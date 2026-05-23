"""
MinerU 3.x pipeline。

调用方式：subprocess 调 `mineru` CLI（Python API 在 3.x 不稳）。
mineru 3.x CLI 的输出目录约定（注意：和 2.x 不同！）：
    <out_dir>/<pdf_stem>/<auto|vlm>/
        ├── <stem>.md                     # 主 markdown（不是 full.md）
        ├── <stem>_content_list.json      # 结构化元素列表
        ├── <stem>_layout.pdf
        ├── <stem>_model.json
        ├── <stem>_middle.json
        └── images/

content_list.json 元素的 bbox 是 **归一化到 0-1000 整数**，不是 PDF point。
我们把它换算回 PDF point（× page_width_pt / 1000）以与其他 pipeline 对齐。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from ..schemas import (
    BBox,
    LayoutBlock,
    PipelineMeta,
    PipelineOutput,
    PREVIEW_DIR,
    Section,
    StructuredDocument,
    Table,
)
from ..utils import render_pdf_to_images
from .base import BasePipeline

logger = logging.getLogger(__name__)


# MinerU 3.x content_list.json 中 "type" 字段到统一 schema 的映射
# 注意：MinerU 3.x 没有独立的 "title" 类型，标题是 type:"text" + text_level>=1
_MINERU_TYPE_MAP = {
    "text": "paragraph",
    "image": "figure",
    "table": "table",
    "equation": "formula",
    "list": "list",
    "code": "paragraph",
    "seal": "stamp",
    "chart": "figure",
    "header": "header",
    "footer": "footer",
    "page_number": "footer",
    "aside_text": "paragraph",
    "page_footnote": "footer",
}


class MinerUPipeline(BasePipeline):
    name = "mineru"

    def __init__(
        self,
        device: str | None = None,
        backend: str | None = None,
        dpi: int = 200,
    ) -> None:
        super().__init__(device=device)
        # MinerU 3.x backend 合法值（实测）：
        #   pipeline                 CPU 兜底，兼容性最好
        #   hybrid-auto-engine       3.x 默认，混合方案
        #   hybrid-http-client       走 http server
        #   vlm-auto-engine          GPU VLM 推理
        #   vlm-http-client          走 http server
        # 默认策略：CUDA → vlm-auto-engine，其它（CPU/MPS）→ pipeline
        self.backend = backend or ("vlm-auto-engine" if self.device == "cuda" else "pipeline")
        self.dpi = dpi

    def _process(self, pdf_path: Path, work_dir: Path) -> PipelineOutput:
        # 1) preview images（独立于 MinerU 内部产物，便于横向比较）
        preview_dir = work_dir / PREVIEW_DIR
        pages = render_pdf_to_images(pdf_path, preview_dir, dpi=self.dpi)

        # 2) 调用 mineru CLI
        mineru_out = work_dir / "_mineru_raw"
        mineru_out.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.setdefault("MINERU_MODEL_SOURCE", "modelscope")  # 国内更快

        cmd = [
            "mineru",
            "-p",
            str(pdf_path),
            "-o",
            str(mineru_out),
            "-b",
            self.backend,
        ]
        logger.info("[mineru] running: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.error("[mineru] stdout=%s", proc.stdout[-2000:])
            logger.error("[mineru] stderr=%s", proc.stderr[-2000:])
            raise RuntimeError(f"mineru CLI failed (rc={proc.returncode})")

        # 3) 找到 MinerU 实际写入的目录
        result_dir = _locate_mineru_result(mineru_out, pdf_path.stem)
        if result_dir is None:
            raise RuntimeError(
                f"MinerU output not found under {mineru_out}; stdout={proc.stdout[-500:]}"
            )

        # 4) 读 markdown / content_list.json
        # MinerU 3.x 文件名是 {stem}.md / {stem}_content_list.json
        stem = pdf_path.stem
        candidates_md = [result_dir / f"{stem}.md", result_dir / "full.md"]
        candidates_cl = [
            result_dir / f"{stem}_content_list.json",
            result_dir / "content_list.json",
        ]
        md_path = next((p for p in candidates_md if p.exists()), candidates_md[0])
        cl_path = next((p for p in candidates_cl if p.exists()), candidates_cl[0])
        markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        content_list = (
            json.loads(cl_path.read_text(encoding="utf-8")) if cl_path.exists() else []
        )

        # 拿到每页 PDF point 尺寸用于 bbox 归一化反算
        page_dims = {p.page_index: (p.width_pt, p.height_pt) for p in pages}

        layout_blocks, tables, raw_text = _normalize_mineru(content_list, page_dims)
        sections = _split_sections(markdown)

        structured = StructuredDocument(
            title=sections[0].title if sections else None,
            document_type=None,
            language="zh",
            pages=len(pages),
            sections=sections,
            tables=tables,
        )

        meta = PipelineMeta(
            pipeline_name="mineru",
            pipeline_version=_mineru_version(),
            model="MinerU",
            device=self.device,
            source_pdf=str(pdf_path),
            started_at=datetime.now(),
            finished_at=datetime.now(),
            duration_seconds=0.0,
            notes=f"backend={self.backend}, model_source=modelscope",
        )

        # 5) 复制 MinerU 自己渲染的 images 到 preview 目录（不覆盖 PyMuPDF 渲染图）
        mineru_images = result_dir / "images"
        if mineru_images.exists():
            dst = preview_dir / "mineru_images"
            dst.mkdir(exist_ok=True)
            for f in mineru_images.iterdir():
                shutil.copy(f, dst / f.name)

        return PipelineOutput(
            meta=meta,
            raw_text=raw_text,
            markdown=markdown,
            layout=layout_blocks,
            structured=structured,
            preview_image_paths=[str(p.image_path) for p in pages],
        )


def _locate_mineru_result(out_root: Path, stem: str) -> Path | None:
    """
    MinerU 3.x 输出位置约定：<out>/<stem>/<auto|vlm>/。
    主 markdown 文件名是 {stem}.md（不是 full.md）。
    """
    candidates = [
        out_root / stem / "auto",
        out_root / stem / "vlm",
        out_root / stem,
    ]
    expected_md = (f"{stem}.md", "full.md")  # 兼容 2.x 旧目录
    for c in candidates:
        if c.exists() and any((c / name).exists() for name in expected_md):
            return c
    # 兜底：递归找 {stem}.md
    for p in out_root.rglob(f"{stem}.md"):
        return p.parent
    return None


def _mineru_version() -> str:
    try:
        proc = subprocess.run(
            ["mineru", "--version"], capture_output=True, text=True, check=False
        )
        return (proc.stdout or proc.stderr).strip()
    except Exception:
        return "unknown"


def _normalize_mineru(
    content_list: list[dict],
    page_dims: dict[int, tuple[float, float]],
) -> tuple[list[LayoutBlock], list[Table], str]:
    """
    把 MinerU 3.x content_list.json 归一化到统一 schema。

    关键点（与 2.x 不同）：
    - **没有 "title" 这个 type**，标题是 type:"text" + "text_level" >= 1
    - bbox 是**归一化到 0-1000 整数**，必须乘以页面真实宽高换算回 PDF point
    - `table_caption` / `image_caption` 是 **list[str]**，要 join
    """
    blocks: list[LayoutBlock] = []
    tables: list[Table] = []
    raw_lines: list[str] = []

    for i, item in enumerate(content_list):
        page = item.get("page_idx", 0)
        bbox_raw = item.get("bbox") or []
        bbox = None
        if len(bbox_raw) == 4:
            page_w, page_h = page_dims.get(page, (595.0, 841.0))  # A4 兜底
            # 0-1000 归一化坐标 → PDF point
            bbox = BBox(
                page=page,
                x0=float(bbox_raw[0]) * page_w / 1000.0,
                y0=float(bbox_raw[1]) * page_h / 1000.0,
                x1=float(bbox_raw[2]) * page_w / 1000.0,
                y1=float(bbox_raw[3]) * page_h / 1000.0,
            )

        item_type = item.get("type", "text")
        text_level = item.get("text_level") or 0
        text = item.get("text", "") or ""

        # 标题识别：type=text + text_level>=1
        if item_type == "text" and text_level >= 1:
            block_type = "title"
        else:
            block_type = _MINERU_TYPE_MAP.get(item_type, "other")

        # caption list[str] → str
        caption = item.get("table_caption") or item.get("image_caption") or []
        if isinstance(caption, list):
            caption = " ".join(str(c) for c in caption if c)

        if bbox:
            blocks.append(
                LayoutBlock(
                    bbox=bbox,
                    text=text or caption or "",
                    block_type=block_type,  # type: ignore[arg-type]
                    reading_order=i,
                )
            )
        if text:
            raw_lines.append(text)
        elif caption:
            raw_lines.append(caption)

        if item_type == "table":
            tables.append(
                Table(
                    page=page,
                    bbox=bbox,
                    html=item.get("table_body", ""),
                    caption=caption or None,
                )
            )

    return blocks, tables, "\n".join(raw_lines)


def _split_sections(md: str) -> list[Section]:
    import re

    sections: list[Section] = []
    current_title: str | None = None
    current_level = 1
    buf: list[str] = []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            if current_title is not None:
                sections.append(
                    Section(
                        level=current_level,
                        title=current_title,
                        text="\n".join(buf).strip(),
                        page_start=0,
                        page_end=0,
                    )
                )
            current_title = m.group(2)
            current_level = len(m.group(1))
            buf = []
        else:
            buf.append(line)
    if current_title is not None:
        sections.append(
            Section(
                level=current_level,
                title=current_title,
                text="\n".join(buf).strip(),
                page_start=0,
                page_end=0,
            )
        )
    return sections
