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

历史：原本与 DashScope/PaddleOCR pipeline 共享一个 BasePipeline 抽象基类。
重构后唯一具体实现，抽象基类已 inline 到本文件，避免一抽象一具体的反模式。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..schemas import (
    BBox,
    FILE_LAYOUT,
    FILE_MARKDOWN,
    FILE_PIPELINE_META,
    FILE_RAW_TEXT,
    FILE_STRUCTURED,
    LayoutBlock,
    PipelineMeta,
    PipelineOutput,
    PREVIEW_DIR,
    Section,
    StructuredDocument,
    Table,
)
from ..utils import describe_device, render_pdf_to_images, select_device

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


# 已知 pipeline 产物文件，清空目录时只删这些 + 这些子目录，绝不 rmtree 未知内容
_OWNED_FILES = {
    FILE_RAW_TEXT,
    FILE_MARKDOWN,
    FILE_STRUCTURED,
    FILE_LAYOUT,
    FILE_PIPELINE_META,
    "extraction_result.json",
    "extraction_confidence.json",
}
_OWNED_DIRS = {PREVIEW_DIR, "_mineru_raw"}


class MinerUPipeline:
    """MinerU 3.x PDF 解析。单一职责：run(pdf, out_dir) → PipelineOutput。"""

    name = "mineru"

    def __init__(
        self,
        device: str | None = None,
        backend: str | None = None,
        dpi: int = 200,
    ) -> None:
        self.device = select_device(device)
        logger.info("[%s] device = %s", self.name, describe_device(self.device))
        # MinerU 3.x backend 合法值（实测）：
        #   pipeline                 CPU 兜底，兼容性最好
        #   hybrid-auto-engine       3.x 默认，混合方案
        #   hybrid-http-client       走 http server
        #   vlm-auto-engine          GPU VLM 推理
        #   vlm-http-client          走 http server
        # 默认策略：CUDA → vlm-auto-engine，其它（CPU/MPS）→ pipeline
        self.backend = backend or ("vlm-auto-engine" if self.device == "cuda" else "pipeline")
        self.dpi = dpi

    # ---------- 入口 ----------
    def run(self, pdf_path: str | Path, out_dir: str | Path) -> PipelineOutput:
        """
        统一入口：
        - 清理本 pipeline 写过的旧产物（白名单，绝不递归删未知内容）
        - 计时 + 调 _process
        - 按统一文件名落盘
        """
        pdf_path = Path(pdf_path).resolve()
        out_dir = Path(out_dir).resolve()
        if out_dir.exists():
            for item in out_dir.iterdir():
                if item.is_file() and item.name in _OWNED_FILES:
                    item.unlink()
                elif item.is_dir() and item.name in _OWNED_DIRS:
                    shutil.rmtree(item)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / PREVIEW_DIR).mkdir(exist_ok=True)

        started = datetime.now()
        t0 = time.perf_counter()
        try:
            result = self._process(pdf_path, out_dir)
        except Exception as e:
            logger.exception("[%s] pipeline failed", self.name)
            _dump_failure(out_dir, pdf_path, started, str(e))
            raise
        duration = time.perf_counter() - t0
        finished = datetime.now()

        result.meta.started_at = started
        result.meta.finished_at = finished
        result.meta.duration_seconds = duration
        if not result.meta.source_pdf:
            result.meta.source_pdf = str(pdf_path)
        if not result.meta.device:
            result.meta.device = self.device

        _dump(out_dir, result)
        logger.info("[%s] done in %.2fs, output=%s", self.name, duration, out_dir)
        return result

    # ---------- 实现 ----------
    def _process(self, pdf_path: Path, work_dir: Path) -> PipelineOutput:
        # 1) preview images（独立于 MinerU 内部产物，供下游审阅）
        preview_dir = work_dir / PREVIEW_DIR
        pages = render_pdf_to_images(pdf_path, preview_dir, dpi=self.dpi)

        # 2) 调用 mineru CLI
        mineru_out = work_dir / "_mineru_raw"
        mineru_out.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.setdefault("MINERU_MODEL_SOURCE", "modelscope")  # 国内更快

        cmd = [
            _resolve_mineru(),
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
            # 把 stderr 尾部带进异常——失败日志/DB error_message 才能看到真实原因，
            # 不必回头翻控制台（曾因失败日志没记 stderr 而难定位）。
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            reason = tail[-1] if tail else "no stderr captured"
            raise RuntimeError(f"mineru CLI failed (rc={proc.returncode}): {reason}")

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
        # MinerU 的 markdown 会对 _/*/[]/() 等做反斜杠转义（如 "\_2027年\_6月"），
        # 喂给 rule/LLM 之前清掉这些转义，避免 extraction 抓不到
        raw_text = _unescape_markdown(raw_text)
        markdown_for_extract = _unescape_markdown(markdown)
        sections = _split_sections(markdown_for_extract)

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

        # 5) 复制 MinerU 自己渲染的 images 到 preview 目录
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


# ---------- 落盘工具 ----------


def _dump(out_dir: Path, result: PipelineOutput) -> None:
    (out_dir / FILE_RAW_TEXT).write_text(result.raw_text, encoding="utf-8")
    (out_dir / FILE_MARKDOWN).write_text(result.markdown, encoding="utf-8")
    (out_dir / FILE_STRUCTURED).write_text(
        result.structured.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )
    (out_dir / FILE_LAYOUT).write_text(
        json.dumps(
            [b.model_dump() for b in result.layout],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / FILE_PIPELINE_META).write_text(
        result.meta.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _dump_failure(
    out_dir: Path, pdf_path: Path, started: datetime, err: str
) -> None:
    """失败时仍写一份 meta，附错误信息。"""
    meta = PipelineMeta(
        pipeline_name="mineru",
        source_pdf=str(pdf_path),
        started_at=started,
        finished_at=datetime.now(),
        duration_seconds=(datetime.now() - started).total_seconds(),
        notes=f"FAILED: {err}",
    )
    (out_dir / FILE_PIPELINE_META).write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )


# ---------- 解析 / 归一化 ----------


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


def _resolve_mineru() -> str:
    """
    定位 mineru 可执行文件的绝对路径。

    为什么不能直接用裸字符串 "mineru" 交给 subprocess：
    contract-archive 经 `uv tool install` 安装在隔离 venv 里，mineru 作为同一个 venv 的
    依赖（extra）一起装。但该 venv 的 bin/ 目录**不在**用户 shell 的 PATH 上——
    contract-archive 通过 ~/.local/bin 的 symlink 启动，子进程继承的是 shell PATH，
    于是靠 PATH 解析 "mineru" 必然 FileNotFoundError。

    策略（确定性优先，消除"PATH 里必须有 mineru"这个隐含前提）：
      1. 找与当前解释器同目录的兄弟可执行文件——uv tool / 已激活 venv 场景下，
         mineru 与 python 必然同在一个 bin/，这一步直接命中。
      2. 兜底 shutil.which，兼容用户把 mineru 手动放进 PATH 的开发环境。

    返回：mineru 可执行文件路径。
    抛出：FileNotFoundError（附安装指引），比裸的 [Errno 2] 可读得多。
    """
    sibling = Path(sys.executable).parent / "mineru"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("mineru")
    if found:
        return found
    raise FileNotFoundError(
        "找不到 mineru 可执行文件。它随 contract-archive-cli 的 mineru extra 一起安装：\n"
        "  uv tool install 'contract-archive-cli[mineru]'        # 首次安装\n"
        "  uv tool install 'contract-archive-cli[mineru]' --reinstall   # 已装过 contract-archive-cli 但缺 mineru\n"
        "开发环境：uv sync --extra mineru"
    )


def _mineru_version() -> str:
    try:
        proc = subprocess.run(
            [_resolve_mineru(), "--version"], capture_output=True, text=True, check=False
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


def _unescape_markdown(text: str) -> str:
    """
    剥离 MinerU 在 markdown 里加的反斜杠转义（\\_/\\*/\\[/\\]/\\(/\\)/\\#）。
    再清掉常见的"数字两侧夹下划线"残留（合同填空符号被 OCR 当成下划线）：
    '甲方于_2027年_6月_30日' → '甲方于 2027 年 6 月 30 日'
    """
    import re as _re

    text = _re.sub(r"\\([_*\[\]()#+\-.!`])", r"\1", text)
    # 数字/中文之间的下划线一律视为空白
    text = _re.sub(r"_+", " ", text)
    return text


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
