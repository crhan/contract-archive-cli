"""
PaddleOCR + PP-StructureV3 pipeline。

PP-StructureV3 原生支持 PDF 输入，一次性给出：
- markdown（带表格 HTML）
- layout boxes（含 reading_order）
- tables / formulas

在 Mac arm64 上只能 CPU；在 RTX 5080 上换 paddlepaddle-gpu(cu128) 即可启用 GPU。
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas import (
    BBox,
    LayoutBlock,
    PipelineMeta,
    PipelineOutput,
    PREVIEW_DIR,
    Section,
    StructuredDocument,
    Table,
    TableCell,
)
from ..utils import render_pdf_to_images
from .base import BasePipeline

logger = logging.getLogger(__name__)


# block_type 在 paddle 与统一 schema 间的映射
_PADDLE_BLOCK_MAP = {
    "title": "title",
    "text": "paragraph",
    "paragraph_title": "title",
    "table": "table",
    "table_title": "title",
    "figure": "figure",
    "image": "figure",
    "header": "header",
    "footer": "footer",
    "list": "list",
    "formula": "formula",
    "stamp": "stamp",
    "signature": "signature",
    "seal": "stamp",
}


class PaddleOCRPipeline(BasePipeline):
    name = "paddleocr"

    def __init__(self, device: str | None = None, dpi: int = 200) -> None:
        super().__init__(device=device)
        self.dpi = dpi

    def _process(self, pdf_path: Path, work_dir: Path) -> PipelineOutput:
        from paddleocr import PPStructureV3  # lazy import

        # 1) 预先把 PDF 渲染图存为 preview，让 compare.py 能比较
        preview_dir = work_dir / PREVIEW_DIR
        pages = render_pdf_to_images(pdf_path, preview_dir, dpi=self.dpi)

        # 2) 跑 PP-StructureV3（原生收 PDF）
        device_arg = "gpu" if self.device == "cuda" else "cpu"
        pipeline = PPStructureV3(device=device_arg, use_doc_orientation_classify=True)
        results = list(pipeline.predict(input=str(pdf_path)))
        logger.info("[paddleocr] processed %d pages", len(results))

        # 3) 落临时目录方便 debug，但归一化由我们做
        tmp_out = work_dir / "_paddle_raw"
        tmp_out.mkdir(exist_ok=True)
        md_objs: list = []      # PP-StructureV3 要求传完整 markdown dict 列表
        md_texts: list[str] = []  # 兜底用
        layout_blocks: list[LayoutBlock] = []
        tables: list[Table] = []
        raw_text_pages: list[str] = []

        for idx, res in enumerate(results):
            # res 是一个 dict-like 对象（StructureV3Result）
            try:
                res.save_to_json(str(tmp_out))
                res.save_to_markdown(str(tmp_out))
            except Exception:
                logger.debug("paddle res.save_to_* failed; continuing with in-memory")

            md_obj = getattr(res, "markdown", None)
            if md_obj is not None:
                md_objs.append(md_obj)
                if isinstance(md_obj, dict):
                    md_texts.append(md_obj.get("markdown_texts", ""))

            page_blocks, page_tables, page_text = _normalize_paddle_page(res, idx)
            layout_blocks.extend(page_blocks)
            tables.extend(page_tables)
            raw_text_pages.append(page_text)

        # 4) markdown 合并：必须传整个 res.markdown dict 列表，函数内部会处理
        # markdown_images / page_continuation_flags 等字段
        try:
            full_md = pipeline.concatenate_markdown_pages(md_objs)
            if isinstance(full_md, tuple):
                full_md = full_md[0]
            if isinstance(full_md, dict):
                full_md = full_md.get("markdown_texts", "")
        except Exception as e:
            logger.warning("[paddleocr] concatenate_markdown_pages failed: %s", e)
            full_md = "\n\n---\n\n".join(md_texts)

        sections = _split_sections_from_markdown(full_md or "")
        structured = StructuredDocument(
            title=sections[0].title if sections else None,
            document_type=None,
            language="zh",
            pages=len(pages),
            sections=sections,
            tables=tables,
        )

        meta = PipelineMeta(
            pipeline_name="paddleocr",
            pipeline_version=_safe_paddle_version(),
            model="PP-StructureV3",
            device=self.device,
            source_pdf=str(pdf_path),
            started_at=datetime.now(),
            finished_at=datetime.now(),
            duration_seconds=0.0,
            notes=f"dpi={self.dpi}, device_arg={device_arg}",
        )
        return PipelineOutput(
            meta=meta,
            raw_text="\n".join(raw_text_pages),
            markdown=full_md or "",
            layout=layout_blocks,
            structured=structured,
            preview_image_paths=[str(p.image_path) for p in pages],
        )


def _safe_paddle_version() -> str:
    try:
        import paddleocr

        return getattr(paddleocr, "__version__", "unknown")
    except Exception:
        return "unknown"


def _normalize_paddle_page(
    res: Any, page_idx: int
) -> tuple[list[LayoutBlock], list[Table], str]:
    """
    把 PP-StructureV3 单页结果转成统一 schema。

    PP-StructureV3 dict 结构（实际跑通后再微调）：
      res["layout_det_res"]["boxes"]: [{"bbox":[x,y,x,y], "label":..., "score":...}]
      res["overall_ocr_res"]["dt_polys"] / ["rec_texts"]: detection + recognition
      res["table_res_list"]: 表格列表 with html
      res["parsing_res_list"] 或 res["block_res_list"]: 块级带 reading_order
    """
    blocks: list[LayoutBlock] = []
    tables: list[Table] = []
    text_parts: list[str] = []

    data = res if isinstance(res, dict) else getattr(res, "json", None) or {}

    # 1) layout 块
    layout = data.get("layout_det_res") or data.get("layout_res") or {}
    boxes = layout.get("boxes", []) if isinstance(layout, dict) else []
    for i, b in enumerate(boxes):
        bbox_raw = b.get("bbox") or b.get("coordinate") or []
        if len(bbox_raw) != 4:
            continue
        x0, y0, x1, y1 = bbox_raw
        label = (b.get("label") or b.get("category") or "other").lower()
        block_type = _PADDLE_BLOCK_MAP.get(label, "other")
        blocks.append(
            LayoutBlock(
                bbox=BBox(page=page_idx, x0=x0, y0=y0, x1=x1, y1=y1),
                text=b.get("text", ""),
                block_type=block_type,  # type: ignore[arg-type]
                confidence=b.get("score"),
                reading_order=b.get("order") or i,
            )
        )

    # 2) raw_text：用 overall_ocr_res.rec_texts 拼
    ocr = data.get("overall_ocr_res") or {}
    rec_texts = ocr.get("rec_texts") if isinstance(ocr, dict) else None
    if rec_texts:
        text_parts.extend([t for t in rec_texts if t])

    # 3) 表格
    table_list = data.get("table_res_list") or data.get("tables") or []
    for t in table_list:
        if not isinstance(t, dict):
            continue
        html = t.get("html") or t.get("pred_html") or ""
        bbox_raw = t.get("bbox") or t.get("layout_bbox") or []
        bbox_obj = None
        if len(bbox_raw) == 4:
            bbox_obj = BBox(
                page=page_idx,
                x0=bbox_raw[0],
                y0=bbox_raw[1],
                x1=bbox_raw[2],
                y1=bbox_raw[3],
            )
        cells: list[TableCell] = []
        for c in t.get("cells", []) or []:
            cells.append(
                TableCell(
                    row=c.get("row", 0),
                    col=c.get("col", 0),
                    rowspan=c.get("rowspan", 1),
                    colspan=c.get("colspan", 1),
                    text=c.get("text", ""),
                )
            )
        tables.append(
            Table(
                page=page_idx,
                bbox=bbox_obj,
                html=html,
                cells=cells,
                n_rows=t.get("n_rows", 0),
                n_cols=t.get("n_cols", 0),
            )
        )

    return blocks, tables, "\n".join(text_parts)


def _split_sections_from_markdown(md: str) -> list[Section]:
    """与 dashscope 模块共用同一份逻辑——故意复制一份避免横向依赖。"""
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
