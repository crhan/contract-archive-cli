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

            # 取该页的 PDF point 尺寸，把 PaddleOCR 的图像像素 bbox 换算回 PDF point
            page_w_pt = pages[idx].width_pt if idx < len(pages) else 595.0
            page_h_pt = pages[idx].height_pt if idx < len(pages) else 842.0
            page_blocks, page_tables, page_text = _normalize_paddle_page(
                res, idx, page_w_pt, page_h_pt
            )
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
    res: Any, page_idx: int, page_w_pt: float, page_h_pt: float
) -> tuple[list[LayoutBlock], list[Table], str]:
    """
    把 PP-StructureV3 单页结果转成统一 schema。

    坐标系换算：
      PP-StructureV3 的 bbox 是**输入图像像素**（按内部 dpi 渲染），
      不是 PDF point。换算公式：x_pt = x_px * page_w_pt / img_w_px。
      img_w_px 从 overall_ocr_res.input_img 的形状或 doc_preprocessor_res 获取，
      退而求其次按"bbox 最大值估算"。
    """
    blocks: list[LayoutBlock] = []
    tables: list[Table] = []
    text_parts: list[str] = []

    data = res if isinstance(res, dict) else getattr(res, "json", None) or {}

    # 1) 估算图像像素尺寸：从 overall_ocr_res 或 layout 块的 bbox 最大值
    img_w_px, img_h_px = _estimate_paddle_image_size(data)
    sx = page_w_pt / img_w_px if img_w_px else 1.0
    sy = page_h_pt / img_h_px if img_h_px else 1.0

    # 2) OCR 文本（rec_texts + dt_polys）—— 同时建立 (poly_center → text) 映射用于 layout 关联
    ocr = data.get("overall_ocr_res") or {}
    rec_texts = ocr.get("rec_texts", []) if isinstance(ocr, dict) else []
    dt_polys = ocr.get("dt_polys", []) if isinstance(ocr, dict) else []
    text_parts.extend([t for t in rec_texts if t])

    poly_text_map: list[tuple[float, float, float, float, str]] = []  # (x0,y0,x1,y1,text)
    for poly, t in zip(dt_polys, rec_texts):
        # poly 可能是 numpy array，不能直接 `not poly`
        if poly is None or len(poly) == 0 or not t:
            continue
        try:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            poly_text_map.append((min(xs), min(ys), max(xs), max(ys), t))
        except (IndexError, TypeError, ValueError):
            continue

    # 3) layout 块（含坐标换算 + 文本关联）
    layout = data.get("layout_det_res") or data.get("layout_res") or {}
    boxes = layout.get("boxes", []) if isinstance(layout, dict) else []
    for i, b in enumerate(boxes):
        bbox_raw = b.get("bbox") or b.get("coordinate") or []
        if len(bbox_raw) != 4:
            continue
        x0, y0, x1, y1 = bbox_raw
        label = (b.get("label") or b.get("category") or "other").lower()
        block_type = _PADDLE_BLOCK_MAP.get(label, "other")
        # 用 OCR polys 关联 text（block bbox 包住的 OCR poly 的 text 拼起来）
        contained_texts = [
            t for (px0, py0, px1, py1, t) in poly_text_map
            if px0 >= x0 - 5 and py0 >= y0 - 5 and px1 <= x1 + 5 and py1 <= y1 + 5
        ]
        blocks.append(
            LayoutBlock(
                bbox=BBox(
                    page=page_idx,
                    x0=x0 * sx,
                    y0=y0 * sy,
                    x1=x1 * sx,
                    y1=y1 * sy,
                ),
                text=" ".join(contained_texts),
                block_type=block_type,  # type: ignore[arg-type]
                confidence=b.get("score"),
                reading_order=b.get("order") or i,
            )
        )

    # 4) 表格
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


def _estimate_paddle_image_size(data: dict) -> tuple[float, float]:
    """
    从 PP-StructureV3 输出里估算输入图像的像素尺寸。
    优先：doc_preprocessor_res 的 input_img / output_img 形状；
    退路：layout boxes 的 bbox 最大值。
    """
    # 尝试 doc_preprocessor_res（注意：里面的 img 是 numpy array，
    # 不能用 `or`，否则触发 ndarray truth-value 异常）
    pre = data.get("doc_preprocessor_res") or {}
    if isinstance(pre, dict):
        img = pre.get("output_img")
        if img is None:
            img = pre.get("input_img")
        if img is not None and hasattr(img, "shape"):
            h, w = img.shape[:2]
            return float(w), float(h)

    # 退路：layout boxes 的 bbox 最大值
    layout = data.get("layout_det_res") or {}
    boxes = layout.get("boxes", []) if isinstance(layout, dict) else []
    max_x = max_y = 0.0
    for b in boxes:
        bb = b.get("bbox") or []
        if len(bb) == 4:
            max_x = max(max_x, bb[2])
            max_y = max(max_y, bb[3])
    return (max_x or 595.0, max_y or 842.0)


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
