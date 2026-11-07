"""
PDF text acquisition pipeline.

The pipeline has two paths:
- native PDF text layer for pages with usable embedded text
- DashScope VL OCR for pages that need image-based extraction

It writes the normalized archive artifacts consumed by extraction:
raw_text.txt, markdown.md, structured.json, layout.json, pipeline_meta.json,
and preview_images/page_NNN.png when pages are rendered.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from ..schemas import (
    FILE_LAYOUT,
    FILE_MARKDOWN,
    FILE_PIPELINE_META,
    FILE_RAW_TEXT,
    FILE_STRUCTURED,
    PipelineMeta,
    PipelineOutput,
    PREVIEW_DIR,
    Section,
    StructuredDocument,
)
from ..utils import (
    PageRoute,
    PdfPageInfo,
    TextLayerStats,
    analyze_text_layer,
    classify_pages,
    describe_device,
    extract_pages_text,
    extract_text_layer,
    inspect_pdf_pages,
    render_pdf_to_images,
    routing_summary,
    select_device,
)
from ..utils.page_router import MODE_OCR, MODE_TEXT
from .vl_ocr import _MARK_FAILED, _MARK_ILLEGIBLE, ocr_pages

logger = logging.getLogger(__name__)


_OWNED_FILES = {
    FILE_RAW_TEXT,
    FILE_MARKDOWN,
    FILE_STRUCTURED,
    FILE_LAYOUT,
    FILE_PIPELINE_META,
    "extraction_result.json",
    "extraction_confidence.json",
}
_OWNED_DIRS = {PREVIEW_DIR}


class OCRPipeline:
    """Extract readable document text from PDF into archive artifacts."""

    name = "ocr"

    def __init__(
        self,
        device: str | None = None,
        dpi: int = 200,
        prefer_text_layer: bool = True,
        allow_vl_fallback: bool = True,
        vl_ocr_max_pages: int | None = None,
        vl_ocr_dpi: int | None = None,
    ) -> None:
        self.device = select_device(device)
        logger.info("[%s] device = %s", self.name, describe_device(self.device))
        self.dpi = dpi
        self.prefer_text_layer = prefer_text_layer
        self.allow_vl_fallback = allow_vl_fallback
        self.vl_ocr_max_pages = vl_ocr_max_pages or _env_int(
            "CONTRACT_ARCHIVE_VL_OCR_MAX_PAGES", 500
        )
        self.vl_ocr_dpi = vl_ocr_dpi or _env_int("CONTRACT_ARCHIVE_VL_OCR_DPI", 160)

    def run(self, pdf_path: str | Path, out_dir: str | Path) -> PipelineOutput:
        """
        Run OCR and write normalized artifacts.

        The output directory is cleaned by a whitelist so unrelated files are
        not removed when reusing an archive path.
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
        except Exception:
            # 不把 PDF/上游异常的原始文本写进日志或 pipeline_meta；它可能包含文档内容。
            logger.error("[%s] pipeline failed", self.name)
            _dump_failure(out_dir, pdf_path, started)
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

    def _process(self, pdf_path: Path, work_dir: Path) -> PipelineOutput:
        page_infos = inspect_pdf_pages(pdf_path)
        text_stats = analyze_text_layer(pdf_path)

        routes = classify_pages(pdf_path)
        if not self.prefer_text_layer:
            routes = [replace(r, mode=MODE_OCR) for r in routes]
        summary = routing_summary(routes)

        if summary["ocr_pages"] == 0:
            logger.info("[pdf] all %s page(s) native-text; using text layer", summary["total"])
            text = extract_text_layer(pdf_path)
            return _output_from_text(
                pdf_path=pdf_path,
                page_infos=page_infos,
                text=text,
                device=self.device,
                source="native-text-layer",
                notes=_text_stats_note("backend=native-text-layer", text_stats),
                page_routing=summary,
            )

        logger.info(
            "[pdf] hybrid routing: %s text / %s ocr / %s table (of %s)",
            summary["text_pages"],
            summary["ocr_pages"],
            summary["table_pages"],
            summary["total"],
        )
        hybrid = self._try_hybrid(
            pdf_path,
            work_dir,
            page_infos,
            routes,
            text_stats,
            summary,
        )
        if hybrid is not None:
            return hybrid
        if not self.allow_vl_fallback:
            raise RuntimeError("document requires image OCR but remote OCR is disabled")
        raise RuntimeError("VL OCR produced no usable document text")

    def _try_hybrid(
        self,
        pdf_path: Path,
        work_dir: Path,
        page_infos: list[PdfPageInfo],
        routes: list[PageRoute],
        text_stats: TextLayerStats,
        summary: dict[str, int],
    ) -> PipelineOutput | None:
        """Use native text pages plus VL OCR pages, then stitch pages in order."""
        if not self.allow_vl_fallback:
            return None
        if len(page_infos) > self.vl_ocr_max_pages:
            logger.warning(
                "[vl-ocr] skip hybrid: pages=%s exceeds max=%s",
                len(page_infos),
                self.vl_ocr_max_pages,
            )
            return None

        preview_dir = work_dir / PREVIEW_DIR
        pages = _render_previews_safe(pdf_path, preview_dir, self.vl_ocr_dpi)
        img_by_idx = {p.page_index: p.image_path for p in pages}

        ocr_route_idxs = {r.page_index for r in routes if r.mode == MODE_OCR}
        ocr_pairs = [
            (r.page_index, img_by_idx[r.page_index])
            for r in routes
            if r.mode == MODE_OCR and r.page_index in img_by_idx
        ]
        missing_render_idxs = ocr_route_idxs - set(img_by_idx)
        page_results = (
            ocr_pages(
                [img for _, img in ocr_pairs],
                page_labels=[idx + 1 for idx, _ in ocr_pairs],
            )
            if ocr_pairs
            else []
        )

        # 无凭证/整批调用不可用时，不能悄悄漏掉 OCR 页：每个本应 OCR 的已渲染页
        # 都落一个明确失败标记，供 PipelineMeta 与上游把它视为必需阶段失败。
        if page_results is None:
            page_results = []
            missing_ocr_result_idxs = {idx for idx, _ in ocr_pairs}
        else:
            missing_ocr_result_idxs = {
                idx for idx, _ in ocr_pairs[len(page_results):]
            }

        ocr_body_by_idx = {
            idx: res.body for (idx, _), res in zip(ocr_pairs, page_results)
        }
        ocr_ok = any(res.ok for res in page_results)
        failed_pages = {
            idx + 1
            for (idx, _), res in zip(ocr_pairs, page_results)
            if res.failed
        }
        failed_pages.update(idx + 1 for idx in missing_render_idxs)
        failed_pages.update(idx + 1 for idx in missing_ocr_result_idxs)
        truncated_pages = {
            idx + 1
            for (idx, _), res in zip(ocr_pairs, page_results)
            if res.truncated
        }

        native_by_idx = extract_pages_text(
            pdf_path, [r.page_index for r in routes if r.mode == MODE_TEXT]
        )

        blocks: list[str] = []
        native_ok = False
        for r in routes:
            idx = r.page_index
            if idx in ocr_body_by_idx:
                body = ocr_body_by_idx[idx]
            elif idx in ocr_route_idxs:
                # 区分渲染缺页和远端 OCR 调用失败：两者都进 failed_pages，但原文
                # 标记让人工知道补跑该从哪一步开始。
                body = (
                    "[本页渲染失败，无法进行 OCR]"
                    if idx in missing_render_idxs
                    else _MARK_FAILED
                )
            else:
                native = native_by_idx.get(idx, "").strip()
                if native:
                    native_ok = True
                    body = native
                else:
                    body = _MARK_ILLEGIBLE
            blocks.append(f"## 第 {idx + 1} 页\n\n{body}")

        if not ocr_ok and not native_ok:
            logger.warning("[vl-ocr] hybrid produced no usable page; preserving page failure markers")

        return _output_from_text(
            pdf_path=pdf_path,
            page_infos=page_infos,
            text="\n\n".join(blocks).strip(),
            device=self.device,
            source="hybrid-text-vl",
            notes=_text_stats_note("backend=hybrid-text-vl", text_stats),
            preview_image_paths=[str(p.image_path) for p in pages],
            page_routing=summary,
            ocr_failed_pages=sorted(failed_pages),
            ocr_truncated_pages=sorted(truncated_pages),
        )


def _dump(out_dir: Path, result: PipelineOutput) -> None:
    (out_dir / FILE_RAW_TEXT).write_text(result.raw_text, encoding="utf-8")
    (out_dir / FILE_MARKDOWN).write_text(result.markdown, encoding="utf-8")
    (out_dir / FILE_STRUCTURED).write_text(
        result.structured.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )
    (out_dir / FILE_LAYOUT).write_text(
        json.dumps([b.model_dump() for b in result.layout], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / FILE_PIPELINE_META).write_text(
        result.meta.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _dump_failure(out_dir: Path, pdf_path: Path, started: datetime) -> None:
    meta = PipelineMeta(
        pipeline_name="ocr",
        source_pdf=str(pdf_path),
        started_at=started,
        finished_at=datetime.now(),
        duration_seconds=(datetime.now() - started).total_seconds(),
        notes="FAILED: OCR pipeline failure",
    )
    (out_dir / FILE_PIPELINE_META).write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _text_stats_note(prefix: str, stats: TextLayerStats) -> str:
    return (
        f"{prefix}; pages={stats.pages}, chars={stats.non_ws_chars}, "
        f"printable_ratio={stats.printable_ratio:.3f}, cjk_ratio={stats.cjk_ratio:.3f}, "
        f"control_ratio={stats.control_ratio:.3f}"
    )


def _render_previews_safe(pdf_path: Path, preview_dir: Path, dpi: int) -> list:
    try:
        return render_pdf_to_images(pdf_path, preview_dir, dpi=dpi)
    except Exception:  # noqa: BLE001 - previews are useful but not primary text output
        logger.warning("[pdf] preview render failed; preserving affected OCR pages as failures")
        return []


def _output_from_text(
    *,
    pdf_path: Path,
    page_infos: list[PdfPageInfo],
    text: str,
    device: str,
    source: str,
    notes: str,
    preview_image_paths: list[str] | None = None,
    page_routing: dict[str, int] | None = None,
    ocr_failed_pages: list[int] | None = None,
    ocr_truncated_pages: list[int] | None = None,
) -> PipelineOutput:
    markdown = _clean_markdown_text(text)
    sections = _split_sections(markdown)
    if not sections:
        title = _guess_title(markdown) or pdf_path.stem
        sections = [
            Section(
                level=1,
                title=title,
                text=markdown.strip(),
                page_start=0,
                page_end=max(0, len(page_infos) - 1),
            )
        ]
    structured = StructuredDocument(
        title=sections[0].title if sections else None,
        document_type=None,
        language="zh",
        pages=len(page_infos),
        sections=sections,
        tables=[],
    )
    now = datetime.now()
    return PipelineOutput(
        meta=PipelineMeta(
            pipeline_name="ocr",
            pipeline_version="native-text+vl-ocr",
            model=source,
            device=device,
            source_pdf=str(pdf_path),
            started_at=now,
            finished_at=now,
            duration_seconds=0.0,
            notes=notes,
            page_routing=page_routing or {},
            ocr_failed_pages=ocr_failed_pages or [],
            ocr_truncated_pages=ocr_truncated_pages or [],
        ),
        raw_text=markdown,
        markdown=markdown,
        layout=[],
        structured=structured,
        preview_image_paths=preview_image_paths or [],
    )


def _guess_title(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip(" #\t")
        if 4 <= len(line) <= 80:
            return line
    return None


def _clean_markdown_text(text: str) -> str:
    """Normalize markdown escapes and OCR filler underscores before extraction."""
    import re as _re

    text = _re.sub(r"\\([_*\[\]()#+\-.!`])", r"\1", text)
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
