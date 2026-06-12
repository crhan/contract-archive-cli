"""
PDF 公共工具：分页转图片、获取页面尺寸。

之所以用 PyMuPDF (fitz) 而不是 pdf2image：
- 不依赖系统 poppler，纯 Python wheel，跨平台 (macOS arm64 / Linux x86_64) 都有预编译
- 速度更快，性能稳定
- 同时能拿到原始 PDF 的页面 mediabox 用于 layout 坐标对齐
"""
from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageImage:
    """单页渲染结果。"""

    page_index: int  # 0-based
    image_path: Path  # PNG 文件绝对路径
    width_px: int
    height_px: int
    width_pt: float  # PDF 原始页面宽 (point, 1 pt = 1/72 inch)
    height_pt: float
    dpi: int


@dataclass
class PdfPageInfo:
    """PDF page metadata that does not require rendering the page bitmap."""

    page_index: int
    width_pt: float
    height_pt: float
    image_count: int


@dataclass
class TextLayerStats:
    """Quick quality signal for a PDF's embedded text layer."""

    pages: int
    chars: int
    non_ws_chars: int
    printable_chars: int
    cjk_chars: int
    control_chars: int
    replacement_chars: int

    @property
    def printable_ratio(self) -> float:
        return self.printable_chars / self.non_ws_chars if self.non_ws_chars else 0.0

    @property
    def cjk_ratio(self) -> float:
        return self.cjk_chars / self.non_ws_chars if self.non_ws_chars else 0.0

    @property
    def control_ratio(self) -> float:
        return self.control_chars / self.chars if self.chars else 0.0

    @property
    def replacement_ratio(self) -> float:
        return self.replacement_chars / self.chars if self.chars else 0.0

    @property
    def usable(self) -> bool:
        return is_text_layer_usable(self)


def inspect_pdf_pages(pdf_path: str | Path) -> list[PdfPageInfo]:
    """Return page dimensions/image counts without rasterizing every page."""
    pdf_path = Path(pdf_path)
    infos: list[PdfPageInfo] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            infos.append(
                PdfPageInfo(
                    page_index=idx,
                    width_pt=page.rect.width,
                    height_pt=page.rect.height,
                    image_count=len(page.get_images(full=True)),
                )
            )
    return infos


def render_pdf_to_images(
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = 200,
    prefix: str = "page",
) -> list[PageImage]:
    """
    将 PDF 每页渲染成 PNG，返回元数据列表。

    :param pdf_path: 输入 PDF
    :param out_dir: 输出目录（会自动创建）
    :param dpi: 渲染 DPI；200 是 OCR 通用甜点（精度足够、文件不大）。
                 原扫描件 400 DPI 时建议 dpi >= 300，否则会丢字。
    :param prefix: 输出文件名前缀，最终形如 page_001.png
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    scale = dpi / 72.0  # PyMuPDF 默认 72 DPI
    matrix = fitz.Matrix(scale, scale)

    results: list[PageImage] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_path = out_dir / f"{prefix}_{idx + 1:03d}.png"
            pix.save(img_path)
            results.append(
                PageImage(
                    page_index=idx,
                    image_path=img_path.resolve(),
                    width_px=pix.width,
                    height_px=pix.height,
                    width_pt=page.rect.width,
                    height_pt=page.rect.height,
                    dpi=dpi,
                )
            )

    return results


def extract_text_layer(pdf_path: str | Path, max_chars: int | None = None) -> str:
    """
    抽取 PDF 文字层。扫描版会返回空字符串或纯空白。
    用于快速判断是否需要走 OCR。
    """
    pdf_path = Path(pdf_path)
    chunks: list[str] = []
    total = 0
    with fitz.open(pdf_path) as doc:
        for page in doc:
            chunk = page.get_text()
            if max_chars is not None and total + len(chunk) > max_chars:
                chunk = chunk[: max(0, max_chars - total)]
            chunks.append(chunk)
            total += len(chunk)
            if max_chars is not None and total >= max_chars:
                break
    return "\n".join(chunks)


def analyze_text_layer(pdf_path: str | Path, max_chars: int = 20000) -> TextLayerStats:
    """
    Inspect the embedded text layer without running OCR.

    Some generated PDFs expose a text layer that is technically non-empty but
    unusable because the font encoding maps glyphs to control/extended garbage.
    Those files must still go through OCR/VL instead of being treated as native
    text PDFs.
    """
    text = extract_text_layer(pdf_path, max_chars=max_chars)
    chars = len(text)
    non_ws = [c for c in text if not c.isspace()]
    printable = sum((c in string.printable) or ("\u4e00" <= c <= "\u9fff") for c in non_ws)
    cjk = sum("\u4e00" <= c <= "\u9fff" for c in non_ws)
    control = sum(
        unicodedata.category(c) in {"Cc", "Cf", "Cs", "Co", "Cn"}
        and c not in "\n\t\r"
        for c in text
    )
    replacement = text.count("\ufffd")
    try:
        pages = len(inspect_pdf_pages(pdf_path))
    except Exception:
        pages = 0
    return TextLayerStats(
        pages=pages,
        chars=chars,
        non_ws_chars=len(non_ws),
        printable_chars=printable,
        cjk_chars=cjk,
        control_chars=control,
        replacement_chars=replacement,
    )


def is_text_layer_usable(stats: TextLayerStats, min_chars: int = 200) -> bool:
    """Heuristic gate for using native PDF text instead of OCR."""
    if stats.non_ws_chars < min_chars:
        return False
    if stats.control_ratio > 0.02 or stats.replacement_ratio > 0.005:
        return False
    return stats.printable_ratio >= 0.85 or stats.cjk_ratio >= 0.12


def is_scanned_pdf(pdf_path: str | Path, min_chars: int = 50) -> bool:
    """
    简单判断：没有可用文字层即视为需要 OCR。
    注意：部分 PDF 有非空但乱码的文字层，不能只按字符数判断。
    """
    return not is_text_layer_usable(analyze_text_layer(pdf_path), min_chars=min_chars)
