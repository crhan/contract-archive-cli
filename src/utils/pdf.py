"""
PDF 公共工具：分页转图片、获取页面尺寸。

之所以用 PyMuPDF (fitz) 而不是 pdf2image：
- 不依赖系统 poppler，纯 Python wheel，跨平台 (macOS arm64 / Linux x86_64) 都有预编译
- 速度更快，性能稳定
- 同时能拿到原始 PDF 的页面 mediabox 用于 layout 坐标对齐
"""
from __future__ import annotations

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


def extract_text_layer(pdf_path: str | Path) -> str:
    """
    抽取 PDF 文字层。扫描版会返回空字符串或纯空白。
    用于快速判断是否需要走 OCR。
    """
    pdf_path = Path(pdf_path)
    chunks: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            chunks.append(page.get_text())
    return "\n".join(chunks)


def is_scanned_pdf(pdf_path: str | Path, min_chars: int = 50) -> bool:
    """
    简单判断：文字层字符数低于阈值即视为扫描版。
    阈值默认 50（一页纯文字 PDF 至少几百字符）。
    """
    text = extract_text_layer(pdf_path).strip()
    return len(text) < min_chars
