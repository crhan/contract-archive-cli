"""
PDF 公共工具：分页转图片、获取页面尺寸。

之所以用 PyMuPDF (fitz) 而不是 pdf2image：
- 不依赖系统 poppler，纯 Python wheel，跨平台 (macOS arm64 / Linux x86_64) 都有预编译
- 速度更快，性能稳定
- 同时能拿到原始 PDF 的页面 mediabox 用于 layout 坐标对齐
"""
from __future__ import annotations

import base64
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
    pages_with_text: int = 0  # 含实质文本的页数；扫描件夹少量文本页时远小于 pages

    @property
    def printable_ratio(self) -> float:
        return self.printable_chars / self.non_ws_chars if self.non_ws_chars else 0.0

    @property
    def text_coverage(self) -> float:
        """含实质文本的页占比。扫描件即便夹几页原生文本，覆盖率也极低。"""
        return self.pages_with_text / self.pages if self.pages else 0.0

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
    pages: list[int] | None = None,
) -> list[PageImage]:
    """
    将 PDF 每页渲染成 PNG，返回元数据列表。

    :param pdf_path: 输入 PDF
    :param out_dir: 输出目录（会自动创建）
    :param dpi: 渲染 DPI；200 是 OCR 通用甜点（精度足够、文件不大）。
                 原扫描件 400 DPI 时建议 dpi >= 300，否则会丢字。
    :param prefix: 输出文件名前缀，最终形如 page_001.png
    :param pages: 只渲这些 0-based 页索引（None=全部）。供 vision 融合按需补渲选中页，
                  不必为了几页重渲整份（native-text 快路下 MinerU 跳过 preview 渲染）。
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    scale = dpi / 72.0  # PyMuPDF 默认 72 DPI
    matrix = fitz.Matrix(scale, scale)
    want = set(pages) if pages is not None else None

    results: list[PageImage] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            if want is not None and idx not in want:
                continue
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


def encode_image_data_uri(path: str | Path) -> str:
    """本地图片 → data URI（base64 内联）。

    DashScope OpenAI 兼容接口不收 file://，图片必须 base64 内联。逐页 OCR、看图抽字段、
    签章核查三处共用，集中一处避免散落多份同款实现。
    """
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def extract_pages_text(
    pdf_path: str | Path, page_indices: set[int] | list[int]
) -> dict[int, str]:
    """抽取指定页（0-based）的原生文本层，返回 {page_index: text}。

    供页级混合提取用：文本页走原生抽取、扫描页走 VL，再按页序拼回。只取需要的页，
    不整份 get_text 后再切分。
    """
    wanted = set(page_indices)
    if not wanted:
        return {}
    out: dict[int, str] = {}
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            if idx in wanted:
                out[idx] = page.get_text()
    return out


def count_text_pages(pdf_path: str | Path, page_min_chars: int = 20) -> tuple[int, int]:
    """逐页统计含实质文本的页数，返回 (pages_with_text, total_pages)。

    扫描件即便夹了少量原生文本页（如打印叠加的保单信息页），文本层覆盖率也极低。
    据此把"扫描件夹文本页"与"真·电子文档"分开，不被文本总量的质量比例误导
    （后者只看 printable/cjk 比例，对夹页扫描件失效）。逐页统计独立于
    analyze_text_layer 的 max_chars 采样截断，长文档也不会失真。
    """
    pdf_path = Path(pdf_path)
    pages_with_text = 0
    total = 0
    with fitz.open(pdf_path) as doc:
        for page in doc:
            total += 1
            # 只需判断是否够门槛，数到 page_min_chars 即停，长条款页不必数满。
            non_ws = 0
            for ch in page.get_text():
                if not ch.isspace():
                    non_ws += 1
                    if non_ws >= page_min_chars:
                        pages_with_text += 1
                        break
    return pages_with_text, total


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
        pages_with_text, pages = count_text_pages(pdf_path)
    except Exception:
        pages_with_text, pages = 0, 0
    return TextLayerStats(
        pages=pages,
        chars=chars,
        non_ws_chars=len(non_ws),
        printable_chars=printable,
        cjk_chars=cjk,
        control_chars=control,
        replacement_chars=replacement,
        pages_with_text=pages_with_text,
    )


def is_text_layer_usable(
    stats: TextLayerStats,
    min_chars: int = 200,
    min_coverage: float = 0.5,
    min_scanned_pages: int = 2,
) -> bool:
    """Heuristic gate for using native PDF text instead of OCR."""
    if stats.non_ws_chars < min_chars:
        return False
    if stats.control_ratio > 0.02 or stats.replacement_ratio > 0.005:
        return False
    if not (stats.printable_ratio >= 0.85 or stats.cjk_ratio >= 0.12):
        return False
    # 覆盖率门槛：文档若同时（1）带文本的页占比过低、（2）有相当数量的纯扫描页，
    # 几乎必是扫描件夹了少量打印叠加页（如保单信息页），整份当电子文档会丢掉那些
    # 纯扫描页的内容。实测一份 184 页保险合同仅 ~8% 页有文本层，却因 printable/cjk
    # 比例达标骗过质量判据，导致一百多页扫描条款被跳过 OCR。
    #
    # 用"绝对纯扫描页数"而非纯页数阈值把关：既拦住 184 页夹几页、也拦住 4 页夹 1 页
    # 这种小份夹页（旧逻辑按 pages<5 豁免会放过后者）；又放过真·短凭证——页少但几乎
    # 页页有文本、纯扫描页数为 0，自然不触发。
    scanned_pages = stats.pages - stats.pages_with_text
    if stats.text_coverage < min_coverage and scanned_pages >= min_scanned_pages:
        return False
    return True


def is_scanned_pdf(pdf_path: str | Path, min_chars: int = 50) -> bool:
    """
    简单判断：没有可用文字层即视为需要 OCR。
    注意：部分 PDF 有非空但乱码的文字层，不能只按字符数判断。
    """
    return not is_text_layer_usable(analyze_text_layer(pdf_path), min_chars=min_chars)
