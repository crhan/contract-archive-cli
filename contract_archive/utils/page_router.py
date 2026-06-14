"""页级分流：逐页判定走"原生文本抽取"还是"VL 看图 OCR"。

取代整份"二选一"（is_text_layer_usable 整份判 native-text OR 整份 OCR）。混合版式
PDF（扫描件夹打印页、复杂表格混排）下二选一系统性丢数据：当成 native 会跳过扫描页，
当成 OCR 又把干净电子页也丢给模型（慢、且 VL 转写未必比原生文本准）。

分流让每页各取所长：
- **主判据**（启发式）：单页文本层质量——非空白字符量 + printable/cjk/control/replacement
  比例，与整份 is_text_layer_usable 同源，只是按页粒度算。够实质且非乱码 → text。
- **加分项**：含表格的页即便有文本层也改走 VL——原生 get_text 把表格结构抹平成一串文本，
  VL（VL_OCR_PAGE_PROMPT 要求转 markdown 表格）保表格结构更可靠。find_tables 在部分
  PyMuPDF 版本/畸形页会抛错或极慢，一律 try/except 兜底为"无表格"，缺了不影响主判据。
"""
from __future__ import annotations

import logging
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# 单页文本层"够用"门槛（与整份 is_text_layer_usable 同源，按页粒度判）。
# 整份阈值 min_chars=200 是全文级；单页用更低的 50：一页正文普遍几十到几百字，
# 低于 50 多半是封面/空白/扫描页，交给 VL 更稳。
_PAGE_MIN_NON_WS = 50
_MAX_CONTROL_RATIO = 0.02
_MAX_REPLACEMENT_RATIO = 0.005
_MIN_PRINTABLE_RATIO = 0.85
_MIN_CJK_RATIO = 0.12

MODE_TEXT = "text"
MODE_OCR = "ocr"


@dataclass(frozen=True)
class PageRoute:
    """单页分流决策。"""

    page_index: int  # 0-based
    mode: str  # MODE_TEXT | MODE_OCR
    has_text: bool  # 文本层是否够实质且非乱码
    has_tables: bool  # 是否检出表格（加分项；失败/不支持 → False）
    char_count: int  # 非空白字符数（调试/日志）

    @property
    def reason(self) -> str:
        if not self.has_text:
            return "no-usable-text"  # 扫描/空白/乱码页
        if self.has_tables:
            return "table-present"  # 有文本层但含表格，改走 VL 保结构
        return "native-text"


def _page_text_usable(text: str) -> tuple[bool, int]:
    """单页文本层是否够用。返回 (usable, non_ws_chars)。判据同整份 is_text_layer_usable。"""
    non_ws = [c for c in text if not c.isspace()]
    n = len(non_ws)
    if n < _PAGE_MIN_NON_WS:
        return False, n
    chars = len(text)
    printable = sum((c in string.printable) or ("一" <= c <= "鿿") for c in non_ws)
    cjk = sum("一" <= c <= "鿿" for c in non_ws)
    control = sum(
        unicodedata.category(c) in {"Cc", "Cf", "Cs", "Co", "Cn"} and c not in "\n\t\r"
        for c in text
    )
    replacement = text.count("�")
    control_ratio = control / chars if chars else 0.0
    replacement_ratio = replacement / chars if chars else 0.0
    printable_ratio = printable / n
    cjk_ratio = cjk / n
    if control_ratio > _MAX_CONTROL_RATIO or replacement_ratio > _MAX_REPLACEMENT_RATIO:
        return False, n  # 乱码字体编码：有文字层但全是控制字/替换符
    if not (printable_ratio >= _MIN_PRINTABLE_RATIO or cjk_ratio >= _MIN_CJK_RATIO):
        return False, n
    return True, n


def find_tables(page: "fitz.Page") -> bool:
    """检测页面是否含**实质**表格（加分项）。任何异常都兜底为 False——缺失不影响主判据。

    要求 ≥2 行 ≥2 列才算：PyMuPDF 的 find_tables 会把对齐文本/单行表头误判成"表格"，
    门槛过松会把大量干净文本页无谓推给 VL（贵且未必更准）。只有真·多行多列表格才值得 VL。
    """
    try:
        for tbl in page.find_tables().tables:
            if getattr(tbl, "row_count", 0) >= 2 and getattr(tbl, "col_count", 0) >= 2:
                return True
        return False
    except Exception as e:  # noqa: BLE001 - 加分项失败不能影响分流主判据
        logger.debug("find_tables failed on a page, treat as no-table: %s", e)
        return False


def classify_pages(pdf_path: str | Path) -> list[PageRoute]:
    """逐页分流：文本层够用且无表格 → MODE_TEXT；扫描/空白/乱码或含表格 → MODE_OCR。"""
    routes: list[PageRoute] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            usable, n = _page_text_usable(page.get_text())
            # 扫描页本就要 OCR，省去一次（偏慢的）find_tables；仅对文本页加测表格。
            has_tables = find_tables(page) if usable else False
            mode = MODE_TEXT if (usable and not has_tables) else MODE_OCR
            routes.append(PageRoute(idx, mode, usable, has_tables, n))
    return routes


def routing_summary(routes: list[PageRoute]) -> dict[str, int]:
    """分流计数摘要（进 PipelineMeta.page_routing，供评测/调试看一份文档怎么被分流）。"""
    return {
        "total": len(routes),
        "text_pages": sum(r.mode == MODE_TEXT for r in routes),
        "ocr_pages": sum(r.mode == MODE_OCR for r in routes),
        "table_pages": sum(r.has_tables for r in routes),
    }
