"""页级分流 classify_pages / find_tables / routing_summary 单测。

合成 PDF（fitz）覆盖：纯文本页、扫描/空白页、文本太少页、乱码文本层、含表格页。
find_tables 的 try/except 兜底单独验。不碰网络。
"""
from __future__ import annotations

import fitz

from contract_archive.utils import classify_pages, routing_summary
from contract_archive.utils.page_router import (
    MODE_OCR,
    MODE_TEXT,
    PageRoute,
    find_tables,
)


# 用英文正文：fitz 默认 helvetica 无 CJK 字形，插入中文保存后会被替换成 U+00B7，
# 重新抽取就不是 CJK 了（夹具假象）。分流判据本身语言无关，英文足以验"够实质 vs 稀疏/扫描"。
_TEXT_BODY = "This insurance policy certificate page carries plenty of readable english content present here."


def _write_pdf(path, page_texts):
    """每个元素是一页的文本（空串 = 纯扫描空白页，无文本层）。"""
    doc = fitz.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 100), text, fontsize=11)
        doc.save(str(path))
    finally:
        doc.close()


def test_substantial_text_page_routes_to_text(tmp_path):
    pdf = tmp_path / "text.pdf"
    _write_pdf(pdf, [_TEXT_BODY])  # 足量英文正文（>50 非空白字），无表格 → text
    routes = classify_pages(pdf)
    assert len(routes) == 1
    assert routes[0].mode == MODE_TEXT
    assert routes[0].has_text is True
    assert routes[0].reason == "native-text"


def test_blank_scanned_page_routes_to_ocr(tmp_path):
    pdf = tmp_path / "blank.pdf"
    _write_pdf(pdf, [""])  # 无文本层
    routes = classify_pages(pdf)
    assert routes[0].mode == MODE_OCR
    assert routes[0].has_text is False
    assert routes[0].reason == "no-usable-text"


def test_sparse_text_page_routes_to_ocr(tmp_path):
    pdf = tmp_path / "sparse.pdf"
    _write_pdf(pdf, ["短"])  # 远低于 50 字门槛
    routes = classify_pages(pdf)
    assert routes[0].mode == MODE_OCR
    assert routes[0].char_count < 50


def test_mixed_doc_routes_per_page(tmp_path):
    pdf = tmp_path / "mixed.pdf"
    _write_pdf(
        pdf,
        [
            _TEXT_BODY,                         # 文本页
            "",                                 # 扫描空白页
            "Another page with " + _TEXT_BODY,  # 文本页
        ],
    )
    routes = classify_pages(pdf)
    assert [r.mode for r in routes] == [MODE_TEXT, MODE_OCR, MODE_TEXT]
    assert routing_summary(routes) == {
        "total": 3,
        "text_pages": 2,
        "ocr_pages": 1,
        "table_pages": 0,
    }


def test_garbled_text_layer_routes_to_ocr(tmp_path):
    pdf = tmp_path / "garbled.pdf"
    # 大量控制字混入：有"文字层"但全是垃圾 → 主判据判不可用 → ocr
    _write_pdf(pdf, ["\x01\x02\x03\x04\x05" * 60 + "保险"])
    routes = classify_pages(pdf)
    assert routes[0].mode == MODE_OCR
    assert routes[0].has_text is False


def test_table_page_routes_to_ocr_even_with_text(tmp_path):
    """含实质表格的页即便有文本层也走 VL——原生 get_text 抹平表格结构。"""
    doc = fitz.open()
    page = doc.new_page()
    # 足量英文正文让 has_text 成立，再叠一个闭合边框的 3x3 表格让 find_tables 命中
    page.insert_text((72, 72), _TEXT_BODY, fontsize=10)
    x0, y0, cw, ch = 72, 120, 80, 30
    rows, cols = 3, 3
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * ch), (x0 + cols * cw, y0 + r * ch))
    for c in range(cols + 1):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + rows * ch))
    for r in range(rows):
        for c in range(cols):
            page.insert_text((x0 + c * cw + 5, y0 + r * ch + 18), f"cell{r}{c}", fontsize=8)
    pdf = tmp_path / "table.pdf"
    doc.save(str(pdf))
    doc.close()

    routes = classify_pages(pdf)
    assert routes[0].has_text is True  # 有足量文本层
    assert routes[0].has_tables is True  # 但含表格
    assert routes[0].mode == MODE_OCR  # → 改走 VL
    assert routes[0].reason == "table-present"


def test_find_tables_swallows_errors():
    """find_tables 任何异常都兜底为 False，不影响主判据。"""

    class _BoomPage:
        def find_tables(self):
            raise RuntimeError("boom")

    assert find_tables(_BoomPage()) is False


def test_page_route_is_frozen():
    r = PageRoute(0, MODE_TEXT, True, False, 100)
    try:
        r.mode = MODE_OCR  # type: ignore[misc]
    except Exception as e:
        assert isinstance(e, (AttributeError, TypeError))
    else:
        raise AssertionError("PageRoute 应为 frozen，不可改字段")
