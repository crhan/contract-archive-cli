import fitz

from contract_archive.utils.pdf import (
    TextLayerStats,
    count_text_pages,
    is_text_layer_usable,
)


def _write_pdf(path, page_texts):
    """合成一个 PDF：每个元素是一页的文本（空串 = 纯扫描空白页，无文本层）。"""
    doc = fitz.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 100), text, fontsize=11)
        doc.save(str(path))
    finally:
        doc.close()


def test_garbled_text_layer_is_not_usable():
    stats = TextLayerStats(
        pages=7,
        chars=18000,
        non_ws_chars=15600,
        printable_chars=7800,
        cjk_chars=0,
        control_chars=4300,
        replacement_chars=0,
    )

    assert is_text_layer_usable(stats) is False


def test_readable_chinese_text_layer_is_usable():
    # 2 页都有可用中文文本，pages_with_text=2（与文本统计自洽）；纯扫描页数为 0，
    # 不触发覆盖率门槛。
    stats = TextLayerStats(
        pages=2,
        chars=2000,
        non_ws_chars=1600,
        printable_chars=1500,
        cjk_chars=900,
        control_chars=0,
        replacement_chars=0,
        pages_with_text=2,
    )

    assert is_text_layer_usable(stats) is True


def test_sparse_coverage_scanned_pdf_is_not_usable():
    # 184 页扫描合同，仅 2 页保单信息页有原生文本层（~9500 字、质量很高），
    # 覆盖率 ~1%、纯扫描页 182。质量比例达标但必须判为需要 OCR，否则条款扫描图被整体跳过。
    stats = TextLayerStats(
        pages=184,
        chars=11000,
        non_ws_chars=9500,
        printable_chars=9400,
        cjk_chars=6000,
        control_chars=0,
        replacement_chars=0,
        pages_with_text=2,
    )

    assert is_text_layer_usable(stats) is False


def test_small_scanned_pdf_with_one_text_page_is_not_usable():
    # 4 页扫描件夹 1 页保单文本（质量很高），覆盖率 25%、纯扫描页 3。
    # 旧的 pages<5 豁免会放过它，导致 3 页扫描条款被跳过 OCR；按"纯扫描页数"判据应判需 OCR。
    stats = TextLayerStats(
        pages=4,
        chars=11000,
        non_ws_chars=9500,
        printable_chars=9400,
        cjk_chars=6000,
        control_chars=0,
        replacement_chars=0,
        pages_with_text=1,
    )

    assert is_text_layer_usable(stats) is False


def test_short_all_text_certificate_is_usable():
    # 真·短凭证：3 页页页有原生文本，纯扫描页数为 0，不该被覆盖率门槛误伤。
    stats = TextLayerStats(
        pages=3,
        chars=4000,
        non_ws_chars=3000,
        printable_chars=2900,
        cjk_chars=1800,
        control_chars=0,
        replacement_chars=0,
        pages_with_text=3,
    )

    assert is_text_layer_usable(stats) is True


def test_full_coverage_electronic_pdf_is_usable():
    # 真·电子保单：100 页几乎页页有文本，覆盖率高，应继续用文本层、不浪费 OCR。
    stats = TextLayerStats(
        pages=100,
        chars=60000,
        non_ws_chars=50000,
        printable_chars=49000,
        cjk_chars=30000,
        control_chars=0,
        replacement_chars=0,
        pages_with_text=98,
    )

    assert is_text_layer_usable(stats) is True


def test_count_text_pages_counts_only_substantial_pages(tmp_path):
    pdf = tmp_path / "mixed.pdf"
    _write_pdf(
        pdf,
        [
            "This page carries plenty of real text content to count.",  # 文本页
            "",                                                          # 纯扫描空白页
            "short",                                                     # 文本太少，低于门槛
            "Another page with more than enough characters present.",    # 文本页
        ],
    )

    pages_with_text, total = count_text_pages(pdf)

    assert total == 4
    assert pages_with_text == 2


def test_count_text_pages_all_blank(tmp_path):
    pdf = tmp_path / "blank.pdf"
    _write_pdf(pdf, ["", "", ""])

    assert count_text_pages(pdf) == (0, 3)


def test_count_text_pages_respects_page_min_chars(tmp_path):
    pdf = tmp_path / "tuned.pdf"
    _write_pdf(pdf, ["abcde"])  # 5 个非空白字符

    assert count_text_pages(pdf, page_min_chars=10) == (0, 1)
    assert count_text_pages(pdf, page_min_chars=3) == (1, 1)
