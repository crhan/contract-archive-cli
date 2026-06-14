from contract_archive.utils.pdf import TextLayerStats, is_text_layer_usable


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
    stats = TextLayerStats(
        pages=2,
        chars=2000,
        non_ws_chars=1600,
        printable_chars=1500,
        cjk_chars=900,
        control_chars=0,
        replacement_chars=0,
    )

    assert is_text_layer_usable(stats) is True


def test_sparse_coverage_scanned_pdf_is_not_usable():
    # 184 页扫描合同，仅 2 页保单信息页有原生文本层（~9500 字、质量很高），
    # 覆盖率 ~1%。质量比例达标但必须判为需要 OCR，否则 182 页条款扫描图被整体跳过。
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
