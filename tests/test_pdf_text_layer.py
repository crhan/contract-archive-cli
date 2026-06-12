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
