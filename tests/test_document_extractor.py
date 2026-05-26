"""
document_extractor 纯逻辑测试（不联网）：幻觉主体的确定性过滤。

PII 一律占位：示例置业 / 张三 + 虚构号码。
"""
from contract_archive.extraction.document_extractor import _filter_identities_by_text
from contract_archive.schemas import LabeledValue, PersonIdentity


def _pid(name, **ids):
    return PersonIdentity(
        name=name,
        role=None,
        identifiers=[LabeledValue(label=k, value=v) for k, v in ids.items()],
    )


def test_drops_name_absent_from_text():
    """正文只有『示例置业』，LLM 幻觉出正文不存在的『示例奥业』→ 丢弃后者。"""
    text = "出卖人（以下简称甲方）：示例置业有限公司\n买受人：张三"
    ids = [
        _pid("示例置业有限公司", 电话="0571-88880000"),
        _pid("示例奥业有限公司", 印章="990011223344"),   # 幻觉：正文无此名
        _pid("张三", 身份证号="110101199001011234"),
    ]
    kept = _filter_identities_by_text(ids, text)
    assert [p.name for p in kept] == ["示例置业有限公司", "张三"]


def test_keeps_name_with_internal_ocr_spaces():
    """正文里名字被 OCR 夹了空格，去空白后仍能匹配，不误丢。"""
    text = "出卖人：示例 置业 有限公司"
    kept = _filter_identities_by_text([_pid("示例置业有限公司", 电话="x")], text)
    assert [p.name for p in kept] == ["示例置业有限公司"]


def test_empty_text_keeps_all():
    """正文为空（无可校验）时不做删减——宁可不滤，避免误杀。"""
    ids = [_pid("示例置业有限公司", 电话="x")]
    assert _filter_identities_by_text(ids, "") == ids
