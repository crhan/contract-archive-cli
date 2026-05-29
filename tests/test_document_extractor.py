"""
document_extractor 纯逻辑测试（不联网）：幻觉主体的确定性过滤。

PII 一律占位：示例置业 / 张三 + 虚构号码。
"""
from contract_archive.extraction.document_extractor import (
    _coerce_labeled_amounts,
    _filter_identities_by_text,
)
from contract_archive.schemas import LabeledValue, PersonIdentity


def _pid(name, **ids):
    return PersonIdentity(
        name=name,
        role=None,
        identifiers=[LabeledValue(label=k, value=v) for k, v in ids.items()],
    )


# ---- is_total_component 代码强制不变量（纠正 LLM 误标，治"合计重复累加"）----


def test_installment_item_forced_out_of_total():
    """C1 病灶：LLM 把总价与其分期项都标 total_component → 合计 2×总价。
    代码不变量：is_installment=True 的项强制 is_total_component=False。"""
    raw = [
        {"label": "总价款", "text": "12279889元", "is_total_component": True},
        {"label": "首期房价款", "text": "1849889元",
         "is_total_component": True, "is_installment": True},   # LLM 误标 total
        {"label": "剩余房款", "text": "10430000元",
         "is_total_component": True, "is_installment": True},   # LLM 误标 total
    ]
    out = _coerce_labeled_amounts(raw)
    by_label = {a.label: a for a in out}
    assert by_label["总价款"].is_total_component is True
    assert by_label["首期房价款"].is_total_component is False
    assert by_label["剩余房款"].is_total_component is False
    # 合计 = 仅总价款，不重复累加
    total = sum(a.value for a in out if a.is_total_component)
    assert total == 12279889.0


def test_unit_price_forced_out_of_total():
    """单价项（unit 非空）即便被 LLM 误标 total_component，也压成 False（量纲不同）。"""
    raw = [
        {"label": "物业服务费", "text": "2.25元/月·㎡", "unit": "元/月·㎡",
         "is_total_component": True},
    ]
    out = _coerce_labeled_amounts(raw)
    assert out[0].is_total_component is False


def test_independent_components_keep_total():
    """无单一汇总项、各独立组成项（收入证明）→ 各自 total_component 保留。"""
    raw = [
        {"label": "年度税前收入", "text": "500000元", "is_total_component": True},
        {"label": "年度股权应税收益", "text": "120000元", "is_total_component": True},
    ]
    out = _coerce_labeled_amounts(raw)
    assert all(a.is_total_component for a in out)
    assert sum(a.value for a in out if a.is_total_component) == 620000.0


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
