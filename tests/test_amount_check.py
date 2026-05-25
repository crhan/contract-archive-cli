"""
金额自洽校验的纯逻辑测试：amounts + 合计 → amount 类缺陷。
不联网、不调 LLM——只验证 check_amount_consistency 的确定性判定与边界。
"""
from __future__ import annotations

from contract_archive.extraction.amount_check import check_amount_consistency
from contract_archive.schemas import LabeledAmount


def _amt(label, value, *, total=False, installment=False, evidence=""):
    """构造 LabeledAmount 的简写。"""
    return LabeledAmount(
        label=label, text=f"{value}", value=value,
        is_total_component=total, is_installment=installment, evidence=evidence,
    )


def test_installment_sum_exceeds_total_flagged():
    """29号病灶：首期50W＋余款15W=65W ≠ 总价20W → 规则A 报分期失衡。"""
    amounts = [
        _amt("转让总价", 200000.0, total=True),
        _amt("首期款", 500000.0, installment=True, evidence="第3页第三条"),
        _amt("余款", 150000.0, installment=True, evidence="第3页第三条"),
    ]
    issues = check_amount_consistency(amounts, 200000.0)
    assert len(issues) == 1
    assert issues[0].category == "amount"
    assert issues[0].item == "分期款超过总价"
    # 出处拼接了各分期项，能翻回原文
    assert "首期款" in issues[0].evidence and "余款" in issues[0].evidence


def test_installment_below_total_not_flagged():
    """认购/预售：预付房款＋定金 < 房屋总价是正常的（余款待签正式合同再付），不报。

    这是 id=3 认购协议的真实场景——曾因规则 A 用'≠'报负差而误报，收紧为'只报正差'后修复。
    """
    amounts = [
        _amt("房屋总价", 12279889.0, total=True),
        _amt("预付房款", 500000.0, installment=True),
        _amt("定金", 500000.0, installment=True),
    ]
    assert check_amount_consistency(amounts, 12279889.0) == []


def test_installment_sum_matches_total_ok():
    """首期5W＋余款15W=20W=总价 → 自洽，不报（这才是原文该有的数字）。"""
    amounts = [
        _amt("转让总价", 200000.0, total=True),
        _amt("首期款", 50000.0, installment=True),
        _amt("余款", 150000.0, installment=True),
    ]
    assert check_amount_consistency(amounts, 200000.0) == []


def test_no_total_skips_check():
    """无合计基准（None/0）不校验——没有总价参照谈不上自洽，硬比会误报。"""
    amounts = [_amt("首期", 500000.0, installment=True)]
    assert check_amount_consistency(amounts, None) == []
    assert check_amount_consistency(amounts, 0) == []


def test_unit_prices_below_total_not_flagged():
    """服务费/占用费等小额单价远小于合计，不误报（它们非分期、非合计组件）。"""
    amounts = [
        _amt("转让总价", 200000.0, total=True),
        _amt("车位服务管理费", 100.0),   # 100元/月
        _amt("逾期占用费", 300.0),       # 每日300元
    ]
    assert check_amount_consistency(amounts, 200000.0) == []


def test_unmarked_single_amount_exceeds_total_flagged():
    """规则B 兜底：LLM 漏标 is_installment，但首期50W > 合计20W 仍能报出。"""
    amounts = [
        _amt("转让总价", 200000.0, total=True),
        _amt("首期款", 500000.0, evidence="第3页"),  # 漏标 installment
    ]
    issues = check_amount_consistency(amounts, 200000.0)
    assert len(issues) == 1
    assert issues[0].item == "首期款超过合计"
    assert issues[0].category == "amount"
    assert issues[0].evidence == "第3页"


def test_within_tolerance_not_flagged():
    """分期和与合计差在容差内（max(1元, 合计1%)）不报——容忍分/角舍入。"""
    amounts = [
        _amt("总价", 100.0, total=True),
        _amt("首期", 60.0, installment=True),
        _amt("余款", 40.5, installment=True),  # 和=100.5，差0.5 ≤ tol(=1)
    ]
    assert check_amount_consistency(amounts, 100.0) == []


def test_installment_and_overflow_dont_double_report():
    """分期项只走规则A、不走规则B：65W>20W 的首期不会再被规则B重复报一次。"""
    amounts = [
        _amt("转让总价", 200000.0, total=True),
        _amt("首期款", 500000.0, installment=True),
        _amt("余款", 150000.0, installment=True),
    ]
    issues = check_amount_consistency(amounts, 200000.0)
    # 只有规则A 一条，不会因首期50W>20W 再触发规则B
    assert len(issues) == 1
