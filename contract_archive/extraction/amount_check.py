"""
金额自洽性校验：纯确定性数值规则，不依赖 LLM 算术。

为什么不交给 LLM：抽取时 LLM 会忠实抄录原文金额，但对"首期 500000 却 > 总价
200000"这类数量矛盾毫无警觉（实测 29 号车位即 LLM 照抄矛盾数字而未报）。算术与
比较交给代码，LLM 只负责语义标注（is_total_component / is_installment）。

两条规则，都只产出"疑似异常、请人工核对"——属辅助筛查、非终判：
  规则A 分期超额：各分期付款项(is_installment)之和 > 总价(合计) 即报（如首期50W＋
        余款15W=65W > 总价20W，首期多打一个0）。**只报正差**——分期和 < 总价是认购预付、
        首付+贷款等正常场景（认购预付50W+定金50W ≪ 房屋总价1228W），报了是误报。
  规则B 单项越界：未计入合计、也未标分期的单项金额却 > 合计，疑似多填/笔误，
        作为 LLM 漏标分期时的兜底。
"""
from __future__ import annotations

from ..schemas import CompletenessIssue, LabeledAmount


def _tolerance(total: float) -> float:
    """金额比较容差：取「1元」与「合计的 1%」中较大者。

    既容忍分/角小数舍入，又不会把数量级矛盾（50W vs 20W）当成噪声放过。
    """
    return max(1.0, abs(total) * 0.01)


def _merge_evidence(amounts: list[LabeledAmount]) -> str:
    """拼接各分期项的出处，便于翻回原文逐笔核对。"""
    parts = [f"{a.label}：{a.evidence}" for a in amounts if a.evidence]
    return "；".join(parts)


def check_amount_consistency(
    amounts: list[LabeledAmount], computed_total: float | None
) -> list[CompletenessIssue]:
    """
    金额自洽校验 → amount 类缺陷列表。

    无合计基准（computed_total 为空或 ≤0）时不校验——没有"总价"做参照，
    谈不上自洽，硬比会误报。
    """
    issues: list[CompletenessIssue] = []
    if not computed_total or computed_total <= 0:
        return issues
    tol = _tolerance(computed_total)

    # 规则A：分期付款项之和 **超过** 总价才报（只报正差）。
    # 不报负差：认购/预售先付定金预付款、首付+贷款等，已列分期项之和本就小于总价
    # 属正常（认购预付50W+定金50W ≪ 房屋总价1228W），报负差是误报。
    # 分期之和 > 总价 才是硬矛盾（29号首期50W＋余款15W=65W > 总价20W，首期误填）。
    installments = [a for a in amounts if a.is_installment and a.value is not None]
    if installments:
        inst_sum = round(sum(a.value for a in installments), 2)
        overshoot = round(inst_sum - computed_total, 2)
        if overshoot > tol:
            labels = "＋".join(a.label for a in installments)
            issues.append(CompletenessIssue(
                item="分期款超过总价",
                category="amount",
                detail=(
                    f"分期款之和（{labels}）={inst_sum:,.0f}元，"
                    f"超过总价/合计 {computed_total:,.0f}元（多 {overshoot:,.0f}元），"
                    f"疑似金额笔误，请人工核对"
                ),
                evidence=_merge_evidence(installments),
            ))

    # 规则B：未计入合计、也未标分期的单项却超过合计（兜底 LLM 漏标分期的场景）。
    for a in amounts:
        if (
            not a.is_total_component
            and not a.is_installment
            and a.value is not None
            and a.value > computed_total + tol
        ):
            issues.append(CompletenessIssue(
                item=f"{a.label}超过合计",
                category="amount",
                detail=(
                    f"{a.label}={a.value:,.0f}元 超过合计 {computed_total:,.0f}元，"
                    f"疑似多填/笔误，请人工核对"
                ),
                evidence=a.evidence,
            ))
    return issues
