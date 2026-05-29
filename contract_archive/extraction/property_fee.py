"""
周期性费用的月度估算：纯确定性派生，不依赖 LLM 算术。

为什么交给代码（同 amount_check.py 的哲学）：合同只给**单价**（物业服务费 2.25 元/月·㎡、
服务费 4.55、能耗费 0.8），买受人真正关心的是"一个月实付多少钱"。LLM 照抄单价可靠，
但让它把同量纲单价相加再乘建筑面积，既易算错又不可审计。算术与量纲判断交给代码，
LLM 只负责抽出每项单价及其 unit。

口径：
  月物业费 ≈ Σ(按建筑面积计价的物业类单价，元/月·㎡) × 建筑面积
  - **只并入「元/月·㎡」量纲**的项（unit 含 ㎡/平方米 且 含 月）；车位管理费"元/个/月"
    量纲不同（应乘车位数而非建筑面积），不并入——混加是量纲错误。
  - 建筑面积取「建筑面积/预测建筑面积」，**排除「套内/分摊」**——合同物业费明示按建筑面积计。

产出"估算、供参考"——单价/面积任一抽不到即返回 (None, None)，不硬凑。
"""
from __future__ import annotations

import re
from typing import Optional

from ..schemas import LabeledAmount, LabeledValue

# 面积数值解析：从"286.92 平方米""286.92㎡"里取第一个数。
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
# 建筑面积标签里需排除的限定词——要的是「总建筑面积」，非套内/分摊。
_AREA_EXCLUDE = ("套内", "分摊", "共有")


def _is_per_sqm_month(unit: Optional[str]) -> bool:
    """unit 是否为「元/月·㎡」量纲（按建筑面积、按月计的单价）。"""
    u = unit or ""
    has_sqm = "㎡" in u or "平方米" in u or "平米" in u or "m²" in u or "m2" in u.lower()
    has_month = "月" in u
    return has_sqm and has_month


def _find_building_area(fields: list[LabeledValue]) -> Optional[float]:
    """
    从 fields 找建筑面积数值（㎡）。优先「预测建筑面积」或纯「建筑面积」，
    排除「套内建筑面积/分摊共有建筑面积」（物业费按总建筑面积计）。抽不到返回 None。
    """
    candidates: list[tuple[str, float]] = []
    for f in fields:
        label = f.label or ""
        if "建筑面积" in label and not any(x in label for x in _AREA_EXCLUDE):
            m = _NUM_RE.search(f.value or "")
            if m:
                candidates.append((label, float(m.group(1))))
    if not candidates:
        return None
    # 优先"预测建筑面积"/纯"建筑面积"，否则取首个候选。
    for label, value in candidates:
        if "预测" in label or label.strip() == "建筑面积":
            return value
    return candidates[0][1]


def estimate_monthly_property_fee(
    amounts: list[LabeledAmount], fields: list[LabeledValue]
) -> tuple[Optional[float], Optional[str]]:
    """
    估算每月物业费 = Σ(按㎡·月计价的单价) × 建筑面积。

    返回 (月费数值, 算式说明)；缺单价或缺建筑面积时返回 (None, None)。
    算式说明 _text 保留各单价与面积，供人工翻回原文核对。
    """
    per_sqm = [a for a in amounts if a.value is not None and _is_per_sqm_month(a.unit)]
    if not per_sqm:
        return None, None
    area = _find_building_area(fields)
    if not area:
        return None, None

    unit_sum = round(sum(a.value for a in per_sqm if a.value is not None), 4)
    monthly = round(unit_sum * area, 2)
    parts = "＋".join(f"{a.label}{a.value:g}" for a in per_sqm)
    text = f"（{parts}）={unit_sum:g}元/月·㎡ × {area:g}㎡ ≈ {monthly:,.2f}元/月"
    return monthly, text
