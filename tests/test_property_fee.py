"""
月物业费派生估算的纯逻辑测试：amounts(单价) + fields(建筑面积) → 月物业费。
不联网、不调 LLM——只验证 estimate_monthly_property_fee 的确定性算术与边界。
"""
from __future__ import annotations

from contract_archive.extraction.property_fee import estimate_monthly_property_fee
from contract_archive.schemas import LabeledAmount, LabeledValue


def _price(label, value, unit):
    """构造单价类 LabeledAmount 的简写。"""
    return LabeledAmount(label=label, text=f"{value}{unit}", value=value, unit=unit)


def _area(label, value):
    return LabeledValue(label=label, value=f"{value} 平方米")


def test_sum_per_sqm_prices_times_area():
    """三项按㎡单价之和 × 建筑面积 = 月物业费（示例苑 102 室真实口径）。"""
    amounts = [
        _price("物业服务费", 2.25, "元/月·㎡"),
        _price("服务费", 4.55, "元/月·㎡"),
        _price("能耗费", 0.8, "元/月·㎡"),
        _price("地下车位管理费", 100.0, "元/个/月"),  # 量纲不同，不并入
        LabeledAmount(label="总价款", text="12279889元", value=12279889.0,
                      unit=None, is_total_component=True),  # 绝对金额，不并入
    ]
    fields = [
        _area("预测建筑面积", 286.92),
        _area("套内建筑面积", 274.86),
        _area("分摊共有建筑面积", 12.06),
    ]
    value, text = estimate_monthly_property_fee(amounts, fields)
    assert value == 2180.59          # (2.25+4.55+0.8) × 286.92 = 7.6 × 286.92
    assert "286.92" in text and "7.6" in text


def test_carpark_only_no_estimate():
    """只有"元/个/月"量纲（无按㎡项）→ 不估算（量纲不匹配建筑面积）。"""
    amounts = [_price("地下车位管理费", 100.0, "元/个/月")]
    fields = [_area("预测建筑面积", 286.92)]
    assert estimate_monthly_property_fee(amounts, fields) == (None, None)


def test_missing_area_no_estimate():
    """有单价但抽不到建筑面积 → 不硬凑。"""
    amounts = [_price("物业服务费", 2.25, "元/月·㎡")]
    assert estimate_monthly_property_fee(amounts, []) == (None, None)


def test_only_inner_area_excluded():
    """只有套内/分摊面积（无总建筑面积）→ 排除后无可用面积，不估算。"""
    amounts = [_price("物业服务费", 2.25, "元/月·㎡")]
    fields = [
        _area("套内建筑面积", 274.86),
        _area("分摊共有建筑面积", 12.06),
    ]
    assert estimate_monthly_property_fee(amounts, fields) == (None, None)


def test_plain_building_area_label():
    """面积标签为纯"建筑面积"（无"预测"前缀）也能识别。"""
    amounts = [_price("物业费", 3.0, "元/月·平方米")]
    fields = [_area("建筑面积", 100.0)]
    value, _ = estimate_monthly_property_fee(amounts, fields)
    assert value == 300.0


def test_absolute_amount_not_treated_as_price():
    """绝对金额（unit=None）不被当作单价并入——避免量纲污染。"""
    amounts = [
        LabeledAmount(label="合同总价", text="100万元", value=1000000.0, unit=None),
    ]
    fields = [_area("建筑面积", 100.0)]
    assert estimate_monthly_property_fee(amounts, fields) == (None, None)
