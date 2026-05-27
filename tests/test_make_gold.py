"""make_gold 脱敏加固单测：中文大写转换器 + 证明类金额随机缩放。"""
from __future__ import annotations

import pytest

from evals.make_gold import int_to_cap, scrub_income_amounts


@pytest.mark.parametrize("n,exp", [
    (0, "零"),
    (4883, "肆仟捌佰捌拾叁"),
    (10000, "壹万"),
    (106643, "壹拾万陆仟陆佰肆拾叁"),
    (621106, "陆拾贰万壹仟壹佰零陆"),         # 组内零
    (1279720, "壹佰贰拾柒万玖仟柒佰贰拾"),
    (1000006, "壹佰万零陆"),                  # 组间零
    (12480360, "壹仟贰佰肆拾捌万零叁佰陆拾"),  # 组间零 + 进位
    (35964000, "叁仟伍佰玖拾陆万肆仟"),
])
def test_int_to_cap(n, exp):
    assert int_to_cap(n) == exp


def _income_env():
    return {
        "doc_type": "证明",
        "primary_amount_value": 1279720.0,
        "primary_amount_text": "人民币 壹佰贰拾柒万玖仟柒佰贰拾元整",
        "computed_total_value": 1900826.0,
        "amounts": [
            {"label": "年收入", "text": "人民币 壹佰贰拾柒万玖仟柒佰贰拾元整",
             "value": 1279720.0, "is_total_component": True},
            {"label": "股权收益", "text": "人民币陆拾贰万壹仟壹佰零陆元整",
             "value": 621106.0, "is_total_component": True},
            {"label": "公积金", "text": "肆仟捌佰捌拾叁元整",
             "value": 4883.0, "is_total_component": False},
        ],
    }


def test_scrub_income_scales_to_ten_million_and_consistent():
    env = _income_env()
    text = ("年收入人民币 壹佰贰拾柒万玖仟柒佰贰拾元整；"
            "股权人民币陆拾贰万壹仟壹佰零陆元整；公积金肆仟捌佰捌拾叁元整。")
    new_env, new_text, changed = scrub_income_amounts(env, text, seed=42)
    assert changed
    # 主金额到千万级
    assert 10_000_000 <= new_env["primary_amount_value"] <= 100_000_000
    # 旧大写已从文本消失（去真值）
    assert "壹佰贰拾柒万玖仟柒佰贰拾元整" not in new_text
    assert "陆拾贰万壹仟壹佰零陆元整" not in new_text
    # gold.text 与缩放后数值一致（大写可还原回 value）
    for a in new_env["amounts"]:
        assert int_to_cap(int(a["value"])) in a["text"]
        assert a["text"] in new_text          # input 与 gold 文本一致
    # computed_total = 两个 is_total_component 之和
    comps = [a["value"] for a in new_env["amounts"] if a["is_total_component"]]
    assert new_env["computed_total_value"] == pytest.approx(sum(comps))
    # 比例保持（缩放不破坏内部关系）：股权/年收入 比值不变
    ratio_old = 621106.0 / 1279720.0
    a_year = next(a for a in new_env["amounts"] if a["label"] == "年收入")
    a_eq = next(a for a in new_env["amounts"] if a["label"] == "股权收益")
    assert a_eq["value"] / a_year["value"] == pytest.approx(ratio_old, rel=0.001)


def test_scrub_income_scrubs_summary_arabic():
    """收入常被 LLM 写进 summary 的阿拉伯数字（含小数）——也要被替换掉，不只改 amounts。"""
    env = _income_env()
    env["summary"] = "证明其近12个月税前总收入1279720元及股权应税收益621106.71元。"
    new_env, _, _ = scrub_income_amounts(env, "", seed=7)
    assert "1279720" not in new_env["summary"]
    assert "621106" not in new_env["summary"]   # 含 621106.71 的整数段也清掉
    # summary 里应出现缩放后的新主金额
    assert str(int(new_env["primary_amount_value"])) in new_env["summary"]


def test_scrub_income_skips_non_certificate():
    env = {"doc_type": "合同协议", "primary_amount_value": 200000.0,
           "amounts": [{"label": "总价", "text": "贰拾万元整", "value": 200000.0,
                        "is_total_component": True}]}
    new_env, new_text, changed = scrub_income_amounts(env, "总价贰拾万元整", seed=1)
    assert changed is False
    assert new_env["primary_amount_value"] == 200000.0
