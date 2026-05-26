"""
cli_render 纯渲染函数单测（无需 DB / typer）。

这些函数对入参做鸭子类型，用轻量 stub 即可测——这正是把它们从 cli.py
拆出来的收益：脱离命令上下文独立验证。
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from contract_archive.cli_render import (
    color_legend,
    display_amount,
    extracted_terms,
    local_time,
    period_str,
    render_highlighted,
    seal_rows_to_dict,
    subject_of,
)


class _Row:
    """鸭子类型 stub：cli_render 只用到这些属性 + details()。"""

    def __init__(self, **kw):
        self._details = kw.pop("details", {})
        self.primary_amount_value = kw.pop("primary_amount_value", None)
        self.party_a = kw.pop("party_a", None)
        self.party_b = kw.pop("party_b", None)

    def details(self) -> dict:
        return self._details


def test_period_str():
    assert period_str({"period_start": "2025-01-01", "period_end": "2025-12-31"}) == \
        " [dim][2025-01-01~2025-12-31][/dim]"
    assert period_str({}) == ""
    assert period_str({"period_start": "2025-01-01", "period_end": None}) == \
        " [dim][2025-01-01~?][/dim]"


def test_display_amount():
    # 有计算合计：优先显示并标 *
    assert display_amount(_Row(details={"computed_total_value": 123456.0},
                               primary_amount_value=999.0)) == "¥123,456[cyan]*[/cyan]"
    # 无计算合计：回退抽取主金额
    assert display_amount(_Row(details={}, primary_amount_value=20000.0)) == "¥20,000"
    # 都没有
    assert display_amount(_Row(details={}, primary_amount_value=None)) == "-"


def test_subject_of():
    assert subject_of(_Row(details={"parties": ["甲", "乙", "丙"]})) == "甲、乙"  # 取前 2
    assert subject_of(_Row(details={}, party_a="张三", party_b="李四")) == "张三、李四"  # 回退甲乙方
    assert subject_of(_Row(details={})) == "-"
    long = subject_of(_Row(details={"parties": ["这是一个非常非常非常非常非常长的公司名称有限公司"]}))
    assert len(long) == 20 and long.endswith("…")


def test_local_time_timezone(monkeypatch):
    """UTC 存储 → 本地时区展示；解析失败原样返回。"""
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    time.tzset()
    assert local_time("2026-05-24T23:05:04Z") == "2026-05-25 07:05:04"  # +8
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    assert local_time("2026-05-24T23:05:04Z") == "2026-05-24 23:05:04"
    assert local_time(None) == "-"
    assert local_time("garbage") == "garbage"


# ---------- raw 原文高亮 ----------

_ESC = "\033["  # ANSI 前缀；纯文本路径里绝不该出现


class _ExtractRow:
    """extracted_terms 的鸭子 stub：只需顶层抽取列 + details()。"""

    def __init__(self, **kw):
        self.contract_name = kw.get("contract_name")
        self.party_a = kw.get("party_a")
        self.party_b = kw.get("party_b")
        self.amount_text = kw.get("amount_text")
        self.sign_date = kw.get("sign_date")
        self.expire_date = kw.get("expire_date")
        self.risk_clauses = kw.get("risk_clauses", [])
        self.obligations = kw.get("obligations", [])
        self._details = kw.get("details", {})

    def details(self) -> dict:
        return self._details


def test_extracted_terms_collects_and_categorizes():
    row = _ExtractRow(
        contract_name="车位转让协议",
        party_a="示例置业有限公司",
        party_b="张三",
        amount_text="人民币贰万元整",
        sign_date="2025-03-15",
        risk_clauses=["违约金不超过合同总金额的20%"],
        obligations=[SimpleNamespace(evidence="乙方应于签约当日支付定金")],
        details={
            "parties": ["示例置业有限公司", "张三"],
            "amounts": [{"text": "¥20000", "value": 20000.0, "evidence": ""}],
            "key_dates": [{"label": "签订日", "date": "2025-03-15"}],
            "fields": [{"label": "车位号", "value": "B2-108"}],
            "seals": [{"owner": "示例置业有限公司", "raw_text": "示例置业有限公司 合同专用章"}],
        },
    )
    terms = extracted_terms(row)
    assert terms["示例置业有限公司"] == "party"
    assert terms["张三"] == "party"               # 2 字保留
    assert terms["人民币贰万元整"] == "amount"
    assert terms["¥20000"] == "amount"
    assert terms["2025-03-15"] == "date"
    assert terms["B2-108"] == "field"
    assert terms["违约金不超过合同总金额的20%"] == "risk"
    assert terms["乙方应于签约当日支付定金"] == "field"
    assert terms["示例置业有限公司 合同专用章"] == "field"


def test_extracted_terms_drops_short_and_nonstr():
    row = _ExtractRow(party_a="甲", party_b="李四", details={"fields": [{"value": "1"}]})
    terms = extracted_terms(row)
    assert "甲" not in terms and "1" not in terms   # <2 字丢弃，防满屏误命中
    assert "李四" in terms


def test_render_highlighted_verbatim_hit_and_passthrough():
    text = "甲方：示例置业有限公司\n乙方：张三"
    out = render_highlighted(text, {"示例置业有限公司": "party", "张三": "party"})
    assert "示例置业有限公司" in out and _ESC in out
    # 命中串被同一对 ANSI 包裹
    assert "\033[1;36m示例置业有限公司\033[0m" in out
    # 空 terms 原样返回（保护管道纯文本）
    assert render_highlighted(text, {}) == text


def test_render_highlighted_longer_term_wins_no_double_wrap():
    """长串优先：'示例置业有限公司'整体命中，内部'示例置业'不再单独着色（不重叠）。"""
    text = "盖章单位：示例置业有限公司"
    out = render_highlighted(text, {"示例置业有限公司": "party", "示例置业": "field"})
    assert "\033[1;36m示例置业有限公司\033[0m" in out
    assert out.count(_ESC) == 2  # 仅一对 on/off，没有嵌套的第二种颜色


def test_render_highlighted_normalized_date_misses():
    """ISO 日期是规范化值，原文写法不同 → substring 命不中 → 不高亮（无误标）。"""
    text = "签订日期：2025年3月15日"
    assert render_highlighted(text, {"2025-03-15": "date"}) == text


def test_render_highlighted_escapes_regex_specials():
    text = "出租方：示例(中国)有限公司"
    out = render_highlighted(text, {"示例(中国)有限公司": "party"})
    assert "\033[1;36m示例(中国)有限公司\033[0m" in out


def test_color_legend_only_used_categories():
    assert color_legend({}) == ""
    legend = color_legend({"示例公司": "party", "¥1": "amount"})
    assert "当事人" in legend and "金额" in legend
    assert "风险" not in legend and "日期" not in legend


def test_seal_rows_to_dict():
    from contract_archive.archive import SealRow

    rows = [SealRow(seal_id=1, doc_id=3, title="认购协议",
                    owner="示例", seal_type="合同专用章", raw_text="示例 合同专用章")]
    assert seal_rows_to_dict(rows) == [
        {"doc_id": 3, "title": "认购协议", "owner": "示例",
         "seal_type": "合同专用章", "raw_text": "示例 合同专用章"}
    ]
    assert seal_rows_to_dict([]) == []
