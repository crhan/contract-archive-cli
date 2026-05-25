"""
cli_render 纯渲染函数单测（无需 DB / typer）。

这些函数对入参做鸭子类型，用轻量 stub 即可测——这正是把它们从 cli.py
拆出来的收益：脱离命令上下文独立验证。
"""
from __future__ import annotations

import time

from contract_archive.cli_render import (
    display_amount,
    local_time,
    period_str,
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


def test_seal_rows_to_dict():
    from contract_archive.archive import SealRow

    rows = [SealRow(seal_id=1, doc_id=3, title="认购协议",
                    owner="示例", seal_type="合同专用章", raw_text="示例 合同专用章")]
    assert seal_rows_to_dict(rows) == [
        {"doc_id": 3, "title": "认购协议", "owner": "示例",
         "seal_type": "合同专用章", "raw_text": "示例 合同专用章"}
    ]
    assert seal_rows_to_dict([]) == []
