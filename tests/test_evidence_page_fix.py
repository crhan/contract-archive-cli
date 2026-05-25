"""
出处页码校正的纯逻辑测试：用 content_list 的 page_idx 覆盖 LLM 猜的页码。
不联网——构造 blocks / 临时 content_list.json 验证反查与重写。
"""
from __future__ import annotations

import json

from contract_archive.extraction.evidence_page_fix import (
    _correct_evidence,
    _find_page,
    correct_evidence_pages,
)
from contract_archive.schemas import (
    Completeness,
    CompletenessIssue,
    DocumentExtraction,
    LabeledAmount,
)

# 29 号实景：占用费段真实 page_idx=5（PDF 第6页），LLM 却填了第5页。
_OCCUPY = "4、本协议签订后……逾期未交还的应按每日300元的标准支付车位占用费直至交还时止。"
_BLOCKS = [
    ("车位使用权转让总价（人民币)：200000元整。车位按个计价", 0),
    ("乙方应在本协议签订当日支付首期车位使用权转让价款（人民币）500000元整", 1),
    (_OCCUPY, 5),
]


def test_find_page_hits_correct_idx():
    assert _find_page("逾期未交还的应按每日300元的标准支付车位占用费", _BLOCKS) == 5
    assert _find_page("乙方应在本协议签订当日支付首期车位使用权转让价款", _BLOCKS) == 1


def test_find_page_too_short_or_miss_returns_none():
    assert _find_page("第6页", _BLOCKS) is None           # 太短
    assert _find_page("完全不存在的一段文字内容啊", _BLOCKS) is None  # 不匹配


def test_correct_single_evidence_fixes_off_by_one():
    """占用费：LLM 填第5页 → 校正为真实第6页。"""
    ev = "第5页 + 逾期未交还的应按每日300元的标准支付车位占用费直至交还时止。"
    assert _correct_evidence(ev, _BLOCKS).startswith("第6页 + ")


def test_signature_evidence_untouched():
    """签章式出处无'+片段'，正则不匹配，原样保留（VL 给的页码本就准）。"""
    ev = "据落款页图：第8页"
    assert _correct_evidence(ev, _BLOCKS) == ev


def test_concatenated_evidence_each_pair_corrected():
    """amount 类 issue 的拼接 evidence：逐对各自校正。"""
    ev = ("首期款：第7页 + 乙方应在本协议签订当日支付首期车位使用权转让价款；"
          "总价：第9页 + 车位使用权转让总价（人民币)：200000元整")
    out = _correct_evidence(ev, _BLOCKS)
    assert "第2页 + 乙方应在本协议签订当日" in out  # page_idx=1 → 第2页
    assert "第1页 + 车位使用权转让总价" in out      # page_idx=0 → 第1页


def test_unmatched_fragment_keeps_original_page():
    """反查不到片段时不瞎改，保留 LLM 原页码（诚实降级）。"""
    ev = "第3页 + 这段文字在原文里根本不存在所以查不到"
    assert _correct_evidence(ev, _BLOCKS) == ev


def test_correct_evidence_pages_no_content_list_returns_false(tmp_path):
    env = DocumentExtraction(amounts=[LabeledAmount(label="x", text="1", evidence="第1页 + y")])
    assert correct_evidence_pages(env, tmp_path) is False


def test_correct_evidence_pages_end_to_end(tmp_path):
    """端到端：造 content_list.json，校正 amounts 与 completeness issue 的页码。"""
    auto = tmp_path / "_mineru_raw" / "doc" / "auto"
    auto.mkdir(parents=True)
    (auto / "doc_content_list.json").write_text(
        json.dumps([{"text": t, "page_idx": p} for t, p in _BLOCKS], ensure_ascii=False),
        encoding="utf-8",
    )
    env = DocumentExtraction(
        amounts=[LabeledAmount(
            label="逾期占用费", text="每日300元",
            evidence="第5页 + 逾期未交还的应按每日300元的标准支付车位占用费",
        )],
        completeness=Completeness(status="incomplete", issues=[
            CompletenessIssue(item="x", category="amount",
                              evidence="第99页 + 乙方应在本协议签订当日支付首期车位使用权转让价款"),
            CompletenessIssue(item="甲方签章", category="signature",
                              evidence="据落款页图：第8页"),
        ]),
    )
    assert correct_evidence_pages(env, tmp_path) is True
    assert env.amounts[0].evidence.startswith("第6页 + ")          # 5→6
    assert env.completeness.issues[0].evidence.startswith("第2页 + ")  # 99→2
    assert env.completeness.issues[1].evidence == "据落款页图：第8页"   # 签章不动
