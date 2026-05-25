"""
多模态签章核查的纯逻辑测试：VL 结果 → 签章缺陷 issues。
不调 VL（不联网）——只验证 _issues_from_vision 的判定与健壮性。
"""
from __future__ import annotations

from pathlib import Path

from contract_archive.extraction.vision_seal import _issues_from_vision, _signature_evidence


def test_only_missing_reported():
    """28 号场景：甲方空白=缺；乙方有签字/章=不报。只列缺的。"""
    parsed = {
        "units": [
            {
                "agreement": "主协议",
                "parties": [
                    {"role": "甲方", "has_seal": False, "has_signature": False},
                    {"role": "乙方", "has_seal": False, "has_signature": True},
                ],
            },
            {
                "agreement": "补充协议",
                "parties": [
                    {"role": "甲方", "has_seal": False, "has_signature": False},
                    {"role": "乙方", "has_seal": True, "has_signature": False},
                ],
            },
        ]
    }
    issues = _issues_from_vision(parsed)
    assert [i.item for i in issues] == ["主协议·甲方签章", "补充协议·甲方签章"]
    assert all(i.category == "signature" for i in issues)


def test_empty_when_all_signed():
    parsed = {
        "units": [
            {
                "agreement": "主协议",
                "parties": [
                    {"role": "甲方", "has_seal": True, "has_signature": False},
                    {"role": "乙方", "has_seal": False, "has_signature": True},
                ],
            }
        ]
    }
    assert _issues_from_vision(parsed) == []


def test_robust_against_garbage():
    assert _issues_from_vision({}) == []
    assert _issues_from_vision({"units": "bad"}) == []
    # 无 role 的 party 跳过，不误报
    assert _issues_from_vision({"units": [{"parties": [{"has_seal": False}]}]}) == []


def test_signature_evidence_from_page_names():
    imgs = [Path("/x/preview_images/page_008.png"), Path("/x/preview_images/page_009.png")]
    assert _signature_evidence(imgs) == "据落款页图：第 8、9 页"


def test_issue_falls_back_when_no_page():
    """unit 没回填 page 时，用 fallback 出处。"""
    parsed = {"units": [{"agreement": "主协议", "parties": [
        {"role": "甲方", "has_seal": False, "has_signature": False}]}]}
    issues = _issues_from_vision(parsed, "据落款页图：第 8、9 页")
    assert issues[0].evidence == "据落款页图：第 8、9 页"


def test_issue_uses_each_unit_page():
    """各落款区用自己回填的 page：主协议→第8页、补充协议→第9页，不再笼统堆叠。"""
    parsed = {"units": [
        {"agreement": "主协议", "page": 8, "parties": [
            {"role": "甲方", "has_seal": False, "has_signature": False}]},
        {"agreement": "补充协议", "page": 9, "parties": [
            {"role": "甲方", "has_seal": False, "has_signature": False}]},
    ]}
    issues = _issues_from_vision(parsed, "fallback")
    assert issues[0].evidence == "据落款页图：第 8 页"
    assert issues[1].evidence == "据落款页图：第 9 页"
