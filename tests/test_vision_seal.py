"""
多模态签章核查的纯逻辑测试：VL 结果 → 签章缺陷 issues。
不调 VL（不联网）——只验证 _issues_from_vision 的判定与健壮性。
"""
from __future__ import annotations

from contract_archive.extraction.vision_seal import _issues_from_vision


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
