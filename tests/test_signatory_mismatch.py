"""
回归：落款签字人 vs 当事人名单交叉核对（VL 方案）。
场景源自真实 case 4a26ef74b79b——补充协议乙方落款"王五"，而乙方是张三、李四。
"""
from __future__ import annotations

from contract_archive.extraction.vision_seal import _signatory_mismatch_issues
from contract_archive.schemas import DocumentExtraction


def _env(parties):
    return DocumentExtraction(doc_type="合同协议", parties=parties)


def test_signer_not_in_parties_is_flagged():
    """补充协议乙方落款王五，不在 [示例置业,张三,李四] → 报不符。"""
    env = _env(["示例置业有限公司", "张三", "李四"])
    parsed = {"units": [{
        "agreement": "补充协议", "page": 9,
        "parties": [{"role": "乙方", "has_signature": True, "signature_name": "王五"}],
    }]}
    issues = _signatory_mismatch_issues(env, parsed)
    assert len(issues) == 1
    assert issues[0].category == "signature"
    assert "王五" in issues[0].detail and "落款人与当事人不符" in issues[0].item
    assert "第 9 页" in issues[0].evidence


def test_signer_in_parties_no_issue():
    """主协议乙方落款张三，在名单内 → 不报。"""
    env = _env(["示例置业有限公司", "张三", "李四"])
    parsed = {"units": [{
        "agreement": "主协议", "page": 8,
        "parties": [{"role": "乙方", "has_signature": True, "signature_name": "张三"}],
    }]}
    assert _signatory_mismatch_issues(env, parsed) == []


def test_substring_match_no_false_positive():
    """名字带后缀（张三（买受人））双向子串匹配 → 不误报。"""
    env = _env(["张三"])
    parsed = {"units": [{"agreement": "主协议", "page": 1,
                         "parties": [{"role": "乙方", "signature_name": "张三（买受人）"}]}]}
    assert _signatory_mismatch_issues(env, parsed) == []


def test_no_signature_name_not_judged():
    """无 signature_name（只盖章/空白）不在此核查范围 → 不报。"""
    env = _env(["张三"])
    parsed = {"units": [{"agreement": "主协议", "page": 1,
                         "parties": [{"role": "乙方", "has_seal": True, "seal_owner": "示例公司"}]}]}
    assert _signatory_mismatch_issues(env, parsed) == []


def test_no_parties_no_judge():
    """当事人名单为空时无从核对 → 不报。"""
    parsed = {"units": [{"agreement": "主协议", "page": 1,
                         "parties": [{"role": "乙方", "signature_name": "王五"}]}]}
    assert _signatory_mismatch_issues(_env([]), parsed) == []
