"""
多模态签章核查的纯逻辑测试：VL 结果 → 签章缺陷 issues。
不调 VL（不联网）——只验证 _issues_from_vision 的判定与健壮性。
"""
from __future__ import annotations

from pathlib import Path

from contract_archive.extraction.vision_seal import (
    _attach_seal_identities,
    _issues_from_vision,
    _signature_evidence,
)
from contract_archive.schemas import DocumentExtraction, LabeledValue, PersonIdentity


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


# ---- 印章绑主体（跨合同章号核对的入口）。PII 用占位：示例置业 + 假章号 ----


def _parsed_with_seal(owner, seal_text, role="甲方"):
    return {"units": [{"agreement": "主协议", "page": 8, "parties": [
        {"role": role, "has_seal": True, "has_signature": False,
         "seal_owner": owner, "seal_text": seal_text}]}]}


def test_attach_seal_creates_party_identity():
    """VL 读出甲方章 → 以章主体名建 person_identity，章号作 印章 标识。"""
    env = DocumentExtraction(doc_type="合同协议")
    _attach_seal_identities(env, _parsed_with_seal("示例置业有限公司", "合同专用章 123456"))
    pid = next(p for p in env.person_identities if p.name == "示例置业有限公司")
    assert any(i.label == "印章" and i.value == "合同专用章 123456" for i in pid.identifiers)


def test_attach_seal_appends_to_existing_party():
    """章主体与头部已抽的主体同名 → 追加印章标识，不新建主体（对应关系落点）。"""
    env = DocumentExtraction(doc_type="合同协议", person_identities=[
        PersonIdentity(name="示例置业有限公司", role="甲方",
                       identifiers=[LabeledValue(label="银行账号", value="6222000")])])
    _attach_seal_identities(env, _parsed_with_seal("示例置业有限公司", "合同专用章 123456"))
    assert len(env.person_identities) == 1
    assert {i.label for i in env.person_identities[0].identifiers} == {"银行账号", "印章"}


def test_attach_seal_skips_when_no_owner_or_no_seal():
    env = DocumentExtraction(doc_type="合同协议")
    # 有章但读不出 owner/seal_text → 没法核对，不绑
    _attach_seal_identities(env, {"units": [{"parties": [
        {"role": "甲方", "has_seal": True, "seal_owner": "", "seal_text": ""}]}]})
    # 无章 → 不绑
    _attach_seal_identities(env, {"units": [{"parties": [
        {"role": "甲方", "has_seal": False, "seal_owner": "X", "seal_text": "Y"}]}]})
    assert env.person_identities == []


def test_attach_seal_no_duplicate():
    """同一章号重复出现（多落款页）不重复追加。"""
    env = DocumentExtraction(doc_type="合同协议")
    _attach_seal_identities(env, _parsed_with_seal("示例置业有限公司", "合同专用章 123456"))
    _attach_seal_identities(env, _parsed_with_seal("示例置业有限公司", "合同专用章 123456"))
    seals = [i for i in env.person_identities[0].identifiers if i.label == "印章"]
    assert len(seals) == 1
