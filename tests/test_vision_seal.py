"""
多模态签章核查的纯逻辑测试：VL 结果 → 签章缺陷 issues。
不调 VL（不联网）——只验证 _issues_from_vision 的判定与健壮性。
"""
from __future__ import annotations

from pathlib import Path

from contract_archive.extraction.vision_seal import (
    _attach_seal_identities,
    _issues_from_vision,
    _seal_number,
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


# ---- 印章绑主体（跨合同章号核对的入口）。PII 用占位：示例置业 + 虚构章号 ----

_FAKE_NO = "990011223344"  # 虚构章号占位


def _parsed_with_seal(owner="示例置业有限公司", seal_no=_FAKE_NO, role="甲方", seal_text=None):
    return {"units": [{"agreement": "主协议", "page": 8, "parties": [
        {"role": role, "has_seal": True, "has_signature": False,
         "seal_owner": owner, "seal_no": seal_no,
         "seal_text": seal_text if seal_text is not None else f"{owner} 合同专用章 {seal_no}"}]}]}


def test_attach_seal_binds_to_head_party_by_role():
    """头部已抽出甲方主体 → 章号按 role 匹配绑到它，不被 VL 章面误读带偏。"""
    env = DocumentExtraction(doc_type="合同协议", person_identities=[
        PersonIdentity(name="正确置业有限公司", role="甲方",
                       identifiers=[LabeledValue(label="银行账号", value="6222000")])])
    # VL 把章面主体名误读成别的，但 role=甲方 → 仍锚定到头部"正确置业"
    _attach_seal_identities(env, _parsed_with_seal(owner="误读置业有限公司", seal_no=_FAKE_NO))
    assert len(env.person_identities) == 1
    pid = env.person_identities[0]
    assert pid.name == "正确置业有限公司"        # 用头部名，不被章面误读带偏
    assert {i.label for i in pid.identifiers} == {"银行账号", "印章"}
    assert any(i.value == _FAKE_NO for i in pid.identifiers if i.label == "印章")


def test_attach_seal_binds_across_role_synonym():
    """认购协议：头部主体 role='出卖人'，VL 落款 role='甲方' → 阵营归组匹配，
    章号绑到正确头部主体，不再因'出卖人'不含'甲'字而漏匹配（浙典/浙奥分裂成因之一）。"""
    env = DocumentExtraction(doc_type="合同协议", person_identities=[
        PersonIdentity(name="示例置业有限公司", role="出卖人",
                       identifiers=[LabeledValue(label="电话", value="0571-1")])])
    _attach_seal_identities(env, _parsed_with_seal(owner="误读置业有限公司",
                                                   seal_no=_FAKE_NO, role="甲方"))
    assert len(env.person_identities) == 1                 # 不新建主体
    pid = env.person_identities[0]
    assert pid.name == "示例置业有限公司"                   # 绑到头部出卖人，非章面误读
    assert any(i.label == "印章" and i.value == _FAKE_NO for i in pid.identifiers)


def test_attach_seal_fallback_to_owner_when_no_head_match():
    """头部没抽到对应 role 主体 → 退回用章面 owner 兜底建主体。"""
    env = DocumentExtraction(doc_type="合同协议")
    _attach_seal_identities(env, _parsed_with_seal(owner="示例置业有限公司"))
    pid = next(p for p in env.person_identities if p.name == "示例置业有限公司")
    assert any(i.label == "印章" and i.value == _FAKE_NO for i in pid.identifiers)


def test_seal_number_prefers_seal_no_then_extracts():
    assert _seal_number({"seal_no": _FAKE_NO}) == _FAKE_NO
    # seal_no 缺 → 从 seal_text 提最长数字串
    assert _seal_number({"seal_text": f"合同专用章 (5) {_FAKE_NO}"}) == _FAKE_NO
    assert _seal_number({"seal_text": "无编号"}) == ""


def test_attach_seal_skips_when_no_seal_or_no_number():
    env = DocumentExtraction(doc_type="合同协议")
    # 无章 → 不绑
    _attach_seal_identities(env, {"units": [{"parties": [
        {"role": "甲方", "has_seal": False, "seal_no": _FAKE_NO}]}]})
    # 有章但读不出编号 → 不绑（没法核对）
    _attach_seal_identities(env, {"units": [{"parties": [
        {"role": "甲方", "has_seal": True, "seal_no": "", "seal_text": "模糊"}]}]})
    assert env.person_identities == []


def test_attach_seal_no_duplicate():
    """同一章号重复出现（多落款页）不重复追加。"""
    env = DocumentExtraction(doc_type="合同协议")
    _attach_seal_identities(env, _parsed_with_seal())
    _attach_seal_identities(env, _parsed_with_seal())
    seals = [i for i in env.person_identities[0].identifiers if i.label == "印章"]
    assert len(seals) == 1
