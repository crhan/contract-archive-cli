"""
多模态签章核查的纯逻辑测试：VL 结果 → 签章缺陷 issues。
不调 VL（不联网）——只验证 _issues_from_vision 的判定与健壮性。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from contract_archive.extraction import vision_seal
from contract_archive.extraction.vision_seal import (
    _attach_seal_identities,
    _issues_from_vision,
    _seal_number,
    _signature_evidence,
)
from contract_archive.schemas import (
    Completeness,
    CompletenessIssue,
    DocumentExtraction,
    LabeledValue,
    PersonIdentity,
    SubAgreement,
)


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


# ---- LLM response boundary: malformed results must never clear text-derived issues ----


def _seal_settings():
    return SimpleNamespace(
        dashscope_vl_model="qwen-vl-test",
        dashscope_api_key="test-key",
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
    )


def _env_with_text_signature_issue() -> DocumentExtraction:
    return DocumentExtraction(
        doc_type="合同协议",
        completeness=Completeness(
            status="incomplete",
            issues=[
                CompletenessIssue(
                    item="文本抽取的甲方签章",
                    category="signature",
                    detail="文本层显示缺签章",
                )
            ],
        ),
    )


def test_invalid_seal_shapes_preserve_text_completeness(monkeypatch):
    """Invalid or partial VL JSON is a warning, never evidence that all parties signed."""
    monkeypatch.setattr(vision_seal, "load_settings", _seal_settings)
    monkeypatch.setattr(
        vision_seal, "locate_signature_pages", lambda _ocr_dir: [Path("page_001.png")]
    )
    valid_party = {"role": "甲方", "has_seal": True, "has_signature": False}
    invalid_payloads = [
        {"unexpected": "shape"},
        {"units": []},
        {"units": [{"parties": [{"role": "甲方", "has_seal": "false", "has_signature": False}]}]},
        # A mixed valid/invalid response cannot be partially accepted: the missing
        # verdict must not make the original issue disappear.
        {
            "units": [
                {"parties": [valid_party]},
                {"parties": [{"role": "乙方", "has_seal": False, "has_signature": "false"}]},
            ]
        },
    ]

    for payload in invalid_payloads:
        monkeypatch.setattr(
            vision_seal,
            "_call_vl",
            lambda *_args, payload=payload: (json.dumps(payload, ensure_ascii=False), None),
        )
        env = _env_with_text_signature_issue()

        assert vision_seal.augment_completeness_with_vision(env, Path("/unused")) is False
        assert env.completeness is not None
        assert env.completeness.status == "incomplete"
        assert [issue.item for issue in env.completeness.issues] == ["文本抽取的甲方签章"]
        assert [(issue.stage, issue.severity, issue.error.code) for issue in env.stage_issues] == [
            ("seal_vision", "warning", "INVALID_RESPONSE")
        ]


def test_augment_preserves_text_completeness_when_image_encoding_fails(monkeypatch):
    """Image-read failures are safe optional-stage warnings, not a successful seal check."""
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(vision_seal, "load_settings", _seal_settings)
    monkeypatch.setattr(
        vision_seal, "locate_signature_pages", lambda _ocr_dir: [Path("page_001.png")]
    )

    def broken_encoder(_path):
        raise RuntimeError("image bytes must not leak into the stage issue")

    monkeypatch.setattr(vision_seal, "encode_image_data_uri", broken_encoder)
    env = _env_with_text_signature_issue()

    assert vision_seal.augment_completeness_with_vision(env, Path("/unused")) is False
    assert env.completeness is not None
    assert env.completeness.status == "incomplete"
    assert [issue.item for issue in env.completeness.issues] == ["文本抽取的甲方签章"]
    assert len(env.stage_issues) == 1
    assert env.stage_issues[0].stage == "seal_vision"
    assert env.stage_issues[0].severity == "warning"
    assert env.stage_issues[0].error.message == "VL seal verification request failed"


def test_call_vl_empty_choices_returns_safe_error(monkeypatch):
    """Malformed SDK responses are caught with the image encoding and API call."""
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(vision_seal, "encode_image_data_uri", lambda _path: "data:image/png;base64,test")
    monkeypatch.setattr(
        vision_seal,
        "observed_chat_completion",
        lambda *_args, **_kwargs: SimpleNamespace(choices=[]),
    )

    text, error = vision_seal._call_vl(
        [Path("page_001.png")], "qwen-vl-test", "test-key", "https://dashscope.aliyuncs.com/api/v1"
    )

    assert text is None
    assert error is not None
    assert error.message == "VL seal verification request failed"


# ---- Replacement coverage: a VL response may clear old signature issues only slot by slot ----


def _signature_issue(item: str, page: int) -> CompletenessIssue:
    return CompletenessIssue(
        item=item,
        category="signature",
        detail="文本层疑似缺签章",
        evidence=f"第 {page} 页",
    )


def _vision_unit(agreement: str, page: int, *parties: dict) -> dict:
    return {"agreement": agreement, "page": page, "parties": list(parties)}


def _signed(role: str) -> dict:
    return {"role": role, "has_seal": True, "has_signature": False}


def _run_vision(monkeypatch, env: DocumentExtraction, payload: dict, pages: list[int]) -> bool:
    images = [Path(f"page_{page:03d}.png") for page in pages]
    monkeypatch.setattr(vision_seal, "locate_signature_pages", lambda _ocr_dir: images)
    monkeypatch.setattr(
        vision_seal,
        "_read_seals_on_images_with_issue",
        lambda _images: (payload, None),
    )
    return vision_seal.augment_completeness_with_vision(env, Path("/unused"))


def test_vision_partial_coverage_preserves_all_old_signature_issues(monkeypatch):
    old = [_signature_issue("主协议·甲方签章", 8), _signature_issue("主协议·乙方签章", 8)]
    env = DocumentExtraction(doc_type="合同协议", completeness=Completeness(status="incomplete", issues=old))
    payload = {"units": [_vision_unit("主协议", 8, _signed("甲方"))]}

    assert _run_vision(monkeypatch, env, payload, [8]) is False
    assert env.completeness is not None
    assert env.completeness.issues == old
    assert [(issue.stage, issue.severity, issue.error.code) for issue in env.stage_issues] == [
        ("seal_vision", "warning", "INCOMPLETE_COVERAGE")
    ]


def test_vision_missing_supplementary_slot_preserves_all_old_issues(monkeypatch):
    old = [
        _signature_issue("主协议·甲方签章", 8),
        _signature_issue("补充协议·乙方签章", 9),
    ]
    env = DocumentExtraction(
        doc_type="合同协议",
        sub_agreements=[SubAgreement(title="补充协议")],
        completeness=Completeness(status="incomplete", issues=old),
    )
    payload = {"units": [_vision_unit("主协议", 8, _signed("甲方"))]}

    assert _run_vision(monkeypatch, env, payload, [8, 9]) is False
    assert env.completeness is not None
    assert env.completeness.issues == old
    assert env.stage_issues[-1].error.code == "INCOMPLETE_COVERAGE"


def test_vision_wrong_page_cannot_cover_signature_issue(monkeypatch):
    old = [_signature_issue("主协议·甲方签章", 8)]
    env = DocumentExtraction(doc_type="合同协议", completeness=Completeness(status="incomplete", issues=old))
    payload = {"units": [_vision_unit("主协议", 9, _signed("甲方"))]}

    assert _run_vision(monkeypatch, env, payload, [8, 9]) is False
    assert env.completeness is not None
    assert env.completeness.issues == old
    assert env.stage_issues[-1].error.code == "INCOMPLETE_COVERAGE"


def test_vision_missing_or_unrequested_page_cannot_cover_signature_issue(monkeypatch):
    old = [_signature_issue("主协议·甲方签章", 8)]
    payloads = [
        {"units": [{"agreement": "主协议", "parties": [_signed("甲方")]}]},
        {"units": [_vision_unit("主协议", 9, _signed("甲方"))]},
    ]

    for payload in payloads:
        env = DocumentExtraction(
            doc_type="合同协议", completeness=Completeness(status="incomplete", issues=old)
        )
        assert _run_vision(monkeypatch, env, payload, [8]) is False
        assert env.completeness is not None
        assert env.completeness.issues == old
        assert env.stage_issues[-1].error.code == "INCOMPLETE_COVERAGE"


def test_vision_unknown_agreement_or_ambiguous_role_cannot_clear(monkeypatch):
    old = [_signature_issue("主协议·甲方签章", 8)]
    payloads = [
        {"units": [_vision_unit("未知协议", 8, _signed("甲方"))]},
        {"units": [_vision_unit("地下车位补充协议", 8, _signed("甲方"))]},
        {"units": [_vision_unit("主协议", 8, _signed("甲乙双方"))]},
    ]

    for payload in payloads:
        env = DocumentExtraction(
            doc_type="合同协议", completeness=Completeness(status="incomplete", issues=old)
        )
        assert _run_vision(monkeypatch, env, payload, [8]) is False
        assert env.completeness is not None
        assert env.completeness.issues == old
        assert env.stage_issues[-1].error.code == "INCOMPLETE_COVERAGE"


def test_vision_invalid_extra_unit_invalidates_otherwise_covered_response(monkeypatch):
    old = [_signature_issue("主协议·甲方签章", 8)]
    invalid_extras = [
        _vision_unit("未知协议", 8, _signed("乙方")),
        _vision_unit("主协议", 8, _signed("甲乙双方")),
        _vision_unit("主协议", 9, _signed("乙方")),
    ]

    for extra in invalid_extras:
        env = DocumentExtraction(
            doc_type="合同协议", completeness=Completeness(status="incomplete", issues=old)
        )
        payload = {"units": [_vision_unit("主协议", 8, _signed("甲方")), extra]}

        assert _run_vision(monkeypatch, env, payload, [8]) is False
        assert env.completeness is not None
        assert env.completeness.issues == old
        assert env.stage_issues[-1].error.code == "INCOMPLETE_COVERAGE"


def test_vision_duplicate_slot_cannot_clear_signature_issue(monkeypatch):
    old = [_signature_issue("主协议·甲方签章", 8)]
    env = DocumentExtraction(doc_type="合同协议", completeness=Completeness(status="incomplete", issues=old))
    payload = {
        "units": [
            _vision_unit("主协议", 8, _signed("甲方")),
            _vision_unit(
                "主协议", 8, {"role": "甲方", "has_seal": False, "has_signature": False}
            ),
        ]
    }

    assert _run_vision(monkeypatch, env, payload, [8]) is False
    assert env.completeness is not None
    assert env.completeness.issues == old
    assert env.stage_issues[-1].error.code == "INCOMPLETE_COVERAGE"


def test_vision_one_slot_cannot_clear_duplicate_old_signature_issues(monkeypatch):
    old = [
        _signature_issue("主协议·甲方签章", 8),
        CompletenessIssue(
            item="主协议·甲方签章",
            category="signature",
            detail="另一条重复的文本问题",
            evidence="第 8 页",
        ),
    ]
    env = DocumentExtraction(doc_type="合同协议", completeness=Completeness(status="incomplete", issues=old))
    payload = {"units": [_vision_unit("主协议", 8, _signed("甲方"))]}

    assert _run_vision(monkeypatch, env, payload, [8]) is False
    assert env.completeness is not None
    assert env.completeness.issues == old
    assert env.stage_issues[-1].error.code == "INCOMPLETE_COVERAGE"


def test_vision_full_main_and_supplement_coverage_can_complete(monkeypatch):
    old = [
        _signature_issue("主协议·甲方签章", 8),
        _signature_issue("主协议·乙方签章", 8),
        _signature_issue("补充协议·乙方签章", 9),
    ]
    env = DocumentExtraction(
        doc_type="合同协议",
        sub_agreements=[SubAgreement(title="补充协议")],
        completeness=Completeness(status="incomplete", issues=old),
    )
    payload = {
        "units": [
            _vision_unit("主合同", 8, _signed("甲方"), _signed("乙方")),
            _vision_unit("补充协议一", 9, _signed("乙方")),
        ]
    }

    assert _run_vision(monkeypatch, env, payload, [8, 9]) is True
    assert env.completeness is not None
    assert env.completeness.status == "complete"
    assert env.completeness.issues == []


def test_vision_role_alias_buyer_matches_party_b_slot(monkeypatch):
    old = [_signature_issue("买受人签章", 8)]
    env = DocumentExtraction(doc_type="合同协议", completeness=Completeness(status="incomplete", issues=old))
    payload = {"units": [_vision_unit("本协议", 8, _signed("乙方"))]}

    assert _run_vision(monkeypatch, env, payload, [8]) is True
    assert env.completeness is not None
    assert env.completeness.status == "complete"
    assert env.completeness.issues == []


def test_vision_omitted_old_agreement_requires_no_sub_agreements_and_one_page(monkeypatch):
    old = _signature_issue("甲方签章", 8)
    with_sub_agreement = DocumentExtraction(
        doc_type="合同协议",
        sub_agreements=[SubAgreement(title="补充协议")],
        completeness=Completeness(status="incomplete", issues=[old]),
    )
    payload = {"units": [_vision_unit("主协议", 8, _signed("甲方"))]}

    assert _run_vision(monkeypatch, with_sub_agreement, payload, [8]) is False
    assert with_sub_agreement.completeness is not None
    assert with_sub_agreement.completeness.issues == [old]

    ambiguous_evidence = _signature_issue("甲方签章", 8)
    ambiguous_evidence.evidence = "第 8、9 页"
    no_sub_agreement = DocumentExtraction(
        doc_type="合同协议",
        completeness=Completeness(status="incomplete", issues=[ambiguous_evidence]),
    )
    assert _run_vision(monkeypatch, no_sub_agreement, payload, [8]) is False
    assert no_sub_agreement.completeness is not None
    assert no_sub_agreement.completeness.issues == [ambiguous_evidence]


def test_vision_all_positive_does_not_promote_unknown_or_none(monkeypatch):
    payload = {"units": [_vision_unit("主协议", 8, _signed("甲方"))]}

    none_env = DocumentExtraction(doc_type="合同协议", completeness=None)
    assert _run_vision(monkeypatch, none_env, payload, [8]) is True
    assert none_env.completeness is None

    unknown_env = DocumentExtraction(
        doc_type="合同协议", completeness=Completeness(status="unknown", issues=[])
    )
    assert _run_vision(monkeypatch, unknown_env, payload, [8]) is True
    assert unknown_env.completeness is not None
    assert unknown_env.completeness.status == "unknown"
    assert unknown_env.completeness.issues == []


def test_vision_new_defect_can_make_none_completeness_incomplete(monkeypatch):
    env = DocumentExtraction(doc_type="合同协议", completeness=None)
    payload = {
        "units": [
            _vision_unit(
                "主协议", 8, {"role": "甲方", "has_seal": False, "has_signature": False}
            )
        ]
    }

    assert _run_vision(monkeypatch, env, payload, [8]) is True
    assert env.completeness is not None
    assert env.completeness.status == "incomplete"
    assert [issue.item for issue in env.completeness.issues] == ["主协议·甲方签章"]


def test_vision_full_coverage_preserves_non_signature_issues(monkeypatch):
    signature = _signature_issue("主协议·甲方签章", 8)
    field = CompletenessIssue(item="签订日期", category="field", evidence="第 1 页")
    env = DocumentExtraction(
        doc_type="合同协议", completeness=Completeness(status="incomplete", issues=[signature, field])
    )
    payload = {"units": [_vision_unit("主协议", 8, _signed("甲方"))]}

    assert _run_vision(monkeypatch, env, payload, [8]) is True
    assert env.completeness is not None
    assert env.completeness.status == "incomplete"
    assert env.completeness.issues == [field]
