"""Contract-specialisation obligation replacement boundaries.

These tests intentionally exercise ``extract_contract -> _contract_specialized``
and mock only its LLM boundary.  They prevent a truthy-but-incomplete JSON object
from erasing obligations already extracted by the generic document pass.
"""
from __future__ import annotations

from contract_archive.extraction import contract_extractor
from contract_archive.extraction import doc_type_handlers as handlers
from contract_archive.extraction.llm_extractor import LlmResult
from contract_archive.schemas import DocumentExtraction, ObligationItem


def _generic_obligations() -> list[ObligationItem]:
    return [ObligationItem(actor="party_b", action="保留的通用义务")]


def _mock_llm(monkeypatch, parsed: dict, *, usage: dict | None = None) -> None:
    monkeypatch.setattr(
        contract_extractor,
        "call_llm_extract",
        lambda *_args, **_kwargs: LlmResult(
            parsed=parsed,
            model="contract-test",
            usage=usage,
        ),
    )


def test_truthy_random_json_cannot_clear_generic_obligations_and_keeps_metadata(monkeypatch):
    usage = {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
    _mock_llm(monkeypatch, {"contract_name": "仍可解析的合同", "unrelated": "x"}, usage=usage)
    generic = _generic_obligations()
    env = DocumentExtraction(
        doc_type="合同协议",
        obligations=generic,
        llm_usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    )

    ext, _ = handlers._contract_specialized("正文", env, True)

    assert ext.contract_name == "仍可解析的合同"
    assert ext.llm_model == "contract-test"
    assert ext.llm_usage == usage
    assert ext.extraction_error is not None
    assert ext.extraction_error.code == "INVALID_RESPONSE"
    assert env.obligations == generic
    assert env.llm_usage == {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}
    assert [(issue.stage, issue.severity, issue.error.code) for issue in env.stage_issues] == [
        ("contract_extract", "error", "INVALID_RESPONSE")
    ]


def test_mixed_invalid_obligations_list_cannot_partially_replace_generic_values(monkeypatch):
    _mock_llm(
        monkeypatch,
        {
            "obligations": [
                {"actor": "甲方", "action": "付款", "deadline": None, "evidence": "第1页"},
                {"actor": "未知方", "action": "交付", "deadline": None, "evidence": "第2页"},
            ]
        },
    )
    generic = _generic_obligations()
    env = DocumentExtraction(doc_type="合同协议", obligations=generic)

    ext, _ = handlers._contract_specialized("正文", env, True)

    assert ext.extraction_error is not None
    assert ext.extraction_error.code == "INVALID_RESPONSE"
    assert env.obligations == generic


def test_missing_null_or_non_list_obligations_cannot_clear_generic_values(monkeypatch):
    for payload in ({}, {"obligations": None}, {"obligations": "not-a-list"}):
        _mock_llm(monkeypatch, payload)
        generic = _generic_obligations()
        env = DocumentExtraction(doc_type="合同协议", obligations=generic)

        ext, _ = handlers._contract_specialized("正文", env, True)

        assert ext.extraction_error is not None
        assert ext.extraction_error.code == "INVALID_RESPONSE"
        assert env.obligations == generic


def test_non_string_deadline_or_evidence_is_invalid_and_preserves_generic_obligations(monkeypatch):
    for item in (
        {"actor": "party_a", "action": "付款", "deadline": 20260101, "evidence": None},
        {"actor": "party_a", "action": "付款", "deadline": None, "evidence": 42},
    ):
        _mock_llm(monkeypatch, {"obligations": [item]})
        generic = _generic_obligations()
        env = DocumentExtraction(doc_type="合同协议", obligations=generic)

        ext, _ = handlers._contract_specialized("正文", env, True)

        assert ext.extraction_error is not None
        assert ext.extraction_error.code == "INVALID_RESPONSE"
        assert env.obligations == generic


def test_non_string_or_empty_action_cannot_clear_generic_obligations(monkeypatch):
    for action in (None, {}, [], 42, "   "):
        _mock_llm(
            monkeypatch,
            {"obligations": [{"actor": "party_a", "action": action, "deadline": None}]},
        )
        generic = _generic_obligations()
        env = DocumentExtraction(doc_type="合同协议", obligations=generic)

        ext, _ = handlers._contract_specialized("正文", env, True)

        assert ext.extraction_error is not None
        assert ext.extraction_error.code == "INVALID_RESPONSE"
        assert env.obligations == generic


def test_non_string_actor_cannot_clear_generic_obligations(monkeypatch):
    for actor in (None, {}, [], 42):
        _mock_llm(
            monkeypatch,
            {"obligations": [{"actor": actor, "action": "付款", "deadline": None}]},
        )
        generic = _generic_obligations()
        env = DocumentExtraction(doc_type="合同协议", obligations=generic)

        ext, _ = handlers._contract_specialized("正文", env, True)

        assert ext.extraction_error is not None
        assert ext.extraction_error.code == "INVALID_RESPONSE"
        assert env.obligations == generic


def test_explicit_empty_obligations_list_can_clear_generic_obligations(monkeypatch):
    _mock_llm(monkeypatch, {"obligations": []})
    env = DocumentExtraction(doc_type="合同协议", obligations=_generic_obligations())

    ext, _ = handlers._contract_specialized("正文", env, True)

    assert ext.extraction_error is None
    assert ext.obligations == []
    assert env.obligations == []


def test_valid_obligations_use_existing_actor_aliases_and_replace_generic_values(monkeypatch):
    _mock_llm(
        monkeypatch,
        {
            "obligations": [
                {
                    "actor": "甲方",
                    "action": "支付款项",
                    "deadline": None,
                    "evidence": "第1页",
                }
            ]
        },
    )
    env = DocumentExtraction(doc_type="合同协议", obligations=_generic_obligations())

    ext, _ = handlers._contract_specialized("正文", env, True)

    assert ext.extraction_error is None
    assert [(item.actor, item.action, item.evidence) for item in env.obligations] == [
        ("party_a", "支付款项", "第1页")
    ]


def test_no_llm_preserves_generic_obligations_without_calling_contract_llm(monkeypatch):
    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("--no-llm must not call the contract LLM")

    monkeypatch.setattr(contract_extractor, "call_llm_extract", unexpected_call)
    generic = _generic_obligations()
    env = DocumentExtraction(doc_type="合同协议", obligations=generic)

    ext, _ = handlers._contract_specialized("正文", env, False)

    assert ext.extraction_error is None
    assert env.obligations == generic
    assert env.stage_issues == []
