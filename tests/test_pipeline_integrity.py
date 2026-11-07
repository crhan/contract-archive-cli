"""Regression coverage for pipeline integrity and safe LLM stage boundaries."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import fitz

from contract_archive.errors import ErrorInfo
from contract_archive.extraction import doc_type_handlers as handlers
from contract_archive.extraction import document_extractor as document
from contract_archive.extraction import llm_extractor
from contract_archive.extraction import text_fields
from contract_archive.pipelines import ocr_pipeline as ocr
from contract_archive.pipelines import vl_ocr
from contract_archive.pipelines.vl_ocr import _MARK_FAILED, _PageResult
from contract_archive.schemas import ContractExtraction, DocumentExtraction, ObligationItem
from contract_archive.utils import NO_THINKING, PageImage, PageRoute, TextLayerStats


def _settings():
    return SimpleNamespace(
        dashscope_model="text-test",
        dashscope_api_key="test-key",
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
    )


def _install_openai(monkeypatch, content: str) -> dict:
    seen: dict = {}

    class Client:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            seen["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content), finish_reason="stop"
                    )
                ],
                usage=None,
            )

    module = types.ModuleType("openai")
    module.OpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", module)
    return seen


def test_default_document_call_sends_full_text_and_keeps_no_thinking(monkeypatch):
    monkeypatch.setattr(document, "load_settings", _settings)
    seen = _install_openai(monkeypatch, '{"doc_type":"其他"}')
    full_text = "开头-" + ("中段证据-" * 20) + "-结尾"

    result = document.call_llm_document(full_text)

    assert result.error is None
    sent = seen["request"]["messages"][1]["content"]
    assert full_text in sent
    assert "[...省略中段...]" not in sent
    assert seen["request"]["extra_body"] == NO_THINKING


def test_explicit_text_limit_returns_structured_error_without_a_request(monkeypatch):
    monkeypatch.setattr(document, "load_settings", _settings)
    monkeypatch.setattr(llm_extractor, "load_settings", _settings)
    monkeypatch.setattr(text_fields, "load_settings", _settings)
    seen = _install_openai(monkeypatch, '{"unused":true}')

    doc_result = document.call_llm_document("abcdef", max_chars=3)
    contract_result = llm_extractor.call_llm_extract("abcdef", max_chars=3)
    field_result = text_fields.read_fields_in_text(
        "abcdef", {"字段": "测试字段"}, max_chars=3
    )

    assert doc_result.error and doc_result.error.code == "INPUT_TOO_LONG"
    assert contract_result.error and contract_result.error.code == "INPUT_TOO_LONG"
    assert field_result.error and field_result.error.code == "INPUT_TOO_LONG"
    assert "request" not in seen


def test_required_document_failure_becomes_stage_issue(monkeypatch):
    err = llm_extractor.output_truncated_error()
    monkeypatch.setattr(
        document,
        "call_llm_document",
        lambda *_args, **_kwargs: llm_extractor.LlmResult(
            parsed={}, model="text-test", usage={"total_tokens": 11}, error=err
        ),
    )

    env = document.extract_document("正文")

    assert env.extraction_error == err
    assert env.llm_usage == {"total_tokens": 11}
    assert [(i.stage, i.severity, i.error.code) for i in env.stage_issues] == [
        ("document_extract", "error", "OUTPUT_TRUNCATED")
    ]


def test_contract_specialization_failure_keeps_generic_obligations(monkeypatch):
    generic = [ObligationItem(actor="party_a", action="保留的通用义务")]
    contract_error = ErrorInfo(
        code="TIMEOUT", category="transient", message="safe", retryable=True
    )
    monkeypatch.setattr(
        handlers,
        "extract_contract",
        lambda *_args, **_kwargs: (
            ContractExtraction(
                llm_usage={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
                extraction_error=contract_error,
            ),
            handlers.ExtractionConfidence(),
        ),
    )
    env = DocumentExtraction(
        doc_type="合同协议",
        obligations=generic,
        llm_usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    )

    handlers._contract_specialized("正文", env, True)

    assert env.obligations == generic
    assert env.extraction_error == contract_error
    assert env.llm_usage == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
    assert [(i.stage, i.severity, i.error.code) for i in env.stage_issues] == [
        ("contract_extract", "error", "TIMEOUT")
    ]


def test_contract_specialization_successful_empty_obligations_clear_generic(monkeypatch):
    generic = [ObligationItem(actor="party_a", action="应被清除的通用误抽")]
    monkeypatch.setattr(
        handlers,
        "extract_contract",
        lambda *_args, **_kwargs: (
            ContractExtraction(obligations=[]),
            handlers.ExtractionConfidence(),
        ),
    )
    env = DocumentExtraction(doc_type="合同协议", obligations=generic)

    handlers._contract_specialized("正文", env, True)

    assert env.obligations == []
    assert env.extraction_error is None
    assert env.stage_issues == []


def test_hybrid_ocr_persists_failed_and_truncated_page_numbers(tmp_path, monkeypatch):
    """A missing render, remote failure, and truncation remain auditable as 1-based pages."""
    pdf = tmp_path / "scan.pdf"
    source = fitz.open()
    for _ in range(3):
        source.new_page()
    source.save(pdf)
    source.close()

    routes = [PageRoute(i, "ocr", False, False, 0) for i in range(3)]
    infos = [
        SimpleNamespace(page_index=i, width_pt=100.0, height_pt=100.0, image_count=0)
        for i in range(3)
    ]
    stats = TextLayerStats(
        pages=3,
        chars=0,
        non_ws_chars=0,
        printable_chars=0,
        cjk_chars=0,
        control_chars=0,
        replacement_chars=0,
    )
    rendered = [
        PageImage(0, tmp_path / "page_001.png", 100, 100, 100.0, 100.0, 72),
        PageImage(1, tmp_path / "page_002.png", 100, 100, 100.0, 100.0, 72),
        # Page 3 deliberately absent: rendering failed before that page.
    ]

    monkeypatch.setattr(ocr, "inspect_pdf_pages", lambda _path: infos)
    monkeypatch.setattr(ocr, "analyze_text_layer", lambda _path: stats)
    monkeypatch.setattr(ocr, "classify_pages", lambda _path: routes)
    monkeypatch.setattr(ocr, "_render_previews_safe", lambda *_args: rendered)
    monkeypatch.setattr(
        ocr,
        "ocr_pages",
        lambda *_args, **_kwargs: [
            _PageResult("partial page", ok=True, failed=False, truncated=True),
            _PageResult(_MARK_FAILED, ok=False, failed=True, truncated=False),
        ],
    )

    out_dir = tmp_path / "out"
    result = ocr.OCRPipeline(vl_ocr_dpi=72).run(pdf, out_dir)

    assert result.meta.ocr_failed_pages == [2, 3]
    assert result.meta.ocr_truncated_pages == [1]
    assert "[本页渲染失败，无法进行 OCR]" in result.raw_text
    persisted = json.loads((out_dir / "pipeline_meta.json").read_text(encoding="utf-8"))
    assert persisted["ocr_failed_pages"] == [2, 3]
    assert persisted["ocr_truncated_pages"] == [1]


def test_completion_paths_use_observer_and_keep_no_thinking():
    """Guard the runtime boundary for every direct completion call site."""
    base = Path(__file__).parents[1] / "contract_archive"
    direct_calls = {
        "extraction/llm_extractor.py": "contract_extract",
        "extraction/vl_extract.py": "fusion_vision",
        "extraction/fusion.py": "fusion_adjudicate",
        "extraction/vision_seal.py": "seal_vision",
        "pipelines/vl_ocr.py": "ocr_page",
    }
    for relative, stage in direct_calls.items():
        source = (base / relative).read_text(encoding="utf-8")
        assert "observed_chat_completion(" in source
        assert f'"{stage}"' in source
        assert "extra_body=NO_THINKING" in source
        assert "chat.completions.create(" not in source

    # Text/document paths share the JSON helper above, but must still name their
    # stable ledger stages when they invoke it.
    for relative, stage in {
        "extraction/document_extractor.py": "document_extract",
        "extraction/text_fields.py": "fusion_text",
    }.items():
        assert f'stage="{stage}"' in (base / relative).read_text(encoding="utf-8")


def test_ocr_request_uses_model_default_output_limit(monkeypatch):
    captured: dict = {}
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="正文"), finish_reason="stop")]
    )

    def observe(_client, stage, **kwargs):
        captured["stage"] = stage
        captured["kwargs"] = kwargs
        return response

    monkeypatch.setattr(vl_ocr, "observed_chat_completion", observe)
    monkeypatch.setattr(vl_ocr, "encode_image_data_uri", lambda _path: "data:image/png;base64,x")

    result = vl_ocr._ocr_one_page(object(), "ocr-model", 1, Path("page.png"), 1)

    assert result.ok is True
    assert captured["stage"] == "ocr_page"
    assert captured["kwargs"]["extra_body"] == NO_THINKING
    assert "max_tokens" not in captured["kwargs"]
