"""Offline tests for the durable evaluation runner (no model/API calls)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from contract_archive.config import load_settings
from contract_archive.utils.llm_runtime import record_llm_call
from evals import run
from evals.pricing import record_cost
from evals.run import DEFAULT_CASES, evalset_dir, load_cases, select_cases


def test_evalset_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_ARCHIVE_EVALSET_DIR", str(tmp_path / "private" / "dataset"))
    assert evalset_dir() == tmp_path / "private" / "dataset"


def test_evalset_dir_falls_back_to_builtin(monkeypatch):
    monkeypatch.delenv("CONTRACT_ARCHIVE_EVALSET_DIR", raising=False)
    assert evalset_dir() == DEFAULT_CASES


def _case(suite_dir, name, *, text=None, pdf=False, gold=True):
    d = suite_dir / name
    d.mkdir(parents=True)
    if gold:
        (d / "gold.json").write_text(json.dumps({"doc_type": "其他"}), encoding="utf-8")
    if text is not None:
        (d / "input.txt").write_text(text, encoding="utf-8")
    if pdf:
        (d / "source.pdf").write_bytes(b"%PDF fake")
    return d


def test_load_cases_recognizes_text_and_pdf(tmp_path):
    suite = tmp_path / "extraction"
    _case(suite, "c_text", text="纯文本 case")
    _case(suite, "c_pdf", pdf=True)
    _case(suite, "c_both", text="文本", pdf=True)
    _case(suite, "c_nogold", text="无金标准", gold=False)  # 应跳过
    _case(suite, "c_neither", gold=True)  # 既无 input 也无 pdf → 跳过

    cases = {c["case_id"]: c for c in load_cases(suite)}
    assert set(cases) == {"c_text", "c_pdf", "c_both"}
    assert cases["c_text"]["pdf"] is None and cases["c_text"]["input"] == "纯文本 case"
    assert cases["c_pdf"]["pdf"] is not None and cases["c_pdf"]["input"] is None
    assert cases["c_both"]["pdf"] is not None and cases["c_both"]["input"] == "文本"


def test_select_cases_auto_prefers_pdf_and_explicit_modes_are_bounded(tmp_path):
    suite = tmp_path / "extraction"
    _case(suite, "both", text="text", pdf=True)
    _case(suite, "text_only", text="text-only")
    cases = load_cases(suite)

    auto = {case["case_id"]: case for case in select_cases(cases, input_mode="auto")}
    assert auto["both"]["source"] == "pdf"
    assert auto["text_only"]["source"] == "text"

    text = select_cases(cases, input_mode="text", case_ids=["both"])
    assert text[0]["source"] == "text"
    with pytest.raises(ValueError, match="unavailable"):
        select_cases(cases, input_mode="pdf", case_ids=["text_only"])


class _FakeEnvelope:
    def __init__(self, *, text_model: str, calls: list[dict] | None = None) -> None:
        self.llm_model = text_model
        self.llm_calls = calls or [
            {
                "stage": "document_extract",
                "model": text_model,
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                "usage_known": True,
                "latency_s": 0.01,
                "status": "ok",
                "error": None,
            }
        ]
        self.extraction_error = None
        self.stage_issues = []

    def model_dump(self, **_kwargs):
        return {"doc_type": "其他", "llm_model": self.llm_model, "llm_calls": self.llm_calls}


def _fake_pdf_case(suite, name):
    case_dir = _case(suite, name, pdf=True)
    # Distinct case PDFs must have distinct content-addressed OCR keys. (The
    # runner intentionally shares OCR for byte-identical PDFs.)
    (case_dir / "source.pdf").write_bytes(f"%PDF fake {name}".encode("utf-8"))


def _install_pdf_fakes(
    monkeypatch,
    *,
    partial=False,
    include_vision_calls=False,
    failed_vision_call=False,
    record_ocr_calls=False,
):
    """Install deterministic production-boundary fakes; no OCR/LLM network happens."""
    seen = {"ocr": [], "extract": []}

    class FakePipeline:
        def run(self, _source, ocr_dir):
            ocr_model = load_settings().dashscope_ocr_model
            seen["ocr"].append(ocr_model)
            if record_ocr_calls:
                record_llm_call(
                    "ocr_page",
                    ocr_model,
                    usage={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                    latency_s=0.02,
                )
            ocr_dir.mkdir(parents=True, exist_ok=True)
            (ocr_dir / "raw_text.txt").write_text("fake OCR text", encoding="utf-8")
            return SimpleNamespace(
                meta={
                    "ocr_failed_pages": [1] if partial else [],
                    "ocr_truncated_pages": [],
                    "pipeline_name": "ocr",
                }
            )

    def fake_full_extraction(_text, _ocr_dir):
        settings = load_settings()
        seen["extract"].append(
            (
                settings.dashscope_model,
                settings.dashscope_vl_model,
                settings.dashscope_vl_extract_model,
            )
        )
        calls = None
        if include_vision_calls:
            calls = [
                # This fixed-text call must never be treated as a vision-axis
                # candidate request.
                {
                    "stage": "fusion_text",
                    "model": settings.dashscope_model,
                    "usage": None,
                    "usage_known": False,
                    "latency_s": 0.01,
                    "status": "ok",
                    "error": None,
                },
                {
                    "stage": "fusion_vision",
                    "model": settings.dashscope_vl_extract_model,
                    "usage": None,
                    "usage_known": False,
                    "latency_s": 0.01,
                    "status": "ok",
                    "error": None,
                },
            ]
            if failed_vision_call:
                # A real optional seal check can time out and leave a warning
                # while the text-derived extraction remains available.  The
                # evaluation axis must nevertheless record that its candidate
                # request failed.
                calls.append(
                    {
                        "stage": "seal_vision",
                        "model": settings.dashscope_vl_model,
                        "usage": None,
                        "usage_known": False,
                        "latency_s": 0.01,
                        "status": "error",
                        "error": "TimeoutError",
                    }
                )
        envelope = _FakeEnvelope(text_model=settings.dashscope_model, calls=calls)
        if failed_vision_call:
            envelope.stage_issues = [
                {"stage": "seal_vision", "severity": "warning", "error": {"code": "TIMEOUT"}}
            ]
        return envelope

    monkeypatch.setattr(run, "OCRPipeline", FakePipeline)
    monkeypatch.setattr(run, "run_full_extraction", fake_full_extraction)
    return seen


@pytest.fixture(autouse=True)
def _clean_eval_config(monkeypatch, tmp_path):
    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_LLM_MODEL",
        "DASHSCOPE_VL_MODEL",
        "DASHSCOPE_OCR_MODEL",
        "DASHSCOPE_VL_EXTRACT_MODEL",
        "DASHSCOPE_TIMEOUT_S",
        "CONTRACT_ARCHIVE_LLM_CONCURRENCY",
        "CONTRACT_ARCHIVE_VL_OCR_MAX_PAGES",
        "CONTRACT_ARCHIVE_VL_OCR_DPI",
        "CONTRACT_ARCHIVE_VL_OCR_RETRIES",
        "CONTRACT_ARCHIVE_VISION_FUSION_MAX_PAGES",
        "CONTRACT_ARCHIVE_FUSION_THRESHOLD",
        "COMPUTE_DEVICE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def test_pdf_matrix_caches_per_pdf_and_resume_skips_complete_identities(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    suite = dataset / "extraction"
    _fake_pdf_case(suite, "a")
    _fake_pdf_case(suite, "b")
    monkeypatch.setenv("DASHSCOPE_LLM_MODEL", "configured-text")
    monkeypatch.setenv("DASHSCOPE_OCR_MODEL", "configured-ocr")
    seen = _install_pdf_fakes(monkeypatch, record_ocr_calls=True)
    out = tmp_path / "out"

    argv = [
        "--models", "candidate-a,candidate-b",
        "--cases-dir", str(dataset),
        "--input-mode", "pdf",
        "--repeat", "2",
        "--out", str(out),
    ]
    assert run.main(argv) == 0
    # A/B PDFs each OCR exactly once; model/repeat extractions use copytree cache.
    assert seen["ocr"] == ["configured-ocr", "configured-ocr"]
    assert len(seen["extract"]) == 8
    assert {item[0] for item in seen["extract"]} == {"candidate-a", "candidate-b"}
    assert load_settings().dashscope_model == "configured-text"  # scoped override restored

    records = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(records) == 8
    assert {record["llm_model"] for record in records} == {"candidate-a", "candidate-b"}
    assert sum(bool(record["cache_hit"]) for record in records) == 6
    assert all(record["status"] == "ok" for record in records)
    assert all(record["candidate_request"]["status"] == "verified" for record in records)
    first, cached = records[0], records[1]
    assert [call["stage"] for call in first["calls"]] == ["ocr_page", "document_extract"]
    assert first["calls"][0]["cache_hit"] is False
    assert cached["calls"][0]["cache_hit"] is True
    assert cached["cached_ocr_usage"] == {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}
    assert cached["actual_usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}

    # A same-identity resume does not append/retry any successful record.
    assert run.main(argv) == 0
    assert len(seen["ocr"]) == 2
    assert len(seen["extract"]) == 8
    assert len((out / "results.jsonl").read_text().splitlines()) == 8


def test_axis_overrides_and_vision_ignores_fusion_text_call(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    suite = dataset / "extraction"
    _fake_pdf_case(suite, "one")
    monkeypatch.setenv("DASHSCOPE_LLM_MODEL", "base-text")
    monkeypatch.setenv("DASHSCOPE_VL_MODEL", "base-vl")
    monkeypatch.setenv("DASHSCOPE_VL_EXTRACT_MODEL", "base-vl-extract")
    monkeypatch.setenv("DASHSCOPE_OCR_MODEL", "base-ocr")
    seen = _install_pdf_fakes(monkeypatch, include_vision_calls=True)

    ocr_out = tmp_path / "ocr-out"
    assert run.main([
        "--models", "ocr-candidate", "--axis", "ocr", "--input-mode", "pdf",
        "--cases-dir", str(dataset), "--out", str(ocr_out),
    ]) == 0
    assert seen["ocr"] == ["ocr-candidate"]
    assert seen["extract"][-1] == ("base-text", "base-vl", "base-vl-extract")
    ocr_record = json.loads((ocr_out / "results.jsonl").read_text())
    assert ocr_record["model"] == "ocr-candidate"
    assert ocr_record["llm_model"] == "base-text"
    assert ocr_record["candidate_request"]["status"] == "not_evaluated"

    vision_out = tmp_path / "vision-out"
    assert run.main([
        "--models", "vision-candidate", "--axis", "vision", "--input-mode", "pdf",
        "--cases-dir", str(dataset), "--out", str(vision_out),
    ]) == 0
    assert seen["extract"][-1] == ("base-text", "vision-candidate", "vision-candidate")
    vision_record = json.loads((vision_out / "results.jsonl").read_text())
    assert vision_record["status"] == "ok"
    assert vision_record["candidate_request"]["status"] == "verified"
    assert vision_record["candidate_request"]["observed_stages"] == ["fusion_vision"]
    assert load_settings().dashscope_vl_model == "base-vl"  # restoration matters across matrix rows


def test_candidate_request_requires_a_successful_call_and_accepts_cached_ocr():
    cached_ocr = [
        {
            "stage": "ocr_page",
            "model": "ocr-candidate",
            "status": "ok",
            "error": None,
            "cache_hit": True,
        }
    ]
    assert run.candidate_request_validation(
        cached_ocr, axis="ocr", candidate="ocr-candidate"
    )["status"] == "verified"

    mixed_vision = [
        {
            "stage": "fusion_vision",
            "model": "vision-candidate",
            "status": "ok",
            "error": None,
        },
        {
            "stage": "seal_vision",
            "model": "vision-candidate",
            "status": "error",
            "error": "TimeoutError",
        },
    ]
    assert run.candidate_request_validation(
        mixed_vision, axis="vision", candidate="vision-candidate"
    )["status"] == "request_failed"
    assert run.candidate_request_validation(
        [{"stage": "ocr_page", "model": None, "status": "ok", "error": None}],
        axis="ocr",
        candidate="ocr-candidate",
    )["status"] == "missing_model"
    assert run.candidate_request_validation(
        [{"stage": "ocr_page", "model": "other", "status": "ok", "error": None}],
        axis="ocr",
        candidate="ocr-candidate",
    )["status"] == "mismatch"


def test_optional_vision_timeout_warning_cannot_verify_fallback_result(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    suite = dataset / "extraction"
    _fake_pdf_case(suite, "vision-timeout")
    monkeypatch.setenv("DASHSCOPE_LLM_MODEL", "base-text")
    monkeypatch.setenv("DASHSCOPE_VL_MODEL", "base-vl")
    monkeypatch.setenv("DASHSCOPE_VL_EXTRACT_MODEL", "base-vl-extract")
    _install_pdf_fakes(
        monkeypatch,
        include_vision_calls=True,
        failed_vision_call=True,
    )

    result = run.run_pdf_case(
        suite / "vision-timeout" / "source.pdf",
        "vision-candidate",
        axis="vision",
        cache_dir=tmp_path / "ocr-cache",
    )

    # The optional production enhancement fell back to the text result, but its
    # failed candidate-axis request is still disqualifying evaluation evidence.
    assert result["pred"]["doc_type"] == "其他"
    assert result["candidate_request"]["status"] == "request_failed"
    assert result["status"] == "error"
    assert result["error"] == "CANDIDATE_REQUEST_FAILED"
    assert result["calls"][-1]["stage"] == "seal_vision"
    assert result["calls"][-1]["status"] == "error"


def test_pdf_ocr_exception_preserves_observed_ledger_without_publishing_cache(tmp_path, monkeypatch):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF fake OCR exception")
    cache_dir = tmp_path / "ocr-cache"

    class FailingPipeline:
        def run(self, _source, _ocr_dir):
            record_llm_call(
                "ocr_page",
                load_settings().dashscope_ocr_model,
                latency_s=0.01,
                error=TimeoutError("synthetic provider timeout"),
            )
            raise TimeoutError("synthetic provider timeout")

    monkeypatch.setattr(run, "OCRPipeline", FailingPipeline)
    result = run.run_pdf_case(
        pdf_path,
        "ocr-candidate",
        axis="ocr",
        cache_dir=cache_dir,
    )

    assert result["status"] == "error"
    assert result["error"] == "TimeoutError"
    assert result["candidate_request"]["status"] == "request_failed"
    assert result["ocr_latency_s"] is not None
    assert len(result["calls"]) == 1
    call = result["calls"][0]
    assert call["stage"] == "ocr_page"
    assert call["model"] == "ocr-candidate"
    assert call["usage"] is None and call["usage_known"] is False
    assert call["latency_s"] == pytest.approx(0.01)
    assert call["status"] == "error" and call["error"] == "TimeoutError"
    assert call["cache_hit"] is False
    assert not record_cost(result).known  # failed request with unknown usage is never free
    assert [path for path in cache_dir.iterdir() if path.name != ".locks"] == []


def test_partial_ocr_is_recorded_and_not_treated_as_success(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    _fake_pdf_case(dataset / "extraction", "partial")
    seen = _install_pdf_fakes(monkeypatch, partial=True)
    out = tmp_path / "out"
    argv = [
        "--models", "candidate", "--input-mode", "pdf", "--cases-dir", str(dataset), "--out", str(out),
    ]
    assert run.main(argv) == 1
    record = json.loads((out / "results.jsonl").read_text())
    assert record["status"] == "partial"
    assert record["cache_hit"] is False
    assert len(seen["ocr"]) == 1

    # Failed/partial identities remain visible and are not silently re-run.
    assert run.main(argv) == 1
    assert len(seen["ocr"]) == 1
    # Only an explicit retry appends a last-wins replacement identity.
    assert run.main([*argv, "--retry-failed"]) == 1
    assert len(seen["ocr"]) == 2
    assert len((out / "results.jsonl").read_text().splitlines()) == 2


def test_text_exception_keeps_observed_call_ledger(monkeypatch):
    def exploding_extract(_text):
        record_llm_call(
            "document_extract",
            "candidate",
            usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            latency_s=0.01,
        )
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(run, "extract_document", exploding_extract)
    result = run.run_extraction_case("private text", "candidate")
    assert result["status"] == "error"
    assert result["error"] == "RuntimeError"
    assert result["calls"] == [
        {
            "stage": "document_extract",
            "model": "candidate",
            "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            "usage_known": True,
            "latency_s": pytest.approx(0.01),
            "status": "ok",
            "error": None,
            "cache_hit": False,
        }
    ]
    assert result["usage"] == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


def test_stage_error_is_not_downgraded_to_a_success():
    envelope = SimpleNamespace(
        extraction_error=None,
        stage_issues=[{"stage": "document_extract", "severity": "error", "error": {"code": "EXTRACT_EMPTY"}}],
    )
    assert run._status_for(envelope, {"ocr_failed_pages": [], "ocr_truncated_pages": []}, None) == "error"
    assert run._envelope_error_code(envelope) == "EXTRACT_EMPTY"


def test_manifest_rejects_input_gold_or_config_changes_and_never_leaks_key(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    suite = dataset / "extraction"
    _case(suite, "text", text="original")
    original_input = (suite / "text" / "input.txt").read_bytes()
    original_gold = (suite / "text" / "gold.json").read_bytes()

    def fake_text_case(_text, candidate, *, axis):
        return {
            "pred": {"doc_type": "其他"}, "llm_model": "actual", "calls": [], "usage": None,
            "actual_usage": None, "cached_ocr_usage": None, "status": "ok", "error": None,
            "candidate_request": {"axis": axis, "candidate_model": candidate, "status": "not_evaluated", "observed_models": [], "observed_stages": []},
            "model_validation": {"axis": axis, "candidate_model": candidate, "status": "not_evaluated", "observed_models": [], "observed_stages": []},
            "latency_s": 0.0, "ocr_latency_s": None, "extract_latency_s": 0.0,
            "logical_total_latency_s": 0.0, "actual_ocr_latency_s": None,
            "wall_latency_s": 0.0, "cache_hit": False, "extraction_calls_known": True,
        }

    monkeypatch.setattr(run, "run_extraction_case", fake_text_case)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-very-secret")
    out = tmp_path / "out"
    argv = ["--models", "candidate", "--input-mode", "text", "--cases-dir", str(dataset), "--out", str(out)]
    assert run.main(argv) == 0
    manifest_text = (out / "run_meta.json").read_text(encoding="utf-8")
    assert "sk-very-secret" not in manifest_text
    assert "text_model" not in json.loads(manifest_text)["config"]["models"]

    (suite / "text" / "input.txt").write_text("changed input", encoding="utf-8")
    assert run.main(argv) == 1
    (suite / "text" / "input.txt").write_bytes(original_input)
    (suite / "text" / "gold.json").write_text('{"doc_type":"合同协议"}', encoding="utf-8")
    assert run.main(argv) == 1
    (suite / "text" / "gold.json").write_bytes(original_gold)
    monkeypatch.setenv("DASHSCOPE_TIMEOUT_S", "99")
    assert run.main(argv) == 1

    # The text-axis configured default is deliberately excluded from identity:
    # it is merely another candidate source, not a fixed comparison condition.
    monkeypatch.delenv("DASHSCOPE_TIMEOUT_S")
    monkeypatch.setenv("DASHSCOPE_LLM_MODEL", "different-candidate-default")
    assert run.main([
        "--models", "another-candidate", "--input-mode", "text",
        "--cases-dir", str(dataset), "--out", str(out),
    ]) == 0
