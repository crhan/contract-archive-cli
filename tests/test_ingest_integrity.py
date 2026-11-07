"""Production ingest integrity: call ledgers and required-stage status gates."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest

from contract_archive.archive import ArchivePaths, get_document, ingest_pdf, open_archive_db, re_extract
from contract_archive.archive.ingest import run_full_extraction
from contract_archive.errors import ErrorInfo
from contract_archive.schemas import (
    ContractExtraction,
    DocumentExtraction,
    ExtractionConfidence,
    FILE_EXTRACTION,
    ObligationItem,
    PipelineMeta,
    StageIssue,
)
from contract_archive.utils.llm_runtime import record_llm_call


class _Pipeline:
    """Small OCR double that can emit page metadata and one observed OCR call."""

    def __init__(self, *, failed: list[int] | None = None, truncated: list[int] | None = None):
        self.failed = failed or []
        self.truncated = truncated or []

    def run(self, pdf_path: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "raw_text.txt").write_text("sample OCR text", encoding="utf-8")
        meta = PipelineMeta(
            pipeline_name="ocr",
            source_pdf=str(pdf_path),
            started_at=datetime.now(),
            finished_at=datetime.now(),
            duration_seconds=0.01,
            ocr_failed_pages=self.failed,
            ocr_truncated_pages=self.truncated,
        )
        (out_dir / "pipeline_meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
        record_llm_call(
            "ocr_page",
            "fake-ocr",
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )


@pytest.fixture
def archive(tmp_path):
    paths = ArchivePaths(tmp_path / "archive")
    paths.ensure()
    conn = open_archive_db(paths.db_path)
    yield paths, conn
    conn.close()


@pytest.fixture
def source_pdf(tmp_path) -> Path:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF fake integrity test")
    return source


def _successful_extraction(_text: str, llm_enabled: bool):
    assert llm_enabled is True
    record_llm_call(
        "document_extract",
        "fake-text",
        usage={"prompt_tokens": 10, "completion_tokens": 11, "total_tokens": 21},
    )
    record_llm_call(
        "contract_extract",
        "fake-contract",
        usage={"prompt_tokens": 20, "completion_tokens": 21, "total_tokens": 41},
    )
    return (
        ContractExtraction(contract_name="完整账本合同"),
        ExtractionConfidence(overall=0.8),
        DocumentExtraction(doc_type="其他", title="完整账本合同"),
    )


def test_ingest_persists_one_complete_ledger_including_ocr(
    archive, source_pdf, monkeypatch
):
    paths, conn = archive
    monkeypatch.setattr("contract_archive.archive.ingest._run_extraction", _successful_extraction)

    result = ingest_pdf(source_pdf, paths, conn, pipeline=_Pipeline())

    assert result.status == "ok"
    row = get_document(conn, result.doc_id)
    assert row is not None
    details = row.details()
    assert [call["stage"] for call in details["llm_calls"]] == [
        "ocr_page", "document_extract", "contract_extract"
    ]
    assert details["llm_usage"] == {
        "input_tokens": 32,
        "output_tokens": 35,
        "total_tokens": 67,
    }


def test_ocr_page_loss_forces_partial_but_keeps_extracted_title(
    archive, source_pdf, monkeypatch
):
    paths, conn = archive
    monkeypatch.setattr("contract_archive.archive.ingest._run_extraction", _successful_extraction)

    result = ingest_pdf(source_pdf, paths, conn, pipeline=_Pipeline(failed=[2], truncated=[4]))

    assert result.status == "partial"
    assert result.error is not None and result.error.code == "OCR_INCOMPLETE"
    row = get_document(conn, result.doc_id)
    assert row is not None
    assert row.status == "partial"
    assert row.title == "完整账本合同"
    issue = next(item for item in row.details()["stage_issues"] if item["stage"] == "ocr_page")
    assert issue["severity"] == "error"
    assert issue["error"]["code"] == "OCR_INCOMPLETE"


def test_specialized_error_forces_partial_without_erasing_contract_fields(
    archive, source_pdf, monkeypatch
):
    paths, conn = archive
    error = ErrorInfo(code="CONTRACT_DOWN", category="infra", message="specialized failed", retryable=True)

    obligation = ObligationItem(actor="party_a", action="继续履约")

    def extraction(_text: str, llm_enabled: bool):
        assert llm_enabled is False
        env = DocumentExtraction(
            doc_type="合同协议",
            title="仍可展示的合同",
            obligations=[obligation],
            extraction_error=error,
            stage_issues=[StageIssue(stage="contract_extract", error=error, severity="error")],
        )
        return (
            ContractExtraction(contract_name="仍可展示的合同", obligations=[obligation]),
            ExtractionConfidence(overall=0.8),
            env,
        )

    monkeypatch.setattr("contract_archive.archive.ingest._run_extraction", extraction)
    # Disable optional vision post-processing; the fake extractor still models a
    # mandatory specialized-stage error which must govern final status.
    result = ingest_pdf(source_pdf, paths, conn, llm_enabled=False, pipeline=_Pipeline())

    assert result.status == "partial"
    row = get_document(conn, result.doc_id)
    assert row is not None
    assert row.status == "partial"
    assert row.contract_name == "仍可展示的合同"
    assert row.title == "仍可展示的合同"
    assert [item.action for item in row.obligations] == ["继续履约"]


def test_reextract_ledger_excludes_historical_ocr_but_run_full_owns_its_ledger(
    archive, source_pdf, monkeypatch
):
    paths, conn = archive
    monkeypatch.setattr("contract_archive.archive.ingest._run_extraction", _successful_extraction)
    first = ingest_pdf(source_pdf, paths, conn, pipeline=_Pipeline())

    second = re_extract(first.doc_id, paths, conn)
    assert second.status == "ok"
    row = get_document(conn, first.doc_id)
    assert row is not None
    assert [call["stage"] for call in row.details()["llm_calls"]] == [
        "document_extract", "contract_extract"
    ]

    ocr_dir = paths.resolve_document_dir(row) / "ocr"
    full = run_full_extraction("sample OCR text", ocr_dir)
    assert [call["stage"] for call in full.llm_calls] == [
        "document_extract", "contract_extract"
    ]
    assert full.llm_usage == {
        "input_tokens": 30,
        "output_tokens": 32,
        "total_tokens": 62,
    }


def test_run_full_exception_and_ocr_meta_both_remain_visible(tmp_path, monkeypatch):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    meta = PipelineMeta(
        pipeline_name="ocr",
        source_pdf="fixture.pdf",
        started_at=datetime.now(),
        finished_at=datetime.now(),
        duration_seconds=0.01,
        ocr_failed_pages=[3],
    )
    (ocr_dir / "pipeline_meta.json").write_text(meta.model_dump_json(), encoding="utf-8")

    def exploding(_text: str, *, llm_enabled: bool):
        assert llm_enabled is True
        record_llm_call(
            "document_extract",
            "fake-text",
            usage={"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        )
        raise RuntimeError("extract boom")

    monkeypatch.setattr("contract_archive.archive.ingest._run_extraction", exploding)
    env = run_full_extraction("text", ocr_dir)

    assert [call["stage"] for call in env.llm_calls] == ["document_extract"]
    assert env.llm_usage == {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9}
    assert env.extraction_error is not None
    assert {(issue.stage, issue.error.code, issue.severity) for issue in env.stage_issues} >= {
        ("extraction", "UNKNOWN", "error"),
        ("ocr_page", "OCR_INCOMPLETE", "error"),
    }


def test_fatal_ocr_failure_persists_safe_observed_call_ledger(archive, source_pdf):
    paths, conn = archive
    secret = "OCR-UPSTREAM-SECRET-MUST-NOT-PERSIST-IN-LEDGER"

    class RecordThenFail:
        def run(self, _pdf: Path, _out_dir: Path) -> None:
            record_llm_call("ocr_page", "fake-ocr", error=TimeoutError(secret))
            raise RuntimeError("OCR pipeline failed")

    result = ingest_pdf(source_pdf, paths, conn, pipeline=RecordThenFail())

    assert result.status == "failed"
    row = get_document(conn, result.doc_id)
    assert row is not None
    details = row.details()
    assert details["llm_usage"] is None
    assert details["llm_calls"] == [
        {
            "stage": "ocr_page",
            "model": "fake-ocr",
            "usage": None,
            "usage_known": False,
            "latency_s": None,
            "status": "error",
            "error": "TimeoutError",
        }
    ]
    assert secret not in json.dumps(details)


def test_failed_reingest_preserves_current_row_and_stages_attempt_ledger(archive, source_pdf):
    paths, conn = archive
    first = ingest_pdf(source_pdf, paths, conn, llm_enabled=False, pipeline=_Pipeline())
    before = get_document(conn, first.doc_id)
    assert before is not None

    class RecordThenFail:
        def run(self, _pdf: Path, _out_dir: Path) -> None:
            record_llm_call("ocr_page", "fake-ocr", error=TimeoutError("attempt secret"))
            raise RuntimeError("OCR pipeline failed")

    failed = ingest_pdf(
        source_pdf,
        paths,
        conn,
        reingest=True,
        llm_enabled=False,
        pipeline=RecordThenFail(),
    )

    assert failed.status == "failed"
    after = get_document(conn, first.doc_id)
    assert after is not None
    assert after.details_json == before.details_json
    staged = list(paths.runs_dir.glob(f"{first.sha256[:12]}-*"))
    assert len(staged) == 1
    attempt = json.loads((staged[0] / FILE_EXTRACTION).read_text(encoding="utf-8"))
    assert attempt["llm_calls"][0]["error"] == "TimeoutError"
    assert "attempt secret" not in json.dumps(attempt)
