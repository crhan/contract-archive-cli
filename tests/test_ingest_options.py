"""
ingest 的 Agent-Ready 选项单测：--dry-run / --max-files / --progress ndjson。

dry-run 与 max-files 真跑（不调 MinerU/LLM）；ndjson 模式 mock ingest_pdf 验证事件流格式。
"""
import json
from pathlib import Path

from typer.testing import CliRunner

import contract_archive.cli as climod
from contract_archive.archive.ingest import IngestResult
from contract_archive.cli import app

runner = CliRunner()


def _make_pdfs(d: Path, n: int) -> None:
    for i in range(n):
        (d / f"f{i}.pdf").write_bytes(f"%PDF-1.4 fake {i}".encode())


def test_dry_run_previews_without_side_effect(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_pdfs(src, 2)
    arch = tmp_path / "arch"
    result = runner.invoke(
        app, ["ingest", str(src), "--dry-run", "--format", "json", "--archive", str(arch)]
    )
    assert result.exit_code == 0, result.output
    d = json.loads(result.stdout)
    assert d["dry_run"] is True
    assert d["total"] == 2 and d["new"] == 2 and d["already_ingested"] == 0
    assert d["estimated_llm_calls"] == 2
    assert {f["action"] for f in d["files"]} == {"new"}
    assert not arch.exists()      # dry-run 绝不建库


def test_max_files_guard_rejects_with_exit_2(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_pdfs(src, 3)
    arch = tmp_path / "arch"
    result = runner.invoke(
        app, ["ingest", str(src), "--max-files", "2", "--archive", str(arch), "--no-llm"]
    )
    assert result.exit_code == 2
    assert not arch.exists()      # 被拦在建库之前


def test_progress_ndjson_emits_stream(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _make_pdfs(src, 2)
    arch = tmp_path / "arch"

    # 不真加载 MinerU 模型、不真跑解析。
    monkeypatch.setattr(climod, "MinerUPipeline", lambda **kw: object())

    def fake_ingest(pdf, paths, conn, **kw):
        return IngestResult(
            pdf_path=pdf, sha256="sha_" + pdf.stem, status="ok", doc_id=1,
            mineru_duration_s=0.1, llm_duration_s=0.1,
        )

    monkeypatch.setattr(climod, "ingest_pdf", fake_ingest)

    result = runner.invoke(
        app, ["ingest", str(src), "--progress", "ndjson", "--archive", str(arch), "--no-llm"]
    )
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    file_events = [e for e in events if e.get("event") == "file_done"]
    summary_events = [e for e in events if e.get("event") == "summary"]
    assert len(file_events) == 2
    assert all("seq" in e and "total" in e and e["status"] == "ok" for e in file_events)
    assert [e["seq"] for e in file_events] == [1, 2]
    assert len(summary_events) == 1
    assert summary_events[0]["ok"] == 2
