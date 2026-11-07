"""Generation-based archive publication safety tests.

These tests exercise the file/DB boundary directly.  They deliberately keep OCR
small and local: the safety contract must not depend on an external model.
"""
from __future__ import annotations

from contextlib import contextmanager
import multiprocessing
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from contract_archive.archive import (
    ArchivePaths,
    get_document,
    ingest_pdf,
    open_archive_db,
    re_extract,
    update_extraction,
)
from contract_archive.cli_render import archived_doc_dir
from contract_archive.archive.ingest import _new_staging, _publish_staging
from contract_archive.schemas import ContractExtraction, ExtractionConfidence


class SafetyPipeline:
    """Minimal picklable OCR stub used both in-process and from forked workers."""

    name = "safety-ocr"

    def __init__(self, text: str = "# safety document\n", delay_s: float = 0.0):
        self.text = text
        self.delay_s = delay_s

    def run(self, _pdf_path: Path, out_dir: Path):
        if self.delay_s:
            import time

            time.sleep(self.delay_s)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "markdown.md").write_text(self.text, encoding="utf-8")
        (out_dir / "raw_text.txt").write_text(self.text, encoding="utf-8")


@pytest.fixture
def archive_paths(tmp_path) -> ArchivePaths:
    return ArchivePaths(tmp_path / "archive")


@pytest.fixture
def conn(archive_paths):
    archive_paths.ensure()
    connection = open_archive_db(archive_paths.db_path)
    yield connection
    connection.close()


@pytest.fixture
def source_pdf(tmp_path) -> Path:
    path = tmp_path / "input" / "source.pdf"
    path.parent.mkdir()
    path.write_bytes(b"%PDF-1.4 safety\n" + b"s" * 4096)
    return path


def _ingest(source_pdf: Path, paths: ArchivePaths, conn, text: str = "# safety document\n"):
    return ingest_pdf(source_pdf, paths, conn, llm_enabled=False, pipeline=SafetyPipeline(text))


def _current_dir(paths: ArchivePaths, conn, doc_id: int) -> Path:
    row = get_document(conn, doc_id)
    assert row is not None
    return paths.resolve_document_dir(row)


def test_source_snapshot_is_a_copy_not_a_link(archive_paths, conn, source_pdf):
    result = _ingest(source_pdf, archive_paths, conn)
    source_snapshot = _current_dir(archive_paths, conn, result.doc_id) / "source.pdf"
    original_bytes = source_snapshot.read_bytes()

    # A hard link would make this mutation silently rewrite the archived source as well.
    source_pdf.write_bytes(b"%PDF-1.4 rewritten\n" + b"x" * 4096)

    assert source_snapshot.read_bytes() == original_bytes
    assert source_snapshot.read_bytes() != source_pdf.read_bytes()


@pytest.mark.parametrize(
    "relative",
    [Path("documents"), Path("tmp"), Path("tmp") / "runs", Path("locks")],
)
def test_controlled_archive_directory_symlinks_are_rejected_before_writes(tmp_path, relative):
    """No archive bootstrap path may create a child through a poisoned symlink."""
    root = tmp_path / "archive"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    poisoned = root / relative
    poisoned.parent.mkdir(parents=True, exist_ok=True)
    poisoned.symlink_to(outside, target_is_directory=True)

    paths = ArchivePaths(root)
    with pytest.raises(ValueError, match="symlinked archive directory"):
        paths.ensure()

    assert list(outside.iterdir()) == []


def test_staging_and_publish_recheck_controlled_directories(tmp_path):
    """Validation is repeated at staging allocation and immediately before rename."""
    paths = ArchivePaths(tmp_path / "archive")
    paths.ensure()
    sha = "a" * 64
    outside = tmp_path / "outside"
    outside.mkdir()

    paths.runs_dir.rmdir()
    paths.runs_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked archive directory"):
        _new_staging(paths, sha)
    assert list(outside.iterdir()) == []

    paths.runs_dir.unlink()
    paths.ensure()
    staging = paths.staging_dir(sha, "123e4567-e89b-42d3-a456-426614174000")
    staging.mkdir()
    generation = paths.generation_dir(sha, "223e4567-e89b-42d3-a456-426614174000")
    paths.documents_dir.rmdir()
    paths.documents_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked archive directory"):
        _publish_staging(paths, staging, generation)
    assert list(outside.iterdir()) == []


def test_source_changed_after_slot_lock_refuses_old_sha_dedup(
    archive_paths, conn, source_pdf, monkeypatch
):
    first = _ingest(source_pdf, archive_paths, conn, "# original\n")
    before = get_document(conn, first.doc_id)
    assert before is not None
    old_dir = archive_paths.resolve_document_dir(before)
    calls = 0

    class MustNotRun:
        def run(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("OCR must not run after source hash changes under slot lock")

    real_slot_lock = ArchivePaths.slot_lock

    @contextmanager
    def replace_source_after_lock(self, sha):
        with real_slot_lock(self, sha):
            source_pdf.write_bytes(b"%PDF-1.4 replacement B\n" + b"b" * 4096)
            yield

    monkeypatch.setattr(ArchivePaths, "slot_lock", replace_source_after_lock)
    with pytest.raises(ValueError, match="source changed while waiting.*retry"):
        ingest_pdf(source_pdf, archive_paths, conn, pipeline=MustNotRun())

    after = get_document(conn, first.doc_id)
    assert after is not None
    assert after.sha256 == before.sha256
    assert after.output_dir == before.output_dir
    assert old_dir.exists()
    assert calls == 0
    assert conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"] == 1


def test_reingest_db_failure_keeps_old_pointer_and_generation(
    archive_paths, conn, source_pdf, monkeypatch
):
    first = _ingest(source_pdf, archive_paths, conn, "# first generation\n")
    before = get_document(conn, first.doc_id)
    assert before is not None
    old_dir = archive_paths.resolve_document_dir(before)
    old_markdown = (old_dir / "ocr" / "markdown.md").read_text(encoding="utf-8")

    def fail_insert_risks(*_args, **_kwargs):
        raise RuntimeError("injected DB commit failure")

    # Raise after replace_document has issued its UPDATE/child-table DELETE statements, so the
    # repository transaction itself must roll everything back rather than only avoiding a call.
    monkeypatch.setattr("contract_archive.archive.repository._insert_risks", fail_insert_risks)
    with pytest.raises(RuntimeError, match="injected DB commit failure"):
        ingest_pdf(
            source_pdf,
            archive_paths,
            conn,
            reingest=True,
            llm_enabled=False,
            pipeline=SafetyPipeline("# replacement generation\n"),
        )

    after = get_document(conn, first.doc_id)
    assert after is not None
    assert after.output_dir == before.output_dir
    assert after.status == before.status
    assert (old_dir / "ocr" / "markdown.md").read_text(encoding="utf-8") == old_markdown
    # The already-published new generation is an auditable orphan, not a replacement/deletion.
    assert len(list(archive_paths.documents_dir.glob(f"{first.sha256[:12]}.gen-*"))) == 2


def test_reextract_copies_generation_and_switches_pointer_for_success_and_partial(
    archive_paths, conn, source_pdf, monkeypatch
):
    first = _ingest(source_pdf, archive_paths, conn, "# original OCR\n")
    original = get_document(conn, first.doc_id)
    assert original is not None
    original_dir = archive_paths.resolve_document_dir(original)

    successful = re_extract(first.doc_id, archive_paths, conn, llm_enabled=False)
    assert successful.status == "ok"
    after_success = get_document(conn, first.doc_id)
    assert after_success is not None
    success_dir = archive_paths.resolve_document_dir(after_success)
    assert success_dir != original_dir
    assert original_dir.exists() and success_dir.exists()
    assert (success_dir / "source.pdf").read_bytes() == (original_dir / "source.pdf").read_bytes()
    assert (success_dir / "source.pdf").stat().st_ino != (original_dir / "source.pdf").stat().st_ino
    assert after_success.ingested_at == original.ingested_at
    assert after_success.ocr_duration_s == original.ocr_duration_s

    def fail_extract(*_args, **_kwargs):
        raise RuntimeError("injected extraction failure")

    monkeypatch.setattr("contract_archive.archive.ingest._run_extraction", fail_extract)
    partial = re_extract(first.doc_id, archive_paths, conn, llm_enabled=False)
    assert partial.status == "partial"
    after_partial = get_document(conn, first.doc_id)
    assert after_partial is not None
    partial_dir = archive_paths.resolve_document_dir(after_partial)
    assert partial_dir != success_dir
    assert partial_dir.exists() and success_dir.exists()
    assert after_partial.output_dir == str(partial_dir)


def test_reextract_db_failure_leaves_old_pointer_and_metadata(
    archive_paths, conn, source_pdf, monkeypatch
):
    first = _ingest(source_pdf, archive_paths, conn)
    before = get_document(conn, first.doc_id)
    assert before is not None
    old_dir = archive_paths.resolve_document_dir(before)

    def fail_insert_risks(*_args, **_kwargs):
        raise RuntimeError("injected extraction commit failure")

    monkeypatch.setattr("contract_archive.archive.repository._insert_risks", fail_insert_risks)
    with pytest.raises(RuntimeError, match="injected extraction commit failure"):
        re_extract(first.doc_id, archive_paths, conn, llm_enabled=False)

    after = get_document(conn, first.doc_id)
    assert after is not None
    assert after.output_dir == before.output_dir
    assert after.status == before.status
    assert archive_paths.resolve_document_dir(after) == old_dir


def test_failed_reingest_preserves_existing_ok_row_and_generation(
    archive_paths, conn, source_pdf
):
    first = _ingest(source_pdf, archive_paths, conn)
    update_extraction(
        conn,
        first.doc_id,
        status="ok",
        llm_duration_s=0.1,
        error_message=None,
        extraction=ContractExtraction(contract_name="不可丢失的原字段", party_a="甲方"),
        confidence=ExtractionConfidence(overall=0.8),
    )
    before = get_document(conn, first.doc_id)
    assert before is not None
    old_dir = archive_paths.resolve_document_dir(before)
    old_details = before.details_json

    class FailingPipeline:
        name = "failing"

        def run(self, *_args, **_kwargs):
            raise RuntimeError("OCR failed on reingest")

    result = ingest_pdf(
        source_pdf,
        archive_paths,
        conn,
        reingest=True,
        llm_enabled=False,
        pipeline=FailingPipeline(),
    )
    assert result.status == "failed"
    assert result.doc_id == first.doc_id

    after = get_document(conn, first.doc_id)
    assert after is not None
    assert after.output_dir == before.output_dir
    assert after.status == "ok"
    assert after.contract_name == "不可丢失的原字段"
    assert after.details_json == old_details
    assert old_dir.exists()
    assert list(archive_paths.failed_logs_dir.glob("*.log"))


def _ingest_in_subprocess(pdf_path: str, archive_root: str) -> tuple[str, int | None]:
    """Top-level worker so multiprocessing can pickle it independently of pytest state."""
    paths = ArchivePaths(Path(archive_root))
    connection = open_archive_db(paths.db_path)
    try:
        result = ingest_pdf(
            Path(pdf_path),
            paths,
            connection,
            llm_enabled=False,
            pipeline=SafetyPipeline("# concurrent OCR\n", delay_s=0.2),
        )
        return result.status, result.doc_id
    finally:
        connection.close()


def test_same_pdf_multiprocess_creates_one_row_and_one_generation(archive_paths, conn, source_pdf):
    # The fixture has already migrated the DB before child processes open independent connections.
    context = multiprocessing.get_context("fork")
    with context.Pool(2) as pool:
        outcomes = pool.starmap(
            _ingest_in_subprocess,
            [(str(source_pdf), str(archive_paths.root))] * 2,
        )

    statuses = sorted(status for status, _doc_id in outcomes)
    assert statuses == ["ok", "skipped"]
    sha = next(doc.sha256 for doc in [get_document(conn, outcomes[0][1] or outcomes[1][1])] if doc)
    assert conn.execute("SELECT COUNT(*) AS count FROM documents WHERE sha256 = ?", (sha,)).fetchone()["count"] == 1
    assert len(list(archive_paths.documents_dir.glob(f"{sha[:12]}.gen-*"))) == 1
    assert archive_paths.slot_lock_path(sha).is_file()  # retained: never unlink/recreate a lock inode


def test_resolve_document_dir_supports_legacy_and_relocated_generation(tmp_path):
    sha = "a" * 64
    old_root = tmp_path / "old-archive"
    new_paths = ArchivePaths(tmp_path / "new-archive")
    new_paths.ensure()

    legacy = new_paths.doc_dir(sha)
    legacy.mkdir()
    legacy_row = SimpleNamespace(
        sha256=sha,
        output_dir=str(old_root / "documents" / sha[:12]),
    )
    assert new_paths.resolve_document_dir(legacy_row) == legacy.resolve()

    generation_name = f"{sha[:12]}.gen-123e4567-e89b-42d3-a456-426614174000"
    generation = new_paths.documents_dir / generation_name
    generation.mkdir()
    generation_row = SimpleNamespace(
        sha256=sha,
        output_dir=str(old_root / "documents" / generation_name),
    )
    assert new_paths.resolve_document_dir(generation_row) == generation.resolve()
    assert archived_doc_dir(generation_row, new_paths.root) == generation.resolve()

    bad_row = SimpleNamespace(sha256=sha, output_dir="/etc/passwd")
    with pytest.raises(ValueError, match="untrusted document directory"):
        new_paths.resolve_document_dir(bad_row)
    assert archived_doc_dir(bad_row, new_paths.root).parent == new_paths.documents_dir.resolve()

    outside = tmp_path / "outside"
    outside.mkdir()
    generation.rmdir()
    generation.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked document directory"):
        new_paths.resolve_document_dir(generation_row)


def test_raw_uses_relocated_generation_pointer(archive_paths, conn, source_pdf):
    """The read-side command must not trust an old absolute output_dir."""
    from typer.testing import CliRunner

    from contract_archive.archive import checkpoint
    from contract_archive.cli import app

    result = _ingest(source_pdf, archive_paths, conn, "relocated raw payload")
    row = get_document(conn, result.doc_id)
    assert row is not None
    name = Path(row.output_dir).name
    conn.execute(
        "UPDATE documents SET output_dir = ? WHERE id = ?",
        (str(archive_paths.root.parent / "old-place" / "documents" / name), row.id),
    )
    checkpoint(conn)

    cli_result = CliRunner().invoke(
        app, ["raw", str(row.id), "--archive", str(archive_paths.root)]
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "relocated raw payload" in cli_result.stdout


def test_purge_candidates_verify_full_sha_and_reject_symlinks_before_db_delete(
    archive_paths, conn, source_pdf, tmp_path
):
    """A same-prefix collision is retained; an unsafe candidate aborts purge."""
    from typer.testing import CliRunner

    from contract_archive.archive import checkpoint
    from contract_archive.cli import app

    first = _ingest(source_pdf, archive_paths, conn)
    second = re_extract(first.doc_id, archive_paths, conn, llm_enabled=False)
    assert second.status == "ok"
    row = get_document(conn, first.doc_id)
    assert row is not None

    # Same short prefix, different full SHA: its controlled generation name is
    # intentionally discoverable but source hash verification excludes it.
    other_sha = row.sha256[:12] + "b" * 52
    collision = archive_paths.generation_dir(
        other_sha, "123e4567-e89b-42d3-a456-426614174000"
    )
    collision.mkdir(parents=True)
    (collision / "source.pdf").write_bytes(b"different PDF bytes")
    current_dirs = archive_paths.document_dirs_for_sha(row.sha256)
    assert collision not in current_dirs

    # A controlled-looking symlink is a validation failure, not a file to follow
    # or a reason to delete the DB row.
    unsafe = archive_paths.generation_dir(
        row.sha256, "223e4567-e89b-42d3-a456-426614174000"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    unsafe.symlink_to(outside, target_is_directory=True)
    checkpoint(conn)

    refused = CliRunner().invoke(
        app,
        [
            "delete", str(row.id), "--archive", str(archive_paths.root),
            "--purge-files", "--yes",
        ],
    )
    assert refused.exit_code == 1, refused.output
    assert get_document(conn, row.id) is not None
    assert collision.exists()

    unsafe.unlink()
    checkpoint(conn)
    deleted = CliRunner().invoke(
        app,
        [
            "delete", str(row.id), "--archive", str(archive_paths.root),
            "--purge-files", "--yes",
        ],
    )
    assert deleted.exit_code == 0, deleted.output
    assert get_document(conn, row.id) is None
    assert collision.exists()  # prefix collision never gets purged
    assert not archive_paths.resolve_document_dir(
        sha256=row.sha256, output_dir=row.output_dir
    ).exists()


def test_purge_rmtree_failure_is_nonzero_and_reports_exact_residue(
    archive_paths, conn, source_pdf, monkeypatch
):
    """A committed DB delete must never be reported as a fully successful purge."""
    from typer.testing import CliRunner

    from contract_archive.archive import checkpoint
    from contract_archive.cli import app

    first = _ingest(source_pdf, archive_paths, conn)
    re_extract(first.doc_id, archive_paths, conn, llm_enabled=False)
    row = get_document(conn, first.doc_id)
    assert row is not None
    expected_residue = archive_paths.document_dirs_for_sha(row.sha256)
    assert len(expected_residue) == 2
    attempted: list[Path] = []

    def disk_error(path, *_args, **_kwargs):
        attempted.append(Path(path))
        raise OSError("simulated disk error")

    monkeypatch.setattr(shutil, "rmtree", disk_error)
    checkpoint(conn)
    result = CliRunner().invoke(
        app,
        [
            "delete", str(row.id), "--archive", str(archive_paths.root),
            "--purge-files", "--yes",
        ],
    )

    assert result.exit_code == 1, result.output
    assert get_document(conn, row.id) is None
    assert set(attempted) == set(expected_residue)
    assert all(path.exists() for path in expected_residue)
    assert "数据库记录已删除，但以下目录未清理" in result.output
    assert all(str(path) in result.output for path in expected_residue)
    assert "✓ deleted DB row" not in result.output
