"""Schema creation/migration concurrency and rollback safety."""
from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path

import pytest

from contract_archive.archive import open_archive_db
from contract_archive.archive import db as archive_db


def _open_fresh_archive_db(path_text: str) -> int:
    """Picklable worker: concurrent processes all initialize the same empty DB."""
    conn = open_archive_db(Path(path_text))
    try:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
        ).fetchone()
        return archive_db.get_schema_version(conn)
    finally:
        conn.close()


def test_fresh_archive_db_initialization_is_serialized_across_processes(tmp_path):
    db_path = tmp_path / "fresh-archive" / "db.sqlite"
    context = multiprocessing.get_context("fork")
    with context.Pool(4) as pool:
        versions = pool.map(_open_fresh_archive_db, [str(db_path)] * 4)

    assert versions == [5, 5, 5, 5]
    # The permanent inode is part of the locking correctness contract.
    assert (db_path.parent / "locks" / "schema.lock").is_file()


def test_failed_migration_script_rolls_back_schema_and_version(tmp_path, monkeypatch):
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_broken.sql").write_text(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        CREATE TABLE must_not_survive (id INTEGER PRIMARY KEY);
        INSERT INTO schema_version(version, applied_at) VALUES (1, 'now');
        INSERT INTO definitely_missing_table(value) VALUES ('boom');
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(archive_db, "MIGRATIONS_DIR", migrations)
    conn = archive_db.connect(tmp_path / "db.sqlite")
    try:
        with pytest.raises(sqlite3.OperationalError, match="definitely_missing_table"):
            archive_db.migrate(conn)
        assert not conn.in_transaction
        assert archive_db.get_schema_version(conn) == 0
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'must_not_survive'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone() is None
    finally:
        conn.close()
