"""
SQLite 连接 + schema 迁移。

设计要点：
- 用 stdlib sqlite3，不引 SQLAlchemy（单表单进程，ORM 是负债）
- 每个连接强制执行 PRAGMA：WAL + foreign_keys=ON + busy_timeout=5000
- schema_version 表 + migrations/*.sql 文件按版本顺序执行
- 退出前 wal_checkpoint(TRUNCATE)：清空 -wal 文件，避免拷贝 db 时丢数据
"""
from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
MIGRATION_PATTERN = re.compile(r"^(\d{3})_.+\.sql$")


def utc_now_iso() -> str:
    """统一时间戳格式（带 Z 后缀的 UTC ISO8601，字典序 = 时间序）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: Path) -> sqlite3.Connection:
    """
    打开连接 + 应用必要 PRAGMA。

    注意：
    - foreign_keys 不是持久 PRAGMA，每次新连接默认 OFF，必须手动开
    - busy_timeout 防止并发写时立即 SQLITE_BUSY
    - row_factory 改 sqlite3.Row 让结果支持 row["col_name"] 访问
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        isolation_level=None,  # 自动提交模式，事务用显式 BEGIN/COMMIT 控制
        timeout=10.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """显式事务：BEGIN IMMEDIATE 立即获取写锁，避免升级死锁。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def checkpoint(conn: sqlite3.Connection) -> None:
    """强制 WAL checkpoint，清空 -wal 文件。退出前调用，避免拷 db 丢数据。"""
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError as e:
        logger.warning("wal_checkpoint failed: %s", e)


def get_schema_version(conn: sqlite3.Connection) -> int:
    """读 schema_version 表。表不存在视为版本 0（新库）。"""
    try:
        row = conn.execute(
            "SELECT MAX(version) AS v FROM schema_version"
        ).fetchone()
        return int(row["v"]) if row and row["v"] is not None else 0
    except sqlite3.OperationalError:
        return 0


def discover_migrations() -> list[tuple[int, Path]]:
    """扫描 migrations/ 目录，按版本号升序返回 [(version, path), ...]。"""
    found: list[tuple[int, Path]] = []
    if not MIGRATIONS_DIR.exists():
        return found
    for f in MIGRATIONS_DIR.iterdir():
        m = MIGRATION_PATTERN.match(f.name)
        if m:
            found.append((int(m.group(1)), f))
    found.sort(key=lambda x: x[0])
    return found


def migrate(conn: sqlite3.Connection) -> int:
    """
    应用所有未应用的迁移。返回最终 schema_version。

    注意：executescript() 内部会自动 COMMIT 当前事务，所以不能再用 transaction()
    包裹（会出现 "no transaction is active" 错误）。失败回滚由 SQLite 自身的
    事务语义保证——脚本里若有 BEGIN/COMMIT，executescript 会按 SQL 内容执行。
    本工具的 migration 文件不写 BEGIN/COMMIT，让 SQLite 走自动事务即可。
    """
    current = get_schema_version(conn)
    applied = 0
    for version, path in discover_migrations():
        if version <= current:
            continue
        sql = path.read_text(encoding="utf-8")
        logger.info("applying migration %s (version=%d)", path.name, version)
        conn.executescript(sql)
        applied += 1
    final = get_schema_version(conn)
    if applied:
        logger.info("migrations applied: %d, schema_version=%d", applied, final)
    return final


def open_archive_db(db_path: Path) -> sqlite3.Connection:
    """打开档案库 DB（必要时建表 + 迁移）。"""
    conn = connect(db_path)
    migrate(conn)
    return conn
