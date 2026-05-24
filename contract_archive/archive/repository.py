"""
档案库 DAO（数据访问层）。

只暴露业务操作，调用方不直接拼 SQL。所有写操作：
- 显式事务（with transaction(conn)）
- INSERT documents 用 ON CONFLICT(sha256) DO NOTHING 避免吃 autoincrement seq
- reingest 时 risk_clauses 先 DELETE 再批量 INSERT，同一事务

不引入 ORM —— 单表项目，dict ↔ row 手写更轻。
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..schemas import ContractExtraction, ExtractionConfidence, ObligationItem
from .db import transaction, utc_now_iso

logger = logging.getLogger(__name__)


# ---------- 类型 ----------


@dataclass
class DocumentRow:
    """单条档案记录（documents 表 + risk_clauses 聚合）。"""

    id: int
    sha256: str
    source_path: str
    output_dir: str
    ingested_at: str
    mineru_duration_s: Optional[float]
    llm_duration_s: Optional[float]
    status: str
    error_message: Optional[str]
    contract_name: Optional[str]
    party_a: Optional[str]
    party_b: Optional[str]
    amount_text: Optional[str]
    amount_cents: Optional[int]
    sign_date: Optional[str]
    expire_date: Optional[str]
    auto_renewal: Optional[int]
    overall_confidence: Optional[float]
    risk_clauses: list[str] = field(default_factory=list)
    obligations: list[ObligationItem] = field(default_factory=list)

    @property
    def amount_value(self) -> Optional[float]:
        """amount_cents → 元，方便展示。"""
        return None if self.amount_cents is None else self.amount_cents / 100.0

    @property
    def short_sha(self) -> str:
        return self.sha256[:12]


# ---------- 工具 ----------


def _amount_to_cents(value: Optional[float]) -> Optional[int]:
    """元 → 分。None 透传。四舍五入到分，防 0.005 漂移。"""
    if value is None:
        return None
    return int(round(value * 100))


def _row_to_document(
    row: sqlite3.Row,
    risks: list[str],
    obligations: list[ObligationItem],
) -> DocumentRow:
    return DocumentRow(
        id=row["id"],
        sha256=row["sha256"],
        source_path=row["source_path"],
        output_dir=row["output_dir"],
        ingested_at=row["ingested_at"],
        mineru_duration_s=row["mineru_duration_s"],
        llm_duration_s=row["llm_duration_s"],
        status=row["status"],
        error_message=row["error_message"],
        contract_name=row["contract_name"],
        party_a=row["party_a"],
        party_b=row["party_b"],
        amount_text=row["amount_text"],
        amount_cents=row["amount_cents"],
        sign_date=row["sign_date"],
        expire_date=row["expire_date"],
        auto_renewal=row["auto_renewal"],
        overall_confidence=row["overall_confidence"],
        risk_clauses=risks,
        obligations=obligations,
    )


# ---------- 查询 ----------


def find_by_sha(conn: sqlite3.Connection, sha256: str) -> Optional[int]:
    """sha256 → id；不存在返回 None。ingest 去重用。"""
    row = conn.execute(
        "SELECT id FROM documents WHERE sha256 = ?", (sha256,)
    ).fetchone()
    return row["id"] if row else None


def get_document(conn: sqlite3.Connection, doc_id: int) -> Optional[DocumentRow]:
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    if not row:
        return None
    return _hydrate(conn, row)


def find_by_sha_prefix(
    conn: sqlite3.Connection, prefix: str
) -> list[DocumentRow]:
    """
    sha 前缀查（show 命令支持）。
    前缀必须 >= 4 字符以避免误命中。
    """
    if len(prefix) < 4:
        raise ValueError("sha prefix must be >= 4 chars to disambiguate")
    rows = conn.execute(
        "SELECT * FROM documents WHERE sha256 LIKE ? ORDER BY ingested_at DESC",
        (prefix + "%",),
    ).fetchall()
    return [_hydrate(conn, r) for r in rows]


def _hydrate(conn: sqlite3.Connection, row: sqlite3.Row) -> DocumentRow:
    """从主表行 + 子表数据组装 DocumentRow。"""
    return _row_to_document(
        row,
        _load_risks(conn, row["id"]),
        _load_obligations(conn, row["id"]),
    )


def _load_risks(conn: sqlite3.Connection, doc_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT clause_text FROM risk_clauses WHERE doc_id = ? ORDER BY id",
        (doc_id,),
    ).fetchall()
    return [r["clause_text"] for r in rows]


@dataclass
class TodoItem:
    """跨合同 obligations 视图（list_obligations 的返回行）。"""

    obligation_id: int
    doc_id: int
    contract_name: Optional[str]
    party_a: Optional[str]
    party_b: Optional[str]
    actor: str
    action: str
    deadline: Optional[str]
    evidence: str


def list_obligations(
    conn: sqlite3.Connection,
    *,
    actor: Optional[str] = None,
    before: Optional[str] = None,
    after: Optional[str] = None,
    include_undated: bool = False,
    limit: int = 50,
) -> list[TodoItem]:
    """
    跨合同列 obligations（待办看板）。

    默认只返回带 deadline 的，按 deadline 升序。
    include_undated=True 时同时返回无日期义务（排在末尾）。
    """
    where: list[str] = []
    params: list[Any] = []
    if not include_undated:
        where.append("o.deadline IS NOT NULL")
    if actor:
        if actor not in ("party_a", "party_b", "both"):
            raise ValueError(f"actor must be party_a/party_b/both, got {actor!r}")
        where.append("o.actor = ?")
        params.append(actor)
    if before:
        where.append("(o.deadline IS NOT NULL AND o.deadline <= ?)")
        params.append(before)
    if after:
        where.append("(o.deadline IS NOT NULL AND o.deadline >= ?)")
        params.append(after)

    sql = """
        SELECT o.id AS oid, o.doc_id, o.actor, o.action, o.deadline, o.evidence,
               d.contract_name, d.party_a, d.party_b
          FROM obligations o JOIN documents d ON d.id = o.doc_id
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    # NULL deadline 排到最后（IS NULL 排序：SQLite NULLS FIRST 默认，反过来）
    sql += " ORDER BY (o.deadline IS NULL), o.deadline ASC, o.doc_id LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        TodoItem(
            obligation_id=r["oid"],
            doc_id=r["doc_id"],
            contract_name=r["contract_name"],
            party_a=r["party_a"],
            party_b=r["party_b"],
            actor=r["actor"],
            action=r["action"],
            deadline=r["deadline"],
            evidence=r["evidence"] or "",
        )
        for r in rows
    ]


def _load_obligations(conn: sqlite3.Connection, doc_id: int) -> list[ObligationItem]:
    rows = conn.execute(
        """SELECT actor, action, deadline, evidence
             FROM obligations WHERE doc_id = ?
             ORDER BY ordering, id""",
        (doc_id,),
    ).fetchall()
    return [
        ObligationItem(
            actor=r["actor"],
            action=r["action"],
            deadline=r["deadline"],
            evidence=r["evidence"] or "",
        )
        for r in rows
    ]


def list_documents(
    conn: sqlite3.Connection,
    limit: int = 50,
    order_by: str = "ingested_at",
    status: Optional[str] = None,
) -> list[DocumentRow]:
    """list 命令实现。status=None 表示全部。"""
    allowed_order = {"ingested_at", "sign_date", "expire_date", "amount_cents"}
    if order_by not in allowed_order:
        raise ValueError(f"order_by must be one of {allowed_order}")

    sql = "SELECT * FROM documents"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += f" ORDER BY {order_by} DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [_hydrate(conn, r) for r in rows]


@dataclass
class SearchFilter:
    """search 命令的过滤参数。所有 None 字段被忽略。"""

    name: Optional[str] = None         # LIKE 模糊匹配 contract_name
    party: Optional[str] = None        # LIKE 模糊匹配 party_a 或 party_b
    amount_min_cents: Optional[int] = None
    amount_max_cents: Optional[int] = None
    signed_after: Optional[str] = None
    signed_before: Optional[str] = None
    expire_before: Optional[str] = None
    auto_renewal: Optional[bool] = None
    has_risk: bool = False
    status: Optional[str] = None
    # 义务过滤：跨表 EXISTS 查询
    deadline_before: Optional[str] = None   # 找近期到期的待办
    deadline_after: Optional[str] = None
    actor: Optional[str] = None             # party_a / party_b / both
    limit: int = 50


def search_documents(
    conn: sqlite3.Connection, flt: SearchFilter
) -> list[DocumentRow]:
    """
    多字段过滤查询。
    - name / party 走 FTS5（trigram tokenizer 支持中文子串）
    - 其他字段走主表索引
    - FTS 用 IN 子查询绑定 id，避免 JOIN 后字段冲突
    """
    where: list[str] = []
    params: list[Any] = []

    if flt.name:
        where.append("contract_name LIKE ?")
        params.append(f"%{flt.name}%")
    if flt.party:
        where.append("(party_a LIKE ? OR party_b LIKE ?)")
        like = f"%{flt.party}%"
        params.append(like)
        params.append(like)
    if flt.amount_min_cents is not None:
        where.append("amount_cents >= ?")
        params.append(flt.amount_min_cents)
    if flt.amount_max_cents is not None:
        where.append("amount_cents <= ?")
        params.append(flt.amount_max_cents)
    if flt.signed_after:
        where.append("sign_date >= ?")
        params.append(flt.signed_after)
    if flt.signed_before:
        where.append("sign_date <= ?")
        params.append(flt.signed_before)
    if flt.expire_before:
        where.append("expire_date <= ?")
        params.append(flt.expire_before)
    if flt.auto_renewal is not None:
        where.append("auto_renewal = ?")
        params.append(1 if flt.auto_renewal else 0)
    if flt.has_risk:
        where.append("EXISTS (SELECT 1 FROM risk_clauses WHERE doc_id = documents.id)")
    if flt.status:
        where.append("status = ?")
        params.append(flt.status)

    # 义务过滤：用一个 EXISTS 子查询带 AND 链，所有 obligation 条件命中同一条 obligation
    obl_where: list[str] = []
    if flt.deadline_before:
        obl_where.append("deadline IS NOT NULL AND deadline <= ?")
        params.append(flt.deadline_before)
    if flt.deadline_after:
        obl_where.append("deadline IS NOT NULL AND deadline >= ?")
        params.append(flt.deadline_after)
    if flt.actor:
        if flt.actor not in ("party_a", "party_b", "both"):
            raise ValueError(f"actor must be party_a/party_b/both, got {flt.actor!r}")
        obl_where.append("actor = ?")
        params.append(flt.actor)
    if obl_where:
        # 注意把 obligation 条件参数插到主表参数之前——这里 params 是按顺序拼接的，
        # 所以 obligation 条件必须在 SQL 中先出现：放到 where 列表开头
        clause = (
            "EXISTS (SELECT 1 FROM obligations WHERE doc_id = documents.id AND "
            + " AND ".join(obl_where)
            + ")"
        )
        # 因为 params 是按 where 列表顺序追加的，且 obl 参数已经追加在末尾，
        # 这里把 clause 也加到末尾保持参数顺序一致
        where.append(clause)

    sql = "SELECT * FROM documents"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ingested_at DESC LIMIT ?"
    params.append(flt.limit)

    rows = conn.execute(sql, params).fetchall()
    return [_hydrate(conn, r) for r in rows]


# ---------- 写入 ----------


def insert_document(
    conn: sqlite3.Connection,
    *,
    sha256: str,
    source_path: str,
    output_dir: str,
    status: str,
    mineru_duration_s: Optional[float],
    llm_duration_s: Optional[float],
    error_message: Optional[str],
    extraction: Optional[ContractExtraction],
    confidence: Optional[ExtractionConfidence],
) -> Optional[int]:
    """
    新增一条档案。sha256 冲突时返回 None（已存在），不消耗 autoincrement seq。
    单事务：documents + risk_clauses 全部原子写入。
    """
    ext = extraction or ContractExtraction()
    conf = confidence or ExtractionConfidence()

    with transaction(conn):
        cursor = conn.execute(
            """
            INSERT INTO documents (
              sha256, source_path, output_dir, ingested_at,
              mineru_duration_s, llm_duration_s, status, error_message,
              contract_name, party_a, party_b,
              amount_text, amount_cents,
              sign_date, expire_date, auto_renewal,
              overall_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO NOTHING
            """,
            (
                sha256,
                source_path,
                output_dir,
                utc_now_iso(),
                mineru_duration_s,
                llm_duration_s,
                status,
                error_message,
                ext.contract_name,
                ext.party_a,
                ext.party_b,
                ext.amount,
                _amount_to_cents(ext.amount_value),
                ext.sign_date,
                ext.expire_date,
                None if ext.auto_renewal is None else int(ext.auto_renewal),
                conf.overall,
            ),
        )
        if cursor.rowcount == 0:
            return None  # 冲突，sha256 已存在
        doc_id = cursor.lastrowid
        _insert_risks(conn, doc_id, ext.risk_clauses)
        _insert_obligations(conn, doc_id, ext.obligations)
        return doc_id


def update_extraction(
    conn: sqlite3.Connection,
    doc_id: int,
    *,
    status: str,
    llm_duration_s: Optional[float],
    error_message: Optional[str],
    extraction: ContractExtraction,
    confidence: ExtractionConfidence,
) -> None:
    """
    复跑抽取（mineru 产物已存在）后更新字段。同一事务：
    risk_clauses 显式 DELETE 再 INSERT，避免重复堆积。
    """
    with transaction(conn):
        conn.execute(
            """
            UPDATE documents SET
              status = ?,
              llm_duration_s = ?,
              error_message = ?,
              contract_name = ?, party_a = ?, party_b = ?,
              amount_text = ?, amount_cents = ?,
              sign_date = ?, expire_date = ?, auto_renewal = ?,
              overall_confidence = ?
            WHERE id = ?
            """,
            (
                status,
                llm_duration_s,
                error_message,
                extraction.contract_name,
                extraction.party_a,
                extraction.party_b,
                extraction.amount,
                _amount_to_cents(extraction.amount_value),
                extraction.sign_date,
                extraction.expire_date,
                None if extraction.auto_renewal is None else int(extraction.auto_renewal),
                confidence.overall,
                doc_id,
            ),
        )
        conn.execute("DELETE FROM risk_clauses WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM obligations WHERE doc_id = ?", (doc_id,))
        _insert_risks(conn, doc_id, extraction.risk_clauses)
        _insert_obligations(conn, doc_id, extraction.obligations)


def replace_document(
    conn: sqlite3.Connection,
    doc_id: int,
    *,
    source_path: str,
    output_dir: str,
    status: str,
    mineru_duration_s: Optional[float],
    llm_duration_s: Optional[float],
    error_message: Optional[str],
    extraction: ContractExtraction,
    confidence: ExtractionConfidence,
) -> None:
    """
    reingest：mineru + 抽取都重跑。比 update_extraction 多更新 source_path/output_dir/mineru_duration。
    sha256 / id / ingested_at 不变。
    """
    with transaction(conn):
        conn.execute(
            """
            UPDATE documents SET
              source_path = ?, output_dir = ?,
              mineru_duration_s = ?, llm_duration_s = ?,
              status = ?, error_message = ?,
              contract_name = ?, party_a = ?, party_b = ?,
              amount_text = ?, amount_cents = ?,
              sign_date = ?, expire_date = ?, auto_renewal = ?,
              overall_confidence = ?
            WHERE id = ?
            """,
            (
                source_path,
                output_dir,
                mineru_duration_s,
                llm_duration_s,
                status,
                error_message,
                extraction.contract_name,
                extraction.party_a,
                extraction.party_b,
                extraction.amount,
                _amount_to_cents(extraction.amount_value),
                extraction.sign_date,
                extraction.expire_date,
                None if extraction.auto_renewal is None else int(extraction.auto_renewal),
                confidence.overall,
                doc_id,
            ),
        )
        conn.execute("DELETE FROM risk_clauses WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM obligations WHERE doc_id = ?", (doc_id,))
        _insert_risks(conn, doc_id, extraction.risk_clauses)
        _insert_obligations(conn, doc_id, extraction.obligations)


def _insert_risks(
    conn: sqlite3.Connection, doc_id: int, clauses: Iterable[str]
) -> None:
    """批量插 risk_clauses（severity 留空，未来增强）。"""
    rows = [(doc_id, c) for c in clauses if c and c.strip()]
    if not rows:
        return
    conn.executemany(
        "INSERT INTO risk_clauses(doc_id, clause_text) VALUES (?, ?)",
        rows,
    )


def _insert_obligations(
    conn: sqlite3.Connection,
    doc_id: int,
    items: Iterable[ObligationItem],
) -> None:
    """批量插 obligations，ordering 按列表顺序递增。"""
    rows = [
        (doc_id, it.actor, it.action, it.deadline, it.evidence, i)
        for i, it in enumerate(items)
        if it.action and it.action.strip()
    ]
    if not rows:
        return
    conn.executemany(
        """INSERT INTO obligations(doc_id, actor, action, deadline, evidence, ordering)
             VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )


def delete_document(conn: sqlite3.Connection, doc_id: int) -> Optional[str]:
    """
    删档案记录。返回 output_dir 路径（让调用方决定是否删文件）。
    DB 中 risk_clauses 由 ON DELETE CASCADE 自动级联。
    """
    with transaction(conn):
        row = conn.execute(
            "SELECT output_dir FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return row["output_dir"]


# ---------- 统计 ----------


@dataclass
class Stats:
    total: int
    by_status: dict[str, int]
    by_sign_month: dict[str, int]   # 'YYYY-MM' → count
    new_this_month: int
    expiring_within_30d: int


def collect_stats(conn: sqlite3.Connection) -> Stats:
    total = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]

    by_status = {
        r["status"]: r["c"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS c FROM documents GROUP BY status"
        )
    }

    by_sign_month = {
        r["m"]: r["c"]
        for r in conn.execute(
            """
            SELECT substr(sign_date, 1, 7) AS m, COUNT(*) AS c
              FROM documents WHERE sign_date IS NOT NULL
              GROUP BY m ORDER BY m
            """
        )
    }

    new_this_month = conn.execute(
        "SELECT COUNT(*) AS c FROM documents WHERE substr(ingested_at, 1, 7) = strftime('%Y-%m', 'now')"
    ).fetchone()["c"]

    expiring_within_30d = conn.execute(
        """
        SELECT COUNT(*) AS c FROM documents
        WHERE expire_date IS NOT NULL
          AND expire_date >= date('now')
          AND expire_date <= date('now', '+30 days')
        """
    ).fetchone()["c"]

    return Stats(
        total=total,
        by_status=by_status,
        by_sign_month=by_sign_month,
        new_this_month=new_this_month,
        expiring_within_30d=expiring_within_30d,
    )
