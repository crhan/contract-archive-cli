"""
CLI 渲染层：把 DocumentRow / IngestResult 等数据对象格式化成展示字符串或 JSON dict。

这里只放与 typer/console 无关的纯函数（输入数据对象，输出字符串/dict），
便于单测、也让 cli.py 专注命令定义与参数解析。函数对入参做鸭子类型，不依赖
具体 model 类型，避免反向 import。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def status_color(s: str) -> str:
    """status 着色：ok 绿 / partial 黄 / failed 红。"""
    color = {"ok": "green", "partial": "yellow", "failed": "red"}.get(s, "white")
    return f"[{color}]{s}[/{color}]"


def subject_of(r) -> str:
    """list 用的『主体』列：优先信封 parties，回退合同甲乙方。截断防撑宽。"""
    parties = r.details().get("parties") or []
    if not parties:
        parties = [p for p in (r.party_a, r.party_b) if p]
    if not parties:
        return "-"
    s = "、".join(parties[:2])
    return s if len(s) <= 20 else s[:19] + "…"


def display_amount(r) -> str:
    """
    list 金额列：有计算合计（computed_total_value）优先显示并标 *，
    否则回退抽取的主金额（primary_amount_value），都没有则 '-'。
    """
    total = r.details().get("computed_total_value")
    if isinstance(total, (int, float)):
        return f"¥{total:,.0f}[cyan]*[/cyan]"
    if r.primary_amount_value is not None:
        return f"¥{r.primary_amount_value:,.0f}"
    return "-"


def completeness_mark(r) -> str:
    """
    list『完整』列：仅合同有值。疑似缺红色警示，其余从简。
    'incomplete'→红警；'complete'→绿勾；'unknown'→黄问号；None（非合同/未判）→灰横。
    """
    s = getattr(r, "completeness_status", None)
    return {
        "incomplete": "[red]⚠ 疑似缺[/red]",
        "complete": "[green]✓[/green]",
        "unknown": "[yellow]?[/yellow]",
    }.get(s, "[dim]-[/dim]")


def period_str(a: dict) -> str:
    """金额覆盖区间的展示标注，如 ' [2025-01-01~2025-12-31]'；无区间返回空串。"""
    start, end = a.get("period_start"), a.get("period_end")
    if not start and not end:
        return ""
    return f" [dim][{start or '?'}~{end or '?'}][/dim]"


def local_time(iso_utc: Optional[str]) -> str:
    """
    入库时间 UTC ISO（'2026-05-24T23:05:04Z'）→ 本地时区展示串。
    存储保持 UTC（可移植、可比较），仅展示时转本地。解析失败则原样返回。
    """
    if not iso_utc:
        return "-"
    try:
        dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return iso_utc


def ingest_result_to_dict(r) -> dict:
    """IngestResult → JSON 友好 dict（pdf_path 转字符串）。"""
    return {
        "pdf_path": str(r.pdf_path),
        "sha256": r.sha256,
        "status": r.status,
        "doc_id": r.doc_id,
        "mineru_duration_s": r.mineru_duration_s,
        "llm_duration_s": r.llm_duration_s,
        "error_message": r.error_message,
        "skipped_reason": r.skipped_reason,
    }


def seal_rows_to_dict(rows) -> list[dict]:
    """SealRow 列表 → JSON 友好 dict 列表（seals --format json 用）。"""
    return [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "owner": r.owner,
            "seal_type": r.seal_type,
            "raw_text": r.raw_text,
        }
        for r in rows
    ]


def row_to_dict(r) -> dict:
    """DocumentRow → JSON 友好 dict（list/search/show 的 --format json 用）。"""
    details = r.details()
    return {
        "id": r.id,
        "sha256": r.sha256,
        "status": r.status,
        "doc_type": r.doc_type,
        "title": r.title,
        "summary": r.summary,
        "primary_date": r.primary_date,
        "primary_amount_value": r.primary_amount_value,
        "computed_total_value": details.get("computed_total_value"),
        "seals": details.get("seals"),
        "sub_agreements": details.get("sub_agreements"),
        "completeness": details.get("completeness"),
        "completeness_status": r.completeness_status,
        "details": details,
        "contract_name": r.contract_name,
        "party_a": r.party_a,
        "party_b": r.party_b,
        "amount_text": r.amount_text,
        "amount_value": r.amount_value,
        "sign_date": r.sign_date,
        "expire_date": r.expire_date,
        "auto_renewal": bool(r.auto_renewal) if r.auto_renewal is not None else None,
        "risk_clauses": r.risk_clauses,
        "obligations": [
            {"actor": o.actor, "action": o.action,
             "deadline": o.deadline, "evidence": o.evidence}
            for o in r.obligations
        ],
        "overall_confidence": r.overall_confidence,
        "source_path": r.source_path,
        "output_dir": r.output_dir,
        "ingested_at": r.ingested_at,
    }
