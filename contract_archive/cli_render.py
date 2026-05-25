"""
CLI 渲染层：把 DocumentRow / IngestResult 等数据对象格式化成展示字符串 / JSON dict / rich Table。

这里只放与 typer/console 无关的纯函数（输入数据对象，输出字符串/dict/Table 对象，
不碰 stdout——打印交给 cli.py 的 console），便于单测、也让 cli.py 专注命令定义与参数解析。
函数对入参做鸭子类型，不依赖具体 model 类型，避免反向 import。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from rich.table import Table


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
        # 结构化错误（code/category/retryable）；成功/跳过为 None。供 Agent 判可否重试。
        "error": r.error.model_dump() if r.error else None,
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
        "llm_model": details.get("llm_model"),
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


def build_show_table(row) -> Table:
    """
    show 命令的单文档详情表（纯函数：输入 DocumentRow，输出 rich Table，不打印）。

    从 cli.py 下沉至此，让 cli.py 专注命令定义与参数解析。逻辑与原 show 完全一致，
    仅末尾 return 而非 console.print。
    """
    table = Table(title=f"Document #{row.id} · {row.doc_type or '?'} ({status_color(row.status)})")
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    table.add_row("sha256", row.sha256)
    table.add_row("source_path", row.source_path)
    table.add_row("output_dir", row.output_dir)
    table.add_row("ingested_at", local_time(row.ingested_at))
    # mineru_s/llm_s（执行耗时）是运维遥测，不属于档案内容——不在 show 展示。
    # DB 列仍保留并由 ingest 写入，需要时可查 jsonl 日志。
    if row.error_message:
        table.add_row("[red]error[/red]", row.error_message)

    # ---- 通用信封（任何文档类型）----
    table.add_row("", "")
    table.add_row("[bold]doc_type[/bold]", row.doc_type or "-")
    table.add_row("[bold]title[/bold]", row.title or row.contract_name or "-")
    if row.summary:
        table.add_row("summary", row.summary)

    # 合同有专属列（party/到期/续约），日期走表列；其余类型走 details 的主体/日期。
    det = row.details()
    is_contract = bool(row.contract_name or row.party_a or row.party_b)
    if is_contract:
        table.add_row("", "")
        table.add_row("party_a", row.party_a or "-")
        table.add_row("party_b", row.party_b or "-")
        table.add_row("sign_date", row.sign_date or "-")
        table.add_row("expire_date", row.expire_date or "-")
        table.add_row(
            "auto_renewal",
            "是" if row.auto_renewal == 1 else ("否" if row.auto_renewal == 0 else "-"),
        )
    else:
        parties = det.get("parties") or []
        if parties:
            table.add_row("主体", "\n".join(f"• {p}" for p in parties))
        key_dates = det.get("key_dates") or []
        if key_dates:
            table.add_row(
                "日期",
                "\n".join(f"• {d.get('label', '')}: {d.get('date') or '-'}" for d in key_dates),
            )

    # ---- 金额 / 合计 / 字段：所有类型通用 ----
    # 合同此前只显示单个主金额，付款明细（首期/余款）和付款方式（fields）被吞；提到这里通用展示。
    amounts = det.get("amounts") or []
    if amounts:
        lines = []
        for a in amounts:
            v = a.get("value")
            vs = f"（¥{v:,.2f}）" if isinstance(v, (int, float)) else ""
            mark = " [cyan]✓计入合计[/cyan]" if a.get("is_total_component") else ""
            if a.get("is_installment"):
                mark += " [magenta]分期[/magenta]"
            lines.append(
                f"• {a.get('label', '')}: {a.get('text', '')}{vs}{period_str(a)}{mark}"
            )
            ev = a.get("evidence") or ""
            if ev:
                lines.append(f"    [dim]↳ 出处：{ev}[/dim]")
        table.add_row("金额", "\n".join(lines))
    elif row.amount_text:  # details 无 amounts 的旧数据/回退，至少显示表列主金额
        table.add_row(
            "金额",
            f"{row.amount_text} (¥{row.amount_value:,.2f})" if row.amount_value is not None else row.amount_text,
        )
    total = det.get("computed_total_value")
    if isinstance(total, (int, float)):
        table.add_row(
            "[bold]合计(计算)[/bold]",
            f"[cyan]¥{total:,.2f}[/cyan] [dim](上方标✓项之和，非抽取值)[/dim]",
        )
    fields = det.get("fields") or []
    if fields:
        table.add_row(
            "字段",
            "\n".join(f"• {f.get('label', '')}: {f.get('value', '')}" for f in fields),
        )

    # 印章：跨文档类型通用，放分支外（合同恰恰最常盖章）。det 只在非合同分支定义，
    # 这里用 row.details() 现取，避免合同分支 NameError。
    seals = row.details().get("seals") or []
    if seals:
        lines = []
        for s in seals:
            owner = s.get("owner") or "?"
            stype = s.get("seal_type")
            head = owner + (f" · {stype}" if stype else "")
            raw = s.get("raw_text") or ""
            lines.append(f"• {head}  [dim]{raw}[/dim]" if raw else f"• {head}")
        table.add_row("[bold]印章[/bold]", "\n".join(lines))

    # 附属协议（补充协议等）：一份 PDF 可能含主协议 + N 份补充协议，各有独立签章。
    subs = row.details().get("sub_agreements") or []
    if subs:
        lines = []
        for sub in subs:
            head = f"[bold]{sub.get('title') or '附属协议'}[/bold]"
            if sub.get("sign_date"):
                head += f"  [dim]{sub['sign_date']}[/dim]"
            lines.append(head)
            if sub.get("summary"):
                lines.append(f"  {sub['summary']}")
            sseals = sub.get("seals") or []
            if sseals:
                for s in sseals:
                    owner = s.get("owner") or "?"
                    stype = s.get("seal_type")
                    lines.append(f"  印章: {owner}" + (f" · {stype}" if stype else ""))
            else:
                lines.append("  [dim]印章: 无[/dim]")
        table.add_row("[bold]补充协议[/bold]", "\n".join(lines))

    # 完整性核查：仅合同有此块（非合同 details 里 completeness 为 None）。
    comp = row.details().get("completeness")
    if comp:
        status = comp.get("status")
        issues = comp.get("issues") or []
        if status == "complete":
            table.add_row("[bold]完整性[/bold]", "[green]✓ 要素与签章齐全[/green]")
        elif status == "incomplete":
            lines = ["[red]⚠ 疑似不完整[/red] [dim](签章经落款页核查；要素/金额据原文，可翻回核对)[/dim]"]
            cat_label = {"signature": "签章", "amount": "金额", "field": "要素"}
            for it in issues:
                cat = cat_label.get(it.get("category"), "要素")
                detail = it.get("detail") or ""
                tail = f" — [dim]{detail}[/dim]" if detail else ""
                lines.append(f"• [{cat}] {it.get('item', '')}{tail}")
                evidence = it.get("evidence") or ""
                if evidence:
                    lines.append(f"    [dim]↳ 出处：{evidence}[/dim]")
            table.add_row("[bold]完整性[/bold]", "\n".join(lines))
        else:  # unknown
            table.add_row("[bold]完整性[/bold]", "[yellow]? 信息不足，未能判定[/yellow]")

    table.add_row(
        "llm_model",
        row.details().get("llm_model") or "[dim]- (旧抽取未记录，重抽后显示)[/dim]",
    )
    table.add_row(
        "overall_confidence",
        f"{row.overall_confidence:.2f}" if row.overall_confidence is not None else "-",
    )
    if row.obligations:
        table.add_row("", "")
        for actor_key, label in (
            ("party_a", "[bold]甲方动作[/bold]"),
            ("party_b", "[bold]乙方动作[/bold]"),
            ("both",    "[bold]双方动作[/bold]"),
        ):
            items = [o for o in row.obligations if o.actor == actor_key]
            if not items:
                continue
            lines = []
            for o in items:
                dl = o.deadline or "[dim]无日期[/dim]"
                lines.append(f"• [{dl}] {o.action}")
            table.add_row(label, "\n".join(lines))
    if row.risk_clauses:
        table.add_row(
            "[bold]risk_clauses[/bold]",
            "\n".join(f"• {c}" for c in row.risk_clauses),
        )
    return table
