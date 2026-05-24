"""
本地合同档案库 CLI。

子命令：
  ingest <path>         扫描 PDF 文件/目录，跑 MinerU + 抽取，结果入库
  list                  列出档案；status 颜色标注，支持排序
  search                按字段过滤（合同名/甲乙方/金额/日期/自动续约/风险）
  show <ident>          查看单条详情（id 或 sha 前缀 >=4 字符）
  extract <id>          只重跑抽取（不重跑 MinerU），适合 partial 修复 / 改 prompt 后重抽
  stats                 总数 / status 分布 / 按月签订分布 / 近 30 天到期数
  delete <id>           删除档案记录；默认仅删 DB 行，--purge-files 同时删文件
  vacuum                VACUUM 数据库（碎片整理）

档案库路径优先级：--archive flag > OCR_ARCHIVE_DIR env > ./archive
"""
from __future__ import annotations

import json as _json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .archive import (
    ArchivePaths,
    SearchFilter,
    checkpoint,
    collect_stats,
    delete_document,
    discover_pdfs,
    find_by_sha_prefix,
    get_document,
    ingest_pdf,
    list_documents,
    list_obligations,
    open_archive_db,
    re_extract,
    search_documents,
)
from .pipelines import MinerUPipeline

app = typer.Typer(help="本地合同档案库 CLI (MinerU + qwen3.7-max)")
console = Console()
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------- 全局 archive 路径解析 ----------


def _resolve_archive(archive_opt: Optional[Path]) -> ArchivePaths:
    """--archive flag > OCR_ARCHIVE_DIR env > ./archive"""
    if archive_opt:
        root = archive_opt
    elif os.getenv("OCR_ARCHIVE_DIR"):
        root = Path(os.getenv("OCR_ARCHIVE_DIR"))
    else:
        root = Path("./archive")
    return ArchivePaths(root=root.resolve())


_archive_opt = typer.Option(
    None,
    "--archive",
    "-a",
    help="档案库根目录；不传则用 OCR_ARCHIVE_DIR 或 ./archive",
)


# ---------- ingest ----------


@app.command()
def ingest(
    path: Path = typer.Argument(
        ..., exists=True, readable=True, help="PDF 文件或目录（目录会递归扫 *.pdf）"
    ),
    archive: Optional[Path] = _archive_opt,
    reingest: bool = typer.Option(
        False, "--reingest", help="忽略 sha256 去重，强制重跑 MinerU + 抽取"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="只跑 rule 抽取，跳过 LLM（调试用，无 API key 时也用）"
    ),
    limit: int = typer.Option(
        0, "--limit", help="最多处理 N 个文件（0 = 无限制；试跑用）"
    ),
) -> None:
    """跑 MinerU + 抽取，把合同入库。"""
    paths = _resolve_archive(archive)
    paths.ensure()
    conn = open_archive_db(paths.db_path)

    pdfs = discover_pdfs(path)
    if limit > 0:
        pdfs = pdfs[:limit]
    if not pdfs:
        console.print("[yellow]no PDFs found[/yellow]")
        raise typer.Exit(0)

    console.print(f"[cyan]found {len(pdfs)} PDF(s); archive={paths.root}[/cyan]")
    # 复用一个 MinerUPipeline 实例（避免每次重新加载模型）
    pipeline = MinerUPipeline()

    summary = {"ok": 0, "partial": 0, "failed": 0, "skipped": 0}
    for i, pdf in enumerate(pdfs, 1):
        console.rule(f"[bold cyan][{i}/{len(pdfs)}] {pdf.name}[/bold cyan]")
        try:
            result = ingest_pdf(
                pdf,
                paths,
                conn,
                reingest=reingest,
                llm_enabled=not no_llm,
                pipeline=pipeline,
            )
        except Exception as e:
            console.print(f"[red]✗ unexpected error: {e}[/red]")
            logging.getLogger(__name__).exception("ingest crashed")
            summary["failed"] += 1
            continue
        summary[result.status] = summary.get(result.status, 0) + 1
        _print_ingest_result(result)

    checkpoint(conn)
    conn.close()
    console.rule("[bold]summary[/bold]")
    console.print(
        f"ok={summary['ok']} partial={summary['partial']} "
        f"failed={summary['failed']} skipped={summary['skipped']}"
    )
    if summary["failed"]:
        raise typer.Exit(1)


def _print_ingest_result(r) -> None:
    color = {"ok": "green", "partial": "yellow", "failed": "red", "skipped": "blue"}.get(
        r.status, "white"
    )
    mineru_s = f"{r.mineru_duration_s:.1f}s" if r.mineru_duration_s is not None else "-"
    llm_s = f"{r.llm_duration_s:.1f}s" if r.llm_duration_s is not None else "-"
    msg = (
        f"[{color}]{r.status:8s}[/{color}] id={r.doc_id} "
        f"sha={r.sha256[:12]} mineru={mineru_s} llm={llm_s}"
    )
    if r.error_message:
        msg += f"  [red]err={r.error_message[:80]}[/red]"
    if r.skipped_reason:
        msg += f"  [blue]{r.skipped_reason}[/blue]"
    console.print(msg)


# ---------- list ----------


@app.command("list")
def list_cmd(
    archive: Optional[Path] = _archive_opt,
    limit: int = typer.Option(50, "--limit", "-n"),
    order_by: str = typer.Option(
        "ingested_at",
        "--order-by",
        help="ingested_at | sign_date | expire_date | amount_cents",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status",
        help="过滤状态：ok | partial | failed；默认 None=全部",
    ),
    fmt: str = typer.Option("table", "--format", help="table | json"),
) -> None:
    """列出档案库内已索引合同。"""
    paths = _resolve_archive(archive)
    if not paths.db_path.exists():
        console.print(f"[yellow]archive empty: {paths.db_path} not found[/yellow]")
        raise typer.Exit(0)
    conn = open_archive_db(paths.db_path)
    rows = list_documents(conn, limit=limit, order_by=order_by, status=status)
    conn.close()

    if fmt == "json":
        print(_json.dumps([_row_to_dict(r) for r in rows], ensure_ascii=False, indent=2))
        return

    table = Table(title=f"Archive · {paths.root} ({len(rows)} of total)")
    table.add_column("id", style="cyan", justify="right")
    table.add_column("status")
    table.add_column("contract_name", overflow="fold")
    table.add_column("party_a", overflow="fold")
    table.add_column("party_b", overflow="fold")
    table.add_column("amount", justify="right")
    table.add_column("sign_date")
    table.add_column("expire_date")
    table.add_column("conf", justify="right")
    table.add_column("ingested", style="dim")
    for r in rows:
        status_styled = _status_color(r.status)
        amount = f"¥{r.amount_value:,.0f}" if r.amount_value is not None else "-"
        conf = f"{r.overall_confidence:.2f}" if r.overall_confidence is not None else "-"
        table.add_row(
            str(r.id),
            status_styled,
            r.contract_name or "-",
            r.party_a or "-",
            r.party_b or "-",
            amount,
            r.sign_date or "-",
            r.expire_date or "-",
            conf,
            r.ingested_at[:10],
        )
    console.print(table)


def _status_color(s: str) -> str:
    color = {"ok": "green", "partial": "yellow", "failed": "red"}.get(s, "white")
    return f"[{color}]{s}[/{color}]"


def _row_to_dict(r) -> dict:
    return {
        "id": r.id,
        "sha256": r.sha256,
        "status": r.status,
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


# ---------- search ----------


@app.command()
def search(
    archive: Optional[Path] = _archive_opt,
    name: Optional[str] = typer.Option(None, "--name", help="合同名包含（LIKE）"),
    party: Optional[str] = typer.Option(
        None, "--party", help="甲方 OR 乙方包含（LIKE）"
    ),
    amount_min: Optional[float] = typer.Option(
        None, "--amount-min", help="金额下限（元）"
    ),
    amount_max: Optional[float] = typer.Option(
        None, "--amount-max", help="金额上限（元）"
    ),
    signed_after: Optional[str] = typer.Option(
        None, "--signed-after", help="签订日 ≥ YYYY-MM-DD"
    ),
    signed_before: Optional[str] = typer.Option(
        None, "--signed-before", help="签订日 ≤ YYYY-MM-DD"
    ),
    expire_before: Optional[str] = typer.Option(
        None, "--expire-before", help="到期日 ≤ YYYY-MM-DD（找快到期）"
    ),
    auto_renewal: Optional[bool] = typer.Option(
        None,
        "--auto-renewal/--no-auto-renewal",
        help="是否自动续约",
    ),
    has_risk: bool = typer.Option(False, "--has-risk", help="只显示有风险条款的"),
    deadline_before: Optional[str] = typer.Option(
        None,
        "--deadline-before",
        help="存在 deadline ≤ YYYY-MM-DD 的义务（找近期待办合同）",
    ),
    deadline_after: Optional[str] = typer.Option(
        None, "--deadline-after", help="存在 deadline ≥ YYYY-MM-DD 的义务"
    ),
    actor: Optional[str] = typer.Option(
        None, "--actor",
        help="义务 actor: party_a | party_b | both",
    ),
    status: Optional[str] = typer.Option(None, "--status"),
    limit: int = typer.Option(50, "--limit", "-n"),
    fmt: str = typer.Option("table", "--format", help="table | json"),
) -> None:
    """多字段 AND 过滤查询。"""
    paths = _resolve_archive(archive)
    if not paths.db_path.exists():
        console.print(f"[yellow]archive empty: {paths.db_path}[/yellow]")
        raise typer.Exit(0)
    conn = open_archive_db(paths.db_path)
    flt = SearchFilter(
        name=name,
        party=party,
        amount_min_cents=int(round(amount_min * 100)) if amount_min is not None else None,
        amount_max_cents=int(round(amount_max * 100)) if amount_max is not None else None,
        signed_after=signed_after,
        signed_before=signed_before,
        expire_before=expire_before,
        auto_renewal=auto_renewal,
        has_risk=has_risk,
        deadline_before=deadline_before,
        deadline_after=deadline_after,
        actor=actor,
        status=status,
        limit=limit,
    )
    rows = search_documents(conn, flt)
    conn.close()

    if fmt == "json":
        print(_json.dumps([_row_to_dict(r) for r in rows], ensure_ascii=False, indent=2))
        return

    table = Table(title=f"Search · {len(rows)} hit(s)")
    table.add_column("id", style="cyan", justify="right")
    table.add_column("name", overflow="fold")
    table.add_column("party_a", overflow="fold")
    table.add_column("party_b", overflow="fold")
    table.add_column("amount", justify="right")
    table.add_column("sign_date")
    table.add_column("expire_date")
    table.add_column("risks", justify="right")
    for r in rows:
        amount = f"¥{r.amount_value:,.0f}" if r.amount_value is not None else "-"
        table.add_row(
            str(r.id),
            r.contract_name or "-",
            r.party_a or "-",
            r.party_b or "-",
            amount,
            r.sign_date or "-",
            r.expire_date or "-",
            str(len(r.risk_clauses)),
        )
    console.print(table)


# ---------- show ----------


@app.command()
def show(
    ident: str = typer.Argument(..., help="档案 id (整数) 或 sha 前缀 (>=4 字符)"),
    archive: Optional[Path] = _archive_opt,
    fmt: str = typer.Option("table", "--format", help="table | json"),
) -> None:
    """显示单条档案详情。"""
    paths = _resolve_archive(archive)
    if not paths.db_path.exists():
        console.print(f"[yellow]archive empty: {paths.db_path}[/yellow]")
        raise typer.Exit(1)
    conn = open_archive_db(paths.db_path)
    row = _resolve_ident(conn, ident)
    conn.close()

    if not row:
        console.print(f"[red]not found: {ident}[/red]")
        raise typer.Exit(1)

    if fmt == "json":
        print(_json.dumps(_row_to_dict(row), ensure_ascii=False, indent=2))
        return

    table = Table(title=f"Document #{row.id} ({_status_color(row.status)})")
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    table.add_row("sha256", row.sha256)
    table.add_row("source_path", row.source_path)
    table.add_row("output_dir", row.output_dir)
    table.add_row("ingested_at", row.ingested_at)
    table.add_row(
        "mineru_s", f"{row.mineru_duration_s:.2f}" if row.mineru_duration_s else "-"
    )
    table.add_row("llm_s", f"{row.llm_duration_s:.2f}" if row.llm_duration_s else "-")
    if row.error_message:
        table.add_row("[red]error[/red]", row.error_message)
    table.add_row("", "")
    table.add_row("[bold]contract_name[/bold]", row.contract_name or "-")
    table.add_row("party_a", row.party_a or "-")
    table.add_row("party_b", row.party_b or "-")
    table.add_row(
        "amount",
        f"{row.amount_text or '-'} (¥{row.amount_value:,.2f})"
        if row.amount_value is not None
        else (row.amount_text or "-"),
    )
    table.add_row("sign_date", row.sign_date or "-")
    table.add_row("expire_date", row.expire_date or "-")
    table.add_row(
        "auto_renewal",
        "是" if row.auto_renewal == 1 else ("否" if row.auto_renewal == 0 else "-"),
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
    console.print(table)


def _resolve_ident(conn, ident: str):
    """
    show/extract/delete 共用：ident 可能是 id 或 sha 前缀。
    消歧规则：
      - 全数字且 <= 18 位 → 先按 id 查；查不到再按 sha 前缀
      - 含非数字字符 → 按 sha 前缀（必须 >=4 字符）
      - sha 前缀多匹配 → 报错列候选
    """
    if ident.isdigit() and len(ident) <= 18:
        try:
            doc_id = int(ident)
            row = get_document(conn, doc_id)
            if row:
                return row
        except ValueError:
            pass
        # 数字也可能是 sha 前缀（罕见但合法），fallthrough
    if len(ident) < 4:
        console.print(
            f"[red]ident {ident!r} 不是有效 id；如要按 sha 前缀查询请提供 ≥4 字符[/red]"
        )
        return None
    matches = find_by_sha_prefix(conn, ident.lower())
    if not matches:
        return None
    if len(matches) > 1:
        console.print(f"[red]sha prefix {ident!r} 命中 {len(matches)} 条，请提供更长前缀：[/red]")
        for m in matches[:10]:
            console.print(
                f"  id={m.id} sha={m.short_sha} name={m.contract_name or '-'}"
            )
        return None
    return matches[0]


# ---------- extract ----------


@app.command()
def extract(
    ident: str = typer.Argument(..., help="档案 id 或 sha 前缀"),
    archive: Optional[Path] = _archive_opt,
    no_llm: bool = typer.Option(False, "--no-llm", help="只跑 rule，跳过 LLM"),
) -> None:
    """
    只重跑合同字段抽取（不重跑 MinerU）。用于：
      - partial 状态修复
      - 改 prompt / rule 后批量再抽取
    """
    paths = _resolve_archive(archive)
    conn = open_archive_db(paths.db_path)
    row = _resolve_ident(conn, ident)
    if not row:
        console.print(f"[red]not found: {ident}[/red]")
        conn.close()
        raise typer.Exit(1)

    console.print(f"[cyan]re-extracting id={row.id} sha={row.short_sha}[/cyan]")
    result = re_extract(row.id, paths, conn, llm_enabled=not no_llm)
    checkpoint(conn)
    conn.close()
    _print_ingest_result(result)


# ---------- stats ----------


@app.command()
def stats(archive: Optional[Path] = _archive_opt) -> None:
    """档案库统计：总数 / status 分布 / 按月签订分布 / 近 30 天到期数。"""
    paths = _resolve_archive(archive)
    if not paths.db_path.exists():
        console.print(f"[yellow]archive empty: {paths.db_path}[/yellow]")
        raise typer.Exit(0)
    conn = open_archive_db(paths.db_path)
    s = collect_stats(conn)
    conn.close()

    table = Table(title=f"Archive Stats · {paths.root}")
    table.add_column("metric", style="cyan")
    table.add_column("value")
    table.add_row("total", str(s.total))
    table.add_row(
        "by_status",
        ", ".join(f"{k}={v}" for k, v in sorted(s.by_status.items())) or "-",
    )
    table.add_row("new_this_month", str(s.new_this_month))
    table.add_row("expiring_within_30d", str(s.expiring_within_30d))
    table.add_row(
        "by_sign_month",
        "\n".join(f"{m}: {c}" for m, c in s.by_sign_month.items()) or "-",
    )
    console.print(table)


# ---------- delete ----------


@app.command()
def delete(
    ident: str = typer.Argument(..., help="档案 id 或 sha 前缀"),
    archive: Optional[Path] = _archive_opt,
    purge_files: bool = typer.Option(
        False,
        "--purge-files",
        help="同时删除 archive/documents/<sha-short>/（默认只删 DB 行）",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认提示"),
) -> None:
    """
    删除单条档案。默认仅删 DB 记录，保留文件；--purge-files 才删 archive 内的产物。
    源 PDF（用户原文件）任何情况下都不会被删除。
    """
    paths = _resolve_archive(archive)
    conn = open_archive_db(paths.db_path)
    row = _resolve_ident(conn, ident)
    if not row:
        console.print(f"[red]not found: {ident}[/red]")
        conn.close()
        raise typer.Exit(1)

    console.print(
        f"about to delete: id={row.id} sha={row.short_sha} name={row.contract_name or '-'}"
    )
    console.print(f"  source PDF: {row.source_path} [dim](不会被删除)[/dim]")
    console.print(
        f"  archive dir: {row.output_dir} "
        + ("[red](会被删除)[/red]" if purge_files else "[dim](保留)[/dim]")
    )

    if not yes:
        confirm = typer.confirm("继续？", default=False)
        if not confirm:
            console.print("[yellow]aborted[/yellow]")
            conn.close()
            raise typer.Exit(0)

    output_dir = delete_document(conn, row.id)
    checkpoint(conn)
    conn.close()

    if purge_files and output_dir:
        from shutil import rmtree

        out = Path(output_dir)
        if out.exists():
            rmtree(out)
            console.print(f"[green]✓ removed {out}[/green]")
    console.print(f"[green]✓ deleted DB row id={row.id}[/green]")


# ---------- todo ----------


@app.command()
def todo(
    archive: Optional[Path] = _archive_opt,
    actor: Optional[str] = typer.Option(
        None, "--actor", help="party_a | party_b | both"
    ),
    before: Optional[str] = typer.Option(
        None, "--before", help="deadline ≤ YYYY-MM-DD"
    ),
    after: Optional[str] = typer.Option(
        None, "--after", help="deadline ≥ YYYY-MM-DD"
    ),
    include_undated: bool = typer.Option(
        False, "--include-undated", help="同时显示无 deadline 的义务"
    ),
    within_days: Optional[int] = typer.Option(
        None,
        "--within-days",
        help="便捷选项：deadline 在今天到 N 天内（等价于 --after today --before today+N）",
    ),
    limit: int = typer.Option(50, "--limit", "-n"),
    fmt: str = typer.Option("table", "--format", help="table | json"),
) -> None:
    """
    跨合同列出待办义务（"催办看板"）。按 deadline 升序。

    用例：
      ocr-cli todo --within-days 30           本月需要做的事
      ocr-cli todo --actor party_b            乙方所有待办
      ocr-cli todo --actor party_a --before 2026-12-31
      ocr-cli todo --include-undated          含无日期的（如"签订当日支付定金"）
    """
    from datetime import date, timedelta

    if within_days is not None:
        today = date.today().isoformat()
        before = before or (date.today() + timedelta(days=within_days)).isoformat()
        after = after or today

    paths = _resolve_archive(archive)
    if not paths.db_path.exists():
        console.print(f"[yellow]archive empty: {paths.db_path}[/yellow]")
        raise typer.Exit(0)
    conn = open_archive_db(paths.db_path)
    items = list_obligations(
        conn,
        actor=actor,
        before=before,
        after=after,
        include_undated=include_undated,
        limit=limit,
    )
    conn.close()

    if fmt == "json":
        print(
            _json.dumps(
                [
                    {
                        "doc_id": it.doc_id,
                        "contract_name": it.contract_name,
                        "actor": it.actor,
                        "action": it.action,
                        "deadline": it.deadline,
                        "evidence": it.evidence,
                    }
                    for it in items
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    table = Table(title=f"Todo · {len(items)} obligation(s)")
    table.add_column("deadline", style="cyan")
    table.add_column("actor")
    table.add_column("action", overflow="fold")
    table.add_column("contract", overflow="fold", style="dim")
    table.add_column("doc", justify="right", style="dim")
    actor_label = {"party_a": "甲方", "party_b": "乙方", "both": "双方"}
    for it in items:
        deadline = it.deadline or "[dim]无日期[/dim]"
        table.add_row(
            deadline,
            actor_label.get(it.actor, it.actor),
            it.action,
            it.contract_name or "-",
            f"#{it.doc_id}",
        )
    console.print(table)


# ---------- vacuum ----------


@app.command()
def vacuum(archive: Optional[Path] = _archive_opt) -> None:
    """VACUUM 数据库（碎片整理，建议大批量 ingest 后跑一次）。"""
    paths = _resolve_archive(archive)
    if not paths.db_path.exists():
        console.print(f"[yellow]archive empty: {paths.db_path}[/yellow]")
        raise typer.Exit(0)
    conn = open_archive_db(paths.db_path)
    console.print("[cyan]running VACUUM ANALYZE...[/cyan]")
    conn.execute("VACUUM")
    conn.execute("ANALYZE")
    checkpoint(conn)
    conn.close()
    console.print("[green]✓ done[/green]")


if __name__ == "__main__":
    app()
