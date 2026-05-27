"""
只读查询/展示命令：list / search / show / raw / stats / todo / seals。

这些命令不写库、不调用付费 API，是档案库的"读侧"。它们用 @app.command 挂到
cli_common 的主 app 上——import 本模块即触发注册（见 cli.py 的组装段）。

依赖方向：本模块只 import cli_common（基础设施）+ archive 读函数 + cli_render
（纯渲染），绝不回头 import cli（写命令模块），以保持 DAG、避免循环 import。
"""
from __future__ import annotations

import json as _json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .archive import (
    SearchFilter,
    Stats,
    collect_stats,
    list_documents,
    list_obligations,
    list_seals,
    load_document_text,
    open_archive_db,
    search_documents,
)
from .cli_common import (
    Actor,
    ColorWhen,
    DocStatus,
    DocType,
    OrderBy,
    OutputFormat,
    _archive_empty,
    _archive_opt,
    _resolve_archive,
    _resolve_ident,
    app,
    console,
    err_console,
)
from .cli_render import (
    build_list_table,
    build_search_table,
    build_show_table,
    color_legend,
    extracted_terms,
    render_highlighted,
    row_to_dict,
    seal_rows_to_dict,
)

# ---------- list ----------


@app.command("list")
def list_cmd(
    archive: Optional[Path] = _archive_opt,
    limit: int = typer.Option(50, "--limit", "-n"),
    order_by: OrderBy = typer.Option(
        OrderBy.ingested_at, "--order-by", help="排序字段"
    ),
    status: Optional[DocStatus] = typer.Option(
        None, "--status", help="过滤状态；默认全部"
    ),
    doc_type: Optional[DocType] = typer.Option(
        None, "--type", help="按文档类型过滤"
    ),
    incomplete: bool = typer.Option(
        False, "--incomplete", help="只列疑似不完整的合同（缺签章/缺要素）"
    ),
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", help="table | json"),
) -> None:
    """列出档案库内已索引文档。"""
    paths = _resolve_archive(archive)
    if _archive_empty(paths, fmt):
        return
    conn = open_archive_db(paths.db_path)
    rows = list_documents(
        conn,
        limit=limit,
        order_by=order_by.value,
        status=status.value if status else None,
        doc_type=doc_type.value if doc_type else None,
        incomplete=incomplete,
    )
    conn.close()

    if fmt is OutputFormat.json:
        print(_json.dumps([row_to_dict(r) for r in rows], ensure_ascii=False, indent=2))
        return

    console.print(build_list_table(rows, paths.root))


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
    actor: Optional[Actor] = typer.Option(
        None, "--actor", help="义务 actor"
    ),
    status: Optional[DocStatus] = typer.Option(None, "--status", help="过滤状态"),
    has_seal: Optional[bool] = typer.Option(
        None, "--has-seal/--no-seal", help="有/无印章（默认不过滤）"
    ),
    seal_owner: Optional[str] = typer.Option(
        None, "--seal-owner", help="盖章主体包含（LIKE）"
    ),
    seal_type: Optional[str] = typer.Option(
        None, "--seal-type", help="印章类型包含（LIKE），如 合同专用章/公章"
    ),
    subject: Optional[str] = typer.Option(
        None, "--subject", help="主体包含（LIKE），覆盖所有文档类型（含合同甲乙方）"
    ),
    limit: int = typer.Option(50, "--limit", "-n"),
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", help="table | json"),
) -> None:
    """多字段 AND 过滤查询。"""
    paths = _resolve_archive(archive)
    if _archive_empty(paths, fmt):
        return
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
        actor=actor.value if actor else None,
        status=status.value if status else None,
        has_seal=has_seal,
        seal_owner=seal_owner,
        seal_type=seal_type,
        subject=subject,
        limit=limit,
    )
    rows = search_documents(conn, flt)
    conn.close()

    if fmt is OutputFormat.json:
        print(_json.dumps([row_to_dict(r) for r in rows], ensure_ascii=False, indent=2))
        return

    console.print(build_search_table(rows))


# ---------- show ----------


@app.command()
def show(
    ident: str = typer.Argument(..., help="档案 id (整数) 或 sha 前缀 (>=4 字符)"),
    archive: Optional[Path] = _archive_opt,
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", help="table | json"),
) -> None:
    """显示单条档案详情。"""
    paths = _resolve_archive(archive)
    # show 请求的是具体一条；库不存在/查不到都是错误（exit 1），提示走 stderr。
    if not paths.db_path.exists():
        err_console.print(f"[yellow]archive empty: {paths.db_path}[/yellow]")
        raise typer.Exit(1)
    conn = open_archive_db(paths.db_path)
    row = _resolve_ident(conn, ident)
    conn.close()

    if not row:
        err_console.print(f"[red]not found: {ident}[/red]")
        raise typer.Exit(1)

    if fmt is OutputFormat.json:
        print(_json.dumps(row_to_dict(row), ensure_ascii=False, indent=2))
        return

    console.print(build_show_table(row))


# ---------- raw ----------


@app.command()
def raw(
    ident: str = typer.Argument(..., help="档案 id (整数) 或 sha 前缀 (>=4 字符)"),
    archive: Optional[Path] = _archive_opt,
    color: ColorWhen = typer.Option(
        ColorWhen.auto, "--color",
        help="auto=仅 TTY 上色（管道纯文本）| always（配 less -R）| never",
    ),
) -> None:
    """
    打印文档原文（MinerU OCR 输出的纯文本）到 stdout。

    与 show 互补：show 看 LLM 抽出的结构化字段，raw 看抽取所依据的原始文本——
    这正是抽取时喂给 LLM 的同一份内容（raw_text.txt，缺失则退回 markdown.md），
    用于核对抽取结果是否忠于原文。

    交互终端下默认按抽取来源给命中关键字着色（当事人/金额/日期/风险/字段），
    一眼看出哪些被 LLM 识别到；管道（非 TTY）时输出纯文本，不破坏 raw|grep / raw|less。
    """
    paths = _resolve_archive(archive)
    # 同 show：请求的是具体一条，库不存在/查不到都按错误处理（exit 1），提示走 stderr。
    if not paths.db_path.exists():
        err_console.print(f"[yellow]archive empty: {paths.db_path}[/yellow]")
        raise typer.Exit(1)
    conn = open_archive_db(paths.db_path)
    row = _resolve_ident(conn, ident)
    conn.close()

    if not row:
        err_console.print(f"[red]not found: {ident}[/red]")
        raise typer.Exit(1)

    # output_dir 可能为空串（失败入库的记录），Path("")/"mineru" 会落到不存在目录，
    # load_document_text 返回 ""，统一走下面的"无原文"分支，无需单独判空。
    mineru_dir = Path(row.output_dir) / "mineru"
    text = load_document_text(mineru_dir)
    if not text:
        err_console.print(
            f"[red]no OCR text for id={row.id} sha={row.short_sha}: {mineru_dir}[/red]"
        )
        raise typer.Exit(1)

    # 上色判定：always 强制；auto 仅当 stdout 是 TTY；never 禁用。
    # 管道默认纯文本——保住 raw|grep / raw|less 的既有行为（不破坏 userspace）。
    use_color = color is ColorWhen.always or (
        color is ColorWhen.auto and sys.stdout.isatty()
    )
    if not use_color:
        print(text)
        return

    terms = extracted_terms(row)
    # 图例走 stderr：解释颜色含义，又不污染 stdout 的原文（even with | less -R）。
    legend = color_legend(terms)
    if legend and sys.stderr.isatty():
        print(legend, file=sys.stderr)
    sys.stdout.write(render_highlighted(text, terms))
    if not text.endswith("\n"):
        sys.stdout.write("\n")


# ---------- stats ----------


@app.command()
def stats(
    archive: Optional[Path] = _archive_opt,
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", help="table | json"),
) -> None:
    """档案库统计：总数 / status 分布 / 按月签订分布 / 近 30 天到期数。"""
    paths = _resolve_archive(archive)
    # 库不存在 = 零文档档案：合成零值 Stats，走同一条渲染路径，
    # 不为"空库"单开分支——json 形状始终是对象（不会退化成 list 的 []）。
    if paths.db_path.exists():
        conn = open_archive_db(paths.db_path)
        s = collect_stats(conn)
        conn.close()
    else:
        s = Stats(
            total=0, by_status={}, by_sign_month={},
            new_this_month=0, expiring_within_30d=0,
        )

    if fmt is OutputFormat.json:
        print(_json.dumps(asdict(s), ensure_ascii=False, indent=2))
        return

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


# ---------- todo ----------


@app.command()
def todo(
    archive: Optional[Path] = _archive_opt,
    actor: Optional[Actor] = typer.Option(
        None, "--actor", help="义务 actor"
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
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", help="table | json"),
) -> None:
    """
    跨合同列出待办义务（"催办看板"）。按 deadline 升序。

    用例：
      contract-archive todo --within-days 30           本月需要做的事
      contract-archive todo --actor party_b            乙方所有待办
      contract-archive todo --actor party_a --before 2026-12-31
      contract-archive todo --include-undated          含无日期的（如"签订当日支付定金"）
    """
    from datetime import date, timedelta

    if within_days is not None:
        today = date.today().isoformat()
        before = before or (date.today() + timedelta(days=within_days)).isoformat()
        after = after or today

    paths = _resolve_archive(archive)
    if _archive_empty(paths, fmt):
        return
    conn = open_archive_db(paths.db_path)
    items = list_obligations(
        conn,
        actor=actor.value if actor else None,
        before=before,
        after=after,
        include_undated=include_undated,
        limit=limit,
    )
    conn.close()

    if fmt is OutputFormat.json:
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


# ---------- seals ----------


@app.command("seals")
def seals_cmd(
    archive: Optional[Path] = _archive_opt,
    owner: Optional[str] = typer.Option(None, "--owner", help="盖章主体包含（LIKE）"),
    seal_type: Optional[str] = typer.Option(
        None, "--type", help="印章类型包含（LIKE），如 合同专用章/公章"
    ),
    limit: int = typer.Option(200, "--limit", "-n"),
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", help="table | json"),
) -> None:
    """
    跨文档列印章：某主体有哪些章、各出现在哪些文档（按主体/类型聚合阅读）。

    用例：
      contract-archive seals                  全部印章
      contract-archive seals --owner 示例公司   某公司的章
      contract-archive seals --type 合同专用章
    """
    paths = _resolve_archive(archive)
    if _archive_empty(paths, fmt):
        return
    conn = open_archive_db(paths.db_path)
    rows = list_seals(conn, owner=owner, seal_type=seal_type, limit=limit)
    conn.close()

    if fmt is OutputFormat.json:
        print(_json.dumps(seal_rows_to_dict(rows), ensure_ascii=False, indent=2))
        return

    table = Table(title=f"Seals · {len(rows)} 枚")
    table.add_column("owner", overflow="fold", style="magenta")
    table.add_column("type")
    table.add_column("raw_text", overflow="fold", style="dim")
    table.add_column("doc", overflow="fold")
    table.add_column("id", justify="right", style="dim")
    for r in rows:
        table.add_row(
            r.owner or "?",
            r.seal_type or "-",
            r.raw_text,
            r.title or "-",
            f"#{r.doc_id}",
        )
    console.print(table)
