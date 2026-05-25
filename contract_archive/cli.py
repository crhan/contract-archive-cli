"""
本地合同档案库 CLI。

子命令：
  ingest <path>         扫描 PDF 文件/目录，跑 MinerU + 抽取，结果入库
  list                  列出档案；status 颜色标注，支持排序；--incomplete 只列疑似不完整合同
  search                按字段过滤（合同名/甲乙方/金额/日期/自动续约/风险）
  show <ident>          查看单条详情（id 或 sha 前缀 >=4 字符）
  extract <id>          只重跑抽取（不重跑 MinerU），适合 partial 修复 / 改 prompt 后重抽
  stats                 总数 / status 分布 / 按月签订分布 / 近 30 天到期数
  seals                 跨文档列印章（某主体有哪些章、各在哪些文档）
  delete <id>           删除档案记录；默认仅删 DB 行，--purge-files 同时删文件
  vacuum                VACUUM 数据库（碎片整理）
  config                查看/设置全局配置（XDG ~/.config/contract-archive/config.json）

档案库路径优先级：--archive flag > CONTRACT_ARCHIVE_DIR env > config archive.dir > XDG 默认 (~/.local/share/contract-archive)
"""
from __future__ import annotations

import json as _json
import logging
import os
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import __version__
from .errors import classify_exception
from .archive import (
    ArchivePaths,
    SearchFilter,
    checkpoint,
    default_archive_root,
    collect_stats,
    delete_document,
    discover_pdfs,
    find_by_sha_prefix,
    get_document,
    ingest_pdf,
    list_documents,
    list_obligations,
    list_seals,
    open_archive_db,
    re_extract,
    search_documents,
    Stats,
)
from .archive.paths import sha256_of_file
from .archive.repository import find_by_sha
from .pipelines import MinerUPipeline
from .config import load_settings
from .cli_config import config_app
from .cli_introspect import register as register_introspect
from .cli_render import (
    build_show_table,
    completeness_mark,
    display_amount,
    ingest_result_to_dict,
    local_time,
    row_to_dict,
    seal_rows_to_dict,
    status_color,
    subject_of,
)

# ---------- 参数枚举（parse-time 校验：坏值由 typer 报 exit 2，不再漏到数据层 ValueError）----------


class OutputFormat(str, Enum):
    """--format：人类表格 or 机器 JSON。"""

    table = "table"
    json = "json"


class ProgressFormat(str, Enum):
    """--progress：none=现状（汇总在末尾）；ndjson=逐文件向 stdout 吐事件流，供 agent 流式消费。"""

    none = "none"
    ndjson = "ndjson"


class OrderBy(str, Enum):
    """list --order-by。成员必须与 repository.list_documents 的 allowed_order 白名单一致。"""

    ingested_at = "ingested_at"
    primary_date = "primary_date"
    primary_amount_cents = "primary_amount_cents"
    sign_date = "sign_date"
    expire_date = "expire_date"
    amount_cents = "amount_cents"


class DocStatus(str, Enum):
    """--status：入库状态。"""

    ok = "ok"
    partial = "partial"
    failed = "failed"


class DocType(str, Enum):
    """list --type：文档类型。值即 CLI choice，与抽取信封的类型枚举一致。"""

    contract = "合同协议"
    proof = "证明"
    invoice = "发票票据"
    report = "报告"
    certificate = "证件"
    other = "其他"


class Actor(str, Enum):
    """--actor：义务主体。成员必须与 repository 的 party_a/party_b/both 校验一致。"""

    party_a = "party_a"
    party_b = "party_b"
    both = "both"


# ---------- 双 console：数据走 stdout（可管道），诊断/进度/错误走 stderr ----------

console = Console()                    # 主数据：表格 + JSON
err_console = Console(stderr=True)      # 人类消息：状态/进度/错误/确认


def _version_cb(value: bool) -> None:
    """--version 的 eager 回调：版本号打到 stdout（机器可消费），随即退出。"""
    if value:
        print(f"contract-archive {__version__}")
        raise typer.Exit()


app = typer.Typer(
    help="本地合同档案库 CLI (MinerU + qwen3.7-max)",
    context_settings={"help_option_names": ["-h", "--help"]},
    # 默认 typer 会在 traceback 里 dump 局部变量，可能带出敏感内容，关掉。
    pretty_exceptions_show_locals=False,
    epilog=(
        "示例：\n"
        "  contract-archive ingest ./input            # 扫描目录入库\n"
        "  contract-archive list --format json | jq   # 机器可读，管道友好\n"
        "  contract-archive todo --within-days 30      # 近 30 天待办义务\n"
        "\n文档：https://github.com/crhan/contract-archive-cli"
    ),
)
app.add_typer(config_app, name="config")
# introspection 命令（capabilities/describe/schema）：给机器发现能力用，见 cli_introspect。
register_introspect(app)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        is_eager=True,
        callback=_version_cb,
        help="打印版本并退出",
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="禁用彩色输出（管道/日志归档时用）"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="DEBUG 级日志（更啰嗦，排查用）"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="仅 WARNING 及以上（更安静）"
    ),
) -> None:
    """
    全局选项在所有子命令前生效。flag 优先级高于环境变量：
      --no-color 覆盖 NO_COLOR/TTY 自动探测；--verbose/--quiet 覆盖 LOG_LEVEL。
    """
    # dotenv 放到这里加载——保证 flag 解析后再读 env，且 CONTRACT_ARCHIVE_DIR 等及时可用。
    # override=False 显式声明：shell 已 export 的变量压过 .env（即 env > 项目 .env），
    # 与 config 层 env>file>default 的优先级语义一致。
    load_dotenv(override=False)

    if no_color:
        console.no_color = True
        err_console.no_color = True

    # 日志默认 stderr；--verbose/--quiet 胜过 LOG_LEVEL env。
    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    else:
        level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ---------- 全局 archive 路径解析 ----------


def _resolve_archive(archive_opt: Optional[Path]) -> ArchivePaths:
    """
    --archive flag > CONTRACT_ARCHIVE_DIR env > config archive.dir > XDG 默认。

    env 与 config 的合并交给 load_settings()（其 archive_dir 已是 env>config 短路结果，
    env 严格优先、空串当未设），这里只在 flag 之后接住它，再回退 XDG 默认。
    统一 expanduser：修掉历史上 CONTRACT_ARCHIVE_DIR=~/x 不展开 ~ 的坑。
    """
    if archive_opt:
        root = archive_opt
    else:
        configured = load_settings().archive_dir
        root = Path(configured) if configured else default_archive_root()
    return ArchivePaths(root=root.expanduser().resolve())


_archive_opt = typer.Option(
    None,
    "--archive",
    "-a",
    help="档案库根目录；不传则用 CONTRACT_ARCHIVE_DIR 或 XDG 默认 ~/.local/share/contract-archive",
)


def _archive_empty(paths: ArchivePaths, fmt: OutputFormat) -> bool:
    """
    读命令统一空库守卫。返回 True 表示库不存在、调用方应直接 return。
      - json 模式：往 stdout 打 `[]`，保证管道消费者（jq）拿到合法 JSON
      - table 模式：往 stderr 打人类提示，不污染 stdout
    """
    if paths.db_path.exists():
        return False
    if fmt is OutputFormat.json:
        print("[]")
    else:
        err_console.print(f"[yellow]archive empty: {paths.db_path} not found[/yellow]")
    return True


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
    fmt: OutputFormat = typer.Option(
        OutputFormat.table, "--format", help="table | json（json 把汇总+逐条结果打到 stdout）"
    ),
    progress: ProgressFormat = typer.Option(
        ProgressFormat.none, "--progress",
        help="none | ndjson（ndjson 逐文件向 stdout 吐 JSON 事件流，供 agent 流式消费）",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="只预览将处理哪些文件 + 预计 API 调用，不跑 MinerU/不调 LLM/不写库",
    ),
    max_files: int = typer.Option(
        0, "--max-files",
        help="最多处理 N 个文件，超过则报错退出（0=不限；防误喂大目录烧钱，agent 应主动设）",
    ),
) -> None:
    """跑 MinerU + 抽取，把合同入库。"""
    paths = _resolve_archive(archive)

    pdfs = discover_pdfs(path)
    if limit > 0:
        pdfs = pdfs[:limit]

    # --dry-run：预览不建库/不跑 MinerU/不调 LLM，提前返回（预览不受 --max-files 限制）。
    if dry_run:
        _ingest_dry_run(pdfs, paths, fmt)
        return

    # --max-files 护栏：超上限直接报错退出，防 agent 误喂大目录烧钱（0=不限，保持兼容）。
    if max_files > 0 and len(pdfs) > max_files:
        err_console.print(
            f"[red]发现 {len(pdfs)} 个 PDF，超过 --max-files {max_files}；"
            f"确需处理请调大 --max-files[/red]"
        )
        raise typer.Exit(2)

    paths.ensure()
    conn = open_archive_db(paths.db_path)

    summary = {"ok": 0, "partial": 0, "failed": 0, "skipped": 0}
    if not pdfs:
        # 进度/提示走 stderr；json 模式仍向 stdout 吐合法 JSON，便于管道消费。
        err_console.print("[yellow]no PDFs found[/yellow]")
        if fmt is OutputFormat.json:
            print(_json.dumps(
                {"archive": str(paths.root), "summary": summary, "results": []},
                ensure_ascii=False, indent=2,
            ))
        raise typer.Exit(0)

    err_console.print(f"[cyan]found {len(pdfs)} PDF(s); archive={paths.root}[/cyan]")
    # 复用一个 MinerUPipeline 实例（避免每次重新加载模型）
    pipeline = MinerUPipeline()

    results: list[dict] = []
    for i, pdf in enumerate(pdfs, 1):
        err_console.rule(f"[bold cyan][{i}/{len(pdfs)}] {pdf.name}[/bold cyan]")
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
            err_console.print(f"[red]✗ unexpected error: {e}[/red]")
            logging.getLogger(__name__).exception("ingest crashed")
            summary["failed"] += 1
            fail_dict = {
                "pdf_path": str(pdf), "sha256": None, "status": "failed",
                "doc_id": None, "mineru_duration_s": None, "llm_duration_s": None,
                "error_message": str(e),
                "error": classify_exception(e).model_dump(),
                "skipped_reason": None,
            }
            results.append(fail_dict)
            if progress is ProgressFormat.ndjson:
                _emit_progress(i, len(pdfs), fail_dict)
            continue
        summary[result.status] = summary.get(result.status, 0) + 1
        result_dict = ingest_result_to_dict(result)
        results.append(result_dict)
        _print_ingest_result(result)
        if progress is ProgressFormat.ndjson:
            _emit_progress(i, len(pdfs), result_dict)

    checkpoint(conn)
    conn.close()
    err_console.rule("[bold]summary[/bold]")
    err_console.print(
        f"ok={summary['ok']} partial={summary['partial']} "
        f"failed={summary['failed']} skipped={summary['skipped']}"
    )
    if progress is ProgressFormat.ndjson:
        # 流式模式：末行吐 summary 事件（逐文件事件已在循环里吐过）。
        print(_json.dumps(
            {"event": "summary", "archive": str(paths.root), **summary},
            ensure_ascii=False,
        ))
    elif fmt is OutputFormat.json:
        print(_json.dumps(
            {"archive": str(paths.root), "summary": summary, "results": results},
            ensure_ascii=False, indent=2,
        ))
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
    err_console.print(msg)


def _emit_progress(seq: int, total: int, result_dict: dict) -> None:
    """--progress ndjson：每处理完一个文件，向 stdout 吐一行 file_done 事件（机器流式消费）。"""
    print(_json.dumps(
        {"event": "file_done", "seq": seq, "total": total, **result_dict},
        ensure_ascii=False,
    ))


def _ingest_dry_run(pdfs: list[Path], paths: ArchivePaths, fmt: OutputFormat) -> None:
    """
    预览将处理哪些文件 + 预计 API 调用，不产生任何副作用（不建库/不跑 MinerU/不调 LLM）。

    用已有库做 sha256 去重预览（库不存在则全部视为新增，且不创建库）；成本预估：
    每个新增文件至少 1 次 LLM 文本抽取，合同还会有最多 1 次 VL 签章核查。
    """
    conn = open_archive_db(paths.db_path) if paths.db_path.exists() else None
    files: list[dict] = []
    new_count = 0
    for pdf in pdfs:
        sha = sha256_of_file(pdf)
        existing = find_by_sha(conn, sha) if conn is not None else None
        action = "skip" if existing else "new"
        if action == "new":
            new_count += 1
        files.append({
            "pdf_path": str(pdf), "sha256": sha,
            "action": action, "existing_doc_id": existing,
        })
    if conn is not None:
        conn.close()

    payload = {
        "dry_run": True,
        "archive": str(paths.root),
        "total": len(pdfs),
        "new": new_count,
        "already_ingested": len(pdfs) - new_count,
        "estimated_llm_calls": new_count,
        "estimated_vl_calls_max": new_count,
        "files": files,
    }
    if fmt is OutputFormat.json:
        print(_json.dumps(payload, ensure_ascii=False, indent=2))
        return
    err_console.print(
        f"[cyan]dry-run: 共 {len(pdfs)} 个 PDF，新增 {new_count}，"
        f"已存在 {len(pdfs) - new_count}[/cyan]"
    )
    err_console.print(
        f"预计 LLM 文本抽取 {new_count} 次，VL 签章核查最多 {new_count} 次（仅合同）"
    )
    for f in files:
        mark = "[green]new [/green]" if f["action"] == "new" else "[blue]skip[/blue]"
        err_console.print(f"  {mark} {f['sha256'][:12]} {f['pdf_path']}")


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

    table = Table(
        title=f"Archive · {paths.root} ({len(rows)} of total)",
        caption="amount 带 * 为计算合计（如收入证明=年税前+股权），无 * 为抽取的主金额",
        caption_justify="left",
    )
    table.add_column("id", style="cyan", justify="right")
    table.add_column("status")
    table.add_column("type", style="magenta")
    table.add_column("完整")  # 合同完整性：⚠ 疑似缺 / ✓ / -（非合同）
    table.add_column("title", overflow="fold")
    table.add_column("主体", overflow="fold")  # 区分同类文档（谁的/和谁签的）
    table.add_column("date")
    table.add_column("amount", justify="right")
    table.add_column("ingested", style="dim")
    for r in rows:
        table.add_row(
            str(r.id),
            status_color(r.status),
            r.doc_type or "-",
            completeness_mark(r),
            r.title or r.contract_name or "-",
            subject_of(r),
            r.primary_date or "-",
            display_amount(r),
            local_time(r.ingested_at)[:10],  # 本地日期，与 show 一致
        )
    console.print(table)


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
        err_console.print(
            f"[red]ident {ident!r} 不是有效 id；如要按 sha 前缀查询请提供 ≥4 字符[/red]"
        )
        return None
    matches = find_by_sha_prefix(conn, ident.lower())
    if not matches:
        return None
    if len(matches) > 1:
        err_console.print(f"[red]sha prefix {ident!r} 命中 {len(matches)} 条，请提供更长前缀：[/red]")
        for m in matches[:10]:
            err_console.print(
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
        err_console.print(f"[red]not found: {ident}[/red]")
        conn.close()
        raise typer.Exit(1)

    err_console.print(f"[cyan]re-extracting id={row.id} sha={row.short_sha}[/cyan]")
    result = re_extract(row.id, paths, conn, llm_enabled=not no_llm)
    checkpoint(conn)
    conn.close()
    _print_ingest_result(result)


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
        err_console.print(f"[red]not found: {ident}[/red]")
        conn.close()
        raise typer.Exit(1)

    # 非交互环境（管道/CI）下不能交互确认，typer.confirm 会读到 EOF 崩。
    # clig.dev：危险动作在非 TTY 下应明确要求显式 --yes，而不是糊涂地中止。
    if not yes and not sys.stdin.isatty():
        err_console.print(
            "[red]拒绝在非交互环境删除：请加 --yes 确认[/red]"
        )
        conn.close()
        raise typer.Exit(1)

    err_console.print(
        f"about to delete: id={row.id} sha={row.short_sha} name={row.contract_name or '-'}"
    )
    err_console.print(f"  source PDF: {row.source_path} [dim](不会被删除)[/dim]")
    err_console.print(
        f"  archive dir: {row.output_dir} "
        + ("[red](会被删除)[/red]" if purge_files else "[dim](保留)[/dim]")
    )

    if not yes:
        confirm = typer.confirm("继续？", default=False)
        if not confirm:
            err_console.print("[yellow]aborted[/yellow]")
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
            err_console.print(f"[green]✓ removed {out}[/green]")
    err_console.print(f"[green]✓ deleted DB row id={row.id}[/green]")


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


# ---------- vacuum ----------


@app.command()
def vacuum(archive: Optional[Path] = _archive_opt) -> None:
    """VACUUM 数据库（碎片整理，建议大批量 ingest 后跑一次）。"""
    paths = _resolve_archive(archive)
    if not paths.db_path.exists():
        err_console.print(f"[yellow]archive empty: {paths.db_path}[/yellow]")
        raise typer.Exit(0)
    conn = open_archive_db(paths.db_path)
    err_console.print("[cyan]running VACUUM ANALYZE...[/cyan]")
    conn.execute("VACUUM")
    conn.execute("ANALYZE")
    checkpoint(conn)
    conn.close()
    err_console.print("[green]✓ done[/green]")


if __name__ == "__main__":
    app()
