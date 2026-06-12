"""
本地合同档案库 CLI —— 入口与组装模块。

子命令：
  ingest <path>         扫描 PDF 文件/目录，跑 OCR + 抽取，结果入库
  list                  列出档案；status 颜色标注，支持排序；--incomplete 只列疑似不完整合同
  search                按字段过滤（合同名/甲乙方/金额/日期/自动续约/风险）
  show <ident>          查看单条详情（id 或 sha 前缀 >=4 字符）
  raw <ident>           打印文档原文（OCR 文本），与 show 互补，可管道给 grep/less
  extract <id>          只重跑抽取（不重跑 OCR），适合 partial 修复 / 改 prompt 后重抽
  stats                 总数 / status 分布 / 按月签订分布 / 近 30 天到期数
  seals                 跨文档列印章（某主体有哪些章、各在哪些文档）
  delete <id>           删除档案记录；默认仅删 DB 行，--purge-files 同时删文件
  vacuum                VACUUM 数据库（碎片整理）
  config                查看/设置全局配置（XDG ~/.config/contract-archive/config.json）

档案库路径优先级：--archive flag > CONTRACT_ARCHIVE_DIR env > config archive.dir > XDG 默认 (~/.local/share/contract-archive)

代码组织（命令空间扁平，全挂在同一个 app 上）：
  cli_common.py   app 实例 + 全局 callback + 参数 Enum + 双 console + 路径/ident 解析（叶子）
  cli_query.py    只读命令：list/search/show/raw/stats/todo/seals
  cli.py（本文件）写命令：ingest/extract/delete/vacuum + 组装 sub-app 与 introspection

ingest 留在本模块（而非 cli_query）有硬约束：测试用 monkeypatch.setattr(cli, "MinerUPipeline"/
"ingest_pdf") 打桩，命令体必须引用本模块全局名才能让桩生效——别图整齐把它挪走。
"""
from __future__ import annotations

import json as _json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from .errors import classify_exception
from .archive import (
    ArchivePaths,
    checkpoint,
    delete_document,
    discover_pdfs,
    ingest_pdf,
    open_archive_db,
    re_extract,
)
from .archive.paths import sha256_of_file
from .archive.repository import find_by_sha
from .pipelines import MinerUPipeline
from .cli_config import config_app
from .cli_introspect import register as register_introspect
from .cli_party import party_app
from .cli_render import ingest_result_to_dict
from .cli_common import (
    OutputFormat,
    ProgressFormat,
    _archive_opt,
    _resolve_archive,
    _resolve_ident,
    app,
    console,  # noqa: F401  —— re-export：历史上有调用方/测试经 cli.console 访问
    err_console,
    not_found_json,
)

# ---------- ingest ----------


@app.command()
def ingest(
    path: Path = typer.Argument(
        ..., exists=True, readable=True, help="PDF 文件或目录（目录会递归扫 *.pdf）"
    ),
    archive: Optional[Path] = _archive_opt,
    reingest: bool = typer.Option(
        False, "--reingest", help="忽略 sha256 去重，强制重跑 OCR + 抽取"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm",
        help="跳过 LLM 抽取（无 API key 时也用）：仅入库 OCR 产物，抽取字段留空，可后续 extract 补抽",
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
        help="只预览将处理哪些文件 + 预计 API 调用，不跑 OCR/不调 LLM/不写库",
    ),
    max_files: int = typer.Option(
        0, "--max-files",
        help="最多处理 N 个文件，超过则报错退出（0=不限；防误喂大目录烧钱，agent 应主动设）",
    ),
) -> None:
    """跑 OCR + 抽取，把合同入库。"""
    paths = _resolve_archive(archive)

    pdfs = discover_pdfs(path)
    if limit > 0:
        pdfs = pdfs[:limit]

    # --dry-run：预览不建库/不跑 OCR/不调 LLM，提前返回（预览不受 --max-files 限制）。
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
        # 进度/提示走 stderr；json/ndjson 模式仍向 stdout 吐合法 JSON，便于管道消费。
        err_console.print("[yellow]no PDFs found[/yellow]")
        if progress is ProgressFormat.ndjson:
            # 流式消费方即便空输入也应收到终止 summary 事件（与非空路径一致）。
            print(_json.dumps(
                {"event": "summary", "archive": str(paths.root), **summary},
                ensure_ascii=False,
            ))
        elif fmt is OutputFormat.json:
            print(_json.dumps(
                {"archive": str(paths.root), "summary": summary, "results": []},
                ensure_ascii=False, indent=2,
            ))
        conn.close()
        raise typer.Exit(0)

    err_console.print(f"[cyan]found {len(pdfs)} PDF(s); archive={paths.root}[/cyan]")
    # 复用一个 OCR pipeline 实例（避免每次重新加载模型）
    pipeline = MinerUPipeline(allow_vl_fallback=not no_llm)

    results: list[dict] = []
    try:
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
    except KeyboardInterrupt:
        # Ctrl-C：先让 finally 跑 checkpoint+close，再把中断抛出去（退出码 130）。
        err_console.print(
            f"\n[yellow]中断：已处理 {len(results)}/{len(pdfs)} 个，checkpoint 后退出[/yellow]"
        )
        raise
    finally:
        # 正常结束 / 循环内异常 / Ctrl-C 三条退出路径都无条件 checkpoint+close：
        # per-file tmp→rename 已保证数据一致，这里兜 WAL 合并回主库 + 连接关闭的整洁性。
        try:
            checkpoint(conn)
            conn.close()
        except Exception:  # noqa: BLE001 — 清理失败不能掩盖原始异常/中断
            logging.getLogger(__name__).debug("ingest 清理失败", exc_info=True)

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
    ocr_s = f"{r.mineru_duration_s:.1f}s" if r.mineru_duration_s is not None else "-"
    llm_s = f"{r.llm_duration_s:.1f}s" if r.llm_duration_s is not None else "-"
    msg = (
        f"[{color}]{r.status:8s}[/{color}] id={r.doc_id} "
        f"sha={r.sha256[:12]} ocr={ocr_s} llm={llm_s}"
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
    预览将处理哪些文件 + 预计 API 调用，不产生任何副作用（不建库/不跑 OCR/不调 LLM）。

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


# ---------- extract ----------


@app.command()
def extract(
    ident: str = typer.Argument(..., help="档案 id 或 sha 前缀"),
    archive: Optional[Path] = _archive_opt,
    no_llm: bool = typer.Option(
        False, "--no-llm", help="跳过 LLM（抽取字段留空，rule 已退役）"
    ),
    fmt: OutputFormat = typer.Option(
        OutputFormat.table, "--format", help="table | json（json 把结构化结果+error 打到 stdout）"
    ),
) -> None:
    """
    只重跑合同字段抽取（不重跑 OCR）。用于：
      - partial 状态修复
      - 改 prompt 后批量再抽取
    """
    paths = _resolve_archive(archive)
    conn = open_archive_db(paths.db_path)
    row = _resolve_ident(conn, ident)
    if not row:
        # json 模式吐 not_found 信封到 stdout（与 show 一致，别让 | jq 拿空输入）；table 走 stderr。
        if fmt is OutputFormat.json:
            not_found_json(ident)
        else:
            err_console.print(f"[red]not found: {ident}[/red]")
        conn.close()
        raise typer.Exit(1)

    err_console.print(f"[cyan]re-extracting id={row.id} sha={row.short_sha}[/cyan]")
    result = re_extract(row.id, paths, conn, llm_enabled=not no_llm)
    checkpoint(conn)
    conn.close()
    if fmt is OutputFormat.json:
        # 结构化结果（含 re_extract 的 error/retryable）供 agent 消费，与 ingest --format json 一致。
        print(_json.dumps(ingest_result_to_dict(result), ensure_ascii=False, indent=2))
    else:
        _print_ingest_result(result)
    # 抽取失败（空抽取/LLM 异常，re_extract 返回 status=partial + error）必须以非零退出，
    # 否则纯 shell 调用方靠 $? 完全发现不了 extract 失败（此前一律 exit 0 是 bug）。
    if result.error is not None:
        raise typer.Exit(1)


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


# ---------- 组装：挂上只读命令、子 app、introspection ----------
#
# import cli_query 仅为触发其 @app.command 注册（它只依赖 cli_common，不回头 import 本
# 模块，无循环）。放写命令之后，让 --help 里 ingest 等写命令仍排在前、贴近历史顺序。
from . import cli_query  # noqa: E402,F401

app.add_typer(config_app, name="config")
app.add_typer(party_app, name="party")
# introspection 命令（capabilities/describe/schema）：给机器发现能力用，见 cli_introspect。
register_introspect(app)


def main_entry() -> None:
    """
    console_scripts 入口：在 app() 外包一层顶层异常钩子。

    受控退出（命令里的 typer.Exit/Abort 在 click standalone 模式下已转成 SystemExit，
    是 BaseException、不被下面 except Exception 接住）原样放行；未预期异常（如底层
    sqlite OperationalError）翻成一行人话错误走 stderr（不污染 stdout 管道），默认不打
    完整 traceback——加 -v/--verbose 才用 rich 展开（show_locals=False 防 secret 泄露）。

    此前入口直挂裸 app()、typer pretty 异常开着，底层异常直接 dump 一坨 rich traceback，
    用户/agent 看到的是实现细节而非可操作信息。配合 app 的 pretty_exceptions_enable=False。
    """
    try:
        app()
    except Exception as exc:  # noqa: BLE001 — 顶层兜底，把未预期异常翻成人话
        info = classify_exception(exc)
        err_console.print(f"[red]意外错误[/red] [{info.code}] {info.message}")
        if "-v" in sys.argv or "--verbose" in sys.argv:
            err_console.print_exception(show_locals=False)
        else:
            err_console.print("[dim]这可能是 bug；加 -v 看完整 traceback[/dim]")
        raise SystemExit(1)


if __name__ == "__main__":
    main_entry()
