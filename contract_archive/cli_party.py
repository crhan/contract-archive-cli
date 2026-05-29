"""
`party` 子命令：管理 known_parties 身份基准库（查看/录入/删除主体固有标识）。

独立文件——cli.py 已逼近 1000 行红线，不能再塞。known_parties.json 含真实 PII，
故本命令只在本地档案库读写，不提供导出/分享。基准的"首见入库"由 ingest 自动完成，
本命令组负责人工查看与修正：set 覆盖（纠正被 OCR 读错的首见基准）、rm 删除。
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .archive.party_registry import PartyRegistry
from .archive.paths import ArchivePaths, default_archive_root
# 复用 cli_common 的全局 console（理由同 cli_config）：自建实例会让全局 --no-color 失效。
from .cli_common import OutputFormat, console, err_console
from .config import load_settings

# pretty_exceptions_show_locals=False：防 traceback 把 PII 等局部变量 dump 到终端。
party_app = typer.Typer(
    help="管理 known_parties 身份基准库（主体固有标识的跨文档核对基准）",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,  # clig.dev：裸 `party` 列出 list/show/set/rm，而非报 Missing command
    context_settings={"help_option_names": ["-h", "--help"]},
)

_archive_opt = typer.Option(
    None,
    "--archive",
    "-a",
    help="档案库根目录；不传则用 CONTRACT_ARCHIVE_DIR 或 XDG 默认",
)


def _resolve_archive(archive_opt: Optional[Path]) -> ArchivePaths:
    """与 cli._resolve_archive 同逻辑：flag > env/config > XDG 默认。隔离实现以避免循环 import。"""
    if archive_opt:
        root = archive_opt
    else:
        configured = load_settings().archive_dir
        root = Path(configured) if configured else default_archive_root()
    return ArchivePaths(root=root.expanduser().resolve())


def _load_registry(archive_opt: Optional[Path]) -> PartyRegistry:
    return PartyRegistry.load(_resolve_archive(archive_opt).known_parties_path)


@party_app.command("list")
def list_parties(
    archive: Optional[Path] = _archive_opt,
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", help="table | json"),
) -> None:
    """列出基准库里所有主体及其固有标识。"""
    reg = _load_registry(archive)
    parties = reg.all_parties()
    if fmt is OutputFormat.json:
        # known_parties 是跨文档身份核对基准，agent 据此核对身份——给机读出口，别只剩表格。
        # 空库吐合法 {}（与其它命令空集合吐 [] 同一套契约）。注意：含真实 PII，仍只到本地 stdout。
        print(_json.dumps(parties, ensure_ascii=False, indent=2))
        return
    if not parties:
        err_console.print("[yellow]known_parties 为空——入库文档后会自动录入首见标识。[/yellow]")
        return
    table = Table(title=f"known_parties · {len(parties)} 个主体")
    table.add_column("主体", style="cyan", no_wrap=True)
    table.add_column("标识")
    table.add_column("值", overflow="fold")
    table.add_column("首见", style="dim")
    for name, ids in parties.items():
        first = True
        for label, rec in ids.items():
            table.add_row(name if first else "", label, rec.get("value", ""), str(rec.get("first_seen_doc", ""))[:12])
            first = False
    console.print(table)


@party_app.command("show")
def show_party(
    name: str = typer.Argument(..., help="主体名（姓名或机构全称）"),
    archive: Optional[Path] = _archive_opt,
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", help="table | json"),
) -> None:
    """查看某主体的全部标识基准。"""
    reg = _load_registry(archive)
    ids = reg.get(name)
    if not ids:
        # json 模式吐 not_found 信封到 stdout（别让 | jq 拿空输入）；table 走 stderr。
        if fmt is OutputFormat.json:
            print(_json.dumps({"error": "not_found", "name": name}, ensure_ascii=False))
        else:
            err_console.print(f"[red]未找到主体: {name}[/red]")
        raise typer.Exit(1)
    if fmt is OutputFormat.json:
        print(_json.dumps(ids, ensure_ascii=False, indent=2))
        return
    table = Table(title=f"{name} · {len(ids)} 项标识")
    table.add_column("标识", style="cyan")
    table.add_column("值", overflow="fold")
    table.add_column("角色", style="dim")
    table.add_column("首见出处", style="dim")
    for label, rec in ids.items():
        table.add_row(label, rec.get("value", ""), rec.get("role", ""), str(rec.get("first_seen_doc", "")))
    console.print(table)


@party_app.command("set")
def set_party(
    name: str = typer.Argument(..., help="主体名"),
    label: str = typer.Argument(..., help="标识名，如 身份证号 / 电话 / 银行账号"),
    value: str = typer.Argument(..., help="标识值"),
    archive: Optional[Path] = _archive_opt,
) -> None:
    """手动录入/修正某主体的标识基准（覆盖既有值；用于纠正首见时被 OCR 读错的基准）。"""
    paths = _resolve_archive(archive)
    reg = PartyRegistry.load(paths.known_parties_path)
    try:
        reg.set(name, label, value)
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    reg.save()
    console.print(f"[green]已设置[/green] {name}·{label} → {paths.known_parties_path}")
    err_console.print(
        "[yellow]注意：known_parties.json 明文存 PII，已设为仅本人可读(0600)，请勿提交或分享。[/yellow]"
    )


@party_app.command("rm")
def rm_party(
    name: str = typer.Argument(..., help="主体名"),
    label: Optional[str] = typer.Argument(None, help="标识名；省略则删除该主体全部标识"),
    archive: Optional[Path] = _archive_opt,
) -> None:
    """删除某主体的某标识；不给 label 则删除整个主体。"""
    paths = _resolve_archive(archive)
    reg = PartyRegistry.load(paths.known_parties_path)
    target = f"{name}·{label}" if label else name
    if reg.remove(name, label):
        reg.save()
        console.print(f"[green]已删除[/green] {target}")
    else:
        err_console.print(f"[red]未找到: {target}[/red]")
        raise typer.Exit(1)
