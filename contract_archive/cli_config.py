"""
`config` 子命令：查看 / 设置 / 删除全局配置。

独立文件——cli.py 已逼近 1000 行红线，config 命令组不能再往里塞。
只做 show / set / unset 三个核心命令（砍掉 meeting-asr 的 keys/path/import-env/--json：
单用户本地工具用不上，path 并进 show 抬头即可）。
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .config import config_path, find_key, set_value, unset_value, visible_items

console = Console()
err_console = Console(stderr=True)

# pretty_exceptions_show_locals=False：防 traceback 把 api_key 等局部变量 dump 到终端。
config_app = typer.Typer(
    help="查看/设置全局配置（XDG ~/.config/contract-archive/config.json）",
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@config_app.command("show")
def show(
    reveal: bool = typer.Option(False, "--reveal", help="显示 secret 明文（默认掩码）"),
) -> None:
    """列出各配置项当前生效值（优先级：环境变量 > 配置文件 > 默认）。"""
    cfg = config_path()
    table = Table(
        title=f"Config · {cfg}",
        caption="值来源优先级：环境变量(含 .env) > 配置文件 > 默认值",
        caption_justify="left",
    )
    table.add_column("key", style="cyan")
    table.add_column("value", overflow="fold")
    for name, shown in visible_items(reveal=reveal):
        table.add_row(name, shown)
    console.print(table)
    if not cfg.exists():
        err_console.print(
            "[dim]配置文件尚不存在；`config set` 后自动创建。当前值来自环境变量/默认。[/dim]"
        )


@config_app.command("set")
def set_cmd(
    key: str = typer.Argument(..., help="配置键，如 dashscope.api_key"),
    value: str = typer.Argument(..., help="配置值"),
) -> None:
    """设置一个配置项，写入 ~/.config/contract-archive/config.json（文件权限 0600）。"""
    try:
        cfg = set_value(key, value)
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    k = find_key(key)
    console.print(f"[green]已设置[/green] {k.name} → {cfg}")
    if k.secret:
        err_console.print(
            "[yellow]注意：该文件明文存储 secret，已设为仅本人可读(0600)；请勿提交或分享。[/yellow]"
        )


@config_app.command("unset")
def unset_cmd(
    key: str = typer.Argument(..., help="配置键，如 dashscope.api_key"),
) -> None:
    """从配置文件删除一个配置项（不影响环境变量/默认值）。"""
    try:
        cfg = unset_value(key)
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]已删除[/green] {key.strip()} ← {cfg}")
