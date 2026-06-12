"""
CLI 基础设施（被各命令模块共享的叶子模块，不依赖任何 cli_* 命令模块）。

放这里的都是"框架级"共享件，与具体命令无关：
  - app          主 Typer 实例 + 全局 callback（命令在别处用 @app.command 挂上来）
  - 参数 Enum    parse-time 校验，坏值由 typer 报 exit 2，不漏到数据层
  - 双 console   数据走 stdout（可管道）/ 诊断走 stderr
  - _resolve_*   档案库路径解析、ident 消歧、空库守卫（多个命令共用）

依赖方向（保持 DAG，禁止反向）：cli_common ← cli_query / cli（写命令）。
"""
from __future__ import annotations

import json as _json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from . import __version__
from .archive import (
    ArchivePaths,
    default_archive_root,
    find_by_sha_prefix,
    get_document,
)
from .config import load_settings

# ---------- 参数枚举（parse-time 校验：坏值由 typer 报 exit 2，不再漏到数据层 ValueError）----------


class OutputFormat(str, Enum):
    """--format：人类表格 or 机器 JSON。"""

    table = "table"
    json = "json"


class ColorWhen(str, Enum):
    """--color：auto=仅 TTY 上色（管道纯文本）；always=强制（配 less -R）；never=禁用。"""

    auto = "auto"
    always = "always"
    never = "never"


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
    insurance = "保险凭证"
    travel = "旅行资料"
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


def color_disabled() -> bool:
    """
    全局是否应禁用颜色：--no-color flag（落到 console.no_color）或 NO_COLOR 环境变量。

    rich console 自身已尊重 NO_COLOR + 被 callback 的 --no-color 置过 no_color；但 raw 命令的
    高亮不经 console（直写 ANSI 转义码），故需显式查这个开关。NO_COLOR 规范：非空即禁用
    （空串不算），故 bool(os.environ.get("NO_COLOR")) 恰好对（空串 falsy）。
    """
    return bool(console.no_color) or bool(os.environ.get("NO_COLOR"))


def _version_cb(value: bool) -> None:
    """--version 的 eager 回调：版本号打到 stdout（机器可消费），随即退出。"""
    if value:
        print(f"contract-archive {__version__}")
        raise typer.Exit()


app = typer.Typer(
    help="本地文档档案库 CLI（合同/证明/发票等，OCR + qwen3.7-max）",
    # clig.dev：无参数应展示帮助，而非报 "Missing command" 错误框。
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    # 关掉 typer 自带的 rich traceback 接管：未预期异常改由 cli.main_entry 的顶层钩子翻成
    # 人话（默认一行错误 + 提示 -v 展开），别直接 dump 一坨实现细节给用户/agent。
    pretty_exceptions_enable=False,
    # show_locals=False 留作防御纵深：万一将来 enable 翻开，也不把局部变量（可能含 secret）dump 出。
    pretty_exceptions_show_locals=False,
    epilog=(
        "示例：\n"
        "  contract-archive ingest ./input            # 扫描目录入库\n"
        "  contract-archive list --format json | jq   # 机器可读，管道友好\n"
        "  contract-archive todo --within-days 30      # 近 30 天待办义务\n"
        "\n文档：https://github.com/crhan/contract-archive-cli"
    ),
)


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
        level: str | int = "DEBUG"
    elif quiet:
        level = "WARNING"
    else:
        level = _resolve_log_level(os.getenv("LOG_LEVEL", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _resolve_log_level(raw: str) -> str | int:
    """
    LOG_LEVEL 白名单归一：合法名（大小写不敏感）原样、纯数字转 int，其余降级 INFO 并 warning。

    此前把原始字符串直喂 logging.basicConfig，`LOG_LEVEL=bogus` 会抛 ValueError 把所有命令打挂
    （连只读的 list/show 都崩）。坏 env 不该让命令崩——与 config 层「坏配置不崩、warning 后降级」
    一个取向，消除「文件配置坏了优雅、env 坏了硬崩」这个特殊情况。
    """
    norm = (raw or "").strip().upper()
    if norm in _VALID_LOG_LEVELS:
        return norm
    if norm.isdigit():
        return int(norm)
    err_console.print(f"[yellow]无效 LOG_LEVEL={raw!r}，降级为 INFO[/yellow]")
    return "INFO"


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


def not_found_json(ident: str) -> None:
    """
    show/extract 的 json 模式未命中时吐合法 JSON 错误信封到 stdout（调用方随后 Exit(1)）。

    与空集合命令吐 `[]`、stats 吐零值对象同一套 JSON 契约：json 模式永不让 stdout 为空，
    保证 `| jq` / json.loads 拿到可解析对象。退出码仍非零，让 shell 也能判失败。
    """
    print(_json.dumps({"error": "not_found", "ident": ident}, ensure_ascii=False))


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
