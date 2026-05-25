"""
introspection 命令：让机器（Agent / MCP wrapper）发现能力与结构，无需读源码或解析 --help。

- capabilities      列所有命令 + 副作用/破坏性/幂等元数据（自动遍历 typer app + 安全表）
- describe <cmd>    单命令的参数 schema（名称/类型/必填/默认/可选值/帮助）
- schema <type>     核心数据结构的 JSON Schema（pydantic 直出）

输出一律 JSON 到 stdout，可 `| jq` 消费。为什么手工维护安全元数据：
side_effects / destructive 无法从函数签名推断，必须人为声明——这恰是 Agent 调用前最该知道的
（这命令花不花钱？会不会删数据？能不能安全重试？）。命令清单本身由 click 内省自动生成，
新增命令不会漏，只是缺省按只读安全值兜底，提醒维护者补 META。
"""
from __future__ import annotations

import json as _json
from enum import Enum
from typing import Any

import click
import typer

from . import __version__
from .errors import ErrorInfo
from .schemas import ContractExtraction, DocumentExtraction, ExtractionConfidence

# 输出 schema 版本：未来字段演进时 +1，消费方据此判断兼容性。
INTROSPECT_SCHEMA_VERSION = "1"

# 命令安全元数据。side_effects 取值：
#   read / fs_write / db_write / network / cost(消耗付费 API token)
# destructive=True 表示会删除/不可逆覆盖已有数据；idempotent=True 表示重复执行结果一致。
# 未列出的命令按「只读、非破坏、幂等」兜底（见 _command_entry）。
COMMAND_META: dict[str, dict[str, Any]] = {
    "ingest": {
        "summary": "跑 MinerU + 抽取，把合同入库",
        "side_effects": ["fs_write", "db_write", "network", "cost"],
        "destructive": False, "idempotent": True,
    },
    "extract": {
        "summary": "只重跑抽取（不重跑 MinerU），修复 partial / 改 prompt 后重抽",
        "side_effects": ["fs_write", "db_write", "network", "cost"],
        "destructive": False, "idempotent": True,
    },
    "list": {"summary": "列出档案", "side_effects": ["read"], "destructive": False, "idempotent": True},
    "search": {"summary": "多字段过滤查询", "side_effects": ["read"], "destructive": False, "idempotent": True},
    "show": {"summary": "查看单条详情", "side_effects": ["read"], "destructive": False, "idempotent": True},
    "stats": {"summary": "档案库统计", "side_effects": ["read"], "destructive": False, "idempotent": True},
    "todo": {"summary": "跨合同列待办义务", "side_effects": ["read"], "destructive": False, "idempotent": True},
    "seals": {"summary": "跨文档列印章", "side_effects": ["read"], "destructive": False, "idempotent": True},
    "delete": {
        "summary": "删除单条档案（--purge-files 同时删文件）",
        "side_effects": ["fs_write", "db_write"],
        "destructive": True, "idempotent": True, "requires_confirmation": True,
    },
    "vacuum": {
        "summary": "VACUUM 数据库（碎片整理）",
        "side_effects": ["db_write"], "destructive": False, "idempotent": True,
    },
    "config": {"summary": "查看/设置全局配置", "side_effects": ["read", "fs_write"], "destructive": False, "idempotent": True},
    "capabilities": {"summary": "列命令能力与副作用元数据", "side_effects": ["read"], "destructive": False, "idempotent": True},
    "describe": {"summary": "打印单命令参数 schema", "side_effects": ["read"], "destructive": False, "idempotent": True},
    "schema": {"summary": "打印核心数据结构 JSON Schema", "side_effects": ["read"], "destructive": False, "idempotent": True},
}

# 可经 `schema <type>` 暴露的数据结构（pydantic 直出 JSON Schema）。
SCHEMA_TYPES: dict[str, type] = {
    "document": DocumentExtraction,    # 通用抽取信封（list/show --json 的核心内容）
    "contract": ContractExtraction,    # 合同专属字段
    "confidence": ExtractionConfidence,
    "error": ErrorInfo,                # 失败结果里的结构化错误
}

# register() 时由 cli.py 注入主 app；capabilities/describe 据此内省命令树（避免循环 import）。
_APP: typer.Typer | None = None


def _click_group() -> click.Group:
    """主 app 的 click group——命令内省的入口。"""
    if _APP is None:  # 理论上 register() 必先于命令执行；防御性报错而非静默
        raise RuntimeError("introspect 未注册到 app")
    return typer.main.get_command(_APP)  # type: ignore[return-value]


def _param_to_dict(p: click.Parameter) -> dict[str, Any]:
    """click 参数 → JSON 友好 dict：名称/种类/类型/必填/可选值/默认/帮助。"""
    is_arg = isinstance(p, click.Argument)
    entry: dict[str, Any] = {
        "name": p.name,
        "kind": "argument" if is_arg else "option",
        "type": getattr(p.type, "name", "string"),
        "required": bool(p.required),
    }
    if not is_arg:
        entry["flags"] = list(p.opts) + list(p.secondary_opts or [])
    choices = list(getattr(p.type, "choices", []) or [])
    if choices:
        entry["choices"] = choices
    default = p.default
    if isinstance(default, Enum):
        default = default.value
    if default is not None and not is_arg and isinstance(default, (str, int, float, bool)):
        entry["default"] = default
    help_text = getattr(p, "help", None)
    if help_text:
        entry["help"] = help_text
    return entry


def _command_params(cmd: click.Command) -> list[dict[str, Any]]:
    """命令的全部参数（剔除 click 自动加的 --help）。"""
    return [_param_to_dict(p) for p in cmd.params if p.name != "help"]


def _command_entry(name: str, cmd: click.Command, *, with_params: bool) -> dict[str, Any]:
    """组装单命令的能力描述。META 缺失时按只读安全值兜底。"""
    meta = COMMAND_META.get(name, {})
    summary = meta.get("summary") or (cmd.help or "").strip().splitlines()[0] if cmd.help else ""
    entry = {
        "name": name,
        "summary": meta.get("summary") or summary,
        "side_effects": meta.get("side_effects", ["read"]),
        "destructive": meta.get("destructive", False),
        "idempotent": meta.get("idempotent", True),
        "requires_confirmation": meta.get("requires_confirmation", False),
    }
    if with_params:
        entry["params"] = _command_params(cmd)
    return entry


def _emit(payload: dict[str, Any]) -> None:
    """结构化输出统一走 stdout（机器消费），保证可 `| jq`。"""
    print(_json.dumps(payload, ensure_ascii=False, indent=2))


def capabilities_cmd() -> None:
    """列出所有命令及其副作用/破坏性/幂等元数据（机器可读 JSON）。"""
    group = _click_group()
    commands = [
        _command_entry(name, cmd, with_params=False)
        for name, cmd in sorted(group.commands.items())
    ]
    _emit({
        "schema_version": INTROSPECT_SCHEMA_VERSION,
        "tool": "contract-archive",
        "version": __version__,
        "commands": commands,
    })


def describe_cmd(
    command: str = typer.Argument(..., help="要描述的命令名，如 ingest / list"),
) -> None:
    """打印单个命令的参数 schema（名称/类型/必填/默认/可选值/帮助）。"""
    group = _click_group()
    cmd = group.commands.get(command)
    if cmd is None:
        # 未知命令是用户错——提示走 stderr，退出码 2（与 typer 参数错一致）。
        typer.echo(f"unknown command: {command}", err=True)
        raise typer.Exit(2)
    entry = _command_entry(command, cmd, with_params=True)
    entry["schema_version"] = INTROSPECT_SCHEMA_VERSION
    entry["help"] = (cmd.help or "").strip()
    _emit(entry)


def schema_cmd(
    type_name: str = typer.Argument(
        ..., help=f"数据结构名，可选：{', '.join(sorted(SCHEMA_TYPES))}"
    ),
) -> None:
    """打印核心数据结构的 JSON Schema（pydantic 直出）。"""
    model = SCHEMA_TYPES.get(type_name)
    if model is None:
        typer.echo(
            f"unknown type: {type_name}; available: {', '.join(sorted(SCHEMA_TYPES))}",
            err=True,
        )
        raise typer.Exit(2)
    _emit(model.model_json_schema())


def register(app: typer.Typer) -> None:
    """把 introspection 命令挂到主 app。cli.py 在创建 app 后调用，避免循环 import。"""
    global _APP
    _APP = app
    app.command("capabilities")(capabilities_cmd)
    app.command("describe")(describe_cmd)
    app.command("schema")(schema_cmd)
