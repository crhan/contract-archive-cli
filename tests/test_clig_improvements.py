"""
clig.dev 打磨轮的回归锁：超时旋钮 / LOG_LEVEL 健壮 / 颜色一致 / not_found JSON 信封 /
config 与 party 的机读出口 / party rm 守卫 / seal flag 别名。

这些是「文档与行为对齐 + 机器可消费」的承重点，回归了会静默坑到 agent/脚本，必须守住。
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from contract_archive import cli_common, config as cfg
from contract_archive.cli import app

runner = CliRunner()

ENV_NAMES = [
    "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_LLM_MODEL",
    "DASHSCOPE_VL_MODEL", "CONTRACT_ARCHIVE_DIR", "NO_COLOR", "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """清相关 env + XDG 指向 tmp，隔离开发机真实配置/环境（同 test_config）。"""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


# ---------- get_timeout_s：坏值不崩、回退默认 ----------


def test_get_timeout_s_missing_returns_default(monkeypatch):
    monkeypatch.delenv("X_TO", raising=False)
    assert cfg.get_timeout_s("X_TO", 300.0) == 300.0


def test_get_timeout_s_valid(monkeypatch):
    monkeypatch.setenv("X_TO", "42.5")
    assert cfg.get_timeout_s("X_TO", 1.0) == 42.5


@pytest.mark.parametrize("bad", ["bogus", "-5", "0", "  "])
def test_get_timeout_s_bad_falls_back(monkeypatch, bad):
    monkeypatch.setenv("X_TO", bad)
    assert cfg.get_timeout_s("X_TO", 7.0) == 7.0


# ---------- _resolve_log_level：白名单归一，坏值降级 INFO 不崩 ----------


@pytest.mark.parametrize("raw,expected", [
    ("INFO", "INFO"), ("debug", "DEBUG"), ("Warning", "WARNING"),
    ("10", 10), ("bogus", "INFO"), ("", "INFO"),
])
def test_resolve_log_level(raw, expected):
    assert cli_common._resolve_log_level(raw) == expected


# ---------- color_disabled：--no-color flag 或 NO_COLOR env ----------


def test_color_disabled_reads_no_color_env(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    saved = cli_common.console.no_color
    try:
        cli_common.console.no_color = False
        assert cli_common.color_disabled() is False
        monkeypatch.setenv("NO_COLOR", "1")
        assert cli_common.color_disabled() is True
        monkeypatch.setenv("NO_COLOR", "")  # 空串不算（NO_COLOR 规范）
        assert cli_common.color_disabled() is False
        cli_common.console.no_color = True   # --no-color flag 落点
        assert cli_common.color_disabled() is True
    finally:
        cli_common.console.no_color = saved


def test_subcommand_consoles_are_shared():
    """config/party 复用 cli_common 的全局 console——否则全局 --no-color 触达不到它们。"""
    from contract_archive import cli_config, cli_party
    assert cli_config.console is cli_common.console
    assert cli_config.err_console is cli_common.err_console
    assert cli_party.console is cli_common.console
    assert cli_party.err_console is cli_common.err_console


# ---------- not_found_json：json 模式未命中吐合法信封 ----------


def test_not_found_json_emits_valid_json(capsys):
    cli_common.not_found_json("abc")
    out = capsys.readouterr().out
    assert json.loads(out) == {"error": "not_found", "ident": "abc"}


def test_show_json_not_found_envelope_and_exit(tmp_path):
    """show 未命中（库不存在）：json 模式 stdout 吐合法信封，仍以非零退出。"""
    r = runner.invoke(app, ["show", "99999", "--format", "json", "-a", str(tmp_path / "arch")])
    assert r.exit_code == 1
    assert json.loads(r.stdout) == {"error": "not_found", "ident": "99999"}


def test_extract_json_not_found_envelope(tmp_path):
    r = runner.invoke(app, ["extract", "99999", "--format", "json", "-a", str(tmp_path / "arch")])
    assert r.exit_code == 1
    assert json.loads(r.stdout) == {"error": "not_found", "ident": "99999"}


# ---------- config show --format json：机读配置发现 ----------


def test_config_describe_items_source_and_mask(tmp_path):
    # 直接测 describe_items（不经 CLI callback 的 load_dotenv，避免项目 .env 干扰）。
    p = tmp_path / "c.json"
    cfg.set_value("dashscope.model", "qwen-x", p)
    items = {d["key"]: d for d in cfg.describe_items(path=p)}
    assert items["dashscope.model"]["value"] == "qwen-x"
    assert items["dashscope.model"]["source"] == "file"
    assert items["dashscope.base_url"]["source"] == "default"  # 有默认、无 env/file
    assert items["dashscope.api_key"]["secret"] is True
    assert items["dashscope.api_key"]["value"] == "<unset>"    # 无 env/file/default
    assert items["dashscope.api_key"]["env"] == "DASHSCOPE_API_KEY"


def test_config_describe_items_masks_secret_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-secret")
    items = {d["key"]: d for d in cfg.describe_items(path=tmp_path / "none.json")}
    assert items["dashscope.api_key"]["value"] == "********"   # 默认掩码
    assert items["dashscope.api_key"]["source"] == "env"


def test_config_show_json_cli_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # 避开项目 .env 被 callback 的 load_dotenv 读到
    r = runner.invoke(app, ["config", "show", "--format", "json"])
    assert r.exit_code == 0
    keys = {d["key"] for d in json.loads(r.stdout)}
    assert "dashscope.api_key" in keys and "archive.dir" in keys


# ---------- party：list/show json + rm 守卫 ----------


def test_party_list_json_empty_is_object(tmp_path):
    r = runner.invoke(app, ["party", "list", "--format", "json", "-a", str(tmp_path / "arch")])
    assert r.exit_code == 0
    assert json.loads(r.stdout) == {}


def test_party_show_json_not_found(tmp_path):
    r = runner.invoke(app, ["party", "show", "张三", "--format", "json", "-a", str(tmp_path / "arch")])
    assert r.exit_code == 1
    assert json.loads(r.stdout) == {"error": "not_found", "name": "张三"}


def test_party_rm_whole_subject_non_interactive_refused(tmp_path):
    """非交互（CliRunner 的 stdin 非 TTY）下删整个主体须显式 --yes，否则拒绝退出 1。"""
    r = runner.invoke(app, ["party", "rm", "某主体", "-a", str(tmp_path / "arch")])
    assert r.exit_code == 1
    assert "请加 --yes" in r.output


# ---------- seals 印章 flag 别名：--seal-type/--seal-owner 与旧 --type/--owner 并存 ----------


@pytest.mark.parametrize("flag", ["--seal-type", "--type", "--seal-owner", "--owner"])
def test_seals_flag_aliases_accepted(tmp_path, flag):
    """别名/旧名都不应触发 usage 错误（exit 2）；空库下正常退出。"""
    r = runner.invoke(app, ["seals", flag, "公章", "-a", str(tmp_path / "arch")])
    assert r.exit_code != 2, r.output
