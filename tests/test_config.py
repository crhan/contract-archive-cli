"""
config 层测试：优先级 env>file>default、掩码、set/unset、archive 解析、健壮性。

隔离要点（autouse fixture）：清掉相关 env + 把 XDG_CONFIG_HOME 指向 tmp，
保证测试绝不读开发机真实 env / ~/.config（否则 export 了真 key 会假阳性）。
load_settings 无缓存，monkeypatch 后即时生效，无需清缓存。
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from contract_archive import cli, config as cfg
from contract_archive.archive.paths import default_archive_root

ENV_NAMES = [
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "DASHSCOPE_LLM_MODEL",
    "DASHSCOPE_VL_MODEL",
    "DASHSCOPE_OCR_MODEL",
    "CONTRACT_ARCHIVE_DIR",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """清相关 env + XDG_CONFIG_HOME 指向 tmp，隔离开发机真实配置/环境。"""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def test_config_path_follows_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "x"))
    assert cfg.config_path() == tmp_path / "x" / "contract-archive" / "config.json"


def test_config_path_relative_xdg_falls_back(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/not/abs")  # 非绝对路径不生效
    assert cfg.config_path() == Path.home() / ".config" / "contract-archive" / "config.json"


def test_load_settings_defaults_when_empty():
    """无 env、无配置文件：不报错，api_key 空串，base_url/model 走默认。"""
    s = cfg.load_settings()
    assert s.dashscope_api_key == ""
    assert s.dashscope_base_url == cfg.DEFAULT_DASHSCOPE_BASE_URL
    assert s.dashscope_model == cfg.DEFAULT_DASHSCOPE_MODEL
    assert s.archive_dir is None


def test_config_file_overrides_default(tmp_path):
    p = tmp_path / "c.json"
    cfg.save_config_values({"dashscope.model": "qwen-max"}, p)
    assert cfg.load_settings(p).dashscope_model == "qwen-max"


def test_ocr_model_defaults():
    assert cfg.load_settings().dashscope_ocr_model == cfg.DEFAULT_DASHSCOPE_OCR_MODEL


def test_ocr_model_env_over_default(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_OCR_MODEL", "qwen-vl-ocr-custom")
    assert cfg.load_settings().dashscope_ocr_model == "qwen-vl-ocr-custom"


def test_ocr_model_file_over_default(tmp_path):
    p = tmp_path / "c.json"
    cfg.save_config_values({"dashscope.ocr_model": "qwen-vl-ocr-file"}, p)
    assert cfg.load_settings(p).dashscope_ocr_model == "qwen-vl-ocr-file"


def test_env_overrides_config(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    cfg.save_config_values({"dashscope.model": "from-file"}, p)
    monkeypatch.setenv("DASHSCOPE_LLM_MODEL", "from-env")
    assert cfg.load_settings(p).dashscope_model == "from-env"  # env 压过 file


def test_set_get_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    cfg.set_value("dashscope.api_key", "sk-test", p)
    assert cfg.load_settings(p).dashscope_api_key == "sk-test"


def test_unset_reverts_to_default(tmp_path):
    p = tmp_path / "c.json"
    cfg.set_value("dashscope.model", "x", p)
    cfg.unset_value("dashscope.model", p)
    assert cfg.load_settings(p).dashscope_model == cfg.DEFAULT_DASHSCOPE_MODEL


def test_set_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError):
        cfg.set_value("nope.bad", "x", tmp_path / "c.json")


def test_unknown_key_in_file_ignored(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"nope.bad": "x", "dashscope.model": "ok"}), encoding="utf-8")
    values = cfg.load_config_values(p)
    assert "nope.bad" not in values
    assert values["dashscope.model"] == "ok"


def test_corrupt_file_does_not_crash(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ not json", encoding="utf-8")
    assert cfg.load_config_values(p) == {}  # 坏文件不崩，返回空


def test_save_permissions(tmp_path):
    p = tmp_path / "sub" / "c.json"
    cfg.save_config_values({"dashscope.model": "x"}, p)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700


def test_display_value_masks_secret():
    api = cfg.find_key("dashscope.api_key")
    assert cfg.display_value(api, "sk-secret", reveal=False) == "********"
    assert cfg.display_value(api, "sk-secret", reveal=True) == "sk-secret"
    assert cfg.display_value(api, None, reveal=False) == "<unset>"
    assert cfg.display_value(api, "", reveal=False) == "<unset>"  # 空串 secret 也显示 <unset>
    model = cfg.find_key("dashscope.model")
    assert cfg.display_value(model, "qwen", reveal=False) == "qwen"  # 非 secret 不掩码


def test_archive_dir_env_over_config(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    cfg.save_config_values({"archive.dir": "/from/file"}, p)
    monkeypatch.setenv("CONTRACT_ARCHIVE_DIR", "/from/env")
    assert cfg.load_settings(p).archive_dir == "/from/env"


def test_archive_dir_empty_env_treated_unset(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    cfg.save_config_values({"archive.dir": "/from/file"}, p)
    monkeypatch.setenv("CONTRACT_ARCHIVE_DIR", "")  # 空串当未设，回落 config
    assert cfg.load_settings(p).archive_dir == "/from/file"


def test_whitespace_env_treated_unset(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    cfg.save_config_values({"archive.dir": "/from/file"}, p)
    monkeypatch.setenv("CONTRACT_ARCHIVE_DIR", "   ")  # 纯空白也当未设，与空串一致
    assert cfg.load_settings(p).archive_dir == "/from/file"


def test_non_dict_payload_ignored(tmp_path):
    """合法 JSON 但不是对象（数组/数字）也不崩，返回 {}。"""
    p = tmp_path / "c.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert cfg.load_config_values(p) == {}


# ---------- _resolve_archive（CLI 层优先级链 + ~ 展开 bug fix 的回归锁）----------
# autouse fixture 已 delenv CONTRACT_ARCHIVE_DIR + 把 XDG_CONFIG_HOME 指向 tmp，
# 故 config 文件落在隔离的 tmp 下，不污染真实 ~/.config。


def test_resolve_archive_flag_wins(monkeypatch):
    monkeypatch.setenv("CONTRACT_ARCHIVE_DIR", "/from/env")
    cfg.save_config_values({"archive.dir": "/from/config"})
    assert cli._resolve_archive(Path("/from/flag")).root == Path("/from/flag").resolve()


def test_resolve_archive_env_over_config(monkeypatch):
    cfg.save_config_values({"archive.dir": "/from/config"})
    monkeypatch.setenv("CONTRACT_ARCHIVE_DIR", "/from/env")
    assert cli._resolve_archive(None).root == Path("/from/env").resolve()


def test_resolve_archive_config_when_no_env():
    cfg.save_config_values({"archive.dir": "/from/config"})
    assert cli._resolve_archive(None).root == Path("/from/config").resolve()


def test_resolve_archive_default_when_unset():
    assert cli._resolve_archive(None).root == default_archive_root().resolve()


def test_resolve_archive_expands_tilde(monkeypatch):
    monkeypatch.setenv("CONTRACT_ARCHIVE_DIR", "~/myarchive")  # 锁住 ~ 展开 bug fix
    assert cli._resolve_archive(None).root == (Path.home() / "myarchive").resolve()
