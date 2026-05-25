"""
全局配置：XDG 配置文件 + 进程环境的统一加载。

设计（对齐 clig.dev + 本项目的简洁取向）：
- 配置文件走 XDG：$XDG_CONFIG_HOME/contract-archive/config.json（默认 ~/.config/...）。
- 优先级 env > 配置文件 > 默认值，纯只读短路：`os.getenv() or file or default`。
  cli.py 的 load_dotenv() 已把项目 .env 注入 os.environ（override=False，shell export
  仍优先），所以 .env 天然落在 env 层——保留老 .env 用户零中断。
- load_settings() 只读，**绝不回写 os.environ**；任何字段缺失都不报错
  （api_key 缺失返回空串，由调用方在真要调 LLM 时降级，沿用既有"返回 {}+warning"语义）。
- secret（api_key）落盘是明文，靠目录 0700 + 文件 0600 + 展示掩码保护；
  不上 keyring（对单用户本地 CLI 过重，边际收益低）。
- 故意不收：XDG_DATA_HOME（由 archive/paths.py 决定 data 位置，收进来是循环依赖）、
  MinerU 子进程 environ（基础设施，非配置项）；COMPUTE_DEVICE/LOG_LEVEL 是运行时旋钮，
  保持 env-only，不做持久配置。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

APP_CONFIG_DIR = "contract-archive"
CONFIG_FILENAME = "config.json"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_DASHSCOPE_MODEL = "qwen3.7-max"
DEFAULT_DASHSCOPE_VL_MODEL = "qwen3.6-flash"  # 多模态签章核查（OpenAI 兼容接口）；更准用 qwen3.6-plus


@dataclass(frozen=True, slots=True)
class ConfigKey:
    """一个受支持的全局配置键。"""

    name: str               # 用户输入名 / 文件键名，如 "dashscope.api_key"
    env_name: str           # 对应环境变量名（严格沿用现存名，保证老 .env 值仍被读到）
    secret: bool = False    # 敏感项，展示时掩码
    default: str | None = None


# 只收 4 个：DashScope 三件套 + 档案库路径。env_name 必须 = 现存环境变量名。
CONFIG_KEYS: tuple[ConfigKey, ...] = (
    ConfigKey("dashscope.api_key", "DASHSCOPE_API_KEY", secret=True),
    ConfigKey("dashscope.base_url", "DASHSCOPE_BASE_URL", default=DEFAULT_DASHSCOPE_BASE_URL),
    ConfigKey("dashscope.model", "DASHSCOPE_LLM_MODEL", default=DEFAULT_DASHSCOPE_MODEL),
    ConfigKey("dashscope.vl_model", "DASHSCOPE_VL_MODEL", default=DEFAULT_DASHSCOPE_VL_MODEL),
    ConfigKey("archive.dir", "CONTRACT_ARCHIVE_DIR"),
)
_KEYS_BY_NAME = {k.name: k for k in CONFIG_KEYS}


@dataclass(slots=True)
class Settings:
    """运行时配置：全局配置文件 + 进程环境合并后的取值。"""

    dashscope_api_key: str
    dashscope_base_url: str
    dashscope_model: str
    dashscope_vl_model: str
    archive_dir: str | None
    config_path: Path


def find_key(name: str) -> ConfigKey | None:
    """按 name 查配置键定义；未注册返回 None。"""
    return _KEYS_BY_NAME.get(name.strip())


def config_path() -> Path:
    """XDG 配置文件路径：$XDG_CONFIG_HOME/contract-archive/config.json（默认 ~/.config/...）。"""
    return _xdg_base_dir("XDG_CONFIG_HOME", Path.home() / ".config") / APP_CONFIG_DIR / CONFIG_FILENAME


def _xdg_base_dir(env_name: str, fallback: Path) -> Path:
    """返回绝对 XDG 基目录，否则回退（与 archive/paths.py 同风格：仅绝对路径生效）。"""
    raw = os.getenv(env_name)
    if not raw:
        return fallback
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else fallback


def load_config_values(path: Path | None = None) -> dict[str, str]:
    """
    读配置文件为 {name: value}。文件不存在/损坏/含未知键都不报错——
    只读路径必须健壮（坏配置不能让所有命令崩），未知键跳过并 warning。
    """
    cfg = path or config_path()
    if not cfg.exists():
        return {}
    try:
        payload = json.loads(cfg.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("配置文件读取失败，忽略 %s: %s", cfg, e)
        return {}
    if not isinstance(payload, dict):
        logger.warning("配置文件必须是 JSON 对象，忽略: %s", cfg)
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        name = str(raw_key).strip()
        if name not in _KEYS_BY_NAME:
            logger.warning("配置文件含未知键，跳过: %s", name)
            continue
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value:
            out[name] = value
    return out


def save_config_values(values: dict[str, str], path: Path | None = None) -> Path:
    """写配置文件；目录 0700 / 文件 0600（每次都 chmod，防 umask 宽松导致 secret 可被他人读）。"""
    cfg = path or config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.parent.chmod(0o700)
    cfg.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cfg.chmod(0o600)
    return cfg


def _read_value(values: dict[str, str], key: ConfigKey) -> str | None:
    """
    优先级 os.getenv(含 .env 注入) > 配置文件 > 默认值。只读，不回写 os.environ，不报错。

    strip 后判 truthy，把"未设 / 空串 / 纯空白"三者一视同仁地回落下一层——一把消除
    特殊情况（空串与空白串行为一致），与历史 _resolve_archive 的 truthy 语义一致。
    默认值在此兜底（base_url/model 有 default），故 load_settings 无需再 `or DEFAULT`，
    默认保持单一真相源（只在 CONFIG_KEYS 里定义一次）。
    """
    for candidate in (os.getenv(key.env_name), values.get(key.name), key.default):
        if candidate and candidate.strip():
            return candidate.strip()
    return None


def load_settings(path: Path | None = None) -> Settings:
    """
    加载运行时配置。惰性（每次现读，无缓存——CLI 一次进程读几次，缓存只会让
    `config set` 同进程不生效 + 污染测试）。任何字段缺失都不报错。
    """
    values = load_config_values(path)

    def read(name: str) -> str | None:
        return _read_value(values, _KEYS_BY_NAME[name])

    # 默认值的单一真相源是 CONFIG_KEYS 里的 default（_read_value 已兜底）。
    # 这里的 `or DEFAULT_*` 不是第二默认源（引用同一常量），只是把 read() 的 str|None
    # 类型收敛成 Settings 字段要求的 str。api_key 无 default，`or ""` 同理收敛。
    return Settings(
        dashscope_api_key=read("dashscope.api_key") or "",
        dashscope_base_url=read("dashscope.base_url") or DEFAULT_DASHSCOPE_BASE_URL,
        dashscope_model=read("dashscope.model") or DEFAULT_DASHSCOPE_MODEL,
        dashscope_vl_model=read("dashscope.vl_model") or DEFAULT_DASHSCOPE_VL_MODEL,
        archive_dir=read("archive.dir"),
        config_path=path or config_path(),
    )


def set_value(key: str, value: str, path: Path | None = None) -> Path:
    """设置一个配置项。key 必须是注册表里的 name，否则报错列出支持的键。"""
    name = _validate_key(key)
    values = load_config_values(path)
    values[name] = value.strip()
    return save_config_values(values, path)


def unset_value(key: str, path: Path | None = None) -> Path:
    """从配置文件删除一个配置项（不影响环境变量）。"""
    name = _validate_key(key)
    values = load_config_values(path)
    values.pop(name, None)
    return save_config_values(values, path)


def _validate_key(key: str) -> str:
    name = key.strip()
    if name not in _KEYS_BY_NAME:
        supported = ", ".join(k.name for k in CONFIG_KEYS)
        raise ValueError(f"不支持的配置键: {key}。支持的键: {supported}")
    return name


def display_value(key: ConfigKey, value: str | None, *, reveal: bool) -> str:
    """展示用：secret 不 reveal 时掩码，空值显示 <unset>。"""
    if not value:
        return "<unset>"
    if key.secret and not reveal:
        return "********"
    return value


def visible_items(*, reveal: bool = False, path: Path | None = None) -> list[tuple[str, str]]:
    """config show 用：按注册表顺序给出 (name, 展示值)，值已按 env>file>default 解析。"""
    values = load_config_values(path)
    return [(k.name, display_value(k, _read_value(values, k), reveal=reveal)) for k in CONFIG_KEYS]
