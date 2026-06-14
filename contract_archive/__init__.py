"""本地文档档案库 CLI：OCR 解析 + qwen3.7-max 字段抽取 + SQLite 索引。"""
from importlib.metadata import PackageNotFoundError, version

try:
    # 单一真相源：版本号只在 pyproject.toml 维护，这里从已安装包元数据读，根除
    # "硬编码 __version__ 与 pyproject 脱节"（历史上 --version 长期报陈旧 0.2.6）。
    __version__ = version("contract-archive-cli")
except PackageNotFoundError:  # 直接在源码树跑（未安装）→ 占位，不崩
    __version__ = "0.0.0+dev"
