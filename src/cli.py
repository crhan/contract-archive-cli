"""
ocr-cli 主命令行。

本文件在 Phase 1 重构中被清空到最小框架——旧的 run/extract/compare/serve/benchmark
命令均已删除。Phase 3c 会重新加入面向档案库的子命令：
  ingest / list / search / show / extract / stats / delete
"""
from __future__ import annotations

import logging
import os

import typer
from dotenv import load_dotenv
from rich.console import Console

app = typer.Typer(help="本地合同档案库 CLI (MinerU + qwen3.7-max)")
console = Console()
load_dotenv()  # 自动加载项目根目录 .env

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


if __name__ == "__main__":
    app()
