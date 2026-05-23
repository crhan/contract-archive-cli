#!/usr/bin/env bash
# 一键安装脚本（macOS / Linux 通用）
# 用 uv 管理 venv 和依赖

set -euo pipefail

cd "$(dirname "$0")/.."

# 1) 检查 uv
if ! command -v uv &>/dev/null; then
    echo "[setup] uv 未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version)"

# 2) 检查 .env
if [[ ! -f .env ]]; then
    echo "[setup] .env 不存在，从 .env.example 复制（请记得填 DASHSCOPE_API_KEY）"
    cp .env.example .env
fi

# 3) 安装依赖
EXTRAS="${1:-dashscope}"
echo "[setup] uv sync --extra $EXTRAS"
case "$EXTRAS" in
    base)        uv sync ;;
    dashscope)   uv sync --extra dashscope ;;
    paddleocr)   uv sync --extra paddleocr ;;
    mineru)      uv sync --extra mineru ;;
    all)         uv sync --extra all ;;
    *) echo "unknown extras: $EXTRAS (base/dashscope/paddleocr/mineru/all)"; exit 2 ;;
esac

echo "[setup] done. venv at .venv"
echo "[setup] 试运行：uv run ocr-cli --help"
