#!/usr/bin/env bash
# 一键安装脚本（macOS / Linux 通用）
# uv 管理 venv + 依赖

set -euo pipefail

cd "$(dirname "$0")/.."

# 1) 检查 uv
if ! command -v uv &>/dev/null; then
    echo "[setup] uv 未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version)"

# UV_LINK_MODE=copy 避免 uv 默认的 hardlink 模式偶发丢文件
# （实测会导致 cv2/pptx 包只装一部分子目录，触发奇怪的 ImportError）
export UV_LINK_MODE=copy

# 2) 检查 .env
if [[ ! -f .env ]]; then
    echo "[setup] .env 不存在，从 .env.example 复制（请记得填 DASHSCOPE_API_KEY）"
    cp .env.example .env
fi

# 3) 安装依赖
echo "[setup] uv sync"
uv sync

echo "[setup] done. venv at .venv"
echo "[setup] 试运行：uv run contract-archive --help"
