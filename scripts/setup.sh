#!/usr/bin/env bash
# 一键安装脚本（macOS / Linux 通用）
# uv 管理 venv + 依赖；MinerU 走 optional extras（体积大 >1GB）

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
# base = 只装核心（pydantic/typer/dashscope SDK），可跑 extract / list / search / show；
# mineru = 加装 MinerU（>1GB，首次跑 ingest 时还会下模型）
EXTRAS="${1:-mineru}"
echo "[setup] uv sync --extra $EXTRAS"
case "$EXTRAS" in
    base)        uv sync ;;
    mineru)      uv sync --extra mineru ;;
    *)
        echo "unknown extras: $EXTRAS"
        echo "可选：base (DB+抽取，无 OCR) / mineru (含 MinerU OCR)"
        exit 2
        ;;
esac

echo "[setup] done. venv at .venv"
echo "[setup] 试运行：uv run ocr-cli --help"
