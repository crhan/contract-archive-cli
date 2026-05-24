# 本地合同档案库 CLI 容器镜像
#
# 使用：
#   docker build -t contract-archive -f docker/Dockerfile .
#   docker run --rm -it -v $PWD/archive:/app/archive -v $PWD/input:/app/input \
#     --env-file .env contract-archive contract-archive ingest /app/input
#
# 注意：MinerU 首次启动会从 modelscope 下载模型（>1GB），
# 建议挂载 -v ~/.cache/modelscope:/root/.cache/modelscope 复用本机缓存。
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

# 系统依赖：PyMuPDF 不需要 poppler；libgl1 给 MinerU OCR backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# 利用 layer cache：先复制依赖描述
COPY pyproject.toml uv.lock ./
RUN uv sync --extra mineru --no-install-project

COPY contract_archive ./contract_archive
COPY README.md ./README.md
RUN uv sync --extra mineru

CMD ["uv", "run", "contract-archive", "--help"]
