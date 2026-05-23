# 基础镜像：用 Python 3.12 + uv，覆盖 dashscope 那一路 + FastAPI
# 这层故意只装最小依赖，体积小、build 快、纯 CPU
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

# 系统依赖：PyMuPDF 不需要 poppler；只装 build essentials 兜底
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# 先复制依赖描述文件以利用 layer cache
COPY pyproject.toml ./

# base：只装 dashscope + 通用基础依赖
RUN uv sync --extra dashscope --no-install-project

# 拷代码
COPY src ./src
COPY README.md ./README.md
RUN uv sync --extra dashscope

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ===== heavy stage：含 paddleocr + mineru，体积大 =====
FROM base AS heavy
RUN uv sync --extra all
CMD ["tail", "-f", "/dev/null"]
