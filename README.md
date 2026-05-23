# Document Intelligence Playground

> **目标**：在同一份 PDF 上，横向对比 **阿里百炼 (qwen-vl-ocr)**、**PaddleOCR (PP-StructureV3)**、**MinerU 3.x** 三路 OCR + Document Parsing 能力，并在统一 schema 上跑合同语义抽取 (rule + LLM hybrid)。
>
> 这不是归档系统，是验证场。

## ✦ 功能

```
            ┌─ DashScope (qwen-vl-ocr) ─┐
PDF ───────┼─ PaddleOCR (PP-StructureV3)┤── 统一 schema ── compare.py ── 对比报告
            └─ MinerU 3.x ──────────────┘                     │
                                                              └── Semantic Extraction
                                                                  rule + LLM (qwen3.7-max)
```

每路 OCR 都产出**同一份 schema**：

```
output/<pipeline>/
├── raw_text.txt              # 纯文本
├── markdown.md               # 带结构 markdown
├── structured.json           # title/document_type/pages/sections/tables/entities
├── layout.json               # [{bbox, page, text, block_type, ...}]
├── pipeline_meta.json        # 模型/版本/设备/耗时
├── preview_images/           # 每页 PNG
├── extraction_result.json    # 合同字段（rule + LLM hybrid）
└── extraction_confidence.json
```

## ✦ 当前阶段硬件路线

| 阶段 | 硬件 | 三路情况 |
| --- | --- | --- |
| **现在** | MacBook (Apple Silicon, arm64) | dashscope ✅ / paddleocr ⚠CPU / mineru ⚠CPU |
| 未来 | Linux + RTX 5080 (sm_120) | 全部 GPU；切 `--profile gpu` |

设备选择策略：`COMPUTE_DEVICE=auto` 时自动 `MPS → CUDA → CPU` 三档降级。

## ✦ 快速开始 (macOS Native, 推荐)

### 1. 安装

```bash
# 1) 装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 装最轻量的 dashscope 依赖（先跑通这一路）
./scripts/setup.sh dashscope
```

### 2. 填 API key

```bash
# .env 已经从 .env.example 自动复制；填入 DashScope key
$EDITOR .env
```

`.env` 字段说明：

| 字段 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 必填。从 [百炼控制台](https://dashscope.console.aliyun.com/) 获取 |
| `DASHSCOPE_OCR_MODEL` | OCR 模型，默认 `qwen-vl-ocr-latest` |
| `DASHSCOPE_LLM_MODEL` | 语义抽取模型，默认 `qwen3.7-max`（严格按需求保留） |
| `COMPUTE_DEVICE` | `auto` / `mps` / `cuda` / `cpu` |

### 3. 跑样本 PDF

```bash
# 三路全跑 + extraction + 对比报告（先确保只装了 dashscope，paddleocr/mineru 会优雅 skip）
./scripts/run_sample.sh dashscope

# 或单跑某一路
uv run ocr-cli run --pipeline dashscope ./input/sample_contract.pdf
uv run ocr-cli extract ./output/dashscope
uv run ocr-cli compare ./output
```

### 4. 装重型依赖（PaddleOCR / MinerU）

```bash
./scripts/setup.sh paddleocr    # 装 paddlepaddle (cpu) + paddleocr[all]
./scripts/setup.sh mineru       # 装 mineru[core]
./scripts/setup.sh all          # 一次全装
```

> **Mac 注意**：`paddlepaddle` 在 arm64 上只有 CPU wheel，跑 9 页扫描合同约 30–120s；`mineru` CPU backend (-b pipeline) 同样会慢。建议先用 dashscope 验证整套链路通了，再装重型路线。

## ✦ Docker 用法（Linux 推荐）

```bash
# 只跑 FastAPI + dashscope（最轻）
docker compose up api

# 含 paddleocr/mineru 的重型容器
docker compose --profile heavy up

# Linux + NVIDIA GPU
docker compose --profile gpu up
```

> **Mac 警告**：Docker Desktop for Mac 不能直通 Apple GPU，重型 pipeline 在 Mac docker 里会 CPU 跑且更慢。Mac 阶段优先 native venv。

## ✦ HTTP API

```bash
# 启服务
uv run ocr-cli serve --port 8000

# 上传 PDF 让指定 pipeline 处理
curl -F 'file=@input/sample_contract.pdf' http://localhost:8000/ocr/dashscope

# 三路 + extraction + compare 一把梭
curl -F 'file=@input/sample_contract.pdf' http://localhost:8000/pipeline/all
```

## ✦ Benchmark

```bash
uv run ocr-cli benchmark ./input/sample_contract.pdf --rounds 1
```

会落 `./output/benchmark.json`，含每路 duration/raw_chars/md_chars/layout_blocks。

## ✦ 项目结构

```
ocr-cli/
├── pyproject.toml          # uv 依赖管理（extras: dashscope/paddleocr/mineru/all）
├── docker-compose.yml
├── docker/
│   ├── Dockerfile          # base + heavy stage
│   └── Dockerfile.gpu      # 未来 5080 用
├── .env.example
├── scripts/
│   ├── setup.sh
│   └── run_sample.sh
├── src/
│   ├── cli.py              # ocr-cli 入口
│   ├── compare.py          # 对比报告生成
│   ├── schemas/            # pydantic 统一 schema
│   ├── pipelines/
│   │   ├── base.py
│   │   ├── dashscope_pipeline.py
│   │   ├── paddleocr_pipeline.py
│   │   └── mineru_pipeline.py
│   ├── extraction/
│   │   ├── rule_extractor.py
│   │   ├── llm_extractor.py     # qwen3.7-max
│   │   └── hybrid.py
│   ├── api/
│   │   └── app.py              # FastAPI
│   └── utils/
│       ├── device.py           # MPS → CUDA → CPU 自动降级
│       └── pdf.py              # PyMuPDF 渲染
├── input/                 # 放待处理 PDF
└── output/                # 三路输出 + compare_report.md
```

## ✦ 设计纪律

- **三路彼此独立**：任何一路依赖装不上不影响其他两路（lazy import + optional extras）
- **统一 schema 是契约**：所有 pipeline 必须把自己的输出折算到 `BBox(pt)` / `LayoutBlock` / `StructuredDocument`
- **rule + LLM hybrid**：单字段命中策略明确，每个字段都有 `value_source` 记录来源
- **API key 不出包**：仅从 env 读取，所有日志路径都过滤

## ✦ 未来迁移到 RTX 5080

1. 切到 Linux 主机
2. `cp .env .env` 一份
3. `docker compose --profile gpu up`
4. `COMPUTE_DEVICE=cuda` 自动启用 CUDA
5. PaddlePaddle 改用 [paddlepaddle-sm120-wheels](https://github.com/horhe-dvlp/paddlepaddle-sm120-wheels) 社区 wheel（官方 wheel 在 2026.5 还没合 sm_120）

代码层面 0 改动。
