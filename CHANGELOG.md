# Changelog

本项目变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)，
版本号语义化（单用户本地工具，破坏性变更会在此显著标注）。

## [Unreleased]

### 变更
- **OCR 阶段改用专用 OCR 模型 `qwen-vl-ocr`，逐页调用**（此前用通用 VL 模型 `qwen3.6-flash`
  把整份 PDF 全部页塞进一个请求）。新增配置键 `dashscope.ocr_model`（env `DASHSCOPE_OCR_MODEL`，
  默认 `qwen-vl-ocr-latest`）；签章核查仍用 `dashscope.vl_model`，互不影响。
  - 动机：旧实现只有上下文极大的通用 VL 才扛得住"一次塞全部页"，慢且易超时；专用 OCR 模型
    `qwen-vl-ocr` maxInput 仅 30000，必须逐页。实测 91 页保单：572s（旧 923s，快 38%）、
    91/91 页成功、输出更完整（91984 vs 85157 字符）、单价更低（0.3/0.5 元每百万 token）。
  - `vl_ocr_max_pages` 默认从 10 放宽到 500：逐页后它不再是单请求页数上限，退化为"防超大 PDF
    烧太多次调用"的安全阀；保单条款全文普遍 90+ 页，旧默认 10 会让前置 VL 跳过、回退 mineru。

## [0.2.0] — 2026-05-29

按 [clig.dev](https://clig.dev/) 做的一轮 human-first 打磨（均为非破坏增量）。

### 新增
- `config show --format json`：机器可发现配置旋钮（key/env/secret/default/value/source）。
- `party list` / `party show` 增 `--format json`。
- `show` / `extract` / `party show` 在 `--format json` 下未命中时吐合法 `{"error":"not_found",...}`
  信封到 stdout（此前 stdout 全空，破坏 `| jq`），仍以非零退出。
- 顶层异常钩子：未预期异常翻成一行人话错误，`-v`/`--verbose` 才展开完整 traceback。
- 无参数运行 `contract-archive` / `config` / `party` 展示帮助（含命令清单），不再报 `Missing command`。
- `seals` 增 `--seal-owner` / `--seal-type` 别名（与 `search` 词汇统一；旧 `--owner`/`--type` 保留）。
- `party rm` 删整个主体时增 `--yes` + 非交互守卫（比照 `delete`）。
- 超时旋钮：`DASHSCOPE_TIMEOUT_S`（默认 300s）、`CONTRACT_ARCHIVE_MINERU_TIMEOUT_S`（默认 1800s）。

### 修复
- **MinerU 子进程 / LLM / VL 调用此前全无 timeout**：畸形/超大 PDF 可永久挂死整条 ingest，
  上游 hang 时静默等近 10 分钟。现均有显式上限。
- `LOG_LEVEL` 非法值（如 `bogus`）此前让所有命令崩 traceback；现降级 INFO + warning。
- `extract` 失败（空抽取/LLM 异常）此前一律 exit 0，shell 无法靠 `$?` 发现失败；现 exit 1。
- `--no-color` / `NO_COLOR` 此前对 `raw` 高亮和 `config`/`party` 命令无效；现全命令树一致生效。
- `ingest` 批量 Ctrl-C 此前跳过末尾 checkpoint；现 try/finally 保证清理。
- `describe <未知命令>` 现列出全部可选命令。
- 文案对齐现实：`--no-llm` 不再谎称「只跑 rule」（rule 已退役）；README 项目结构/设计纪律/
  配置/命令清单全面订正；`--limit` 补 help。

### 变更
- 命令入口 `contract_archive.cli:app` → `contract_archive.cli:main_entry`（包顶层异常钩子）。
  **全局安装需 `uv tool install ... --reinstall` 才更新入口脚本。**
- 自称从「合同档案库」统一为「文档档案库」（工具早已支持合同/证明/发票/报告等）。

## [0.1.x] — 历史（未单独发版）

- **Phase 2**：退役 rule 抽取与 rule/LLM hybrid 合并，合同与通用文档抽取统一为纯 LLM。
- **Phase 1**（agent-ready 加固）：结构化错误模型（`code`/`category`/`retryable`）；
  `capabilities`/`describe`/`schema` 机器发现命令；`ingest --progress ndjson` 流式进度；
  `ingest --dry-run` + `--max-files` 成本闸；XDG 配置 + `config` 子命令组。
- 初始：MinerU 解析 + qwen3.7-max 字段抽取 + SQLite 索引 + list/search/show/raw/stats/todo/seals。
