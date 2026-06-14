# Changelog

本项目变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)，
版本号语义化（单用户本地工具，破坏性变更会在此显著标注）。

## [0.3.1] — 2026-06-14

### 修复
- **`--version` 报告陈旧版本**：`contract_archive/__init__.py` 历史硬编码 `__version__='0.2.6'`，
  历次只 bump `pyproject.toml` 漏改它 → 装上 0.2.7/0.3.0 后 `--version` 仍显 `0.2.6`。改为
  `importlib.metadata.version()` 从已安装包元数据动态读取，版本号**单一真相源 = pyproject**，根除脱节。
- **发版 CI**：`release.yml` 的 `astral-sh/setup-uv@v8` 不存在（该 action 最新 major 为 v7），
  致 v0.3.0 发版 CI 在 "Set up job" 解析 action 失败、未发布。改回 `@v7`。

## [0.3.0] — 2026-06-14

多源融合提取 + 文档类型路由泛化 + 评测私有化（PR #1）+ 评测调试驱动的 6 项融合改进（PR #2）。
含若干**破坏性变更**（抽取流水线行为变化），单用户本地工具，升级后建议对关键文档重抽一遍核对。

### 破坏性变更
- **页级分流取代整份"二选一"**：mineru 改**混合提取**——逐页判 text/ocr（主判据=单页文本层质量，
  含表格的文本页也走 VL），文本页原生抽取、扫描/表格页 VL OCR，按页序拼回。取代旧的"整份 native
  OR 整份 OCR"，混合版式（夹扫描页/夹表格页）不再系统性丢数据。
- **文档类型路由泛化**：`doc_type → handler` 映射（`extraction/doc_type_handlers.py`）取代散落各处的
  `if doc_type == "合同协议"`。本项目处理**所有** PDF 类型（合同/保险/证明/发票/旅行/证件…）——先识别
  doc_type（通用信封）再据类型走特化（特化抽取/后处理/是否开多源融合）。
- **评测私有化**：评测**数据集**（原始 PDF + 真实金标准，**不脱敏**）迁私有仓库，评测**框架代码**留主仓库；
  框架读 env `CONTRACT_ARCHIVE_EVALSET_DIR` 定位数据集，删 `make_gold` 三层脱敏机制。

### 新增
- **多源融合**（保险为首个落地类型）：A(文本 `read_fields_in_text`)/C(看图 `read_fields_on_images`)两路
  **并发**按同一组高价值概念键抽候选——**一致直接采信**（省一次 LLM）、**矛盾才据原图评判**
  （`fusion.fuse_sources`）。结论只写 `field_verdicts`/`fusion_overall_confidence` **sidecar，绝不回写原
  `amounts/fields`**（保护 evidence/unit/is_total_component 与 computed_total 的勾稽不变量）。
- 保险特化概念键 `INSURANCE_FIELD_DEFS`：投保人/被保险人（分列且各带口径，防文本路把投保人误当被保险人）、
  保单号/保险期间/各档保额（一般/特定医疗/重疾/身故各独立键）/年度限额/免赔额/赔付比例（社保内外）/
  等待期/保证续保年限。
- **并发基建** `utils.map_concurrent`（保序、单项失败隔离）+ `merge_usage`；逐页 OCR、看图抽字段、多字段
  评判都走它。旋钮 `CONTRACT_ARCHIVE_LLM_CONCURRENCY`（默认 4）。
- `agent_fallback.escalate_low_confidence` 兜底接口（本期 **no-op** 仅标记 low_confidence，未来插 agentic 只改这一处）。
- 配置键 `dashscope.vl_extract_model`（env `DASHSCOPE_VL_EXTRACT_MODEL`，看图抽字段模型，默认 `qwen3.6-flash`）。
- 评测支持**原始 PDF 全链路评测**（OCR→类型路由→特化→多源融合，对照 gold）；融合 `field_verdicts` 纳入评分门禁。

### 变更
- 逐页 OCR 由串行改**并发**（`map_concurrent`），计数器竞态用结构化结果消除。实测 91 页保单提速明显。
- 文本层判据加覆盖率门槛 + "绝对纯扫描页数"判据：修扫描件夹文本页被错跳 OCR、小份夹页扫描件漏网。
- **评测调试（6 份真实保单实测 field_verdicts micro-F1 = 0.979）驱动的 6 项融合改进**：
  ① 补 `保额_身故` 概念键（意外/寿险身故金额原无键，doc36 身故300万由全漏→捕获）；
  ② `_normalize_value` 整年月归一（"12个月"=="1年"）；
  ③ 评判 prompt 投保人/被保险人**取投保单结构化栏**（签名常代签，防被签名栏误覆盖真值）；
  ④ `年度限额` 定义锐化（仅医疗类，排除身故限额双标）；
  ⑤ vision 选页**封面页优先纳入**（大文档封面不被表格页挤出截断窗）；
  ⑥ native-text 快路（ocr_pages==0）下按需补渲 vision 选中页（`render_pdf_to_images` 加 `pages` 参数）。
- CI：`checkout` v4→v6、`setup-uv` v5→v8（消除 Node 20 弃用警告）。

## [0.2.7] — 2026-06-13

### 变更
- **OCR 阶段改用专用 OCR 模型 `qwen-vl-ocr`，逐页调用**（此前用通用 VL 模型 `qwen3.6-flash`
  把整份 PDF 全部页塞进一个请求）。新增配置键 `dashscope.ocr_model`（env `DASHSCOPE_OCR_MODEL`，
  默认 `qwen-vl-ocr-latest`）；签章核查仍用 `dashscope.vl_model`，互不影响。
  - 动机：旧实现只有上下文极大的通用 VL 才扛得住"一次塞全部页"，慢且易超时；专用 OCR 模型
    `qwen-vl-ocr` maxInput 仅 30000，必须逐页。实测 91 页保单：572s（旧 923s，快 38%）、
    91/91 页成功、输出更完整（91984 vs 85157 字符）、单价更低（0.3/0.5 元每百万 token）。
  - `vl_ocr_max_pages` 默认从 10 放宽到 500：逐页后它不再是单请求页数上限，退化为"防超大 PDF
    烧太多次调用"的安全阀；保单条款全文普遍 90+ 页，旧默认 10 会让前置 VL 跳过、回退 mineru。
- **逐页 OCR 健壮性加固**（逐页后单份要发几十上百个请求，失败点随页数累积）：
  - 单页异常态分标，不再都塞 `[看不清]`——`[本页 OCR 调用失败]`（请求级失败）/
    `[本页输出达模型上限被截断]`（`finish_reason==length`，单页输出硬上限 8192 token）/
    `[看不清]`（模型正常返回但本页无文本）。把"技术失败"与"原文模糊"分开，前者可事后审计/补跑。
  - SDK 重试由默认 2 调高到 4（env `CONTRACT_ARCHIVE_VL_OCR_RETRIES`）：429/超时/5xx 由 openai
    SDK 自动指数退避（读 `Retry-After`），避免偶发抖动直接丢一整页内容。
  - 补 `tests/test_vl_ocr.py`（单页失败隔离 / 全失败回退 / 截断标记 / 空输入 / 缺凭证 / 重试旋钮）
    与 config 层 `dashscope.ocr_model` 覆盖。

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
