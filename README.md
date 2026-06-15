# 本地文档档案库 CLI

> 把各类文档 PDF 批量入库、归档、可追溯——合同、协议、证明、发票、报告……
> OCR 提取版面文本，qwen3.7-max LLM 判类型 + 抽字段，索引到本地 SQLite，
> 支持按类型/字段过滤检索。

**LLM-first**：文档类型与字段抽取都交给 LLM，统一归一化到一个「通用信封」
（doc_type / title / summary / 主体 / 金额 / 日期 / 柔性字段）。加新文档类型
**无需写代码**——LLM 自行决定抽什么。死代码 rule 仅保留为确定性数值归一化
（中文大写金额→数值、日期→ISO）。合同另有一份调校过的专属 prompt（同样纯 LLM），
仍保留全部合同字段与查询（甲乙方/到期日/自动续约/风险条款/义务清单）。

历史：本项目最初是多路 OCR 对比 playground，后重构为档案库 CLI；
再从「合同专用」扩展为「通用文档档案库」。

## ✦ 数据流

```
PDF ─► sha256 去重 ─► OCR 提取（页级分流：文本页原生 / 扫描·表格页 VL OCR 混合提取）
                          │
                          ▼
              LLM 判类型 ─► doc_type → handler 特化抽取
                          ├─ 合同：专属 prompt + 看落款页签章核查
                          └─ 保险：多源融合（A 文本 / C 看图两路评判 → field_verdicts sidecar）
                                          │
              ┌───────────────────────────┴──┐
              ▼                              ▼
   db.sqlite (通用信封 + 索引)     documents/<sha-12>/
                                    ├── source.pdf  (硬链接)
                                    ├── ocr/markdown.md ...
                                    ├── extraction_result.json  (通用信封 + 融合 sidecar)
                                    └── ingest.log

档案库默认在 XDG 数据目录：~/.local/share/contract-archive/
```

## ✦ 安装

```bash
# 1) 装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 装依赖
./scripts/setup.sh
```

> **uv hardlink 坑**：uv 默认 `UV_LINK_MODE=hardlink` 偶发只装包的一部分文件
> （实测 `cv2`/`pptx` 会丢，触发 `module 'cv2' has no attribute 'INTER_NEAREST'`
> 或 `cannot import name 'Presentation' from 'pptx'`）。`scripts/setup.sh` 已
> 显式 `export UV_LINK_MODE=copy` 规避。手动 `uv sync` 时建议也带上。
> 已损坏的包可以 `uv pip install --force-reinstall --no-deps <包名>` 修。

## ✦ 全局安装（可选）

如果想在任意目录用 `contract-archive`（不必 `cd` 项目目录或 `uv run`），用 `uv tool install`：

```bash
# 用 --reinstall 而非 --force：版本号没变时 --force 会命中 uv 缓存里的旧 wheel，
# 把过时代码装进去（实测会停在旧版本）；--reinstall 强制重建，更新才可靠。
UV_LINK_MODE=copy uv tool install --reinstall "/path/to/contract-archive-cli"
```

`uv tool install` 会在 `~/.local/bin/contract-archive` 装独立 venv（与项目 venv 隔离）。
然后从任意目录：

```bash
# 用环境变量指定档案库
CONTRACT_ARCHIVE_DIR=~/contracts contract-archive list

# 或显式 --archive（per-command 选项，放在子命令之后）
contract-archive list --archive ~/contracts
contract-archive ingest ~/Documents/new_contract.pdf --archive ~/contracts
```

`DASHSCOPE_API_KEY` 建议用 `contract-archive config set dashscope.api_key ...` 配置；
也可以通过 shell env / `.env` 提供。

**开发者（改了源码要即时生效）**：加 `--editable`，全局命令指向本仓库源码而非快照——
改完 `.py` 直接生效，不必每次重装：

```bash
UV_LINK_MODE=copy uv tool install --editable --reinstall "/path/to/contract-archive-cli"
```

> 不加 `--editable` 装的是「当下代码的快照」：之后改了源码、或仓库升了版本，都得
> 重新 `--reinstall` 才更新。如果发现 `contract-archive --version` 跟仓库 `pyproject.toml`
> 对不上，多半就是装了旧快照——重装即可。

卸载：

```bash
uv tool uninstall contract-archive-cli
# 数据/配置不随之删除，需手动清理：
#   ~/.local/share/contract-archive   档案库数据（db.sqlite + documents/）
#   ~/.config/contract-archive        config.json
```

## ✦ 配置

两种方式，优先级 **环境变量(含 .env) > config 文件 > 默认值**。
日常使用推荐 `config` 命令；`.env.example` 仍保留，用于项目内开发、Docker、临时覆盖模型，
以及 env-only 的运行时旋钮。

```bash
# 方式一：config 命令（落 ~/.config/contract-archive/config.json，权限 0600，比项目 .env 更安全）
contract-archive config set dashscope.api_key sk-xxx
contract-archive config show              # 看各项当前生效值与来源（secret 默认掩码）
contract-archive config show --format json   # 机读：含 key/env/secret/default/value/source
contract-archive config unset dashscope.api_key

# 方式二：项目 .env（开发 / Docker / 临时覆盖用，仍支持）
cp .env.example .env
$EDITOR .env   # 填入 DASHSCOPE_API_KEY
```

| 环境变量 | config 键 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | `dashscope.api_key` | 必填。[百炼控制台](https://dashscope.console.aliyun.com/) 申请 |
| `DASHSCOPE_LLM_MODEL` | `dashscope.model` | 默认 `qwen3.7-max`（用户百炼账户的特定别名；若 404 换 `qwen-max` / `qwen3-max`） |
| `DASHSCOPE_BASE_URL` | `dashscope.base_url` | 默认 `https://dashscope.aliyuncs.com/api/v1`；海外换 `https://dashscope-intl.aliyuncs.com/api/v1` |
| `DASHSCOPE_OCR_MODEL` | `dashscope.ocr_model` | 逐页 VL OCR 模型，默认 `qwen-vl-ocr-latest` |
| `DASHSCOPE_VL_MODEL` | `dashscope.vl_model` | 签章核查视觉模型，默认 `qwen3.6-flash` |
| `DASHSCOPE_VL_EXTRACT_MODEL` | `dashscope.vl_extract_model` | 多源融合"看图抽字段"视觉模型，默认 `qwen3.6-flash` |
| `CONTRACT_ARCHIVE_DIR` | `archive.dir` | 档案库根目录，默认 XDG `~/.local/share/contract-archive`；CLI `--archive` 优先 |
| `COMPUTE_DEVICE` | — | `auto` / `mps` / `cuda` / `cpu`（记录运行环境，默认自动选择） |
| `LOG_LEVEL` | — | `DEBUG`/`INFO`/`WARNING`/...，默认 `INFO`；`--verbose`/`--quiet` 覆盖之 |
| `DASHSCOPE_TIMEOUT_S` | — | LLM/VL 调用超时秒数，默认 `300` |
| `CONTRACT_ARCHIVE_LLM_CONCURRENCY` | — | LLM 调用并发度（OCR/看图/评判共用线程池），默认 `4` |
| `CONTRACT_ARCHIVE_VL_OCR_MAX_PAGES` | — | 单文档允许走 VL OCR 的最大页数，默认 `500` |
| `CONTRACT_ARCHIVE_VL_OCR_DPI` | — | VL OCR 渲染页图 DPI，默认 `160` |
| `CONTRACT_ARCHIVE_VL_OCR_RETRIES` | — | 逐页 VL OCR SDK 重试次数，默认 `4` |
| `CONTRACT_ARCHIVE_VISION_FUSION_MAX_PAGES` | — | 融合看图单文档最多看几页（优先表格/扫描页），默认 `20` |
| `CONTRACT_ARCHIVE_FUSION_THRESHOLD` | — | 融合低置信阈值 `[0,1]`，低于触发 agent 兜底，默认 `0.6` |
| `CONTRACT_ARCHIVE_EVALSET_DIR` | — | 评测私有数据集根目录（仅开发/质量门禁用）；不设回退主仓库内合成 cases |

> 标 `—` 的是运行时旋钮，保持 env-only、不进 config 文件层。

## ✦ 用法

```bash
# 入库单个 PDF
uv run contract-archive ingest path/to/合同.pdf

# 批量入库整个目录（递归扫 *.pdf，sha256 去重）
uv run contract-archive ingest ~/Documents/contracts/

# 跳过字段抽取：仅入库 OCR 产物，抽取字段留空，可后续 extract 补抽
uv run contract-archive ingest path/to/合同.pdf --no-llm

# 强制重跑（已 ingest 过的也再跑一遍，覆盖旧记录）
uv run contract-archive ingest path/to/合同.pdf --reingest

# 试跑前 3 个
uv run contract-archive ingest ~/Documents/contracts/ --limit 3

# 成本/进度（agent 友好）
uv run contract-archive ingest ~/Documents/contracts/ --dry-run        # 只预览扫到几个、预计几次 API 调用，不建库不烧钱
uv run contract-archive ingest ~/Documents/contracts/ --max-files 20   # 超 20 个直接报错退出，防误喂大目录
uv run contract-archive ingest ~/Documents/contracts/ --progress ndjson  # 每文件一行 JSON 事件，供 agent 流式消费
```

### 查询

```bash
# 列出全部（按入库时间倒序，默认 50 条）
uv run contract-archive list

# 按签订日排序，只看 partial 的
uv run contract-archive list --order-by sign_date --status partial

# 输出 JSON 供脚本消费
uv run contract-archive list --format json | jq '.[] | .contract_name'

# 多字段过滤（全部 AND）
uv run contract-archive search --party 张三 --amount-min 100000 --signed-after 2024-01-01
uv run contract-archive search --expire-before 2026-12-31 --has-risk
uv run contract-archive search --name 车位 --auto-renewal

# 看单条详情（id 或 sha 前缀 ≥4 字符）
uv run contract-archive show 5
uv run contract-archive show a3f9c2b1

# 看原文：show 看 LLM 抽出的字段，raw 看抽取依据的 OCR 原始文本（同一份喂给 LLM 的内容）
# 交互终端下按抽取来源给命中关键字着色（当事人/金额/日期/风险/字段），一眼看出哪些被识别到
uv run contract-archive raw 5
uv run contract-archive raw a3f9c2b1 | grep 违约           # 管道时自动纯文本，不破坏 grep
uv run contract-archive raw 5 --color always | less -R     # 强制上色配 less -R
```

### 待办看板（义务清单）

每份合同抽取时会拆出双方"动作"（递交资料/付款/交付/签字等）作为
独立的 `obligations` 表，每条带 `actor` (甲方/乙方/双方) + `deadline`：

```bash
# 跨合同列出所有待办（按 deadline 升序，NULL 排最后）
contract-archive todo --include-undated

# 未来 30 天内要做的事
contract-archive todo --within-days 30

# 只看甲方任务 / 只看乙方任务
contract-archive todo --actor party_a
contract-archive todo --actor party_b --before 2026-12-31

# 找"近 30 天内有截止动作的合同"（不是单条 obligation，而是合同列）
contract-archive search --deadline-before 2026-06-30 --actor party_b
```

`contract-archive show <id>` 会按甲方/乙方/双方分组展示该合同所有义务，
与原本的 `risk_clauses`（违约罚则）严格区分。

### 身份核对（known_parties 基准库）

抽取时把每个主体（自然人/机构）与其固有标识（身份证号/电话/银行账号/开户行/税号…）
**精确绑定到人**（`person_identities`），不像扁平字段那样把多人号码混成一条。
入库时与 `known_parties` 基准库比对，采用「首见入库、再见校对」：

- **首见**：某主体的某标识第一次出现 → 录入为基准（记首见出处）。
- **再见**：同主体同标识再出现 → 与基准比对，不一致即在 `show` 的「身份核对」块报
  `identity` 缺陷（疑似 OCR 读错或信息被改），**不覆盖基准**。
- 比较前归一化剥离分隔符噪声（空格/；/：不误报），但多/少/错位的真实数字差异会被抓出。
- 不分自然人/机构——身份证、电话、银行账号、开户行一律核对。

```bash
contract-archive party list                       # 列出所有已知主体及标识
contract-archive party show 张三                   # 查看某主体的标识基准
contract-archive party set 张三 身份证号 1101...   # 手动修正基准（纠正被 OCR 读错的首见值）
contract-archive party rm 张三 电话                # 删除某标识；省略标识则删整个主体
```

> 基准库 `known_parties.json` 存档案库根目录，**含真实 PII**（身份证/电话/账号），
> 文件权限 0600、列入 `.gitignore`，绝不入库或分享。

### 抽取层管理

LLM 跑挂或想升级 prompt 后批量再抽取——不重跑 OCR：

```bash
uv run contract-archive extract 5            # 复跑 id=5 的抽取
uv run contract-archive extract 5 --no-llm   # 跳过 LLM（抽取字段留空，rule 已退役）
```

### 统计与维护

```bash
uv run contract-archive stats                # 总数 / status 分布 / 按月签订 / 近 30 天到期
uv run contract-archive delete 5             # 默认仅删 DB 行，交互确认
uv run contract-archive delete 5 --purge-files -y    # 同时删 archive/documents/<sha>/，无确认
uv run contract-archive vacuum               # 大批量 ingest 后整理碎片
```

> **注意**：`delete` 不会删用户原 PDF 文件——`source_path` 字段记录的是入库时
> 的源路径，源文件归用户所有。

### 印章总览

```bash
uv run contract-archive seals                      # 跨文档列全部印章
uv run contract-archive seals --seal-owner 示例公司   # 某主体的章（--owner 同义）
uv run contract-archive seals --seal-type 合同专用章   # 按印章类型（--type 同义）
```

### 机器发现 / agent 接入

把本 CLI 包成 MCP / OpenAI tool，或让 agent 自动调用时，用这几个命令免去硬编码——输出皆 JSON：

```bash
uv run contract-archive capabilities          # 全部命令 + 副作用/破坏性/幂等元数据
uv run contract-archive describe ingest       # 单命令参数 schema（名称/类型/必填/默认/可选值）
uv run contract-archive schema document        # 核心数据结构 JSON Schema（document/contract/confidence/error）
```

数据命令（list/search/show/stats/todo/seals/party/extract/ingest）都支持 `--format json`，
stdout 纯净可 `| jq`；失败结果带结构化 `error`（`code`/`category`/`retryable`），供 agent 判是否重试。

## ✦ 档案库目录结构

```
archive/
├── db.sqlite                     # 索引表
├── db.sqlite-wal / -shm          # WAL 模式产物（运行时）
├── ingest.jsonl                  # 总日志（每次 ingest 一行 JSON）
└── documents/
    └── a3f9c2b1/                 # sha256 前 12 位
        ├── source.pdf            # 硬链接源 PDF（跨盘 fallback copy）
        ├── ocr/
        │   ├── markdown.md
        │   ├── layout.json       # bbox 已归一到 PDF point
        │   ├── structured.json
        │   ├── raw_text.txt
        │   ├── pipeline_meta.json
        │   └── preview_images/
        ├── extraction_result.json    # 抽取字段（通用信封）
        ├── extraction_confidence.json
        └── ingest.log            # 单合同 stderr
```

## ✦ Docker

```bash
docker build -t contract-archive -f docker/Dockerfile .
docker run --rm -it \
  -v $PWD/archive:/app/archive \
  -v $PWD/input:/app/input \
  --env-file .env \
  contract-archive uv run contract-archive ingest /app/input
```

## ✦ 项目结构

```
contract-archive-cli/
├── pyproject.toml          # uv 依赖管理
├── docker/Dockerfile
├── .env.example
├── scripts/
│   └── setup.sh
├── contract_archive/
│   ├── cli.py              # 入口 main_entry + 写命令 ingest/extract/delete/vacuum + 组装
│   ├── cli_common.py       # app 实例 + 全局 callback + 参数 Enum + 双 console + 路径/ident 解析
│   ├── cli_query.py        # 只读命令 list/search/show/raw/stats/todo/seals
│   ├── cli_config.py       # config show/set/unset 子命令组
│   ├── cli_party.py        # party list/show/set/rm（known_parties PII 基准库）
│   ├── cli_introspect.py   # capabilities/describe/schema 机器发现命令
│   ├── cli_render.py       # 纯渲染层（Table / JSON dict / raw 高亮）
│   ├── schemas/            # pydantic schema（BBox/LayoutBlock/DocumentExtraction 等）
│   ├── pipelines/
│   │   ├── ocr_pipeline.py   # native text + VL OCR 混合提取 + markdown 清洗
│   │   └── vl_ocr.py         # qwen-vl-ocr 逐页 OCR 调用
│   ├── utils/
│   │   ├── pdf.py            # PyMuPDF 文本层分析 / 渲染
│   │   └── page_router.py    # 页级 text/ocr 分流
│   ├── extraction/              # 纯 LLM 抽取（rule/hybrid 自 Phase 2 退役）
│   │   ├── document_extractor.py  # 通用文档判类型 + 抽信封
│   │   ├── contract_extractor.py  # 合同专属字段（专属 prompt）
│   │   ├── llm_extractor.py        # DashScope OpenAI 兼容口调用
│   │   ├── vision_seal.py          # 落款页 VL 签章核查
│   │   ├── doc_type_handlers.py    # doc_type → 特化/后处理/融合配置
│   │   ├── fusion.py               # 多源候选评判 → field_verdicts sidecar
│   │   ├── text_fields.py / vl_extract.py / insurance_extractor.py
│   │   ├── normalize.py / amount_check.py / evidence_page_fix.py / property_fee.py
│   ├── archive/
│   │   ├── db.py                # SQLite 连接 + migrations 引擎
│   │   ├── repository.py        # DAO + 搜索查询构造
│   │   ├── ingest.py            # 入库流水线（hash → OCR → extract → rename → DB）
│   │   ├── party_registry.py    # known_parties 身份基准库
│   │   ├── paths.py             # 档案库路径约定 + 硬链接工具
│   │   └── migrations/          # 001_init … 005_completeness（5 个）
│   ├── errors.py               # 结构化错误模型（code/category/retryable）
│   ├── config.py               # XDG 配置 + env>file>default
├── archive/                # 档案库数据（gitignored）
├── docs/
├── evals/                  # 评测框架；真实数据在私有 evalset 仓库
├── input/                  # 用户放待处理 PDF
└── tests/
```

## ✦ 设计纪律

- **统一 schema**：OCR 产物统一写入 raw_text/markdown/structured/layout/meta；markdown 反斜杠转义在喂给抽取层前清洗
- **纯 LLM 抽取**：自 Phase 2 退役 rule/hybrid，字段全由 LLM 抽取；旧合同置信度列只保留 `llm`/`missing` 语义。死代码 rule 仅保留为确定性数值归一化（中文大写金额→数值、日期→ISO）
- **类型路由集中化**：先抽通用信封的 `doc_type`，再经 `doc_type_handlers.py` 决定特化抽取、类型专属后处理和是否启用融合。
- **融合只写 sidecar**：保险等高价值字段的多源融合结果只进入 `field_verdicts` / `fusion_overall_confidence`，不回写 `amounts` / `fields`。
- **API key 不出包**：仅从 env 读，日志不打印响应体
- **sha256 去重**：流式 hash 后查 UNIQUE 索引；命中即 skip 避免重复 OCR 后才发现重复
- **事务边界**：tmp 目录跑全 → `os.rename` 到 documents/ → DB INSERT；任一阶段失败回滚干净，DB 不留半成品
- **partial 状态可修复**：OCR OK 但 LLM 挂时 markdown 仍可用，`extract <id>` 命令只重跑抽取层
