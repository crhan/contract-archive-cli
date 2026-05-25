# 本地文档档案库 CLI

> 把各类文档 PDF 批量入库、归档、可追溯——合同、协议、证明、发票、报告……
> MinerU 解析版面文本，qwen3.7-max LLM 判类型 + 抽字段，索引到本地 SQLite，
> 支持按类型/字段过滤检索。

**LLM-first**：文档类型与字段抽取都交给 LLM，统一归一化到一个「通用信封」
（doc_type / title / summary / 主体 / 金额 / 日期 / 柔性字段）。加新文档类型
**无需写代码**——LLM 自行决定抽什么。死代码 rule 仅保留为确定性数值归一化
（中文大写金额→数值、日期→ISO）。合同另有专属抽取层（rule+LLM hybrid 交叉验证），
仍保留全部合同字段（甲乙方/到期日/自动续约/风险条款/义务清单）。

历史：本项目最初是 DashScope / PaddleOCR / MinerU 三路 OCR 对比 playground，
选定 MinerU 后重构为档案库 CLI；再从「合同专用」扩展为「通用文档档案库」。

## ✦ 数据流

```
PDF ─► sha256 去重 ─► MinerU 解析 ─► LLM 判类型 + 抽字段（合同再走 hybrid）
                                          │
              ┌───────────────────────────┴──┐
              ▼                              ▼
   db.sqlite (通用信封 + 索引)     documents/<sha-12>/
                                    ├── source.pdf  (硬链接)
                                    ├── mineru/markdown.md ...
                                    ├── extraction_result.json  (通用信封)
                                    └── ingest.log

档案库默认在 XDG 数据目录：~/.local/share/contract-archive/
```

## ✦ 安装

```bash
# 1) 装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 装依赖（mineru extras 会拉 MinerU 包，首次跑 ingest 还会从 modelscope 下模型 >1GB）
./scripts/setup.sh mineru
```

> **uv hardlink 坑**：uv 默认 `UV_LINK_MODE=hardlink` 偶发只装包的一部分文件
> （实测 `cv2`/`pptx` 会丢，触发 `module 'cv2' has no attribute 'INTER_NEAREST'`
> 或 `cannot import name 'Presentation' from 'pptx'`）。`scripts/setup.sh` 已
> 显式 `export UV_LINK_MODE=copy` 规避。手动 `uv sync` 时建议也带上。
> 已损坏的包可以 `uv pip install --force-reinstall --no-deps <包名>` 修。

如果只想用 list/search/show 查已有的档案库（机器上有别的人 ingest 好的 db.sqlite），
跳过 mineru extras 也可以：

```bash
./scripts/setup.sh base
```

## ✦ 全局安装（可选）

如果想在任意目录用 `contract-archive`（不必 `cd` 项目目录或 `uv run`），用 `uv tool install`：

```bash
UV_LINK_MODE=copy uv tool install --reinstall --with mineru --force /path/to/contract-archive-cli
```

`uv tool install` 会在 `~/.local/bin/contract-archive` 装独立 venv（与项目 venv 隔离）。
然后从任意目录：

```bash
# 用环境变量指定档案库
CONTRACT_ARCHIVE_DIR=~/contracts contract-archive list

# 或显式 --archive
contract-archive --archive ~/contracts list
contract-archive --archive ~/contracts ingest ~/Documents/new_contract.pdf
```

`DASHSCOPE_API_KEY` 需通过 shell env 提供（建议放进 `~/.zshrc` 或专用 shell wrapper）。

## ✦ 配置

```bash
cp .env.example .env
$EDITOR .env   # 填入 DASHSCOPE_API_KEY
```

| 字段 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 必填。[百炼控制台](https://dashscope.console.aliyun.com/) 申请 |
| `DASHSCOPE_LLM_MODEL` | 默认 `qwen3.7-max`（用户百炼账户的特定别名；若 404 换 `qwen-max` / `qwen3-max`） |
| `CONTRACT_ARCHIVE_DIR` | 档案库根目录，默认 XDG `~/.local/share/contract-archive`；CLI `--archive` 优先 |
| `COMPUTE_DEVICE` | `auto` / `mps` / `cuda` / `cpu`（MinerU 走子进程，主要影响其内部 backend 选择） |

## ✦ 用法

```bash
# 入库单个 PDF
uv run contract-archive ingest path/to/合同.pdf

# 批量入库整个目录（递归扫 *.pdf，sha256 去重）
uv run contract-archive ingest ~/Documents/contracts/

# 调试：只跑 rule 抽取跳过 LLM（不需要 API key）
uv run contract-archive ingest path/to/合同.pdf --no-llm

# 强制重跑（已 ingest 过的也再跑一遍，覆盖旧记录）
uv run contract-archive ingest path/to/合同.pdf --reingest

# 试跑前 3 个
uv run contract-archive ingest ~/Documents/contracts/ --limit 3
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

### 抽取层管理

LLM 跑挂或想升级 prompt 后批量再抽取——不重跑 MinerU：

```bash
uv run contract-archive extract 5            # 复跑 id=5 的抽取
uv run contract-archive extract 5 --no-llm   # 只跑 rule
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

## ✦ 档案库目录结构

```
archive/
├── db.sqlite                     # 索引表
├── db.sqlite-wal / -shm          # WAL 模式产物（运行时）
├── ingest.jsonl                  # 总日志（每次 ingest 一行 JSON）
└── documents/
    └── a3f9c2b1/                 # sha256 前 12 位
        ├── source.pdf            # 硬链接源 PDF（跨盘 fallback copy）
        ├── mineru/
        │   ├── markdown.md
        │   ├── layout.json       # bbox 已归一到 PDF point
        │   ├── structured.json
        │   ├── raw_text.txt
        │   ├── pipeline_meta.json
        │   └── preview_images/
        ├── extracted.json        # 抽取字段
        ├── extraction_confidence.json
        └── ingest.log            # 单合同 stderr
```

## ✦ Docker

```bash
docker build -t contract-archive -f docker/Dockerfile .
docker run --rm -it \
  -v $PWD/archive:/app/archive \
  -v $PWD/input:/app/input \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  --env-file .env \
  contract-archive uv run contract-archive ingest /app/input
```

挂载 modelscope 缓存复用本机 MinerU 模型。Mac 容器不直通 GPU，强烈推荐 native venv 跑。

## ✦ 项目结构

```
contract-archive-cli/
├── pyproject.toml          # uv 依赖管理（extras: mineru）
├── docker/Dockerfile
├── .env.example
├── scripts/
│   ├── setup.sh
│   └── run_sample.sh
├── contract_archive/
│   ├── cli.py              # contract-archive 入口
│   ├── schemas/            # pydantic schema（BBox/LayoutBlock/ContractExtraction 等）
│   ├── pipelines/
│   │   └── mineru_pipeline.py   # MinerU subprocess 调用 + 坐标归一化 + markdown 清洗
│   ├── extraction/
│   │   ├── rule_extractor.py    # 正则抽取
│   │   ├── llm_extractor.py     # qwen3.7-max
│   │   └── hybrid.py            # rule + LLM 字段级合并
│   ├── archive/
│   │   ├── db.py                # SQLite 连接 + migrations 引擎
│   │   ├── repository.py        # DAO + 搜索查询构造
│   │   ├── ingest.py            # 入库流水线（hash → MinerU → extract → rename → DB）
│   │   ├── paths.py             # 档案库路径约定 + 硬链接工具
│   │   └── migrations/001_init.sql
│   └── utils/                   # 设备选择 / PyMuPDF PDF 渲染
├── archive/                # 档案库数据（gitignored）
├── input/                  # 用户放待处理 PDF
└── output.legacy/          # 旧 pipeline 历史产物（重构前的对比数据，可删）
```

## ✦ 设计纪律

- **统一 schema**：MinerU 的 0-1000 归一化坐标全部反算成 PDF point；markdown 反斜杠转义在喂给抽取层前清洗
- **rule + LLM hybrid**：字段级合并策略（金额/日期信 rule 原文证据，实体名信 LLM 规整），每字段附 `value_source`（rule/llm/merged/missing）+ 置信度
- **API key 不出包**：仅从 env 读，日志不打印响应体
- **sha256 去重**：流式 hash 后查 UNIQUE 索引；命中即 skip 避免 MinerU 跑一次几分钟才发现重复
- **事务边界**：tmp 目录跑全 → `os.rename` 到 documents/ → DB INSERT；任一阶段失败回滚干净，DB 不留半成品
- **partial 状态可修复**：MinerU OK 但 LLM 挂时 markdown 仍可用，`extract <id>` 命令只重跑抽取层
- **不并发 MinerU**：每个 subprocess 会加载 GB 级模型，并发反而 OOM；默认 workers=1
