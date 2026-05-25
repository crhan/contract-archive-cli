# Agent-Ready CLI 审查规范与实例

本文档分两部分：

- **Part A — 审查方法论 / Rubric**：一套**可复用**的「Agent-Ready CLI」审查标准。
  讲清楚**怎么审、审什么、每项的判定标准、证据怎么取、怎么打分**。
  以后对任何 CLI 做同类审查，直接套这一部分。
- **Part B — 首次应用实例**：把这套规范第一次应用到本仓库 `contract-archive-cli`
  的结果（评分 / 发现 / 改造计划）。它是 Part A 的「样例答卷」，会随代码演进而过期，
  Part A 才是长期资产。

审查的立场（一句话）：**不是审 CLI「能不能用」，而是审它能不能被 AI Agent / 自动化系统
稳定地发现、调用、推理、重试、恢复、审计**。三套思想做骨架：CLIG（human-first UX）、
JSON Schema（结构化输入输出）、MCP / Tool-Calling（discoverability / 副作用声明 /
机器可读 / 安全边界 / 幂等）。

---

# Part A — 审查方法论 / Rubric

## A.0 怎么用这份规范

1. **先定边界**：明确被审对象的入口（`pyproject.toml [project.scripts]` / `bin` /
   `console_scripts`）、所有子命令、所有产出通道（stdout / stderr / 文件 / 网络 / DB）。
2. **逐维度审**：A.2 列了 10 个维度。每个维度按统一模板走：
   `审查目标 → 如何审 → 判定标准 → 证据要求 → 满分参考`。
3. **取证不空谈**：每条结论必须挂 `file_path:line_number` + 关键代码片段。
   **没有可证伪的代码证据，不写结论**（只能写「疑似」并给出待验证实验）。
4. **打分**：用 A.1 的三轴评分 + 严重度分级。
5. **产出**：按 Part B 的结构出报告（Executive Summary / Detailed Findings /
   Future Architecture / Migration Plan）。

**审查者纪律**（避免常见翻车）：
- 一次只下一个判断，先列假设再用 grep / 跑命令验证，不要「看着像就是」。
- 区分「代码里没有」和「我没找到」——grep 不到要换关键词再确认，别急着判缺失。
- 给改进建议时永远问：**会不会破坏现有用法（human / 现有脚本）**？破坏性改动单独标注。
- 实用主义：对**单用户本地工具**，全套 async task daemon 往往是过度设计；
  对**多 agent 编排后端**，结构化错误和能力发现是刚需。**评分要结合定位**，不要一把尺子量到底。

## A.1 评分体系（打分锚点）

三条独立轴，各自 0–10。**Human CLI 高不代表 Agent-Ready 高**——这正是本审查的意义。

### 轴 1：整体成熟度（工程质量基线）
| 分段 | 锚点 |
|---|---|
| 8–10 | 有测试、有文档、错误处理完整、依赖干净、提交纪律好 |
| 5–7 | 主干可靠，局部有技术债（超长文件 / 死依赖 / 测试覆盖不均） |
| 2–4 | 能跑但脆弱，错误处理稀疏，无测试 |
| 0–1 | 玩具 / demo 质量 |

### 轴 2：Human CLI（CLIG 友好度）
检查点见维度 1/2/6。锚点：
| 分段 | 锚点 |
|---|---|
| 8–10 | stdout/stderr 分离、`--help` 带示例、尊重 `NO_COLOR`/TTY、危险操作有确认、错误信息可操作 |
| 5–7 | 基本可用，个别地方污染 stdout 或缺 `--help` 细节 |
| 2–4 | 输出混乱、颜色硬编码、无 `--help` 体系 |
| 0–1 | 反人类 |

### 轴 3：Agent-Ready（本审查重点）
这是承重轴。检查点贯穿全部 10 维度，但**以下 5 项为「承重墙」，缺一项扣 1.5–2 分**：
1. **结构化输出**：有稳定 `--json` / NDJSON，stdout 纯净可解析。
2. **结构化错误 + 退出码分类**：agent 能区分 `可重试 / 配置错 / 用户错 / 基础设施错`。
3. **能力 / schema 发现**：有 machine-readable 的命令/参数/输出描述。
4. **非交互安全**：不会在 CI/headless 卡 prompt；危险操作有 `--yes` 旁路 + isatty 守卫。
5. **幂等 / 可恢复**：重复执行安全，中断后能恢复或安全重跑。

| 分段 | 锚点 |
|---|---|
| 8–10 | 5 道承重墙齐全 + 长任务有进度/任务模型 + 有 trace/audit |
| 5–7 | 输出/幂等/非交互到位，但缺结构化错误或能力发现 |
| 2–4 | 只有零星 `--json`，错误是裸文本，无发现机制 |
| 0–1 | 纯人类终端工具，机器无从下手 |

### 严重度分级（每条 Finding 必须标）
判定锚点 = **「Agent 在自动调用时，这个缺陷会不会导致错误决策 / 烧钱 / 数据损坏 / 卡死」**。
| 级别 | 判定标准 | 典型 |
|---|---|---|
| **Critical** | Agent 会据此做出**错误且有害**的决策，或导致数据损坏/无限烧钱 | 错误不可区分导致无脑重试不可重试的操作；破坏性命令无确认且可被 LLM 参数注入 |
| **High** | 严重阻碍自动化，但有 workaround；或封装为 tool 的硬阻塞 | 无能力发现（wrapper 只能硬编码）；长任务全阻塞无进度 |
| **Medium** | 影响稳定性/成本/可维护性，agent 能勉强工作 | 无 dry-run/预算闸；输入无大小护栏；接口不统一 |
| **Low** | 卫生问题，不影响 agent 正确性 | 死依赖；超长文件；pretty-JSON 而非 NDJSON |

## A.2 十个审查维度（统一模板）

> 每个维度模板：**① 审查目标 ② 如何审（可操作动作）③ 判定标准（红绿灯 + 严重度）
> ④ 证据要求 ⑤ 满分参考**。

---

### 维度 1 — CLI 结构设计

**① 审查目标**：命令结构是否稳定、无歧义、可被 agent 推理和组合。坏例 `tool do <x>`，
好例 `tool project build`（noun-verb，语义自解释）。

**② 如何审**
- 列全部子命令：看 `@app.command` / `add_typer` / argparse subparser / `cobra.Command`。
- 检查命名一致性：动词时态统一？同义动词混用（`list` vs `show` vs `get`）？
- 检查歧义：一个命令是否做多件事？参数能否相互矛盾？
- 检查可组合：读命令能否 `| jq` 串联？输出能否喂回输入？

**③ 判定标准**
- 🟢 noun-verb 或扁平动词且语义清晰、无重载、命名一致。
- 🟡 个别命名不一致或单命令职责偏宽 → Low/Medium。
- 🔴 命令语义靠位置/标志重载切换、agent 无法静态推断要调哪个 → High。

**④ 证据要求**：命令清单表（命令 → 一句话职责 → 副作用类型）。

**⑤ 满分参考**：命令是「名词+动词」或清晰扁平动词；每个命令单一职责；
读写分明；help 里每命令一句话能说清。

---

### 维度 2 — Structured Output（结构化输出）

**① 审查目标**：是否有稳定、机器可解析的输出；数据与诊断是否分流。

**② 如何审**
- grep `--json` / `--format` / `--output` / `print_json` / `json.dumps` / `json.NewEncoder`。
- 确认 **stdout 只放数据，stderr 放进度/日志/错误**：grep `stderr` / `Console(stderr=True)` /
  `os.Stderr` / `eprintln`。看进度条、spinner、彩色 `rule` 是否误入 stdout。
- 跑一遍 `<cmd> --json | jq .` 和 `<cmd> --json 2>/dev/null | jq .`，看是否被 ANSI/日志污染。
- 检查空结果：空集合返回合法 `[]`/`{}` 还是什么都不打（破坏管道）？
- 检查输出是否带 **schema 版本**，字段是否稳定。

**③ 判定标准**
- 🟢 所有数据命令有 `--json`；stdout 纯净；空结果合法；尊重 `--no-color`/`NO_COLOR`。
- 🟡 有 `--json` 但 pretty-print 非 NDJSON、或无 schema 版本 → Low。
- 🔴 无机器可读输出，或 stdout 混入彩色/进度无法解析 → High。

**④ 证据要求**：每个命令的输出形状（指向序列化函数）；stdout/stderr 分流的代码位置。

**⑤ 满分参考**：数据走 stdout 且可 `--json`/NDJSON，诊断走 stderr，
输出有顶层信封 + `schema_version`，空结果合法，`NO_COLOR` 受尊重。

---

### 维度 3 — Schema / Capability Discovery（能力发现）

**① 审查目标**：机器能否**自动发现**有哪些命令、各吃什么参数、吐什么形状——
这是能否自动生成 MCP/OpenAI tool 定义的前提。

**② 如何审**
- grep `capabilit` / `describe` / `schema` / `introspect`（命令意义上的，排除注释）。
- 看是否能 `tool capabilities --json` / `tool describe <cmd> --json` / `tool schema <type> --json`。
- 看输出形状是否由可导出的类型定义（pydantic `model_json_schema()` / TypeScript 类型 /
  protobuf），还是散落在序列化函数里的隐式 dict。
- 看 capability 元数据是否声明 **side_effects / destructive / idempotent**。

**③ 判定标准**
- 🟢 有 `capabilities`/`describe`/`schema` 且含副作用元数据 + 参数 JSON Schema。
- 🟡 有类型定义可导出但无专门命令暴露 → Medium。
- 🔴 完全没有，wrapper 必须硬编码每个命令/参数/输出 → High。

**④ 证据要求**：introspection 命令位置；输出 schema 来源（类型定义 file:line）。

**⑤ 满分参考**：`capabilities --json` 自动遍历命令元数据生成，每命令含
`side_effects/destructive/idempotent/args_schema/output_schema`；`schema <type>` 直出 JSON Schema。

---

### 维度 4 — Agent Safety Model（安全模型）

**① 审查目标**：agent 自动调用时能否误删、误部署、越权、烧钱、无限执行。

**② 如何审**
- 列破坏性操作：grep `delete` / `rm` / `unlink` / `rmtree` / `DROP` / `DELETE FROM` /
  `TRUNCATE` / `overwrite` / `--force`。每个问：有无 confirm / dry-run / 预览 / 备份 / 回滚？
- 检查确认在非交互下的行为：`isatty` 守卫 + `--yes` 旁路，还是糊涂中止 / 直接执行？
- **路径注入**：写路径是否由**用户输入或 LLM 抽取的字段**拼接？grep `Path(` / `os.path.join` /
  `open(` 配合用户/模型来源变量。理想是路径由内容 hash / 固定规则派生。
- **外部副作用/成本**：grep 网络调用（`requests`/`httpx`/`.create(`/SDK），是否计费？
  有无速率/预算/次数闸？
- **密钥**：grep `api_key`/`API_KEY`/`getenv`/`.env`，是否会被打进日志/错误/traceback
  （检查 `pretty_exceptions_show_locals` 之类）？

**③ 判定标准**
- 🟢 破坏性操作有确认+isatty+`--yes`；路径不吃不可信输入；密钥默认掩码、不进日志。
- 🟡 有成本但无 dry-run/预算闸；破坏性操作有确认但无预览 → Medium。
- 🔴 破坏性操作无任何护栏、或路径吃 LLM 字段、或密钥会泄进输出 → Critical/High。

**④ 证据要求**：破坏性操作清单（操作 → 护栏现状 → file:line）；路径派生来源；密钥流向。

**⑤ 满分参考**：危险操作默认拒绝执行需显式 `--yes`，非 TTY 不交互；提供 `--dry-run`/预览；
路径由 hash/固定规则派生；成本有上限闸；密钥掩码且永不入日志。

---

### 维度 5 — Idempotency / Retry Safety（幂等与重试）

**① 审查目标**：命令能否安全重复执行；中断后是否留半成品；能否恢复。

**② 如何审**
- grep 去重机制：`hash`/`md5`/`sha`/`INSERT OR REPLACE`/`ON CONFLICT`/`exists`/`already`。
- 看「重复对同一输入执行」的代码路径：会重复写入还是 skip？
- 看失败时中间产物：是否落在临时区、成功后才原子 `rename`/`commit`？还是边写边污染最终区？
- grep `resume`/`checkpoint`/`cache`/`skip`：中断后能否复用已完成阶段？
- 多阶段任务：每阶段产物是否落盘可单独重跑（如「只重抽取不重 OCR」）？

**③ 判定标准**
- 🟢 内容寻址去重 + 临时区+原子提交 + 失败可安全重跑 + 阶段可单独重跑。
- 🟡 幂等但无阶段级恢复，必须整体重跑 → Medium。
- 🔴 重复执行会重复写/损坏数据，或半成品污染最终区 → High/Critical。

**④ 证据要求**：去重判定 file:line；事务/原子边界位置；阶段恢复入口。

**⑤ 满分参考**：sha/内容 hash 去重；`tmp → rename` 原子边界；失败状态可自动重试；
提供「只重某阶段」的命令。

---

### 维度 6 — Non-Interactive Execution（非交互执行）

**① 审查目标**：CI / headless / agent 环境下会不会卡在 prompt。

**② 如何审**
- grep `confirm`/`prompt`/`input(`/`Confirm.ask`/`Prompt.ask`/`sys.stdin`/`isatty`/`scanln`。
- 每个交互点问：非 TTY 下行为？是 EOF 崩溃、糊涂中止，还是要求显式 `--yes`/`--no-input`？
- 是否有全局 `--no-input`/`--yes`/`--force`？是否尊重 `CI` 环境变量？

**③ 判定标准**
- 🟢 无交互或交互点都有 isatty 守卫 + 显式旁路标志。
- 🟡 个别交互点在非 TTY 下行为不明 → Medium。
- 🔴 关键路径会阻塞在 prompt 等输入 → High（对 agent 是致命卡死）。

**④ 证据要求**：所有交互点清单 + 非交互行为 + 旁路标志。

**⑤ 满分参考**：默认非交互友好；危险操作非 TTY 下要求 `--yes` 否则明确报错退出（非挂起）。

---

### 维度 7 — Error Model（错误模型）

**① 审查目标**：错误是否结构化、可分类、机器可据此决策（尤其重试与否）。

**② 如何审**
- grep `raise .*Exit`/`sys.exit`/`os.Exit`/`return.*err`：统计退出码种类。是否只有 0/1？
- 看错误如何呈现：自由文本 `str(e)` 拼接，还是带 `code`/`category`/`retryable` 的结构？
- 检查能否区分 5 类：**user / validation / transient(可重试) / permission / infra**。
- grep 空 catch / 吞异常：`except.*pass`/`except Exception`/`catch {}`，关键路径是否静默吞错？
- 看外部调用（API）异常是否映射到分类（如 429→transient/retryable）。

**③ 判定标准**
- 🟢 退出码分类 + JSON 错误含 `code/category/retryable`；外部错误映射到分类。
- 🟡 退出码 0/1 但 JSON 有部分结构化字段 → Medium。
- 🔴 错误全是裸文本、退出码只有 0/1，agent 只能正则匹配错误串 → **Critical**（重试决策无依据）。

**④ 证据要求**：退出码使用统计；错误对象结构 file:line；外部异常映射点。

**⑤ 满分参考**：稳定退出码小表（≤8 个，分类清晰）+ 结构化 error 信封含 `retryable`，
批量结果逐条带 error 分类。

```json
{ "error": { "code": "RATE_LIMITED", "category": "transient",
             "message": "...", "retryable": true, "retry_after_s": 30 } }
```

---

### 维度 8 — Long-Running Workflow（长流程编排）

**① 审查目标**：CLI 能否作为 agent orchestration 后端——长任务可发起、查询、流式、取消、恢复。

**② 如何审**
- 找最慢命令，估算耗时构成（外部 OCR/LLM/编译）。是同步阻塞还是异步？
- grep `task`/`job`/`status`/`async`/`--follow`/`--watch`：有无 task id / 状态查询 /
  日志流 / 取消 / resume？
- 看进度反馈：是给人看的彩色进度条，还是机器可读的 NDJSON 事件流？
- 批量：单项失败会中断整批吗？批量进度可观测吗？

**③ 判定标准**（**强烈结合定位**：单用户本地工具不强求 task daemon）
- 🟢（编排后端定位）有 task id + status + logs + cancel + resume。
- 🟢（本地工具定位）至少 NDJSON 流式进度 + sha 幂等可重跑。
- 🟡 长任务全阻塞，进度只给人看（stderr 彩色） → High（对 agent）/ Low（对人）。
- 🔴 长任务无任何进度、无法取消、中断后只能从头来 → High。

**④ 证据要求**：最慢命令耗时构成；进度通道；批量失败处理 file:line。

**⑤ 满分参考**：长任务支持 `--progress ndjson` 流式事件；可选 task 模型（start/status/logs/cancel）；
批量单项失败不中断，逐项可观测。

---

### 维度 9 — Observability（可观测性）

**① 审查目标**：能否追踪、审计、复现一次执行——agent 系统要把 CLI 动作绑回编排步骤。

**② 如何审**
- grep `trace`/`correlation`/`request_id`：有无可透传的关联 id（`--trace-id`）？
- 看日志：`logging`/结构化（jsonl）还是 `print`？有无统一审计落点（谁/何时/做了什么）？
- 变更操作（delete/deploy/write）是否有审计记录？
- 可复现：执行用的模型/版本/参数是否记录进产物（单一真相源）？

**③ 判定标准**
- 🟢 `--trace-id` 透传 + 结构化日志 + 变更审计 + 执行元数据落产物。
- 🟡 有部分结构化日志但无 trace id、审计只覆盖部分命令 → Medium。
- 🔴 只有 `print`，无审计、无元数据 → Medium/High。

**④ 证据要求**：日志机制 file:line；审计落点；执行元数据记录点。

**⑤ 满分参考**：全局 `--trace-id` 进日志与产物；变更走统一 audit jsonl；
产物记录模型/版本/参数。

---

### 维度 10 — MCP Compatibility（MCP / Tool-Calling 兼容潜力）

**① 审查目标**：当前 CLI 能否低成本包装为 MCP Server / OpenAI Tool / Claude Tool / Workflow Node。

**② 如何审**（这是 1–9 的综合）
- 读命令能否 1:1 映射为 tool（依赖维度 2 的 `--json` + stdout 纯净）？
- tool 定义能否自动生成（依赖维度 3 的能力发现）？
- 错误能否映射为规范 tool error（依赖维度 7）？
- 长任务能否塞进 request/response 模型（依赖维度 8）？
- 输出形状是否有版本（依赖维度 2 的 schema_version）？

**③ 判定标准**
- 🟢 读命令可自动生成 tool；错误结构化；长任务有流式/任务模型。
- 🟡 读命令可手工封装，但要硬编码 schema、错误要正则解析 → High 阻塞。
- 🔴 输出不可解析、无 schema、错误裸文本 → 无法可靠封装。

**④ 证据要求**：逐条列出封装阻塞点 + 对应根因维度。

**⑤ 满分参考**：`capabilities` + `schema` + 结构化错误三者就位后，读命令自动生成 MCP tools，
写/长命令走 NDJSON 流或 task 模型。

---

## A.3 审查产出模板（套这个写报告）

```
# Executive Summary
  - 三轴评分（整体成熟度 / Human CLI / Agent-Ready）+ 一句话锚点理由
  - 最大架构问题（一句话）
  - 最优先 3 项改进（投入产出比排序，标明是否破坏性）

# Detailed Findings   （按 Critical/High/Medium/Low 分组）
  每条：问题 / 原因 / Agent 场景影响 / 推荐方案 / 示例(命令或 schema) / 证据(file:line)

# Proposed Future Architecture
  - 推荐 command namespace / JSON 信封 / capability schema / safety model / async model

# Migration Plan
  - Phase 1 非破坏增量 / Phase 2 半破坏(需公告) / Phase 3 投机性(按需)
  - 每项标注：是否 breaking change
```

---
---

# Part B — 首次应用实例：contract-archive-cli

> 被审对象：本仓库 `contract-archive`（typer CLI，入口 `contract_archive.cli:app`）。
> 审查时间：首次。下文是 Part A 规范在此 CLI 上的一次完整应用，作为样例答卷。

## B.1 Executive Summary

| 轴 | 评分 | 锚点理由 |
|---|---|---|
| 整体成熟度 | **7.5 / 10** | 有测试、迁移、配置体系、提交纪律好；扣分于 `cli.py` 1050 行超红线、死依赖 |
| Human CLI | **8.5 / 10** | 双 console 分流、`--format`、`--no-color`、isatty 守卫、help 带示例 |
| Agent-Ready | **5 / 10** | 输出/幂等/非交互三道承重墙在；缺结构化错误、能力发现两道墙 |

**最大架构问题**：把「给人看」做对了，但**没把「给机器决策」做对**——
错误是自由文本、退出码只有 0/1、输出形状无 schema 无版本、零 introspection 命令。
Agent 调用时**拿不到据以决策的信号**（该重试？该改配置？该放弃？无从判断）。

**最优先三项**（均非破坏性，纯增量）：
1. **结构化错误模型 + 退出码分类**（Critical，维度 7）。
2. **能力 / schema 发现命令**（High，维度 3）。
3. **ingest 的 NDJSON 流式进度 + dry-run**（High，维度 8 / 维度 4 成本闸）。

## B.2 Detailed Findings

### 🔴 Critical

#### C1 — 无结构化错误模型，退出码只有 0/1（维度 7）
- **问题**：错误靠 `f"mineru: {e}"`（`contract_archive/archive/ingest.py:199`）、
  `f"extract: {e}"`（`ingest.py:249`）拼字符串；退出码仅 `Exit(0)/Exit(1)`，
  所有失败场景同为 `1`（`cli.py:318/542/769/845/854`）。
- **原因**：错误处理是「打印给人看」思路（`cli.py:292`），降级藏在 `status` 字符串
  （`failed`/`partial`），无 `category`/`retryable`。
- **Agent 影响**：DashScope `429 限流`（应退避重试）与「API key 没配」（重试无用，应停下）
  对 agent 不可区分，除非正则匹配 `str(e)`——供应商一改措辞就崩。**重试决策无依据**。
- **方案**：失败结果带结构化 error；退出码分类小表（见 B.3）；外部异常在
  `extraction/llm_extractor.py:111` 处映射（`openai.RateLimitError→TRANSIENT/retryable`，
  `AuthenticationError→CONFIG`）。
- **示例**：
  ```json
  {"status":"failed","sha256":"a1b2..","error":{"code":"RATE_LIMITED",
   "category":"transient","message":"DashScope 429","retryable":true,"retry_after_s":30}}
  ```

### 🟠 High

#### H1 — 无能力 / schema 发现机制（维度 3）
- **问题**：无 `capabilities`/`describe`/`schema` 命令；JSON 输出形状由
  `cli_render.row_to_dict`（`cli_render.py:107-144`）隐式定义，30+ 字段，无 schema 无版本。
- **原因**：从「人类 CLI」长出，没把机器消费方当一等公民。`schemas/document.py` 已有
  pydantic 模型（`model_json_schema()` 一行可出 schema），但未暴露。
- **Agent 影响**：封装 MCP/OpenAI tool 必须人肉抄每个命令/参数/枚举（如 `OrderBy`
  `cli.py:79-87`）；CLI 一加枚举值，所有 wrapper 手动同步。
- **方案**：`capabilities --json`（遍历 `app.registered_commands` 自动生成，含
  `side_effects/destructive/idempotent`）+ `describe <cmd> --json` + `schema <type> --json`。

#### H2 — ingest 长任务全阻塞，进度只给人看（维度 8）
- **问题**：`ingest` 对目录全同步串行（`cli.py:280-303`），单文件 MinerU 5–60s + LLM 2–10s；
  进度是 `err_console.rule(...)`（`cli.py:281`，stderr 彩色，给人看）；机器可读 JSON 要等
  整批跑完才一次性吐（`cli.py:312-316`）。无 task id / status / cancel。
- **Agent 影响**：发起可能跑 1 小时的调用，期间无法 poll 进度、无法判断卡在第几个、
  无法 graceful cancel（只能 SIGKILL，靠 sha 幂等救回）。
- **方案**（先做便宜档）：`ingest --progress ndjson`，每处理完一个文件吐一行 JSON 事件，
  末行 summary（非破坏，现有 `--format json` 不动）。完整 task 模型**暂缓**（单用户本地工具属过度设计）。
  ```jsonl
  {"event":"file_done","seq":1,"total":200,"sha256":"a1b2","status":"ok","doc_id":12}
  {"event":"summary","ok":1,"partial":0,"failed":0,"skipped":0}
  ```

### 🟡 Medium

#### M1 — 烧钱副作用未声明，ingest 无 dry-run / 预算闸（维度 4）
- **问题**：每 ingest 一个 PDF 调两次付费 API（`llm_extractor.py:158` 文本 +
  `vision_seal.py:114` 签章）；有 `--no-llm`/`--limit` 但无 dry-run、无 token/次数上限。
- **Agent 影响**：误喂大目录或循环 ingest → 直接烧钱 + 触发限流，无护栏。
- **方案**：`ingest --dry-run`（不调 API，报「将处理 N 个、去重后 M 新、预计 2M 次调用」）+
  `--max-files`；`capabilities` 里把 `cost` 标进 `side_effects`。

#### M2 — 输入仅校验后缀，无大小 / 超时护栏（维度 4）
- **问题**：`ingest` 校验 `exists/readable`（`cli.py:240`）+ `.pdf`（`ingest.py:565`），
  但不查文件大小；畸形/超大 PDF 可能让 MinerU OOM 或挂死。
- **方案**：`--max-file-mb` + MinerU subprocess `--timeout-s` 透出，超时归类 `INFRA/TRANSIENT`。

#### M3 — evals 是另一套接口（argparse `python -m`），且无去重（维度 1 / 5）
- **问题**：主 CLI 是 typer，评测线是 `python -m evals.run`（argparse）。`evals.run` 无
  `(model,case)` 去重，重跑 append 重复行，`report` 只按 `repeat_idx` 取首条。
- **方案**：不急合并；先给 `evals.run` 加去重对齐主线幂等；长期可收编为 `contract-archive eval run`。

#### M4 — 无 trace id，变更操作无统一审计（维度 9）
- **问题**：只有 `ingest` 写 `ingest.jsonl`（`ingest.py:332-344`）；delete/vacuum/extract 无审计；
  日志无 trace id（`cli.py:186-189`）。
- **方案**：全局 `--trace-id` 透传；delete/extract 写统一 audit jsonl。

### 🟢 Low

- **L1 死依赖**：`fastapi`/`uvicorn[standard]`/`python-multipart` 在 base deps 但全仓库零引用
  （`pyproject.toml`）。删，或真建 server。
- **L2 pretty-JSON 非 NDJSON + 无顶层信封**：`--format json` 都 `indent=2`；建议加
  `schema_version` 顶层信封（走新选项，不破坏旧形状）。
- **L3 `cli.py` 1050 行超红线**：项目自定 1000 行/文件、50 行/函数；`show` 函数近 190 行
  （`cli.py:524-711`）。渲染逻辑应下沉到 `cli_render.py`。
- **L4 VACUUM 无确认**（`cli.py:1042`）：只压缩不毁数据，风险低，不强求。

### ✅ 做对的地方（不要动）
- 双 console 分流（`cli.py:119-120`）；空库返回合法 `[]`/`{}`（`cli.py:219-231`）。
- delete 的 `--yes` + isatty 守卫（`cli.py:849-854`）——非交互安全已达标。
- sha256 幂等 + `ON CONFLICT DO NOTHING` + 失败自动重试（`ingest.py:149-169`）。
- 路径由 sha 派生、不吃 LLM 字段；参数化 SQL + `order_by` 白名单。
- `pretty_exceptions_show_locals=False` 防密钥泄进 traceback（`cli.py:134`）。

## B.3 Proposed Future Architecture

**命令 namespace**：保留现有扁平动词（单用户工具不必强行 noun-verb），仅补 introspection：
```
# 新增（给机器）
contract-archive capabilities --json          # 命令清单 + 副作用元数据
contract-archive describe <command> --json    # 单命令参数 schema
contract-archive schema <document|ingest_result> --json
```

**退出码小表**：
| code | 含义 | retryable |
|---|---|---|
| 0 | OK | — |
| 2 | USAGE（typer 已用，保留） | 否 |
| 3 | NOT_FOUND | 否 |
| 4 | CONFIG（缺 API key） | 否 |
| 5 | PARTIAL（批量部分失败） | 看情况 |
| 7 | TRANSIENT（限流/网络） | 是 |
| 8 | INFRA（MinerU 崩/DB 锁） | 看情况 |

退出码粗判，JSON 里 `retryable` 细判（批量可能混合多种结果）。

**JSON 信封**（走 `--envelope`/`json2` 过渡，保留旧形状一个周期）：
```json
{ "schema_version": "1", "ok": true, "data": [ ... ], "error": null }
```

**安全模型矩阵**：
| 命令 | side_effects | destructive | idempotent | 护栏 |
|---|---|---|---|---|
| list/search/show/stats/seals/todo | read | 否 | 是 | 无需 |
| ingest | fs_write, network, **cost** | 否 | 是(sha) | dry-run/max-files/max-api-calls |
| extract | network, cost, db_write | 否 | 是 | cost 闸 |
| delete | fs_write, db_write | **是** | 是 | 已有 `--yes`+isatty ✅ |

**async 模型**：先 NDJSON 流式进度；task daemon 暂不做（无并发编排需求前属臆想）。

## B.4 Migration Plan

**Phase 1 — 非破坏增量**（先做，纯增，零影响现有用法）
1. 结构化错误对象（C1）：`IngestResult` 加 `error` 字段，保留旧 `error_message`。
2. `capabilities`/`describe`/`schema` 命令（H1）：新文件 `cli_introspect.py`。
3. `ingest --progress ndjson`（H2）。
4. `ingest --dry-run` + `--max-files`（M1）。
5. 删死依赖（L1）。
6. 拆 `cli.py`（L3）：渲染下沉 `cli_render.py`，降至 1000 行内（有 `tests/test_cli_render.py` 兜底）。

**Phase 2 — 半破坏（需公告 CHANGELOG）**
7. 退出码分类（C1）：`ingest` 部分失败 `1→5`，NOT_FOUND→`3`。**改变脚本退出码判断，需灰度**。
8. `--trace-id` + 统一 audit jsonl（M4）。
9. evals 去重（M3）。

**Phase 3 — 投机性（按需，可能永不做）**
10. JSON 信封 + schema_version（L2，走新选项）。
11. 完整 task 模型（H2 重档）——无 orchestrator 接入前不做。
12. MCP server 封装——Phase 1 就位后读命令可自动生成 tool。

## B.5 MCP 兼容结论
读命令（list/search/show/stats/todo/seals）**现在就接近可封装**（有 `--json` + stdout 纯净 +
空库合法）。四个封装阻塞，全在 Phase 1 解决：① 无 schema（H1）② 错误裸文本无法判重试（C1）
③ ingest 长阻塞进度不可解析（H2）④ 输出无版本（L2）。**做完 Phase 1，即从「subprocess 硬包」
升级到「可自动生成 MCP/OpenAI tool 定义」**。
