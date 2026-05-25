# 调研：面向 LLM / Agent 的开源测评能力栈（2026-05）

目的：为「能否用更便宜更快的模型替换 qwen3.7-max」选型评测方案。结论先行——
**通用 agent benchmark（τ-bench / SWE-bench / GAIA）对本项目没用**，要的是**领域特定的
离线结构化抽取评测**；而主流框架（promptfoo 等）也不能无脑套用，原因见 §2。

---

## 1. 开源测评框架横评

| 框架 | 定位 | OpenAI兼容口(DashScope) | LLM-judge | 多模态 | 离线自托管 | License | JSON逐字段打分契合度 |
|---|---|---|---|---|---|---|---|
| **promptfoo** | 配置驱动轻量评测/红队 CLI | ✅ apiBaseUrl | ✅ llm-rubric/g-eval | ✅(provider层) | ✅ 100%本地 | MIT | 高 |
| **Inspect AI** (UK AISI) | 代码优先 solver/scorer | ✅ openai-api/* + BASE_URL | ✅ model-graded | ✅ | ✅ | MIT | 高 |
| **DeepEval** | pytest 式 LLM 单测 | ⚠️ 需包 DeepEvalBaseLLM | ✅ G-Eval | ✅ | ✅ | Apache-2.0 | 中高 |
| **OpenAI Evals** | OpenAI 自家注册表 | ⚠️ 绑 OpenAI 平台 | ✅ | 弱 | ⚠️ | MIT | 中低 |
| **lm-evaluation-harness** | 学术 benchmark 跑分器 | ✅ | 弱 | 弱 | ✅ | MIT | **低** |
| **LangSmith** | SaaS 可观测+评测 | ✅ | ✅ | ✅ | ⚠️ 仅企业版 | 商业 | 中 |
| **Braintrust** | SaaS 评测/CI 门禁 | ✅ | ✅ | ✅ | ⚠️ 混合架构需回连 | 商业 | 中 |

要点：
- **promptfoo**：一份 YAML 把多模型列成 providers 自动出 side-by-side 矩阵；断言体系完整
  （`is-json`+schema、`javascript`/`python` 自定义断言、`llm-rubric`/`g-eval`）；
  `PROMPTFOO_DISABLE_TELEMETRY=1` 真离线。是"快速跑多模型对比"的最轻栈。短板：Node/TS 工具链。
- **Inspect AI**：AISI 官方，代码优先，复杂多步打分逻辑比 YAML 顺手；纯本地、活跃。
- **DeepEval**：pytest 党友好，内置 `JsonCorrectnessMetric`/`GEval`；接 DashScope 要自己包一层。
- **lm-eval-harness**：few-shot 学术刷分器，打分范式是 loglikelihood/精确匹配，**不为
  "固定 JSON schema 逐字段语义打分"设计**，领域抽取评测用它=削足适履。
- **LangSmith / Braintrust**：能力强但商业闭源 SaaS，与"离线+轻量 Python"取向相左。

## 2. 为什么本项目选「薄自建」而非直接上 promptfoo

调研里 promptfoo 是最热推荐，但本项目有一个决定性约束：

> **要测的是 `extract_document()` 整条链路的产物，不是裸 LLM JSON。**

换模型在生产里就是改 `dashscope.model` 一个 config 字段。而 `extract_document` 在 LLM
返回后还做了大量**后处理**：`_coerce_*` 容错、`normalize_date`/`parse_money_value` 归一化、
`computed_total` 求和、`_coerce_completeness` 的"有缺项却标 complete 则纠正"、金额自洽校验。
这些是模型表现的**放大器或遮羞布**——便宜模型可能 JSON 字段错了但被后处理吃掉，也可能
`is_total_component` 标错导致合计算错。**只有测整条链才看得见生产真实事故。**

promptfoo / DeepEval 想在配置或 metric 里自己持有 prompt+调用，测的是裸 JSON，跳过后处理——
那测的不是生产实际产物。要让它们调真实链路，得写 python/exec provider 把本项目包成
HTTP/子进程，而**领域打分（嵌套 DocumentExtraction 的对齐、归一化比较）无论如何都得自己写**。
于是框架只剩"编排+查看器"的边角价值，却引入 Node 工具链分裂。

权衡后：**核心是一个操作 DocumentExtraction 的纯 Python 打分库 + 薄 runner**（复用生产归一化
函数当同一把尺子），决策用 gate（见 README）。promptfoo 留作"快速裸 JSON 探针"的可选外挂，
Inspect AI 是未来想要查看器/并行/分布式时的升级路径——但都不进核心，避免过度工程。

## 3. DashScope 候选模型对比（替换 qwen3.7-max）

> 查询日期 2026-05-25，价格**估算、以阿里云百炼控制台为准**。带小版本号的是当前在售命名；
> **无版本别名（qwen-plus 等）会随官方升级漂移，跑分务必锁 snapshot**。

### 文本抽取（替换 qwen3.7-max 的直接目标）

| 模型 | 定位 | 上下文 | 相对max能力 | 输入¥/M | 输出¥/M | JSON结构化 |
|---|---|---|---|---|---|---|
| qwen3.7-max（现用） | 旗舰 | 1M | 100% | ~2.5 | ~10 | ✅(非思考) |
| **qwen3.6-plus / qwen-plus** ⭐ | 均衡主力 | 1M | ~85-90% | ~0.8 | ~4.8 | ✅(非思考) |
| **qwen3.6-flash / qwen-flash** | 极速极廉 | 1M | ~70-78% | ~0.2 | ~2 | ✅(非思考) |
| qwen3-235b-a22b | 开源旗舰 | 256K | ~88%(非思考) | ~5(折RMB) | ~20 | ✅ |
| qwen-turbo | 老廉价款 | 1M | ~60% | ~0.3 | ~0.6 | ✅ |

**最可能接近 max 而成本砍到 ~1/5：`qwen-plus`**；极致降本测 `qwen-flash`（精度敏感字段大概率
掉点，须实测）；开源对照 `qwen3-235b-a22b`（关思考）。能力档百分比是定位估计、非官方分，
**最终以本评测集跑出的逐字段准确率为准**。

### 多模态（签章核查）

| 模型 | 定位 | 输入¥/M | 输出¥/M |
|---|---|---|---|
| qwen-vl-max（现用） | 上代最强 | ~5.6 | ~22.4 |
| **qwen3-vl-plus** ⭐ | VL 最强 | ~1.5 | ~4.5 |
| **qwen3-vl-flash** ⭐ | 均衡性价比 | ~0.35 | ~2.8 |

签章"有无章/签字"是低复杂度判定，**首选 qwen3-vl-flash**，精度不够再升 qwen3-vl-plus。

### JSON Mode 必须知道的坑

- **思考模式不支持** `response_format={"type":"json_object"}`——抽取链路务必关思考。
- 开 json_object 时**别设 max_tokens**，否则 JSON 可能被截断成非法串。
- prompt（system 或 user）里**必须出现 "JSON" 字样**，否则接口报错（本项目 prompt 已含）。
- `qwen-vl-max`/`qwen-vl-plus` 的 **latest/snapshot 版不支持** json_object，要用换 qwen3-vl 系列。
- 全部模型都可走 OpenAI 兼容口：`https://dashscope.aliyuncs.com/compatible-mode/v1`。

## 4. 评测方法学（指导本项目打分）

- **先分流再聚合**：能确定性比对的字段绝不交给 LLM-judge。
  - 确定性/归一化：doc_type 分类、ISO 日期（比归一化值）、键值字段。
  - **金额 exact**：数值+币种+is_total_component 整体匹配，错一个算 FN。**不用比值容差**
    （容差会放过量级错误，金额最不能错）。
  - 列表字段（当事人/金额/印章/issues/补充协议）：**贪心对齐→TP/FP/FN→P/R/F1**，
    干净处理多抽(FP)/漏抽(FN)；嵌套递归对齐。先贪心，证明有歧义再上匈牙利。
  - **完整性 issues**：漏报远比误报致命 → **F-beta(β≥2) 偏召回**；对齐 key=(类型,页码)；
    关键缺陷类型（签章/金额不一致）单设召回硬门槛 + 误报率上限（防"无脑全报"刷召回）。
  - **相对日期**（"上年度"→起止）：gold 存原文表述+reference date+期望，单拉 sub-metric，
    不混进 ISO 日期 F1（考的是推理不是抽取）。
- **LLM-judge 收窄**：只用于 title/summary 这类无唯一正解的开放文本。换**异家族**强模型当裁判，
  analytic rubric + 小刻度 + reasoning-before-score + 多采样取众数 + 盲测(匿名+随机顺序) +
  golden 子集校准。完整性核查有客观对错→走确定性对齐，不用 judge。
- **聚合 = gate 不是平均**：默认不可替换，候选须逐项非劣（bootstrap CI 下界 ≥ champion−δ），
  详见 README §决策框架。

## 来源（节选，查询于 2026-05）

- promptfoo 断言/离线/兼容口：promptfoo.dev/docs（expected-outputs、providers/openai、telemetry）
- Inspect AI：inspect.aisi.org.uk（providers、scorers）
- DeepEval：deepeval.com/docs（metrics-json-correctness、custom-llms）
- lm-eval-harness：github.com/EleutherAI/lm-evaluation-harness
- 阿里云百炼价格/JSON Mode/兼容口：help.aliyun.com/zh/model-studio（model-pricing、json-mode、
  compatibility-of-openai-with-dashscope、vision）
- 抽取评测（exact vs relaxed、numeric tolerance、consistency）：arxiv 2510.15727
- 列表对齐（Hungarian/greedy→TP/FP/FN）：Kuhn-Munkres / arxiv 2011.10881
- Ragas 可复用 metric（Factual Correctness=claim分解+P/R/F1）：docs.ragas.io
- LLM-judge 偏置与缓解（位置/冗长/自偏好+golden校准）：arxiv 2602.02219、arxiv 2410.21819
