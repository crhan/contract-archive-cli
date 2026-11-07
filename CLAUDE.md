# contract-archive-cli — Agent 须知

记录本项目里容易踩的坑与约定。遇到反直觉的地方，更新这里，帮后来的 agent 少走弯路。

## DashScope 平台接口：优先用 OpenAI 兼容接口

调用阿里云百炼（DashScope）平台的模型，**一律走 OpenAI 兼容接口**
（`base_url` 用 `https://dashscope.aliyuncs.com/compatible-mode/v1`，`openai` SDK 的
`chat.completions.create`），**不用原生 `dashscope.Generation.call`**。

为什么：
- **原生端点不认部分模型 id**：实测 `qwen3.6-flash` 经原生 `/api/v1` 报
  `400 InvalidParameter: url error`，而经 `/compatible-mode/v1` 正常。第三方托管模型
  （`deepseek-v4-*`、`glm-5.1` 等）也只在兼容口稳定。
- **统一一条 transport**：VL 签章线本来就走兼容口；文本线也统一过去后，全项目一个 SDK、
  一种响应结构，少一类"原生 vs 兼容"的隐藏分叉。
- **可移植**：OpenAI 标准接口，换供应商/自建网关成本低。

实践要点（JSON 抽取场景）：
- `base_url` 由配置的 `/api/v1` 做 `.replace("/api/v1", "/compatible-mode/v1")` 得到。
- 开 `response_format={"type":"json_object"}`；**别设 max_tokens**（否则 JSON 可能被截断成非法串）；
  prompt 里必须出现 "JSON" 字样；**关思考模式**（思考模型不支持 json_object）。
- **关思考靠显式传 `extra_body=NO_THINKING`**（`utils.llm.NO_THINKING`），新增任何
  `chat.completions.create` 都必须带上，别就地写字面量。新一代模型（qwen3.8-max 起）
  **默认开思考**：实测同份保单 qwen3.7-max 87s，qwen3.8-max 默认态 897s（两次 300s 超时
  重试）。漏传不会报错，只会静默退化成思考态——评测跑出来的就是退化态，选型结论被污染。
- usage 从 `resp.usage`（`prompt_tokens`/`completion_tokens`/`total_tokens`）读，归一化成
  `input_tokens`/`output_tokens`/`total_tokens`。

## OCR 模型 ID：默认稳定 ID，不做静默替换

OCR 默认值是 `qwen-vl-ocr`。当前项目验证的账号与 OpenAI 兼容端点中，
`qwen-vl-ocr-latest` 返回 `404 model_not_found`，而 `qwen-vl-ocr` 正常完成请求；官方
[Qwen-VL-OCR 文档](https://help.aliyun.com/zh/model-studio/qwen-vl-ocr)列出稳定 ID
`qwen-vl-ocr`。这只是当前环境的兼容性证据，不得推断 `-latest` 已在所有区域或账号下线，
也不是模型质量升级结论。

用户显式设置的 `DASHSCOPE_OCR_MODEL` 或 `dashscope.ocr_model` 必须原样使用：不要因失败
自动重写为默认值、隐藏 fallback，或改变评测候选标签。默认值只适用于没有显式配置的新设置。

## 文档类型路由：doc_type → handler 映射，别再散落 if doc_type==

本项目处理**所有** PDF 类型（合同/保险/证明/发票/旅行/证件…），不只合同。流程是
**先识别 doc_type（通用信封 `extract_document`）→ 据类型走特化**。类型特化**集中在**
`extraction/doc_type_handlers.py` 的 `DOC_TYPE_HANDLERS` 映射，**不要**在 ingest/各处再写
`if doc_type == "合同协议"`。

加新类型 = 注册一条 `DocTypeHandler`：
- `specialized_extractor`：第二层特化抽取（合同→`extract_contract`，就地 enrich 信封）。
- `post_processors`：类型专属后处理（合同=看落款页签章核查）。签名 `(envelope, OCR_dir)->bool`。
- `enable_vision_fusion` + `vision_fusion_fields`：是否开多源融合 + 高价值概念键定义（保险开）。

通用后处理（页码校正 `correct_evidence_pages` / 身份核对 `PartyRegistry`）**类型无关**，留在
ingest，不进 handler。`document_extractor` 里"completeness/金额自洽仅合同"是信封构造的内聚领域
逻辑（与 `computed_total` 勾稽），**留原处不外移**。

## 多源融合（保险首个 case）：A 文本 / C 看图两路评判，sidecar 绝不回写原字段

复杂表格/混合版式下单源抽取系统性丢数据。融合让 A(文本 `read_fields_in_text`)/C(看图
`read_fields_on_images`)两路**并发**按同一组概念键抽候选，**一致直接采信（省一次 LLM）、矛盾才据
原图评判**（`fusion.fuse_sources`）。入口 `fusion.run_vision_fusion`，ingest 据
`enable_vision_fusion` 调用。

**铁律（正确性，非兼容性，别违反）**：
- 融合结论**只写 `field_verdicts` / `fusion_overall_confidence` sidecar，绝不回写原
  `amounts/fields`**——原字段带着 `evidence/unit/is_total_component` 与 `computed_total` 的勾稽
  不变量，回写会破坏。
- **概念键每概念独立**（一般/特定医疗/重疾各一键，见 `INSURANCE_FIELD_DEFS`）——独立键让融合按键
  各管各，根除"A 概念值覆盖 B 概念"的对齐错位。
- 评判 prompt **独立依据原图、勿受候选主导、矛盾以图为准并标 low_confidence**；被保险人 vs 投保人
  口径写进字段定义（文本路常把投保人误当被保险人，靠看图纠正）。
- low_confidence 长尾留 `agent_fallback.escalate_low_confidence` 接口（**本期 no-op**，未来插
  agentic 兜底只改这一处）。

## 并发：能并发的 LLM 调用都走 `utils.map_concurrent`

openai SDK 同步阻塞、GIL 在网络等待时释放，线程池足够（不引 asyncio）。逐页 OCR、看图抽字段、
多字段评判都用 `map_concurrent`（**保序**、单项失败隔离）。**铁律**：`OpenAI` client 与
`sanitized_httpx_proxy_env`（改进程级 `os.environ`，多线程进退会竞态）必须在**并发块外层一次性
构造**，worker 只复用 client。并发度旋钮 `CONTRACT_ARCHIVE_LLM_CONCURRENCY`（默认 4）。

## 页级分流：混合提取取代整份"二选一"

`utils.page_router.classify_pages` 逐页判 text/ocr（主判据=单页文本层质量；加分项=含表格的文本页
也走 VL）。OCR 据此做**混合提取**：文本页原生抽取、扫描/表格页 VL OCR，按页序拼回——取代旧的
"整份 native OR 整份 OCR"，混合版式不再丢数据。

## 档案 generation：DB 指针只能在发布后切换

新产物一律发布为 `documents/<sha12>.gen-<uuid>/`，是**不可变 generation**；
`documents.output_dir` 只是当前一代的 DB pointer，不能把 `<sha12>/` 当成新写入目录。

- 读、渲染、删除等路径必须经 `ArchivePaths.resolve_document_dir`：它只采纳受控 basename、在当前
  archive 根重新锚定，因此整库搬迁后仍可用；`documents/` 或 generation 是 symlink 时必须拒绝，别直接
  信任 DB 里的绝对路径。
- 持有 `slot_lock(sha)` 的范围是锁内去重复查、快照、发布直到 DB 事务提交；锁文件要保留，不能 unlink
  后重建。`source.pdf` 是校验完整 SHA 的独立 copy，不得恢复硬链接。
- 先将 staging fsync/rename 成 durable generation，再提交 DB 行或切 `output_dir`。旧代和发布后提交失败
  留下的 orphan 都是审计材料，**没有自动 GC**；重跑若未成功发布/提交，必须保留上次有效 pointer/元数据。

## 抽取完整性与阶段状态：不能拿残缺结果冒充成功

- 默认把完整 OCR 文本交给抽取，禁止静默截中段/尾部；调用方若有明确 `max_chars` 预算，超限应返回
  `INPUT_TOO_LONG`，模型以长度结束则是 `OUTPUT_TRUNCATED`，而不是解析部分 JSON 后标成功。
- `stage_issues` 中 required 的 `error` 必须把可用 OCR 的结果降为 `partial`；可选后处理仅记
  `warning`，不应无故改变状态或抹掉已抽出的字段。

## 运行时实验：ContextVar 传模型选择与调用账本

评测或临时实验用 `utils.llm_runtime.model_overrides` / `collect_llm_calls`，不要改进程环境或持久配置。
`map_concurrent` 会把该 ContextVar 复制到每个 worker；SDK 请求经 `observed_chat_completion` 记录实际
model、usage、耗时和安全错误类别，账本不保留 prompt、响应或密钥。

## 评测：换模型/改 prompt/改流水线 → 跑评测过 gate 才提交

`evals/` 是离线评测脚手架。两阶段：`evals.run`（跑→`results.jsonl` 增量累积）+ `evals.report`
（读全量→gate 决策报告）。

**数据私有化**：评测**框架代码**留公开主仓库；评测**数据集**（原始 PDF + 真实金标准，**不脱敏**）
放私有仓库（`git.crhan.com`）。框架读 `CONTRACT_ARCHIVE_EVALSET_DIR` 定位数据集，不设则回退主仓库
内合成 cases（CI smoke）。**改抽取 prompt/模型/流水线后，必须**指向私有数据集跑一遍过四重 gate、
确认无回归再提交。原始 PDF case 走整条生产链路（`ingest.run_full_extraction`），文本 case 供模型对比。
详见 `evals/README.md`。

模型替换的实测要跑 text / OCR / vision 三轴，并优先用真实 `source.pdf`；runner 的不可变 manifest 和
仅完整成功才发布的 OCR cache 是可比性的一部分。供应商未返回 usage 时保持 unknown，绝不可按零成本计。
