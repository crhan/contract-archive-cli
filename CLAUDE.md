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
- usage 从 `resp.usage`（`prompt_tokens`/`completion_tokens`/`total_tokens`）读，归一化成
  `input_tokens`/`output_tokens`/`total_tokens`。

## 换模型评测

`evals/` 是离线评测脚手架，判断能否用更便宜模型替换抽取主力模型。两阶段：
`evals.run`（跑模型→`results.jsonl` 增量累积）+ `evals.report`（读全量→gate 决策报告）。
换模型走 `extract_document(text, model=m)`，测的是整条生产链路而非裸 JSON。详见 `evals/README.md`。
