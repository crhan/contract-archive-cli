"""
LLM 调用的共享请求参数。

**关思考模式（`enable_thinking=False`）是全项目铁律**，每个 `chat.completions.create`
都必须带上，原因有二：

1. **思考模型不支持 `response_format={"type":"json_object"}`**——本项目所有抽取都靠
   JSON 信封，思考态下 JSON 约束失效，解析直接崩。
2. **延迟塌方**。新一代模型（qwen3.8-max 起）**默认开思考**：实测同一份保单，
   qwen3.7-max 87s / 5.7k 输出 token，qwen3.8-max 默认态要 897s（= 两次 300s 超时
   重试 + 一次成功）/ 15.9k 输出 token。不显式关，换新模型就等于换来 10 倍延迟。

老模型（qwen3.7-max / qwen3.6-flash / qwen-vl-ocr-latest）本就不思考，带上此参数
无副作用——已逐个实测接受。故五处调用统一传，不做 per-model 分支。
"""
from __future__ import annotations

# 传给 openai SDK 的 extra_body。别就地写字面量字典——散落五处会漏改，
# 漏掉哪处，哪处就在换新模型时静默退化成思考态。
NO_THINKING: dict[str, object] = {"enable_thinking": False}
