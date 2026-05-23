"""
LLM-based 合同字段抽取。

模型：DashScope qwen3.7-max（严格按用户指定的 model id，不要替换）

策略：
- 一次性把 markdown / raw_text 喂给 LLM，让其返回结构化 JSON
- 用 JSON Schema/示例约束输出
- 失败/超时时返回空字典，由 hybrid 层 fallback 到 rule
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


LLM_SYSTEM_PROMPT = """你是一名严谨的法律助理。请从给定的合同文本中抽取结构化字段。
要求：
1. 只输出 JSON，不要任何解释、前缀、Markdown 代码块标记。
2. 抽不到的字段填 null，禁止猜测。
3. 日期统一为 ISO 8601 (YYYY-MM-DD)。
4. 金额保留原文（如"人民币贰万元整"）放到 amount 字段。
5. risk_clauses 是字符串数组，每条简明扼要（≤80 字），仅列违约/赔偿/争议解决/不可抗力/保密等敏感条款。

JSON 字段定义：
{
  "contract_name": "合同名称",
  "party_a": "甲方全称",
  "party_b": "乙方全称",
  "amount": "合同金额原文",
  "sign_date": "签订日期 ISO 8601",
  "expire_date": "到期/终止日期 ISO 8601",
  "auto_renewal": true/false/null,
  "risk_clauses": ["风险条款1", "风险条款2"]
}
"""


def call_llm_extract(
    document_text: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_chars: int = 24000,
) -> dict[str, Any]:
    """
    调用 DashScope qwen3.7-max 进行合同字段抽取。

    :param document_text: 已 OCR 得到的合同全文（推荐用 markdown 版本）
    :param model: 默认从 DASHSCOPE_LLM_MODEL env 读，最终默认 qwen3.7-max
    :param max_chars: 截断阈值，避免超过模型上下文
    """
    import dashscope  # lazy import

    model = model or os.getenv("DASHSCOPE_LLM_MODEL", "qwen3.7-max")
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
    base_url = base_url or os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
    )
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip LLM extraction")
        return {}

    dashscope.base_http_api_url = base_url

    if len(document_text) > max_chars:
        # 头 1/3 尾 2/3：合同尾部承载签字/金额/到期日期等关键信息，权重更高
        head_size = max_chars // 3
        tail_size = max_chars - head_size
        head = document_text[:head_size]
        tail = document_text[-tail_size:]
        document_text = head + "\n\n[...省略中段...]\n\n" + tail

    user_msg = f"以下是合同正文，请抽取字段：\n\n{document_text}"

    try:
        resp = dashscope.Generation.call(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            result_format="message",
            temperature=0.1,  # 抽取任务降随机性
            top_p=0.5,
            response_format={"type": "json_object"},  # qwen3.x 支持 JSON 模式
        )
    except TypeError:
        # 老版本 SDK 不接受 response_format/temperature，回退最小参数
        resp = dashscope.Generation.call(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            result_format="message",
        )
    except Exception as e:
        logger.exception("DashScope LLM call failed: %s", e)
        return {}

    text = _extract_text(resp)
    if not text:
        logger.warning("LLM empty response")
        return {}

    parsed = _parse_json_loose(text)
    if not parsed:
        logger.warning("LLM response not parseable as JSON: %s", text[:200])
    return parsed


def _extract_text(resp: Any) -> str:
    try:
        choices = resp["output"]["choices"]
        content = choices[0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
        return ""
    except (KeyError, IndexError, TypeError):
        try:
            return resp["output"]["text"]
        except Exception:
            return ""


def _parse_json_loose(text: str) -> dict[str, Any]:
    """
    LLM 偶尔会带 markdown 代码块或前缀文字，做一次 best-effort 解析。
    """
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ``` 包裹
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # 抓第一个 {...} 块
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # 修一下常见问题：单引号 / trailing comma
        repaired = m.group(0).replace("'", '"')
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return {}
