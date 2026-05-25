"""
LLM-based 合同字段抽取。

模型：DashScope qwen3.7-max（严格按用户指定的 model id，不要替换）

策略：
- 一次性把 markdown / raw_text 喂给 LLM，让其返回结构化 JSON
- 用 JSON Schema/示例约束输出
- 失败/超时时返回空字典，由调用方（contract_extractor）处理为空抽取
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config import load_settings

logger = logging.getLogger(__name__)


LLM_SYSTEM_PROMPT = """你是一名严谨的法律助理。请从给定的合同文本中抽取结构化字段。
铁律：
1. 只输出 JSON，不要任何解释、前缀、Markdown 代码块标记。
2. 抽不到的字段填 null，禁止猜测、禁止拼凑。
3. 如果合同里某个日期是占位符或空白（如"___年__月__日"、"2026年5月_日"），返回 null，不要补全。
4. 日期统一为 ISO 8601 (YYYY-MM-DD)。
5. 金额保留原文（如"人民币贰万元整"或"210000 元整"）。
6. 签订日期(sign_date)：仅取合同最后落款/签字处的日期；如未明确则 null，不要把"付款日"或"交付日"当签订日。
7. 到期日期(expire_date)：仅取合同明确的有效期/失效日；车位转让、买卖等一次性合同通常没有，应填 null。
8. party_a / party_b：填全称（含公司类型如"有限公司"或买受人完整身份描述）。
9. risk_clauses 是字符串数组，每条 ≤80 字，仅列违约/赔偿/争议解决/不可抗力/保密/管辖等"出问题后果"型条款。
10. obligations 是动作清单——"X 方应/须于 Y 之前做 Z"型条款。
    与 risk_clauses 严格区分：
      - 动作类（要做某事、按时交付、提交资料、付款、验收、盖章、签订其他合同）→ obligations
      - 后果类（违约金、解除权、争议解决、滞纳金、赔偿、不可抗力免责）→ risk_clauses
    每条 obligation 必须含：actor、action、deadline（若无明确日期则 null）、evidence（原文片段≤120字）
    actor 只能是 "party_a"|"party_b"|"both"，不要写实际人名/公司名。
    action 用动宾短语，≤30字，例如"递交审贷资料"、"交付车位"、"支付定金"。
    deadline 是 ISO 'YYYY-MM-DD'；原文为"签订本协议当日"/"30 日内"等相对时间无法换算时填 null。
    宁缺毋滥：抽不出动作不要硬凑；典型合同 obligations 5-15 条为正常。

JSON 字段定义：
{
  "contract_name": "合同名称",
  "party_a": "甲方全称",
  "party_b": "乙方全称（如有多人/多主体用顿号分隔）",
  "amount": "合同金额原文",
  "sign_date": "签订日期 ISO 8601 或 null",
  "expire_date": "到期/终止日期 ISO 8601 或 null",
  "auto_renewal": true/false/null,
  "risk_clauses": ["违约金/赔偿/解除/争议解决等罚则", "..."],
  "obligations": [
    {
      "actor": "party_a"|"party_b"|"both",
      "action": "动宾短语",
      "deadline": "YYYY-MM-DD 或 null",
      "evidence": "原文片段"
    }
  ]
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

    # 统一从 config 层取（env > 配置文件 > 默认）；显式传参仍优先（param or settings）。
    settings = load_settings()
    model = model or settings.dashscope_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
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
