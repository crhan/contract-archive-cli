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
from dataclasses import dataclass
from typing import Any, Optional

from ..config import get_timeout_s, load_settings
from ..errors import ErrorInfo, classify_exception, config_missing
from ..utils.http_env import sanitized_httpx_proxy_env

logger = logging.getLogger(__name__)


@dataclass
class LlmResult:
    """
    一次 LLM 调用的产物 + 元数据。

    把 parsed / model / usage 一起作为返回值传出，让"实际用了哪个模型""花了多少
    token"成为可信的返回值，而非靠外部 monkeypatch 拦截偷取（SDK 一升级猴补就静默失效）。
    评测据 usage 算成本；model 是单一真相源，杜绝"记录的模型≠实际跑的模型"。
    失败路径返回 parsed={}，与历史"返回空 dict"语义一致（调用方判 `if not res.parsed`）。
    """

    parsed: dict[str, Any]                # 解析后的 JSON；失败为空 dict
    model: str                            # 本次实际请求的 model id
    usage: dict[str, Any] | None = None   # token 用量（DashScope resp["usage"]）；读不到为 None
    error: Optional[ErrorInfo] = None     # 结构化错误（缺 key / API 异常）；成功为 None


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
) -> LlmResult:
    """
    调用 DashScope LLM（OpenAI 兼容口）进行合同字段抽取。

    见 CLAUDE.md：DashScope 一律走 OpenAI 兼容接口（原生 Generation 不认部分模型 id）。
    :param document_text: 已 OCR 得到的合同全文（推荐用 markdown 版本）
    :param model: 默认从 DASHSCOPE_LLM_MODEL env 读，最终默认 qwen3.7-max
    :param max_chars: 截断阈值，避免超过模型上下文
    :return: LlmResult（parsed/model/usage）；失败时 parsed={}，调用方判 `if not res.parsed`
    """
    # 统一从 config 层取（env > 配置文件 > 默认）；显式传参仍优先（param or settings）。
    settings = load_settings()
    model = model or settings.dashscope_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip LLM extraction")
        return LlmResult(
            parsed={}, model=model,
            error=config_missing("DASHSCOPE_API_KEY 缺失，跳过 LLM 抽取"),
        )

    user_msg = f"以下是合同正文，请抽取字段：\n\n{_truncate_middle(document_text, max_chars)}"
    try:
        content, usage = _call_openai_compat(LLM_SYSTEM_PROMPT, user_msg, model, api_key, base_url)
    except Exception as e:  # noqa: BLE001 — 外部调用降级返回空，但保留结构化 error 供上层判重试
        logger.exception("DashScope LLM call failed: %s", e)
        return LlmResult(parsed={}, model=model, error=classify_exception(e))

    if not content:
        logger.warning("LLM empty response")
        return LlmResult(parsed={}, model=model, usage=usage)
    parsed = _parse_json_loose(content)
    if not parsed:
        logger.warning("LLM response not parseable as JSON: %s", content[:200])
    return LlmResult(parsed=parsed, model=model, usage=usage)


def _truncate_middle(text: str, max_chars: int) -> str:
    """超长则头 1/3 尾 2/3 截断——尾部承载签字/金额/到期日等关键信息，权重更高。"""
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    return text[:head] + "\n\n[...省略中段...]\n\n" + text[-(max_chars - head):]


def _usage_from_openai(resp: Any) -> dict[str, Any] | None:
    """OpenAI 兼容响应的 token 用量 → 归一化 input/output/total_tokens。读不到返回 None。"""
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    out = {
        "input_tokens": getattr(u, "prompt_tokens", None),
        "output_tokens": getattr(u, "completion_tokens", None),
        "total_tokens": getattr(u, "total_tokens", None),
    }
    return out if any(v is not None for v in out.values()) else None


def _call_openai_compat(
    system_prompt: str, user_content: str, model: str, api_key: str, base_url: str
) -> tuple[str, dict[str, Any] | None]:
    """
    经 DashScope 的 OpenAI 兼容接口调文本模型，返回 (content, usage)。失败抛异常由调用方降级。

    见 CLAUDE.md：DashScope 一律走兼容口（原生 Generation 不认部分模型 id，如 qwen3.6-flash）。
    开 json_object；**不设 max_tokens**（避免 JSON 被截断成非法串）；各 prompt 已含 "JSON" 字样。

    显式 timeout（默认 300s，DASHSCOPE_TIMEOUT_S 可调）：不设则吃 SDK 默认 ~600s，
    上游 hang 时 CI/agent 会静默干等近 10 分钟。300s 给长合同（截断后约 6 万字，
    且故意不设 max_tokens）留足头寸，又不至无界等待。超时异常由调用方 except 兜底降级。
    """
    from openai import OpenAI

    compat_url = base_url.replace("/api/v1", "/compatible-mode/v1")
    with sanitized_httpx_proxy_env():
        client = OpenAI(
            api_key=api_key, base_url=compat_url,
            timeout=get_timeout_s("DASHSCOPE_TIMEOUT_S", 300.0),
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            top_p=0.5,
            response_format={"type": "json_object"},
        )
    return (resp.choices[0].message.content or ""), _usage_from_openai(resp)


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
