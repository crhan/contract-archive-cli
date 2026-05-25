"""
通用文档抽取（LLM-first，跨类型）。

与合同专用抽取（自带一份调校过的合同 prompt + ContractExtraction schema）相对，
这里是面向任意类型的通用路径：一次调用完成「判类型 + 抽字段」，
结果归一化到 DocumentExtraction 信封。两者都是纯 LLM（Phase 2 起无 rule）。
死代码 rule 仅保留为确定性数值归一化（中文大写金额→数值、日期→ISO），
不参与字段抽取——加新文档类型只需扩 prompt 里的举例，无需写代码。

设计动机：用户要的是「整理各类文档让其可追溯」，核心吃 LLM 能力，
尽量少依赖死代码规则体系。
"""
from __future__ import annotations

import logging
from typing import Any

from ..schemas import (
    DOC_TYPES,
    DocumentExtraction,
    LabeledAmount,
    LabeledDate,
    LabeledValue,
)
from .llm_extractor import _extract_text, _parse_json_loose
from .normalize import coerce_obligations, normalize_date, parse_money_value

logger = logging.getLogger(__name__)


DOC_EXTRACT_SYSTEM_PROMPT = f"""你是一名严谨的文档档案管理助理。给定一份文档的 OCR 文本，
判断它属于哪类文档，并抽取结构化字段，用于建立可检索、可追溯的个人文档档案库。

铁律：
1. 只输出 JSON，不要任何解释、前缀、Markdown 代码块标记。
2. 抽不到的字段填 null 或空数组，禁止猜测、禁止拼凑。
3. 日期统一 ISO 8601 (YYYY-MM-DD)；占位/空白日期（"___年__月__日"）填 null。
4. 金额保留原文（含大写与币种），不要自己换算成数字。
5. 照实抽取，包括身份证号、电话等个人信息（这是用户本人的私人档案，需完整留存）。

doc_type 从以下规范类型择一（更细的归类写进 title）：
{("、".join(DOC_TYPES))}

JSON 字段定义：
{{
  "doc_type": "上面六类之一",
  "title": "简短但能区分同类文档的标题——务必嵌入关键当事人/标的物/编号，不要只写泛化文档名。如『张三在职收入证明』『XX花园3幢102室商品房认购协议』『18号地下车位转让协议(乙方李四)』",
  "summary": "一句话摘要，含关键主体+金额/数字+日期+标的，便于日后检索回忆与区分同类",
  "parties": ["涉及的人或机构全称", "..."],
  "primary_date": "该文档最重要的日期 ISO（合同=签订日，证明=出具日，发票=开票日）或 null",
  "primary_amount": "该文档最重要的金额原文（合同=合同额，收入证明=年收入）或 null",
  "key_dates": [{{"label": "出具日/签订日/到期日/入职日 等", "date": "YYYY-MM-DD"}}],
  "amounts": [{{"label": "年收入/月均收入/公积金(个人)/合同金额 等", "text": "金额原文"}}],
  "fields": [{{"label": "字段名", "value": "字段值"}}],
  "obligations": [
    {{"actor": "party_a|party_b|both", "action": "动宾短语", "deadline": "YYYY-MM-DD 或 null", "evidence": "原文片段"}}
  ]
}}

字段抽取要点：
- fields 是该类型专属的键值对，由文档内容自行决定抽哪些。例如：
  · 收入证明 → 持证人、身份证号、用人单位、职位、入职日期、联系人、联系电话、单位地址
  · 发票 → 发票号、税号、开票方、购买方、税额
  · 证件 → 证件号、有效期、签发机关
  把不属于 parties/amounts/key_dates 的有价值信息都放进 fields。
- amounts 列出文档里**所有**金额（不止主金额），各带语义 label。
- obligations 仅当文档含明确"谁该在何时做什么"的待办/义务（合同尤甚）；
  证明、发票等通常为空数组。actor 只能是 party_a|party_b|both。
"""


def call_llm_document(
    document_text: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_chars: int = 24000,
) -> dict[str, Any]:
    """
    调 DashScope LLM 做通用文档抽取，返回解析后的 dict（失败返回 {}）。

    与 llm_extractor.call_llm_extract 同样的调用骨架，但用通用文档 prompt。
    刻意不复用其函数体——合同那条路保持不动，避免改动牵连。
    """
    import dashscope  # lazy import
    import os

    model = model or os.getenv("DASHSCOPE_LLM_MODEL", "qwen3.7-max")
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
    base_url = base_url or os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
    )
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip LLM document extraction")
        return {}

    dashscope.base_http_api_url = base_url

    if len(document_text) > max_chars:
        # 头 1/3 尾 2/3：落款/金额/日期等关键信息多在尾部，权重更高
        head_size = max_chars // 3
        document_text = (
            document_text[:head_size]
            + "\n\n[...省略中段...]\n\n"
            + document_text[-(max_chars - head_size):]
        )

    messages = [
        {"role": "system", "content": DOC_EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"以下是文档正文，请判类型并抽取字段：\n\n{document_text}"},
    ]
    try:
        resp = dashscope.Generation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format="message",
            temperature=0.1,
            top_p=0.5,
            response_format={"type": "json_object"},
        )
    except TypeError:
        # 老版本 SDK 不接受 response_format/temperature
        resp = dashscope.Generation.call(
            api_key=api_key, model=model, messages=messages, result_format="message"
        )
    except Exception as e:
        logger.exception("DashScope document LLM call failed: %s", e)
        return {}

    text = _extract_text(resp)
    if not text:
        logger.warning("LLM empty response (document extract)")
        return {}
    parsed = _parse_json_loose(text)
    if not parsed:
        logger.warning("LLM document response not parseable: %s", text[:200])
    return parsed


def _coerce_labeled_amounts(raw: Any) -> list[LabeledAmount]:
    """LLM amounts 数组 → LabeledAmount，顺手算数值。跳过非法项。"""
    if not isinstance(raw, list):
        return []
    out: list[LabeledAmount] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("value") or "").strip()
        if not text:
            continue
        label = str(item.get("label", "")).strip() or "金额"
        out.append(LabeledAmount(label=label, text=text, value=parse_money_value(text)))
    return out


def _coerce_labeled_dates(raw: Any) -> list[LabeledDate]:
    """LLM key_dates 数组 → LabeledDate，日期归一化到 ISO。"""
    if not isinstance(raw, list):
        return []
    out: list[LabeledDate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        date = item.get("date")
        date = normalize_date(date) if isinstance(date, str) and date else None
        if not label and not date:
            continue
        out.append(LabeledDate(label=label or "日期", date=date))
    return out


def _coerce_labeled_values(raw: Any) -> list[LabeledValue]:
    """LLM fields 数组 → LabeledValue。跳过空值项。"""
    if not isinstance(raw, list):
        return []
    out: list[LabeledValue] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = item.get("value")
        value = "" if value is None else str(value).strip()
        if not label or not value:
            continue
        out.append(LabeledValue(label=label, value=value))
    return out


def _coerce_parties(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(p).strip() for p in raw if p and str(p).strip()]


def extract_document(
    document_text: str,
    llm_enabled: bool = True,
) -> DocumentExtraction:
    """
    通用文档抽取主入口：LLM 判类型 + 抽字段 → DocumentExtraction 信封。

    :param llm_enabled: False（或无 API key）时返回空信封（doc_type 留默认）。
                        通用路径不依赖 rule，关掉 LLM 就没有可抽的东西——诚实返回空。
    """
    if not llm_enabled:
        return DocumentExtraction()

    raw = call_llm_document(document_text)
    if not raw:
        return DocumentExtraction()

    doc_type = str(raw.get("doc_type", "")).strip()
    if doc_type not in DOC_TYPES:
        doc_type = "其他"

    primary_amount_text = (raw.get("primary_amount") or None)
    if isinstance(primary_amount_text, str):
        primary_amount_text = primary_amount_text.strip() or None

    primary_date = raw.get("primary_date")
    primary_date = normalize_date(primary_date) if isinstance(primary_date, str) and primary_date else None

    return DocumentExtraction(
        doc_type=doc_type,
        title=(str(raw["title"]).strip() if raw.get("title") else None),
        summary=(str(raw["summary"]).strip() if raw.get("summary") else None),
        parties=_coerce_parties(raw.get("parties")),
        primary_date=primary_date,
        primary_amount_text=primary_amount_text,
        primary_amount_value=parse_money_value(primary_amount_text),
        key_dates=_coerce_labeled_dates(raw.get("key_dates")),
        amounts=_coerce_labeled_amounts(raw.get("amounts")),
        fields=_coerce_labeled_values(raw.get("fields")),
        obligations=coerce_obligations(raw.get("obligations")),
        raw_evidence={},
    )
