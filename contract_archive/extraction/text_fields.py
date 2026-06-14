"""文本看字段（A 路）：从已 OCR 的全文里抽一组高价值概念的候选值，与看图路(C)融合。

与 vl_extract（看图）对偶：同一组 fields_spec，这里读文本抽取，产出 source="text" 候选。
全文一把喂、单次 LLM 调用（不像看图路需逐页并发）。走主文本模型（DASHSCOPE_LLM_MODEL）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..config import load_settings
from ..errors import ErrorInfo, classify_exception, config_missing
from ..schemas import FieldCandidate
from .llm_extractor import _call_openai_compat, _parse_json_loose, _truncate_middle
from .vl_extract import candidate_from_raw

logger = logging.getLogger(__name__)


TEXT_FIELDS_SYSTEM = """你是严谨的字段抽取助理。从给定文档全文中抽取下列字段，以 JSON 返回。

铁律：
1. 据实抽取，文中没有的字段填 null，禁止猜测、禁止跨段脑补。
2. 值保留原文（含币种/单位/百分比），如 "200万元" "1万元" "100%" "90天"。
3. 只输出 JSON，不要解释、不要 Markdown 代码块标记。

要抽取的字段（key: 含义）：
{fields}

返回 JSON（每个字段一个对象；没有填 null）：
{{
{schema}
}}
"""


@dataclass
class TextFieldsResult:
    """文本看字段产物：概念键 → 文本候选列表（每键至多一条，全文单次抽取）。"""

    by_key: dict[str, list[FieldCandidate]] = field(default_factory=dict)
    model: str = ""
    usage: Optional[dict] = None
    error: Optional[ErrorInfo] = None


def read_fields_in_text(
    document_text: str,
    fields_spec: dict[str, str],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_chars: int = 24000,
) -> TextFieldsResult:
    """从全文抽取 fields_spec 描述的字段，产出 source="text" 候选。无凭证/空输入 → 空结果。"""
    settings = load_settings()
    model = model or settings.dashscope_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
    if not fields_spec or not document_text:
        return TextFieldsResult(model=model)
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip text field extraction")
        return TextFieldsResult(
            model=model, error=config_missing("DASHSCOPE_API_KEY 缺失，跳过文本看字段")
        )

    system = _build_system(fields_spec)
    user = f"以下是文档全文，请抽取字段：\n\n{_truncate_middle(document_text, max_chars)}"
    try:
        content, usage = _call_openai_compat(system, user, model, api_key, base_url)
    except Exception as e:  # noqa: BLE001 — 外部调用降级，保留 error 供上层判
        logger.exception("text field extraction failed: %s", e)
        return TextFieldsResult(model=model, error=classify_exception(e))

    parsed = _parse_json_loose(content)
    by_key: dict[str, list[FieldCandidate]] = {}
    for k in fields_spec:
        cand = candidate_from_raw(parsed.get(k), source="text")
        if cand is not None:
            by_key[k] = [cand]
    return TextFieldsResult(by_key=by_key, model=model, usage=usage)


def _build_system(fields_spec: dict[str, str]) -> str:
    fields = "\n".join(f'- "{k}": {desc}' for k, desc in fields_spec.items())
    schema = ",\n".join(
        f'  "{k}": {{"value": <值或 null>, "evidence": <原文片段或 null>}}' for k in fields_spec
    )
    return TEXT_FIELDS_SYSTEM.format(fields=fields, schema=schema)
