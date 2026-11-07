"""
合同字段抽取（LLM-only）。

Phase 2 起退役了 rule_extractor + rule/LLM hybrid 合并：合同抽取与其余文档类型
一样走纯 LLM，只对 LLM 输出做确定性数值归一化（金额→数值、日期→ISO）。
保留 ContractExtraction schema 与合同专属列/搜索（party/到期/续约/风险/义务），
是因为这些是文档化的能力，不破坏。

合同的 LLM prompt 仍是那份调校过的 LLM_SYSTEM_PROMPT（见 llm_extractor）。
"""
from __future__ import annotations

import logging

from ..schemas import (
    ContractExtraction,
    ExtractionConfidence,
    FieldConfidence,
)
from ..errors import ErrorInfo, extract_empty
from .llm_extractor import call_llm_extract
from .normalize import (
    coerce_bool,
    coerce_obligations,
    coerce_str_list,
    normalize_date,
    parse_money_value,
)

logger = logging.getLogger(__name__)

# 参与 overall 置信度均值的字段（与历史口径一致）
_SCORED_FIELDS = (
    "contract_name", "party_a", "party_b", "amount",
    "sign_date", "expire_date", "auto_renewal", "risk_clauses",
)


def _invalid_obligations_error() -> ErrorInfo:
    """Return a safe error when the specialised obligations payload is unusable.

    A specialised extraction is allowed to replace the generic obligations only
    when it explicitly supplied a fully valid list.  Treating a missing or partly
    malformed value as ``[]`` would turn an absent model field into permission to
    erase useful generic extraction results in the doc-type handler.
    """
    return ErrorInfo(
        code="INVALID_RESPONSE",
        category="validation",
        message="LLM returned invalid obligations",
        retryable=False,
    )


def _coerce_explicit_obligations(raw: dict) -> tuple[list, ErrorInfo | None]:
    """Coerce only an explicit, entirely valid obligations list.

    ``coerce_obligations`` owns the accepted actor aliases and normalisation.
    Validate each item around that established coercer rather than recreating a
    second alias table here.  Every semantic string field is deliberately strict:
    silently stringifying a number/object makes malformed LLM JSON look like an
    authoritative empty or valid result.
    """
    if "obligations" not in raw or not isinstance(raw.get("obligations"), list):
        return [], _invalid_obligations_error()

    obligations_raw = raw["obligations"]
    for item in obligations_raw:
        if not isinstance(item, dict):
            return [], _invalid_obligations_error()
        actor = item.get("actor")
        action = item.get("action")
        if not isinstance(actor, str) or not actor.strip():
            return [], _invalid_obligations_error()
        if not isinstance(action, str) or not action.strip():
            return [], _invalid_obligations_error()
        if any(
            item.get(field) is not None and not isinstance(item.get(field), str)
            for field in ("deadline", "evidence")
        ):
            return [], _invalid_obligations_error()
        # This uses the existing actor aliases and action semantics.  A one-item
        # result proves that the item did not have an unknown actor or blank
        # action; anything else invalidates the entire replacement list.
        if len(coerce_obligations([item])) != 1:
            return [], _invalid_obligations_error()

    # An explicit [] is meaningful: it is the one safe way for the specialised
    # extractor to clear generic obligations.
    return coerce_obligations(obligations_raw), None


def _build_confidence(ext: ContractExtraction) -> ExtractionConfidence:
    """
    LLM-only 置信度：有值即 llm/0.7，无值 missing/0.0。
    无 rule 交叉验证，故不再有 merged/0.9 与 rule_hit 维度——诚实反映"只有 LLM 一票"。
    """
    conf = ExtractionConfidence()

    def fc(present: bool) -> FieldConfidence:
        return FieldConfidence(
            value_source="llm" if present else "missing",
            confidence=0.7 if present else 0.0,
            rule_hit=False,
            llm_agreed=None,
        )

    conf.contract_name = fc(bool(ext.contract_name))
    conf.party_a = fc(bool(ext.party_a))
    conf.party_b = fc(bool(ext.party_b))
    conf.amount = fc(bool(ext.amount))
    conf.sign_date = fc(bool(ext.sign_date))
    conf.expire_date = fc(bool(ext.expire_date))
    conf.auto_renewal = fc(ext.auto_renewal is not None)
    conf.risk_clauses = fc(bool(ext.risk_clauses))
    scores = [getattr(conf, f).confidence for f in _SCORED_FIELDS]
    conf.overall = sum(scores) / len(scores)
    return conf


def extract_contract(
    document_text: str,
    llm_enabled: bool = True,
    model: str | None = None,
    max_chars: int | None = None,
) -> tuple[ContractExtraction, ExtractionConfidence]:
    """
    合同字段抽取主入口（LLM-only）。

    :param llm_enabled: False（或无 API key / LLM 失败）时返回空结果——
                        纯 LLM 路径无 rule 兜底，抽不到比硬塞更诚实。
    :param model: 覆盖抽取所用 model（默认 None=走 settings.dashscope_model）；
                  合同线是双 LLM 调用之一，评测换模型时与 extract_document 同步穿透才保真。
    """
    if not llm_enabled:
        return ContractExtraction(), ExtractionConfidence()

    if max_chars is None:
        result = call_llm_extract(document_text, model=model)
    else:
        result = call_llm_extract(document_text, model=model, max_chars=max_chars)
    raw = result.parsed
    if not raw:
        # Keep the failed specialisation observable to the doc-type handler.  An
        # empty object with no original transport/parse error is also an omitted
        # obligations field, so it must not be mistaken for an authoritative [].
        return (
            ContractExtraction(
                llm_model=result.model,
                llm_usage=result.usage,
                extraction_error=result.error or _invalid_obligations_error(),
            ),
            ExtractionConfidence(),
        )

    if not isinstance(raw, dict):
        return (
            ContractExtraction(
                llm_model=result.model,
                llm_usage=result.usage,
                extraction_error=result.error or extract_empty("LLM returned invalid JSON"),
            ),
            ExtractionConfidence(),
        )

    # A truthy JSON object without an explicit valid obligations list is not a
    # successful specialised obligation extraction.  Still retain every other
    # parsable contract field and the call metadata for auditability; the handler
    # will use ``extraction_error`` to preserve the generic obligations.
    obligations, obligations_error = _coerce_explicit_obligations(raw)

    ext = ContractExtraction(
        contract_name=(raw.get("contract_name") or None),
        party_a=(raw.get("party_a") or None),
        party_b=(raw.get("party_b") or None),
        amount=(raw.get("amount") or None),
        sign_date=normalize_date(raw.get("sign_date")) if isinstance(raw.get("sign_date"), str) else None,
        expire_date=normalize_date(raw.get("expire_date")) if isinstance(raw.get("expire_date"), str) else None,
        auto_renewal=coerce_bool(raw.get("auto_renewal")),
        risk_clauses=coerce_str_list(raw.get("risk_clauses")),
        obligations=obligations,
        llm_model=result.model,
        llm_usage=result.usage,
        extraction_error=obligations_error or result.error,
    )
    ext.amount_value = parse_money_value(ext.amount)
    ext.raw_evidence = {
        k: f"[LLM] {getattr(ext, k)}"
        for k in ("contract_name", "party_a", "party_b", "amount", "sign_date")
        if getattr(ext, k)
    }
    return ext, _build_confidence(ext)
