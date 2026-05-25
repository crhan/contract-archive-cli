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

    raw = call_llm_extract(document_text, model=model).parsed
    if not raw:
        return ContractExtraction(), ExtractionConfidence()

    ext = ContractExtraction(
        contract_name=(raw.get("contract_name") or None),
        party_a=(raw.get("party_a") or None),
        party_b=(raw.get("party_b") or None),
        amount=(raw.get("amount") or None),
        sign_date=normalize_date(raw.get("sign_date")) if isinstance(raw.get("sign_date"), str) else None,
        expire_date=normalize_date(raw.get("expire_date")) if isinstance(raw.get("expire_date"), str) else None,
        auto_renewal=coerce_bool(raw.get("auto_renewal")),
        risk_clauses=coerce_str_list(raw.get("risk_clauses")),
        obligations=coerce_obligations(raw.get("obligations")),
    )
    ext.amount_value = parse_money_value(ext.amount)
    ext.raw_evidence = {
        k: f"[LLM] {getattr(ext, k)}"
        for k in ("contract_name", "party_a", "party_b", "amount", "sign_date")
        if getattr(ext, k)
    }
    return ext, _build_confidence(ext)
