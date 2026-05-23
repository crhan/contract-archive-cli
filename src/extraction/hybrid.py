"""
Rule + LLM hybrid 合并器。

合并策略（按字段）：
- rule + LLM 都命中且值一致 → confidence 0.9, source=merged
- 只 rule 命中     → confidence 0.6, source=rule
- 只 LLM 命中      → confidence 0.7, source=llm
- 都未命中         → missing, confidence 0.0
- 两者冲突         → 取 LLM 值（一般更准），confidence 0.5

`risk_clauses` 特殊处理：取并集去重，confidence = max(0.6, len(union)/10)
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..schemas import ContractExtraction, ExtractionConfidence, FieldConfidence
from .llm_extractor import call_llm_extract
from .rule_extractor import RuleResult, extract_rules

logger = logging.getLogger(__name__)


def normalize_date(value: str | None) -> str | None:
    """把规则抽到的原文日期粗糙归一化到 ISO 8601。失败原样返回。"""
    if not value:
        return None
    m = re.match(
        r"^((?:19|20)\d{2}|二[〇零]{1,3}[一二三四五六七八九十]{1,3})"
        r"[年\-./](\d{1,2}|[一二三四五六七八九十]{1,3})"
        r"[月\-./](\d{1,2}|[一二三四五六七八九十]{1,3})",
        value,
    )
    if not m:
        return value
    y, mo, d = m.group(1), m.group(2), m.group(3)
    if not y.isdigit():
        y = _cn_year_to_int(y)
    if not mo.isdigit():
        mo = _cn_num_to_int(mo)
    if not d.isdigit():
        d = _cn_num_to_int(d)
    try:
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    except (ValueError, TypeError):
        return value


_CN_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_year_to_int(s: str) -> str:
    digits = [str(_CN_DIGITS.get(ch, "")) for ch in s if ch in _CN_DIGITS]
    return "".join(digits) or s


def _cn_num_to_int(s: str) -> str:
    if "十" in s:
        parts = s.split("十")
        tens = _CN_DIGITS.get(parts[0], 1) if parts[0] else 1
        ones = _CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return str(tens * 10 + ones)
    return "".join(str(_CN_DIGITS.get(ch, "")) for ch in s) or s


def parse_money_value(value: str | None) -> float | None:
    """从原文金额抽出数值（人民币元）。中文大写暂不解析，返回 None。"""
    if not value:
        return None
    m = re.search(r"([0-9]+(?:[,，]\d{3})*(?:\.\d+)?)\s*(万元|万|千元|百元|元|圆)?", value)
    if not m:
        return None
    num = float(m.group(1).replace(",", "").replace("，", ""))
    unit = m.group(2) or "元"
    multiplier = {"万元": 10000, "万": 10000, "千元": 1000, "百元": 100,
                  "元": 1, "圆": 1}.get(unit, 1)
    return num * multiplier


def extract_contract(
    document_text: str,
    llm_enabled: bool = True,
) -> tuple[ContractExtraction, ExtractionConfidence]:
    """
    主入口：跑 rule + （可选）LLM，合并为统一 ContractExtraction + ExtractionConfidence。
    """
    rule_res = extract_rules(document_text)
    llm_res: dict[str, Any] = {}
    if llm_enabled:
        llm_res = call_llm_extract(document_text)

    extraction = ContractExtraction()
    conf = ExtractionConfidence()
    evidence: dict[str, str] = {}

    for field_name in (
        "contract_name",
        "party_a",
        "party_b",
        "amount",
        "sign_date",
        "expire_date",
        "auto_renewal",
    ):
        rule_hit = rule_res.get(field_name)
        rule_val = rule_hit.value if rule_hit else None
        llm_val = llm_res.get(field_name)

        # 日期归一化
        if field_name in ("sign_date", "expire_date"):
            rule_val = normalize_date(rule_val) if isinstance(rule_val, str) else rule_val
            llm_val = normalize_date(llm_val) if isinstance(llm_val, str) else llm_val

        merged_val, fc = _merge_field(rule_val, llm_val, field_name)
        setattr(extraction, field_name, merged_val)
        setattr(conf, field_name, fc)
        if rule_hit:
            evidence[field_name] = rule_hit.evidence
        elif llm_val:
            evidence[field_name] = f"[LLM] {merged_val}"

    # 数值化金额
    if extraction.amount:
        extraction.amount_value = parse_money_value(extraction.amount)

    # 风险条款单独合并
    risks_rule = (
        rule_res.get("risk_clauses").value.split("|")  # type: ignore[union-attr]
        if rule_res.get("risk_clauses") and rule_res.get("risk_clauses").value
        else []
    )
    risks_llm = llm_res.get("risk_clauses") or []
    if not isinstance(risks_llm, list):
        risks_llm = []
    merged_risks = _merge_risk_lists(risks_rule, risks_llm)
    extraction.risk_clauses = merged_risks
    conf.risk_clauses = FieldConfidence(
        value_source="merged"
        if merged_risks and risks_rule and risks_llm
        else ("llm" if risks_llm else ("rule" if risks_rule else "missing")),
        confidence=min(1.0, max(0.5, len(merged_risks) / 10.0))
        if merged_risks
        else 0.0,
        rule_hit=bool(risks_rule),
        llm_agreed=bool(risks_llm and risks_rule),
    )

    extraction.raw_evidence = evidence

    # 总分：各字段加权平均
    scores = [
        conf.contract_name.confidence,
        conf.party_a.confidence,
        conf.party_b.confidence,
        conf.amount.confidence,
        conf.sign_date.confidence,
        conf.expire_date.confidence,
        conf.auto_renewal.confidence,
        conf.risk_clauses.confidence,
    ]
    conf.overall = sum(scores) / len(scores)

    return extraction, conf


# 这些字段在原文里有强证据（金额数字、日期、自动续约关键词），
# rule/LLM 冲突时优先信 rule（LLM 容易幻觉）；
# 实体规整类（合同名/甲方/乙方）冲突时优先信 LLM（rule 通常只抓到候选片段）。
_RULE_WINS_ON_CONFLICT = {"amount", "sign_date", "expire_date", "auto_renewal"}


def _merge_field(
    rule_val: Any, llm_val: Any, field_name: str = ""
) -> tuple[Any, FieldConfidence]:
    """单字段合并 → (最终值, FieldConfidence)"""
    if rule_val is None and llm_val is None:
        return None, FieldConfidence(value_source="missing", confidence=0.0)
    if rule_val is not None and llm_val is None:
        return rule_val, FieldConfidence(
            value_source="rule", confidence=0.6, rule_hit=True, llm_agreed=None
        )
    if rule_val is None and llm_val is not None:
        return llm_val, FieldConfidence(
            value_source="llm", confidence=0.7, rule_hit=False, llm_agreed=None
        )
    # 两者都有：比较一致性
    if _values_equivalent(rule_val, llm_val):
        # 实体类合并时优先取 LLM（更规整、含全称）；可验证字段取 rule（带原文证据）
        merged = rule_val if field_name in _RULE_WINS_ON_CONFLICT else llm_val
        return merged, FieldConfidence(
            value_source="merged", confidence=0.9, rule_hit=True, llm_agreed=True
        )
    # 冲突：按字段策略
    if field_name in _RULE_WINS_ON_CONFLICT:
        return rule_val, FieldConfidence(
            value_source="rule", confidence=0.5, rule_hit=True, llm_agreed=False
        )
    return llm_val, FieldConfidence(
        value_source="llm", confidence=0.5, rule_hit=True, llm_agreed=False
    )


def _values_equivalent(a: Any, b: Any) -> bool:
    """
    模糊比较：
    - 布尔直接等
    - 字符串完全相等 → 等价
    - 子串包含但需满足：双方都 >=6 字符，且较短一方至少占较长一方 60% 长度，
      避免 "张三" 吃掉 "张三投资集团有限公司" 这类把改进当一致的误判
    """
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    sa = str(a).strip().lower()
    sb = str(b).strip().lower()
    if sa == sb:
        return True
    if not sa or not sb:
        return False
    short, long = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    if len(short) >= 6 and short in long and len(short) >= 0.6 * len(long):
        return True
    return False


def _merge_risk_lists(a: list[str], b: list[str]) -> list[str]:
    """合并风险条款列表（按近似去重）。"""
    out: list[str] = []
    seen: set[str] = set()
    for s in list(a) + list(b):
        if not s:
            continue
        key = re.sub(r"\s+", "", s)[:40]
        if key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out
