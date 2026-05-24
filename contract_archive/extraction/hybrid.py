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

from ..schemas import (
    ContractExtraction,
    ExtractionConfidence,
    FieldConfidence,
    ObligationItem,
)
from .llm_extractor import call_llm_extract
from .rule_extractor import RuleResult, extract_rules

logger = logging.getLogger(__name__)


def normalize_date(value: str | None) -> str | None:
    """把规则抽到的原文日期粗糙归一化到 ISO 8601。失败原样返回。"""
    if not value:
        return None
    # 兼容 OCR 输出里数字与"年/月"之间出现的空格
    m = re.match(
        r"^((?:19|20)\d{2}|二[〇零]{1,3}[一二三四五六七八九十]{1,3})"
        r"\s*[年\-./]\s*(\d{1,2}|[一二三四五六七八九十]{1,3})"
        r"\s*[月\-./]\s*(\d{1,2}|[一二三四五六七八九十]{1,3})",
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


_CN_MONEY_DIGITS = {
    "零": 0, "壹": 1, "贰": 2, "叁": 3, "肆": 4, "伍": 5,
    "陆": 6, "柒": 7, "捌": 8, "玖": 9,
    # 小写也支持（合同里偶尔混用）
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "两": 2,
}
_CN_MONEY_UNITS = {
    "拾": 10, "佰": 100, "仟": 1000, "万": 10000, "亿": 100000000,
    "十": 10, "百": 100, "千": 1000,
}


def _cn_money_to_value(s: str) -> float | None:
    """
    中文大写金额 → 数值（元）。
    "壹仟贰佰贰拾柒万玖仟捌佰捌拾玖元整" → 12279889.0
    "贰万元" → 20000.0
    解析失败返回 None。
    """
    s = (s.replace("元", "").replace("圆", "")
          .replace("整", "").replace("人民币", "").strip())
    if not s:
        return None
    total = 0
    section = 0   # 当前"万"以下的累加
    digit = 0     # 上一个数字
    saw_any = False
    for ch in s:
        if ch in _CN_MONEY_DIGITS:
            digit = _CN_MONEY_DIGITS[ch]
            saw_any = True
        elif ch in _CN_MONEY_UNITS:
            unit = _CN_MONEY_UNITS[ch]
            if unit >= 10000:
                # 万 / 亿 触发段结算
                section = (section + digit) * unit
                total += section
                section = 0
            else:
                # 拾/佰/仟：digit 为 0 时按 1（"拾" 单独 = 10）
                section += (digit if digit else 1) * unit
            digit = 0
            saw_any = True
        elif ch.isspace():
            continue
        else:
            return None  # 不认识的字符（含混阿拉伯/中文），交给阿拉伯路径
    section += digit
    total += section
    return float(total) if saw_any and total > 0 else None


def parse_money_value(value: str | None) -> float | None:
    """从原文金额抽出数值（人民币元）。先试阿拉伯数字，再试中文大写。"""
    if not value:
        return None
    # 阿拉伯数字优先（精度高，覆盖大多数情况）
    m = re.search(r"([0-9]+(?:[,，]\d{3})*(?:\.\d+)?)\s*(万元|万|千元|百元|元|圆)?", value)
    if m:
        num = float(m.group(1).replace(",", "").replace("，", ""))
        unit = m.group(2) or "元"
        multiplier = {"万元": 10000, "万": 10000, "千元": 1000, "百元": 100,
                      "元": 1, "圆": 1}.get(unit, 1)
        return num * multiplier
    # 退化到中文大写
    return _cn_money_to_value(value)


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
        rule_conf = rule_hit.confidence if rule_hit else 0.0
        llm_val = llm_res.get(field_name)

        # 日期归一化
        if field_name in ("sign_date", "expire_date"):
            rule_val = normalize_date(rule_val) if isinstance(rule_val, str) else rule_val
            llm_val = normalize_date(llm_val) if isinstance(llm_val, str) else llm_val

        merged_val, fc = _merge_field(rule_val, llm_val, field_name, rule_conf=rule_conf)
        setattr(extraction, field_name, merged_val)
        setattr(conf, field_name, fc)
        if rule_hit:
            evidence[field_name] = rule_hit.evidence
        elif llm_val:
            evidence[field_name] = f"[LLM] {merged_val}"

    # 数值化金额
    if extraction.amount:
        extraction.amount_value = parse_money_value(extraction.amount)

    # obligations 完全由 LLM 提供（rule 无法可靠区分动作/罚则）
    extraction.obligations = _coerce_obligations(llm_res.get("obligations"))

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
    rule_val: Any,
    llm_val: Any,
    field_name: str = "",
    rule_conf: float = 0.7,
) -> tuple[Any, FieldConfidence]:
    """
    单字段合并 → (最终值, FieldConfidence)。

    rule_conf < 0.7 视为弱信号——即使 field_name 在 _RULE_WINS_ON_CONFLICT
    名单里，冲突时也让 LLM 赢（rule 的兜底启发式不应压过 LLM 的整体判断）。
    """
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
    # 冲突：按字段策略 + rule 置信度门槛
    if field_name in _RULE_WINS_ON_CONFLICT and rule_conf >= 0.7:
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


def _coerce_obligations(raw: Any) -> list[ObligationItem]:
    """
    把 LLM 返回的 obligations 数组（dict 列表）转 ObligationItem。
    LLM 偶尔会返回 actor 写成"甲方"/"乙方"中文 —— 这里做兜底归一。
    跳过缺 actor 或 action 的非法项，不抛异常。
    """
    if not isinstance(raw, list):
        return []
    actor_alias = {
        "甲方": "party_a", "Party A": "party_a", "partyA": "party_a",
        "乙方": "party_b", "Party B": "party_b", "partyB": "party_b",
        "双方": "both",   "Both": "both",
    }
    out: list[ObligationItem] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        actor = str(item.get("actor", "")).strip()
        actor = actor_alias.get(actor, actor)
        if actor not in ("party_a", "party_b", "both"):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        deadline = item.get("deadline")
        if isinstance(deadline, str):
            deadline = normalize_date(deadline) or None
        else:
            deadline = None
        evidence = str(item.get("evidence", "")).strip()[:500]
        out.append(
            ObligationItem(
                actor=actor,  # type: ignore[arg-type]
                action=action[:200],
                deadline=deadline,
                evidence=evidence,
            )
        )
    return out
