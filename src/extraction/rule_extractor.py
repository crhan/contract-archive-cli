"""
Rule-based 合同字段抽取。

策略：
- 正则在合同文本里很可靠的字段（金额、日期、签订日期、自动续约关键词）走规则
- 实体型字段（甲方/乙方/合同名）规则可作"候选"，最终由 LLM 仲裁
- 风险条款由关键词 + 句子级提取

返回的是字段级别的 RuleHit 列表，由 hybrid 层与 LLM 结果合并。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 日期：2024年5月23日 / 2024-05-23 / 2024/5/23 / 2024.05.23 / 二〇二四年五月二十三日
# 注意：
# - 年份分支必须用 (?:19|20)\d{2}，否则 "19" 会被单独匹配（优先级 bug）
# - "日" 前必须紧跟数字或中文数字，不能是空白；避免抓到合同占位符 "2026年5月___日"
DATE_PATTERNS = [
    re.compile(
        r"(?P<y>(?:19|20)\d{2}|二[〇零]{1,3}[一二三四五六七八九十]{1,3})"
        r"\s*[年\-./]\s*(?P<m>1[0-2]|0?[1-9]|[一二三四五六七八九十]{1,3})"
        r"\s*[月\-./]\s*(?P<d>3[01]|[12]\d|0?[1-9]|[一二三四五六七八九十]{1,3})"
        r"(?:\s*日)?"
    ),
]

# 金额：必须带单位，否则误抓页码/编号。
# 规则1：带千分位 / 小数点 → 强信号
# 规则2：4+ 位整数 + 单位（"210000 元"）
# 规则3：1-3 位整数但单位是"万元/千元/百元"（"5万元"）
MONEY_PATTERN = re.compile(
    r"(?:人民币|¥|RMB|￥)?\s*"
    r"(?P<num>[0-9]{1,3}(?:[,，]\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,}|\d{1,3}(?=\s*(?:万元|万|千元|百元)))"
    r"\s*(?P<unit>元|圆|万元|万|千元|百元)\s*整?"
)
CHINESE_MONEY_PATTERN = re.compile(
    r"(?:人民币)?[零壹贰叁肆伍陆柒捌玖拾佰仟万亿]{2,}\s*(?:元|圆)\s*整?"
)

# 甲方/乙方候选行
PARTY_A_PATTERN = re.compile(r"甲\s*方[（(]?[^）)]{0,20}[）)]?\s*[:：]\s*([^\n]{2,80})")
PARTY_B_PATTERN = re.compile(r"乙\s*方[（(]?[^）)]{0,20}[）)]?\s*[:：]\s*([^\n]{2,80})")

# 合同名：标题行（第一行非空，或包含"合同/协议/契约"且长度 < 40 的最早一行）
CONTRACT_NAME_HINT = re.compile(r"([^\n]{4,40}(?:合同|协议|契约|约定书|意向书))")

# 自动续约关键词。注意：
# "顺延" 在合同里通常指"逾期顺延"/"工期顺延"，与续约无关，不能纳入。
# "延期" 同理；只有明确含"续约/续签"才算肯定信号。
AUTO_RENEW_TRUE = re.compile(
    r"(自动续(?:约|签)|期满(?:后)?自动续(?:约|签)|期满之日起.{0,12}续(?:约|签)|automatically\s+renew|auto-?renew)"
)
AUTO_RENEW_FALSE = re.compile(
    r"(不(?:自动)?续(?:约|签)|不再续(?:约|签)|期满终止|期满即(?:自动)?终止|无须?续(?:约|签))"
)

# 风险条款关键词
RISK_KEYWORDS = [
    "违约金",
    "赔偿",
    "解除",
    "终止",
    "争议",
    "诉讼",
    "仲裁",
    "保密",
    "知识产权",
    "不可抗力",
    "罚款",
    "滞纳金",
    "管辖",
    "免责",
    "连带责任",
]


@dataclass
class RuleHit:
    field_name: str
    value: str | bool | None
    evidence: str
    confidence: float = 0.7  # rule 命中的默认置信度


@dataclass
class RuleResult:
    hits: list[RuleHit] = field(default_factory=list)

    def get(self, name: str) -> RuleHit | None:
        for h in self.hits:
            if h.field_name == name:
                return h
        return None


def extract_rules(text: str) -> RuleResult:
    """
    跑一遍所有规则。text 是 raw_text.txt 的全部内容。

    Linus 风格：每个 if 都解决一类问题，不要嵌套 if，找不到就 skip。
    """
    res = RuleResult()

    name = _extract_contract_name(text)
    if name:
        res.hits.append(RuleHit("contract_name", name, name, 0.6))

    pa = PARTY_A_PATTERN.search(text)
    if pa:
        res.hits.append(RuleHit("party_a", pa.group(1).strip(), pa.group(0), 0.75))

    pb = PARTY_B_PATTERN.search(text)
    if pb:
        res.hits.append(RuleHit("party_b", pb.group(1).strip(), pb.group(0), 0.75))

    money = _extract_money(text)
    if money:
        res.hits.append(RuleHit("amount", money[0], money[1], 0.7))

    # 1) 优先匹配带标签的签订日期（合同落款处常见格式："签订日期: 2024年5月10日"）
    sign_labeled = _extract_labeled_date(
        text, labels=("签订日期", "签订时间", "签署日期", "落款日期", "签字日期")
    )
    if sign_labeled:
        res.hits.append(RuleHit("sign_date", sign_labeled[0], sign_labeled[1], 0.85))

    # 2) 到期/截止日期同样优先看标签
    expire_labeled = _extract_labeled_date(
        text, labels=("到期日期", "截止日期", "终止日期", "失效日期", "有效期至")
    )
    if expire_labeled:
        res.hits.append(RuleHit("expire_date", expire_labeled[0], expire_labeled[1], 0.85))

    # 3) 兜底：没有标签时退化到"最早=sign, 最晚=expire"的粗糙启发式（低置信度，让 LLM 仲裁）
    if not sign_labeled or not expire_labeled:
        dates = _extract_dates(text)
        if dates:
            sorted_dates = sorted(dates, key=lambda x: x[0])
            if not sign_labeled:
                res.hits.append(
                    RuleHit("sign_date", sorted_dates[0][0], sorted_dates[0][1], 0.4)
                )
            if not expire_labeled and len(sorted_dates) > 1:
                res.hits.append(
                    RuleHit("expire_date", sorted_dates[-1][0], sorted_dates[-1][1], 0.4)
                )

    renewal = _extract_auto_renewal(text)
    if renewal is not None:
        val, evi = renewal
        res.hits.append(RuleHit("auto_renewal", val, evi, 0.7))

    risks = _extract_risk_clauses(text)
    if risks:
        res.hits.append(
            RuleHit(
                "risk_clauses",
                "|".join(risks),
                "; ".join(risks[:3]),
                0.5,
            )
        )

    return res


def _extract_contract_name(text: str) -> str | None:
    # 优先：前 5 行内的标题
    for line in text.splitlines()[:5]:
        line = line.strip()
        if 4 <= len(line) <= 40 and any(
            k in line for k in ("合同", "协议", "契约", "约定书")
        ):
            return line
    m = CONTRACT_NAME_HINT.search(text)
    return m.group(1).strip() if m else None


def _extract_money(text: str) -> tuple[str, str] | None:
    # 先找中文大写（多出现在合同正式金额）
    m = CHINESE_MONEY_PATTERN.search(text)
    if m:
        # 找前后小写阿拉伯数字配合
        window_start = max(0, m.start() - 20)
        window_end = min(len(text), m.end() + 40)
        window = text[window_start:window_end]
        return m.group(0), window

    m = MONEY_PATTERN.search(text)
    if m:
        return m.group(0).strip(), text[
            max(0, m.start() - 20) : min(len(text), m.end() + 20)
        ]
    return None


def _extract_labeled_date(
    text: str, labels: tuple[str, ...]
) -> tuple[str, str] | None:
    """
    从形如 '签订日期: 2024年5月10日' / '签订日期：2024-05-10' 的文本里抽日期。
    多次出现时取最后一个（合同正文里可能预先列标题，最终落款才是真值）。
    """
    label_re = "|".join(re.escape(lbl) for lbl in labels)
    pattern = re.compile(
        rf"(?:{label_re})\s*[:：]?\s*"
        r"((?:19|20)\d{2}\s*[年\-./]\s*(?:1[0-2]|0?[1-9])\s*[月\-./]\s*(?:3[01]|[12]\d|0?[1-9])\s*日?)"
    )
    matches = pattern.findall(text)
    if matches:
        # 取最后一个出现的——合同正文的字段标签通常重复多次，最后那次往往是落款的真值
        return matches[-1].strip(), matches[-1].strip()
    return None


def _extract_dates(text: str) -> list[tuple[str, str]]:
    """返回 [(normalized_date_str, evidence)]"""
    out: list[tuple[str, str]] = []
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            out.append((m.group(0), m.group(0)))
    # 去重保序
    seen = set()
    unique = []
    for v, e in out:
        if v not in seen:
            seen.add(v)
            unique.append((v, e))
    return unique


def _extract_auto_renewal(text: str) -> tuple[bool, str] | None:
    """先判否定（更具体），再判肯定。"""
    m_false = AUTO_RENEW_FALSE.search(text)
    if m_false:
        return False, m_false.group(0)
    m_true = AUTO_RENEW_TRUE.search(text)
    if m_true:
        return True, m_true.group(0)
    return None


def _extract_risk_clauses(text: str) -> list[str]:
    """按句切分，命中 RISK_KEYWORDS 的句子作为风险条款候选。"""
    sentences = re.split(r"(?<=[。！？!?；;])\s*", text)
    hits: list[str] = []
    for s in sentences:
        s = s.strip()
        if 6 < len(s) < 200 and any(k in s for k in RISK_KEYWORDS):
            hits.append(s)
        if len(hits) >= 10:
            break
    return hits
