"""
确定性逐字段打分器：gold vs pred 两个 DocumentExtraction → 逐字段 TP/FP/FN。

设计要点（吸收评测方法学评审）：
- 复用生产 normalize_date / parse_money_value 当同一把尺子，避免评测与生产用不同标准。
- 列表字段顺序无关 → 贪心对齐后映射 TP/FP/FN → P/R/F1，干净处理多抽(FP)/漏抽(FN)。
- 金额用 exact（数值 + is_total_component 布尔整体匹配，错一个算 FN+FP），不用比值容差
  ——容差会放过量级错误，而金额是合同里最不能错的。
- 完整性 issues 用 F-beta(β=2) 偏召回：漏报缺陷远比误报致命；对齐 key=(category, 页码)，
  文本相似度兜底；签章类缺陷单独算召回，供决策硬门槛。
- 主观项（title/summary）这里只做归一化精确匹配当弱信号，质量评判留给 Phase 3 的窄 judge。

纯函数、不依赖网络：可用合成 gold/pred 直接单测打分逻辑（见 tests/test_evals_score.py）。
"""
from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field as dc_field
from difflib import SequenceMatcher
from typing import Callable, Optional

from contract_archive.extraction.normalize import normalize_date
from contract_archive.schemas import (
    DOC_TYPES,
    DocumentExtraction,
    LabeledAmount,
)

# ============================================================================
# 配置：字段权重 / 关键字段 / F-beta / 阈值。显式写在这里、可审计——别藏进代码逻辑。
# 权重体现业务重要性：当事人/金额/完整性 issues 是高风险字段，权重高于次要字段。
# ============================================================================

FIELD_WEIGHTS: dict[str, float] = {
    "doc_type": 2.0,            # 分类错 → 后续字段语义全偏，权重高
    "parties": 2.0,            # 当事人是合同核心主体
    "primary_amount": 2.0,     # 主金额
    "monthly_property_fee": 1.0,  # 月物业费派生估算（单价×面积，代码算）
    "amounts": 1.5,
    "completeness_issues": 3.0,  # 完整性核查是这个项目的痛点来源，最高权重
    "primary_date": 1.0,
    "key_dates": 1.0,
    "fields": 1.0,
    "seals": 1.0,
    "obligations": 1.0,
    "sub_agreements": 1.5,
    "title": 0.5,              # 主观，弱信号
    "summary": 0.5,            # 主观，弱信号
}

# 关键字段：候选模型在这些字段上必须逐项非劣，任一塌方一票否决（不进加权总分平均）。
CRITICAL_FIELDS = ("doc_type", "parties", "primary_amount", "completeness_issues")

ISSUE_FBETA = 2.0          # 完整性 issues 偏召回
STR_SIM_THRESHOLD = 0.6    # 文本相似度匹配阈值
PAGE_RE = re.compile(r"第\s*(\d+)\s*页")

# 替换决策参数（report / seal 共用）：非劣边际 + JSON 解析成功率绝对地板。
DELTA = 0.03               # 候选指标允许比 champion 低的上限（非劣性 margin）
PARSE_FLOOR = 0.98         # JSON 解析成功率绝对地板（一条非法 JSON=整篇抽取归零）


# ============================================================================
# 基础工具
# ============================================================================


def normalize_str(s: Optional[str]) -> str:
    """归一化：NFKC（全角→半角）+ 去所有空白 + casefold。空/None → 空串。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    return s.casefold()


def str_sim(a: str, b: str) -> float:
    """两串相似度 [0,1]。归一化后用 difflib ratio。"""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def extract_page(evidence: str) -> Optional[int]:
    """从 evidence 文本里抽页码（"第X页"）。抽不到返回 None。"""
    if not evidence:
        return None
    m = PAGE_RE.search(evidence)
    return int(m.group(1)) if m else None


# 角色称谓 → 甲方/乙方 归一化映射。合同里"甲方/乙方"与具体角色称谓（出卖人/买受人等）
# 语义等价，但 issue.item 文本不同会让相似度匹配失败（如 gold "主协议·乙方签章" vs
# pred "主协议·买受人签章"），制造虚假 FP/FN、压低 champion 自身得分、污染 gate 判定。
# 打 completeness issue 前先把称谓统一到甲方/乙方，只比"哪一方·哪个要素缺/异常"，
# 不被称谓体系差异干扰。注意：只用于 issue.item 相似度，不动 parties（机构全称不该被并称）。
_ROLE_TO_CANON = {
    # 甲方系（出让/出租/转让/发包/供货/出借/放贷方）
    "出卖人": "甲方", "卖方": "甲方", "出让人": "甲方", "转让人": "甲方", "转让方": "甲方",
    "出租方": "甲方", "出租人": "甲方", "发包方": "甲方", "发包人": "甲方",
    "供方": "甲方", "供货方": "甲方", "出借方": "甲方", "贷款人": "甲方",
    # 乙方系（受让/承租/承包/购买/借款方）
    "买受人": "乙方", "买方": "乙方", "受让人": "乙方", "受让方": "乙方", "购房人": "乙方",
    "承租方": "乙方", "承租人": "乙方", "承包方": "乙方", "承包人": "乙方",
    "需方": "乙方", "采购方": "乙方", "借款人": "乙方", "购买方": "乙方",
}


def normalize_issue_item(s: Optional[str]) -> str:
    """issue.item 归一化：通用 normalize 后，把角色称谓统一到甲方/乙方，消除术语差异虚假失配。"""
    out = normalize_str(s)
    for term, canon in _ROLE_TO_CANON.items():
        out = out.replace(normalize_str(term), canon)
    # 称谓归一后，"乙方买受人签章"会变成"乙方乙方签章"——叠词反而拉低与"乙方签章"的相似度。
    # 去掉相邻叠词，让"角色+同义角色"的冗余写法（gold 或 pred 任一侧）规整到单一称谓。
    out = out.replace("乙方乙方", "乙方").replace("甲方甲方", "甲方")
    return out


def _issue_party(item_norm: str) -> str:
    """从已归一化的 issue.item 判定指向哪一方：'甲'/'乙'/'双'/''（无主体）。
    '双'=同时提及甲乙、或明示「双方」——与单方 '甲'/'乙' 视为不同方、不得互配
    （否则"双方签章"会因与"甲方签章"仅一字之差被 str_sim 误配成 TP）。"""
    if "双方" in item_norm:
        return "双"
    jia, yi = "甲方" in item_norm, "乙方" in item_norm
    if jia and yi:
        return "双"
    if jia:
        return "甲"
    if yi:
        return "乙"
    return ""


# ============================================================================
# 打分单元
# ============================================================================


@dataclass
class FieldScore:
    """单字段打分：统一成 TP/FP/FN，标量字段也折算进来（gold 有 pred 缺=FN 等）。"""

    field: str
    channel: str           # scalar | set | fbeta | class
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    weight: float = 1.0
    beta: float = 1.0
    critical: bool = False
    detail: str = ""

    @property
    def support(self) -> float:
        """gold 侧应有的数量（TP+FN）。0 表示该字段在本 case 无期望。"""
        return self.tp + self.fn

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0   # 无产出且无期望 → 视为不失分

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 1.0

    def fbeta(self) -> float:
        # 真空（无 gold 无 pred，本 case 对该字段无期望）→ 1.0，不失分。
        if self.tp + self.fp + self.fn == 0:
            return 1.0
        p, r = self.precision(), self.recall()
        b2 = self.beta * self.beta
        d = b2 * p + r
        # p=r=0（全错，如标量值抽错→fp=fn=1）→ 0.0。不可落回 1.0：那是把全错当满分。
        return (1 + b2) * p * r / d if d else 0.0


def _align(
    gold: list, pred: list, sim: Callable[[object, object], float], threshold: float
) -> tuple[int, int, int, list[tuple[int, int]]]:
    """
    贪心对齐两个列表：算所有 pred×gold 相似度，从高到低配对（相似度≥阈值才配）。
    返回 (tp, fp, fn, matched_index_pairs)。先上贪心——证明有歧义再上匈牙利。
    """
    cands: list[tuple[float, int, int]] = []
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            s = sim(g, p)
            if s >= threshold:
                cands.append((s, i, j))
    cands.sort(reverse=True, key=lambda x: x[0])
    g_used: set[int] = set()
    p_used: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, i, j in cands:
        if i in g_used or j in p_used:
            continue
        g_used.add(i)
        p_used.add(j)
        matched.append((i, j))
    tp = len(matched)
    return tp, len(pred) - tp, len(gold) - tp, matched


def score_scalar(
    field: str, gold_val: Optional[str], pred_val: Optional[str], *, is_date: bool = False
) -> FieldScore:
    """标量字段：gold/pred 都空→不计；一致→TP；不一致→gold 侧 FN + pred 侧 FP。"""
    norm = (lambda v: normalize_date(v) or "") if is_date else normalize_str
    g = norm(gold_val) if gold_val else ""
    p = norm(pred_val) if pred_val else ""
    fs = FieldScore(field, "scalar", weight=FIELD_WEIGHTS.get(field, 1.0),
                    critical=field in CRITICAL_FIELDS)
    if not g and not p:
        return fs
    if g and g == p:
        fs.tp = 1
    else:
        if g:
            fs.fn = 1
        if p:
            fs.fp = 1
    return fs


def score_scalar_amount(field: str, gold_val: Optional[float], pred_val: Optional[float]) -> FieldScore:
    """
    标量金额按**归一化数值**比，不比原文——同一金额 "¥200,000.00" 与 "人民币贰拾万元整"
    数值相同应算对。生产已用 parse_money_value 把文本算成 value，这里直接比 value。
    """
    fs = FieldScore(field, "scalar", weight=FIELD_WEIGHTS.get(field, 1.0),
                    critical=field in CRITICAL_FIELDS)
    gv = round(gold_val, 2) if gold_val is not None else None
    pv = round(pred_val, 2) if pred_val is not None else None
    if gv is None and pv is None:
        return fs
    if gv is not None and gv == pv:
        fs.tp = 1
    else:
        if gv is not None:
            fs.fn = 1
        if pv is not None:
            fs.fp = 1
    return fs


def _amount_equal(g: LabeledAmount, p: LabeledAmount) -> bool:
    """金额 exact：归一化数值 + is_total_component 布尔都一致才算对。"""
    gv = round(g.value, 2) if g.value is not None else None
    pv = round(p.value, 2) if p.value is not None else None
    return gv == pv and bool(g.is_total_component) == bool(p.is_total_component)


def score_amounts(gold: list[LabeledAmount], pred: list[LabeledAmount]) -> FieldScore:
    """金额列表：按 label 配对 → 配上但内容错的降级为 FN+FP（exact 语义）。"""
    fs = FieldScore("amounts", "set", weight=FIELD_WEIGHTS["amounts"])
    tp, fp, fn, matched = _align(
        gold, pred,
        lambda g, p: 1.0 if normalize_str(g.label) == normalize_str(p.label) else 0.0,
        threshold=1.0,
    )
    wrong = sum(1 for i, j in matched if not _amount_equal(gold[i], pred[j]))
    fs.tp = tp - wrong
    fs.fp = fp + wrong
    fs.fn = fn + wrong
    return fs


def score_str_list(field: str, gold: list[str], pred: list[str]) -> FieldScore:
    """纯字符串列表（如 parties）：归一化相等即配对。"""
    fs = FieldScore(field, "set", weight=FIELD_WEIGHTS.get(field, 1.0),
                    critical=field in CRITICAL_FIELDS)
    fs.tp, fs.fp, fs.fn, _ = _align(
        gold, pred,
        lambda g, p: 1.0 if normalize_str(g) == normalize_str(p) else 0.0,
        threshold=1.0,
    )
    return fs


def score_labeled_dates(gold: list, pred: list) -> FieldScore:
    """key_dates：按 label 配对，内容比归一化日期。"""
    fs = FieldScore("key_dates", "set", weight=FIELD_WEIGHTS["key_dates"])

    def sim(g, p):
        return 1.0 if normalize_str(g.label) == normalize_str(p.label) else 0.0

    tp, fp, fn, matched = _align(gold, pred, sim, threshold=1.0)
    wrong = sum(
        1 for i, j in matched
        if (normalize_date(gold[i].date) or "") != (normalize_date(pred[j].date) or "")
    )
    fs.tp, fs.fp, fs.fn = tp - wrong, fp + wrong, fn + wrong
    return fs


def score_labeled_values(field: str, gold: list, pred: list) -> FieldScore:
    """fields：按 label 配对，内容比归一化 value。"""
    fs = FieldScore(field, "set", weight=FIELD_WEIGHTS.get(field, 1.0))

    def sim(g, p):
        return 1.0 if normalize_str(g.label) == normalize_str(p.label) else 0.0

    tp, fp, fn, matched = _align(gold, pred, sim, threshold=1.0)
    wrong = sum(1 for i, j in matched if normalize_str(gold[i].value) != normalize_str(pred[j].value))
    fs.tp, fs.fp, fs.fn = tp - wrong, fp + wrong, fn + wrong
    return fs


def score_seals(gold: list, pred: list) -> FieldScore:
    """印章：owner+raw_text 文本相似度配对（OCR 残缺，用模糊匹配）。"""
    fs = FieldScore("seals", "set", weight=FIELD_WEIGHTS["seals"])

    def sim(g, p):
        gs = normalize_str((g.owner or "") + (g.raw_text or ""))
        ps = normalize_str((p.owner or "") + (p.raw_text or ""))
        return str_sim(gs, ps)

    fs.tp, fs.fp, fs.fn, _ = _align(gold, pred, sim, threshold=STR_SIM_THRESHOLD)
    return fs


def score_obligations(gold: list, pred: list) -> FieldScore:
    """义务：actor 相等 + action 相似 配对。"""
    fs = FieldScore("obligations", "set", weight=FIELD_WEIGHTS["obligations"])

    def sim(g, p):
        if g.actor != p.actor:
            return 0.0
        return str_sim(normalize_str(g.action), normalize_str(p.action))

    fs.tp, fs.fp, fs.fn, _ = _align(gold, pred, sim, threshold=STR_SIM_THRESHOLD)
    return fs


def score_sub_agreements(gold: list, pred: list) -> FieldScore:
    """补充协议：按 title 相似配对；内容核 sign_date + 印章数量（浅层，深层递归留作增强）。"""
    fs = FieldScore("sub_agreements", "set", weight=FIELD_WEIGHTS["sub_agreements"])

    def sim(g, p):
        return str_sim(normalize_str(g.title), normalize_str(p.title))

    tp, fp, fn, matched = _align(gold, pred, sim, threshold=STR_SIM_THRESHOLD)
    wrong = 0
    for i, j in matched:
        g, p = gold[i], pred[j]
        date_ok = (normalize_date(g.sign_date) or "") == (normalize_date(p.sign_date) or "")
        seal_ok = len(g.seals) == len(p.seals)
        if not (date_ok and seal_ok):
            wrong += 1
    fs.tp, fs.fp, fs.fn = tp - wrong, fp + wrong, fn + wrong
    return fs


def score_completeness_issues(gold: list, pred: list) -> FieldScore:
    """
    完整性 issues：F-beta(β=2) 偏召回。对齐 key=(category, 页码)，文本相似度兜底。
    页码弱匹配：同页=1.0，缺页码=0.6，±1 页=0.7，否则不配。
    """
    fs = FieldScore("completeness_issues", "fbeta", weight=FIELD_WEIGHTS["completeness_issues"],
                    beta=ISSUE_FBETA, critical=True)

    def sim(g, p):
        if g.category != p.category:
            return 0.0
        gp, pp = extract_page(g.evidence), extract_page(p.evidence)
        if gp is None or pp is None:
            page = 0.6
        elif gp == pp:
            page = 1.0
        elif abs(gp - pp) <= 1:
            page = 0.7
        else:
            return 0.0
        gi, pi = normalize_issue_item(g.item), normalize_issue_item(p.item)
        # 称谓归一后"甲方/乙方"仅一字之差，str_sim 会误判为高相似（"甲方签章"vs"乙方签章"≈0.9）；
        # 故先要求明确指向的一方一致，再比要素文本，避免把不同方的同类缺陷误配。
        if _issue_party(gi) and _issue_party(pi) and _issue_party(gi) != _issue_party(pi):
            return 0.0
        return page if str_sim(gi, pi) >= STR_SIM_THRESHOLD else 0.0

    fs.tp, fs.fp, fs.fn, _ = _align(gold, pred, sim, threshold=0.5)
    return fs


# ============================================================================
# 信封级聚合
# ============================================================================


@dataclass
class EnvelopeScore:
    """单 case × 单模型的完整打分。"""

    case_id: str
    fields: list[FieldScore] = dc_field(default_factory=list)
    doc_type_pair: tuple[str, str] = ("", "")   # (gold, pred) 供混淆矩阵
    parse_ok: bool = True                       # pred 是否非空（JSON 解析/调用成功）
    sig_recall_tp: float = 0.0                  # 签章类缺陷召回（关键硬门槛）
    sig_recall_fn: float = 0.0

    def weighted_score(self) -> float:
        """加权字段分（仅参考，不作单独通过条件）。各字段 F-beta × 权重 求加权平均。"""
        num = den = 0.0
        for fs in self.fields:
            num += fs.fbeta() * fs.weight
            den += fs.weight
        return num / den if den else 1.0

    def sig_recall(self) -> Optional[float]:
        d = self.sig_recall_tp + self.sig_recall_fn
        return self.sig_recall_tp / d if d else None


def score_envelope(case_id: str, gold: DocumentExtraction, pred: DocumentExtraction) -> EnvelopeScore:
    """对一份 case 的 gold/pred 信封逐字段打分。"""
    es = EnvelopeScore(case_id=case_id)
    # parse_ok：pred 完全为空（llm_model 缺）= 调用/解析失败，是一票否决的格式问题。
    es.parse_ok = pred.llm_model is not None or bool(pred.doc_type and pred.doc_type != "其他") \
        or bool(pred.parties or pred.fields or pred.amounts)
    es.doc_type_pair = (gold.doc_type, pred.doc_type)

    es.fields.append(score_scalar("doc_type", gold.doc_type, pred.doc_type))
    es.fields.append(score_scalar("title", gold.title, pred.title))
    es.fields.append(score_scalar("summary", gold.summary, pred.summary))
    es.fields.append(score_scalar("primary_date", gold.primary_date, pred.primary_date, is_date=True))
    es.fields.append(score_scalar_amount("primary_amount", gold.primary_amount_value, pred.primary_amount_value))
    # 月物业费派生值：代码确定性算（Σ按㎡单价 × 建筑面积），间接评 LLM 单价+面积抽取质量。
    es.fields.append(score_scalar_amount(
        "monthly_property_fee", gold.monthly_property_fee_value, pred.monthly_property_fee_value))
    es.fields.append(score_str_list("parties", gold.parties, pred.parties))
    es.fields.append(score_amounts(gold.amounts, pred.amounts))
    es.fields.append(score_labeled_dates(gold.key_dates, pred.key_dates))
    es.fields.append(score_labeled_values("fields", gold.fields, pred.fields))
    es.fields.append(score_seals(gold.seals, pred.seals))
    es.fields.append(score_obligations(gold.obligations, pred.obligations))
    es.fields.append(score_sub_agreements(gold.sub_agreements, pred.sub_agreements))

    gold_issues = gold.completeness.issues if gold.completeness else []
    pred_issues = pred.completeness.issues if pred.completeness else []
    es.fields.append(score_completeness_issues(gold_issues, pred_issues))

    # 签章类缺陷召回单算（关键硬门槛：漏报签章缺陷=合同蒙混过关）。
    gold_sig = [i for i in gold_issues if i.category == "signature"]
    pred_sig = [i for i in pred_issues if i.category == "signature"]
    sig_fs = score_completeness_issues(gold_sig, pred_sig)
    es.sig_recall_tp, es.sig_recall_fn = sig_fs.tp, sig_fs.fn
    return es


# ============================================================================
# 非劣性 / 置信区间（决策用 CI 下界，不看点估计）
# ============================================================================


def bootstrap_ci(values: list[float], n: int = 2000, alpha: float = 0.05, seed: int = 0
                 ) -> tuple[float, float, float]:
    """
    对逐 case 指标做 bootstrap 重采样，返回 (均值, CI下界, CI上界)。
    决策依据是 CI 下界——小样本下 CI 会很宽，如实暴露"看不出差异"其实是 power 不足。
    """
    if not values:
        return (1.0, 1.0, 1.0)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, mean, mean)
    rng = random.Random(seed)
    means = []
    k = len(values)
    for _ in range(n):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return (mean, lo, hi)


def is_valid_doc_type(dt: str) -> bool:
    """枚举合规：doc_type 是否在规范类型内（生产已 coerce 到'其他'，此处供 raw 校验复用）。"""
    return dt in DOC_TYPES
