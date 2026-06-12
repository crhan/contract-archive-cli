"""
评测打分器单测：纯函数、不触网。构造合成 gold/pred 验证 TP/FP/FN 与门禁信号正确，
并校验 4 个种子 case 的 gold.json 都是合法 DocumentExtraction。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from contract_archive.schemas import DOC_TYPES, DocumentExtraction
from evals.score import (
    CRITICAL_FIELDS,
    bootstrap_ci,
    score_amounts,
    score_completeness_issues,
    score_envelope,
    score_str_list,
)

CASES_DIR = Path(__file__).resolve().parent.parent / "evals" / "cases" / "extraction"


def _load_gold(case_id: str) -> DocumentExtraction:
    return DocumentExtraction.model_validate(
        json.loads((CASES_DIR / case_id / "gold.json").read_text(encoding="utf-8"))
    )


def test_seed_golds_are_valid_schema():
    """所有种子 gold.json 必须能解析为 DocumentExtraction（否则评测从源头就错）。"""
    case_dirs = [p for p in CASES_DIR.iterdir() if p.is_dir()]
    assert case_dirs, "没有种子 case"
    for cd in case_dirs:
        env = _load_gold(cd.name)
        assert env.doc_type in DOC_TYPES


def test_perfect_match_scores_one():
    """pred == gold → 每字段 fbeta=1，加权分=1。"""
    gold = _load_gold("c01_carpark_with_subagreement")
    pred = copy.deepcopy(gold)
    pred.llm_model = "qwen3.7-max"
    es = score_envelope("c01", gold, pred)
    assert es.parse_ok
    for fs in es.fields:
        assert fs.fbeta() == pytest.approx(1.0), f"{fs.field} 不该失分"
    assert es.weighted_score() == pytest.approx(1.0)


def test_missing_party_drops_recall():
    """漏抽一个当事人 → parties recall 掉、fn=1。"""
    gold = score_str_list("parties", ["示例置业有限公司", "张三"], ["示例置业有限公司"])
    assert gold.tp == 1 and gold.fn == 1 and gold.fp == 0
    assert gold.recall() == pytest.approx(0.5)


def test_amount_wrong_value_is_fn_and_fp():
    """金额数值错（量级错）→ 既 FN 又 FP，不被容差放过。"""
    gold = _load_gold("c01_carpark_with_subagreement")
    pred = copy.deepcopy(gold)
    pred.amounts[0].value = 20000.0  # 20万写成2万，量级错
    fs = score_amounts(gold.amounts, pred.amounts)
    assert fs.tp == 0 and fs.fn == 1 and fs.fp == 1


def test_is_total_component_flip_is_caught():
    """is_total_component 翻转 → 金额算错，判 FN+FP。"""
    gold = _load_gold("c02_income_certificate")
    pred = copy.deepcopy(gold)
    # 把"年度股权应税收益"误标为不计入合计
    pred.amounts[1].is_total_component = False
    fs = score_amounts(gold.amounts, pred.amounts)
    assert fs.fn >= 1 and fs.fp >= 1


def test_missed_signature_issue_kills_recall():
    """漏报补充协议签章缺陷（致命）→ 签章召回为 0。"""
    gold = _load_gold("c01_carpark_with_subagreement")
    pred = copy.deepcopy(gold)
    pred.completeness.issues = []          # 候选模型没看出乙方补充协议没签
    pred.completeness.status = "complete"
    es = score_envelope("c01", gold, pred)
    assert es.sig_recall() == pytest.approx(0.0)
    issue_fs = next(f for f in es.fields if f.field == "completeness_issues")
    assert issue_fs.recall() == pytest.approx(0.0)


def test_false_positive_issue_drops_precision_not_recall():
    """误报一个缺陷（gold 没有）→ precision 掉、recall 不变。"""
    gold_issues = []
    from contract_archive.schemas import CompletenessIssue
    pred_issues = [CompletenessIssue(item="甲方签章", category="signature",
                                     detail="疑似空白", evidence="第 1 页")]
    fs = score_completeness_issues(gold_issues, pred_issues)
    assert fs.fp == 1 and fs.fn == 0


def test_page_mismatch_prevents_match():
    """同类缺陷但页码差很远 → 不应配上（定位错也是错）。"""
    from contract_archive.schemas import CompletenessIssue
    g = [CompletenessIssue(item="乙方签章", category="signature", detail="x", evidence="第 2 页")]
    p = [CompletenessIssue(item="乙方签章", category="signature", detail="x", evidence="第 8 页")]
    fs = score_completeness_issues(g, p)
    assert fs.tp == 0 and fs.fn == 1 and fs.fp == 1


def test_role_synonym_matches_issue():
    """角色称谓归一化：gold「乙方签章」与 pred「买受人签章」同页同义 → 匹配 TP，消除虚假 FP/FN；
    但「出卖人(→甲方)」与「买受人(→乙方)」是不同方，仍须区分、不得误配。"""
    from contract_archive.schemas import CompletenessIssue
    g = [CompletenessIssue(item="主协议·乙方签章", category="signature", detail="x", evidence="第 8 页")]
    p = [CompletenessIssue(item="主协议·买受人签章", category="signature", detail="x", evidence="第 8 页")]
    fs = score_completeness_issues(g, p)
    assert fs.tp == 1 and fs.fp == 0 and fs.fn == 0
    p2 = [CompletenessIssue(item="主协议·出卖人签章", category="signature", detail="x", evidence="第 8 页")]
    fs2 = score_completeness_issues(g, p2)
    assert fs2.tp == 0 and fs2.fn == 1 and fs2.fp == 1


def test_role_redundant_item_not_penalized():
    """角色冗余写法：gold「乙方买受人签章」归一不应产生叠词「乙方乙方」拉低相似度，
    须与 pred「乙方签章」同页匹配（回归：c08 认购协议曾因此虚假 FP+FN）。"""
    from contract_archive.schemas import CompletenessIssue
    g = [CompletenessIssue(item="乙方买受人签章", category="signature", detail="x", evidence="第 5 页")]
    p = [CompletenessIssue(item="主协议·乙方签章", category="signature", detail="x", evidence="第 5 页")]
    fs = score_completeness_issues(g, p)
    assert fs.tp == 1 and fs.fp == 0 and fs.fn == 0


def test_double_party_not_matched_to_single():
    """「双方签章」与「甲方/乙方签章」仅一字之差，不得被 str_sim 误配（_issue_party 三态区分）；
    「双方」对「双方」应匹配。"""
    from contract_archive.schemas import CompletenessIssue
    g = [CompletenessIssue(item="主协议·双方签章", category="signature", detail="x", evidence="第 8 页")]
    for single in ("主协议·甲方签章", "主协议·乙方签章"):
        p = [CompletenessIssue(item=single, category="signature", detail="x", evidence="第 8 页")]
        fs = score_completeness_issues(g, p)
        assert fs.tp == 0 and fs.fn == 1 and fs.fp == 1, f"双方 vs {single} 不该配上"
    p2 = [CompletenessIssue(item="主协议·双方签章", category="signature", detail="x", evidence="第 8 页")]
    assert score_completeness_issues(g, p2).tp == 1


def test_role_alias_seller_buyer_aliases():
    """补充别名：「卖方签章」(→甲方) 与 gold「出卖人签章」(→甲方) 同义匹配；卖方 vs 买方 不配。"""
    from contract_archive.schemas import CompletenessIssue
    g = [CompletenessIssue(item="出卖人签章", category="signature", detail="x", evidence="第 3 页")]
    p = [CompletenessIssue(item="卖方签章", category="signature", detail="x", evidence="第 3 页")]
    assert score_completeness_issues(g, p).tp == 1
    p2 = [CompletenessIssue(item="买方签章", category="signature", detail="x", evidence="第 3 页")]
    fs2 = score_completeness_issues(g, p2)
    assert fs2.tp == 0 and fs2.fn == 1 and fs2.fp == 1


def test_empty_pred_is_parse_failure():
    """空信封（调用/解析失败）→ parse_ok=False。"""
    gold = _load_gold("c03_vat_invoice")
    pred = DocumentExtraction()  # 全空，llm_model=None
    es = score_envelope("c03", gold, pred)
    assert es.parse_ok is False


def test_fbeta_wrong_is_zero_not_one():
    """回归：p=r=0（全错，如标量抽错 fp=fn=1）的 fbeta 必须为 0，不能落回 1.0 把全错当满分。"""
    from evals.score import FieldScore
    assert FieldScore("x", "scalar", tp=0, fp=1, fn=1).fbeta() == 0.0   # 全错
    assert FieldScore("x", "scalar", tp=0, fp=0, fn=0).fbeta() == 1.0   # 真空：无期望
    assert FieldScore("x", "scalar", tp=1, fp=0, fn=0).fbeta() == 1.0   # 全对
    assert FieldScore("x", "set", tp=1, fp=1, fn=1).fbeta() == pytest.approx(0.5)


def test_wrong_scalar_field_scores_zero_in_envelope():
    """信封级回归：primary_amount 抽错 → 该字段 fbeta=0（门禁 field_values 才能正确卡）。"""
    gold = _load_gold("c03_vat_invoice")
    pred = copy.deepcopy(gold)
    pred.llm_model = "x"
    pred.primary_amount_value = 99999.99   # 抽错主金额（按数值比）
    es = score_envelope("c03", gold, pred)
    pa = next(f for f in es.fields if f.field == "primary_amount")
    assert pa.fbeta() == 0.0


def test_primary_amount_scored_by_value_not_text():
    """金额按归一化数值比：文本写法不同但数值相同 → 算对（不假阴性）。"""
    gold = _load_gold("c01_carpark_with_subagreement")
    pred = copy.deepcopy(gold)
    pred.llm_model = "x"
    pred.primary_amount_text = "¥200,000.00"   # 文本不同
    pred.primary_amount_value = 200000.0       # 数值相同
    es = score_envelope("c01", gold, pred)
    pa = next(f for f in es.fields if f.field == "primary_amount")
    assert pa.fbeta() == 1.0


def test_bootstrap_ci_bounds():
    mean, lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 0.0], n=500)
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_critical_fields_are_flagged():
    """score_envelope 标记的 critical 字段须恰好等于 CRITICAL_FIELDS。"""
    gold = _load_gold("c04_lease_complete")
    es = score_envelope("c04", gold, copy.deepcopy(gold))
    flagged = {fs.field for fs in es.fields if fs.critical}
    assert flagged == set(CRITICAL_FIELDS)
