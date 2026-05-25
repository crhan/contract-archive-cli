"""
评测报告：把 results/ 聚合成 gate 决策表。

核心理念（吸收方法学评审）：替换是风险问题，不是精度问题。**默认不可替换，候选必须
逐项证明非劣**才放行——不用"加权平均比大小"（平均会抹平关键文档的塌方）。

门禁（全过才有资格谈成本）：
  1. JSON 解析成功率 ≥ PARSE_FLOOR（便宜模型最常见的退化是吐非法 JSON，一条失败=整篇归零）
  2. 签章类缺陷召回 ≥ champion − δ（漏报签章=合同蒙混过关，最致命）
  3. 每个关键字段：候选 per-case 指标 bootstrap CI 下界 ≥ champion 均值 − δ
  4. 以上全过 → ELIGIBLE，才比成本/延迟（约束优化：质量是闸门，成本是目标）

加权总分只作参考列，不作单独通过条件。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional

from contract_archive.schemas import DocumentExtraction

from .run import DEFAULT_CASES
from .score import CRITICAL_FIELDS, EnvelopeScore, bootstrap_ci, score_envelope

# 决策参数（显式、可审计）
DELTA = 0.03          # 非劣边际：候选指标允许比 champion 低的上限
PARSE_FLOOR = 0.98    # JSON 解析成功率绝对地板

# 价格表（¥ / 百万 token，查询日期 2026-05-25，**估算，以阿里云百炼控制台为准**）。
# 文本模型；带小版本号的是当前在售命名，无版本别名会随官方升级漂移（评测请锁 snapshot）。
PRICE_RMB_PER_M: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "qwen3.7-max": (2.5, 10.0),
    "qwen3-max": (2.5, 10.0),
    "qwen-max": (2.5, 10.0),
    "qwen3.6-plus": (0.8, 4.8),
    "qwen-plus": (0.8, 4.8),
    "qwen3.6-flash": (0.2, 2.0),
    "qwen-flash": (0.2, 2.0),
    "qwen-turbo": (0.3, 0.6),
    "qwen3-235b-a22b": (0.7 * 7, 2.8 * 7),  # 国际美元价×约7 粗折人民币，待核
}
DEFAULT_PRICE = (2.5, 10.0)


# ============================================================================
# 加载 + 打分
# ============================================================================


def load_gold(cases_dir: Path, suite: str, case_id: str) -> DocumentExtraction:
    gold_path = cases_dir / suite / case_id / "gold.json"
    return DocumentExtraction.model_validate(json.loads(gold_path.read_text(encoding="utf-8")))


def load_results(results_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict]]]:
    """读 results 目录 → (run_meta, {model: [case_record, ...]})。"""
    run_meta = json.loads((results_dir / "run_meta.json").read_text(encoding="utf-8"))
    by_model: dict[str, list[dict]] = {}
    for model in run_meta["models"]:
        model_dir = results_dir / model.replace("/", "_")
        records = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(model_dir.glob("*.json"))]
        by_model[model] = records
    return run_meta, by_model


class ModelAgg:
    """单模型在整个 suite 上的聚合指标。"""

    def __init__(self, model: str):
        self.model = model
        self.envelopes: list[EnvelopeScore] = []
        self.latencies: list[float] = []
        self.cost_per_doc: list[float] = []
        self.determinism_flags: list[bool] = []   # 多次跑结果是否一致

    def parse_rate(self) -> float:
        if not self.envelopes:
            return 0.0
        return sum(1 for e in self.envelopes if e.parse_ok) / len(self.envelopes)

    def field_values(self, field: str) -> list[float]:
        """该字段逐 case 的 F-beta（仅取有 gold 或有 pred 的 case，避免空 case 注水）。"""
        vals = []
        for e in self.envelopes:
            for fs in e.fields:
                if fs.field == field and (fs.tp + fs.fp + fs.fn) > 0:
                    vals.append(fs.fbeta())
        return vals

    def sig_recall(self) -> Optional[float]:
        tp = sum(e.sig_recall_tp for e in self.envelopes)
        fn = sum(e.sig_recall_fn for e in self.envelopes)
        return tp / (tp + fn) if (tp + fn) else None

    def weighted_scores(self) -> list[float]:
        return [e.weighted_score() for e in self.envelopes]

    def p50_p95(self) -> tuple[float, float]:
        if not self.latencies:
            return (0.0, 0.0)
        s = sorted(self.latencies)
        return (statistics.median(s), s[min(len(s) - 1, int(0.95 * len(s)))])

    def avg_cost_per_1k(self) -> Optional[float]:
        vals = [c for c in self.cost_per_doc if c is not None]
        return (sum(vals) / len(vals)) * 1000 if vals else None


def _cost_of(model: str, usage: Optional[dict]) -> Optional[float]:
    """据 usage 估单文档成本（¥）。usage 缺失返回 None。"""
    if not usage:
        return None
    pin, pout = PRICE_RMB_PER_M.get(model, DEFAULT_PRICE)
    itok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    otok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return itok / 1e6 * pin + otok / 1e6 * pout


def aggregate(by_model: dict[str, list[dict]], run_meta: dict, cases_dir: Path) -> dict[str, ModelAgg]:
    suite = run_meta["suite"]
    out: dict[str, ModelAgg] = {}
    for model, records in by_model.items():
        agg = ModelAgg(model)
        for rec in records:
            gold = load_gold(cases_dir, suite, rec["case_id"])
            runs = rec["runs"]
            pred = DocumentExtraction.model_validate(runs[0]["pred"])
            agg.envelopes.append(score_envelope(rec["case_id"], gold, pred))
            agg.latencies.append(runs[0]["latency_s"])
            agg.cost_per_doc.append(_cost_of(model, runs[0].get("usage")))
            if len(runs) > 1:
                # 自一致性：多次跑的 doc_type + parties 是否完全一致
                types = {r["pred"].get("doc_type") for r in runs}
                agg.determinism_flags.append(len(types) == 1)
        out[model] = agg
    return out


# ============================================================================
# Gate 决策
# ============================================================================


def gate_verdict(champ: ModelAgg, cand: ModelAgg) -> tuple[bool, list[str]]:
    """候选 vs champion 跑全部门禁。返回 (是否通过, 失败原因列表)。"""
    reasons: list[str] = []

    # 门禁 1：解析成功率绝对地板
    if cand.parse_rate() < PARSE_FLOOR:
        reasons.append(f"JSON 解析成功率 {cand.parse_rate():.0%} < 地板 {PARSE_FLOOR:.0%}")

    # 门禁 2：签章缺陷召回非劣
    cs, ch = cand.sig_recall(), champ.sig_recall()
    if ch is not None and cs is not None and cs < ch - DELTA:
        reasons.append(f"签章缺陷召回 {cs:.0%} < champion {ch:.0%} − {DELTA:.0%}（漏报致命）")

    # 门禁 3：每个关键字段 CI 下界非劣
    for f in CRITICAL_FIELDS:
        champ_vals, cand_vals = champ.field_values(f), cand.field_values(f)
        if not champ_vals or not cand_vals:
            continue
        champ_mean = sum(champ_vals) / len(champ_vals)
        _, cand_lo, _ = bootstrap_ci(cand_vals)
        if cand_lo < champ_mean - DELTA:
            reasons.append(
                f"关键字段 {f}：候选 CI 下界 {cand_lo:.2f} < champion 均值 {champ_mean:.2f} − {DELTA}")

    return (len(reasons) == 0, reasons)


# ============================================================================
# 渲染
# ============================================================================


def render_markdown(run_meta: dict, aggs: dict[str, ModelAgg], champion: str) -> str:
    champ = aggs[champion]
    lines: list[str] = []
    lines.append("# 换模型评测报告（gate 决策）\n")
    lines.append(f"- suite: `{run_meta['suite']}`　champion: `{champion}`　"
                 f"case 数: {len(champ.envelopes)}　repeat: {run_meta.get('repeat', 1)}")
    lines.append(f"- 决策规则：**默认不可替换**，候选须逐项非劣（δ={DELTA}，解析地板={PARSE_FLOOR:.0%}）才放行")
    n = len(champ.envelopes)
    if n < 30:
        lines.append(f"- ⚠️ **样本仅 {n} 例，CI 会很宽、统计效力不足**——结论仅供形态参考，"
                     f"真实决策需按 README 扩到每分层 80-150 例。")
    lines.append("")

    # 主表
    lines.append("## 决策总表\n")
    lines.append("| 模型 | 解析率 | 加权分(均值/CI下界) | 签章召回 | p50延迟 | p95延迟 | ¥/千文档 | 裁决 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model, agg in aggs.items():
        ws = agg.weighted_scores()
        mean, lo, _ = bootstrap_ci(ws)
        p50, p95 = agg.p50_p95()
        cost = agg.avg_cost_per_1k()
        cost_s = f"{cost:.1f}" if cost is not None else "—"
        sig = agg.sig_recall()
        sig_s = f"{sig:.0%}" if sig is not None else "—"
        if model == champion:
            verdict = "🏆 champion"
        else:
            ok, reasons = gate_verdict(champ, agg)
            verdict = "✅ 可替换" if ok else "❌ 不可替换"
        lines.append(f"| `{model}` | {agg.parse_rate():.0%} | {mean:.2f} / {lo:.2f} | "
                     f"{sig_s} | {p50:.2f}s | {p95:.2f}s | {cost_s} | {verdict} |")
    lines.append("")

    # 候选失败原因
    for model, agg in aggs.items():
        if model == champion:
            continue
        ok, reasons = gate_verdict(champ, agg)
        lines.append(f"### `{model}` 门禁明细")
        if ok:
            cost, ccost = agg.avg_cost_per_1k(), champ.avg_cost_per_1k()
            save = f"，成本约为 champion 的 {cost / ccost:.0%}" if (cost and ccost) else ""
            lines.append(f"- ✅ 通过全部质量门禁{save}。**建议：在此基础上可替换。**\n")
        else:
            lines.append("- ❌ 未通过，失败门禁：")
            for r in reasons:
                lines.append(f"  - {r}")
            lines.append("  - **省再多成本也不替换**（质量是闸门，不可被成本买通）。\n")

    # 关键字段逐项明细（每模型）
    lines.append("## 关键字段逐项（micro F-beta / TP·FP·FN）\n")
    lines.append("| 模型 | " + " | ".join(CRITICAL_FIELDS) + " |")
    lines.append("|---|" + "---|" * len(CRITICAL_FIELDS))
    for model, agg in aggs.items():
        cells = []
        for f in CRITICAL_FIELDS:
            tp = sum(fs.tp for e in agg.envelopes for fs in e.fields if fs.field == f)
            fp = sum(fs.fp for e in agg.envelopes for fs in e.fields if fs.field == f)
            fn = sum(fs.fn for e in agg.envelopes for fs in e.fields if fs.field == f)
            beta = next((fs.beta for e in agg.envelopes for fs in e.fields if fs.field == f), 1.0)
            micro = _micro_fbeta(tp, fp, fn, beta)
            cells.append(f"{micro:.2f} ({tp:.0f}/{fp:.0f}/{fn:.0f})")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines.append("")

    # doc_type 混淆（仅列错分）
    lines.append("## doc_type 误分类（gold → pred）\n")
    any_err = False
    for model, agg in aggs.items():
        errs = [(e.case_id, e.doc_type_pair) for e in agg.envelopes
                if e.doc_type_pair[0] != e.doc_type_pair[1]]
        if errs:
            any_err = True
            lines.append(f"- `{model}`: " + "；".join(
                f"{cid}: {g}→{p}" for cid, (g, p) in errs))
    if not any_err:
        lines.append("- 无误分类。")
    lines.append("")

    lines.append("---\n> 价格为 2026-05-25 估算，以阿里云百炼控制台为准；延迟受网络/限流抖动影响，"
                 "样本少时仅供量级参考。完整方法学与样本量要求见 `evals/README.md`。")
    return "\n".join(lines)


def _micro_fbeta(tp: float, fp: float, fn: float, beta: float) -> float:
    if tp + fp + fn == 0:
        return 1.0
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    b2 = beta * beta
    d = b2 * p + r
    return (1 + b2) * p * r / d if d else 0.0


def build_report(results_dir: Path, cases_dir: Path, champion: Optional[str]) -> str:
    run_meta, by_model = load_results(results_dir)
    aggs = aggregate(by_model, run_meta, cases_dir)
    champ = champion or run_meta["models"][0]
    if champ not in aggs:
        raise ValueError(f"champion `{champ}` 不在结果中：{list(aggs)}")
    # 让 champion 排在表首
    ordered = {champ: aggs[champ], **{m: a for m, a in aggs.items() if m != champ}}
    return render_markdown(run_meta, ordered, champ)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测报告：gate 决策表")
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--champion", default=None, help="基准模型（默认 run_meta 里第一个）")
    ap.add_argument("--out", type=Path, default=None, help="报告输出路径（默认 results_dir/report.md）")
    args = ap.parse_args(argv)

    md = build_report(args.results_dir, args.cases_dir, args.champion)
    out = args.out or (args.results_dir / "report.md")
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n✅ 报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
