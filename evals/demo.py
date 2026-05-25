"""
离线 demo：不需要 API key，伪造三个模型在 4 个种子 case 上的预测，端到端跑通
run→report 文件流，产出 evals/sample_report.md，演示 gate 决策表的形态。

伪造的三个模型（仅为展示报表，不是真实跑分）：
- qwen3.7-max（champion）：近满分。
- qwen-plus（候选-好）：仅次要字段小差异，关键字段与签章缺陷全对 → 应判可替换。
- qwen-flash（候选-差）：c01 漏报补充协议乙方签章缺陷（致命），c03 返回空信封（解析失败）
  → 应被门禁一票否决。

重跑：uv run --no-sync python -m evals.demo
真实评测请用 evals.run（真调模型），本文件只造演示数据。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from contract_archive.schemas import DocumentExtraction

from .report import build_report
from .run import DEFAULT_CASES, DEFAULT_RESULTS

DEMO_DIR = DEFAULT_RESULTS / "demo"
SAMPLE_REPORT = Path(__file__).resolve().parent / "sample_report.md"
SUITE = "extraction"


def _load_gold(case_id: str) -> DocumentExtraction:
    return DocumentExtraction.model_validate(
        json.loads((DEFAULT_CASES / SUITE / case_id / "gold.json").read_text(encoding="utf-8"))
    )


def _fake_usage(in_tok: int, out_tok: int) -> dict:
    return {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok}


def _champion_pred(gold: DocumentExtraction) -> DocumentExtraction:
    pred = copy.deepcopy(gold)
    pred.llm_model = "qwen3.7-max"
    return pred


def _good_candidate_pred(case_id: str, gold: DocumentExtraction) -> DocumentExtraction:
    """qwen-plus：关键字段全对，仅在次要字段制造无伤大雅的小差异。"""
    pred = copy.deepcopy(gold)
    pred.llm_model = "qwen-plus"
    if case_id == "c02_income_certificate" and pred.fields:
        pred.fields = pred.fields[:-1]            # 漏一个次要键值字段（联系人）
    if case_id == "c04_lease_complete":
        pred.summary = (pred.summary or "") + "（措辞略有不同）"  # 摘要措辞不同，主观项
    return pred


def _bad_candidate_pred(case_id: str, gold: DocumentExtraction) -> DocumentExtraction:
    """qwen-flash：致命退化。"""
    if case_id == "c03_vat_invoice":
        return DocumentExtraction()               # 返回空信封：JSON 解析失败
    pred = copy.deepcopy(gold)
    pred.llm_model = "qwen-flash"
    if case_id == "c01_carpark_with_subagreement" and pred.completeness:
        pred.completeness.issues = [i for i in pred.completeness.issues if i.category != "signature"]
        pred.completeness.status = "complete"     # 漏报补充协议乙方签章缺陷（致命）
    return pred


def generate() -> Path:
    """伪造三模型预测，写成与 evals.run 同格式的 results.jsonl（重跑前先清空，幂等）。"""
    cases = sorted(p.name for p in (DEFAULT_CASES / SUITE).iterdir() if p.is_dir())
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = DEMO_DIR / "results.jsonl"
    jsonl.unlink(missing_ok=True)
    models = ["qwen3.7-max", "qwen-plus", "qwen-flash"]
    builders = {
        "qwen3.7-max": lambda cid, g: _champion_pred(g),
        "qwen-plus": _good_candidate_pred,
        "qwen-flash": _bad_candidate_pred,
    }
    usages = {"qwen3.7-max": (1800, 650), "qwen-plus": (1800, 640), "qwen-flash": (1800, 600)}
    latencies = {"qwen3.7-max": 2.6, "qwen-plus": 1.5, "qwen-flash": 0.7}

    with jsonl.open("w", encoding="utf-8") as f:
        for model in models:
            for cid in cases:
                gold = _load_gold(cid)
                pred = builders[model](cid, gold)
                usage = _fake_usage(*usages[model]) if pred.llm_model else None
                rec = {
                    "suite": SUITE, "case_id": cid, "model": model, "repeat_idx": 0,
                    "meta": json.loads((DEFAULT_CASES / SUITE / cid / "meta.json").read_text(encoding="utf-8")),
                    "pred": pred.model_dump(mode="json"),
                    "latency_s": latencies[model], "usage": usage, "llm_model": pred.llm_model,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return DEMO_DIR


def main() -> int:
    results_dir = generate()
    md = build_report(results_dir, DEFAULT_CASES, champion="qwen3.7-max")
    SAMPLE_REPORT.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n✅ demo 报告已写入 {SAMPLE_REPORT}（results 在 {results_dir}，已 gitignore）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
