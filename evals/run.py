"""
评测执行器：跑 cases × 候选模型，调项目自己的 extract_document() 整条链路，
计时 + 取 token usage，原始产出落 results/<timestamp>/<model>/<case>.json。

用法：
  uv run --no-sync python -m evals.run --models qwen3.7-max,qwen-plus --suite extraction
  uv run --no-sync python -m evals.run --models qwen3.7-max --repeat 3   # 自一致性

需 DASHSCOPE_API_KEY（走 .env / config）。无 key 时各 case 会得到空信封，
report 会把它标成 parse 失败——这本身就是"该候选不可用"的有效信号。

注意：换模型在生产里就是改 dashscope.model；这里用 extract_document(text, model=m)
显式覆盖，跑的就是生产实际链路（prompt+后处理+归一化+completeness 纠正）。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from contract_archive.extraction import extract_document

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVALS_DIR / "cases"
DEFAULT_RESULTS = EVALS_DIR / "results"


def load_cases(suite_dir: Path) -> list[dict[str, Any]]:
    """加载一个 suite 下所有 case：每个目录含 input.txt + gold.json + meta.json。"""
    cases: list[dict[str, Any]] = []
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"suite 目录不存在: {suite_dir}")
    for case_dir in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
        input_txt = case_dir / "input.txt"
        gold_json = case_dir / "gold.json"
        if not (input_txt.exists() and gold_json.exists()):
            continue
        meta = {}
        meta_path = case_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cases.append({
            "case_id": case_dir.name,
            "input": input_txt.read_text(encoding="utf-8"),
            "meta": meta,
        })
    return cases


def run_extraction_case(text: str, model: str) -> dict[str, Any]:
    """跑一次抽取，返回 {pred(信封 json), latency_s, usage}。"""
    start = time.perf_counter()
    envelope = extract_document(text, model=model)
    latency = time.perf_counter() - start
    return {
        "pred": envelope.model_dump(mode="json"),
        "latency_s": round(latency, 3),
        "usage": envelope.llm_usage,
        "llm_model": envelope.llm_model,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测执行器：跑 cases × 候选模型")
    ap.add_argument("--models", required=True, help="逗号分隔的 model id 列表（建议锁 snapshot）")
    ap.add_argument("--suite", default="extraction", choices=["extraction"],
                    help="评测套件（seal 见 Phase 2）")
    ap.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--out", type=Path, default=None, help="结果目录（默认 results/<时间戳>）")
    ap.add_argument("--repeat", type=int, default=1, help=">1 时同输入重复跑，供自一致性检查")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = load_cases(args.cases_dir / args.suite)
    if not cases:
        print(f"⚠️  {args.cases_dir / args.suite} 下没有可用 case")
        return 1

    out_dir = args.out or (DEFAULT_RESULTS / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps({
        "suite": args.suite, "models": models, "repeat": args.repeat,
        "cases_dir": str(args.cases_dir), "ts": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for model in models:
        model_dir = out_dir / model.replace("/", "_")
        model_dir.mkdir(exist_ok=True)
        for case in cases:
            runs = [run_extraction_case(case["input"], model) for _ in range(max(1, args.repeat))]
            record = {
                "case_id": case["case_id"], "model": model, "suite": args.suite,
                "meta": case["meta"], "runs": runs,
            }
            (model_dir / f"{case['case_id']}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{model}] {case['case_id']}: {runs[0]['latency_s']}s "
                  f"model={runs[0]['llm_model']}")

    print(f"\n✅ 结果已写入 {out_dir}\n   下一步：uv run --no-sync python -m evals.report {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
