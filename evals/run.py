"""
评测一阶段：跑 cases × 候选模型，调项目自己的 extract_document() 整条链路，
计时 + 取 token usage，把每条 (模型×case×repeat) 结果 **append** 进 results.jsonl。

两阶段设计：
  一阶段(run)：只管跑模型、把结果落 JSONL（增量累积——同一 --out 目录可多次 run 不同模型，
              新模型 append 进去，已有结果不重跑）。
  二阶段(report)：读全量 results.jsonl，按模型分组对比、出 gate 决策报告。

用法：
  uv run --no-sync python -m evals.run --models qwen3.7-max,qwen-plus --out evals/results/r1
  uv run --no-sync python -m evals.run --models deepseek-v4-pro --out evals/results/r1  # 增量追加
  uv run --no-sync python -m evals.run --models qwen3.7-max --repeat 3                  # 自一致性

需 DASHSCOPE_API_KEY（走 .env / config）。换模型在生产里就是改 dashscope.model；这里
用 extract_document(text, model=m) 显式覆盖，跑的就是生产实际链路（prompt+后处理+归一化）。
百炼托管的第三方模型（deepseek-*/glm-* 等）同一 key+endpoint 直接传 model id 即可。
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
RESULTS_JSONL = "results.jsonl"


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
    """跑一次抽取，返回 {pred(信封 json), latency_s, usage, llm_model}。"""
    start = time.perf_counter()
    envelope = extract_document(text, model=model)
    latency = time.perf_counter() - start
    return {
        "pred": envelope.model_dump(mode="json"),
        "latency_s": round(latency, 3),
        "usage": envelope.llm_usage,
        "llm_model": envelope.llm_model,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测一阶段：跑 cases × 候选模型 → results.jsonl")
    ap.add_argument("--models", required=True, help="逗号分隔的 model id 列表（建议锁 snapshot）")
    ap.add_argument("--suite", default="extraction", choices=["extraction"],
                    help="评测套件（seal 见 evals.seal）")
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
    jsonl_path = out_dir / RESULTS_JSONL

    for model in models:
        for case in cases:
            for r in range(max(1, args.repeat)):
                rec = run_extraction_case(case["input"], model)
                append_jsonl(jsonl_path, {
                    "suite": args.suite, "case_id": case["case_id"], "model": model,
                    "repeat_idx": r, "meta": case["meta"], **rec,
                })
                tag = "" if args.repeat == 1 else f" r{r}"
                print(f"  [{model}] {case['case_id']}{tag}: {rec['latency_s']}s "
                      f"model={rec['llm_model']}")

    print(f"\n✅ 已追加到 {jsonl_path}\n"
          f"   下一步（二阶段）：uv run --no-sync python -m evals.report {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
