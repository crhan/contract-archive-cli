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
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from contract_archive.extraction import extract_document

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVALS_DIR / "cases"
DEFAULT_RESULTS = EVALS_DIR / "results"
RESULTS_JSONL = "results.jsonl"


def evalset_dir() -> Path:
    """评测集根目录：CONTRACT_ARCHIVE_EVALSET_DIR（私有数据集，git.crhan.com）优先，
    否则回退主仓库内合成 cases（保 CI smoke 不依赖私有数据）。结构均为 <root>/<suite>/<case>/。"""
    raw = os.getenv("CONTRACT_ARCHIVE_EVALSET_DIR")
    return Path(raw).expanduser() if raw and raw.strip() else DEFAULT_CASES


def load_cases(suite_dir: Path) -> list[dict[str, Any]]:
    """加载一个 suite 下所有 case：每目录含 gold.json + (input.txt 或 source.pdf) + meta.json。

    source.pdf（原始 PDF）→ 走整条生产链路评测（OCR→类型路由→特化→多源融合）；
    input.txt（纯文本）→ 走文本抽取（旧路径，供模型对比）。两者皆可，优先 PDF。
    """
    cases: list[dict[str, Any]] = []
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"suite 目录不存在: {suite_dir}")
    for case_dir in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
        gold_json = case_dir / "gold.json"
        input_txt = case_dir / "input.txt"
        source_pdf = case_dir / "source.pdf"
        if not gold_json.exists() or not (input_txt.exists() or source_pdf.exists()):
            continue
        meta = {}
        meta_path = case_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cases.append({
            "case_id": case_dir.name,
            "input": input_txt.read_text(encoding="utf-8") if input_txt.exists() else None,
            "pdf": source_pdf if source_pdf.exists() else None,
            "meta": meta,
        })
    return cases


def run_extraction_case(text: str, model: str) -> dict[str, Any]:
    """跑一次文本抽取（供模型对比），返回 {pred(信封 json), latency_s, usage, llm_model}。"""
    start = time.perf_counter()
    envelope = extract_document(text, model=model)
    latency = time.perf_counter() - start
    return {
        "pred": envelope.model_dump(mode="json"),
        "latency_s": round(latency, 3),
        "usage": envelope.llm_usage,
        "llm_model": envelope.llm_model,
    }


def run_pdf_case(pdf_path: Path) -> dict[str, Any]:
    """跑原始 PDF 的整条生产链路（OCR → 类型路由 → 特化 → 多源融合），对照 gold 测全链路。

    用生产默认模型（CONTRACT_ARCHIVE 配置），不做 per-model 覆盖——PDF case 是"改 prompt/流水线
    后的全链路回归门禁"，模型对比仍走文本 case。融合需 source.pdf 选页，故复制进临时目录。
    """
    import shutil
    import tempfile

    from contract_archive.archive import load_document_text
    from contract_archive.archive.ingest import run_full_extraction
    from contract_archive.pipelines import MinerUPipeline

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        shutil.copy(pdf_path, work / "source.pdf")  # _select_fusion_images 读 mineru_dir.parent/source.pdf
        mineru_dir = work / "mineru"
        MinerUPipeline().run(pdf_path, mineru_dir)
        text = load_document_text(mineru_dir)
        envelope = run_full_extraction(text, mineru_dir)
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
    ap.add_argument("--cases-dir", type=Path, default=None,
                    help="评测集根目录（默认 CONTRACT_ARCHIVE_EVALSET_DIR 或主仓库内合成 cases）")
    ap.add_argument("--out", type=Path, default=None, help="结果目录（默认 results/<时间戳>）")
    ap.add_argument("--repeat", type=int, default=1, help=">1 时同输入重复跑，供自一致性检查")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases_root = args.cases_dir or evalset_dir()
    cases = load_cases(cases_root / args.suite)
    if not cases:
        print(f"⚠️  {cases_root / args.suite} 下没有可用 case")
        return 1

    out_dir = args.out or (DEFAULT_RESULTS / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / RESULTS_JSONL

    for case in cases:
        if case["pdf"] is not None:
            # 原始 PDF：整条生产链路全链路回归，跑一次（生产默认模型，不做 per-model 覆盖）。
            rec = run_pdf_case(case["pdf"])
            append_jsonl(jsonl_path, {
                "suite": args.suite, "case_id": case["case_id"], "model": rec["llm_model"],
                "repeat_idx": 0, "meta": case["meta"], "source": "pdf", **rec,
            })
            print(f"  [pdf] {case['case_id']}: {rec['latency_s']}s model={rec['llm_model']}")
            continue
        # 纯文本：模型对比（每个候选模型各跑一遍）。
        for model in models:
            for r in range(max(1, args.repeat)):
                rec = run_extraction_case(case["input"], model)
                append_jsonl(jsonl_path, {
                    "suite": args.suite, "case_id": case["case_id"], "model": model,
                    "repeat_idx": r, "meta": case["meta"], "source": "text", **rec,
                })
                tag = "" if args.repeat == 1 else f" r{r}"
                print(f"  [{model}] {case['case_id']}{tag}: {rec['latency_s']}s "
                      f"model={rec['llm_model']}")

    print(f"\n✅ 已追加到 {jsonl_path}\n"
          f"   下一步（二阶段）：uv run --no-sync python -m evals.report {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
