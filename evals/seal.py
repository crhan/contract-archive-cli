"""
VL 签章核查评测线（自包含：gen / run / report / demo）。

被测对象：check_seals_on_images(images, model) —— 看落款页图判甲/乙方有无盖章/签字，
输出签章缺陷 issues。评测对 issues 打分（复用 score.score_completeness_issues），
gate 同样偏召回（漏报签章缺陷=合同蒙混过关，最致命）。

诚实声明（两位评审都点了）：本模块自带的 PIL **合成**落款页与真实 MinerU 抠出的
淡红/模糊红章分布不同——合成图只验"调用链通不通 + 打分逻辑对不对"，**不代表真实
识别精度**。要测真实精度，把脱敏真实落款页放 cases/seal/<id>/private/（已 gitignore），
gold 标注哪方缺签章即可。

用法：
  uv run --no-sync python -m evals.seal gen                     # 生成合成 plumbing 图
  uv run --no-sync python -m evals.seal run --models qwen3-vl-flash,qwen-vl-max
  uv run --no-sync python -m evals.seal report <results_dir>
  uv run --no-sync python -m evals.seal demo                    # 无 key 产 sample_seal_report.md
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

from contract_archive.schemas import CompletenessIssue

from .run import DEFAULT_CASES, DEFAULT_RESULTS
from .score import DELTA, bootstrap_ci, score_completeness_issues

SUITE = "seal"
SAMPLE_REPORT = Path(__file__).resolve().parent / "sample_seal_report.md"

# VL 价格（¥/百万 token，2026-05-25 估算，以官方为准；图像按分辨率折算 token，比文本贵）
VL_PRICE_RMB_PER_M: dict[str, tuple[float, float]] = {
    "qwen-vl-max": (5.6, 22.4),
    "qwen3-vl-plus": (1.5, 4.5),
    "qwen-vl-plus": (1.5, 4.5),
    "qwen3-vl-flash": (0.35, 2.8),
}
DEFAULT_VL_PRICE = (5.6, 22.4)


# ---------------------------------------------------------------------------
# gen：PIL 合成落款页（plumbing 用）
# ---------------------------------------------------------------------------


def _cjk_font(size: int):
    from PIL import ImageFont
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_page(path: Path, party_b_signed: bool) -> None:
    """画一张落款页：甲方恒有红章；乙方按 party_b_signed 决定有无手写签字。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 1100), "white")
    d = ImageDraw.Draw(img)
    f = _cjk_font(28)
    fs = _cjk_font(22)
    d.text((60, 60), "示例置业有限公司 地下车位使用权转让协议", font=f, fill="black")
    d.text((60, 700), "甲方（盖章）：示例置业有限公司", font=fs, fill="black")
    # 甲方红章：红色圆圈 + 中心文字
    d.ellipse((520, 670, 640, 790), outline="red", width=4)
    d.text((536, 715), "示例置业\n合同专用章", font=_cjk_font(16), fill="red")
    d.text((60, 850), "乙方（签字）：", font=fs, fill="black")
    if party_b_signed:
        # 手写签字（潦草线条近似）
        d.line((220, 870, 260, 845), fill="blue", width=3)
        d.line((260, 845, 290, 880), fill="blue", width=3)
        d.line((290, 880, 330, 850), fill="blue", width=3)
        d.text((340, 855), "张三", font=fs, fill="blue")
    d.text((640, 1040), "第 1 页 共 1 页", font=_cjk_font(18), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def gen_synthetic() -> None:
    """生成两个合成 plumbing case：双方齐全 / 乙方落款空白。"""
    base = DEFAULT_CASES / SUITE
    specs = [
        ("s01_both_signed", True, {"issues": []}),
        ("s02_party_b_blank", False, {"issues": [{
            "item": "主协议·乙方签章", "category": "signature",
            "detail": "落款页图像显示该处空白，无红章也无手写签字",
            "evidence": "据落款页图：第 1 页",
        }]}),
    ]
    for cid, signed, gold in specs:
        cdir = base / cid
        _draw_page(cdir / "page_01.png", party_b_signed=signed)
        (cdir / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
        (cdir / "meta.json").write_text(json.dumps({
            "stratum": "合成-plumbing", "difficulty": "n/a",
            "exercises": ["check_seals_on_images 调用链", "签章缺陷打分"],
            "warning": "PIL 合成图，仅验调用链与打分逻辑，不代表真实红章识别精度",
            "provenance": "合成匿名（示例置业/张三），可入库",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已生成合成落款页到 {base}")


# ---------------------------------------------------------------------------
# 加载 + 打分
# ---------------------------------------------------------------------------


def _load_seal_cases(cases_dir: Path) -> list[dict]:
    base = cases_dir / SUITE
    cases = []
    for cdir in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        imgs = sorted(cdir.glob("page_*.png"))
        # 也收 private/ 下的脱敏真实图（gitignore）
        imgs += sorted((cdir / "private").glob("page_*.png")) if (cdir / "private").is_dir() else []
        gold_path = cdir / "gold.json"
        if not imgs or not gold_path.exists():
            continue
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        cases.append({"case_id": cdir.name, "images": imgs, "gold": gold})
    return cases


def _issues(raw: list[dict]) -> list[CompletenessIssue]:
    return [CompletenessIssue.model_validate(i) for i in raw]


def _cost_of(model: str, usage: Optional[dict]) -> Optional[float]:
    if not usage:
        return None
    pin, pout = VL_PRICE_RMB_PER_M.get(model, DEFAULT_VL_PRICE)
    itok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    otok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return itok / 1e6 * pin + otok / 1e6 * pout


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run(models: list[str], cases_dir: Path, out_dir: Path) -> Path:
    import time

    from contract_archive.extraction.vision_seal import check_seals_on_images

    cases = _load_seal_cases(cases_dir)
    if not cases:
        raise FileNotFoundError(f"{cases_dir / SUITE} 下没有 seal case（先跑 `seal gen`）")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps({
        "suite": SUITE, "models": models, "ts": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for model in models:
        mdir = out_dir / model.replace("/", "_")
        mdir.mkdir(exist_ok=True)
        for case in cases:
            start = time.perf_counter()
            issues = check_seals_on_images(case["images"], model=model)
            latency = round(time.perf_counter() - start, 3)
            rec = {
                "case_id": case["case_id"], "model": model,
                "pred_issues": None if issues is None else [i.model_dump() for i in issues],
                "gold": case["gold"], "latency_s": latency,
            }
            (mdir / f"{case['case_id']}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{model}] {case['case_id']}: {latency}s "
                  f"issues={'FAIL' if issues is None else len(issues)}")
    return out_dir


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class SealAgg:
    def __init__(self, model: str):
        self.model = model
        self.recall_vals: list[float] = []   # 逐 case 签章缺陷召回
        self.fbeta_vals: list[float] = []
        self.tp = self.fp = self.fn = 0.0
        self.parse_ok = 0
        self.total = 0
        self.latencies: list[float] = []

    def parse_rate(self) -> float:
        return self.parse_ok / self.total if self.total else 0.0

    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        return self.tp / d if d else None

    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0


def _aggregate(results_dir: Path) -> tuple[dict, dict[str, SealAgg]]:
    run_meta = json.loads((results_dir / "run_meta.json").read_text(encoding="utf-8"))
    aggs: dict[str, SealAgg] = {}
    for model in run_meta["models"]:
        agg = SealAgg(model)
        mdir = results_dir / model.replace("/", "_")
        for p in sorted(mdir.glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            agg.total += 1
            gold = _issues(rec["gold"].get("issues", []))
            if rec["pred_issues"] is None:    # VL 调用/解析失败
                agg.fn += len(gold)
                agg.recall_vals.append(0.0 if gold else 1.0)
                continue
            agg.parse_ok += 1
            pred = _issues(rec["pred_issues"])
            fs = score_completeness_issues(gold, pred)
            agg.tp += fs.tp
            agg.fp += fs.fp
            agg.fn += fs.fn
            agg.recall_vals.append(fs.recall())
            agg.fbeta_vals.append(fs.fbeta())
            agg.latencies.append(rec["latency_s"])
        aggs[model] = agg
    return run_meta, aggs


def report(results_dir: Path, champion: Optional[str]) -> str:
    run_meta, aggs = _aggregate(results_dir)
    champ = champion or run_meta["models"][0]
    champ_agg = aggs[champ]
    lines = ["# VL 签章核查换模型评测报告（gate 决策）\n"]
    n = champ_agg.total
    lines.append(f"- champion: `{champ}`　case 数: {n}　δ={DELTA}")
    lines.append("- ⚠️ 若 case 多为 PIL 合成图：只反映调用链与打分逻辑，**不代表真实红章识别精度**。\n")
    lines.append("| 模型 | 解析率 | 签章召回 | 加权F2(均值/CI下界) | p50延迟 | 裁决 |")
    lines.append("|---|---|---|---|---|---|")
    champ_recall = champ_agg.recall()
    for model, agg in [(champ, champ_agg)] + [(m, a) for m, a in aggs.items() if m != champ]:
        rec = agg.recall()
        rec_s = f"{rec:.0%}" if rec is not None else "—"
        mean, lo, _ = bootstrap_ci(agg.fbeta_vals) if agg.fbeta_vals else (1.0, 1.0, 1.0)
        if model == champ:
            verdict = "🏆 champion"
        else:
            ok = agg.parse_rate() >= 0.98 and (
                champ_recall is None or rec is None or rec >= champ_recall - DELTA)
            verdict = "✅ 可替换" if ok else "❌ 不可替换（签章召回/解析未达标）"
        lines.append(f"| `{model}` | {agg.parse_rate():.0%} | {rec_s} | {mean:.2f}/{lo:.2f} | "
                     f"{agg.p50():.2f}s | {verdict} |")
    lines.append("\n> 价格/精度以官方与真实样本为准；签章是低复杂度判定，性价比可优先 qwen3-vl-flash，"
                 "精度兜底 qwen3-vl-plus。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# demo：无 key 伪造预测
# ---------------------------------------------------------------------------


def demo() -> int:
    cases = _load_seal_cases(DEFAULT_CASES)
    if not cases:
        gen_synthetic()
        cases = _load_seal_cases(DEFAULT_CASES)
    demo_dir = DEFAULT_RESULTS / "seal_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    models = ["qwen-vl-max", "qwen3-vl-flash", "qwen-vl-plus"]
    (demo_dir / "run_meta.json").write_text(json.dumps({
        "suite": SUITE, "models": models, "note": "DEMO 伪造数据"}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    for model in models:
        mdir = demo_dir / model.replace("/", "_")
        mdir.mkdir(exist_ok=True)
        for case in cases:
            gold_issues = case["gold"].get("issues", [])
            if model == "qwen-vl-max":
                pred = copy.deepcopy(gold_issues)           # 满分
            elif model == "qwen3-vl-flash":
                pred = copy.deepcopy(gold_issues)           # 也对（签章判定简单）
            else:  # qwen-vl-plus：在 s02 漏报缺陷（演示一票否决）
                pred = [] if case["case_id"].startswith("s02") else copy.deepcopy(gold_issues)
            (mdir / f"{case['case_id']}.json").write_text(json.dumps({
                "case_id": case["case_id"], "model": model, "pred_issues": pred,
                "gold": case["gold"], "latency_s": {"qwen-vl-max": 3.2, "qwen3-vl-flash": 1.1,
                                                     "qwen-vl-plus": 2.0}[model],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    md = report(demo_dir, champion="qwen-vl-max")
    SAMPLE_REPORT.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n✅ seal demo 报告已写入 {SAMPLE_REPORT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VL 签章核查评测线")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen", help="生成合成落款页")
    rp = sub.add_parser("run", help="跑真实 VL 评测")
    rp.add_argument("--models", required=True)
    rp.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES)
    rp.add_argument("--out", type=Path, default=DEFAULT_RESULTS / datetime.now().strftime("seal_%Y%m%d_%H%M%S"))
    rep = sub.add_parser("report", help="出 gate 报告")
    rep.add_argument("results_dir", type=Path)
    rep.add_argument("--champion", default=None)
    sub.add_parser("demo", help="无 key 产 sample_seal_report.md")
    args = ap.parse_args(argv)

    if args.cmd == "gen":
        gen_synthetic()
    elif args.cmd == "run":
        out = run([m.strip() for m in args.models.split(",") if m.strip()], args.cases_dir, args.out)
        print(f"\n✅ 结果在 {out}，下一步：python -m evals.seal report {out}")
    elif args.cmd == "report":
        print(report(args.results_dir, args.champion))
    elif args.cmd == "demo":
        return demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
