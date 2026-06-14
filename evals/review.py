"""
盲标助手：破除"审核 champion 输出"的偏向——只给你看 input.txt + 空白模板，
你从原文从头标高风险字段，填完再和 gold.json diff。想偷看 gold 都看不到。

用法：
  python -m evals.review <case_id>           # 打印原文 + 生成空白 blind.json（不覆盖已填的）
  python -m evals.review <case_id> --diff     # 填完 blind.json 后，与 gold.json 列差异

只针对高风险三件套：parties / amounts(数值+是否计入合计/是否分期) / completeness.issues。
case 先在私有数据集（CONTRACT_ARCHIVE_EVALSET_DIR）下找，找不到再去主仓库内合成 cases/。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .run import DEFAULT_CASES, evalset_dir
from .score import normalize_str

EVALS_DIR = Path(__file__).resolve().parent
BLIND_TEMPLATE = {
    "_说明": "只看 input.txt 填下面三类，别开 gold.json！填完跑：python -m evals.review <case_id> --diff",
    "_format": {
        "amounts 每项": {"label": "如 转让价款/年收入", "value": 0.0,
                         "is_total_component": "true=计入文档主合计", "is_installment": "true=分期/部分付款项"},
        "issue 每项": {"item": "缺陷/异常名", "category": "signature|field|amount|identity",
                      "page": "页码数字", "detail": "缺/异常了什么"},
    },
    "parties": [],
    "amounts": [],
    "completeness_issues": [],
}


def find_case(case_id: str) -> Optional[Path]:
    # 私有数据集优先（evalset_dir 已含"未设 env 则回退主仓库 cases"），再兜底主仓库合成 cases。
    for base in (evalset_dir() / "extraction", DEFAULT_CASES / "extraction"):
        d = base / case_id
        if (d / "input.txt").exists():
            return d
    return None


def cmd_start(case_dir: Path) -> None:
    print("=" * 70)
    print(f"原文 input.txt（{case_dir.name}）—— 只据此标注，别开 gold.json")
    print("=" * 70)
    print((case_dir / "input.txt").read_text(encoding="utf-8"))
    blind = case_dir / "blind.json"
    if blind.exists():
        print(f"\n⚠️ {blind} 已存在（不覆盖你已填的）。编辑它，填完跑 --diff。")
    else:
        blind.write_text(json.dumps(BLIND_TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ 已生成空白模板：{blind}")
    print(f"   填完执行：uv run --no-sync python -m evals.review {case_dir.name} --diff")


def _amount_key(a: dict) -> Optional[float]:
    v = a.get("value")
    return round(float(v), 2) if isinstance(v, (int, float)) else None


def cmd_diff(case_dir: Path) -> int:
    blind_path, gold_path = case_dir / "blind.json", case_dir / "gold.json"
    if not blind_path.exists():
        print(f"❌ 没有 {blind_path}，先跑不带 --diff 的命令生成模板并填写。")
        return 1
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    print(f"=== 盲标 vs gold 差异（{case_dir.name}）===")
    print("（⚠️ 你标了 gold 没有 = champion 可能漏抽，gold 该补；gold 有你没标 = 你漏了或 champion 多报）\n")

    # parties
    bset = {normalize_str(p) for p in blind.get("parties", []) if p}
    graw = gold.get("parties", [])
    gset = {normalize_str(p): p for p in graw}
    print("【parties】")
    only_b = [p for p in blind.get("parties", []) if normalize_str(p) not in gset]
    only_g = [graw[i] for i, p in enumerate(graw) if normalize_str(p) not in bset]
    print(f"  你标了 gold 没有: {only_b or '（无）'}")
    print(f"  gold 有你没标:   {only_g or '（无）'}")

    # amounts（按数值）
    print("\n【amounts（按数值比）】")
    b_amt = {_amount_key(a): a for a in blind.get("amounts", []) if _amount_key(a) is not None}
    g_amt = {_amount_key(a): a for a in gold.get("amounts", []) if _amount_key(a) is not None}
    print(f"  你标了 gold 没有的金额: {sorted(set(b_amt) - set(g_amt)) or '（无）'}")
    print(f"  gold 有你没标的金额:   {sorted(set(g_amt) - set(b_amt)) or '（无）'}")
    for v in sorted(set(b_amt) & set(g_amt)):
        ba, ga = b_amt[v], g_amt[v]
        for fld in ("is_total_component", "is_installment"):
            if bool(ba.get(fld)) != bool(ga.get(fld)):
                print(f"  ⚠️ {v}: {fld} 你={bool(ba.get(fld))} gold={bool(ga.get(fld))}")

    # completeness issues
    print("\n【completeness.issues】")
    b_iss = blind.get("completeness_issues", [])
    g_iss = (gold.get("completeness") or {}).get("issues", [])
    print(f"  你标的 {len(b_iss)} 条 / gold 的 {len(g_iss)} 条：")
    for i in b_iss:
        print(f"    你: [{i.get('category')}] {i.get('item')} (第{i.get('page')}页)")
    for i in g_iss:
        ev = i.get("evidence", "")
        print(f"    gold: [{i.get('category')}] {i.get('item')}  出处:{ev[:40]}")
    print("\n  → 逐条人工对：你标了 gold 漏的就给 gold 补上；gold 多报的核实是否误报。")
    print("\n核完后：把 input.txt + 修正 gold.json + meta.json 复制到 evals/cases/extraction/<有意义id>/ 提交。")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="盲标助手：只看原文标注，再与 gold diff")
    ap.add_argument("case_id")
    ap.add_argument("--diff", action="store_true", help="与 gold.json 列差异（先填好 blind.json）")
    args = ap.parse_args(argv)

    case_dir = find_case(args.case_id)
    if case_dir is None:
        print(f"❌ 找不到 case「{args.case_id}」（在 cases_private/extraction/ 或 cases/extraction/ 下）")
        return 1
    return cmd_diff(case_dir) if args.diff else (cmd_start(case_dir) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
