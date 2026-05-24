"""
多 PDF × 多 pipeline 横向对比脚本。

用法：
  python scripts/multi_compare.py output/contract output/29 output/renggou

每个传入目录下都期望有 dashscope/paddleocr/mineru 子目录（缺的会优雅跳过）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def load_dir(d: Path) -> dict:
    """读出单个 pipeline 目录的核心指标，缺失/异常返回 None 字段。"""
    out = {
        "exists": d.exists(),
        "raw_chars": 0,
        "md_chars": 0,
        "md_titles": 0,
        "layout_blocks": 0,
        "layout_with_text": 0,
        "layout_types": {},
        "tables": 0,
        "duration_s": 0.0,
        "model": "",
    }
    if not d.exists():
        return out

    rt = d / "raw_text.txt"
    md = d / "markdown.md"
    layout = d / "layout.json"
    struct = d / "structured.json"
    meta = d / "pipeline_meta.json"

    if rt.exists():
        out["raw_chars"] = len(rt.read_text(encoding="utf-8"))
    if md.exists():
        text = md.read_text(encoding="utf-8")
        out["md_chars"] = len(text)
        out["md_titles"] = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("#"))
    if layout.exists():
        ls = json.loads(layout.read_text(encoding="utf-8"))
        out["layout_blocks"] = len(ls)
        out["layout_with_text"] = sum(1 for b in ls if b.get("text"))
        out["layout_types"] = dict(Counter(b.get("block_type", "?") for b in ls))
    if struct.exists():
        s = json.loads(struct.read_text(encoding="utf-8"))
        out["tables"] = len(s.get("tables", []) or [])
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        out["duration_s"] = float(m.get("duration_seconds", 0))
        out["model"] = m.get("model", "")
    return out


def print_table(rows: list[dict], cols: list[tuple[str, str, int]]):
    """简易表格打印，cols=[(key, header, width)]"""
    header = " | ".join(h.ljust(w) for _, h, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " | ".join(str(r.get(k, "")).ljust(w) for k, _, w in cols)
        print(line)


def main(roots: list[str]):
    pipelines = ("dashscope", "paddleocr", "mineru")
    rows = []
    for root in roots:
        pdf_label = Path(root).name
        for pl in pipelines:
            data = load_dir(Path(root) / pl)
            if not data["exists"]:
                continue
            data["pdf"] = pdf_label
            data["pipeline"] = pl
            data["text_ratio"] = (
                f"{data['layout_with_text']}/{data['layout_blocks']}"
                if data["layout_blocks"]
                else "0/0"
            )
            data["layout_types_str"] = ",".join(
                f"{k}:{v}" for k, v in sorted(data["layout_types"].items(), key=lambda x: -x[1])
            )[:50]
            rows.append(data)

    print("\n## 性能 + 体量")
    print_table(rows, [
        ("pdf", "PDF", 12),
        ("pipeline", "pipeline", 10),
        ("duration_s", "duration_s", 11),
        ("raw_chars", "raw_chars", 10),
        ("md_chars", "md_chars", 10),
        ("md_titles", "titles", 7),
    ])

    print("\n## Layout")
    print_table(rows, [
        ("pdf", "PDF", 12),
        ("pipeline", "pipeline", 10),
        ("layout_blocks", "blocks", 8),
        ("text_ratio", "with_text", 12),
        ("layout_types_str", "type 分布", 50),
    ])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/multi_compare.py <out_dir1> <out_dir2> ...")
        sys.exit(2)
    main(sys.argv[1:])
