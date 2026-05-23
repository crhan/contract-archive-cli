"""
横向对比三路 OCR + extraction 结果。

评估维度：
- OCR 质量：raw_text 字数、字符多样性、可解析性（与 markdown 文本相似度）
- markdown 质量：标题数、列表数、表格 HTML 数
- 表格质量：表格 cell 总数、含 HTML 的表格数
- extraction 质量：抽到字段数、置信度均值
- layout 保真度：layout block 数量、平均 bbox 面积、跨页一致性
- 耗时：duration_seconds

注意：这是"相互对比"——三路彼此参考，不是与 ground truth 对比。
合同 ground truth 抽取另由 extraction_result.json 的 overall confidence 兜底。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .pipelines import get_pipeline
from .schemas import (
    FILE_EXTRACTION,
    FILE_EXTRACTION_CONF,
    FILE_LAYOUT,
    FILE_MARKDOWN,
    FILE_PIPELINE_META,
    FILE_RAW_TEXT,
    FILE_STRUCTURED,
)

logger = logging.getLogger(__name__)

PIPELINES = ("dashscope", "paddleocr", "mineru")


def _load(out_dir: Path) -> dict[str, Any]:
    """从某路输出目录读出全部统计指标。"""
    raw = (out_dir / FILE_RAW_TEXT).read_text(encoding="utf-8") if (out_dir / FILE_RAW_TEXT).exists() else ""
    md = (out_dir / FILE_MARKDOWN).read_text(encoding="utf-8") if (out_dir / FILE_MARKDOWN).exists() else ""
    layout = json.loads((out_dir / FILE_LAYOUT).read_text(encoding="utf-8")) if (out_dir / FILE_LAYOUT).exists() else []
    structured = json.loads((out_dir / FILE_STRUCTURED).read_text(encoding="utf-8")) if (out_dir / FILE_STRUCTURED).exists() else {}
    meta = json.loads((out_dir / FILE_PIPELINE_META).read_text(encoding="utf-8")) if (out_dir / FILE_PIPELINE_META).exists() else {}
    extraction = json.loads((out_dir / FILE_EXTRACTION).read_text(encoding="utf-8")) if (out_dir / FILE_EXTRACTION).exists() else {}
    extraction_conf = json.loads((out_dir / FILE_EXTRACTION_CONF).read_text(encoding="utf-8")) if (out_dir / FILE_EXTRACTION_CONF).exists() else {}

    # markdown 统计
    import re as _re
    md_titles = sum(1 for line in md.splitlines() if line.lstrip().startswith("#"))
    md_lists = sum(1 for line in md.splitlines() if line.lstrip().startswith(("-", "*", "+")))
    # 表格分隔行：|---| / |:--:| / | --- | 等任一格式
    md_table_sep = sum(
        1 for line in md.splitlines() if _re.match(r"^\s*\|?\s*:?-{3,}", line)
    )
    md_tables_html = md.count("<table") + md_table_sep

    # 表格统计
    tables = structured.get("tables", []) or []
    table_cells_total = sum(len(t.get("cells", []) or []) for t in tables)
    tables_with_html = sum(1 for t in tables if t.get("html"))

    # layout 面积
    bbox_areas = []
    for b in layout:
        bb = b.get("bbox") or {}
        try:
            area = (bb["x1"] - bb["x0"]) * (bb["y1"] - bb["y0"])
            if area > 0:
                bbox_areas.append(area)
        except (KeyError, TypeError):
            continue
    avg_area = sum(bbox_areas) / len(bbox_areas) if bbox_areas else 0.0

    return {
        "raw_chars": len(raw),
        "md_chars": len(md),
        "md_titles": md_titles,
        "md_lists": md_lists,
        "md_tables_marker": md_tables_html,
        "layout_blocks": len(layout),
        "layout_avg_area": avg_area,
        "tables": len(tables),
        "table_cells_total": table_cells_total,
        "tables_with_html": tables_with_html,
        "sections": len(structured.get("sections", []) or []),
        "duration_s": float(meta.get("duration_seconds", 0.0)),
        "model": meta.get("model", ""),
        "device": meta.get("device", ""),
        "notes": meta.get("notes", ""),
        "extraction_filled": sum(
            1 for k in ("contract_name", "party_a", "party_b", "amount", "sign_date", "expire_date", "auto_renewal") if extraction.get(k) not in (None, "")
        ),
        "extraction_overall": float(extraction_conf.get("overall", 0.0)),
        "extraction_risks": len(extraction.get("risk_clauses", []) or []),
    }


def generate_report(out_root: Path) -> str:
    """生成 Markdown 对比报告。"""
    rows = {}
    for name in PIPELINES:
        d = out_root / name
        if not d.exists():
            continue
        try:
            rows[name] = _load(d)
        except Exception as e:
            logger.warning("load %s failed: %s", name, e)

    if not rows:
        return "_no pipeline outputs found_\n"

    lines: list[str] = []
    lines.append("# Document Intelligence Playground · 对比报告")
    lines.append(f"_生成时间：{datetime.now().isoformat(timespec='seconds')}_\n")

    lines.append("## 性能 & 体量")
    lines.append("")
    lines.append("| pipeline | duration(s) | model | device | raw_chars | md_chars | sections |")
    lines.append("|---|---:|---|---|---:|---:|---:|")
    for name, r in rows.items():
        lines.append(
            f"| **{name}** | {r['duration_s']:.2f} | {r['model']} | {r['device']} | "
            f"{r['raw_chars']} | {r['md_chars']} | {r['sections']} |"
        )

    lines.append("\n## OCR 文本量")
    lines.append("")
    lines.append("| pipeline | raw_chars | md_chars | md_titles | md_lists |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['raw_chars']} | {r['md_chars']} | {r['md_titles']} | {r['md_lists']} |"
        )

    lines.append("\n## 表格")
    lines.append("")
    lines.append("| pipeline | tables | with_html | cells_total |")
    lines.append("|---|---:|---:|---:|")
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['tables']} | {r['tables_with_html']} | {r['table_cells_total']} |"
        )

    lines.append("\n## Layout 保真度")
    lines.append("")
    lines.append("| pipeline | blocks | avg_bbox_area(pt²) |")
    lines.append("|---|---:|---:|")
    for name, r in rows.items():
        lines.append(f"| {name} | {r['layout_blocks']} | {r['layout_avg_area']:.0f} |")

    lines.append("\n## 合同语义抽取")
    lines.append("")
    lines.append("| pipeline | filled_fields(/7) | overall_conf | risk_clauses |")
    lines.append("|---|---:|---:|---:|")
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['extraction_filled']} | {r['extraction_overall']:.2f} | {r['extraction_risks']} |"
        )

    lines.append("\n## 备注")
    lines.append("")
    for name, r in rows.items():
        lines.append(f"- **{name}**: {r['notes']}")
    lines.append("")
    return "\n".join(lines)


def benchmark_pipelines(
    pdf: Path, out_root: Path, rounds: int = 1
) -> list[dict[str, Any]]:
    """对每路 pipeline 跑 rounds 次取首次（warm-up 已包含在 duration 内）。"""
    out_root = Path(out_root).resolve()
    pdf = Path(pdf).resolve()
    results: list[dict[str, Any]] = []
    for name in PIPELINES:
        for r in range(rounds):
            sub = out_root / name if rounds == 1 else out_root / f"{name}_r{r + 1}"
            entry = {
                "pipeline": name,
                "round": r + 1,
                "duration_s": 0.0,
                "raw_chars": 0,
                "md_chars": 0,
                "layout_blocks": 0,
                "tables": 0,
                "status": "ok",
            }
            try:
                pl = get_pipeline(name)
                out = pl.run(pdf, sub)
                entry["duration_s"] = out.meta.duration_seconds
                entry["raw_chars"] = len(out.raw_text)
                entry["md_chars"] = len(out.markdown)
                entry["layout_blocks"] = len(out.layout)
                entry["tables"] = len(out.structured.tables)
            except Exception as e:
                entry["status"] = f"failed: {e}"
                logger.exception("[%s] benchmark failed", name)
            results.append(entry)
    return results
