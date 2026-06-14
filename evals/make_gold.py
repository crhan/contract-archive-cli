"""从已入库的真实文档生成 draft gold 评测 case（落**私有**评测集目录，**不脱敏**）。

数据私有化后不再脱敏：评测数据集放私有 git 仓库（git.crhan.com），原始 PDF + 真实金标准
直接入库，代码公开、数据私有。make_gold 把生产链路的产物当 draft gold，省人工初标：
- input.txt：生产 load_document_text 得到的全文（与喂给 extract 的文本一致）。
- source.pdf：archive 里的原始 PDF（若有）→ 评测可走整条链路（OCR→路由→特化→融合）。
- gold.json：champion 抽取结果（draft，需人工对照原文核 parties/amounts/保额等高风险字段后定稿）。

输出到 CONTRACT_ARCHIVE_EVALSET_DIR/extraction/<id>/（默认主仓库内 cases/，建议指向私有数据集）。

用法：
  CONTRACT_ARCHIVE_EVALSET_DIR=~/project/contract-archive-evalset/dataset \
    uv run --no-sync python -m evals.make_gold                       # archive 全部已入库文档
  ... --doc-id e9d1809860f7                                          # 只处理某文档
  ... --crosscheck deepseek-v4-pro                                   # 异家族模型再抽一遍，产 crosscheck.json 破 champion 盲区
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from contract_archive.archive import load_document_text
from contract_archive.config import load_settings
from contract_archive.schemas import DocumentExtraction

from .run import DEFAULT_CASES, evalset_dir

logger = logging.getLogger(__name__)


REVIEW_TEMPLATE = """# DRAFT case 人工核对清单（{doc_id}）

⚠️ 本 case 由 make_gold 从真实文档自动生成（champion 单源抽取），**未经人工核对**。私有数据集，
无需脱敏；但金标准须人工对照原文核高风险字段后才可信。

## gold 正确性（破除 champion 盲区，盲标高风险字段）
- [ ] **不看模型输出**，对照 input.txt / source.pdf 原文，盲标：parties / amounts(数值+is_total_component)
      / 保险高价值字段（保额各类/免赔/赔付比例/被保险人 vs 投保人）/ completeness.issues（缺什么、页码）。
      再与 gold.json diff，以盲标为准修正。
- [ ] 若有 crosscheck.*.json：对比它与 gold，凡 champion 漏抽而 crosscheck 抽到的，回原文核实补上。
- [ ] doc_type / 日期(ISO) / seals / sub_agreements 核一遍。
- [ ] meta.json 记好 doc_type / stratum / difficulty。
"""


def iter_archive_docs(
    archive_dir: Path, only: Optional[str]
) -> list[tuple[str, Path, Path, Path]]:
    """列出 archive 里有 mineru/ + extraction_result.json 的文档
    → (doc_id, doc_dir, mineru_dir, result_json)。"""
    docs_dir = archive_dir / "documents"
    out: list[tuple[str, Path, Path, Path]] = []
    if not docs_dir.is_dir():
        return out
    for d in sorted(p for p in docs_dir.iterdir() if p.is_dir()):
        if only and d.name != only:
            continue
        mineru, result = d / "mineru", d / "extraction_result.json"
        if mineru.is_dir() and result.exists():
            out.append((d.name, d, mineru, result))
    return out


def write_case(
    dataset_dir: Path,
    doc_id: str,
    text: str,
    gold_json: dict,
    source_pdf: Optional[Path],
    crosscheck: Optional[dict],
) -> Path:
    """写一个 draft case 到 dataset_dir/extraction/<doc_id>/（真实数据，不脱敏）。"""
    case_dir = dataset_dir / "extraction" / doc_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "input.txt").write_text(text, encoding="utf-8")
    (case_dir / "gold.json").write_text(
        json.dumps(gold_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (case_dir / "meta.json").write_text(
        json.dumps(
            {
                "doc_type": gold_json.get("doc_type"),
                "stratum": "real(待人工归类)",
                "difficulty": "unknown",
                "provenance": "make_gold champion 抽取 DRAFT；私有数据集不脱敏；需人工核对后定稿",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (case_dir / "REVIEW.md").write_text(REVIEW_TEMPLATE.format(doc_id=doc_id), encoding="utf-8")
    if source_pdf and source_pdf.exists():
        shutil.copy(source_pdf, case_dir / "source.pdf")  # 留原始 PDF → 评测走整条链路
    if crosscheck is not None:
        (case_dir / "crosscheck.json").write_text(
            json.dumps(crosscheck, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return case_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="从真实文档生成 draft gold（不脱敏，落私有评测集目录）"
    )
    ap.add_argument("--archive-dir", type=Path, default=None, help="默认读 config 的 archive 目录")
    ap.add_argument("--doc-id", default=None, help="只处理某个文档（archive documents/ 下的 id）")
    ap.add_argument("--dataset-dir", type=Path, default=None,
                    help="评测集根目录（默认 CONTRACT_ARCHIVE_EVALSET_DIR 或主仓库内 cases/）")
    ap.add_argument("--crosscheck", default=None,
                    help="用一个异家族模型再抽一遍，产 crosscheck.json 供人工对比破 champion 盲区")
    args = ap.parse_args(argv)

    archive_dir = args.archive_dir
    if archive_dir is None:
        settings = load_settings()
        archive_dir = (
            Path(settings.archive_dir)
            if settings.archive_dir
            else Path.home() / ".local/share/contract-archive"
        )
    dataset_dir = args.dataset_dir or evalset_dir()
    # 安全闸：make_gold 写的是**不脱敏**真实数据。绝不能落进主仓库公开的 evals/cases/
    # （会把真实 PII 推上 github）。未显式指向私有数据集就拒绝——必须 --dataset-dir 或
    # 设 CONTRACT_ARCHIVE_EVALSET_DIR 指向私有仓库的 dataset/。
    if dataset_dir.resolve() == DEFAULT_CASES.resolve():
        print(
            "❌ 拒绝把真实（不脱敏）数据写入主仓库公开的 evals/cases/。\n"
            "   请设 CONTRACT_ARCHIVE_EVALSET_DIR 或传 --dataset-dir 指向**私有**评测数据集目录\n"
            "   （如 ~/project/contract-archive-evalset/dataset）。"
        )
        return 2
    docs = iter_archive_docs(archive_dir, args.doc_id)
    if not docs:
        print(f"⚠️  {archive_dir}/documents 下没有可用文档（需含 mineru/ + extraction_result.json）")
        return 1

    print(f"archive: {archive_dir}　待处理 {len(docs)} 个文档　→ 输出到 {dataset_dir}/extraction（私有，不脱敏）\n")
    for doc_id, doc_dir, mineru_dir, result_path in docs:
        raw_text = load_document_text(mineru_dir)
        if not raw_text.strip():
            print(f"  跳过 {doc_id}：mineru 文本为空")
            continue
        gold = json.loads(result_path.read_text(encoding="utf-8"))
        gold["llm_model"] = None  # gold 不带抽取来源
        gold["llm_usage"] = None

        crosscheck = None
        if args.crosscheck:
            from contract_archive.extraction import extract_document

            cc = extract_document(raw_text, model=args.crosscheck)
            crosscheck = cc.model_dump(mode="json")

        # 校验 champion 抽取仍是合法信封（schema 漂移早发现）
        DocumentExtraction.model_validate(gold)
        source_pdf = doc_dir / "source.pdf"
        case_dir = write_case(dataset_dir, doc_id, raw_text, gold, source_pdf, crosscheck)
        print(
            f"  ✓ {doc_id}: doc_type={gold.get('doc_type')}"
            f"{' +pdf' if source_pdf.exists() else ''} → {case_dir}"
        )

    print(
        "\n⚠️  这些是 DRAFT：按各 case 的 REVIEW.md 对照原文盲标高风险字段、核对金标准后再提交私有数据集。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
