"""make_gold（去脱敏后）单测：archive 文档枚举 + draft case 落盘（含 source.pdf 复制）。

脱敏机制已删（数据私有化、放私有仓库 git.crhan.com）。这里验"跑生产链路产物 →
draft gold 落私有数据集（不脱敏）"。
"""
from __future__ import annotations

import json

from evals.make_gold import iter_archive_docs, write_case


def _make_archive_doc(archive_dir, doc_id, with_pdf=True):
    doc = archive_dir / "documents" / doc_id
    (doc / "mineru").mkdir(parents=True)
    (doc / "extraction_result.json").write_text(
        json.dumps({"doc_type": "保险凭证", "title": "某保单"}, ensure_ascii=False),
        encoding="utf-8",
    )
    if with_pdf:
        (doc / "source.pdf").write_bytes(b"%PDF-1.4 fake")
    return doc


def test_iter_archive_docs_lists_valid(tmp_path):
    _make_archive_doc(tmp_path, "doc_a")
    _make_archive_doc(tmp_path, "doc_b")
    # 缺 extraction_result.json 的不算
    (tmp_path / "documents" / "doc_c" / "mineru").mkdir(parents=True)
    docs = iter_archive_docs(tmp_path, only=None)
    assert [d[0] for d in docs] == ["doc_a", "doc_b"]
    assert all(len(d) == 4 for d in docs)  # (doc_id, doc_dir, mineru_dir, result_path)


def test_iter_archive_docs_only_filter(tmp_path):
    _make_archive_doc(tmp_path, "doc_a")
    _make_archive_doc(tmp_path, "doc_b")
    assert [d[0] for d in iter_archive_docs(tmp_path, only="doc_b")] == ["doc_b"]


def test_iter_archive_docs_empty_when_no_documents(tmp_path):
    assert iter_archive_docs(tmp_path, only=None) == []


def test_write_case_emits_real_data_with_pdf(tmp_path):
    doc = _make_archive_doc(tmp_path, "doc_a")
    dataset = tmp_path / "dataset"
    gold = {"doc_type": "保险凭证", "title": "某保单", "parties": ["陈意"]}
    case_dir = write_case(
        dataset, "doc_a", "保单全文……被保险人：陈意", gold, doc / "source.pdf", None
    )
    assert case_dir == dataset / "extraction" / "doc_a"
    # 真实数据原样落盘，不脱敏
    assert "陈意" in (case_dir / "input.txt").read_text(encoding="utf-8")
    assert json.loads((case_dir / "gold.json").read_text(encoding="utf-8"))["parties"] == ["陈意"]
    # 原始 PDF 复制进来 → 评测走整条链路
    assert (case_dir / "source.pdf").exists()
    assert (case_dir / "meta.json").exists()
    assert (case_dir / "REVIEW.md").exists()


def test_write_case_without_pdf_skips_copy(tmp_path):
    dataset = tmp_path / "dataset"
    case_dir = write_case(dataset, "doc_x", "文本", {"doc_type": "证明"}, None, None)
    assert not (case_dir / "source.pdf").exists()


def test_write_case_includes_crosscheck(tmp_path):
    dataset = tmp_path / "dataset"
    case_dir = write_case(
        dataset, "doc_y", "文本", {"doc_type": "证明"}, None, {"doc_type": "证明", "by": "异家族"}
    )
    assert json.loads((case_dir / "crosscheck.json").read_text(encoding="utf-8"))["by"] == "异家族"
