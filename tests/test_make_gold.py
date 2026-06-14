"""make_gold（去脱敏后）单测：archive 文档枚举 + draft case 落盘（含 source.pdf 复制）。

脱敏机制已删（数据私有化、放私有仓库 git.crhan.com）。这里验"跑生产链路产物 →
draft gold 落私有数据集（不脱敏）"。
"""
from __future__ import annotations

import json

from evals.make_gold import iter_archive_docs, write_case
from evals.make_gold import main as make_gold_main
from evals.run import DEFAULT_CASES


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


def test_make_gold_refuses_public_cases_dir(tmp_path, monkeypatch):
    """安全闸：未指向私有数据集（回退主仓库公开 cases）时，拒绝写不脱敏真实数据。"""
    monkeypatch.delenv("CONTRACT_ARCHIVE_EVALSET_DIR", raising=False)
    _make_archive_doc(tmp_path, "doc_a")  # 有文档，确保不是因"无文档"才退出
    rc = make_gold_main(["--archive-dir", str(tmp_path)])
    assert rc == 2  # 拒绝
    # 没把真实数据写进公开 cases
    assert not (DEFAULT_CASES / "extraction" / "doc_a").exists()


def test_make_gold_refuses_any_in_repo_path(tmp_path):
    """守卫扩到整个工作树：仓库内任何位置（非 cases）也拒绝，防真实数据落进公开树被提交。"""
    from evals.make_gold import REPO_ROOT

    _make_archive_doc(tmp_path, "doc_a")
    for sub in ("real_dataset", "evals/private_real"):
        rc = make_gold_main(
            ["--archive-dir", str(tmp_path), "--dataset-dir", str(REPO_ROOT / sub)]
        )
        assert rc == 2, f"工作树内 {sub} 应被拒绝"
        assert not (REPO_ROOT / sub).exists()  # 拒绝即不创建


def test_make_gold_allows_explicit_private_dir(tmp_path):
    """显式 --dataset-dir 指向私有目录 → 放行，真实数据落私有目录。"""
    _make_archive_doc(tmp_path, "doc_a")
    # mineru 文本非空（load_document_text 读 mineru 产物）
    (tmp_path / "documents" / "doc_a" / "mineru" / "raw_text.txt").write_text(
        "保单全文 被保险人：陈意", encoding="utf-8"
    )
    private = tmp_path / "private_dataset"
    rc = make_gold_main(["--archive-dir", str(tmp_path), "--dataset-dir", str(private)])
    assert rc == 0
    assert (private / "extraction" / "doc_a" / "gold.json").exists()
