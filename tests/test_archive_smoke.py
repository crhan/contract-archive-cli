"""
档案库 happy-path smoke test。

不真跑 MinerU（subprocess 需要 GB 级模型加载几分钟），用 stub pipeline
直接写出 mineru/ 目录下的产物。验证：
  - 建表 + migrate
  - ingest 单 PDF → DB 写入 + 文件落盘
  - 重复 ingest → skipped
  - --reingest → replace
  - list / search / show 查询路径
  - extract 复跑只更新抽取层
  - delete 删除 DB 行
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from contract_archive.archive import (
    ArchivePaths,
    SearchFilter,
    collect_stats,
    delete_document,
    discover_pdfs,
    find_by_sha,
    get_document,
    ingest_pdf,
    list_documents,
    open_archive_db,
    re_extract,
    search_documents,
)
from contract_archive.schemas import (
    PipelineMeta,
    PipelineOutput,
    StructuredDocument,
)


# ---------- 假 MinerU pipeline ----------


class StubMineruPipeline:
    """模拟 MinerU 行为：写出 markdown.md + raw_text.txt + pipeline_meta.json。"""

    name = "mineru"

    def __init__(self, markdown_text: str, raw_text: str | None = None):
        self.markdown_text = markdown_text
        self.raw_text = raw_text if raw_text is not None else markdown_text

    def run(self, pdf_path: Path, out_dir: Path) -> PipelineOutput:
        from datetime import datetime

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "markdown.md").write_text(self.markdown_text, encoding="utf-8")
        (out_dir / "raw_text.txt").write_text(self.raw_text, encoding="utf-8")
        (out_dir / "structured.json").write_text(
            StructuredDocument(pages=1).model_dump_json(), encoding="utf-8"
        )
        (out_dir / "layout.json").write_text("[]", encoding="utf-8")
        meta = PipelineMeta(
            pipeline_name="mineru",
            source_pdf=str(pdf_path),
            started_at=datetime.now(),
            finished_at=datetime.now(),
            duration_seconds=0.1,
        )
        (out_dir / "pipeline_meta.json").write_text(
            meta.model_dump_json(), encoding="utf-8"
        )
        return PipelineOutput(meta=meta, raw_text=self.raw_text, markdown=self.markdown_text)


# ---------- fixtures ----------


@pytest.fixture
def archive_root(tmp_path) -> ArchivePaths:
    return ArchivePaths(root=tmp_path / "archive")


@pytest.fixture
def conn(archive_root):
    archive_root.ensure()
    c = open_archive_db(archive_root.db_path)
    yield c
    c.close()


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """造一个假 PDF：流式 hash 只看字节内容，不需要真 PDF 结构。"""
    p = tmp_path / "input" / "demo_contract.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.4 fake demo\n" + b"x" * 1024)
    return p


@pytest.fixture
def sample_markdown() -> str:
    return """# 测试合同

甲方：示例置业有限公司
乙方：张三

合同金额：人民币贰万元整 (¥20000)

签订日期：2025年3月15日
有效期至：2027年3月14日

违约金不超过合同总金额的20%。
"""


def _patch_pipeline(stub: StubMineruPipeline):
    """把 ingest.ingest_pdf 内部的 MinerUPipeline() 替换为 stub。"""
    return patch("contract_archive.archive.ingest.MinerUPipeline", lambda *a, **kw: stub)


def _patch_llm_disabled():
    """让 extract_contract 走 no-llm 路径（不真调 dashscope）。"""
    # 仍然真调 extract_contract，但 llm_enabled=False 时不会走 dashscope


# ---------- 测试 ----------


def test_ingest_happy_path(archive_root, conn, sample_pdf, sample_markdown):
    stub = StubMineruPipeline(markdown_text=sample_markdown)
    with _patch_pipeline(stub):
        result = ingest_pdf(
            sample_pdf, archive_root, conn,
            reingest=False, llm_enabled=False,
        )
    assert result.status == "ok"
    assert result.doc_id == 1
    assert result.sha256
    assert result.mineru_duration_s is not None

    # 文件落盘检查（rename 事务边界后所有 archive 内文件都应到位）
    doc_dir = archive_root.doc_dir(result.sha256)
    assert (doc_dir / "source.pdf").exists()
    assert (doc_dir / "mineru" / "markdown.md").exists()
    assert (doc_dir / "mineru" / "raw_text.txt").exists()
    assert (doc_dir / "extraction_result.json").exists()
    assert (doc_dir / "extraction_confidence.json").exists()
    assert (doc_dir / "ingest.log").exists()

    # 总日志 jsonl
    log_lines = archive_root.ingest_log.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    payload = json.loads(log_lines[0])
    assert payload["status"] == "ok"
    assert payload["sha"] == result.sha256
    assert payload["doc_id"] == result.doc_id

    # DB 记录存在 + 关键状态字段正确（不强断言 rule 抽取的字段值——
    # 那是 extraction 模块的责任，由 extraction 自己的测试覆盖）
    doc = get_document(conn, result.doc_id)
    assert doc.status == "ok"
    assert doc.sha256 == result.sha256
    assert doc.output_dir == str(doc_dir)
    assert doc.error_message is None


def test_ingest_duplicate_skipped(archive_root, conn, sample_pdf, sample_markdown):
    stub = StubMineruPipeline(markdown_text=sample_markdown)
    with _patch_pipeline(stub):
        r1 = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)
        r2 = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)
    assert r1.status == "ok"
    assert r2.status == "skipped"
    assert r2.doc_id == r1.doc_id
    assert r2.skipped_reason


def test_ingest_reingest_replaces(archive_root, conn, sample_pdf, sample_markdown):
    """reingest 应保留 id + sha + ingested_at（archive 流水线契约），不验证
    具体抽取字段值（那由 extraction 模块的测试覆盖）。"""
    stub1 = StubMineruPipeline(markdown_text=sample_markdown)
    with _patch_pipeline(stub1):
        r1 = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)
    orig = get_document(conn, r1.doc_id)

    stub2 = StubMineruPipeline(markdown_text="# 新版合同\n甲方：完全不同的公司")
    with _patch_pipeline(stub2):
        r2 = ingest_pdf(
            sample_pdf, archive_root, conn,
            reingest=True, llm_enabled=False,
        )
    assert r2.status == "ok"
    assert r2.doc_id == r1.doc_id  # id 保留，UPDATE 而非 INSERT
    assert r2.sha256 == r1.sha256  # sha 不变（同 PDF）

    after = get_document(conn, r2.doc_id)
    assert after.ingested_at == orig.ingested_at  # ingested_at 不应被覆盖
    # markdown 内容确实换了
    new_md = (archive_root.doc_dir(r2.sha256) / "mineru" / "markdown.md").read_text(encoding="utf-8")
    assert "新版合同" in new_md


def test_ingest_mineru_failure_writes_failed_status(
    archive_root, conn, sample_pdf, sample_markdown
):
    """MinerU 抛异常时应留 status=failed 记录，不污染 documents/。"""

    class FailingPipeline:
        name = "mineru"
        def run(self, *_a, **_kw):
            raise RuntimeError("simulated mineru crash")

    with patch("contract_archive.archive.ingest.MinerUPipeline", lambda *a, **kw: FailingPipeline()):
        r = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)
    assert r.status == "failed"
    assert "simulated mineru crash" in r.error_message
    # 失败时 tmp 应该被清理，documents/<sha> 不应存在
    assert not archive_root.doc_dir(r.sha256).exists()
    # DB 仍有一条 failed 记录
    doc = get_document(conn, r.doc_id)
    assert doc.status == "failed"


def test_list_and_search(archive_root, conn, sample_pdf, tmp_path):
    """
    两条不同合同入库后 list / search 的管道行为。
    用 update_extraction 直接灌可控的字段值，绕开 rule 抽取的精度问题
    （rule 抽取另有自己的测试覆盖，不在 archive 模块的责任范围）。
    """
    from contract_archive.archive import update_extraction
    from contract_archive.schemas import ContractExtraction, ExtractionConfidence

    md_min = "# placeholder"
    pdf2 = tmp_path / "input" / "other.pdf"
    pdf2.parent.mkdir(parents=True, exist_ok=True)
    pdf2.write_bytes(b"%PDF-1.4 other fake\n" + b"y" * 2048)

    with _patch_pipeline(StubMineruPipeline(markdown_text=md_min)):
        r1 = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)
    with _patch_pipeline(StubMineruPipeline(markdown_text=md_min)):
        r2 = ingest_pdf(pdf2, archive_root, conn, llm_enabled=False)

    # 直接灌可控的字段值
    conf = ExtractionConfidence(); conf.overall = 0.8
    update_extraction(conn, r1.doc_id, status="ok", llm_duration_s=0.1,
        error_message=None,
        extraction=ContractExtraction(
            contract_name="地下车位使用权转让协议",
            party_a="示例置业有限公司", party_b="张三",
            amount="人民币贰万元整", amount_value=20000.0,
            sign_date="2025-03-15", expire_date="2027-03-14",
            risk_clauses=["违约金不超过 20%"]),
        confidence=conf)
    update_extraction(conn, r2.doc_id, status="ok", llm_duration_s=0.1,
        error_message=None,
        extraction=ContractExtraction(
            contract_name="商铺租赁合同", party_a="李四", party_b="王五",
            amount="500000 元", amount_value=500000.0,
            sign_date="2024-08-01", expire_date="2025-07-31"),
        confidence=conf)

    all_docs = list_documents(conn, limit=10)
    assert len(all_docs) == 2

    # LIKE 中文 2 字也命中（trigram 做不到）
    hits = search_documents(conn, SearchFilter(name="商铺"))
    assert len(hits) == 1 and "商铺" in hits[0].contract_name

    hits = search_documents(conn, SearchFilter(party="张三"))
    assert len(hits) == 1 and hits[0].party_b == "张三"

    # 金额 >= 10 万
    hits = search_documents(conn, SearchFilter(amount_min_cents=10000000))
    assert len(hits) == 1 and hits[0].amount_value == 500000.0

    # 签订日 >= 2025-01-01
    hits = search_documents(conn, SearchFilter(signed_after="2025-01-01"))
    assert len(hits) == 1 and hits[0].sign_date == "2025-03-15"

    # 多条件 AND
    hits = search_documents(conn, SearchFilter(
        party="李四", amount_max_cents=100000000, has_risk=False))
    assert len(hits) == 1 and hits[0].id == r2.doc_id

    # has_risk 过滤
    hits = search_documents(conn, SearchFilter(has_risk=True))
    assert len(hits) == 1 and hits[0].id == r1.doc_id


def test_stats(archive_root, conn, sample_pdf):
    """stats 按管道结果统计，不依赖 rule 抽取精度。"""
    from contract_archive.archive import update_extraction
    from contract_archive.schemas import ContractExtraction, ExtractionConfidence

    with _patch_pipeline(StubMineruPipeline(markdown_text="# placeholder")):
        r = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)
    update_extraction(conn, r.doc_id, status="ok", llm_duration_s=0.1,
        error_message=None,
        extraction=ContractExtraction(sign_date="2025-03-15"),
        confidence=ExtractionConfidence())

    s = collect_stats(conn)
    assert s.total == 1
    assert s.by_status == {"ok": 1}
    assert "2025-03" in s.by_sign_month


def test_re_extract_does_not_rerun_mineru(
    archive_root, conn, sample_pdf, sample_markdown
):
    """re_extract 应该只更新抽取字段，mineru_duration_s 保持原值。"""
    with _patch_pipeline(StubMineruPipeline(markdown_text=sample_markdown)):
        r = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)
    orig = get_document(conn, r.doc_id)

    # 即使把 stub 改了，re_extract 不会再调 MinerU
    with patch("contract_archive.archive.ingest.MinerUPipeline", side_effect=AssertionError("should not be called")):
        re_extract(r.doc_id, archive_root, conn, llm_enabled=False)

    after = get_document(conn, r.doc_id)
    assert after.mineru_duration_s == orig.mineru_duration_s
    assert after.contract_name == orig.contract_name


def test_delete_removes_db_row(archive_root, conn, sample_pdf, sample_markdown):
    with _patch_pipeline(StubMineruPipeline(markdown_text=sample_markdown)):
        r = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)
    output_dir = delete_document(conn, r.doc_id)
    assert output_dir
    assert find_by_sha(conn, r.sha256) is None
    # 文件保留（archive 目录还在）
    assert Path(output_dir).exists()


def test_discover_pdfs_recursive(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / "sub" / "c.txt").write_bytes(b"ignore")
    (tmp_path / ".hidden" / "d.pdf").mkdir(parents=True, exist_ok=False)  # 路径中含隐藏目录
    (tmp_path / ".hidden" / "d.pdf" / "skip.pdf").write_bytes(b"%PDF")

    pdfs = discover_pdfs(tmp_path)
    names = sorted(p.name for p in pdfs)
    assert names == ["a.pdf", "b.pdf"]


def test_show_ident_sha_prefix(archive_root, conn, sample_pdf, sample_markdown):
    from contract_archive.archive import find_by_sha_prefix

    with _patch_pipeline(StubMineruPipeline(markdown_text=sample_markdown)):
        r = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)

    matches = find_by_sha_prefix(conn, r.sha256[:8])
    assert len(matches) == 1
    assert matches[0].id == r.doc_id

    with pytest.raises(ValueError, match="prefix must be >= 4"):
        find_by_sha_prefix(conn, "abc")


def test_obligations_storage_and_filter(archive_root, conn, sample_pdf):
    """obligations 写入 + reingest 不堆积 + search/todo 过滤。"""
    from contract_archive.archive import (
        list_obligations, search_documents, update_extraction
    )
    from contract_archive.schemas import (
        ContractExtraction, ExtractionConfidence, ObligationItem
    )

    with _patch_pipeline(StubMineruPipeline(markdown_text="# placeholder")):
        r = ingest_pdf(sample_pdf, archive_root, conn, llm_enabled=False)

    obls = [
        ObligationItem(actor="party_b", action="递交审贷资料",
                       deadline="2026-05-12", evidence="..."),
        ObligationItem(actor="party_b", action="支付定金",
                       deadline=None, evidence="签订当日"),
        ObligationItem(actor="party_a", action="交付车位",
                       deadline="2027-06-30", evidence="..."),
        ObligationItem(actor="both", action="签订商品房买卖合同",
                       deadline=None, evidence="..."),
    ]
    update_extraction(conn, r.doc_id, status="ok", llm_duration_s=0.1,
        error_message=None,
        extraction=ContractExtraction(obligations=obls),
        confidence=ExtractionConfidence())

    # 1) DocumentRow.obligations 完整加载
    doc = get_document(conn, r.doc_id)
    assert len(doc.obligations) == 4
    assert doc.obligations[0].actor == "party_b"
    assert doc.obligations[0].deadline == "2026-05-12"

    # 2) search 跨表 EXISTS 过滤
    assert len(search_documents(conn,
        SearchFilter(deadline_before="2026-06-30"))) == 1
    assert len(search_documents(conn,
        SearchFilter(deadline_before="2026-04-30"))) == 0
    assert len(search_documents(conn,
        SearchFilter(actor="party_a", deadline_after="2027-01-01"))) == 1
    assert len(search_documents(conn,
        SearchFilter(actor="party_a", deadline_before="2026-12-31"))) == 0

    # 3) todo 视图：默认只看带 deadline 的，按 deadline 升序
    todos = list_obligations(conn)
    assert [t.deadline for t in todos] == ["2026-05-12", "2027-06-30"]
    todos = list_obligations(conn, include_undated=True)
    assert len(todos) == 4
    # NULL deadline 排到最后
    assert todos[-1].deadline is None or todos[-2].deadline is None

    # 4) reingest 不堆积
    update_extraction(conn, r.doc_id, status="ok", llm_duration_s=0.1,
        error_message=None,
        extraction=ContractExtraction(obligations=[
            ObligationItem(actor="party_a", action="新动作", deadline="2027-01-01")
        ]),
        confidence=ExtractionConfidence())
    doc = get_document(conn, r.doc_id)
    assert len(doc.obligations) == 1
    assert doc.obligations[0].action == "新动作"


def test_obligations_coerce_chinese_actor(archive_root, conn, sample_pdf):
    """LLM 偶尔返回 actor=甲方/乙方 中文，hybrid 应归一为 party_a/party_b。"""
    from contract_archive.extraction.hybrid import _coerce_obligations

    raw = [
        {"actor": "甲方", "action": "交付", "deadline": "2026-12-31",
         "evidence": "..."},
        {"actor": "乙方", "action": "付款", "deadline": "2025年1月15日",
         "evidence": "..."},
        {"actor": "双方", "action": "签字", "deadline": None, "evidence": ""},
        {"actor": "未知", "action": "应跳过"},        # 非法 actor
        {"actor": "party_a", "action": ""},          # 空 action 跳过
    ]
    out = _coerce_obligations(raw)
    assert [o.actor for o in out] == ["party_a", "party_b", "both"]
    # 日期归一化
    assert out[1].deadline == "2025-01-15"
