"""文档类型路由 doc_type_handlers + ingest._run_extraction 分派单测。

验证：get_handler 映射/回退、合同 handler 声明、_contract_specialized 合并义务/标题、
_run_extraction 据 doc_type 查表分派（合同走特化、非合同走信封启发式）。不碰网络。
"""
from __future__ import annotations

from contract_archive.archive import ingest
from contract_archive.extraction import doc_type_handlers as dth
from contract_archive.extraction.vision_seal import augment_completeness_with_vision
from contract_archive.schemas import ContractExtraction, DocumentExtraction, ExtractionConfidence


def test_get_handler_contract():
    h = dth.get_handler("合同协议")
    assert h.doc_type == "合同协议"
    assert h.specialized_extractor is not None
    assert augment_completeness_with_vision in h.post_processors
    assert h.enable_vision_fusion is False


def test_get_handler_unregistered_falls_back_to_default():
    for t in ("发票票据", "证明", "旅行资料", "其他", "不存在的类型"):
        h = dth.get_handler(t)
        assert h is dth.DEFAULT_HANDLER
        assert h.specialized_extractor is None
        assert h.post_processors == ()
        assert h.enable_vision_fusion is False


def test_contract_specialized_merges_obligations_and_title(monkeypatch):
    from contract_archive.schemas import ObligationItem

    obls = [ObligationItem(actor="party_a", action="付款")]

    def fake_extract_contract(text, llm_enabled=True):
        return ContractExtraction(contract_name=None, obligations=obls), ExtractionConfidence()

    monkeypatch.setattr(dth, "extract_contract", fake_extract_contract)
    env = DocumentExtraction(doc_type="合同协议", title="某买卖合同")
    ext, conf = dth._contract_specialized("正文", env, True)

    # 义务合回信封（合同专属 prompt 对义务更细）
    assert env.obligations == obls
    # 合同抽取没给 contract_name → 回退用信封 title
    assert ext.contract_name == "某买卖合同"


def test_run_extraction_dispatches_contract(monkeypatch):
    """doc_type=合同协议 → 走特化抽取，返回非空 ContractExtraction。"""
    monkeypatch.setattr(
        ingest, "extract_document", lambda text, llm_enabled=True: DocumentExtraction(doc_type="合同协议")
    )
    called = {}

    def fake_specialized(text, env, llm_enabled):
        called["yes"] = True
        return ContractExtraction(contract_name="X"), ExtractionConfidence(overall=0.9)

    # DocTypeHandler 是 frozen，换整条映射而非改属性
    monkeypatch.setitem(
        dth.DOC_TYPE_HANDLERS,
        "合同协议",
        dth.DocTypeHandler("合同协议", specialized_extractor=fake_specialized),
    )

    ext, conf, env = ingest._run_extraction("正文", llm_enabled=True)
    assert called.get("yes") is True
    assert ext.contract_name == "X"
    assert env.doc_type == "合同协议"


def test_run_extraction_non_contract_uses_envelope_heuristic(monkeypatch):
    """非注册类型 → 不跑特化，返回空 ContractExtraction + 信封启发式置信度。"""
    monkeypatch.setattr(
        ingest,
        "extract_document",
        lambda text, llm_enabled=True: DocumentExtraction(doc_type="发票票据", title="增值税发票"),
    )
    ext, conf, env = ingest._run_extraction("正文", llm_enabled=True)
    assert ext.contract_name is None  # 空 ContractExtraction
    assert env.doc_type == "发票票据"
    assert conf.overall >= 0.0  # 走 _envelope_confidence
