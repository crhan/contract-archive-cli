"""
补充协议（sub_agreements）测试：LLM 解析、印章并入子表、落库 round-trip。
不调 LLM——用合成 dict / envelope 直接验证 coerce 与 repository 行为。
"""
from __future__ import annotations

from contract_archive.archive.db import open_archive_db
from contract_archive.archive.repository import (
    _collect_seals,
    get_document,
    insert_document,
)
from contract_archive.extraction.document_extractor import _coerce_sub_agreements
from contract_archive.schemas import DocumentExtraction, Seal, SubAgreement


def test_coerce_sub_agreements_basic():
    raw = [
        {
            "title": "补充协议",
            "summary": "改了车位使用权期限，土地出让金由乙方承担",
            "sign_date": "2026-05-10",
            "seals": [
                {"owner": "示例置业有限公司", "seal_type": "合同专用章", "raw_text": "示例置业合同专用章"}
            ],
            "evidence": "鉴于甲乙双方就...达成如下补充协议",
        }
    ]
    subs = _coerce_sub_agreements(raw)
    assert len(subs) == 1
    assert subs[0].title == "补充协议"
    assert subs[0].sign_date == "2026-05-10"
    assert len(subs[0].seals) == 1
    assert subs[0].seals[0].owner == "示例置业有限公司"


def test_coerce_sub_agreements_skips_garbage():
    # 无 title 跳过；非 dict 跳过；非 list 返回空
    assert _coerce_sub_agreements([{"summary": "x"}, "junk", {"title": "  "}]) == []
    assert _coerce_sub_agreements("notlist") == []
    assert _coerce_sub_agreements(None) == []


def test_collect_seals_merges_sub():
    """主文档 seals + 各补充协议 seals 都进子表，顺序：主在前、补在后。"""
    env = DocumentExtraction(
        seals=[Seal(raw_text="主章", owner="甲公司")],
        sub_agreements=[
            SubAgreement(title="补充协议", seals=[Seal(raw_text="补章", owner="甲公司")]),
        ],
    )
    assert [s.raw_text for s in _collect_seals(env)] == ["主章", "补章"]


def test_collect_seals_no_sub():
    env = DocumentExtraction(seals=[Seal(raw_text="主章")])
    assert len(_collect_seals(env)) == 1


def test_sub_agreements_roundtrip(tmp_path):
    """落库：sub_agreements 进 details_json；补充协议的章并入 document_seals 子表。"""
    conn = open_archive_db(tmp_path / "db.sqlite")
    env = DocumentExtraction(
        doc_type="合同协议",
        title="28号车位转让协议",
        seals=[Seal(raw_text="主章", owner="甲公司", seal_type="合同专用章")],
        sub_agreements=[
            SubAgreement(
                title="补充协议",
                summary="改了期限",
                seals=[Seal(raw_text="补章", owner="甲公司")],
            )
        ],
    )
    doc_id = insert_document(
        conn,
        sha256="a" * 64,
        source_path="/x.pdf",
        output_dir="/o",
        status="ok",
        mineru_duration_s=1.0,
        llm_duration_s=1.0,
        error_message=None,
        extraction=None,
        confidence=None,
        envelope=env,
    )
    row = get_document(conn, doc_id)
    subs = row.details()["sub_agreements"]
    assert len(subs) == 1 and subs[0]["title"] == "补充协议"
    # 主章 + 补章 = 2 条进子表
    n = conn.execute(
        "SELECT COUNT(*) AS c FROM document_seals WHERE doc_id = ?", (doc_id,)
    ).fetchone()["c"]
    assert n == 2
    conn.close()
