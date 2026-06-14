"""多源融合 sidecar（FieldVerdict/FieldCandidate）+ encode_image_data_uri 单测。

只验 schema 结构/默认/序列化与编码工具，不碰网络/LLM。融合逻辑本身在 fusion 模块另测。
"""
from __future__ import annotations

import base64

from contract_archive.schemas import (
    DocumentExtraction,
    FieldCandidate,
    FieldVerdict,
    LabeledAmount,
)
from contract_archive.utils import encode_image_data_uri


def test_field_verdict_defaults():
    v = FieldVerdict(key="保额_一般医疗")
    assert v.value is None
    assert v.source == "adjudicated"
    assert v.confidence == 0.0
    assert v.low_confidence is False
    assert v.candidates == []


def test_field_candidate_carries_source_and_page():
    c = FieldCandidate(source="vision", value="200万", evidence="保障责任表", page=3)
    assert c.source == "vision"
    assert c.page == 3


def test_document_extraction_sidecar_defaults_empty():
    """未融合的文档：field_verdicts 空、fusion_overall_confidence None（零侵入）。"""
    doc = DocumentExtraction(doc_type="保险凭证")
    assert doc.field_verdicts == []
    assert doc.fusion_overall_confidence is None


def test_sidecar_survives_json_roundtrip():
    """sidecar 随 model_dump_json 进 details_json，再读回不丢——零 DB 迁移的前提。"""
    doc = DocumentExtraction(
        doc_type="保险凭证",
        field_verdicts=[
            FieldVerdict(
                key="保额_重疾",
                value="400万",
                source="adjudicated",
                confidence=0.92,
                candidates=[
                    FieldCandidate(source="text", value="400万"),
                    FieldCandidate(source="vision", value="400万", page=2),
                ],
            )
        ],
        fusion_overall_confidence=0.88,
    )
    restored = DocumentExtraction.model_validate_json(doc.model_dump_json())
    assert restored.fusion_overall_confidence == 0.88
    assert len(restored.field_verdicts) == 1
    v = restored.field_verdicts[0]
    assert v.key == "保额_重疾"
    assert v.value == "400万"
    assert [c.source for c in v.candidates] == ["text", "vision"]


def test_verdict_does_not_touch_original_amount_fields():
    """sidecar 与原 amounts 互不影响：融合结论挂 verdict，原金额项的不变量原样保留。"""
    amt = LabeledAmount(label="重大疾病保险金", text="400万元", value=4_000_000.0, is_total_component=True)
    doc = DocumentExtraction(
        doc_type="保险凭证",
        amounts=[amt],
        field_verdicts=[FieldVerdict(key="保额_重疾", value="500万", source="vision")],
    )
    # 原金额项不被 verdict 的不同取值污染
    assert doc.amounts[0].value == 4_000_000.0
    assert doc.amounts[0].is_total_component is True


def test_encode_image_data_uri(tmp_path):
    img = tmp_path / "p.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nFAKEBYTES")
    uri = encode_image_data_uri(img)
    assert uri.startswith("data:image/png;base64,")
    payload = uri.split(",", 1)[1]
    assert base64.b64decode(payload) == b"\x89PNG\r\n\x1a\nFAKEBYTES"
