"""text_fields（A 路）单测 + fusion.run_vision_fusion 端到端编排测。

text_fields 用 fake openai 验文本看字段；run_vision_fusion patch 两个抽取源 + 用真 fuse_sources
验编排（A/C 并发、attach sidecar、usage 合并、不回写原字段、无图退文本路）。不碰网络。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from contract_archive.extraction import fusion, text_fields, vl_extract
from contract_archive.extraction.text_fields import read_fields_in_text
from contract_archive.extraction.vl_extract import VisionFieldsResult
from contract_archive.schemas import DocumentExtraction, FieldCandidate, LabeledAmount

_FIELDS = {"被保险人": "保障对象本人", "保额_重疾": "重大疾病保险金额"}


# ---------- text_fields fake openai ----------


class _Resp:
    def __init__(self, content, usage=None):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = (
            SimpleNamespace(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
            if usage
            else None
        )


def _install_text_fake(monkeypatch, content, usage=None):
    class _Client:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **k: _Resp(content, usage))
            )

    mod = types.ModuleType("openai")
    mod.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", mod)


def _settings(api_key="test-key"):
    return SimpleNamespace(
        dashscope_model="qwen-max-test",
        dashscope_vl_extract_model="qwen-vl-test",
        dashscope_api_key=api_key,
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(text_fields, "load_settings", lambda: _settings())
    monkeypatch.setattr(fusion, "load_settings", lambda: _settings())
    monkeypatch.setattr(fusion, "encode_image_data_uri", lambda p: "data:image/png;base64,FAKE")
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "1")


def _json(d):
    return json.dumps(d, ensure_ascii=False)


# ---------- text_fields ----------


def test_read_fields_in_text_extracts_source_text(monkeypatch):
    _install_text_fake(monkeypatch, _json({"被保险人": {"value": "陈意"}, "保额_重疾": None}))
    res = read_fields_in_text("保单全文……被保险人：陈意", _FIELDS)
    assert res.by_key["被保险人"][0].value == "陈意"
    assert res.by_key["被保险人"][0].source == "text"
    assert res.by_key["被保险人"][0].page is None  # 文本候选无页号
    assert "保额_重疾" not in res.by_key  # null 丢弃


def test_read_fields_in_text_no_key_degrades(monkeypatch):
    monkeypatch.setattr(text_fields, "load_settings", lambda: _settings(api_key=None))
    res = read_fields_in_text("正文", _FIELDS)
    assert res.by_key == {}
    assert res.error is not None


def test_read_fields_in_text_empty_inputs(monkeypatch):
    assert read_fields_in_text("", _FIELDS).by_key == {}
    assert read_fields_in_text("正文", {}).by_key == {}


# ---------- run_vision_fusion 编排 ----------


def _patch_sources(monkeypatch, text_by_key, vision_by_key, text_usage=None, vision_usage=None):
    """patch A/C 两路抽取源，验编排逻辑（真 fuse_sources）。返回调用记录。"""
    from contract_archive.extraction.text_fields import TextFieldsResult

    seen = {"vision_called": False, "vision_images": None}

    def fake_text(document_text, fields, **kw):
        return TextFieldsResult(by_key=text_by_key, usage=text_usage)

    def fake_vision(image_paths, fields, **kw):
        seen["vision_called"] = True
        seen["vision_images"] = list(image_paths)
        return VisionFieldsResult(by_key=vision_by_key, usage=vision_usage)

    monkeypatch.setattr(text_fields, "read_fields_in_text", fake_text)
    monkeypatch.setattr(vl_extract, "read_fields_on_images", fake_vision)
    return seen


def test_run_vision_fusion_agreement_attaches_sidecar(monkeypatch):
    """A/C 一致 → agreed verdict 挂 sidecar、零评判（不碰 openai）。"""
    _patch_sources(
        monkeypatch,
        text_by_key={"保额_重疾": [FieldCandidate(source="text", value="400万")]},
        vision_by_key={"保额_重疾": [FieldCandidate(source="vision", value="400万", page=2)]},
    )
    env = DocumentExtraction(doc_type="保险凭证")
    ok = fusion.run_vision_fusion(
        env, "全文", {2: Path("p2.png")}, fields=_FIELDS
    )
    assert ok is True
    assert len(env.field_verdicts) == 1
    v = env.field_verdicts[0]
    assert v.key == "保额_重疾"
    assert v.value == "400万"
    assert v.source == "agreed"
    assert env.fusion_overall_confidence == v.confidence


def test_run_vision_fusion_disagreement_adjudicates(monkeypatch):
    """A/C 矛盾 → 真 fuse 走评判（fake openai 回定值）。"""
    _patch_sources(
        monkeypatch,
        text_by_key={"被保险人": [FieldCandidate(source="text", value="张三")]},
        vision_by_key={"被保险人": [FieldCandidate(source="vision", value="陈意", page=1)]},
    )
    _install_text_fake(monkeypatch, _json({"value": "陈意", "confidence": 0.9, "low_confidence": False}))
    env = DocumentExtraction(doc_type="保险凭证")
    ok = fusion.run_vision_fusion(env, "全文", {1: Path("p1.png")}, fields=_FIELDS)
    assert ok is True
    assert env.field_verdicts[0].value == "陈意"
    assert env.field_verdicts[0].source == "adjudicated"


def test_run_vision_fusion_no_fields_returns_false(monkeypatch):
    env = DocumentExtraction(doc_type="保险凭证")
    assert fusion.run_vision_fusion(env, "全文", {1: Path("p1")}, fields={}) is False
    assert env.field_verdicts == []


def test_run_vision_fusion_text_only_when_no_images(monkeypatch):
    """无图 → C 路跳过（不调 read_fields_on_images），仅文本路出单源 verdict。"""
    seen = _patch_sources(
        monkeypatch,
        text_by_key={"保单号": [FieldCandidate(source="text", value="PICC9")]},
        vision_by_key={},
    )
    env = DocumentExtraction(doc_type="保险凭证")
    ok = fusion.run_vision_fusion(env, "全文", {}, fields=_FIELDS)
    assert ok is True
    assert seen["vision_called"] is False  # 无图不调看图路
    assert env.field_verdicts[0].value == "PICC9"
    assert env.field_verdicts[0].source == "text"


def test_run_vision_fusion_merges_all_usage(monkeypatch):
    """文本+看图两路 token 并入 envelope.llm_usage（融合总开销）。一致 → 无评判开销。"""
    _patch_sources(
        monkeypatch,
        text_by_key={"保额_重疾": [FieldCandidate(source="text", value="400万")]},
        vision_by_key={"保额_重疾": [FieldCandidate(source="vision", value="400万", page=1)]},
        text_usage={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        vision_usage={"input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
    )
    env = DocumentExtraction(doc_type="保险凭证", llm_usage={"input_tokens": 20, "output_tokens": 2, "total_tokens": 22})
    fusion.run_vision_fusion(env, "全文", {1: Path("p1")}, fields=_FIELDS)
    assert env.llm_usage == {"input_tokens": 170, "output_tokens": 17, "total_tokens": 187}


def test_run_vision_fusion_does_not_touch_amounts(monkeypatch):
    _patch_sources(
        monkeypatch,
        text_by_key={"保额_重疾": [FieldCandidate(source="text", value="400万")]},
        vision_by_key={"保额_重疾": [FieldCandidate(source="vision", value="400万", page=1)]},
    )
    env = DocumentExtraction(
        doc_type="保险凭证",
        amounts=[LabeledAmount(label="重疾", text="400万元", value=4_000_000.0, is_total_component=True)],
    )
    fusion.run_vision_fusion(env, "全文", {1: Path("p1")}, fields=_FIELDS)
    assert env.amounts[0].value == 4_000_000.0
    assert env.amounts[0].is_total_component is True
