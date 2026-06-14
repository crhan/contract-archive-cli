"""多源融合 fuse_sources / adjudicate / attach_verdicts + agent_fallback 单测。

覆盖需求三缺陷的 mock 回归：
① 保额 200/400/800 万独立键互不覆盖 + 各源一致 → 0 次评判（省钱）。
② 被保人=陈意：源矛盾 → 据图评判选 "陈意"、source=adjudicated、候选留证；C 也错 → low_confidence。
③ 值结构不同（A 路 100 万 vs C 路 200 万）→ 触发评判纠正。
另验：无 key 降级、无图降级、attach 不回写原字段、usage 合并、agent_fallback no-op。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from contract_archive.extraction import agent_fallback, fusion
from contract_archive.schemas import DocumentExtraction, FieldCandidate, LabeledAmount


# ---------- fake openai ----------


class _FakeResp:
    def __init__(self, content: str, usage: dict | None = None) -> None:
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


class _FakeClient:
    def __init__(self, behaviors: list, **init_kwargs) -> None:
        self._behaviors = list(behaviors)
        self.init_kwargs = init_kwargs
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        b = self._behaviors[self.calls]
        self.calls += 1
        if isinstance(b, Exception):
            raise b
        content, usage = b if isinstance(b, tuple) else (b, None)
        return _FakeResp(content, usage)


def _install_fake_openai(monkeypatch, behaviors: list) -> dict:
    holder: dict = {"created": 0}

    def factory(**kwargs):
        holder["created"] += 1
        client = _FakeClient(behaviors, **kwargs)
        holder["client"] = client
        return client

    fake_mod = types.ModuleType("openai")
    fake_mod.OpenAI = factory
    monkeypatch.setitem(sys.modules, "openai", fake_mod)
    return holder


def _settings(api_key: str | None = "test-key"):
    return SimpleNamespace(
        dashscope_vl_extract_model="qwen3.6-flash-test",
        dashscope_api_key=api_key,
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(fusion, "load_settings", lambda: _settings())
    monkeypatch.setattr(fusion, "encode_image_data_uri", lambda p: "data:image/png;base64,FAKE")
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "1")


def _t(value, page=None):
    return FieldCandidate(source="text", value=value, page=page)


def _v(value, page=None, evidence=""):
    return FieldCandidate(source="vision", value=value, page=page, evidence=evidence)


def _json(d):
    return json.dumps(d, ensure_ascii=False)


# ---------- ① 独立键 + 一致省钱 ----------


def test_independent_keys_agree_zero_adjudication(monkeypatch):
    """保额三类独立键、各源一致 → 3 个 agreed verdict、0 次 LLM 评判（client 都不构造）。"""
    holder = _install_fake_openai(monkeypatch, [RuntimeError("不应调用评判")])
    text = {
        "保额_一般医疗": [_t("200万")],
        "保额_特定医疗": [_t("200万")],
        "保额_重疾": [_t("400万")],
        "保额_总额": [_t("800万")],
    }
    vision = {
        "保额_一般医疗": [_v("200万元", page=2)],
        "保额_特定医疗": [_v("200万", page=2)],
        "保额_重疾": [_v("400万", page=2)],
        "保额_总额": [_v("800万", page=3)],
    }
    res = fusion.fuse_sources(text, vision, images_by_page={2: Path("p2"), 3: Path("p3")})

    vals = {v.key: v.value for v in res.verdicts}
    # 三类保额互不覆盖，各自正确
    assert vals["保额_一般医疗"] == "200万"
    assert vals["保额_特定医疗"] == "200万"
    assert vals["保额_重疾"] == "400万"
    assert vals["保额_总额"] == "800万"
    assert all(v.source == "agreed" for v in res.verdicts)
    assert all(not v.low_confidence for v in res.verdicts)
    assert res.usage is None  # 没有评判开销
    assert holder["created"] == 0  # client 从未构造 → 0 次评判，省钱


def test_single_source_agreed_lower_confidence(monkeypatch):
    _install_fake_openai(monkeypatch, [])
    res = fusion.fuse_sources({"保单号": [_t("PICC123")]}, {})
    v = res.verdicts[0]
    assert v.value == "PICC123"
    assert v.source == "text"  # 仅文本一源
    assert v.confidence == fusion._CONF_SINGLE_SOURCE
    assert v.low_confidence is False


# ---------- ② 被保人=陈意：矛盾据图评判 ----------


def test_insured_name_adjudicated_from_image(monkeypatch):
    """文本误把投保人当被保险人(张三)，看图给陈意 → 据图评判选陈意、source=adjudicated、候选留证。"""
    _install_fake_openai(
        monkeypatch,
        [_json({"value": "陈意", "confidence": 0.9, "low_confidence": False, "rationale": "图示被保险人：陈意"})],
    )
    text = {"被保险人": [_t("张三")]}  # 错（其实是投保人）
    vision = {"被保险人": [_v("陈意", page=1, evidence="被保险人：陈意")]}
    res = fusion.fuse_sources(
        text, vision, images_by_page={1: Path("p1")},
        field_defs={"被保险人": "保障对象本人，区别于投保人"},
    )
    v = res.verdicts[0]
    assert v.value == "陈意"
    assert v.source == "adjudicated"
    assert v.confidence == 0.9
    assert v.low_confidence is False
    # 候选留证：文本(张三)+看图(陈意)都在审计里
    assert {c.value for c in v.candidates} == {"张三", "陈意"}


def test_both_sources_wrong_yields_low_confidence(monkeypatch):
    """C 也错：源矛盾且评判置信低 → low_confidence（供 agent 兜底关注）。"""
    _install_fake_openai(
        monkeypatch,
        [_json({"value": "存疑", "confidence": 0.3, "low_confidence": True, "rationale": "图上字迹模糊"})],
    )
    res = fusion.fuse_sources(
        {"被保险人": [_t("张三")]}, {"被保险人": [_v("李四", page=1)]},
        images_by_page={1: Path("p1")},
    )
    v = res.verdicts[0]
    assert v.low_confidence is True
    assert v.confidence == 0.3
    assert res.overall_confidence == 0.3


def test_threshold_forces_low_confidence_even_if_model_says_ok(monkeypatch):
    """模型乐观自报 low_confidence=false 但置信 < 阈值 → 仍判 low（阈值兜底）。"""
    _install_fake_openai(
        monkeypatch,
        [_json({"value": "X", "confidence": 0.5, "low_confidence": False})],
    )
    res = fusion.fuse_sources(
        {"k": [_t("A")]}, {"k": [_v("B", page=1)]},
        images_by_page={1: Path("p1")}, threshold=0.6,
    )
    assert res.verdicts[0].low_confidence is True


# ---------- ③ 值不同 → 触发评判 ----------


def test_normalize_value_digit_gate():
    """仅对含数字的值剥币种/单位；纯文本（姓名）不动。"""
    assert fusion._normalize_value("200万元") == "200万"
    assert fusion._normalize_value("200万") == "200万"
    assert fusion._normalize_value("张元") == "张元"  # 无数字 → 不剥"元"
    assert fusion._normalize_value("陈意") == "陈意"


def test_name_values_not_falsely_agreed(monkeypatch):
    """非金额值（姓名）不剥币种字：文本'张元' vs 看图'张'是分歧，应送评判而非误判一致。"""
    _install_fake_openai(monkeypatch, [_json({"value": "张", "confidence": 0.85, "low_confidence": False})])
    res = fusion.fuse_sources(
        {"被保险人": [_t("张元")]}, {"被保险人": [_v("张", page=1)]},
        images_by_page={1: Path("p1")},
    )
    # 不一致 → 走评判（source=adjudicated），而非 agreed 直接采信文本
    assert res.verdicts[0].source == "adjudicated"


def test_differing_values_trigger_adjudication(monkeypatch):
    holder = _install_fake_openai(
        monkeypatch,
        [_json({"value": "200万", "confidence": 0.85, "low_confidence": False, "rationale": "图表第2行=200万"})],
    )
    res = fusion.fuse_sources(
        {"保额_一般医疗": [_t("100万")]},  # A 路错
        {"保额_一般医疗": [_v("200万", page=2)]},  # C 路对
        images_by_page={2: Path("p2")},
    )
    assert holder["client"].calls == 1  # 确实评判了
    assert res.verdicts[0].value == "200万"
    assert res.verdicts[0].source == "adjudicated"


# ---------- 降级路径 ----------


def test_no_api_key_degrades_to_no_image_verdict(monkeypatch):
    monkeypatch.setattr(fusion, "load_settings", lambda: _settings(api_key=None))
    res = fusion.fuse_sources(
        {"被保险人": [_t("张三")]}, {"被保险人": [_v("陈意", page=1)]},
        images_by_page={1: Path("p1")},
    )
    v = res.verdicts[0]
    assert v.value == "陈意"  # 无图无评判时优先采看图源
    assert v.low_confidence is True
    assert v.source == "adjudicated"
    assert res.error is not None


def test_disputed_without_images_degrades(monkeypatch):
    """有 key 但拿不到对应原图 → 无图降级、不发请求。"""
    holder = _install_fake_openai(monkeypatch, [RuntimeError("不应调用")])
    res = fusion.fuse_sources(
        {"k": [_t("A")]}, {"k": [_v("B", page=9)]},  # 候选页 9
        images_by_page={1: Path("p1")},  # 没有第 9 页图
    )
    v = res.verdicts[0]
    assert v.low_confidence is True
    assert v.value == "B"  # 优先看图源
    assert holder["client"].calls == 0  # 无图 → 跳过 LLM


def test_empty_candidates_returns_empty(monkeypatch):
    res = fusion.fuse_sources({}, {})
    assert res.verdicts == []
    assert res.overall_confidence is None


def test_adjudication_usage_merged(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            (_json({"value": "X", "confidence": 0.9}), {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}),
            (_json({"value": "Y", "confidence": 0.9}), {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}),
        ],
    )
    res = fusion.fuse_sources(
        {"k1": [_t("A")], "k2": [_t("C")]},
        {"k1": [_v("B", page=1)], "k2": [_v("D", page=1)]},
        images_by_page={1: Path("p1")},
    )
    assert res.usage == {"input_tokens": 150, "output_tokens": 15, "total_tokens": 165}


# ---------- attach_verdicts 不回写原字段 ----------


def test_attach_verdicts_does_not_touch_original_fields(monkeypatch):
    _install_fake_openai(monkeypatch, [_json({"value": "500万", "confidence": 0.8})])
    doc = DocumentExtraction(
        doc_type="保险凭证",
        amounts=[LabeledAmount(label="重疾", text="400万元", value=4_000_000.0, is_total_component=True)],
        llm_usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    res = fusion.fuse_sources(
        {"保额_重疾": [_t("400万")]}, {"保额_重疾": [_v("500万", page=1)]},
        images_by_page={1: Path("p1")},
    )
    fusion.attach_verdicts(doc, res)
    # sidecar 写入
    assert len(doc.field_verdicts) == 1
    assert doc.fusion_overall_confidence == 0.8
    # 原金额项不变量原样保留——绝不被 verdict 的不同取值污染
    assert doc.amounts[0].value == 4_000_000.0
    assert doc.amounts[0].is_total_component is True


# ---------- agent_fallback no-op ----------


def test_agent_fallback_is_noop():
    doc = DocumentExtraction(doc_type="保险凭证", fusion_overall_confidence=0.3)
    out = agent_fallback.escalate_low_confidence(doc, source_pdf="x.pdf")
    assert out is doc  # 原样返回，不改抽取
