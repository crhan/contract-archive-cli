"""看图抽字段 read_fields_on_images 单测：候选按页标注、null 丢弃、无 key 降级、
单图失败隔离、并发保序、usage 合并。

不碰真实网络：sys.modules["openai"] 换 fake，按 behaviors 返回 JSON 串；
encode_image_data_uri / load_settings mock 掉。index-based fake 非线程安全，
靠 _isolate 强制 CONCURRENCY=1 串行消费；真并发保序另用线程安全 fake。
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from contract_archive.extraction import vl_extract

_SPEC = {"被保险人": "保障对象本人，区别于投保人", "保额_重疾": "重大疾病保险金额"}


class _FakeResp:
    def __init__(self, content: str, usage: dict | None = None) -> None:
        msg = SimpleNamespace(content=content)
        self.choices = [SimpleNamespace(message=msg)]
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
    """按 behaviors 列表逐次响应：str -> JSON content；Exception -> 抛出。"""

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
    holder: dict = {}

    def factory(**kwargs):
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
    monkeypatch.setattr(vl_extract, "load_settings", lambda: _settings())
    monkeypatch.setattr(vl_extract, "encode_image_data_uri", lambda p: "data:image/png;base64,FAKE")
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "1")


def _json(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


# ---------- 用例 ----------


def test_extracts_candidates_by_key(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            _json(
                {
                    "被保险人": {"value": "陈意", "evidence": "被保险人：陈意"},
                    "保额_重疾": {"value": "400万", "evidence": "重大疾病 400万"},
                }
            )
        ],
    )
    res = vl_extract.read_fields_on_images([Path("a.png")], _SPEC)
    assert set(res.by_key) == {"被保险人", "保额_重疾"}
    c = res.by_key["被保险人"][0]
    assert c.source == "vision"
    assert c.value == "陈意"
    assert c.page == 1
    assert "陈意" in c.evidence


def test_flat_string_value_also_accepted(monkeypatch):
    """模型回扁平字符串而非 {value,evidence} 也能规整成候选。"""
    _install_fake_openai(monkeypatch, [_json({"被保险人": "陈意", "保额_重疾": None})])
    res = vl_extract.read_fields_on_images([Path("a.png")], _SPEC)
    assert res.by_key["被保险人"][0].value == "陈意"
    assert "保额_重疾" not in res.by_key  # null 被丢弃


def test_null_and_nullish_dropped(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [_json({"被保险人": {"value": None}, "保额_重疾": {"value": "无"}})],
    )
    res = vl_extract.read_fields_on_images([Path("a.png")], _SPEC)
    assert res.by_key == {}  # null + "无" 都不产生候选


def test_per_page_candidates_tagged_with_label(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            _json({"被保险人": {"value": "陈意"}}),
            _json({"保额_重疾": {"value": "400万"}}),
        ],
    )
    res = vl_extract.read_fields_on_images(
        [Path("a.png"), Path("b.png")], _SPEC, page_labels=[3, 5]
    )
    assert res.by_key["被保险人"][0].page == 3
    assert res.by_key["保额_重疾"][0].page == 5


def test_same_key_multiple_pages_accumulate_in_order(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            _json({"被保险人": {"value": "陈意A"}}),
            _json({"被保险人": {"value": "陈意B"}}),
        ],
    )
    res = vl_extract.read_fields_on_images(
        [Path("a.png"), Path("b.png")], _SPEC, page_labels=[2, 7]
    )
    cands = res.by_key["被保险人"]
    assert [c.value for c in cands] == ["陈意A", "陈意B"]
    assert [c.page for c in cands] == [2, 7]


def test_missing_api_key_returns_error(monkeypatch):
    monkeypatch.setattr(vl_extract, "load_settings", lambda: _settings(api_key=None))
    res = vl_extract.read_fields_on_images([Path("a.png")], _SPEC)
    assert res.by_key == {}
    assert res.error is not None


def test_empty_inputs_return_empty(monkeypatch):
    _install_fake_openai(monkeypatch, [])
    assert vl_extract.read_fields_on_images([], _SPEC).by_key == {}
    assert vl_extract.read_fields_on_images([Path("a.png")], {}).by_key == {}


def test_single_image_failure_isolated(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            _json({"被保险人": {"value": "陈意"}}),
            RuntimeError("boom"),
            _json({"保额_重疾": {"value": "400万"}}),
        ],
    )
    res = vl_extract.read_fields_on_images(
        [Path("a"), Path("b"), Path("c")], _SPEC, page_labels=[1, 2, 3]
    )
    # 中间页失败不影响首尾页的候选
    assert res.by_key["被保险人"][0].value == "陈意"
    assert res.by_key["保额_重疾"][0].value == "400万"


def test_usage_merged(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            (_json({"被保险人": {"value": "A"}}), {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}),
            (_json({"保额_重疾": {"value": "B"}}), {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}),
        ],
    )
    res = vl_extract.read_fields_on_images([Path("a"), Path("b")], _SPEC)
    assert res.usage == {"input_tokens": 150, "output_tokens": 15, "total_tokens": 165}


def test_max_retries_is_two(monkeypatch):
    holder = _install_fake_openai(monkeypatch, [_json({"被保险人": {"value": "A"}})])
    vl_extract.read_fields_on_images([Path("a")], _SPEC)
    assert holder["client"].init_kwargs["max_retries"] == 2


def test_concurrent_preserves_page_order(monkeypatch):
    """真并发（CONCURRENCY=4）下候选仍按页序累积：让靠后页先返回，顺序不乱。"""
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "4")
    monkeypatch.setattr(
        vl_extract, "encode_image_data_uri", lambda p: f"data:image/png;base64,{p.stem}"
    )

    class _OrderFake:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            url = kwargs["messages"][0]["content"][1]["image_url"]["url"]
            n = int(url.rsplit(",", 1)[1][1:])  # "p3" -> 3
            time.sleep((5 - n) * 0.02)  # p4 先返回、p1 最后
            return _FakeResp(_json({"被保险人": {"value": f"page{n}"}}))

    fake_mod = types.ModuleType("openai")
    fake_mod.OpenAI = _OrderFake
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    paths = [Path(f"p{i}.png") for i in range(1, 5)]
    res = vl_extract.read_fields_on_images(paths, _SPEC, page_labels=[1, 2, 3, 4])
    cands = res.by_key["被保险人"]
    assert [c.value for c in cands] == ["page1", "page2", "page3", "page4"]
    assert [c.page for c in cands] == [1, 2, 3, 4]
