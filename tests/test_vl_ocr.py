"""
逐页 OCR（ocr_pdf_images_with_vl）单测：覆盖单页失败隔离、全失败回退、输出截断标记、
空输入、缺凭证、重试旋钮。

不碰真实网络/模型：把 sys.modules["openai"] 换成一个 fake，按预设 behaviors 逐次返回或
抛错；encode_image_data_uri / load_settings 也 mock 掉，与 config / 本机 .env 完全隔离。
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from contract_archive.pipelines import vl_ocr


# ---------- fake OpenAI 兼容客户端 ----------


class _FakeResp:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        msg = SimpleNamespace(content=content)
        self.choices = [SimpleNamespace(message=msg, finish_reason=finish_reason)]


class _FakeClient:
    """按 behaviors 列表逐次响应 create()：元组 -> (content, finish_reason)，异常 -> 抛出。"""

    def __init__(self, behaviors: list, **init_kwargs) -> None:
        self._behaviors = list(behaviors)
        self.init_kwargs = init_kwargs
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        behavior = self._behaviors[self.calls]
        self.calls += 1
        if isinstance(behavior, Exception):
            raise behavior
        content, finish = behavior
        return _FakeResp(content, finish)


def _install_fake_openai(monkeypatch, behaviors: list) -> dict:
    """把 openai 模块换成 fake，返回 holder（事后可断言 client.init_kwargs 等）。"""
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
        dashscope_ocr_model="qwen-vl-ocr-test",
        dashscope_api_key=api_key,
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(vl_ocr, "load_settings", lambda: _settings())
    monkeypatch.setattr(vl_ocr, "encode_image_data_uri", lambda p: "data:image/png;base64,FAKE")
    monkeypatch.delenv("CONTRACT_ARCHIVE_VL_OCR_RETRIES", raising=False)
    # _FakeClient 按 self.calls 自增索引消费 behaviors，非线程安全；强制 CONCURRENCY=1
    # 让逐页 OCR 退化为串行、确定性消费 behaviors 序列。真并发保序另有专测（用线程安全 fake）。
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "1")


# ---------- 用例 ----------


def test_all_pages_ok(monkeypatch):
    _install_fake_openai(monkeypatch, [("第一页内容", "stop"), ("第二页内容", "stop")])
    out = vl_ocr.ocr_pdf_images_with_vl([Path("a.png"), Path("b.png")])
    assert "## 第 1 页\n\n第一页内容" in out
    assert "## 第 2 页\n\n第二页内容" in out
    assert vl_ocr._MARK_FAILED not in out
    assert vl_ocr._MARK_TRUNCATED not in out


def test_single_page_failure_is_isolated_and_marked(monkeypatch):
    """一页抛错不拖垮整份；失败页用 _MARK_FAILED（区别于 [看不清]），其余页照常产出。"""
    _install_fake_openai(
        monkeypatch,
        [("第一页", "stop"), RuntimeError("boom"), ("第三页", "stop")],
    )
    out = vl_ocr.ocr_pdf_images_with_vl([Path("a"), Path("b"), Path("c")])
    assert out is not None
    assert f"## 第 2 页\n\n{vl_ocr._MARK_FAILED}" in out
    assert vl_ocr._MARK_ILLEGIBLE not in out  # 技术失败不能混成"看不清"
    assert "第一页" in out and "第三页" in out


def test_all_pages_fail_returns_none(monkeypatch):
    """全部页失败 -> None，让调用方回退 MinerU。"""
    _install_fake_openai(monkeypatch, [RuntimeError("x"), RuntimeError("y")])
    assert vl_ocr.ocr_pdf_images_with_vl([Path("a"), Path("b")]) is None


def test_blank_page_marked_illegible(monkeypatch):
    """模型正常返回但本页空白 -> [看不清]；只要还有别的可用页就不返回 None。"""
    _install_fake_openai(monkeypatch, [("", "stop"), ("有内容", "stop")])
    out = vl_ocr.ocr_pdf_images_with_vl([Path("a"), Path("b")])
    assert f"## 第 1 页\n\n{vl_ocr._MARK_ILLEGIBLE}" in out
    assert "有内容" in out


def test_truncation_is_marked_but_kept(monkeypatch):
    """finish_reason==length：保留已得内容 + 追加截断标记，且仍算可用页。"""
    _install_fake_openai(monkeypatch, [("超长内容前半截", "length")])
    out = vl_ocr.ocr_pdf_images_with_vl([Path("a")])
    assert out is not None
    assert "超长内容前半截" in out
    assert vl_ocr._MARK_TRUNCATED in out


def test_empty_input_returns_empty_string():
    assert vl_ocr.ocr_pdf_images_with_vl([]) == ""


def test_missing_api_key_returns_none(monkeypatch):
    monkeypatch.setattr(vl_ocr, "load_settings", lambda: _settings(api_key=None))
    assert vl_ocr.ocr_pdf_images_with_vl([Path("a")]) is None


def test_retries_knob_passed_to_client(monkeypatch):
    holder = _install_fake_openai(monkeypatch, [("x", "stop")])
    monkeypatch.setenv("CONTRACT_ARCHIVE_VL_OCR_RETRIES", "7")
    vl_ocr.ocr_pdf_images_with_vl([Path("a")])
    assert holder["client"].init_kwargs["max_retries"] == 7


def test_retries_knob_defaults_to_4(monkeypatch):
    holder = _install_fake_openai(monkeypatch, [("x", "stop")])
    vl_ocr.ocr_pdf_images_with_vl([Path("a")])
    assert holder["client"].init_kwargs["max_retries"] == 4


def test_retries_knob_bad_value_falls_back(monkeypatch):
    holder = _install_fake_openai(monkeypatch, [("x", "stop")])
    monkeypatch.setenv("CONTRACT_ARCHIVE_VL_OCR_RETRIES", "not-an-int")
    vl_ocr.ocr_pdf_images_with_vl([Path("a")])
    assert holder["client"].init_kwargs["max_retries"] == 4


def test_concurrent_preserves_page_order(monkeypatch):
    """真并发（CONCURRENCY=4）下输出仍按页序拼接：让靠后的页先返回，顺序也不能乱。

    其余用例靠 _isolate 强制 CONCURRENCY=1 串行消费 index-based fake；这里要验真并发，
    改用线程安全 fake——据 image_url 里编进的页号回对应内容，不依赖调用计数器。
    """
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "4")
    # 把页号编进 image_url（p1..p4），让 fake 据此识别是哪一页
    monkeypatch.setattr(
        vl_ocr, "encode_image_data_uri", lambda p: f"data:image/png;base64,{p.stem}"
    )

    class _OrderFake:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            url = kwargs["messages"][0]["content"][0]["image_url"]["url"]
            n = int(url.rsplit(",", 1)[1][1:])  # "p3" -> 3
            time.sleep((5 - n) * 0.02)  # p4 最先完成、p1 最后——完成序与页序相反
            return _FakeResp(f"第{n}页正文", "stop")

    fake_mod = types.ModuleType("openai")
    fake_mod.OpenAI = _OrderFake
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    paths = [Path(f"p{i}.png") for i in range(1, 5)]
    out = vl_ocr.ocr_pdf_images_with_vl(paths)
    assert out is not None
    # 即便 p4 先返回、p1 后返回，正文必须按页序 1→2→3→4
    assert out.index("第1页正文") < out.index("第2页正文") < out.index("第3页正文") < out.index(
        "第4页正文"
    )
    # 页标题也保序
    assert out.index("## 第 1 页") < out.index("## 第 4 页")
