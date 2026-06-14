"""utils.concurrency 单测：保序、单项失败隔离、并发度旋钮、真并发集成、usage 合并。

纯逻辑，不碰网络/openai。
"""
from __future__ import annotations

import threading
import time

import pytest

from contract_archive.utils.concurrency import (
    DEFAULT_LLM_CONCURRENCY,
    llm_concurrency,
    map_concurrent,
    merge_usage,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", raising=False)


# ---------- map_concurrent ----------


def test_empty_returns_empty():
    assert map_concurrent(lambda x: x, []) == []


def test_single_item_serial():
    assert map_concurrent(lambda x: x * 2, [5]) == [10]


def test_basic_map():
    assert map_concurrent(lambda x: x * 2, [1, 2, 3, 4], max_workers=4) == [2, 4, 6, 8]


def test_preserves_order_despite_completion_order():
    """乱序完成仍按输入顺序返回：让大的元素先完成，结果顺序不能乱。"""

    def slow(x):
        time.sleep((5 - x) * 0.01)  # x=4 最快完成，x=1 最慢
        return x * 10

    assert map_concurrent(slow, [1, 2, 3, 4], max_workers=4) == [10, 20, 30, 40]


def test_single_failure_isolated_default_none():
    def fn(x):
        if x == 2:
            raise ValueError("boom")
        return x

    assert map_concurrent(fn, [1, 2, 3], max_workers=3) == [1, None, 3]


def test_single_failure_isolated_with_on_error():
    def fn(x):
        if x == 2:
            raise ValueError("boom")
        return x

    out = map_concurrent(fn, [1, 2, 3], max_workers=3, on_error=lambda it, e: f"err{it}")
    assert out == [1, "err2", 3]


# ---------- 并发度旋钮 ----------


def test_concurrency_knob_default():
    assert llm_concurrency() == DEFAULT_LLM_CONCURRENCY


def test_concurrency_knob_custom(monkeypatch):
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "7")
    assert llm_concurrency() == 7


def test_concurrency_knob_bad_value(monkeypatch):
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "not-an-int")
    assert llm_concurrency() == DEFAULT_LLM_CONCURRENCY


def test_concurrency_knob_non_positive(monkeypatch):
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "0")
    assert llm_concurrency() == DEFAULT_LLM_CONCURRENCY


# ---------- 真并发集成（CONCURRENCY=2）----------


def test_runs_in_parallel_capped_by_env(monkeypatch):
    """CONCURRENCY=2：确实并发（峰值并发度=2）且保序。"""
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "2")
    active = 0
    peak = 0
    lock = threading.Lock()

    def fn(x):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return x

    out = map_concurrent(fn, [1, 2, 3, 4])
    assert out == [1, 2, 3, 4]  # 保序
    assert peak == 2  # 确实并发，且被 env=2 封顶


def test_serial_when_concurrency_one(monkeypatch):
    """CONCURRENCY=1：退化为串行，永不并发（峰值=1）。fake openai 测试靠这个确定消费 behaviors。"""
    monkeypatch.setenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY", "1")
    active = 0
    peak = 0
    lock = threading.Lock()

    def fn(x):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.005)
        with lock:
            active -= 1
        return x

    out = map_concurrent(fn, [1, 2, 3])
    assert out == [1, 2, 3]
    assert peak == 1


# ---------- merge_usage ----------


def test_merge_usage_sums():
    a = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    b = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
    assert merge_usage([a, b]) == {
        "input_tokens": 150,
        "output_tokens": 30,
        "total_tokens": 180,
    }


def test_merge_usage_skips_none():
    a = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    assert merge_usage([a, None]) == a


def test_merge_usage_all_none_returns_none():
    assert merge_usage([None, None]) is None
    assert merge_usage([]) is None
