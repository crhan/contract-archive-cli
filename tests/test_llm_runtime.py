"""Runtime-context tests: scoped models, thread propagation, and safe call ledgers."""
from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace

import pytest

from contract_archive.config import load_settings
from contract_archive.utils.concurrency import map_concurrent
from contract_archive.utils.llm_runtime import (
    collect_llm_calls,
    model_overrides,
    observed_chat_completion,
    record_llm_call,
)


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_LLM_MODEL",
        "DASHSCOPE_VL_MODEL",
        "DASHSCOPE_OCR_MODEL",
        "DASHSCOPE_VL_EXTRACT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def test_model_overrides_nest_without_environment_mutation(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_LLM_MODEL", "configured-text")
    monkeypatch.setenv("DASHSCOPE_VL_MODEL", "configured-vl")
    before = os.environ["DASHSCOPE_LLM_MODEL"]

    assert load_settings().dashscope_model == "configured-text"
    with model_overrides(
        text_model="eval-text",
        ocr_model="eval-ocr",
        vl_extract_model="eval-vl-extract",
    ):
        scoped = load_settings()
        assert scoped.dashscope_model == "eval-text"
        assert scoped.dashscope_vl_model == "configured-vl"
        assert scoped.dashscope_ocr_model == "eval-ocr"
        assert scoped.dashscope_vl_extract_model == "eval-vl-extract"

        with model_overrides(vl_model="nested-vl", text_model="nested-text"):
            nested = load_settings()
            assert nested.dashscope_model == "nested-text"
            assert nested.dashscope_vl_model == "nested-vl"
            assert nested.dashscope_ocr_model == "eval-ocr"
            assert nested.dashscope_vl_extract_model == "eval-vl-extract"

        restored_outer = load_settings()
        assert restored_outer.dashscope_model == "eval-text"
        assert restored_outer.dashscope_vl_model == "configured-vl"

    assert load_settings().dashscope_model == "configured-text"
    assert os.environ["DASHSCOPE_LLM_MODEL"] == before


def test_map_concurrent_uses_an_independent_context_for_each_worker(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_LLM_MODEL", "configured-text")
    barrier = threading.Barrier(2, timeout=2)

    def worker(model):
        before = load_settings().dashscope_model
        barrier.wait()
        with model_overrides(text_model=model):
            local = load_settings().dashscope_model
        return before, local, load_settings().dashscope_model

    with model_overrides(text_model="parent-text"):
        results = map_concurrent(worker, ["worker-a", "worker-b"], max_workers=2)
        assert load_settings().dashscope_model == "parent-text"

    assert results == [
        ("parent-text", "worker-a", "parent-text"),
        ("parent-text", "worker-b", "parent-text"),
    ]
    assert load_settings().dashscope_model == "configured-text"


def test_collector_collects_all_concurrent_calls_with_normalized_usage():
    def worker(index):
        record_llm_call(
            "extract",
            f"model-{index}",
            usage={"prompt_tokens": index + 1, "completion_tokens": 2, "ignored": "secret"},
            latency_s=0.01,
        )

    with collect_llm_calls() as collector:
        assert map_concurrent(worker, list(range(8)), max_workers=4) == [None] * 8

    records = collector.records
    assert len(records) == 8
    assert {record["model"] for record in records} == {f"model-{index}" for index in range(8)}
    assert all(record["stage"] == "extract" for record in records)
    assert all(record["status"] == "ok" for record in records)
    assert all(record["usage_known"] is True for record in records)
    for record in records:
        index = int(str(record["model"]).split("-")[1])
        assert record["usage"] == {"input_tokens": index + 1, "output_tokens": 2}
    assert "secret" not in json.dumps(records)


class _Usage:
    prompt_tokens = 11
    completion_tokens = 3
    total_tokens = 14


class _SuccessfulCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(usage=_Usage())


class _FailingCompletions:
    def create(self, **kwargs):
        raise RuntimeError("request contained TOP-SECRET-PROMPT and sk-secret-key")


def test_observed_completion_records_usage_and_never_request_content():
    secret = "TOP-SECRET-PROMPT"
    successful = _SuccessfulCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=successful))

    with collect_llm_calls() as collector:
        response = observed_chat_completion(
            client,
            "document_extract",
            model="qwen-test",
            messages=[{"role": "user", "content": secret}],
        )
        assert response.usage.prompt_tokens == 11

        no_usage_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: SimpleNamespace()))
        )
        observed_chat_completion(no_usage_client, "no_usage", model="qwen-no-usage")

        failing_client = SimpleNamespace(chat=SimpleNamespace(completions=_FailingCompletions()))
        with pytest.raises(RuntimeError, match="TOP-SECRET-PROMPT"):
            observed_chat_completion(
                failing_client,
                "failure",
                model="qwen-failure",
                messages=[{"role": "user", "content": secret}],
            )

    success, no_usage, failure = collector.records
    assert success["usage"] == {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}
    assert success["usage_known"] is True
    assert no_usage["usage"] is None
    assert no_usage["usage_known"] is False
    assert no_usage["status"] == "ok"
    assert failure["status"] == "error"
    assert failure["error"] == "RuntimeError"
    assert failure["usage"] is None
    assert failure["usage_known"] is False
    serialized = json.dumps(collector.records)
    assert secret not in serialized
    assert "sk-secret-key" not in serialized
    assert successful.kwargs["messages"][0]["content"] == secret


def test_record_without_active_collector_is_a_noop():
    assert record_llm_call("extract", "qwen", usage={"prompt_tokens": 1}) is None
