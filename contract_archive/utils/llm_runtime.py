"""Scoped runtime state and safe per-request accounting for LLM calls.

This module deliberately keeps transient evaluation state out of ``os.environ`` and
out of persisted configuration. ``ContextVar`` gives one extraction run its own
model choices and call ledger; :func:`map_concurrent` copies that context into each
worker task.

The ledger contains operational metadata only. It never accepts or stores request
messages, response content, API keys, or arbitrary SDK objects.
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelOverrides:
    """Non-persistent model choices for the current logical run.

    ``None`` means "inherit the surrounding context/configuration". The API does
    not include an API-key field on purpose: model experiments must not alter or
    expose credentials.
    """

    text_model: str | None = None
    vl_model: str | None = None
    ocr_model: str | None = None
    vl_extract_model: str | None = None


_MODEL_OVERRIDES: ContextVar[ModelOverrides] = ContextVar(
    "contract_archive_model_overrides", default=ModelOverrides()
)


def current_model_overrides() -> ModelOverrides:
    """Return the immutable overrides active in this logical execution context."""
    return _MODEL_OVERRIDES.get()


@contextmanager
def model_overrides(
    text_model: str | None = None,
    vl_model: str | None = None,
    ocr_model: str | None = None,
    vl_extract_model: str | None = None,
) -> Iterator[None]:
    """Temporarily override configured model IDs in the current context.

    Omitted (``None``) values inherit any outer override. A nested scope therefore
    changes only the model(s) it names, and its ``ContextVar`` token restores the
    exact outer state even when its body raises. This never mutates environment
    variables or configuration files.
    """
    current = current_model_overrides()
    updated = replace(
        current,
        text_model=current.text_model if text_model is None else text_model,
        vl_model=current.vl_model if vl_model is None else vl_model,
        ocr_model=current.ocr_model if ocr_model is None else ocr_model,
        vl_extract_model=(
            current.vl_extract_model if vl_extract_model is None else vl_extract_model
        ),
    )
    token = _MODEL_OVERRIDES.set(updated)
    try:
        yield
    finally:
        _MODEL_OVERRIDES.reset(token)


LlmCallRecord = dict[str, object]


def _copy_record(record: LlmCallRecord) -> LlmCallRecord:
    """Return a caller-owned record copy, including its small usage mapping."""
    copied = dict(record)
    usage = copied.get("usage")
    if isinstance(usage, dict):
        copied["usage"] = dict(usage)
    return copied


class LlmCallCollector:
    """A lock-protected, JSON-safe collection of call metadata.

    ``records`` is a snapshot rather than the mutable backing list, so concurrent
    workers can keep appending without exposing a caller to a partially-mutated
    container. Completion order is intentionally not promised for parallel calls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[LlmCallRecord] = []

    def _append(self, record: LlmCallRecord) -> None:
        with self._lock:
            self._records.append(_copy_record(record))

    @property
    def records(self) -> list[LlmCallRecord]:
        """Return a stable snapshot of all observed calls so far."""
        with self._lock:
            return [_copy_record(record) for record in self._records]

    @property
    def calls(self) -> list[LlmCallRecord]:
        """Alias for :attr:`records` for cost-reporting callers."""
        return self.records

    def snapshot(self) -> list[LlmCallRecord]:
        """Return a stable snapshot; equivalent to :attr:`records`."""
        return self.records

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __iter__(self) -> Iterator[LlmCallRecord]:
        return iter(self.records)


# Keep an all-caps spelling available for integrations that conventionally spell
# LLM as an initialism. Both names designate the same collector implementation.
LLMCallCollector = LlmCallCollector

_LLM_CALL_COLLECTOR: ContextVar[LlmCallCollector | None] = ContextVar(
    "contract_archive_llm_call_collector", default=None
)


@contextmanager
def collect_llm_calls() -> Iterator[LlmCallCollector]:
    """Collect calls issued within this logical run.

    Nested scopes are independent: calls in an inner collector do not appear in an
    outer one, and exiting the inner scope restores the outer collector.
    """
    collector = LlmCallCollector()
    token = _LLM_CALL_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _LLM_CALL_COLLECTOR.reset(token)


_USAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "total_tokens": ("total_tokens",),
}


def _field(value: object, name: str) -> object | None:
    """Read one field from a mapping or SDK-style object without retaining it."""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _finite_number(value: object) -> int | float | None:
    """Keep finite numeric values only; booleans are not meaningful token counts."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def normalize_llm_usage(usage: object | None) -> dict[str, int | float] | None:
    """Normalize OpenAI-compatible usage without inventing missing token counts.

    Only the three cost-accounting fields are retained. In particular,
    ``prompt_tokens_details`` and arbitrary mapping keys are intentionally ignored
    so a provider-specific payload cannot put request/response content in the
    ledger.
    """
    if usage is None:
        return None

    normalized: dict[str, int | float] = {}
    for output_name, aliases in _USAGE_ALIASES.items():
        for alias in aliases:
            number = _finite_number(_field(usage, alias))
            if number is not None:
                normalized[output_name] = number
                break
    return normalized or None


_ERROR_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _error_code(error: object | None) -> str | None:
    """Convert an error to a safe category, never its message or SDK payload."""
    if error is None:
        return None
    if isinstance(error, BaseException):
        candidate = type(error).__name__
    elif isinstance(error, str):
        candidate = error
    else:
        candidate = type(error).__name__
    return candidate if _ERROR_CODE_RE.fullmatch(candidate) else "error"


def _identifier(value: object, unknown: str) -> str:
    """Avoid converting arbitrary objects (whose repr might contain sensitive data)."""
    return value if isinstance(value, str) and value else unknown


def _latency(value: object | None) -> int | float | None:
    """Keep a non-negative finite duration, or mark it unknown."""
    number = _finite_number(value)
    return number if number is not None and number >= 0 else None


def record_llm_call(
    stage: str,
    model: str | None,
    usage: object | None = None,
    latency_s: float | None = None,
    error: object | None = None,
) -> LlmCallRecord | None:
    """Record one safe LLM-call summary in the active collector, if any.

    ``usage`` may be an OpenAI usage object or mapping. Unknown usage is stored as
    ``None`` with ``usage_known=False`` rather than a made-up zero-token value.
    ``error`` is reduced to an exception class or a simple caller-supplied code;
    exception messages, prompts, responses, credentials, and SDK retry internals
    are never kept. Without :func:`collect_llm_calls`, this is a no-op.
    """
    collector = _LLM_CALL_COLLECTOR.get()
    if collector is None:
        return None

    normalized_usage = normalize_llm_usage(usage)
    error_code = _error_code(error)
    record: LlmCallRecord = {
        "stage": _identifier(stage, "unknown"),
        "model": _identifier(model, "unknown"),
        "usage": normalized_usage,
        "usage_known": normalized_usage is not None,
        "latency_s": _latency(latency_s),
        "status": "error" if error_code is not None else "ok",
        "error": error_code,
    }
    collector._append(record)
    return _copy_record(record)


def observed_chat_completion(client: Any, stage: str, **kwargs: Any) -> Any:
    """Call ``client.chat.completions.create`` and record only safe metadata.

    The supplied ``kwargs`` are forwarded unchanged. This function deliberately
    does not inject a model, alter retry settings, or catch-and-retry failures; it
    records elapsed time and re-raises the original exception unchanged.
    """
    model = kwargs.get("model")
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        record_llm_call(
            stage,
            model if isinstance(model, str) else None,
            latency_s=max(0.0, time.perf_counter() - started),
            error=exc,
        )
        raise

    record_llm_call(
        stage,
        model if isinstance(model, str) else None,
        usage=normalize_llm_usage(_field(response, "usage")),
        latency_s=max(0.0, time.perf_counter() - started),
    )
    return response
