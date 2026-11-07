"""Deterministic, ledger-first LLM cost estimates for evaluation reports.

The evaluation runner records one safe entry per observed SDK call.  This module
prices those entries rather than multiplying a candidate label by aggregate
usage: a PDF pipeline can legitimately call an OCR model and an extraction model
in the same logical document.  The SDK may still retry internally and providers
may apply cache discounts that the ledger does not expose, so every non-zero
amount here is an undiscounted list-price *estimate*, never an invoice.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional


PRICING_AS_OF = "2026-11-07"
TOKEN_TIER_256K = 256_000
TOKEN_TIER_1M = 1_000_000


@dataclass(frozen=True)
class PriceTier:
    """Undiscounted ¥ / million-token list prices through ``max_input_tokens``."""

    max_input_tokens: Optional[int]
    input_rmb_per_m: float
    output_rmb_per_m: float


# 北京原价，核实日期见 PRICING_AS_OF。分档以单次 request 的 input_tokens 判定；官方的
# “256K”在本报告明确按 256,000 token 执行，边界本身有单测，避免 256000/262144 的隐式漂移。
MODEL_PRICE_TIERS: dict[str, tuple[PriceTier, ...]] = {
    "qwen3.8-flash": (PriceTier(None, 0.8, 2.7),),
    "qwen3.7-plus": (
        PriceTier(TOKEN_TIER_256K, 2.0, 8.0),
        PriceTier(TOKEN_TIER_1M, 6.0, 24.0),
    ),
    "qwen3.6-flash": (
        PriceTier(TOKEN_TIER_256K, 1.2, 7.2),
        PriceTier(TOKEN_TIER_1M, 4.8, 28.8),
    ),
    "qwen3.5-ocr": (PriceTier(None, 0.5, 2.0),),
    "qwen3.7-max": (PriceTier(None, 12.0, 36.0),),
    "qwen3.8-max": (PriceTier(None, 12.0, 36.0),),
    "qwen-vl-ocr-latest": (PriceTier(None, 0.3, 0.5),),
    "qwen-vl-ocr": (PriceTier(None, 0.3, 0.5),),
}


@dataclass(frozen=True)
class PricedCall:
    """One ledger call after priceability validation."""

    model: Optional[str]
    stage: str
    amount_rmb: Optional[float]
    reason: Optional[str] = None
    source: str = "ledger"
    cache_hit: bool = False

    @property
    def known(self) -> bool:
        return self.amount_rmb is not None


@dataclass(frozen=True)
class RecordCost:
    """Priceability of one model×case×repeat record."""

    calls: tuple[PricedCall, ...]
    amount_rmb: Optional[float]
    source: str
    reasons: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        return self.amount_rmb is not None


@dataclass(frozen=True)
class CostSummary:
    """Aggregate cost with deliberate unknown propagation."""

    amount_rmb: Optional[float]
    record_costs: tuple[RecordCost, ...]
    reasons: tuple[str, ...] = ()
    cache_hits: int = 0
    logical: bool = False

    @property
    def known(self) -> bool:
        return self.amount_rmb is not None

    @property
    def estimated(self) -> bool:
        # Ledger token counts do not expose provider cache-token discounts or hidden SDK retries.
        return self.known

    @property
    def uses_legacy_usage(self) -> bool:
        return any(record.source == "legacy_usage" for record in self.record_costs)


def _finite_nonnegative(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _usage_value(usage: Mapping[str, Any], names: tuple[str, ...]) -> Optional[float]:
    for name in names:
        if name in usage:
            return _finite_nonnegative(usage[name])
    return None


def _price_tier(model: str, input_tokens: float) -> Optional[PriceTier]:
    for tier in MODEL_PRICE_TIERS.get(model, ()):
        if tier.max_input_tokens is None or input_tokens <= tier.max_input_tokens:
            return tier
    return None


def price_call(call: Mapping[str, Any], *, source: str = "ledger") -> PricedCall:
    """Price one observed call, returning an explicit unknown instead of a zero guess."""
    model = call.get("model")
    stage = call.get("stage")
    safe_model = model if isinstance(model, str) and model else None
    safe_stage = stage if isinstance(stage, str) and stage else "unknown"
    cache_hit = call.get("cache_hit") is True
    if safe_model is None:
        return PricedCall(None, safe_stage, None, "调用缺少实际 model", source, cache_hit)
    if safe_model not in MODEL_PRICE_TIERS:
        return PricedCall(
            safe_model, safe_stage, None, f"未收录模型价格：{safe_model}", source, cache_hit
        )

    if call.get("usage_known") is False:
        return PricedCall(safe_model, safe_stage, None, "调用 usage_known=false", source, cache_hit)
    usage = call.get("usage")
    if not isinstance(usage, Mapping):
        return PricedCall(safe_model, safe_stage, None, "调用缺少 usage", source, cache_hit)
    input_tokens = _usage_value(usage, ("input_tokens", "prompt_tokens"))
    output_tokens = _usage_value(usage, ("output_tokens", "completion_tokens"))
    if input_tokens is None or output_tokens is None:
        return PricedCall(
            safe_model, safe_stage, None, "调用缺少 input_tokens 或 output_tokens", source, cache_hit
        )
    tier = _price_tier(safe_model, input_tokens)
    if tier is None:
        return PricedCall(
            safe_model,
            safe_stage,
            None,
            f"输入 {input_tokens:g} tokens 超出已收录价格档位",
            source,
            cache_hit,
        )
    amount = input_tokens / 1_000_000 * tier.input_rmb_per_m
    amount += output_tokens / 1_000_000 * tier.output_rmb_per_m
    return PricedCall(safe_model, safe_stage, amount, source=source, cache_hit=cache_hit)


def _record_model(record: Mapping[str, Any]) -> Optional[str]:
    """旧结果的 aggregate usage 只能按真实 llm_model 回退，不能默猜候选标签。"""
    for source in (record, record.get("pred")):
        if isinstance(source, Mapping):
            model = source.get("llm_model")
            if isinstance(model, str) and model:
                return model
    return None


def _explicit_no_llm(record: Mapping[str, Any]) -> bool:
    """空 calls 只有显式 no-LLM 标记才能计零，避免未接入账本的旧运行伪装免费。"""
    return record.get("no_llm") is True or record.get("llm_enabled") is False


def record_cost(record: Mapping[str, Any]) -> RecordCost:
    """Price a result record; legacy aggregate usage remains an explicitly weaker fallback."""
    calls = record.get("calls")
    if isinstance(calls, list):
        if not calls:
            if _explicit_no_llm(record):
                return RecordCost((), 0.0, "native")
            return RecordCost((), None, "ledger", ("calls 为空但未确认 no-LLM/native",))
        priced: list[PricedCall] = []
        reasons: list[str] = []
        for item in calls:
            if not isinstance(item, Mapping):
                priced_call = PricedCall(None, "unknown", None, "calls 含非对象条目")
            else:
                priced_call = price_call(item)
            priced.append(priced_call)
            if priced_call.reason:
                reasons.append(priced_call.reason)
        if reasons:
            return RecordCost(tuple(priced), None, "ledger", tuple(reasons))
        return RecordCost(tuple(priced), sum(call.amount_rmb or 0.0 for call in priced), "ledger")

    # Legacy JSONL has only aggregate usage.  It remains useful for historical display, but cannot
    # distinguish OCR/extraction calls or provider retries and is therefore always marked an estimate.
    usage = record.get("usage")
    model = _record_model(record)
    if isinstance(usage, Mapping) and model is not None:
        priced = price_call({"model": model, "stage": "legacy", "usage": usage}, source="legacy_usage")
        if priced.known:
            return RecordCost((priced,), priced.amount_rmb, "legacy_usage")
        return RecordCost((priced,), None, "legacy_usage", (priced.reason or "旧 usage 无法计价",))
    if isinstance(usage, Mapping):
        return RecordCost((), None, "legacy_usage", ("旧 usage 缺少实际 llm_model",))
    return RecordCost((), None, "unknown", ("缺少 calls 与可用 legacy usage",))


def _summary(record_costs: Sequence[RecordCost], *, cache_hits: int, logical: bool) -> CostSummary:
    reasons = tuple(reason for cost in record_costs for reason in cost.reasons)
    if reasons:
        return CostSummary(None, tuple(record_costs), reasons, cache_hits=cache_hits, logical=logical)
    return CostSummary(
        sum(cost.amount_rmb or 0.0 for cost in record_costs),
        tuple(record_costs),
        cache_hits=cache_hits,
        logical=logical,
    )


def actual_cost(records: Iterable[Mapping[str, Any]]) -> CostSummary:
    """Sum every observed record/call in this evaluation run (including repeats)."""
    materialized = list(records)
    record_costs = [_actual_record_cost(record) for record in materialized]
    return _summary(
        record_costs,
        cache_hits=sum(
            1
            for record in materialized
            for call in (record.get("calls") if isinstance(record.get("calls"), list) else ())
            if isinstance(call, Mapping) and call.get("cache_hit") is True
        ),
        logical=False,
    )


def _repeat_idx(record: Mapping[str, Any], fallback: int) -> int:
    value = record.get("repeat_idx", fallback)
    if isinstance(value, bool):
        return fallback
    try:
        repeat_idx = int(value)
    except (TypeError, ValueError):
        return fallback
    return repeat_idx if repeat_idx >= 0 else fallback


def _is_ocr(call: PricedCall) -> bool:
    return "ocr" in call.stage.casefold()


def _actual_record_cost(record: Mapping[str, Any]) -> RecordCost:
    """Price only calls that were actually issued; copied cache hits cost no new request this run."""
    calls = record.get("calls")
    if not isinstance(calls, list):
        return record_cost(record)
    # Do not let an unpriceable cached copy hide a missing usage/unknown-model hole in the ledger.
    # A cost report with any such call is deliberately unknown rather than spuriously zero.
    full_cost = record_cost(record)
    if not full_cost.known:
        return full_cost
    fresh_calls = [call for call in calls if not isinstance(call, Mapping) or call.get("cache_hit") is not True]
    if not fresh_calls:
        if calls:  # A cached ledger entry is evidence of zero *new* invocation, unlike an empty ledger.
            return RecordCost((), 0.0, "ledger")
        return record_cost(record)
    shadow = dict(record)
    shadow["calls"] = fresh_calls
    return record_cost(shadow)


def logical_cost(records: Iterable[Mapping[str, Any]]) -> CostSummary:
    """Estimate one production-like pass per document, counting OCR at most once per case.

    Repeats are useful for quality/determinism but are not a production document multiplier.  The
    canonical (lowest repeat_idx) record contributes non-OCR calls; one canonical observed OCR
    pass (including every page call in that pass) contributes once per case.  Actual run cost is
    available separately via :func:`actual_cost`.
    Unknown usage/model in *any* input record propagates to unknown here too, rather than hiding an
    unpriced call behind de-duplication.
    """
    materialized = list(records)
    all_record_costs = [record_cost(record) for record in materialized]
    cache_hits = sum(
        1
        for record in materialized
        for call in (record.get("calls") if isinstance(record.get("calls"), list) else ())
        if isinstance(call, Mapping) and call.get("cache_hit") is True
    )
    strict = _summary(all_record_costs, cache_hits=cache_hits, logical=True)
    if not strict.known:
        return strict

    indexed = list(enumerate(zip(materialized, all_record_costs)))
    by_case: dict[str, list[tuple[int, Mapping[str, Any], RecordCost]]] = {}
    for index, (record, cost) in indexed:
        raw_case_id = record.get("case_id")
        case_id = raw_case_id if isinstance(raw_case_id, str) and raw_case_id else f"__record_{index}"
        by_case.setdefault(case_id, []).append((index, record, cost))

    selected_calls: list[PricedCall] = []
    for entries in by_case.values():
        entries.sort(key=lambda item: (_repeat_idx(item[1], item[0]), item[0]))
        _, _, canonical_cost = entries[0]
        selected_calls.extend(call for call in canonical_cost.calls if not _is_ocr(call))
        # An OCR pass often emits one ``ocr_page`` ledger item per page.  Select
        # one canonical *pass* (not one arbitrary call) and preserve all of its
        # page calls; later repeats may contain cache copies of the same pass.
        canonical_ocr_cost = next(
            (
                cost
                for _, _, cost in entries
                if any(_is_ocr(call) for call in cost.calls)
            ),
            None,
        )
        if canonical_ocr_cost is not None:
            selected_calls.extend(call for call in canonical_ocr_cost.calls if _is_ocr(call))

    logical_records = tuple(
        RecordCost((call,), call.amount_rmb, "ledger" if call.source == "ledger" else call.source)
        for call in selected_calls
    )
    return _summary(logical_records, cache_hits=cache_hits, logical=True)
