"""Pure, offline regression tests for ledger-first evaluation pricing."""
from __future__ import annotations

import pytest

from evals.pricing import (
    TOKEN_TIER_256K,
    actual_cost,
    logical_cost,
    price_call,
    record_cost,
)


def _call(model: str, input_tokens: int, output_tokens: int = 0, **extra) -> dict:
    return {
        "stage": "extract",
        "model": model,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "usage_known": True,
        "status": "ok",
        **extra,
    }


def test_mixed_actual_models_are_priced_per_call_not_candidate_label():
    record = {
        "model": "candidate-label-that-is-not-a-price-key",
        "calls": [
            _call("qwen3.5-ocr", 1_000, 100),
            _call("qwen3.8-max", 2_000, 50),
        ],
    }

    result = record_cost(record)

    expected = 1_000 / 1_000_000 * 0.5 + 100 / 1_000_000 * 2.0
    expected += 2_000 / 1_000_000 * 12.0 + 50 / 1_000_000 * 36.0
    assert result.known
    assert result.amount_rmb == pytest.approx(expected)
    assert result.source == "ledger"


@pytest.mark.parametrize(
    "call, fragment",
    [
        ({"stage": "extract", "model": "qwen3.8-max", "usage": None, "usage_known": False}, "usage"),
        (_call("unlisted-model", 10), "未收录模型价格"),
        (_call("qwen3.8-max", 10, 0) | {"usage": {"input_tokens": 10}}, "output_tokens"),
    ],
)
def test_missing_usage_or_unknown_model_makes_record_cost_unknown(call, fragment):
    result = record_cost({"calls": [call]})

    assert not result.known
    assert any(fragment in reason for reason in result.reasons)


def test_empty_calls_require_explicit_no_llm_confirmation():
    assert not record_cost({"calls": []}).known
    native = record_cost({"calls": [], "no_llm": True})
    assert native.known and native.amount_rmb == 0.0


def test_legacy_usage_is_supported_but_labeled_as_estimate_source():
    result = record_cost(
        {
            "llm_model": "qwen3.8-max",
            "usage": {"input_tokens": 1_000, "output_tokens": 0},
        }
    )

    assert result.known
    assert result.source == "legacy_usage"
    assert result.amount_rmb == pytest.approx(0.012)


def test_plus_tier_uses_256000_boundary_per_request():
    at_boundary = price_call(_call("qwen3.7-plus", TOKEN_TIER_256K, 1_000))
    above_boundary = price_call(_call("qwen3.7-plus", TOKEN_TIER_256K + 1, 1_000))

    assert at_boundary.amount_rmb == pytest.approx(TOKEN_TIER_256K / 1_000_000 * 2.0 + 1_000 / 1_000_000 * 8.0)
    assert above_boundary.amount_rmb == pytest.approx((TOKEN_TIER_256K + 1) / 1_000_000 * 6.0 + 1_000 / 1_000_000 * 24.0)


def test_cached_ocr_is_once_logical_but_not_a_new_actual_call():
    records = [
        {
            "case_id": "case-a",
            "repeat_idx": 0,
            "calls": [
                _call("qwen3.5-ocr", 1_000, stage="vl_ocr"),
                _call("qwen3.8-max", 1_000),
            ],
        },
        {
            "case_id": "case-a",
            "repeat_idx": 1,
            "calls": [
                _call("qwen3.5-ocr", 1_000, stage="vl_ocr", cache_hit=True),
                _call("qwen3.8-max", 2_000),
            ],
        },
    ]

    actual = actual_cost(records)
    logical = logical_cost(records)

    assert actual.known and logical.known
    assert actual.amount_rmb == pytest.approx(0.0005 + 0.012 + 0.024)
    assert logical.amount_rmb == pytest.approx(0.0005 + 0.012)
    assert actual.cache_hits == logical.cache_hits == 1
    assert logical.logical


def test_logical_cost_keeps_every_page_from_one_canonical_ocr_pass():
    records = [
        {
            "case_id": "two-page-case",
            "repeat_idx": 0,
            "calls": [
                _call("qwen3.5-ocr", 1_000, stage="ocr_page"),
                _call("qwen3.5-ocr", 2_000, stage="ocr_page"),
                _call("qwen3.8-max", 1_000, stage="extract"),
            ],
        },
        {
            "case_id": "two-page-case",
            "repeat_idx": 1,
            "calls": [
                _call("qwen3.5-ocr", 1_000, stage="ocr_page", cache_hit=True),
                _call("qwen3.5-ocr", 2_000, stage="ocr_page", cache_hit=True),
                _call("qwen3.8-max", 2_000, stage="extract"),
            ],
        },
    ]

    logical = logical_cost(records)
    canonical = record_cost(records[0])

    assert logical.known and canonical.known
    assert logical.amount_rmb == pytest.approx(canonical.amount_rmb)
    assert logical.amount_rmb == pytest.approx(0.0005 + 0.001 + 0.012)
    # r1's cached two-page OCR copy and extraction are not another production pass.
    assert len(logical.record_costs) == 3
