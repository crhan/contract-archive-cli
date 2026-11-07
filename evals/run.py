"""Stage one of the extraction evaluation: reproducible model matrices.

Unlike the early text-only harness, PDF cases run the real production chain:
OCR -> document routing/specialised extraction -> post-processing -> vision
fusion. A durable manifest makes an output directory one immutable experiment;
``results.jsonl`` is append-only only for new candidate labels or an explicit
``--retry-failed`` retry.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from contract_archive.archive import load_document_text
from contract_archive.archive.ingest import run_full_extraction
from contract_archive.config import load_settings
from contract_archive.extraction import extract_document
from contract_archive.pipelines import OCRPipeline
from contract_archive.utils.llm_runtime import collect_llm_calls, model_overrides

from .runtime import (
    MANIFEST_SCHEMA_VERSION,
    RESULTS_FILENAME,
    CorruptResults,
    EvalRuntimeError,
    ManifestMismatch,
    OCRArtifactCache,
    OCRCachePayload,
    append_jsonl,
    config_fingerprint,
    create_or_validate_manifest,
    fingerprint_paths,
    latest_records,
    make_ocr_cache_key,
    output_lock,
    project_source_fingerprint,
    read_jsonl_records,
    sanitize_base_url,
    sha256_file,
    validate_records_for_manifest,
)


EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
DEFAULT_CASES = EVALS_DIR / "cases"
DEFAULT_RESULTS = EVALS_DIR / "results"
RESULTS_JSONL = RESULTS_FILENAME  # Backward-compatible public constant.

_ERROR_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_TEXT_STAGES = {"document_extract", "contract_extract", "text_fields"}
_VISION_STAGES = {
    "fusion_vision",
    "fusion_adjudicate",
    "seal_vision",
}
_REQUEST_FAILURE_STATUSES = {"missing_model", "mismatch", "request_failed"}


class _OCRPipelineFailure(RuntimeError):
    """Safe OCR-boundary failure that preserves an already-observed call ledger.

    ``OCRArtifactCache`` deliberately removes an unpublished staging directory
    when its producer raises.  The runner still needs the collector snapshot to
    make the failed request auditable and priceable (or explicitly unknown), so
    this internal exception carries only safe ledger metadata and a stable error
    category—not a provider response or exception message.
    """

    def __init__(
        self,
        error: BaseException,
        *,
        calls: list[dict[str, Any]],
        ocr_latency_s: float,
    ) -> None:
        super().__init__("OCR pipeline invocation failed")
        self.error_code = _safe_error_code(error)
        self.calls = [dict(call) for call in calls]
        self.ocr_latency_s = ocr_latency_s


def evalset_dir() -> Path:
    """Find the private evaluation dataset, with synthetic public smoke fallback."""
    raw = os.getenv("CONTRACT_ARCHIVE_EVALSET_DIR")
    return Path(raw).expanduser() if raw and raw.strip() else DEFAULT_CASES


def load_cases(suite_dir: Path) -> list[dict[str, Any]]:
    """Load candidate cases without yet choosing between text and PDF input."""
    cases: list[dict[str, Any]] = []
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"suite directory does not exist: {suite_dir}")
    for case_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        gold_json = case_dir / "gold.json"
        input_txt = case_dir / "input.txt"
        source_pdf = case_dir / "source.pdf"
        if not gold_json.is_file() or not (input_txt.is_file() or source_pdf.is_file()):
            continue
        meta: dict[str, Any] = {}
        meta_path = case_dir / "meta.json"
        if meta_path.exists():
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"case meta must be a JSON object: {meta_path}")
            meta = loaded
        cases.append(
            {
                "case_id": case_dir.name,
                "input": input_txt.read_text(encoding="utf-8") if input_txt.is_file() else None,
                "input_path": input_txt if input_txt.is_file() else None,
                "pdf": source_pdf if source_pdf.is_file() else None,
                "gold_path": gold_json,
                "meta": meta,
            }
        )
    return cases


def select_cases(
    cases: list[dict[str, Any]], *, input_mode: str, case_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Choose a single immutable input source per case for this run.

    ``auto`` intentionally prefers original PDFs so a case with both artifacts
    exercises production OCR and fusion. ``text`` is retained only for direct
    historical-baseline comparison.
    """
    if input_mode not in {"auto", "text", "pdf"}:
        raise ValueError(f"unsupported input mode: {input_mode}")
    wanted = set(case_ids or [])
    available = {str(case["case_id"]) for case in cases}
    unknown = sorted(wanted - available)
    if unknown:
        raise ValueError(f"unknown case id(s): {', '.join(unknown)}")

    selected: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for case in cases:
        case_id = str(case["case_id"])
        if wanted and case_id not in wanted:
            continue
        pdf = case.get("pdf")
        text_path = case.get("input_path")
        if input_mode == "pdf":
            source_kind, source_path = "pdf", pdf
        elif input_mode == "text":
            source_kind, source_path = "text", text_path
        elif pdf is not None:
            source_kind, source_path = "pdf", pdf
        else:
            source_kind, source_path = "text", text_path
        if source_path is None:
            unavailable.append(case_id)
            continue
        selected.append({**case, "source": source_kind, "source_path": Path(source_path)})

    if unavailable and wanted:
        raise ValueError(
            f"selected input mode `{input_mode}` is unavailable for: {', '.join(unavailable)}"
        )
    if not selected:
        raise ValueError(f"no usable `{input_mode}` cases selected")
    return selected


def _model_overrides_for_axis(axis: str, candidate: str) -> dict[str, str]:
    if axis == "text":
        return {"text_model": candidate}
    if axis == "ocr":
        return {"ocr_model": candidate}
    if axis == "vision":
        # Both general VL seals and vision-fusion extraction are part of the
        # vision axis. Keeping them together prevents a hidden mixed-model run.
        return {"vl_model": candidate, "vl_extract_model": candidate}
    raise ValueError(f"unsupported evaluation axis: {axis}")


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _safe_error_code(error: BaseException) -> str:
    name = type(error).__name__
    return name if _ERROR_CODE_RE.fullmatch(name) else "error"


def _record_calls(calls: list[dict[str, Any]], *, cache_hit: bool) -> list[dict[str, Any]]:
    """Copy a safe ledger and annotate whether its OCR cost was cache-reused."""
    out: list[dict[str, Any]] = []
    for call in calls:
        item = dict(call)
        usage = item.get("usage")
        if isinstance(usage, dict):
            item["usage"] = dict(usage)
        item["cache_hit"] = cache_hit
        out.append(item)
    return out


def aggregate_usage(calls: list[dict[str, Any]]) -> dict[str, int | float] | None:
    """Sum only provider-reported token fields; never manufacture missing usage."""
    totals: dict[str, int | float] = {}
    known = False
    for call in calls:
        usage = call.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0) + value
            known = True
    return totals if known else None


def _pipeline_meta(output: Any) -> dict[str, Any]:
    return _model_dump(getattr(output, "meta", None))


def _meta_has_incomplete_ocr(meta: Mapping[str, Any] | None) -> bool:
    if not meta:
        return False
    return bool(meta.get("ocr_failed_pages") or meta.get("ocr_truncated_pages"))


def _stage_error(envelope: Any) -> bool:
    if envelope is None:
        return False
    if getattr(envelope, "extraction_error", None) is not None:
        return True
    for issue in getattr(envelope, "stage_issues", []) or []:
        dumped = _model_dump(issue)
        severity = dumped.get("severity") if dumped else getattr(issue, "severity", None)
        if severity == "error":
            return True
    return False


def _envelope_error_code(envelope: Any) -> str | None:
    """Expose only a stable structured error code, never document/error text."""
    if envelope is None:
        return None
    candidates: list[Any] = [getattr(envelope, "extraction_error", None)]
    for issue in getattr(envelope, "stage_issues", []) or []:
        dumped = _model_dump(issue)
        severity = dumped.get("severity") if dumped else getattr(issue, "severity", None)
        if severity == "error":
            candidates.append(dumped.get("error") if dumped else getattr(issue, "error", None))
    for error in candidates:
        code = error.get("code") if isinstance(error, dict) else getattr(error, "code", None)
        if isinstance(code, str) and _ERROR_CODE_RE.fullmatch(code):
            return code
    return None


def _status_for(envelope: Any, pipeline_meta: Mapping[str, Any] | None, error: BaseException | None) -> str:
    if error is not None or _stage_error(envelope):
        return "error"
    if _meta_has_incomplete_ocr(pipeline_meta):
        return "partial"
    return "ok"


def _candidate_stage(call: Mapping[str, Any], axis: str) -> bool:
    stage = call.get("stage")
    if not isinstance(stage, str):
        return False
    if axis == "text":
        return stage in _TEXT_STAGES
    if axis == "ocr":
        return stage == "ocr_page"
    # Do not broadly accept ``fusion_*``: ``fusion_text`` deliberately uses the
    # fixed text model during a vision-axis experiment and must not turn a valid
    # candidate run into a false model-mismatch failure.
    return stage in _VISION_STAGES


def candidate_request_validation(
    calls: list[dict[str, Any]], *, axis: str, candidate: str
) -> dict[str, Any]:
    """Show whether an applicable candidate request completed successfully.

    A matching stage/model alone is not evidence of a usable evaluation: the
    request may have timed out after the SDK ledger recorded it.  Missing actual
    models and mixed-model calls remain distinct configuration failures so a
    report can explain them separately from a failed provider request.
    """
    relevant = [call for call in calls if _candidate_stage(call, axis)]
    observed = sorted(
        {str(call.get("model")) for call in relevant if isinstance(call.get("model"), str)}
    )
    if not relevant:
        state = "not_evaluated"
    elif any(
        not isinstance(call.get("model"), str)
        or not call.get("model")
        or call.get("model") == "unknown"
        for call in relevant
    ):
        state = "missing_model"
    elif any(call.get("model") != candidate for call in relevant):
        state = "mismatch"
    elif any(call.get("status") != "ok" or call.get("error") is not None for call in relevant):
        state = "request_failed"
    else:
        state = "verified"
    return {
        "axis": axis,
        "candidate_model": candidate,
        "status": state,
        "observed_models": observed,
        "observed_stages": sorted(
            {str(call.get("stage")) for call in relevant if isinstance(call.get("stage"), str)}
        ),
    }


def _candidate_request_error_code(validation: Mapping[str, Any]) -> str | None:
    """Map a failed candidate-request proof to a stable record-level error code."""
    return {
        "missing_model": "CANDIDATE_MODEL_MISSING",
        "mismatch": "MODEL_MISMATCH",
        "request_failed": "CANDIDATE_REQUEST_FAILED",
    }.get(validation.get("status"))


def _round_time(value: float | int | None) -> float | None:
    return round(float(value), 3) if value is not None else None


def _with_timing(
    *,
    ocr_latency_s: float | None,
    extract_latency_s: float | None,
    cache_hit: bool,
    wall_latency_s: float | None = None,
) -> dict[str, Any]:
    logical_total = (ocr_latency_s or 0.0) + (extract_latency_s or 0.0)
    actual_ocr = 0.0 if cache_hit else ocr_latency_s
    return {
        # ``latency_s`` stays for the existing report reader. It is intentionally
        # logical: each document comparison includes one OCR cost even when a
        # later candidate reused that work from cache.
        "latency_s": _round_time(logical_total),
        "ocr_latency_s": _round_time(ocr_latency_s),
        "extract_latency_s": _round_time(extract_latency_s),
        "logical_total_latency_s": _round_time(logical_total),
        "actual_ocr_latency_s": _round_time(actual_ocr),
        "wall_latency_s": _round_time(wall_latency_s),
        "cache_hit": cache_hit,
    }


def _runtime_knobs() -> dict[str, str | None]:
    """Raw effective knobs whose changes can alter result/cost behaviour."""
    names = (
        "CONTRACT_ARCHIVE_LLM_CONCURRENCY",
        "DASHSCOPE_TIMEOUT_S",
        "CONTRACT_ARCHIVE_VL_OCR_MAX_PAGES",
        "CONTRACT_ARCHIVE_VL_OCR_DPI",
        "CONTRACT_ARCHIVE_VL_OCR_RETRIES",
        "CONTRACT_ARCHIVE_VISION_FUSION_MAX_PAGES",
        "CONTRACT_ARCHIVE_FUSION_THRESHOLD",
        "COMPUTE_DEVICE",
    )
    return {name: os.getenv(name) for name in names}


def runtime_config_for_manifest(axis: str) -> dict[str, Any]:
    """Non-secret, candidate-independent configuration for run identity."""
    settings = load_settings()
    all_models = {
        "text_model": settings.dashscope_model,
        "vl_model": settings.dashscope_vl_model,
        "ocr_model": settings.dashscope_ocr_model,
        "vl_extract_model": settings.dashscope_vl_extract_model,
    }
    excluded = {
        "text": {"text_model"},
        "ocr": {"ocr_model"},
        "vision": {"vl_model", "vl_extract_model"},
    }[axis]
    return {
        "base_url": sanitize_base_url(settings.dashscope_base_url),
        "models": {key: value for key, value in all_models.items() if key not in excluded},
        "runtime_knobs": _runtime_knobs(),
    }


def _project_source_fingerprint() -> dict[str, Any]:
    """Backward-compatible private alias for the shared runtime fingerprint."""
    return project_source_fingerprint()


def _ocr_source_fingerprint() -> dict[str, Any]:
    """Only code that can alter OCR artifacts or their model invocation."""
    relative = (
        "contract_archive/config.py",
        "contract_archive/pipelines/ocr_pipeline.py",
        "contract_archive/pipelines/vl_ocr.py",
        "contract_archive/schemas/document.py",
        "contract_archive/utils/concurrency.py",
        "contract_archive/utils/device.py",
        "contract_archive/utils/http_env.py",
        "contract_archive/utils/llm.py",
        "contract_archive/utils/llm_runtime.py",
        "contract_archive/utils/page_router.py",
        "contract_archive/utils/pdf.py",
        "evals/run.py",
        "evals/runtime.py",
    )
    return fingerprint_paths([PROJECT_ROOT / item for item in relative], root=PROJECT_ROOT)


def _ocr_key_material(pdf_path: Path) -> dict[str, Any]:
    settings = load_settings()
    knobs = _runtime_knobs()
    return {
        "cache_schema_version": 1,
        "pdf_sha256": sha256_file(pdf_path),
        "ocr": {
            "model": settings.dashscope_ocr_model,
            "base_url": sanitize_base_url(settings.dashscope_base_url),
            "vl_ocr_max_pages": knobs["CONTRACT_ARCHIVE_VL_OCR_MAX_PAGES"],
            "vl_ocr_dpi": knobs["CONTRACT_ARCHIVE_VL_OCR_DPI"],
            "vl_ocr_retries": knobs["CONTRACT_ARCHIVE_VL_OCR_RETRIES"],
            "llm_concurrency": knobs["CONTRACT_ARCHIVE_LLM_CONCURRENCY"],
            "timeout_s": knobs["DASHSCOPE_TIMEOUT_S"],
            "compute_device": knobs["COMPUTE_DEVICE"],
        },
        "ocr_source_fingerprint": _ocr_source_fingerprint(),
    }


def _run_ocr_into_cache(pdf_path: Path, cache_dir: Path):
    """Acquire a complete OCR artifact cache entry under the current overrides."""
    key_material = _ocr_key_material(pdf_path)
    cache = OCRArtifactCache(cache_dir)

    def producer(source_pdf: Path, ocr_dir: Path) -> OCRCachePayload:
        started = time.perf_counter()
        with collect_llm_calls() as collector:
            try:
                output = OCRPipeline().run(source_pdf, ocr_dir)
            except Exception as exc:
                raise _OCRPipelineFailure(
                    exc,
                    calls=collector.records,
                    ocr_latency_s=time.perf_counter() - started,
                ) from exc
        latency = time.perf_counter() - started
        meta = _pipeline_meta(output)
        return OCRCachePayload(
            ocr_latency_s=latency,
            ocr_calls=collector.records,
            pipeline_meta=meta,
            complete=not _meta_has_incomplete_ocr(meta),
        )

    return cache.get_or_create(
        cache_key=make_ocr_cache_key(key_material),
        key_material=key_material,
        pdf_path=pdf_path,
        producer=producer,
    )


def _envelope_calls(envelope: Any) -> list[dict[str, Any]]:
    """Read production's internal logical-run ledger without nesting collectors."""
    calls = getattr(envelope, "llm_calls", None)
    if not isinstance(calls, list):
        return []
    return [dict(call) for call in calls if isinstance(call, dict)]


def run_extraction_case(text: str, model: str, *, axis: str = "text") -> dict[str, Any]:
    """Run a legacy text input under scoped model overrides, without env mutation."""
    started = time.perf_counter()
    envelope: Any = None
    caught: BaseException | None = None
    calls: list[dict[str, Any]] = []
    collector = None
    try:
        with model_overrides(**_model_overrides_for_axis(axis, model)):
            # Text extraction currently has no enclosing production run boundary;
            # this collector is therefore the logical boundary for this path.
            with collect_llm_calls() as active_collector:
                collector = active_collector
                envelope = extract_document(text)
    except Exception as exc:  # Keep every matrix identity observable.
        caught = exc
    finally:
        # An observed request may have failed after recording an error ledger
        # entry. Preserve it rather than accidentally presenting the run as a
        # zero-call/zero-cost extraction.
        calls = _envelope_calls(envelope) or (collector.records if collector is not None else [])
    extract_latency = time.perf_counter() - started
    validation = candidate_request_validation(calls, axis=axis, candidate=model)
    status = _status_for(envelope, None, caught)
    if validation["status"] in _REQUEST_FAILURE_STATUSES:
        status = "error"
    error_code = _safe_error_code(caught) if caught is not None else _envelope_error_code(envelope)
    if error_code is None:
        error_code = _candidate_request_error_code(validation)
    result: dict[str, Any] = {
        "pred": _model_dump(envelope),
        "llm_model": getattr(envelope, "llm_model", None) if envelope is not None else None,
        "calls": _record_calls(calls, cache_hit=False),
        "usage": aggregate_usage(calls),
        "actual_usage": aggregate_usage(calls),
        "cached_ocr_usage": None,
        "extraction_calls_known": collector is not None,
        "status": status,
        "error": error_code,
        "candidate_request": validation,
        "model_validation": dict(validation),
    }
    result.update(
        _with_timing(
            ocr_latency_s=None,
            extract_latency_s=extract_latency,
            cache_hit=False,
            wall_latency_s=extract_latency,
        )
    )
    return result


def run_pdf_case(
    pdf_path: Path,
    model: str | None = None,
    *,
    axis: str = "text",
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a PDF through cached OCR and the complete production extraction chain."""
    candidate = model or load_settings().dashscope_model
    cache_dir = cache_dir or (DEFAULT_RESULTS / "ocr-cache")
    started = time.perf_counter()
    cache_result = None
    envelope: Any = None
    extraction_error: BaseException | None = None
    ocr_failure: _OCRPipelineFailure | None = None
    extraction_calls: list[dict[str, Any]] = []
    extract_latency: float | None = None
    try:
        with model_overrides(**_model_overrides_for_axis(axis, candidate)):
            cache_result = _run_ocr_into_cache(pdf_path, cache_dir)
            with tempfile.TemporaryDirectory(prefix="contract-archive-eval-") as raw_work:
                work = Path(raw_work) / "document"
                shutil.copytree(cache_result.artifact_dir, work, copy_function=shutil.copy2)
                ocr_dir = work / "ocr"
                text = load_document_text(ocr_dir)
                extraction_started = time.perf_counter()
                # Do not wrap this in collect_llm_calls: production owns this
                # logical extraction boundary and writes its ledger to the
                # envelope. Nested collectors are intentionally independent.
                envelope = run_full_extraction(text, ocr_dir)
                extract_latency = time.perf_counter() - extraction_started
                extraction_calls = _envelope_calls(envelope)
    except _OCRPipelineFailure as exc:
        # The cache has removed its unpublished staging directory, but a request
        # may already have reached the provider. Preserve that safe snapshot in
        # the result rather than presenting an empty/zero-cost ledger.
        ocr_failure = exc
        extraction_error = exc
    except Exception as exc:  # A failed candidate must not abort the whole matrix.
        extraction_error = exc
    finally:
        if cache_result is not None:
            OCRArtifactCache.cleanup(cache_result)

    wall_latency = time.perf_counter() - started
    ocr_calls = (
        cache_result.payload.ocr_calls
        if cache_result is not None
        else (ocr_failure.calls if ocr_failure is not None else [])
    )
    ocr_meta = cache_result.payload.pipeline_meta if cache_result is not None else {}
    cache_hit = bool(cache_result.cache_hit) if cache_result is not None else False
    ocr_latency = (
        cache_result.payload.ocr_latency_s
        if cache_result is not None
        else (ocr_failure.ocr_latency_s if ocr_failure is not None else None)
    )
    calls = _record_calls(ocr_calls, cache_hit=cache_hit) + _record_calls(
        extraction_calls, cache_hit=False
    )
    validation = candidate_request_validation(calls, axis=axis, candidate=candidate)
    status = _status_for(envelope, ocr_meta, extraction_error)
    if validation["status"] in _REQUEST_FAILURE_STATUSES:
        status = "error"
    error_code = (
        ocr_failure.error_code
        if ocr_failure is not None
        else _safe_error_code(extraction_error)
        if extraction_error is not None
        else _envelope_error_code(envelope)
    )
    if error_code is None:
        error_code = _candidate_request_error_code(validation)
    logical_usage = aggregate_usage(calls)
    actual_calls = extraction_calls if cache_hit else ocr_calls + extraction_calls
    result = {
        "pred": _model_dump(envelope),
        # Keep the text model actually used by document extraction, rather than
        # copying the candidate label for OCR/vision-axis experiments.
        "llm_model": getattr(envelope, "llm_model", None) if envelope is not None else None,
        "calls": calls,
        "usage": logical_usage,
        "actual_usage": aggregate_usage(_record_calls(actual_calls, cache_hit=False)),
        "cached_ocr_usage": aggregate_usage(ocr_calls) if cache_hit else None,
        "extraction_calls_known": envelope is not None,
        "status": status,
        "error": error_code,
        "candidate_request": validation,
        "model_validation": dict(validation),
        "ocr_meta": ocr_meta,
    }
    result.update(
        _with_timing(
            ocr_latency_s=ocr_latency,
            extract_latency_s=extract_latency,
            cache_hit=cache_hit,
            wall_latency_s=wall_latency,
        )
    )
    return result


def _case_manifest_entries(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "case_id": str(case["case_id"]),
            "source": str(case["source"]),
            "input_sha256": sha256_file(Path(case["source_path"])),
            "gold_sha256": sha256_file(Path(case["gold_path"])),
        }
        for case in cases
    ]


def build_run_manifest(
    *,
    suite: str,
    axis: str,
    input_mode: str,
    cases: list[dict[str, Any]],
    repeat: int,
) -> dict[str, Any]:
    config = runtime_config_for_manifest(axis)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "suite": suite,
        "axis": axis,
        "input_mode": input_mode,
        "cases": _case_manifest_entries(cases),
        "expected_case_ids": [str(case["case_id"]) for case in cases],
        "repeat": repeat,
        "source_fingerprint": _project_source_fingerprint(),
        "config": config,
        "config_fingerprint": config_fingerprint(config),
    }


def _base_record(
    *,
    suite: str,
    axis: str,
    case: Mapping[str, Any],
    model: str,
    repeat_idx: int,
    config_fingerprint_value: str,
) -> dict[str, Any]:
    return {
        "suite": suite,
        "axis": axis,
        "case_id": str(case["case_id"]),
        "model": model,  # candidate label, even for OCR/vision axes
        "repeat_idx": repeat_idx,
        "meta": dict(case.get("meta") or {}),
        "source": str(case["source"]),
        "config_fingerprint": config_fingerprint_value,
    }


def _unexpected_error_record(
    error: BaseException, *, axis: str, candidate: str
) -> dict[str, Any]:
    validation = {
        "axis": axis,
        "candidate_model": candidate,
        "status": "not_evaluated",
        "observed_models": [],
        "observed_stages": [],
    }
    return {
        "pred": {},
        "llm_model": None,
        "calls": [],
        "usage": None,
        "actual_usage": None,
        "cached_ocr_usage": None,
        "extraction_calls_known": False,
        "status": "error",
        "error": _safe_error_code(error),
        "candidate_request": validation,
        "model_validation": dict(validation),
        **_with_timing(
            ocr_latency_s=None,
            extract_latency_s=None,
            cache_hit=False,
            wall_latency_s=None,
        ),
    }


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _print_record(record: Mapping[str, Any], *, skipped: bool = False) -> None:
    flag = "skip" if skipped else str(record.get("status", "unknown"))
    print(
        f"  [{flag}] {record.get('model')} {record.get('case_id')}"
        f" r{record.get('repeat_idx')} source={record.get('source')}"
        f" logical={record.get('logical_total_latency_s')}s"
        f" cache_hit={record.get('cache_hit')}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="evaluation stage one: full-pipeline matrix -> results.jsonl")
    ap.add_argument("--models", required=True, help="comma-separated candidate model IDs")
    ap.add_argument("--suite", default="extraction", choices=["extraction"], help="evaluation suite")
    ap.add_argument(
        "--axis",
        default="text",
        choices=["text", "ocr", "vision"],
        help="model family under test (vision overrides both VL model roles)",
    )
    ap.add_argument(
        "--input-mode",
        default="auto",
        choices=["auto", "text", "pdf"],
        help="auto prefers source.pdf; text is for historical baseline comparison",
    )
    ap.add_argument("--case-ids", default=None, help="optional comma-separated cheap smoke subset")
    ap.add_argument(
        "--cases-dir",
        type=Path,
        default=None,
        help="dataset root (defaults to CONTRACT_ARCHIVE_EVALSET_DIR or public smoke cases)",
    )
    ap.add_argument("--out", type=Path, default=None, help="results directory")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="shared OCR cache directory (defaults to <out>/ocr-cache)",
    )
    ap.add_argument("--repeat", type=int, default=1, help="independent extraction repeats")
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="append a new last-wins record only for prior partial/error identities",
    )
    args = ap.parse_args(argv)

    models = _parse_csv(args.models)
    if not models:
        ap.error("--models must contain at least one model ID")
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")
    try:
        cases_root = args.cases_dir or evalset_dir()
        cases = select_cases(
            load_cases(cases_root / args.suite),
            input_mode=args.input_mode,
            case_ids=_parse_csv(args.case_ids),
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1

    out_dir = args.out or (DEFAULT_RESULTS / datetime.now().strftime("%Y%m%d_%H%M%S"))
    cache_dir = args.cache_dir or (out_dir / "ocr-cache")
    manifest = build_run_manifest(
        suite=args.suite,
        axis=args.axis,
        input_mode=args.input_mode,
        cases=cases,
        repeat=args.repeat,
    )
    jsonl_path = out_dir / RESULTS_JSONL

    try:
        with output_lock(out_dir):
            manifest = create_or_validate_manifest(out_dir, manifest)
            existing = read_jsonl_records(jsonl_path)
            validate_records_for_manifest(existing, manifest=manifest)
            previous = latest_records(existing)
            run_records: dict[tuple[str, str, int], dict[str, Any]] = dict(previous)

            for model in models:
                for case in cases:
                    for repeat_idx in range(args.repeat):
                        identity = (str(case["case_id"]), model, repeat_idx)
                        prior = previous.get(identity)
                        if prior is not None and (not args.retry_failed or prior.get("status") == "ok"):
                            _print_record(prior, skipped=True)
                            continue

                        started = time.perf_counter()
                        try:
                            if case["source"] == "pdf":
                                result = run_pdf_case(
                                    Path(case["source_path"]),
                                    model,
                                    axis=args.axis,
                                    cache_dir=cache_dir,
                                )
                            else:
                                text = case.get("input")
                                if not isinstance(text, str):
                                    raise ValueError("text case has no input text")
                                result = run_extraction_case(text, model, axis=args.axis)
                        except Exception as exc:  # defensive: matrix must continue
                            result = _unexpected_error_record(exc, axis=args.axis, candidate=model)
                            result["wall_latency_s"] = _round_time(time.perf_counter() - started)

                        record = _base_record(
                            suite=args.suite,
                            axis=args.axis,
                            case=case,
                            model=model,
                            repeat_idx=repeat_idx,
                            config_fingerprint_value=str(manifest["config_fingerprint"]),
                        )
                        record.update(result)
                        append_jsonl(jsonl_path, record)
                        run_records[identity] = record
                        _print_record(record)

            requested = [
                (str(case["case_id"]), model, repeat_idx)
                for model in models
                for case in cases
                for repeat_idx in range(args.repeat)
            ]
            failed = [identity for identity in requested if run_records.get(identity, {}).get("status") != "ok"]
    except (EvalRuntimeError, ManifestMismatch, CorruptResults, OSError) as exc:
        print(f"error: {exc}")
        return 1

    if failed:
        print(f"\ncompleted with {len(failed)} partial/error result(s): {jsonl_path}")
        return 1
    print(f"\ncompleted: {jsonl_path}\nnext: python -m evals.report {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
