"""Offline durability/safety tests for evals.runtime."""
from __future__ import annotations

import json

import pytest

from evals.runtime import (
    CorruptResults,
    ManifestMismatch,
    OCRArtifactCache,
    OCRCachePayload,
    append_jsonl,
    create_or_validate_manifest,
    make_ocr_cache_key,
    output_lock,
    project_source_fingerprint,
    read_jsonl_records,
    sanitize_base_url,
    sha256_file,
    validate_records_for_manifest,
)


def _manifest():
    return {
        "schema_version": 2,
        "suite": "extraction",
        "axis": "text",
        "cases": [],
        "expected_case_ids": [],
        "repeat": 1,
        "source_fingerprint": project_source_fingerprint(),
        "config": {"base_url": "https://example.test/v1", "models": {}, "runtime_knobs": {}},
        "config_fingerprint": "config-fingerprint",
    }


def _record():
    return {
        "suite": "extraction",
        "axis": "text",
        "case_id": "case",
        "model": "candidate",
        "repeat_idx": 0,
        "config_fingerprint": "config-fingerprint",
        "status": "ok",
    }


def test_sanitize_base_url_never_retains_credentials_or_query_tokens():
    assert (
        sanitize_base_url("https://alice:super-secret@example.test:8443/compatible?token=abc#frag")
        == "https://example.test:8443/compatible"
    )
    assert sanitize_base_url("not-a-url?token=super-secret") == "<invalid-url>"
    assert sanitize_base_url("https://example.test:bad-port/path?token=super-secret") == "<invalid-url>"


def test_project_source_fingerprint_covers_the_shared_runner_source_set():
    fingerprint = project_source_fingerprint()

    assert len(fingerprint["sha256"]) == 64
    assert {
        "evals/run.py",
        "evals/runtime.py",
        "evals/score.py",
    }.issubset({entry["path"] for entry in fingerprint["files"]})


def test_manifest_is_immutable_and_corrupt_jsonl_is_not_appended(tmp_path):
    out = tmp_path / "out"
    manifest = _manifest()
    with output_lock(out):
        assert create_or_validate_manifest(out, manifest) == manifest
        append_jsonl(out / "results.jsonl", _record())
        records = read_jsonl_records(out / "results.jsonl")
        validate_records_for_manifest(records, manifest=manifest)
        changed = dict(manifest)
        changed["repeat"] = 2
        with pytest.raises(ManifestMismatch):
            create_or_validate_manifest(out, changed)

    with (out / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    with pytest.raises(CorruptResults, match="invalid JSON"):
        read_jsonl_records(out / "results.jsonl")


def test_old_nonempty_directory_without_manifest_is_refused(tmp_path):
    out = tmp_path / "legacy"
    out.mkdir()
    (out / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with output_lock(out):
        with pytest.raises(ManifestMismatch, match="no run_meta"):
            create_or_validate_manifest(out, _manifest())


def test_ocr_cache_publishes_only_complete_artifacts_and_reuses_atomically(tmp_path):
    cache = OCRArtifactCache(tmp_path / "cache")
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"private fake PDF")
    key_material = {"pdf_sha256": sha256_file(pdf), "ocr": {"model": "ocr-a"}}
    cache_key = make_ocr_cache_key(key_material)
    calls = {"complete": 0, "partial": 0}

    def complete(source, ocr_dir):
        calls["complete"] += 1
        assert source.read_bytes() == b"private fake PDF"
        ocr_dir.mkdir()
        (ocr_dir / "raw_text.txt").write_text("OCR", encoding="utf-8")
        return OCRCachePayload(
            ocr_latency_s=1.25,
            ocr_calls=[{"stage": "ocr_page", "model": "ocr-a", "usage": None}],
            pipeline_meta={"ocr_failed_pages": [], "ocr_truncated_pages": []},
            complete=True,
        )

    first = cache.get_or_create(
        cache_key=cache_key, key_material=key_material, pdf_path=pdf, producer=complete
    )
    second = cache.get_or_create(
        cache_key=cache_key, key_material=key_material, pdf_path=pdf, producer=complete
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls["complete"] == 1
    assert (first.artifact_dir / "source.pdf").is_file()
    stored = json.loads((first.artifact_dir / "cache_meta.json").read_text())
    assert stored["complete"] is True
    assert stored["ocr_calls"][0]["model"] == "ocr-a"

    partial_material = {"pdf_sha256": sha256_file(pdf), "ocr": {"model": "ocr-b"}}
    partial_key = make_ocr_cache_key(partial_material)

    def partial(_source, ocr_dir):
        calls["partial"] += 1
        ocr_dir.mkdir()
        (ocr_dir / "raw_text.txt").write_text("incomplete", encoding="utf-8")
        return OCRCachePayload(
            ocr_latency_s=0.2,
            ocr_calls=[],
            pipeline_meta={"ocr_failed_pages": [1], "ocr_truncated_pages": []},
            complete=False,
        )

    first_partial = cache.get_or_create(
        cache_key=partial_key, key_material=partial_material, pdf_path=pdf, producer=partial
    )
    assert first_partial.cache_hit is False
    assert first_partial.cleanup_dir is not None
    OCRArtifactCache.cleanup(first_partial)
    second_partial = cache.get_or_create(
        cache_key=partial_key, key_material=partial_material, pdf_path=pdf, producer=partial
    )
    assert second_partial.cache_hit is False
    assert calls["partial"] == 2
    OCRArtifactCache.cleanup(second_partial)
    assert not cache.entry_path(partial_key).exists()
