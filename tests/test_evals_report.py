"""离线验证 report 的 gate：重复 JSONL、覆盖、适用性与证据范围。"""
from __future__ import annotations

import json
from pathlib import Path

from contract_archive.schemas import DocumentExtraction
from evals.report import aggregate, build_report, gate_verdict, load_results
from evals.runtime import (
    MANIFEST_SCHEMA_VERSION,
    config_fingerprint,
    project_source_fingerprint,
    sha256_file,
)


CASES_DIR = Path(__file__).resolve().parent.parent / "evals" / "cases"


def _gold(case_id: str) -> DocumentExtraction:
    path = CASES_DIR / "extraction" / case_id / "gold.json"
    return DocumentExtraction.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _perfect_pred(case_id: str, model: str) -> dict:
    pred = _gold(case_id).model_copy(deep=True)
    pred.llm_model = model
    return pred.model_dump(mode="json")


def _record(model: str, case_id: str, repeat_idx: int = 0, *, pred: dict | None = None, **extra) -> dict:
    candidate_request = {
        "axis": "text",
        "candidate_model": model,
        "status": "verified",
        "observed_models": [model],
        "observed_stages": ["document_extract"],
    }
    return {
        "suite": "extraction",
        "axis": "text",
        "model": model,
        "case_id": case_id,
        "repeat_idx": repeat_idx,
        "pred": pred if pred is not None else _perfect_pred(case_id, model),
        "latency_s": 1.0,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "calls": [
            {
                "stage": "document_extract",
                "model": model,
                "usage": None,
                "usage_known": False,
                "status": "ok",
                "error": None,
            }
        ],
        "candidate_request": candidate_request,
        "model_validation": dict(candidate_request),
        **extra,
    }


def _write_manifest(results_dir: Path, case_ids: list[str], repeat: int, config: dict) -> str:
    fingerprint = config_fingerprint(config)
    cases = []
    for case_id in case_ids:
        case_dir = CASES_DIR / "extraction" / case_id
        cases.append(
            {
                "case_id": case_id,
                "source": "text",
                "input_sha256": sha256_file(case_dir / "input.txt"),
                "gold_sha256": sha256_file(case_dir / "gold.json"),
            }
        )
    (results_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "suite": "extraction",
                "axis": "text",
                "input_mode": "text",
                "cases": cases,
                "expected_case_ids": case_ids,
                "repeat": repeat,
                "source_fingerprint": project_source_fingerprint(),
                "config": config,
                "config_fingerprint": fingerprint,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return fingerprint


def _write_results(
    tmp_path: Path,
    records: list[dict],
    *,
    manifest: bool = False,
    expected_case_ids: list[str] | None = None,
    repeat: int | None = None,
) -> Path:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    if manifest:
        case_ids = expected_case_ids or sorted({record["case_id"] for record in records})
        expected_repeat = repeat if repeat is not None else max(record["repeat_idx"] for record in records) + 1
        config = {"base_url": "https://example.invalid/v1", "llm_concurrency": 4}
        fingerprint = _write_manifest(results_dir, case_ids, expected_repeat, config)
        for record in records:
            record.setdefault("axis", "text")
            record.setdefault("status", "ok")
            record.setdefault("config_fingerprint", fingerprint)
    (results_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return results_dir


def _aggs(results_dir: Path):
    suite, _, by_model = load_results(results_dir)
    return aggregate(by_model, suite, CASES_DIR)


def test_duplicate_identity_uses_last_jsonl_record(tmp_path: Path):
    """append 重跑同一 identity 时，过期的首条记录不能继续主导签章 gate。"""
    bad = _perfect_pred("c01_carpark_with_subagreement", "candidate")
    bad["completeness"]["issues"] = []
    records = [
        _record("champion", "c01_carpark_with_subagreement"),
        _record("candidate", "c01_carpark_with_subagreement", pred=bad, latency_s=9.0),
        _record("candidate", "c01_carpark_with_subagreement", latency_s=2.0),
    ]
    results_dir = _write_results(tmp_path, records)

    _, _, by_model = load_results(results_dir)
    runs = by_model["candidate"]["c01_carpark_with_subagreement"]
    assert len(runs) == 1
    assert runs[0]["latency_s"] == 2.0
    assert _aggs(results_dir)["candidate"].sig_recall() == 1.0


def test_case_or_repeat_coverage_mismatch_cannot_pass(tmp_path: Path):
    """候选少跑一个 repeat，即使已跑的 case 全对，也不能作为可比证据。"""
    case_ids = ["c01_carpark_with_subagreement", "c04_lease_complete"]
    records = [
        _record("champion", case_id, repeat_idx)
        for case_id in case_ids
        for repeat_idx in (0, 1)
    ]
    records.extend(
        _record("candidate", case_id, repeat_idx)
        for case_id in case_ids
        for repeat_idx in ((0, 1) if case_id == case_ids[0] else (0,))
    )
    aggs = _aggs(_write_results(tmp_path, records))

    ok, reasons = gate_verdict(aggs["champion"], aggs["candidate"])
    assert not ok
    assert any("评测覆盖不一致" in reason and "c04_lease_complete#r1" in reason for reason in reasons)


def test_later_empty_repeat_cannot_hide_behind_a_good_r0(tmp_path: Path):
    """质量 gate 按 repeat 配对，不能只看 canonical r0。"""
    case_id = "c01_carpark_with_subagreement"
    records = [
        _record("champion", case_id, 0),
        _record("champion", case_id, 1),
        _record("candidate", case_id, 0),
        _record("candidate", case_id, 1, pred={}),
    ]

    aggs = _aggs(_write_results(tmp_path, records))
    ok, reasons = gate_verdict(aggs["champion"], aggs["candidate"])

    assert not ok
    assert any("候选 r1 JSON 解析成功率" in reason for reason in reasons)


def test_repeat_quality_pairs_each_candidate_repeat_with_its_champion_repeat(tmp_path: Path):
    """r1 的基线只能是 champion r1，不能混回 r0 或伪增 bootstrap 样本。"""
    case_id = "c01_carpark_with_subagreement"
    repeated_baseline = _perfect_pred(case_id, "champion")
    repeated_baseline["doc_type"] = "其他"
    repeated_candidate = _perfect_pred(case_id, "candidate")
    repeated_candidate["doc_type"] = "其他"
    records = [
        _record("champion", case_id, 0),
        _record("champion", case_id, 1, pred=repeated_baseline),
        _record("candidate", case_id, 0),
        _record("candidate", case_id, 1, pred=repeated_candidate),
    ]

    aggs = _aggs(_write_results(tmp_path, records))
    ok, reasons = gate_verdict(aggs["champion"], aggs["candidate"])

    assert ok, reasons


def test_contract_without_signature_positive_is_evidence_insufficient(tmp_path: Path):
    """完整合同不是签章召回正例；二者都没有正例时不可写成整体通过。"""
    results_dir = _write_results(
        tmp_path,
        [_record("champion", "c04_lease_complete"), _record("candidate", "c04_lease_complete")],
        manifest=True,
    )
    aggs = _aggs(results_dir)

    ok, reasons = gate_verdict(aggs["champion"], aggs["candidate"])
    assert not ok
    assert any(reason.startswith("证据不足：合同 gold 没有签章缺陷正例") for reason in reasons)

    report = build_report(results_dir, CASES_DIR, champion="champion")
    assert "证据不足（字段子集）" in report
    assert "不得把字段子集结果写成整体 ELIGIBLE" in report


def test_non_contract_subset_does_not_require_fabricated_signature_gate(tmp_path: Path):
    """证明/发票的 gold completeness=None：签章不适用，但报告必须限制为该类型子集。"""
    case_ids = ["c02_income_certificate", "c03_vat_invoice"]
    results_dir = _write_results(
        tmp_path,
        [_record(model, case_id) for model in ("champion", "candidate") for case_id in case_ids],
        manifest=True,
    )
    aggs = _aggs(results_dir)

    ok, reasons = gate_verdict(aggs["champion"], aggs["candidate"])
    assert ok, reasons
    report = build_report(results_dir, CASES_DIR, champion="champion")
    assert "没有合同签章适用 case" in report
    assert "不能授权全局模型替换" in report


def test_existing_good_contract_cases_still_pass_applicable_gates(tmp_path: Path):
    """有签章正例的完美候选不应因无适用 verdict 等字段被误拒。"""
    case_ids = ["c01_carpark_with_subagreement", "c04_lease_complete"]
    results_dir = _write_results(
        tmp_path,
        [_record(model, case_id) for model in ("champion", "candidate") for case_id in case_ids],
        manifest=True,
    )
    aggs = _aggs(results_dir)

    ok, reasons = gate_verdict(aggs["champion"], aggs["candidate"])
    assert ok, reasons
    report = build_report(results_dir, CASES_DIR, champion="champion")
    assert "✅ 已测范围通过" in report
    assert "N/A（gold 无适用项）" in report


def test_error_status_rejects_partial_record_even_with_a_pred(tmp_path: Path):
    """运行账本已经标 error 时，不能只因保留了部分 pred 就让 gate 放行。"""
    results_dir = _write_results(
        tmp_path,
        [
            _record("champion", "c01_carpark_with_subagreement"),
            _record("candidate", "c01_carpark_with_subagreement", status="error", error={"code": "LLM"}),
        ],
    )
    aggs = _aggs(results_dir)

    ok, reasons = gate_verdict(aggs["champion"], aggs["candidate"])
    assert not ok
    assert any("未完成或报错记录" in reason for reason in reasons)


def test_legacy_jsonl_is_displayable_but_not_a_complete_replacement_verdict(tmp_path: Path):
    """没有 manifest 的历史结果可保留展示，但不能再显示为完整通过。"""
    case_ids = ["c01_carpark_with_subagreement", "c04_lease_complete"]
    results_dir = _write_results(
        tmp_path,
        [_record(model, case_id) for model in ("champion", "candidate") for case_id in case_ids],
    )

    report = build_report(results_dir, CASES_DIR, champion="champion")

    assert "legacy 结果没有不可变 manifest" in report
    assert "证据不足（字段子集）" in report
    assert "不能形成完整可替换结论" in report


def test_manifest_expected_case_prevents_both_models_omitting_it(tmp_path: Path):
    """champion 和 candidate 一起漏 case，不能再因集合相等而误放行。"""
    records = [_record(model, "c01_carpark_with_subagreement") for model in ("champion", "candidate")]
    results_dir = _write_results(
        tmp_path,
        records,
        manifest=True,
        expected_case_ids=["c01_carpark_with_subagreement", "c04_lease_complete"],
    )

    report = build_report(results_dir, CASES_DIR, champion="champion")

    assert "未覆盖 manifest 预期 case/repeat" in report
    assert "c04_lease_complete#r0" in report
    assert "❌ 不可替换" in report


def test_manifest_repeat_and_record_fingerprint_are_required(tmp_path: Path):
    """repeat 与 config fingerprint 都是不可变运行身份的一部分。"""
    records = [_record(model, "c01_carpark_with_subagreement") for model in ("champion", "candidate")]
    records[1]["config_fingerprint"] = "different-config"
    results_dir = _write_results(tmp_path, records, manifest=True, repeat=2)

    report = build_report(results_dir, CASES_DIR, champion="champion")

    assert "c01_carpark_with_subagreement#r1" in report
    assert "record config_fingerprint 不一致" in report


def test_manifest_hash_mismatch_rejects_stale_gold_or_input(tmp_path: Path):
    """报告重算当前文件哈希，不能将改过的评测集误认作旧运行的证据。"""
    records = [_record(model, "c01_carpark_with_subagreement") for model in ("champion", "candidate")]
    results_dir = _write_results(tmp_path, records, manifest=True)
    manifest_path = results_dir / "run_meta.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["gold_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    report = build_report(results_dir, CASES_DIR, champion="champion")

    assert "当前 gold 哈希不匹配" in report
    assert "run manifest 校验失败" in report


def test_manifest_source_fingerprint_mismatch_is_historical_only_not_gate_evidence(tmp_path: Path):
    records = [_record(model, "c01_carpark_with_subagreement") for model in ("champion", "candidate")]
    results_dir = _write_results(tmp_path, records, manifest=True)
    manifest_path = results_dir / "run_meta.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale = project_source_fingerprint()
    manifest["source_fingerprint"] = {**stale, "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    report = build_report(results_dir, CASES_DIR, champion="champion")

    assert "source_fingerprint 与当前评测源码不一致" in report
    assert "❌ 不可替换" in report


def test_report_rechecks_failed_candidate_call_instead_of_trusting_summary(tmp_path: Path):
    records = [_record(model, "c01_carpark_with_subagreement") for model in ("champion", "candidate")]
    candidate = next(record for record in records if record["model"] == "candidate")
    candidate["calls"][0].update({"status": "error", "error": "TimeoutError"})
    # The persisted summary is intentionally forged as "verified". The report
    # must recompute from calls and reject it.
    results_dir = _write_results(tmp_path, records, manifest=True)

    report = build_report(results_dir, CASES_DIR, champion="champion")

    assert "summary 与 calls 不一致" in report
    assert "❌ 不可替换" in report


def test_not_evaluated_candidate_axis_is_only_a_scope_limited_result(tmp_path: Path):
    """纯 native 页可能不触发某个 axis；它可展示，但不可证明该候选 axis 已验证。"""
    case_ids = ["c01_carpark_with_subagreement", "c04_lease_complete"]
    records = [_record(model, case_id) for model in ("champion", "candidate") for case_id in case_ids]
    skipped = next(record for record in records if record["model"] == "candidate" and record["case_id"] == case_ids[1])
    skipped["candidate_request"] = {
        "axis": "text",
        "candidate_model": "candidate",
        "status": "not_evaluated",
        "observed_models": [],
        "observed_stages": [],
    }
    skipped["model_validation"] = dict(skipped["candidate_request"])
    skipped["calls"] = []
    results_dir = _write_results(tmp_path, records, manifest=True)

    report = build_report(results_dir, CASES_DIR, champion="champion")

    assert "未实际调用（not_evaluated）" in report
    assert "⚠️ 证据不足（字段子集）" in report
