"""
评测报告：把 results/ 聚合成 gate 决策表。

核心理念（吸收方法学评审）：替换是风险问题，不是精度问题。**默认不可替换，候选必须
逐项证明非劣**才放行——不用"加权平均比大小"（平均会抹平关键文档的塌方）。

门禁（全过才有资格谈成本）：
  1. JSON 解析成功率 ≥ PARSE_FLOOR（便宜模型最常见的退化是吐非法 JSON，一条失败=整篇归零）
  2. 签章类缺陷召回 ≥ champion − δ（漏报签章=合同蒙混过关，最致命）
  3. 每个关键字段：候选 per-case 指标 bootstrap CI 下界 ≥ champion 均值 − δ
  4. 以上全过 → ELIGIBLE，才比成本/延迟（约束优化：质量是闸门，成本是目标）

加权总分只作参考列，不作单独通过条件。
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from contract_archive.schemas import DocumentExtraction

from .pricing import PRICING_AS_OF, CostSummary, actual_cost, logical_cost
from .run import candidate_request_validation, evalset_dir
from .runtime import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    config_fingerprint,
    project_source_fingerprint,
    sha256_file,
)
from .score import (
    CRITICAL_FIELDS,
    DELTA,
    PARSE_FLOOR,
    EnvelopeScore,
    bootstrap_ci,
    score_envelope,
)

# ============================================================================
# 加载 + 打分
# ============================================================================


def load_gold(cases_dir: Path, suite: str, case_id: str) -> DocumentExtraction:
    gold_path = cases_dir / suite / case_id / "gold.json"
    return DocumentExtraction.model_validate(json.loads(gold_path.read_text(encoding="utf-8")))


CaseRepeatKey = tuple[str, int]


def _repeat_idx(rec: dict[str, Any]) -> int:
    """取并校验 repeat_idx；旧结果没有该字段时按首次运行处理。"""
    raw = rec.get("repeat_idx", 0)
    try:
        repeat_idx = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"非法 repeat_idx: {raw!r}") from exc
    if repeat_idx < 0:
        raise ValueError(f"repeat_idx 不能为负数: {repeat_idx}")
    return repeat_idx


def load_results(results_dir: Path) -> tuple[str, int, dict[str, dict[str, list[dict]]]]:
    """
    读 results.jsonl → (suite, repeat, by_model)。
    by_model[model][case_id] = [记录按 repeat_idx 排序]；保留模型首次出现顺序（dict 有序）。
    同一 (model, case_id, repeat_idx) 被重跑时，JSONL **最后一条覆盖前一条**。一阶段是 append
    模式，旧实现按排序后第一条取值，会让过期结果悄悄主导 gate；这里显式采用最后写入结果，
    兼容已有 JSONL 而不混入两次运行。

    全量读取——这正是两阶段的价值：增量 append 的新模型会自然进入对比。
    """
    jsonl = results_dir / "results.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"没有 {jsonl}（先跑一阶段：python -m evals.run）")
    # 暂存为 identity -> record，dict 赋值的 "last write wins" 正是 legacy append 文件所需语义。
    slots: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    suite = "extraction"
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            suite = rec.get("suite", suite)
            repeat_idx = _repeat_idx(rec)
            slots.setdefault(rec["model"], {}).setdefault(rec["case_id"], {})[repeat_idx] = rec

    by_model: dict[str, dict[str, list[dict]]] = {}
    max_repeat = 1
    for model, cases in slots.items():
        by_model[model] = {}
        for case_id, repeats in cases.items():
            runs = [repeats[idx] for idx in sorted(repeats)]
            by_model[model][case_id] = runs
            if runs:
                max_repeat = max(max_repeat, _repeat_idx(runs[-1]) + 1)
    return suite, max_repeat, by_model


@dataclass(frozen=True)
class ManifestValidation:
    """Report-side validation of the immutable runner manifest."""

    state: str  # verified | legacy | invalid
    manifest: Optional[dict[str, Any]]
    expected_coverage: frozenset[CaseRepeatKey]
    messages: tuple[str, ...] = ()


def _manifest_string(spec: dict[str, Any], *names: str) -> Optional[str]:
    for name in names:
        value = spec.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _case_input_path(case_dir: Path, source_kind: str) -> Optional[Path]:
    if source_kind in {"pdf", "source.pdf"}:
        candidate = case_dir / "source.pdf"
        return candidate if candidate.is_file() else None
    if source_kind in {"text", "input.txt"}:
        candidate = case_dir / "input.txt"
        return candidate if candidate.is_file() else None
    return None


def _validate_manifest(results_dir: Path, cases_dir: Path, suite: str) -> ManifestValidation:
    """Validate reproducibility inputs without reading or emitting document contents."""
    path = results_dir / MANIFEST_FILENAME
    if not path.exists():
        return ManifestValidation(
            "legacy",
            None,
            frozenset(),
            ("证据不足：缺少不可变 run_meta.json，旧 JSONL 只能展示，不能形成完整替换结论",),
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ManifestValidation("invalid", None, frozenset(), (f"run_meta.json 不可读取：{exc}",))
    if not isinstance(raw, dict):
        return ManifestValidation("invalid", None, frozenset(), ("run_meta.json 必须是对象",))

    messages: list[str] = []
    required = {
        "schema_version",
        "suite",
        "axis",
        "input_mode",
        "cases",
        "expected_case_ids",
        "repeat",
        "source_fingerprint",
        "config",
        "config_fingerprint",
    }
    missing = sorted(required - set(raw))
    if missing:
        messages.append("run_meta.json 缺少字段：" + ", ".join(missing))
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        messages.append(
            f"manifest schema_version={raw.get('schema_version')!r}，期望 {MANIFEST_SCHEMA_VERSION}"
        )
    if raw.get("suite") != suite:
        messages.append(f"manifest suite={raw.get('suite')!r} 与 results suite={suite!r} 不一致")
    if not isinstance(raw.get("axis"), str) or not raw.get("axis"):
        messages.append("manifest axis 非法")
    if not isinstance(raw.get("input_mode"), str) or not raw.get("input_mode"):
        messages.append("manifest input_mode 非法")
    source_fingerprint = raw.get("source_fingerprint")
    if not isinstance(source_fingerprint, dict):
        messages.append("manifest source_fingerprint 非法")
    else:
        try:
            current_source_fingerprint = project_source_fingerprint()
        except OSError as exc:
            messages.append(f"无法重算当前 source_fingerprint：{exc}")
        else:
            if source_fingerprint != current_source_fingerprint:
                messages.append("manifest source_fingerprint 与当前评测源码不一致")

    expected_ids_raw = raw.get("expected_case_ids")
    if not isinstance(expected_ids_raw, list) or not all(
        isinstance(case_id, str) and case_id for case_id in expected_ids_raw
    ):
        messages.append("manifest expected_case_ids 非法")
        expected_ids: set[str] = set()
    else:
        expected_ids = set(expected_ids_raw)
        if len(expected_ids) != len(expected_ids_raw):
            messages.append("manifest expected_case_ids 有重复")
        if not expected_ids:
            messages.append("manifest expected_case_ids 为空")

    repeat_raw = raw.get("repeat")
    if isinstance(repeat_raw, bool) or not isinstance(repeat_raw, int) or repeat_raw < 1:
        messages.append("manifest repeat 必须是正整数")
        repeat = 0
    else:
        repeat = repeat_raw

    config = raw.get("config")
    fingerprint = raw.get("config_fingerprint")
    if not isinstance(config, dict) or not isinstance(fingerprint, str) or not fingerprint:
        messages.append("manifest config/config_fingerprint 非法")
    elif config_fingerprint(config) != fingerprint:
        messages.append("manifest config_fingerprint 与 config 内容不一致")

    case_specs = raw.get("cases")
    if not isinstance(case_specs, list) or not all(isinstance(spec, dict) for spec in case_specs):
        messages.append("manifest cases 非法")
        case_by_id: dict[str, dict[str, Any]] = {}
    else:
        case_by_id = {}
        for spec in case_specs:
            case_id = spec.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                messages.append("manifest cases 含非法 case_id")
                continue
            if case_id in case_by_id:
                messages.append(f"manifest cases 重复 case_id：{case_id}")
                continue
            case_by_id[case_id] = spec
        if expected_ids and set(case_by_id) != expected_ids:
            messages.append("manifest cases 与 expected_case_ids 不一致")

    # The case-level hashes, rather than a volatile directory timestamp, bind the report to the currently
    # supplied evaluation set.  Do not print data; only the case id and hash mismatch category are exposed.
    for case_id, spec in case_by_id.items():
        # Runner schema v2 calls this `source`; accept the earlier `source_kind` spelling too so a
        # report can still explain historical manifests rather than crashing on them.
        source_kind = spec.get("source", spec.get("source_kind"))
        if not isinstance(source_kind, str):
            messages.append(f"case {case_id} 缺少 source")
            continue
        input_hash = _manifest_string(spec, "input_sha256", "source_sha256", "input_hash")
        gold_hash = _manifest_string(spec, "gold_sha256", "gold_hash")
        case_dir = cases_dir / suite / case_id
        input_path = _case_input_path(case_dir, source_kind)
        gold_path = case_dir / "gold.json"
        if input_hash is None or gold_hash is None:
            messages.append(f"case {case_id} 缺少 input/gold SHA-256")
            continue
        if input_path is None:
            messages.append(f"case {case_id} 当前输入文件与 source_kind 不匹配")
            continue
        if not gold_path.is_file():
            messages.append(f"case {case_id} 当前 gold.json 缺失")
            continue
        try:
            if sha256_file(input_path) != input_hash:
                messages.append(f"case {case_id} 当前 input 哈希不匹配")
            if sha256_file(gold_path) != gold_hash:
                messages.append(f"case {case_id} 当前 gold 哈希不匹配")
        except OSError:
            messages.append(f"case {case_id} 当前 input/gold 无法读取")

    expected_coverage = (
        frozenset((case_id, repeat_idx) for case_id in expected_ids for repeat_idx in range(repeat))
        if expected_ids and repeat
        else frozenset()
    )
    state = "verified" if not messages else "invalid"
    return ManifestValidation(state, raw, expected_coverage, tuple(messages))


def _candidate_request_from_calls(
    record: dict[str, Any], *, axis: str, candidate: str
) -> dict[str, Any] | None:
    """Recompute candidate proof from the persisted per-call ledger.

    ``candidate_request`` is a convenience summary written by stage one, not a
    trusted source of truth.  A report must inspect the underlying ledger so an
    optional fallback that hid a timed-out candidate request cannot be presented
    as a successful candidate evaluation.
    """
    calls = record.get("calls")
    if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
        return None
    return candidate_request_validation(calls, axis=axis, candidate=candidate)


def _apply_manifest_validation(
    validation: ManifestValidation, aggs: dict[str, "ModelAgg"]
) -> None:
    """Attach immutable-manifest failures to each aggregate so public gate_verdict stays unchanged."""
    for model, agg in aggs.items():
        agg.manifest_state = validation.state
        if validation.state == "legacy":
            agg.manifest_problems.extend(validation.messages)
            continue
        if validation.state == "invalid":
            agg.manifest_problems.extend(
                f"运行清单校验失败：{message}" for message in validation.messages
            )
            continue

        manifest = validation.manifest or {}
        actual_coverage = agg.coverage_keys()
        if actual_coverage != set(validation.expected_coverage):
            missing = set(validation.expected_coverage) - actual_coverage
            extra = actual_coverage - set(validation.expected_coverage)
            details: list[str] = []
            if missing:
                details.append("缺少 " + _format_coverage(missing))
            if extra:
                details.append("多出 " + _format_coverage(extra))
            agg.manifest_problems.append(
                f"运行清单校验失败：{model} 未覆盖 manifest 预期 case/repeat（{'；'.join(details)}）"
            )

        expected_fingerprint = manifest.get("config_fingerprint")
        expected_suite = manifest.get("suite")
        expected_axis = manifest.get("axis")
        bad_fingerprint: list[str] = []
        bad_identity: list[str] = []
        bad_status: list[str] = []
        missing_candidate_call: list[str] = []
        invalid_candidate_validation: list[str] = []
        failed_candidate_call: list[str] = []
        for key, record in sorted(agg.case_repeat_records.items()):
            label = _format_coverage({key})
            if record.get("config_fingerprint") != expected_fingerprint:
                bad_fingerprint.append(label)
            if record.get("suite") != expected_suite or record.get("axis") != expected_axis:
                bad_identity.append(label)
            if record.get("status") not in {"ok", "partial", "error"}:
                bad_status.append(label)
            request = record.get("candidate_request")
            mirror = record.get("model_validation")
            if not isinstance(request, dict):
                invalid_candidate_validation.append(label)
                continue
            if (
                request.get("axis") != expected_axis
                or request.get("candidate_model") != record.get("model")
                or (isinstance(mirror, dict) and mirror != request)
            ):
                invalid_candidate_validation.append(label)
                continue
            recomputed = _candidate_request_from_calls(
                record,
                axis=str(expected_axis),
                candidate=str(record.get("model")),
            )
            if recomputed is None:
                invalid_candidate_validation.append(f"{label}（calls 缺失或非法）")
                continue
            summary_fields = (
                "axis",
                "candidate_model",
                "status",
                "observed_models",
                "observed_stages",
            )
            if any(request.get(field) != recomputed.get(field) for field in summary_fields):
                invalid_candidate_validation.append(f"{label}（summary 与 calls 不一致）")
                continue
            request_status = recomputed["status"]
            observed_models = recomputed["observed_models"]
            observed_stages = recomputed["observed_stages"]
            if request_status == "not_evaluated":
                missing_candidate_call.append(label)
            elif request_status == "request_failed":
                failed_candidate_call.append(label)
            elif (
                request_status != "verified"
                or not isinstance(observed_models, list)
                or record.get("model") not in observed_models
                or not isinstance(observed_stages, list)
                or not observed_stages
            ):
                invalid_candidate_validation.append(f"{label}（{request_status}）")
        if bad_fingerprint:
            agg.manifest_problems.append(
                "运行清单校验失败：record config_fingerprint 不一致：" + ", ".join(bad_fingerprint)
            )
        if bad_identity:
            agg.manifest_problems.append(
                "运行清单校验失败：record suite/axis 不一致：" + ", ".join(bad_identity)
            )
        if bad_status:
            agg.manifest_problems.append(
                "运行清单校验失败：record status 非法：" + ", ".join(bad_status)
            )
        if invalid_candidate_validation:
            agg.manifest_problems.append(
                "运行清单校验失败：candidate_request/model_validation 无法证明实际候选调用："
                + ", ".join(invalid_candidate_validation)
            )
        if failed_candidate_call:
            agg.manifest_problems.append(
                "运行清单校验失败：candidate axis 调用失败（request_failed）："
                + ", ".join(failed_candidate_call)
            )
        if missing_candidate_call:
            agg.manifest_problems.append(
                "证据不足：候选 axis 在这些 case 未实际调用（not_evaluated）："
                + ", ".join(missing_candidate_call)
            )


class ModelAgg:
    """单模型在整个 suite 上的聚合指标。"""

    def __init__(self, model: str):
        self.model = model
        # `envelopes` 保持旧口径：每 case 的最小 repeat_idx 用于质量/成本展示。下面的 identity
        # 索引保留所有 repeat，专供 gate 判断 "模型×case×repeat" 覆盖是否完全可比。
        self.envelopes: list[EnvelopeScore] = []
        self.case_repeat_scores: dict[CaseRepeatKey, EnvelopeScore] = {}
        self.case_repeat_records: dict[CaseRepeatKey, dict[str, Any]] = {}
        self.canonical_case_repeats: dict[str, CaseRepeatKey] = {}
        self.record_problems: dict[CaseRepeatKey, str] = {}
        self.record_warnings: dict[CaseRepeatKey, str] = {}
        self.manifest_problems: list[str] = []
        self.manifest_state = "unvalidated"  # verified | legacy | invalid

        # gold 侧适用性。非合同的 completeness=None，不应被硬塞进签章门禁；合同 case
        # 没有签章缺陷正例时，也不能把 "没有测到" 写成完整签章能力已通过。
        self.gold_doc_types: set[str] = set()
        self.signature_applicable_case_ids: set[str] = set()
        self.signature_positive_case_ids: set[str] = set()

        self.latencies: list[float] = []
        self.determinism_flags: list[bool] = []   # 多次跑结果是否一致

    def coverage_keys(self) -> set[CaseRepeatKey]:
        """实际拥有的 (case_id, repeat_idx)；兼容手工构造的旧 ModelAgg。"""
        if self.case_repeat_scores:
            return set(self.case_repeat_scores)
        return {(e.case_id, 0) for e in self.envelopes}

    def canonical_case_ids(self) -> set[str]:
        if self.canonical_case_repeats:
            return set(self.canonical_case_repeats)
        return {e.case_id for e in self.envelopes}

    def repeat_indices(self) -> set[int]:
        """Return observed repeat identities without treating repeats as samples."""
        if self.case_repeat_scores:
            return {repeat_idx for _, repeat_idx in self.case_repeat_scores}
        return {0} if self.envelopes else set()

    def _scores_for_repeat(self, repeat_idx: int) -> dict[str, EnvelopeScore]:
        """Scores keyed by case for one repeat, preserving case independence."""
        if self.case_repeat_scores:
            return {
                case_id: score
                for (case_id, score_repeat), score in self.case_repeat_scores.items()
                if score_repeat == repeat_idx
            }
        if repeat_idx == 0:
            return {score.case_id: score for score in self.envelopes}
        return {}

    def _field_scores_for_repeat(self, field: str, repeat_idx: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for case_id, envelope in self._scores_for_repeat(repeat_idx).items():
            score = next((fs for fs in envelope.fields if fs.field == field), None)
            if score is not None:
                out[case_id] = score
        return out

    def applicable_field_values_for_repeat(self, field: str, repeat_idx: int) -> dict[str, float]:
        """Gold-applicable field values for one paired repeat."""
        return {
            case_id: score.fbeta()
            for case_id, score in self._field_scores_for_repeat(field, repeat_idx).items()
            if score.support > 0
        }

    def parse_rate_for_repeat(self, repeat_idx: int) -> float:
        scores = list(self._scores_for_repeat(repeat_idx).values())
        if not scores:
            return 0.0
        return sum(1 for score in scores if score.parse_ok) / len(scores)

    def sig_recall_for_repeat(self, repeat_idx: int) -> Optional[float]:
        scores = self._scores_for_repeat(repeat_idx).values()
        tp = sum(score.sig_recall_tp for score in scores)
        fn = sum(score.sig_recall_fn for score in scores)
        return tp / (tp + fn) if (tp + fn) else None

    def _field_scores_by_case(self, field: str) -> dict[str, Any]:
        """每个 canonical case 的字段分数；字段缺失不可被当成 N/A。"""
        out: dict[str, Any] = {}
        for envelope in self.envelopes:
            score = next((fs for fs in envelope.fields if fs.field == field), None)
            if score is not None:
                out[envelope.case_id] = score
        return out

    def applicable_field_values(self, field: str) -> dict[str, float]:
        """仅 gold 有该字段期望的 canonical case；适用性由 gold 侧 support 决定。"""
        return {
            case_id: score.fbeta()
            for case_id, score in self._field_scores_by_case(field).items()
            if score.support > 0
        }

    def parse_rate(self) -> float:
        if not self.envelopes:
            return 0.0
        return sum(1 for e in self.envelopes if e.parse_ok) / len(self.envelopes)

    def field_values(self, field: str) -> list[float]:
        """该字段逐 case 的 F-beta（保留原有展示/兼容语义）。"""
        vals = []
        for e in self.envelopes:
            for fs in e.fields:
                if fs.field == field and (fs.tp + fs.fp + fs.fn) > 0:
                    vals.append(fs.fbeta())
        return vals

    def sig_recall(self) -> Optional[float]:
        tp = sum(e.sig_recall_tp for e in self.envelopes)
        fn = sum(e.sig_recall_fn for e in self.envelopes)
        return tp / (tp + fn) if (tp + fn) else None

    def weighted_scores(self) -> list[float]:
        return [e.weighted_score() for e in self.envelopes]

    def p50_p95(self) -> tuple[float, float]:
        if not self.latencies:
            return (0.0, 0.0)
        s = sorted(self.latencies)
        return (statistics.median(s), s[min(len(s) - 1, int(0.95 * len(s)))])

    def logical_cost_summary(self) -> Optional[CostSummary]:
        """每文档一次的逻辑成本：OCR 每 case 至多一次，重复评测不当作生产吞吐。"""
        if not self.case_repeat_records:
            return None
        return logical_cost(self.case_repeat_records.values())

    def actual_cost_summary(self) -> Optional[CostSummary]:
        """本次评测实际观察到的调用成本：保留 repeat，缓存复制的 OCR 不算新调用。"""
        if not self.case_repeat_records:
            return None
        return actual_cost(self.case_repeat_records.values())

    def logical_cost_per_1k(self) -> Optional[float]:
        summary = self.logical_cost_summary()
        documents = len(self.canonical_case_ids())
        if summary is None or not summary.known or not documents:
            return None
        return summary.amount_rmb / documents * 1000


def _stage_issue_messages(rec: dict[str, Any], severity: str) -> list[str]:
    """兼容运行账本放在顶层或 pred 内的 stage_issues。"""
    messages: list[str] = []
    sources = [rec]
    if isinstance(rec.get("pred"), dict):
        sources.append(rec["pred"])
    for source in sources:
        issues = source.get("stage_issues")
        if not isinstance(issues, list):
            continue
        for issue in issues:
            if not isinstance(issue, dict) or issue.get("severity") != severity:
                continue
            error = issue.get("error")
            code = error.get("code") if isinstance(error, dict) else error
            messages.append(f"{issue.get('stage', 'unknown')}:{code or 'unknown'}")
    return messages


def _record_problem(rec: dict[str, Any]) -> Optional[str]:
    """识别已明确标为失败/partial 的运行记录，避免空 pred 被误当作普通低分。"""
    for key in ("error", "errors"):
        if rec.get(key):
            return f"{key}={rec[key]!r}"
    if rec.get("partial") is True:
        return "partial=true"
    for key in ("stage_status", "status"):
        status = rec.get(key)
        if isinstance(status, str) and status.lower() in {"partial", "failed", "failure", "error"}:
            return f"{key}={status}"
    stage_errors = _stage_issue_messages(rec, "error")
    if stage_errors:
        return "stage_issues=" + ", ".join(stage_errors)
    return None


def _record_warning(rec: dict[str, Any]) -> Optional[str]:
    warnings = _stage_issue_messages(rec, "warning")
    return "stage_issues=" + ", ".join(warnings) if warnings else None


def _prediction_or_empty(rec: dict[str, Any], agg: ModelAgg, key: CaseRepeatKey) -> DocumentExtraction:
    """部分/损坏记录仍可展示其余报告，但必须在 gate 中被 record_problems 拒绝。"""
    raw_pred = rec.get("pred")
    if not isinstance(raw_pred, dict):
        agg.record_problems.setdefault(key, "pred 缺失或不是对象")
        return DocumentExtraction()
    try:
        return DocumentExtraction.model_validate(raw_pred)
    except Exception as exc:  # JSONL 是外部历史产物，报告应给出 gate 失败而非整体崩溃。
        agg.record_problems.setdefault(key, f"pred schema 无效：{exc}")
        return DocumentExtraction()


def aggregate(by_model: dict[str, dict[str, list[dict]]], suite: str, cases_dir: Path
              ) -> dict[str, ModelAgg]:
    out: dict[str, ModelAgg] = {}
    for model, cases in by_model.items():
        agg = ModelAgg(model)
        for case_id, runs in cases.items():
            gold = load_gold(cases_dir, suite, case_id)
            agg.gold_doc_types.add(gold.doc_type)
            if gold.completeness is not None:
                agg.signature_applicable_case_ids.add(case_id)
                if any(issue.category == "signature" for issue in gold.completeness.issues):
                    agg.signature_positive_case_ids.add(case_id)

            # load_results 已按 repeat_idx 排序、并做同 identity 的 last-write-wins；第一条保持
            # 原来的 canonical 统计口径，但所有 repeat 都记入 identity 索引给 gate 校验覆盖。
            canonical_score: Optional[EnvelopeScore] = None
            canonical_run: Optional[dict[str, Any]] = None
            for run in runs:
                repeat_idx = _repeat_idx(run)
                key = (case_id, repeat_idx)
                problem = _record_problem(run)
                if problem:
                    agg.record_problems[key] = problem
                warning = _record_warning(run)
                if warning:
                    agg.record_warnings[key] = warning
                pred = _prediction_or_empty(run, agg, key)
                score = score_envelope(case_id, gold, pred)
                agg.case_repeat_scores[key] = score
                agg.case_repeat_records[key] = run
                if canonical_score is None:
                    canonical_score, canonical_run = score, run
                    agg.canonical_case_repeats[case_id] = key

            if canonical_score is None or canonical_run is None:
                continue
            agg.envelopes.append(canonical_score)
            agg.latencies.append(float(canonical_run.get("latency_s", 0.0)))
            if len(runs) > 1:
                # 自一致性：多次跑的 doc_type 是否完全一致
                types = {
                    score.doc_type_pair[1]
                    for (run_case_id, _), score in agg.case_repeat_scores.items()
                    if run_case_id == case_id
                }
                agg.determinism_flags.append(len(types) == 1)
        out[model] = agg
    return out


# ============================================================================
# Gate 决策
# ============================================================================


def _format_coverage(keys: set[CaseRepeatKey]) -> str:
    return ", ".join(f"{case_id}#r{repeat_idx}" for case_id, repeat_idx in sorted(keys))


def gate_verdict(champ: ModelAgg, cand: ModelAgg) -> tuple[bool, list[str]]:
    """候选 vs champion 跑全部门禁。返回 (是否通过, 失败原因列表)。"""
    reasons: list[str] = []

    # build_report 在这里之前已把 immutable manifest 的完整性结论挂到 aggregate。保持本函数
    # 的公开签名不变，同时让旧 JSONL 只能得到“证据不足”的展示，不会再被写成完整替换通过。
    for agg in (champ, cand):
        for problem in agg.manifest_problems:
            if problem not in reasons:
                reasons.append(problem)

    # gate 的分母是 champion 实际覆盖的所有 (case, repeat)。候选少跑 case/repeat 或多混入
    # 不同批次都不可比，不能再靠已有少数高分 record 静默放行。
    champion_coverage = champ.coverage_keys()
    candidate_coverage = cand.coverage_keys()
    if champion_coverage != candidate_coverage:
        missing = champion_coverage - candidate_coverage
        extra = candidate_coverage - champion_coverage
        details: list[str] = []
        if missing:
            details.append(f"缺少 {_format_coverage(missing)}")
        if extra:
            details.append(f"多出 {_format_coverage(extra)}")
        reasons.append("评测覆盖不一致（模型×case×repeat）：" + "；".join(details))

    for label, agg in (("champion", champ), ("候选", cand)):
        if agg.record_problems:
            details = "; ".join(
                f"{case_id}#r{repeat_idx} {problem}"
                for (case_id, repeat_idx), problem in sorted(agg.record_problems.items())
            )
            reasons.append(f"{label} 存在未完成或报错记录，不能作为完整 gate 证据：{details}")

    # Repeats are independent re-runs of the same cases, not extra bootstrap
    # observations.  Gate each repeat against its matching champion repeat so a
    # good r0 cannot mask an empty/failed r1, while repeated rows never inflate
    # the apparent number of independent cases.
    quality_repeats = sorted(champ.repeat_indices() | cand.repeat_indices()) or [0]

    # 门禁 1：每个 repeat 的解析成功率绝对地板
    for repeat_idx in quality_repeats:
        champ_parse_rate = champ.parse_rate_for_repeat(repeat_idx)
        cand_parse_rate = cand.parse_rate_for_repeat(repeat_idx)
        if champ_parse_rate < PARSE_FLOOR:
            reasons.append(
                f"champion r{repeat_idx} JSON 解析成功率 {champ_parse_rate:.0%} "
                f"< 地板 {PARSE_FLOOR:.0%}，不能作为完整 gate 基准"
            )
        if cand_parse_rate < PARSE_FLOOR:
            reasons.append(
                f"候选 r{repeat_idx} JSON 解析成功率 {cand_parse_rate:.0%} "
                f"< 地板 {PARSE_FLOOR:.0%}"
            )

    # 门禁 2：签章缺陷召回非劣。gold 的 completeness=None 表示该文档类型（如保险/证明/
    # 发票）不适用合同签章核查，不强造签章 gate；但合同适用而 gold 没有任何签章正例时，只能
    # 报已测字段子集，不能据此宣称完整合同替换资格。
    if champ.signature_applicable_case_ids:
        if not champ.signature_positive_case_ids:
            reasons.append("证据不足：合同 gold 没有签章缺陷正例；只能报告字段子集，不能授权整体替换")
        else:
            for repeat_idx in quality_repeats:
                candidate_cases = set(cand._scores_for_repeat(repeat_idx))
                if not champ.signature_positive_case_ids.issubset(candidate_cases):
                    missing = champ.signature_positive_case_ids - candidate_cases
                    reasons.append(
                        f"候选 r{repeat_idx} 缺少签章正例 case：{', '.join(sorted(missing))}"
                    )
                    continue
                ch = champ.sig_recall_for_repeat(repeat_idx)
                cs = cand.sig_recall_for_repeat(repeat_idx)
                if ch is None:
                    reasons.append(
                        f"champion r{repeat_idx} 的签章正例无法评分，不能作为 gate 基准"
                    )
                elif cs is None:
                    reasons.append(
                        f"候选 r{repeat_idx} 缺少签章召回证据（champion 有签章正例）"
                    )
                elif cs < ch - DELTA:
                    reasons.append(
                        f"r{repeat_idx} 签章缺陷召回 {cs:.0%} < champion {ch:.0%} "
                        f"− {DELTA:.0%}（漏报致命）"
                    )

    # 门禁 3：每个关键字段 CI 下界非劣
    for f in CRITICAL_FIELDS:
        # N/A 必须按 gold 的 support 判定。比如非合同没有 completeness、非保险没有
        # field_verdicts，不应让合法类型恒失败；反之 champion 有 gold 期望而 candidate
        # 根本没有该字段分数，不能再靠旧代码的 `continue` 逃过门禁。
        for repeat_idx in quality_repeats:
            champ_by_case = champ.applicable_field_values_for_repeat(f, repeat_idx)
            if not champ_by_case:
                continue
            cand_scores = cand._field_scores_for_repeat(f, repeat_idx)
            missing = set(champ_by_case) - set(cand_scores)
            if missing:
                reasons.append(
                    f"关键字段 {f} r{repeat_idx} 缺少候选证据：{', '.join(sorted(missing))}"
                )
                continue
            champ_vals = list(champ_by_case.values())
            cand_vals = [cand_scores[case_id].fbeta() for case_id in champ_by_case]
            champ_mean = sum(champ_vals) / len(champ_vals)
            _, cand_lo, _ = bootstrap_ci(cand_vals)
            if cand_lo < champ_mean - DELTA:
                reasons.append(
                    f"关键字段 {f} r{repeat_idx}：候选 CI 下界 {cand_lo:.2f} "
                    f"< champion 均值 {champ_mean:.2f} − {DELTA}"
                )

    return (len(reasons) == 0, reasons)


# ============================================================================
# 渲染
# ============================================================================


def _evidence_only(reasons: list[str]) -> bool:
    """字段无已知退化、但合同签章正例等必要证据不足时，单列为范围限制。"""
    return bool(reasons) and all(reason.startswith("证据不足：") for reason in reasons)


def _record_warning_lines(agg: ModelAgg) -> list[str]:
    return [
        f"{case_id}#r{repeat_idx} {warning}"
        for (case_id, repeat_idx), warning in sorted(agg.record_warnings.items())
    ]


def _cost_cell(amount: Optional[float], *, decimals: int) -> str:
    return f"{amount:.{decimals}f}*" if amount is not None else "未知"


def _cost_reasons(summary: Optional[CostSummary]) -> list[str]:
    if summary is None:
        return ["没有可计价的运行记录"]
    return list(dict.fromkeys(summary.reasons))


def _append_cost_detail(lines: list[str], agg: ModelAgg) -> None:
    logical = agg.logical_cost_summary()
    actual = agg.actual_cost_summary()
    logical_per_1k = agg.logical_cost_per_1k()
    lines.append(
        "- 成本账本（*为原价估算）："
        f"逻辑成本 {_cost_cell(logical_per_1k, decimals=2)} ¥/千文档；"
        f"本次实际调用 {_cost_cell(actual.amount_rmb if actual else None, decimals=4)} ¥。"
    )
    if logical is None or not logical.known:
        lines.append("  - 逻辑成本未知：" + "；".join(_cost_reasons(logical)))
    if actual is None or not actual.known:
        lines.append("  - 实际调用成本未知：" + "；".join(_cost_reasons(actual)))
    if (logical and logical.uses_legacy_usage) or (actual and actual.uses_legacy_usage):
        lines.append("  - ⚠️ 仅有旧 aggregate usage，无法按 OCR/抽取实际模型拆分，属 legacy 估算。")
    cache_hits = max(logical.cache_hits if logical else 0, actual.cache_hits if actual else 0)
    if cache_hits:
        lines.append(f"  - OCR cache hit {cache_hits} 次：逻辑成本每文档只计一次 OCR；实际调用不重复计缓存复制。")
    lines.append("  - 未记录缓存 token 折扣，按北京原价估算；SDK 内部重试费用不可见，非账单金额。")


def render_markdown(run_meta: dict, aggs: dict[str, ModelAgg], champion: str) -> str:
    champ = aggs[champion]
    lines: list[str] = []
    lines.append("# 换模型评测报告（gate 决策）\n")
    lines.append(f"- suite: `{run_meta['suite']}`　champion: `{champion}`　"
                 f"case 数: {len(champ.envelopes)}　repeat: {run_meta.get('repeat', 1)}")
    lines.append(f"- 决策规则：**默认不可替换**，候选须逐项非劣（δ={DELTA}，解析地板={PARSE_FLOOR:.0%}）才放行")
    manifest_state = run_meta.get("manifest_state", "legacy")
    if manifest_state == "verified":
        lines.append("- ✅ 不可变 manifest 已校验：case 输入/gold 哈希、预期 coverage 与运行配置一致。")
    elif manifest_state == "legacy":
        lines.append("- ⚠️ **legacy 结果没有不可变 manifest**：可展示已测字段，不能形成完整可替换结论。")
    else:
        lines.append("- ❌ **run manifest 校验失败**：本报告保留已测数据供排障，不可用于替换决策。")
    n = len(champ.envelopes)
    if n < 30:
        lines.append(f"- ⚠️ **样本仅 {n} 例，CI 会很宽、统计效力不足**——结论仅供形态参考，"
                     f"表中的“已测范围通过”不是 ELIGIBLE 或整体替换授权；真实决策需按 README 扩到每分层 80-150 例。")
    doc_types = "、".join(sorted(champ.gold_doc_types)) or "未知"
    lines.append(f"- gold 覆盖类型：{doc_types}。")
    if not champ.signature_applicable_case_ids:
        lines.append("- ⚠️ 本套 gold 没有合同签章适用 case；签章门禁对该类型子集不适用，"
                     "即使字段门禁通过也**不能授权全局模型替换**。")
    elif not champ.signature_positive_case_ids:
        lines.append("- ⚠️ 合同 case 没有签章缺陷正例；签章召回尚未被证实，只能报告字段子集结果。")
    lines.append("")

    # 主表
    lines.append("## 决策总表\n")
    lines.append("| 模型 | 解析率 | 加权分(均值/CI下界) | 签章召回 | p50延迟 | p95延迟 | 逻辑¥/千文档* | 实际调用¥* | 裁决 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model, agg in aggs.items():
        ws = agg.weighted_scores()
        mean, lo, _ = bootstrap_ci(ws)
        p50, p95 = agg.p50_p95()
        logical_cost_per_1k = agg.logical_cost_per_1k()
        actual = agg.actual_cost_summary()
        logical_cost_s = _cost_cell(logical_cost_per_1k, decimals=2)
        actual_cost_s = _cost_cell(actual.amount_rmb if actual else None, decimals=4)
        sig = agg.sig_recall()
        sig_s = f"{sig:.0%}" if sig is not None else "—"
        if model == champion:
            verdict = "🏆 champion"
        else:
            ok, reasons = gate_verdict(champ, agg)
            if ok:
                verdict = "✅ 已测范围通过"
            elif _evidence_only(reasons):
                verdict = "⚠️ 证据不足（字段子集）"
            else:
                verdict = "❌ 不可替换"
        lines.append(f"| `{model}` | {agg.parse_rate():.0%} | {mean:.2f} / {lo:.2f} | "
                     f"{sig_s} | {p50:.2f}s | {p95:.2f}s | {logical_cost_s} | {actual_cost_s} | {verdict} |")
    lines.append("")

    # 候选失败原因
    for model, agg in aggs.items():
        if model == champion:
            continue
        ok, reasons = gate_verdict(champ, agg)
        lines.append(f"### `{model}` 门禁明细")
        if ok:
            cost, ccost = agg.logical_cost_per_1k(), champ.logical_cost_per_1k()
            save = f"，逻辑成本约为 champion 的 {cost / ccost:.0%}" if (cost is not None and ccost) else ""
            lines.append(f"- ✅ 通过当前 gold 适用字段的质量门禁{save}。"
                         "结论仅限已测范围，仍须结合分层样本与适用文档类型后再授权替换。\n")
        elif _evidence_only(reasons):
            lines.append("- ⚠️ 已测字段没有发现对应质量退化，但整体安全证据不足：")
            for r in reasons:
                lines.append(f"  - {r}")
            lines.append("  - **不得把字段子集结果写成整体 ELIGIBLE 或全局替换结论。**\n")
        else:
            lines.append("- ❌ 未通过，失败门禁：")
            for r in reasons:
                lines.append(f"  - {r}")
            lines.append("  - **省再多成本也不替换**（质量是闸门，不可被成本买通）。\n")
        _append_cost_detail(lines, agg)
        lines.append("")
        warnings = _record_warning_lines(agg)
        if warnings:
            lines.append("- ⚠️ 运行 warning（未自动判失败，解释结论范围）：")
            lines.extend(f"  - {warning}" for warning in warnings)
            lines.append("")

    # 关键字段逐项明细（每模型）
    lines.append("## 关键字段逐项（micro F-beta / TP·FP·FN）\n")
    lines.append("| 模型 | " + " | ".join(CRITICAL_FIELDS) + " |")
    lines.append("|---|" + "---|" * len(CRITICAL_FIELDS))
    for model, agg in aggs.items():
        cells = []
        for f in CRITICAL_FIELDS:
            tp = sum(fs.tp for e in agg.envelopes for fs in e.fields if fs.field == f)
            fp = sum(fs.fp for e in agg.envelopes for fs in e.fields if fs.field == f)
            fn = sum(fs.fn for e in agg.envelopes for fs in e.fields if fs.field == f)
            support = sum(fs.support for e in agg.envelopes for fs in e.fields if fs.field == f)
            beta = next((fs.beta for e in agg.envelopes for fs in e.fields if fs.field == f), 1.0)
            if support == 0:
                cells.append("N/A（gold 无适用项）")
            else:
                micro = _micro_fbeta(tp, fp, fn, beta)
                cells.append(f"{micro:.2f} ({tp:.0f}/{fp:.0f}/{fn:.0f})")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines.append("")

    # doc_type 混淆（仅列错分）
    lines.append("## doc_type 误分类（gold → pred）\n")
    any_err = False
    for model, agg in aggs.items():
        errs = [(e.case_id, e.doc_type_pair) for e in agg.envelopes
                if e.doc_type_pair[0] != e.doc_type_pair[1]]
        if errs:
            any_err = True
            lines.append(f"- `{model}`: " + "；".join(
                f"{cid}: {g}→{p}" for cid, (g, p) in errs))
    if not any_err:
        lines.append("- 无误分类。")
    lines.append("")

    lines.append(
        f"---\n> 价格为北京原价（核实 {PRICING_AS_OF}）的逐请求估算；未计缓存 token 折扣，"
        "SDK 内部重试费用不可见。延迟受网络/限流抖动影响，样本少时仅供量级参考。"
        "完整方法学与样本量要求见 `evals/README.md`。"
    )
    return "\n".join(lines)


def _micro_fbeta(tp: float, fp: float, fn: float, beta: float) -> float:
    if tp + fp + fn == 0:
        return 1.0
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    b2 = beta * beta
    d = b2 * p + r
    return (1 + b2) * p * r / d if d else 0.0


def build_report(results_dir: Path, cases_dir: Path, champion: Optional[str]) -> str:
    suite, repeat, by_model = load_results(results_dir)
    aggs = aggregate(by_model, suite, cases_dir)
    if not aggs:
        raise ValueError("results.jsonl 为空")
    manifest = _validate_manifest(results_dir, cases_dir, suite)
    _apply_manifest_validation(manifest, aggs)
    champ = champion or next(iter(aggs))
    if champ not in aggs:
        raise ValueError(f"champion `{champ}` 不在结果中：{list(aggs)}")
    # 让 champion 排在表首
    ordered = {champ: aggs[champ], **{m: a for m, a in aggs.items() if m != champ}}
    manifest_repeat = manifest.manifest.get("repeat") if manifest.manifest else None
    return render_markdown(
        {
            "suite": suite,
            "repeat": manifest_repeat if isinstance(manifest_repeat, int) else repeat,
            "manifest_state": manifest.state,
        },
        ordered,
        champ,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测报告：gate 决策表")
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--cases-dir", type=Path, default=None,
                    help="评测集根目录（默认 CONTRACT_ARCHIVE_EVALSET_DIR 或主仓库内合成 cases）")
    ap.add_argument("--champion", default=None, help="基准模型（默认 JSONL 里首个出现的模型）")
    ap.add_argument("--out", type=Path, default=None, help="报告输出路径（默认 results_dir/report.md）")
    args = ap.parse_args(argv)

    md = build_report(args.results_dir, args.cases_dir or evalset_dir(), args.champion)
    out = args.out or (args.results_dir / "report.md")
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n✅ 报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
