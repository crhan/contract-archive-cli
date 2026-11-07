"""
单 PDF 入库流水线。

流程（每个 PDF 一次调用）：
  1) 流式 SHA256，并持有该 SHA 前缀的跨进程槽锁
  2) 锁内查 documents.sha256；命中 + 非 reingest 直接 skip
  3) 在 tmp/runs/<sha-short>-<uuid>/ 做 source.pdf 独立快照、OCR + 抽取
  4) 完整 staging rename 到 documents/<sha-short>.gen-<uuid>/（不可变 generation）
  5) DB 写入 documents + risk_clauses，或在同一事务中切换 output_dir pointer
  6) 追加一行 ingest.jsonl 总日志
  7) 失败时：新文档可发布 failed generation；已有成功档案保留旧 pointer + 独立失败日志

状态语义：
  - ok       OCR + 抽取都成功
  - partial  OCR 成功但 LLM 失败 → markdown 可用，可后续 extract 命令重跑
  - failed   OCR 失败 → 没有 OCR 产物，但 source.pdf 留档可查
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import stat
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from ..errors import ErrorInfo, classify_exception, extract_empty, ocr_failed
from ..extraction import extract_contract, extract_document
from ..extraction.agent_fallback import escalate_low_confidence
from ..extraction.doc_type_handlers import get_handler
from ..extraction.evidence_page_fix import correct_evidence_pages
from ..extraction.fusion import DEFAULT_FUSION_THRESHOLD, run_vision_fusion
from ..pipelines import OCRPipeline
from ..schemas import (
    FILE_EXTRACTION,
    FILE_EXTRACTION_CONF,
    FILE_MARKDOWN,
    FILE_PIPELINE_META,
    FILE_RAW_TEXT,
    PREVIEW_DIR,
    ContractExtraction,
    DocumentExtraction,
    ExtractionConfidence,
    PipelineMeta,
    StageIssue,
)
from ..utils import classify_pages
from ..utils.llm_runtime import collect_llm_calls, normalize_llm_usage
from ..utils.page_router import MODE_OCR
from ..utils.pdf import render_pdf_to_images
from .party_registry import PartyRegistry
from .paths import ArchivePaths, SHA_SHORT_LEN, link_or_copy, sha256_of_file
from .repository import (
    contract_to_envelope,
    find_by_sha,
    get_document,
    insert_document,
    replace_document,
    update_extraction,
    update_output_dir,
)

logger = logging.getLogger(__name__)


# ---------- 结果类型 ----------


@dataclass
class IngestResult:
    """单 PDF 入库结果，CLI 用来汇总/打印。"""

    pdf_path: Path
    sha256: str
    status: str               # ok | partial | failed | skipped
    doc_id: Optional[int]     # 写入/已存在的 documents.id
    ocr_duration_s: Optional[float] = None
    llm_duration_s: Optional[float] = None
    error_message: Optional[str] = None        # 人类可读错误（同时写入 DB documents.error_message）
    error: Optional[ErrorInfo] = None          # 结构化错误（仅 CLI --format json 输出，不入库）
    skipped_reason: Optional[str] = None


# ---------- 抽取调度（LLM-first） ----------


def _envelope_confidence(env: DocumentExtraction) -> float:
    """
    非合同文档的总体置信度启发式（LLM-first，无 rule 交叉验证）。
    有标题/摘要算基础 0.5，每多一类柔性信息（主体/金额/字段/日期）+0.1，封顶 0.9。
    """
    if not env.title and not env.summary:
        return 0.0
    rich = sum(bool(x) for x in (env.parties, env.amounts, env.fields, env.key_dates))
    return min(0.9, 0.5 + 0.1 * rich)


def _run_extraction(
    document_text: str, llm_enabled: bool
) -> tuple[ContractExtraction, ExtractionConfidence, DocumentExtraction]:
    """
    LLM-first 抽取：先判类型抽通用信封；据 doc_type 查处理器跑第二层特化抽取。
    返回 (合同抽取, 置信度, 通用信封)——三者一并交给 repository 落库。
    """
    if not llm_enabled:
        # 无 LLM：rule 抽取自 Phase 2 已退役，extract_contract(llm_enabled=False) 返回空对象——
        # 即 --no-llm 下抽取字段留空（仅 OCR 产物入库），可后续 `extract <id>` 补抽。
        ext, conf = extract_contract(document_text, llm_enabled=False)
        return ext, conf, contract_to_envelope(ext)

    envelope = extract_document(document_text, llm_enabled=llm_enabled)
    handler = get_handler(envelope.doc_type)
    if handler.specialized_extractor is not None:
        # 第二层特化（合同→extract_contract；保险→insurance）。可就地 enrich envelope。
        ext, conf = handler.specialized_extractor(document_text, envelope, llm_enabled)
        return ext, conf, envelope
    # 无特化的类型：无专属列，overall 走信封启发式
    conf = ExtractionConfidence()
    conf.overall = _envelope_confidence(envelope)
    return ContractExtraction(), conf, envelope


def _attach_runtime_ledger(
    envelope: DocumentExtraction,
    records: list[dict[str, object]],
) -> None:
    """Attach observed calls, preserving envelope metadata when mocks made none."""
    if not records:
        return
    envelope.llm_calls = [dict(record) for record in records]
    totals: dict[str, int | float] = {}
    for record in records:
        usage = normalize_llm_usage(record.get("usage"))
        if usage is None:
            continue
        for name, value in usage.items():
            totals[name] = totals.get(name, 0) + value
    if totals:
        envelope.llm_usage = totals


def _append_stage_issue(
    envelope: DocumentExtraction,
    *,
    stage: str,
    error: ErrorInfo,
    severity: Literal["error", "warning"],
) -> None:
    """Append a stage issue once, preserving extractor-supplied issues."""
    if any(
        issue.stage == stage
        and issue.error.code == error.code
        and issue.severity == severity
        for issue in envelope.stage_issues
    ):
        return
    envelope.stage_issues.append(StageIssue(stage=stage, error=error, severity=severity))


def _ocr_incomplete_error(ocr_dir: Path) -> ErrorInfo | None:
    """Read durable page-level OCR failure markers, preserving old archives."""
    path = ocr_dir / FILE_PIPELINE_META
    if not path.is_file():
        return None
    try:
        meta = PipelineMeta.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - manually created old archives can lack metadata shape
        logger.warning("could not read OCR pipeline metadata %s: %s", path, exc)
        return None
    failed = sorted(set(meta.ocr_failed_pages))
    truncated = sorted(set(meta.ocr_truncated_pages))
    if not failed and not truncated:
        return None
    details: list[str] = []
    if failed:
        details.append(f"failed pages={failed}")
    if truncated:
        details.append(f"truncated pages={truncated}")
    return ErrorInfo(
        code="OCR_INCOMPLETE",
        category="infra",
        message="OCR incomplete: " + "; ".join(details),
        retryable=False,
    )


def _mark_ocr_incomplete(envelope: DocumentExtraction, ocr_dir: Path) -> ErrorInfo | None:
    """Add the required ``ocr_page`` issue when metadata reports page loss."""
    error = _ocr_incomplete_error(ocr_dir)
    if error is not None:
        _append_stage_issue(envelope, stage="ocr_page", error=error, severity="error")
    return error


def _finalize_extraction_status(
    status: str,
    envelope: DocumentExtraction,
    error_message: Optional[str],
    error_info: Optional[ErrorInfo],
) -> tuple[str, Optional[str], Optional[ErrorInfo]]:
    """Turn required stage failures into partial results without dropping fields."""
    if status == "failed":
        return status, error_message, error_info
    required_error = envelope.extraction_error
    if required_error is None:
        required_error = next(
            (issue.error for issue in envelope.stage_issues if issue.severity == "error"),
            None,
        )
    if required_error is None:
        return status, error_message, error_info
    if status == "ok":
        status = "partial"
    if error_info is None:
        error_info = required_error
    if error_message is None:
        error_message = f"{required_error.code}: {required_error.message}"
    return status, error_message, error_info


def run_full_extraction(document_text: str, ocr_dir: Path) -> DocumentExtraction:
    """跑完整抽取链路（类型路由 + 特化 + 通用后处理 + 多源融合），**不落库**——供评测/复用。

    与 ingest 的抽取段同源：_run_extraction（类型路由+特化）→ 类型专属后处理（签章等）
    → 通用后处理（页码校正）→ 多源融合（保险等）。跳过身份核对（跨文档、需基准库，与单文档
    评测无关）。用生产默认模型。任何后处理/融合异常不中断，尽力返回已得信封。
    """
    envelope = DocumentExtraction()
    # Keep this collector inside run_full_extraction.  Nested collectors are
    # intentionally independent in llm_runtime, so an eval run receives exactly
    # its own production calls rather than an outer caller's partial ledger.
    with collect_llm_calls() as collector:
        try:
            _, _, envelope = _run_extraction(document_text, llm_enabled=True)
            for pp in get_handler(envelope.doc_type).post_processors:
                stage = f"post:{getattr(pp, '__name__', 'processor')}"
                try:
                    pp(envelope, ocr_dir)
                except Exception as e:  # noqa: BLE001 — optional post-processing
                    logger.warning("%s 跳过（异常）: %s", stage, e)
                    _append_stage_issue(
                        envelope,
                        stage=stage,
                        error=classify_exception(e),
                        severity="warning",
                    )
            try:
                correct_evidence_pages(envelope, ocr_dir)
            except Exception as e:  # noqa: BLE001
                logger.warning("page-fix 跳过（异常）: %s", e)
                _append_stage_issue(
                    envelope,
                    stage="page_fix",
                    error=classify_exception(e),
                    severity="warning",
                )
            _maybe_run_vision_fusion(envelope, document_text, ocr_dir, logger.info)
        except Exception as e:  # noqa: BLE001 — eval must expose, not erase, failures
            error = classify_exception(e)
            logger.warning("full extraction failed: %s", e)
            if envelope.extraction_error is None:
                envelope.extraction_error = error
            _append_stage_issue(
                envelope,
                stage="extraction",
                error=error,
                severity="error",
            )
        finally:
            _mark_ocr_incomplete(envelope, ocr_dir)
            _attach_runtime_ledger(envelope, collector.records)
    return envelope


def _vision_fusion_max_pages() -> int:
    """vision 融合看图页数上限（CONTRACT_ARCHIVE_VISION_FUSION_MAX_PAGES，默认 20）。坏值回退。"""
    raw = os.getenv("CONTRACT_ARCHIVE_VISION_FUSION_MAX_PAGES")
    if not raw or not raw.strip():
        return 20
    try:
        val = int(raw.strip())
    except ValueError:
        return 20
    return val if val > 0 else 20


def _fusion_threshold() -> float:
    """融合低置信阈值（CONTRACT_ARCHIVE_FUSION_THRESHOLD，默认 DEFAULT_FUSION_THRESHOLD）。坏值回退。"""
    raw = os.getenv("CONTRACT_ARCHIVE_FUSION_THRESHOLD")
    if not raw or not raw.strip():
        return DEFAULT_FUSION_THRESHOLD
    try:
        return float(raw.strip())
    except ValueError:
        return DEFAULT_FUSION_THRESHOLD


def _select_fusion_images(ocr_dir: Path) -> dict[int, Path]:
    """选 vision 融合要看的页图：优先表格页、其次扫描页（高价值表格/扫描所在），映射到 preview 图。

    据 source.pdf 的 classify_pages 复用页级分流选页；上限 _vision_fusion_max_pages 防超大文档
    烧太多看图调用（截断记日志，不静默）。无 source.pdf / 无 preview 图 / 无表格扫描页 → 空（退文本路）。
    返回 {1-based 页号: 页图路径}。
    """
    source_pdf = ocr_dir.parent / "source.pdf"
    preview_dir = ocr_dir / PREVIEW_DIR
    if not source_pdf.exists():
        return {}
    try:
        routes = classify_pages(source_pdf)
    except Exception as e:  # noqa: BLE001 — 分流失败不能中断入库，退文本路
        logger.warning("[fusion] classify_pages 失败，退文本路: %s", e)
        return {}
    table_pages = [r.page_index for r in routes if r.has_tables]
    other_ocr = [r.page_index for r in routes if r.mode == MODE_OCR and not r.has_tables]
    # 封面页（第1页）几乎总含保单号/投保被保人/日期/保额摘要等保单级高价值字段——无论是否判为
    # 表格/扫描页，优先纳入 vision。大文档表格页多时，table-first + 截断会把封面挤出窗口
    # （实测 doc33：61 个表格页排在前，封面被截掉，保单号只能退文本单源）。
    cover = [0] if routes else []
    ordered = cover + [p for p in table_pages + other_ocr if p != 0]  # 封面 → 表格 → 其余扫描
    cap = _vision_fusion_max_pages()
    if len(ordered) > cap:
        logger.info(
            "[fusion] 候选 vision 页 %s 超上限 %s，截断（CONTRACT_ARCHIVE_VISION_FUSION_MAX_PAGES 可调）",
            len(ordered),
            cap,
        )
    selected = ordered[:cap]
    out: dict[int, Path] = {}
    missing: list[int] = []
    for idx in selected:
        img = preview_dir / f"page_{idx + 1:03d}.png"
        if img.exists():
            out[idx + 1] = img  # render_pdf_to_images 用 1-based page_NNN 命名
        else:
            missing.append(idx)
    # native-text 快路不会主动渲染 preview。按需只补渲选中页（不重渲整份），
    # 让 vision 融合照常工作。
    if missing:
        try:
            for pg in render_pdf_to_images(source_pdf, preview_dir, pages=missing):
                if pg.page_index in selected:
                    out[pg.page_index + 1] = Path(pg.image_path)
        except Exception as e:  # noqa: BLE001 — 补渲失败不能中断入库，退已有页
            logger.warning("[fusion] 按需补渲 preview 失败，跳过缺页: %s", e)
    return out


def _maybe_run_vision_fusion(
    envelope: DocumentExtraction,
    document_text: str,
    ocr_dir: Path,
    log: Callable[[str], None],
    confidence: Optional[ExtractionConfidence] = None,
) -> None:
    """据 doc_type 处理器的 enable_vision_fusion 决定是否跑多源融合（如保险）。

    A(文本)/C(看图) 两路抽高价值概念候选 → 评判 → 写 field_verdicts/fusion_overall_confidence
    sidecar（不回写原字段）。融合产出整体置信后，把它写进 confidence.overall（落 documents
    .overall_confidence 列，复用现列），否则查询/列表仍显融合前的旧启发式分。整体置信低于阈值
    → 调 agent_fallback（本期 no-op 仅标记）。任何异常都不中断入库。
    """
    handler = get_handler(envelope.doc_type)
    if not handler.enable_vision_fusion or not handler.vision_fusion_fields:
        return
    threshold = _fusion_threshold()
    try:
        images = _select_fusion_images(ocr_dir)
        if run_vision_fusion(
            envelope,
            document_text,
            images,
            fields=handler.vision_fusion_fields,
            threshold=threshold,
        ):
            log(
                f"[fusion] 多源融合：{len(envelope.field_verdicts)} 项评判，"
                f"overall={envelope.fusion_overall_confidence}"
            )
    except Exception as e:  # noqa: BLE001 — 融合失败不能中断入库
        log(f"[fusion] 跳过（异常）: {e}")
        _append_stage_issue(
            envelope,
            stage="fusion",
            error=classify_exception(e),
            severity="warning",
        )
        return
    conf = envelope.fusion_overall_confidence
    if conf is None:
        return
    try:
        # 融合分写进可查询的 overall_confidence 列（sidecar 与列保持一致，不留旧启发式分）
        if confidence is not None:
            confidence.overall = conf
        if conf < threshold:
            escalate_low_confidence(envelope, source_pdf=str(ocr_dir.parent / "source.pdf"))
    except Exception as e:  # noqa: BLE001 — optional fusion follow-up cannot fail extraction
        log(f"[fusion] 后处理跳过（异常）: {e}")
        _append_stage_issue(
            envelope,
            stage="fusion",
            error=classify_exception(e),
            severity="warning",
        )


def _new_staging(paths: ArchivePaths, sha: str) -> tuple[Path, Path]:
    """创建独占 staging，并返回 ``(staging, generation_target)``。

    两个目录使用同一个 UUID，方便人工追踪一次运行；绝不复用或清理同名目录。
    """
    # Validate/create tmp/runs before allocating a child.  Do not use
    # ``mkdir(parents=True)`` below: a pre-existing symlink parent must be
    # rejected before it can redirect a staging write.
    paths.ensure()
    for _ in range(8):
        run_id = uuid.uuid4().hex
        staging = paths.staging_dir(sha, run_id)
        try:
            staging.mkdir()
        except FileExistsError:
            continue
        return staging, paths.generation_dir(sha, run_id)
    raise RuntimeError(f"could not allocate unique archive staging directory for {sha[:SHA_SHORT_LEN]}")


def _fsync_file(path: Path) -> None:
    """把一个已关闭的普通文件同步到存储介质。"""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    """同步目录项；Linux archive 运行环境支持 O_DIRECTORY。"""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_staging_tree(staging: Path) -> None:
    """发布前递归 fsync staging 的文件和目录，拒绝不受控 symlink。"""
    directories: list[Path] = []
    for current, dirnames, filenames in os.walk(staging, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.append(current_path)
        for dirname in dirnames:
            if (current_path / dirname).is_symlink():
                raise ValueError(f"refusing to publish symlinked archive directory: {current_path / dirname}")
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                raise ValueError(f"refusing to publish symlinked archive file: {path}")
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"refusing to publish non-regular archive file: {path}")
            _fsync_file(path)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _publish_staging(paths: ArchivePaths, staging: Path, generation: Path) -> Path:
    """把完整 staging 原子发布为新 generation，并持久化目录项。

    文件和 staging 目录会先 fsync；rename 后同时 fsync 源/目标父目录。若任一步失败，
    调用方不会切 SQLite pointer，可能留下可人工核对的 staging/orphan generation。SQLite
    自身负责其事务耐久性；这里不把单纯 rename 误表述为跨设备断电的完整系统保证。
    """
    # Staging and generation are derived by _new_staging.  Check their expected
    # parents explicitly so a future caller cannot make rename write outside a
    # controlled archive directory.
    paths.ensure()
    if staging.parent != paths.runs_dir or generation.parent != paths.documents_dir:
        raise ValueError("staging/generation path is outside controlled archive directories")
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError(f"refusing invalid archive staging directory: {staging}")
    if generation.exists() or generation.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing archive generation: {generation}")
    _sync_staging_tree(staging)
    # Revalidate immediately before rename: ensure() catches ordinary
    # replacement/symlink poisoning without claiming a full race-proof sandbox.
    paths.ensure()
    if (
        generation.parent != paths.documents_dir
        or generation.parent.is_symlink()
        or staging.parent != paths.runs_dir
        or staging.parent.is_symlink()
        or staging.is_symlink()
        or not staging.is_dir()
    ):
        raise ValueError(f"refusing invalid archive documents directory: {generation.parent}")
    if generation.exists() or generation.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing archive generation: {generation}")
    source_parent = staging.parent
    os.rename(staging, generation)
    _fsync_directory(generation.parent)
    if source_parent != generation.parent:
        _fsync_directory(source_parent)
    return generation


def _safe_append_jsonl(path: Path, payload: dict) -> None:
    """提交成功后的审计日志是尽力而为，不能反过来把成功操作报成失败。"""
    try:
        _append_jsonl(path, payload)
    except OSError as exc:
        logger.warning("archive audit log write failed after commit (%s): %s", path, exc)


def _ensure_archived_source(
    paths: ArchivePaths,
    conn: sqlite3.Connection,
    row,
    pdf_path: Path,
) -> Path:
    """保证当前 pointer 指向含有正确 source.pdf 快照的 generation。

    已发布 generation 一律不可就地修补。若历史 generation 缺 source.pdf 或内容哈希不符，
    复制实际 ``row.output_dir`` 到 staging、补上校验过的独立快照、发布新 generation 后只
    切换 pointer。这样不会重建/覆盖旧 ``documents/<sha12>`` 目录，也不会误伤前缀碰撞。
    调用方必须已持有该 SHA 的槽锁。
    """
    current_dir = paths.resolve_document_dir(row)
    source_pdf = current_dir / "source.pdf"
    if source_pdf.is_file() and sha256_of_file(source_pdf) == row.sha256:
        return source_pdf
    if not current_dir.is_dir():
        raise FileNotFoundError(f"archive output missing for id={row.id}: {current_dir}")

    staging, generation = _new_staging(paths, row.sha256)
    shutil.copytree(current_dir, staging, copy_function=shutil.copy2, dirs_exist_ok=True)
    link_or_copy(pdf_path, staging / "source.pdf", expected_sha256=row.sha256)
    final_dir = _publish_staging(paths, staging, generation)
    update_output_dir(conn, row.id, str(final_dir))
    return final_dir / "source.pdf"


# ---------- 入口 ----------


def ingest_pdf(
    pdf_path: Path,
    paths: ArchivePaths,
    conn: sqlite3.Connection,
    *,
    reingest: bool = False,
    llm_enabled: bool = True,
    pipeline: Optional[OCRPipeline] = None,
) -> IngestResult:
    """
    单 PDF 入库。

    :param pdf_path: PDF 绝对/相对路径
    :param paths: 档案库根路径对象
    :param conn: 已打开 + 已 migrate 的 sqlite3 连接
    :param reingest: True 时即使 sha256 已存在也强制重跑
    :param llm_enabled: False 时只跑 rule 抽取
    :param pipeline: 可注入的 OCRPipeline 实例（复用模型加载，批量场景必传）
    """
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    paths.ensure()
    logger.info("hashing %s", pdf_path.name)
    sha = sha256_of_file(pdf_path)
    logger.info("sha256=%s", sha[:SHA_SHORT_LEN])

    # 从查重、快照、发布到 DAO 提交始终在同一个物理槽锁内。必须在锁后重新查 DB：
    # 等锁期间另一个进程可能已经完成了同一份 PDF 的入库。也必须重算源文件：
    # 若输入在初始 hash 和获得锁之间被原地替换，不能拿旧 SHA 的锁去 skip/改旧记录。
    with paths.slot_lock(sha):
        locked_sha = sha256_of_file(pdf_path)
        if locked_sha != sha:
            raise ValueError(
                "source changed while waiting for archive slot lock; please retry ingest"
            )
        return _ingest_pdf_locked(
            pdf_path, paths, conn, sha=sha, reingest=reingest,
            llm_enabled=llm_enabled, pipeline=pipeline,
        )


def _ingest_pdf_locked(
    pdf_path: Path,
    paths: ArchivePaths,
    conn: sqlite3.Connection,
    *,
    sha: str,
    reingest: bool,
    llm_enabled: bool,
    pipeline: Optional[OCRPipeline],
) -> IngestResult:
    """Locked ingest boundary with one ledger spanning OCR and extraction."""
    with collect_llm_calls() as collector:
        return _ingest_pdf_locked_with_ledger(
            pdf_path,
            paths,
            conn,
            sha=sha,
            reingest=reingest,
            llm_enabled=llm_enabled,
            pipeline=pipeline,
            collector=collector,
        )


def _ingest_pdf_locked_with_ledger(
    pdf_path: Path,
    paths: ArchivePaths,
    conn: sqlite3.Connection,
    *,
    sha: str,
    reingest: bool,
    llm_enabled: bool,
    pipeline: Optional[OCRPipeline],
    collector,
) -> IngestResult:
    """``ingest_pdf`` work under the slot lock and active runtime collector."""
    existing_id = find_by_sha(conn, sha)
    existing = get_document(conn, existing_id) if existing_id else None
    if existing_id and existing is None:
        raise RuntimeError(f"document id={existing_id} disappeared during ingest")
    if existing_id and not reingest:
        prev_status = existing.status if existing else None
        if prev_status == "failed":
            # 上次失败不算"已入库"——重跑就是想重试，自动按 reingest 处理，
            # 不要 skip 后甩给用户一句"加 --reingest"（UX：见 id=6 排查）。
            logger.info("sha=%s 上次 ingest 失败，自动重试", sha[:SHA_SHORT_LEN])
            reingest = True
        else:
            _ensure_archived_source(paths, conn, existing, pdf_path)
            if prev_status == "partial":
                hint = f"（OCR 已完成、抽取未完成；用 `extract {existing_id}` 只重跑抽取，省去 OCR）"
            else:
                hint = "（已成功入库）"
            return IngestResult(
                pdf_path=pdf_path,
                sha256=sha,
                status="skipped",
                doc_id=existing_id,
                skipped_reason=f"sha256 已在档案库{hint}；要强制重跑整条流程加 --reingest",
            )

    # 每次运行独占 staging；绝不删除同 SHA 前缀下的旧 staging/generation。
    tmp_doc_dir, generation_dir = _new_staging(paths, sha)
    ocr_dir = tmp_doc_dir / "ocr"

    # 单合同 stderr 日志（plain text），与档案库总 jsonl 互补
    log_path = tmp_doc_dir / "ingest.log"
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write(f"# ingest started at {_utc_now()}\n# pdf={pdf_path}\n")
    staged_source = tmp_doc_dir / "source.pdf"
    try:
        link_strategy = link_or_copy(pdf_path, staged_source, expected_sha256=sha)
    except Exception:
        log_handle.close()
        raise
    log_handle.write(f"[source.pdf] {link_strategy}ed from {pdf_path}; sha256 verified\n")

    ocr_duration: Optional[float] = None
    llm_duration: Optional[float] = None
    extraction: Optional[ContractExtraction] = None
    confidence: Optional[ExtractionConfidence] = None
    envelope: Optional[DocumentExtraction] = None
    error_message: Optional[str] = None
    error_info: Optional[ErrorInfo] = None
    status = "ok"

    try:
        # ---- 1. OCR 解析 ----
        pl = pipeline or OCRPipeline()
        t0 = time.perf_counter()
        try:
            pl.run(staged_source, ocr_dir)
            ocr_duration = time.perf_counter() - t0
            log_handle.write(f"\n[ocr] OK in {ocr_duration:.2f}s\n")
        except Exception as e:
            ocr_duration = time.perf_counter() - t0
            status = "failed"
            error_message = f"ocr: {e}"
            log_handle.write(f"\n[ocr] FAILED: {error_message}\n")
            log_handle.write(traceback.format_exc())
            return _commit_failed(
                conn=conn,
                paths=paths,
                pdf_path=pdf_path,
                sha=sha,
                tmp_doc_dir=tmp_doc_dir,
                log_handle=log_handle,
                existing=existing,
                generation_dir=generation_dir,
                ocr_duration=ocr_duration,
                error_message=error_message,
                error=ocr_failed(str(e)),
                ledger_records=collector.records,
            )

        # ---- 2. 抽取（基于 OCR 产物的 raw_text.txt 优先） ----
        document_text = load_document_text(ocr_dir)
        if not document_text:
            log_handle.write("\n[extract] WARNING: no text found in OCR output\n")
        t1 = time.perf_counter()
        try:
            extraction, confidence, envelope = _run_extraction(
                document_text, llm_enabled=llm_enabled
            )
            llm_duration = time.perf_counter() - t1
            log_handle.write(
                f"[extract] OK in {llm_duration:.2f}s (doc_type={envelope.doc_type})\n"
            )
            # 抽取空跑护栏：开了 LLM 却啥都没抽到（最常见是缺 DASHSCOPE_API_KEY——
            # 全局工具需 shell export，不读项目 .env），别静默标 ok 误导用户。
            if (
                llm_enabled
                and not extraction.contract_name
                and not envelope.title
                and not envelope.fields
                and not envelope.amounts
                and not envelope.seals
            ):
                status = "partial"
                # 结构化 error 优先用 envelope 透上来的（精确区分缺 key / 限流 / 网络），
                # 缺失才兜底 EXTRACT_EMPTY；error_message 仍是人类可读提示，不变。
                error_info = envelope.extraction_error or extract_empty("LLM 抽取为空")
                error_message = (
                    "LLM 抽取为空——通常是缺 DASHSCOPE_API_KEY（全局工具需在 shell "
                    "export，不读项目 .env）或 LLM 调用失败；补好后用 `extract <id>` 重抽"
                )
                log_handle.write(f"\n[extract] WARNING: {error_message}\n")
        except Exception as e:
            llm_duration = time.perf_counter() - t1
            status = "partial"
            error_message = f"extract: {e}"
            error_info = classify_exception(e)
            extraction = ContractExtraction()
            confidence = ExtractionConfidence()
            envelope = DocumentExtraction(extraction_error=error_info)
            _append_stage_issue(
                envelope,
                stage="extraction",
                error=error_info,
                severity="error",
            )
            log_handle.write(f"\n[extract] FAILED (status=partial): {error_message}\n")
            log_handle.write(traceback.format_exc())

        # ---- 2.5 类型专属后处理（据 doc_type 查处理器；合同=看落款页图重判签章，其他类型可能无）----
        if status != "failed" and llm_enabled:
            for pp in get_handler(envelope.doc_type).post_processors:
                stage = f"post:{getattr(pp, '__name__', 'processor')}"
                try:
                    if pp(envelope, ocr_dir):
                        log_handle.write(f"[post:{pp.__name__}] 完成\n")
                except Exception as e:  # noqa: BLE001 — 专属后处理失败不能中断入库
                    log_handle.write(f"[post:{pp.__name__}] 跳过（异常）: {e}\n")
                    _append_stage_issue(
                        envelope,
                        stage=stage,
                        error=classify_exception(e),
                        severity="warning",
                    )

            # ---- 2.6 出处页码校正（通用，类型无关）：用 layout 页码覆盖 LLM 猜的页码 ----
            try:
                if correct_evidence_pages(envelope, ocr_dir):
                    log_handle.write("[page-fix] 出处页码已据 layout 校正\n")
            except Exception as e:  # noqa: BLE001 — 页码校正失败不能中断入库
                log_handle.write(f"[page-fix] 跳过（异常）: {e}\n")
                _append_stage_issue(
                    envelope,
                    stage="page_fix",
                    error=classify_exception(e),
                    severity="warning",
                )

            # ---- 2.7 身份基本信息核对：首见入库、再见校对（known_parties 基准库）----
            # 把抽到的 person_identities（精确到人的身份证/电话/银行账号/开户行…）与
            # 跨文档基准库比对：首次见到的录入为基准，再见到不一致即报 identity 缺陷。
            try:
                registry = PartyRegistry.load(paths.known_parties_path)
                id_issues = registry.reconcile(envelope.person_identities, sha)
                if registry.dirty:
                    registry.save()
                envelope.identity_issues = id_issues
                if id_issues:
                    log_handle.write(f"[identity] 身份核对：{len(id_issues)} 项与基准不一致\n")
                elif envelope.person_identities:
                    log_handle.write("[identity] 身份核对：与基准一致（或首见已入库）\n")
            except Exception as e:  # noqa: BLE001 — 核对失败不能中断入库
                log_handle.write(f"[identity] 跳过（异常）: {e}\n")

            # ---- 2.8 多源融合（仅 enable_vision_fusion 的类型，如保险）：A文本/C看图评判 → sidecar ----
            _maybe_run_vision_fusion(
                envelope, document_text, ocr_dir,
                lambda m: log_handle.write(m + "\n"), confidence,
            )

        # The ledger must cover OCR plus both document/contract extraction calls
        # on a new ingest.  Completeness markers are persisted by OCR, not inferred
        # from whether later LLM extraction found a useful title.
        _mark_ocr_incomplete(envelope, ocr_dir)
        _attach_runtime_ledger(envelope, collector.records)
        status, error_message, error_info = _finalize_extraction_status(
            status, envelope, error_message, error_info
        )

        # ---- 3. extracted.json 落盘（写通用信封；即使 partial 也写空对象，便于后续 extract 复跑） ----
        (tmp_doc_dir / FILE_EXTRACTION).write_text(
            envelope.model_dump_json(indent=2), encoding="utf-8"
        )
        (tmp_doc_dir / FILE_EXTRACTION_CONF).write_text(
            confidence.model_dump_json(indent=2), encoding="utf-8"
        )

        # ---- 4. 发布新 immutable generation：sync staging → atomic rename → sync 父目录 ----
        # 先关闭日志文件，让 _publish_staging 能递归 fsync 全部文件；之后绝不修改这个 generation。
        log_handle.flush()
        os.fsync(log_handle.fileno())
        log_handle.close()
        log_handle = None
        final_doc_dir = _publish_staging(paths, tmp_doc_dir, generation_dir)

        # ---- 5. DB 写入 / 切 pointer ----
        if existing_id:
            replace_document(
                conn,
                existing_id,
                source_path=str(pdf_path),
                output_dir=str(final_doc_dir),
                status=status,
                ocr_duration_s=ocr_duration,
                llm_duration_s=llm_duration,
                error_message=error_message,
                extraction=extraction,
                confidence=confidence,
                envelope=envelope,
            )
            doc_id = existing_id
        else:
            doc_id = insert_document(
                conn,
                sha256=sha,
                source_path=str(pdf_path),
                output_dir=str(final_doc_dir),
                status=status,
                ocr_duration_s=ocr_duration,
                llm_duration_s=llm_duration,
                error_message=error_message,
                extraction=extraction,
                confidence=confidence,
                envelope=envelope,
            )
            # 所有本工具进程都持有同一槽锁时不应发生；若外部绕过锁写入，宁可留下
            # 可审计 orphan generation，也不能把未知行当成自己的成功结果。
            if doc_id is None:
                raise RuntimeError(f"sha256 inserted outside archive slot lock: {sha}")

        _safe_append_jsonl(
            paths.ingest_log,
            {
                "ts": _utc_now(),
                "pdf": str(pdf_path),
                "sha": sha,
                "doc_id": doc_id,
                "status": status,
                "ocr_s": ocr_duration,
                "llm_s": llm_duration,
                "error": error_message,
            },
        )

        return IngestResult(
            pdf_path=pdf_path,
            sha256=sha,
            status=status,
            doc_id=doc_id,
            ocr_duration_s=ocr_duration,
            llm_duration_s=llm_duration,
            error_message=error_message,
            error=error_info,
        )
    finally:
        try:
            if log_handle is not None:
                log_handle.close()
        except Exception:
            pass


def _commit_failed(
    *,
    conn: sqlite3.Connection,
    paths: ArchivePaths,
    pdf_path: Path,
    sha: str,
    tmp_doc_dir: Path,
    log_handle,
    generation_dir: Path,
    existing,
    ocr_duration: Optional[float],
    error_message: str,
    error: Optional[ErrorInfo] = None,
    ledger_records: Optional[list[dict[str, object]]] = None,
) -> IngestResult:
    """
    OCR 失败的收尾。

    已有 ok/partial 档案的本次 reingest 失败不能碰旧 DB 行或旧 generation：只返回
    本次失败，并把诊断日志复制到独立位置。新文档（以及此前已是 failed 的文档）可以
    发布一个 failed generation，再按既有失败状态语义写库。
    """
    try:
        log_handle.flush()
        os.fsync(log_handle.fileno())
    except OSError as exc:
        logger.warning("could not sync failed ingest log before preserving it: %s", exc)
    finally:
        log_handle.close()

    # A fatal OCR exception can happen after an observed page request recorded
    # an error.  Persist that safe runtime ledger for a new/previously-failed
    # row; it carries error *categories* only, never the exception text.  An
    # existing ok/partial row remains untouched by this failed reingest path.
    failure_envelope = DocumentExtraction()
    _attach_runtime_ledger(failure_envelope, ledger_records or [])
    try:
        # For a failed reingest of an already-good row, staging is intentionally
        # retained for forensics.  Keep the failed attempt's safe call ledger
        # there rather than overwriting the current DB envelope.
        (tmp_doc_dir / FILE_EXTRACTION).write_text(
            failure_envelope.model_dump_json(indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("could not preserve failed OCR call ledger in staging: %s", exc)

    if existing is not None and existing.status in {"ok", "partial"}:
        paths.failed_logs_dir.mkdir(parents=True, exist_ok=True)
        failed_log = paths.failed_logs_dir / f"{tmp_doc_dir.name}.log"
        try:
            shutil.copy2(tmp_doc_dir / "ingest.log", failed_log)
        except OSError as exc:
            logger.warning("could not preserve failed reingest log at %s: %s", failed_log, exc)
            failed_log = None
        # Keep staging for forensics. It is intentionally neither published nor cleaned up.
        doc_id = existing.id
    else:
        final_doc_dir = _publish_staging(paths, tmp_doc_dir, generation_dir)
        failed_log = final_doc_dir / "ingest.log"
        if existing is not None:
            replace_document(
                conn,
                existing.id,
                source_path=str(pdf_path),
                output_dir=str(final_doc_dir),
                status="failed",
                ocr_duration_s=ocr_duration,
                llm_duration_s=None,
                error_message=error_message,
                extraction=ContractExtraction(),
                confidence=ExtractionConfidence(),
                envelope=failure_envelope,
            )
            doc_id = existing.id
        else:
            doc_id = insert_document(
                conn,
                sha256=sha,
                source_path=str(pdf_path),
                output_dir=str(final_doc_dir),
                status="failed",
                ocr_duration_s=ocr_duration,
                llm_duration_s=None,
                error_message=error_message,
                extraction=ContractExtraction(),
                confidence=ExtractionConfidence(),
                envelope=failure_envelope,
            )
            if doc_id is None:
                raise RuntimeError(f"sha256 inserted outside archive slot lock: {sha}")

    _safe_append_jsonl(
        paths.ingest_log,
        {
            "ts": _utc_now(),
            "pdf": str(pdf_path),
            "sha": sha,
            "doc_id": doc_id,
            "status": "failed",
            "ocr_s": ocr_duration,
            "llm_s": None,
            "error": error_message,
            "log_path": str(failed_log) if failed_log else None,
        },
    )

    return IngestResult(
        pdf_path=pdf_path,
        sha256=sha,
        status="failed",
        doc_id=doc_id,
        ocr_duration_s=ocr_duration,
        error_message=error_message,
        error=error,
    )


# ---------- 复跑抽取（partial 状态修复，不重跑 OCR） ----------


def re_extract(
    doc_id: int,
    paths: ArchivePaths,
    conn: sqlite3.Connection,
    *,
    llm_enabled: bool = True,
) -> IngestResult:
    """
    基于已有 OCR 产物重跑抽取。用于 partial 状态修复或调 prompt 后批量再抽取。
    不动 OCR 产物，不动 sha256/source_path/ingested_at。
    """
    initial = get_document(conn, doc_id)
    if not initial:
        raise ValueError(f"document id={doc_id} not found")
    paths.ensure()
    # 和 ingest 一样，锁后重读 pointer，避免 re_extract 从一代复制、另一代被 reingest 切换。
    with paths.slot_lock(initial.sha256):
        return _re_extract_locked(doc_id, paths, conn, llm_enabled=llm_enabled)


def _re_extract_locked(
    doc_id: int,
    paths: ArchivePaths,
    conn: sqlite3.Connection,
    *,
    llm_enabled: bool,
) -> IngestResult:
    """Locked re-extract boundary with a ledger for this extraction run only."""
    with collect_llm_calls() as collector:
        return _re_extract_locked_with_ledger(
            doc_id,
            paths,
            conn,
            llm_enabled=llm_enabled,
            collector=collector,
        )


def _re_extract_locked_with_ledger(
    doc_id: int,
    paths: ArchivePaths,
    conn: sqlite3.Connection,
    *,
    llm_enabled: bool,
    collector,
) -> IngestResult:
    """Copy the current generation and write a new one under the active ledger."""
    doc = get_document(conn, doc_id)
    if not doc:
        raise ValueError(f"document id={doc_id} not found")
    current_dir = paths.resolve_document_dir(doc)
    current_source = current_dir / "source.pdf"
    if not current_source.is_file():
        raise FileNotFoundError(f"archived source missing for id={doc_id}: {current_source}")
    if sha256_of_file(current_source) != doc.sha256:
        raise ValueError(f"archived source hash mismatch for id={doc_id}: {current_source}")

    staging, generation_dir = _new_staging(paths, doc.sha256)
    shutil.copytree(current_dir, staging, copy_function=shutil.copy2, dirs_exist_ok=True)
    ocr_dir = staging / "ocr"
    if not ocr_dir.exists():
        raise FileNotFoundError(
            f"OCR output missing for id={doc_id}: {ocr_dir}"
        )

    document_text = load_document_text(ocr_dir)
    t0 = time.perf_counter()
    error_message: Optional[str] = None
    error_info: Optional[ErrorInfo] = None
    status = "ok"
    envelope = DocumentExtraction()
    try:
        extraction, confidence, envelope = _run_extraction(
            document_text, llm_enabled=llm_enabled
        )
        # 空抽取护栏（与 ingest_pdf 对齐）：开了 LLM 却啥都没抽到（最常见缺 key），
        # 别静默标 ok 误导用户/agent——据 envelope.extraction_error 给结构化信号。
        if (
            llm_enabled
            and not extraction.contract_name
            and not envelope.title
            and not envelope.fields
            and not envelope.amounts
            and not envelope.seals
        ):
            status = "partial"
            error_info = envelope.extraction_error or extract_empty("LLM 抽取为空")
            error_message = (
                "LLM 抽取为空——通常是缺 DASHSCOPE_API_KEY 或 LLM 调用失败；"
                "补好后重跑 `extract <id>`"
            )
    except Exception as e:
        status = "partial"
        error_message = f"extract: {e}"
        error_info = classify_exception(e)
        extraction = ContractExtraction()
        confidence = ExtractionConfidence()
        envelope = DocumentExtraction(extraction_error=error_info)
        _append_stage_issue(
            envelope,
            stage="extraction",
            error=error_info,
            severity="error",
        )
    llm_duration = time.perf_counter() - t0

    # 类型专属后处理（据 doc_type 查处理器；合同=看落款页图重判签章）。
    if llm_enabled and status == "ok":
        for pp in get_handler(envelope.doc_type).post_processors:
            stage = f"post:{getattr(pp, '__name__', 'processor')}"
            try:
                pp(envelope, ocr_dir)
            except Exception as e:  # noqa: BLE001 — 专属后处理失败不能中断重抽
                logger.warning("post:%s 跳过（异常）: %s", pp.__name__, e)
                _append_stage_issue(
                    envelope,
                    stage=stage,
                    error=classify_exception(e),
                    severity="warning",
                )
        try:
            correct_evidence_pages(envelope, ocr_dir)
        except Exception as e:  # noqa: BLE001 — 页码校正失败不能中断重抽
            logger.warning("page-fix 跳过（异常）: %s", e)
            _append_stage_issue(
                envelope,
                stage="page_fix",
                error=classify_exception(e),
                severity="warning",
            )

        # 身份基本信息核对：首见入库、再见校对。与 ingest 的 2.7 一致——
        # 否则重抽会把已核对出的 identity_issues 清空，造成 ingest/extract 行为分叉。
        try:
            registry = PartyRegistry.load(paths.known_parties_path)
            id_issues = registry.reconcile(envelope.person_identities, doc.sha256)
            if registry.dirty:
                registry.save()
            envelope.identity_issues = id_issues
        except Exception as e:  # noqa: BLE001 — 核对失败不能中断重抽
            logger.warning("identity 跳过（异常）: %s", e)

        # 多源融合（仅 enable_vision_fusion 的类型，如保险）：A文本/C看图评判 → sidecar。
        _maybe_run_vision_fusion(envelope, document_text, ocr_dir, logger.info, confidence)

    # Re-extraction deliberately does not re-run OCR, so this collector contains
    # only calls made by this extraction/post-processing pass.  The copied OCR
    # metadata still has to fail closed if it declares missing/truncated pages.
    _mark_ocr_incomplete(envelope, ocr_dir)
    _attach_runtime_ledger(envelope, collector.records)
    status, error_message, error_info = _finalize_extraction_status(
        status, envelope, error_message, error_info
    )

    # 只写 staging 的新 extracted.json；当前 generation 始终不被就地修改。
    (staging / FILE_EXTRACTION).write_text(
        envelope.model_dump_json(indent=2), encoding="utf-8"
    )
    (staging / FILE_EXTRACTION_CONF).write_text(
        confidence.model_dump_json(indent=2), encoding="utf-8"
    )

    # 先发布文件，再把 DB pointer 和抽取字段在同一事务里切换。DB 失败时旧 generation
    # 及旧 metadata 保持不变，新 generation 留作可审计 orphan。
    final_dir = _publish_staging(paths, staging, generation_dir)

    update_extraction(
        conn,
        doc_id,
        status=status,
        llm_duration_s=llm_duration,
        error_message=error_message,
        extraction=extraction,
        confidence=confidence,
        envelope=envelope,
        output_dir=str(final_dir),
    )

    _safe_append_jsonl(
        paths.ingest_log,
        {
            "ts": _utc_now(),
            "op": "re_extract",
            "doc_id": doc_id,
            "sha": doc.sha256,
            "status": status,
            "llm_s": llm_duration,
            "error": error_message,
        },
    )

    return IngestResult(
        pdf_path=Path(doc.source_path),
        sha256=doc.sha256,
        status=status,
        doc_id=doc_id,
        llm_duration_s=llm_duration,
        error_message=error_message,
        error=error_info or envelope.extraction_error,
    )


# ---------- 工具 ----------


def load_document_text(ocr_dir: Path) -> str:
    """优先 raw_text.txt（已清洗），fallback markdown.md。"""
    raw = ocr_dir / FILE_RAW_TEXT
    md = ocr_dir / FILE_MARKDOWN
    if raw.exists():
        return raw.read_text(encoding="utf-8")
    if md.exists():
        return md.read_text(encoding="utf-8")
    return ""


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- 目录递归收集 PDF ----------


def discover_pdfs(path: Path) -> list[Path]:
    """传入文件返回单元素列表；传入目录递归找 *.pdf，跳过隐藏文件。"""
    path = path.resolve()
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"not a PDF: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    pdfs = sorted(
        p for p in path.rglob("*.pdf")
        if not any(part.startswith(".") for part in p.relative_to(path).parts)
    )
    return pdfs
