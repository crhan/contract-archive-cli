"""
单 PDF 入库流水线。

流程（每个 PDF 一次调用）：
  1) 流式 SHA256
  2) 查 documents.sha256 → 命中 + 非 reingest 直接 skip
  3) 在 tmp/<sha-short>/ 先留 source.pdf，再跑 OCR pipeline + 抽取
  4) 全成功后 os.rename(tmp → documents/<sha-short>/) 是事务边界
  5) DB 写入 documents + risk_clauses（单事务，由 repository 保证）
  6) 追加一行 ingest.jsonl 总日志
  7) 失败时：仍保留 documents/<sha-short>/source.pdf + ingest.log，记 status=failed

状态语义：
  - ok       OCR + 抽取都成功
  - partial  OCR 成功但 LLM 失败 → markdown 可用，可后续 extract 命令重跑
  - failed   OCR 失败 → 没有 OCR 产物，但 source.pdf 留档可查
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..errors import ErrorInfo, classify_exception, extract_empty, mineru_failed
from ..extraction import extract_contract, extract_document
from ..extraction.agent_fallback import escalate_low_confidence
from ..extraction.doc_type_handlers import get_handler
from ..extraction.evidence_page_fix import correct_evidence_pages
from ..extraction.fusion import DEFAULT_FUSION_THRESHOLD, run_vision_fusion
from ..pipelines import MinerUPipeline
from ..schemas import (
    FILE_EXTRACTION,
    FILE_EXTRACTION_CONF,
    FILE_MARKDOWN,
    FILE_RAW_TEXT,
    PREVIEW_DIR,
    ContractExtraction,
    DocumentExtraction,
    ExtractionConfidence,
)
from ..utils import classify_pages
from ..utils.page_router import MODE_OCR
from .party_registry import PartyRegistry
from .paths import ArchivePaths, SHA_SHORT_LEN, link_or_copy, safe_rmtree, sha256_of_file
from .repository import (
    contract_to_envelope,
    find_by_sha,
    get_document,
    insert_document,
    replace_document,
    update_extraction,
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
    mineru_duration_s: Optional[float] = None
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
        # 即 --no-llm 下抽取字段留空（仅 MinerU 产物入库），可后续 `extract <id>` 补抽。
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


def run_full_extraction(document_text: str, mineru_dir: Path) -> DocumentExtraction:
    """跑完整抽取链路（类型路由 + 特化 + 通用后处理 + 多源融合），**不落库**——供评测/复用。

    与 ingest 的抽取段同源：_run_extraction（类型路由+特化）→ 类型专属后处理（签章等）
    → 通用后处理（页码校正）→ 多源融合（保险等）。跳过身份核对（跨文档、需基准库，与单文档
    评测无关）。用生产默认模型。任何后处理/融合异常不中断，尽力返回已得信封。
    """
    _, _, envelope = _run_extraction(document_text, llm_enabled=True)
    for pp in get_handler(envelope.doc_type).post_processors:
        try:
            pp(envelope, mineru_dir)
        except Exception as e:  # noqa: BLE001 — 专属后处理失败不影响其余
            logger.warning("post:%s 跳过（异常）: %s", getattr(pp, "__name__", pp), e)
    try:
        correct_evidence_pages(envelope, mineru_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("page-fix 跳过（异常）: %s", e)
    _maybe_run_vision_fusion(envelope, document_text, mineru_dir, logger.info)
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


def _select_fusion_images(mineru_dir: Path) -> dict[int, Path]:
    """选 vision 融合要看的页图：优先表格页、其次扫描页（高价值表格/扫描所在），映射到 preview 图。

    据 source.pdf 的 classify_pages 复用页级分流选页；上限 _vision_fusion_max_pages 防超大文档
    烧太多看图调用（截断记日志，不静默）。无 source.pdf / 无 preview 图 / 无表格扫描页 → 空（退文本路）。
    返回 {1-based 页号: 页图路径}。
    """
    source_pdf = mineru_dir.parent / "source.pdf"
    preview_dir = mineru_dir / PREVIEW_DIR
    if not source_pdf.exists() or not preview_dir.exists():
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
    out: dict[int, Path] = {}
    for idx in ordered[:cap]:
        img = preview_dir / f"page_{idx + 1:03d}.png"
        if img.exists():
            out[idx + 1] = img  # render_pdf_to_images 用 1-based page_NNN 命名
    return out


def _maybe_run_vision_fusion(
    envelope: DocumentExtraction,
    document_text: str,
    mineru_dir: Path,
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
        images = _select_fusion_images(mineru_dir)
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
        return
    conf = envelope.fusion_overall_confidence
    if conf is None:
        return
    # 融合分写进可查询的 overall_confidence 列（sidecar 与列保持一致，不留旧启发式分）
    if confidence is not None:
        confidence.overall = conf
    if conf < threshold:
        escalate_low_confidence(envelope, source_pdf=str(mineru_dir.parent / "source.pdf"))


def _ensure_archived_source(paths: ArchivePaths, sha: str, pdf_path: Path) -> Path:
    """
    幂等保证 archive 可控目录内有 source.pdf。

    重复 ingest 命中 skip 时也走这里：如果历史产物被误删，当前这次 ingest 仍会
    把源 PDF 补回 documents/<sha-short>/source.pdf。
    """
    source_pdf = paths.doc_dir(sha) / "source.pdf"
    if not source_pdf.exists():
        link_or_copy(pdf_path, source_pdf)
    return source_pdf


# ---------- 入口 ----------


def ingest_pdf(
    pdf_path: Path,
    paths: ArchivePaths,
    conn: sqlite3.Connection,
    *,
    reingest: bool = False,
    llm_enabled: bool = True,
    pipeline: Optional[MinerUPipeline] = None,
) -> IngestResult:
    """
    单 PDF 入库。

    :param pdf_path: PDF 绝对/相对路径
    :param paths: 档案库根路径对象
    :param conn: 已打开 + 已 migrate 的 sqlite3 连接
    :param reingest: True 时即使 sha256 已存在也强制重跑
    :param llm_enabled: False 时只跑 rule 抽取
    :param pipeline: 可注入的 MinerUPipeline 实例（复用模型加载，批量场景必传）
    """
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    paths.ensure()

    logger.info("hashing %s", pdf_path.name)
    sha = sha256_of_file(pdf_path)
    sha_short = sha[:SHA_SHORT_LEN]
    logger.info("sha256=%s", sha_short)

    existing_id = find_by_sha(conn, sha)
    if existing_id and not reingest:
        prev = get_document(conn, existing_id)
        prev_status = prev.status if prev else None
        if prev_status == "failed":
            # 上次失败不算"已入库"——重跑就是想重试，自动按 reingest 处理，
            # 不要 skip 后甩给用户一句"加 --reingest"（UX：见 id=6 排查）。
            logger.info("sha=%s 上次 ingest 失败，自动重试", sha_short)
            reingest = True
        else:
            _ensure_archived_source(paths, sha, pdf_path)
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

    # 在 tmp 跑，全成功后 rename 到 documents/<sha-short>/
    tmp_doc_dir = paths.tmp_dir / sha_short
    safe_rmtree(tmp_doc_dir)
    tmp_doc_dir.mkdir(parents=True, exist_ok=True)
    mineru_dir = tmp_doc_dir / "mineru"

    # 单合同 stderr 日志（plain text），与档案库总 jsonl 互补
    log_path = tmp_doc_dir / "ingest.log"
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write(f"# ingest started at {_utc_now()}\n# pdf={pdf_path}\n")
    link_strategy = link_or_copy(pdf_path, tmp_doc_dir / "source.pdf")
    log_handle.write(f"[source.pdf] {link_strategy}ed from {pdf_path}\n")

    mineru_duration: Optional[float] = None
    llm_duration: Optional[float] = None
    extraction: Optional[ContractExtraction] = None
    confidence: Optional[ExtractionConfidence] = None
    envelope: Optional[DocumentExtraction] = None
    error_message: Optional[str] = None
    error_info: Optional[ErrorInfo] = None
    status = "ok"

    try:
        # ---- 1. OCR 解析 ----
        pl = pipeline or MinerUPipeline(allow_vl_fallback=llm_enabled)
        t0 = time.perf_counter()
        try:
            pl.run(pdf_path, mineru_dir)
            mineru_duration = time.perf_counter() - t0
            log_handle.write(f"\n[ocr] OK in {mineru_duration:.2f}s\n")
        except Exception as e:
            mineru_duration = time.perf_counter() - t0
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
                existing_id=existing_id,
                mineru_duration=mineru_duration,
                error_message=error_message,
                error=mineru_failed(str(e)),
            )

        # ---- 2. 抽取（基于 mineru 产物的 raw_text.txt 优先） ----
        document_text = load_document_text(mineru_dir)
        if not document_text:
            log_handle.write("\n[extract] WARNING: no text found in mineru output\n")
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
            envelope = DocumentExtraction()
            log_handle.write(f"\n[extract] FAILED (status=partial): {error_message}\n")
            log_handle.write(traceback.format_exc())

        # ---- 2.5 类型专属后处理（据 doc_type 查处理器；合同=看落款页图重判签章，其他类型可能无）----
        if status != "failed" and llm_enabled:
            for pp in get_handler(envelope.doc_type).post_processors:
                try:
                    if pp(envelope, mineru_dir):
                        log_handle.write(f"[post:{pp.__name__}] 完成\n")
                except Exception as e:  # noqa: BLE001 — 专属后处理失败不能中断入库
                    log_handle.write(f"[post:{pp.__name__}] 跳过（异常）: {e}\n")

            # ---- 2.6 出处页码校正（通用，类型无关）：用 content_list 的 page_idx 覆盖 LLM 猜的页码 ----
            try:
                if correct_evidence_pages(envelope, mineru_dir):
                    log_handle.write("[page-fix] 出处页码已据 content_list 校正\n")
            except Exception as e:  # noqa: BLE001 — 页码校正失败不能中断入库
                log_handle.write(f"[page-fix] 跳过（异常）: {e}\n")

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
                envelope, document_text, mineru_dir,
                lambda m: log_handle.write(m + "\n"), confidence,
            )

        # ---- 3. extracted.json 落盘（写通用信封；即使 partial 也写空对象，便于后续 extract 复跑） ----
        (tmp_doc_dir / FILE_EXTRACTION).write_text(
            envelope.model_dump_json(indent=2), encoding="utf-8"
        )
        (tmp_doc_dir / FILE_EXTRACTION_CONF).write_text(
            confidence.model_dump_json(indent=2), encoding="utf-8"
        )

        # ---- 4. 事务边界：rename tmp → documents/<sha-short>/ ----
        final_doc_dir = paths.doc_dir(sha)
        safe_rmtree(final_doc_dir)
        final_doc_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_doc_dir.rename(final_doc_dir)
        # rename 之后 log_handle 仍然有效（文件描述符不依赖路径），但 path 已变
        # 为了后续追加，把 handle 关掉再开新的
        log_handle.close()
        log_handle = (final_doc_dir / "ingest.log").open("a", encoding="utf-8")

        # ---- 5. DB 写入 ----
        if existing_id:
            replace_document(
                conn,
                existing_id,
                source_path=str(pdf_path),
                output_dir=str(final_doc_dir),
                status=status,
                mineru_duration_s=mineru_duration,
                llm_duration_s=llm_duration,
                error_message=error_message,
                extraction=extraction,
                confidence=confidence,
                envelope=envelope,
            )
            doc_id = existing_id
            log_handle.write(f"\n[db] replaced id={doc_id} status={status}\n")
        else:
            doc_id = insert_document(
                conn,
                sha256=sha,
                source_path=str(pdf_path),
                output_dir=str(final_doc_dir),
                status=status,
                mineru_duration_s=mineru_duration,
                llm_duration_s=llm_duration,
                error_message=error_message,
                extraction=extraction,
                confidence=confidence,
                envelope=envelope,
            )
            # 极端竞态：sha 在我们 hash 完到 insert 之间被别的 worker 写入
            if doc_id is None:
                doc_id = find_by_sha(conn, sha)
                log_handle.write(
                    f"\n[db] race: sha already inserted by peer, reusing id={doc_id}\n"
                )
            else:
                log_handle.write(f"\n[db] inserted id={doc_id} status={status}\n")

        _append_jsonl(
            paths.ingest_log,
            {
                "ts": _utc_now(),
                "pdf": str(pdf_path),
                "sha": sha,
                "doc_id": doc_id,
                "status": status,
                "mineru_s": mineru_duration,
                "llm_s": llm_duration,
                "error": error_message,
            },
        )

        return IngestResult(
            pdf_path=pdf_path,
            sha256=sha,
            status=status,
            doc_id=doc_id,
            mineru_duration_s=mineru_duration,
            llm_duration_s=llm_duration,
            error_message=error_message,
            error=error_info,
        )
    finally:
        try:
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
    existing_id: Optional[int],
    mineru_duration: Optional[float],
    error_message: str,
    error: Optional[ErrorInfo] = None,
) -> IngestResult:
    """
    OCR 失败的收尾。DB 仍要记一条 status=failed，且保留 archive 内 source.pdf。

    如果是已成功/partial 的文档强制 reingest 失败，保留旧 output_dir 产物，只把本次
    失败日志挪到 archive root；如果是新文档或上次本来就是 failed，则把 tmp 提交成
    documents/<sha-short>/，至少留下 source.pdf + ingest.log。
    """
    log_handle.close()

    final_doc_dir = paths.doc_dir(sha)
    existing = get_document(conn, existing_id) if existing_id else None
    old_output_dir = Path(existing.output_dir) if existing and existing.output_dir else None
    keep_old_outputs = (
        existing is not None
        and existing.status in {"ok", "partial"}
        and old_output_dir is not None
        and old_output_dir.exists()
    )

    failed_log: Optional[Path]
    if keep_old_outputs:
        # 旧 OCR 产物仍可用，不能被一次失败的 reingest 覆盖；但确保留档 PDF 在可控目录内。
        _ensure_archived_source(paths, sha, pdf_path)
        failed_log = paths.root / f"failed_{sha[:SHA_SHORT_LEN]}_{int(time.time())}.log"
        try:
            (tmp_doc_dir / "ingest.log").rename(failed_log)
        except OSError:
            failed_log = None
        safe_rmtree(tmp_doc_dir)
        output_dir = str(old_output_dir)
    else:
        safe_rmtree(final_doc_dir)
        final_doc_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_doc_dir.rename(final_doc_dir)
        output_dir = str(final_doc_dir)
        failed_log = final_doc_dir / "ingest.log"

    if existing_id:
        replace_document(
            conn,
            existing_id,
            source_path=str(pdf_path),
            output_dir=output_dir,
            status="failed",
            mineru_duration_s=mineru_duration,
            llm_duration_s=None,
            error_message=error_message,
            extraction=ContractExtraction(),
            confidence=ExtractionConfidence(),
        )
        doc_id = existing_id
    else:
        doc_id = insert_document(
            conn,
            sha256=sha,
            source_path=str(pdf_path),
            output_dir=output_dir,
            status="failed",
            mineru_duration_s=mineru_duration,
            llm_duration_s=None,
            error_message=error_message,
            extraction=ContractExtraction(),
            confidence=ExtractionConfidence(),
        )

    _append_jsonl(
        paths.ingest_log,
        {
            "ts": _utc_now(),
            "pdf": str(pdf_path),
            "sha": sha,
            "doc_id": doc_id,
            "status": "failed",
            "mineru_s": mineru_duration,
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
        mineru_duration_s=mineru_duration,
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
    基于已有 mineru 产物重跑抽取。用于 partial 状态修复或调 prompt 后批量再抽取。
    不动 MinerU 产物，不动 sha256/source_path/ingested_at。
    """
    doc = get_document(conn, doc_id)
    if not doc:
        raise ValueError(f"document id={doc_id} not found")
    mineru_dir = Path(doc.output_dir) / "mineru"
    if not mineru_dir.exists():
        raise FileNotFoundError(
            f"mineru output missing for id={doc_id}: {mineru_dir}"
        )

    document_text = load_document_text(mineru_dir)
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
        envelope = DocumentExtraction()
    llm_duration = time.perf_counter() - t0

    # 类型专属后处理（据 doc_type 查处理器；合同=看落款页图重判签章）。
    if llm_enabled and status == "ok":
        for pp in get_handler(envelope.doc_type).post_processors:
            try:
                pp(envelope, mineru_dir)
            except Exception as e:  # noqa: BLE001 — 专属后处理失败不能中断重抽
                logger.warning("post:%s 跳过（异常）: %s", pp.__name__, e)
        try:
            correct_evidence_pages(envelope, mineru_dir)
        except Exception as e:  # noqa: BLE001 — 页码校正失败不能中断重抽
            logger.warning("page-fix 跳过（异常）: %s", e)

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
        _maybe_run_vision_fusion(envelope, document_text, mineru_dir, logger.info, confidence)

    # 落盘新 extracted.json（通用信封）
    (Path(doc.output_dir) / FILE_EXTRACTION).write_text(
        envelope.model_dump_json(indent=2), encoding="utf-8"
    )
    (Path(doc.output_dir) / FILE_EXTRACTION_CONF).write_text(
        confidence.model_dump_json(indent=2), encoding="utf-8"
    )

    update_extraction(
        conn,
        doc_id,
        status=status,
        llm_duration_s=llm_duration,
        error_message=error_message,
        extraction=extraction,
        confidence=confidence,
        envelope=envelope,
    )

    _append_jsonl(
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


def load_document_text(mineru_dir: Path) -> str:
    """优先 raw_text.txt（已清洗），fallback markdown.md。"""
    raw = mineru_dir / FILE_RAW_TEXT
    md = mineru_dir / FILE_MARKDOWN
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
