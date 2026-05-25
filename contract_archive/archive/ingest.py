"""
单 PDF 入库流水线。

流程（每个 PDF 一次调用）：
  1) 流式 SHA256
  2) 查 documents.sha256 → 命中 + 非 reingest 直接 skip
  3) 在 tmp/<sha-short>/ 跑 MinerU + 抽取（mineru 失败立刻退出）
  4) 全成功后 os.rename(tmp → documents/<sha-short>/) 是事务边界
  5) DB 写入 documents + risk_clauses（单事务，由 repository 保证）
  6) 追加一行 ingest.jsonl 总日志
  7) 失败时：清 tmp，记 status=failed 或 partial 到 DB（DB 仍要写一条便于查问题）

状态语义：
  - ok       MinerU + 抽取都成功
  - partial  MinerU 成功但 LLM 失败 → markdown 可用，可后续 extract 命令重跑
  - failed   MinerU 失败 → 没有可用产物
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..extraction import extract_contract, extract_document
from ..pipelines import MinerUPipeline
from ..schemas import (
    FILE_EXTRACTION,
    FILE_EXTRACTION_CONF,
    FILE_MARKDOWN,
    FILE_RAW_TEXT,
    ContractExtraction,
    DocumentExtraction,
    ExtractionConfidence,
)
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
    error_message: Optional[str] = None
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
    LLM-first 抽取：先判类型抽通用信封；若是合同，再跑合同抽取补专属列。
    （合同抽取自 Phase 2 起也是纯 LLM，不再有 rule/hybrid。）
    返回 (合同抽取, 置信度, 通用信封)——三者一并交给 repository 落库。
    """
    if not llm_enabled:
        # 无 LLM：通用路径纯靠 LLM、无从抽取；保留合同 rule 抽取作为无 key 兜底
        # （--no-llm 调试能力，见 README），并由其派生信封。
        ext, conf = extract_contract(document_text, llm_enabled=False)
        return ext, conf, contract_to_envelope(ext)

    envelope = extract_document(document_text, llm_enabled=llm_enabled)
    if envelope.doc_type == "合同协议" and llm_enabled:
        ext, conf = extract_contract(document_text, llm_enabled=llm_enabled)
        # 合同义务用合同抽取的（专属 prompt 对义务/罚则区分更细）
        envelope.obligations = ext.obligations
        # 标题若合同抽取没给，回退用信封的
        if not ext.contract_name and envelope.title:
            ext.contract_name = envelope.title
        return ext, conf, envelope
    # 非合同：无合同专属列，overall 走信封启发式
    conf = ExtractionConfidence()
    conf.overall = _envelope_confidence(envelope)
    return ContractExtraction(), conf, envelope


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
            if prev_status == "partial":
                hint = f"（MinerU 已完成、抽取未完成；用 `extract {existing_id}` 只重跑抽取，省去 MinerU）"
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

    mineru_duration: Optional[float] = None
    llm_duration: Optional[float] = None
    extraction: Optional[ContractExtraction] = None
    confidence: Optional[ExtractionConfidence] = None
    envelope: Optional[DocumentExtraction] = None
    error_message: Optional[str] = None
    status = "ok"

    try:
        # ---- 1. MinerU 解析 ----
        pl = pipeline or MinerUPipeline()
        t0 = time.perf_counter()
        try:
            pl.run(pdf_path, mineru_dir)
            mineru_duration = time.perf_counter() - t0
            log_handle.write(f"\n[mineru] OK in {mineru_duration:.2f}s\n")
        except Exception as e:
            mineru_duration = time.perf_counter() - t0
            status = "failed"
            error_message = f"mineru: {e}"
            log_handle.write(f"\n[mineru] FAILED: {error_message}\n")
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
            )

        # ---- 2. 抽取（基于 mineru 产物的 raw_text.txt 优先） ----
        document_text = _load_document_text(mineru_dir)
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
            ):
                status = "partial"
                error_message = (
                    "LLM 抽取为空——通常是缺 DASHSCOPE_API_KEY（全局工具需在 shell "
                    "export，不读项目 .env）或 LLM 调用失败；补好后用 `extract <id>` 重抽"
                )
                log_handle.write(f"\n[extract] WARNING: {error_message}\n")
        except Exception as e:
            llm_duration = time.perf_counter() - t1
            status = "partial"
            error_message = f"extract: {e}"
            extraction = ContractExtraction()
            confidence = ExtractionConfidence()
            envelope = DocumentExtraction()
            log_handle.write(f"\n[extract] FAILED (status=partial): {error_message}\n")
            log_handle.write(traceback.format_exc())

        # ---- 3. extracted.json 落盘（写通用信封；即使 partial 也写空对象，便于后续 extract 复跑） ----
        (tmp_doc_dir / FILE_EXTRACTION).write_text(
            envelope.model_dump_json(indent=2), encoding="utf-8"
        )
        (tmp_doc_dir / FILE_EXTRACTION_CONF).write_text(
            confidence.model_dump_json(indent=2), encoding="utf-8"
        )

        # ---- 4. 硬链接源 PDF（断开后用户挪走原文件也不影响档案副本） ----
        link_strategy = link_or_copy(pdf_path, tmp_doc_dir / "source.pdf")
        log_handle.write(f"[source.pdf] {link_strategy}ed from {pdf_path}\n")

        # ---- 5. 事务边界：rename tmp → documents/<sha-short>/ ----
        final_doc_dir = paths.doc_dir(sha)
        safe_rmtree(final_doc_dir)
        final_doc_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_doc_dir.rename(final_doc_dir)
        # rename 之后 log_handle 仍然有效（文件描述符不依赖路径），但 path 已变
        # 为了后续追加，把 handle 关掉再开新的
        log_handle.close()
        log_handle = (final_doc_dir / "ingest.log").open("a", encoding="utf-8")

        # ---- 6. DB 写入 ----
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
) -> IngestResult:
    """
    MinerU 失败的收尾。DB 仍要记一条 status=failed 便于查问题，但 tmp 目录清掉。
    """
    log_handle.close()
    # 把单合同 log 移到 archive root 下方便查（tmp 即将被清）
    failed_log = paths.root / f"failed_{sha[:SHA_SHORT_LEN]}_{int(time.time())}.log"
    try:
        (tmp_doc_dir / "ingest.log").rename(failed_log)
    except OSError:
        failed_log = None
    safe_rmtree(tmp_doc_dir)

    if existing_id:
        # 失败重跑：保留原 output_dir 不变（旧产物可能还能用），只更状态
        existing = get_document(conn, existing_id)
        replace_document(
            conn,
            existing_id,
            source_path=str(pdf_path),
            output_dir=existing.output_dir if existing else "",
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
            output_dir="",
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
    )


# ---------- 复跑抽取（partial 状态修复，不重跑 MinerU） ----------


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

    document_text = _load_document_text(mineru_dir)
    t0 = time.perf_counter()
    error_message: Optional[str] = None
    status = "ok"
    envelope = DocumentExtraction()
    try:
        extraction, confidence, envelope = _run_extraction(
            document_text, llm_enabled=llm_enabled
        )
    except Exception as e:
        status = "partial"
        error_message = f"extract: {e}"
        extraction = ContractExtraction()
        confidence = ExtractionConfidence()
        envelope = DocumentExtraction()
    llm_duration = time.perf_counter() - t0

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
    )


# ---------- 工具 ----------


def _load_document_text(mineru_dir: Path) -> str:
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
