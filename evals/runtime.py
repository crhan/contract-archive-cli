"""Durable local state for the extraction evaluation runner.

This module deliberately has no model/API dependency.  It owns the small pieces
of state that make an evaluation reproducible and safe to resume:

* immutable ``run_meta.json`` manifests;
* strict JSONL validation before appending;
* process locks for an output directory and for a shared OCR cache entry; and
* atomic, complete-only OCR artifact cache publication.

The cache contains private source PDFs and OCR artifacts, so it is intentionally
local-only.  Nothing here uploads, logs, or serialises document text outside the
evaluation output chosen by the caller.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


MANIFEST_FILENAME = "run_meta.json"
RESULTS_FILENAME = "results.jsonl"
MANIFEST_SCHEMA_VERSION = 2
CACHE_SCHEMA_VERSION = 1
CACHE_META_FILENAME = "cache_meta.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class EvalRuntimeError(RuntimeError):
    """Base error for unsafe or non-reproducible evaluation state."""


class ManifestMismatch(EvalRuntimeError):
    """An existing output directory does not describe this requested run."""


class CorruptResults(EvalRuntimeError):
    """``results.jsonl`` is not safe to append to."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Stable JSON representation used for fingerprints and atomic writes."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file without putting private PDF contents in memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_paths(paths: list[Path], *, root: Path) -> dict[str, Any]:
    """Return a content fingerprint with deterministic repository-relative paths.

    The individual file hashes make a manifest mismatch auditable without putting
    file contents into the manifest.  Callers decide the path set, which lets the
    runner exclude reports, tests, and documentation from its source identity.
    """
    root = root.resolve()
    files: list[dict[str, str]] = []
    for path in sorted({candidate.resolve() for candidate in paths}, key=lambda p: str(p)):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"fingerprint path is outside root: {path}") from exc
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {"sha256": sha256_bytes(canonical_json_bytes(files)), "files": files}


def project_source_fingerprint() -> dict[str, Any]:
    """Fingerprint the source set that can change extraction-evaluation results.

    This intentionally preserves the runner's established source set: production
    Python plus the runner, durable-runtime, and scoring helpers.  Reports use
    the same function to reject a stale run instead of merely type-checking the
    fingerprint saved in its manifest.
    """
    product = list((PROJECT_ROOT / "contract_archive").rglob("*.py"))
    helpers = [
        PROJECT_ROOT / "evals" / "run.py",
        PROJECT_ROOT / "evals" / "runtime.py",
        PROJECT_ROOT / "evals" / "score.py",
    ]
    return fingerprint_paths(product + helpers, root=PROJECT_ROOT)


def sanitize_base_url(value: str) -> str:
    """Remove credentials, query parameters, and fragments from a display URL."""
    try:
        parts = urlsplit(value)
        port = parts.port  # Accessing this can itself reject an invalid port.
    except ValueError:
        # Never retain a malformed string: it could include credentials or a
        # query-string token that urlsplit could not safely interpret.
        return "<invalid-url>"
    if not parts.scheme or not parts.hostname:
        return "<invalid-url>"
    host = parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Fingerprint only caller-supplied, already sanitised non-secret config."""
    return sha256_bytes(canonical_json_bytes(dict(config)))


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability after an atomic rename on Linux."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file, never exposing a half-written manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _read_json_object(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalRuntimeError(f"{kind} is unreadable or invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise EvalRuntimeError(f"{kind} must be a JSON object: {path}")
    return payload


@contextmanager
def output_lock(out_dir: Path) -> Iterator[None]:
    """Serialize mutation of one results directory.

    The lock file is intentionally retained.  ``flock`` is tied to the opened
    inode; deleting/recreating it would permit two waiters to lock different
    inodes, the same trap avoided by the archive's per-document locks.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".run.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def create_or_validate_manifest(out_dir: Path, requested: Mapping[str, Any]) -> dict[str, Any]:
    """Create an immutable manifest once, or require exact identity on resume."""
    manifest_path = out_dir / MANIFEST_FILENAME
    desired = dict(requested)
    if manifest_path.exists():
        existing = _read_json_object(manifest_path, kind="run manifest")
        if existing != desired:
            changed = sorted(set(existing) | set(desired))
            changed = [key for key in changed if existing.get(key) != desired.get(key)]
            raise ManifestMismatch(
                "existing run_meta.json does not match requested evaluation "
                f"(changed: {', '.join(changed) or 'unknown'})"
            )
        return existing

    # Never bless an old/mysterious result directory by adding a new manifest.
    # The lock file is the sole expected pre-manifest artifact created by us.
    leftovers = [p.name for p in out_dir.iterdir() if p.name != ".run.lock"]
    if leftovers:
        raise ManifestMismatch(
            "output directory has no run_meta.json and is not empty; refusing to append "
            f"to legacy/unidentified state: {', '.join(sorted(leftovers))}"
        )
    atomic_write_json(manifest_path, desired)
    return desired


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Strictly read JSONL; any malformed/blank line makes resume unsafe."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CorruptResults(f"cannot read results JSONL: {path}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise CorruptResults(f"blank line in results JSONL at line {number}: {path}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorruptResults(
                f"invalid JSON in results JSONL at line {number}: {path}"
            ) from exc
        if not isinstance(record, dict):
            raise CorruptResults(f"non-object record at line {number}: {path}")
        records.append(record)
    return records


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one complete record while the caller holds :func:`output_lock`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(record)) + b"\n"
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def record_identity(record: Mapping[str, Any]) -> tuple[str, str, int]:
    """Return the resume identity and reject malformed historical records."""
    case_id = record.get("case_id")
    model = record.get("model")
    raw_repeat = record.get("repeat_idx")
    if not isinstance(case_id, str) or not case_id:
        raise CorruptResults("result record has no valid case_id")
    if not isinstance(model, str) or not model:
        raise CorruptResults("result record has no valid model")
    if isinstance(raw_repeat, bool):
        raise CorruptResults("result record has invalid repeat_idx")
    try:
        repeat_idx = int(raw_repeat)
    except (TypeError, ValueError) as exc:
        raise CorruptResults("result record has invalid repeat_idx") from exc
    if repeat_idx < 0:
        raise CorruptResults("result record has negative repeat_idx")
    return case_id, model, repeat_idx


def latest_records(records: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Last JSONL write wins only after an explicit ``--retry-failed`` rerun."""
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in records:
        result[record_identity(record)] = record
    return result


def validate_records_for_manifest(
    records: list[dict[str, Any]], *, manifest: Mapping[str, Any]
) -> None:
    """Reject records that could not have been produced by this manifest."""
    expected_fingerprint = manifest.get("config_fingerprint")
    suite = manifest.get("suite")
    axis = manifest.get("axis")
    for record in records:
        record_identity(record)
        if record.get("config_fingerprint") != expected_fingerprint:
            raise CorruptResults("result record config_fingerprint differs from run_meta.json")
        if record.get("suite") != suite:
            raise CorruptResults("result record suite differs from run_meta.json")
        if record.get("axis") != axis:
            raise CorruptResults("result record axis differs from run_meta.json")
        if record.get("status") not in {"ok", "partial", "error"}:
            raise CorruptResults("result record has invalid status")


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalRuntimeError(f"OCR cache metadata has invalid {field}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise EvalRuntimeError(f"OCR cache metadata has invalid {field}")
    return numeric


def _safe_cache_calls(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise EvalRuntimeError("OCR cache metadata has invalid ocr_calls")
    # json round-trip isolates caller-owned data and guarantees cache metadata is
    # JSON serialisable before it is copied into a result record.
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


@dataclass(frozen=True)
class OCRCachePayload:
    """The small, privacy-safe index stored beside cached OCR artifacts."""

    ocr_latency_s: float
    ocr_calls: list[dict[str, Any]]
    pipeline_meta: dict[str, Any]
    complete: bool


@dataclass(frozen=True)
class OCRCacheResult:
    """A usable cache entry or an ephemeral incomplete OCR run."""

    artifact_dir: Path
    cache_hit: bool
    payload: OCRCachePayload
    cleanup_dir: Path | None = None


class OCRArtifactCache:
    """Shared, keyed OCR artifact cache with complete-only publication.

    ``producer`` receives paths for a private source snapshot and its ``ocr``
    directory.  A false ``payload.complete`` is useful to callers for one run but
    is never renamed into the reusable cache namespace.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def entry_path(self, cache_key: str) -> Path:
        if len(cache_key) != 64 or any(ch not in "0123456789abcdef" for ch in cache_key):
            raise ValueError("cache key must be a lowercase SHA-256 hex digest")
        return self.root / cache_key

    @contextmanager
    def _entry_lock(self, cache_key: str) -> Iterator[None]:
        locks = self.root / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / f"{cache_key}.lock").open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_complete(
        self, entry: Path, *, cache_key: str, key_material: Mapping[str, Any]
    ) -> OCRCacheResult | None:
        if not entry.exists():
            return None
        if entry.is_symlink() or not entry.is_dir():
            raise EvalRuntimeError(f"unsafe OCR cache entry: {entry}")
        meta_path = entry / CACHE_META_FILENAME
        if not meta_path.is_file():
            return None
        try:
            meta = _read_json_object(meta_path, kind="OCR cache metadata")
            if (
                meta.get("schema_version") != CACHE_SCHEMA_VERSION
                or meta.get("cache_key") != cache_key
                or meta.get("key_material") != dict(key_material)
                or meta.get("complete") is not True
            ):
                return None
            source = entry / "source.pdf"
            ocr = entry / "ocr"
            if not source.is_file() or source.is_symlink() or not ocr.is_dir() or ocr.is_symlink():
                return None
            # OCRPipeline always produces raw_text.txt on successful completion;
            # requiring it prevents a crash after directory creation from looking
            # like a usable artifact merely because metadata happened to exist.
            if not (ocr / "raw_text.txt").is_file():
                return None
            payload = OCRCachePayload(
                ocr_latency_s=_finite_nonnegative(meta.get("ocr_latency_s"), field="ocr_latency_s"),
                ocr_calls=_safe_cache_calls(meta.get("ocr_calls")),
                pipeline_meta=dict(meta.get("pipeline_meta") or {}),
                complete=True,
            )
        except EvalRuntimeError:
            raise
        except (TypeError, ValueError):
            return None
        return OCRCacheResult(artifact_dir=entry, cache_hit=True, payload=payload)

    @staticmethod
    def _discard_invalid_entry(entry: Path) -> None:
        """Remove only a known cache path while holding its per-key lock."""
        if not entry.exists():
            return
        if entry.is_symlink():
            raise EvalRuntimeError(f"refusing to remove symlinked OCR cache entry: {entry}")
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    def get_or_create(
        self,
        *,
        cache_key: str,
        key_material: Mapping[str, Any],
        pdf_path: Path,
        producer: Callable[[Path, Path], OCRCachePayload],
    ) -> OCRCacheResult:
        """Return a complete cached OCR result or run/publish it atomically."""
        self.root.mkdir(parents=True, exist_ok=True)
        entry = self.entry_path(cache_key)
        key_data = dict(key_material)
        with self._entry_lock(cache_key):
            cached = self._load_complete(entry, cache_key=cache_key, key_material=key_data)
            if cached is not None:
                return cached
            if entry.exists():
                # An interrupted/old/incompatible entry must never be reused. It
                # is safe to clean because it is exactly ``root/<sha256>`` under
                # this cache's lock, never a caller-provided recursive target.
                self._discard_invalid_entry(entry)

            stage = Path(tempfile.mkdtemp(prefix=f".{cache_key[:12]}-", dir=self.root))
            try:
                source_snapshot = stage / "source.pdf"
                shutil.copy2(pdf_path, source_snapshot)
                expected_pdf_sha = key_data.get("pdf_sha256")
                if (
                    isinstance(expected_pdf_sha, str)
                    and len(expected_pdf_sha) == 64
                    and sha256_file(source_snapshot) != expected_pdf_sha
                ):
                    raise EvalRuntimeError("source PDF changed while creating OCR cache entry")
                ocr_dir = stage / "ocr"
                payload = producer(source_snapshot, ocr_dir)
                if not isinstance(payload, OCRCachePayload):
                    raise TypeError("OCR cache producer must return OCRCachePayload")
                metadata: dict[str, Any] = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "key_material": key_data,
                    "complete": bool(payload.complete),
                    "ocr_latency_s": payload.ocr_latency_s,
                    "ocr_calls": payload.ocr_calls,
                    "pipeline_meta": payload.pipeline_meta,
                }
                # Validate all fields before publishing any directory. This also
                # catches accidental SDK objects in a ledger rather than leaving
                # an unreadable cache entry behind.
                canonical_json_bytes(metadata)
                atomic_write_json(stage / CACHE_META_FILENAME, metadata)

                if not payload.complete:
                    return OCRCacheResult(
                        artifact_dir=stage,
                        cache_hit=False,
                        payload=payload,
                        cleanup_dir=stage,
                    )

                if not (ocr_dir / "raw_text.txt").is_file():
                    raise EvalRuntimeError("complete OCR producer did not write raw_text.txt")
                os.replace(stage, entry)
                _fsync_directory(self.root)
                return OCRCacheResult(artifact_dir=entry, cache_hit=False, payload=payload)
            except Exception:
                if stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)
                raise

    @staticmethod
    def cleanup(result: OCRCacheResult) -> None:
        """Remove a non-reusable temporary OCR run after its record is written."""
        if result.cleanup_dir is not None and result.cleanup_dir.exists():
            shutil.rmtree(result.cleanup_dir, ignore_errors=True)


def make_ocr_cache_key(key_material: Mapping[str, Any]) -> str:
    """Return the content address for the PDF/OCR model/settings/source identity."""
    return sha256_bytes(canonical_json_bytes(dict(key_material)))
