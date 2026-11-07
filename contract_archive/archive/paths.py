"""档案库路径约定、发布槽锁和文件操作小工具。

``documents`` 中的目录是不可变 generation，而不是按 SHA 前缀复用的可变目录：

.. code-block:: text

   archive/
     documents/
       <sha12>.gen-<uuid>/    # 已发布 generation；documents.output_dir 指向当前一代
       <sha12>/               # 仅兼容历史档案，绝不作为新写入目标
     tmp/runs/<sha12>-<uuid>/ # 尚未发布的 staging
     locks/<sha12>.lock       # 长期保留的物理槽锁文件，不能 unlink 后重建

同一个 SHA 前缀仍共用一个锁槽，避免旧档案的前缀碰撞在查重到提交之间相互干扰；
generation 名本身带 UUID，因此不同完整 SHA 即使前缀相同也不会覆盖彼此。
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


SHA_SHORT_LEN = 12
GENERATION_MARKER = ".gen-"

# XDG 数据目录下的应用子目录名（跟 CLI / repo 名对齐）
APP_DIR_NAME = "contract-archive"


def default_archive_root() -> Path:
    """
    无 --archive / CONTRACT_ARCHIVE_DIR 时的默认档案库根，遵循 XDG Base Directory 约定。

    数据类文件（db + 文档产物 + 日志）属于 XDG "data"：
      $XDG_DATA_HOME/contract-archive  （默认 ~/.local/share/contract-archive）

    按 XDG 规范：$XDG_DATA_HOME 仅在被设置且为绝对路径时生效，否则回退默认值。
    """
    xdg_data = os.getenv("XDG_DATA_HOME", "").strip()
    base = Path(xdg_data) if xdg_data and os.path.isabs(xdg_data) else Path.home() / ".local" / "share"
    return base / APP_DIR_NAME


def _ensure_real_directory(path: Path) -> None:
    """Create ``path`` only when it is a real directory, never a symlink.

    Archive-controlled children are all created in parent-before-child order by
    :meth:`ArchivePaths.ensure`.  Checking before ``mkdir`` is important: an
    existing symlink must fail before any operation can write through it.  This
    is a practical same-user safety guard, not a claim of a full dirfd-based
    sandbox against a malicious process racing directory replacement.
    """
    if path.is_symlink():
        raise ValueError(f"refusing symlinked archive directory: {path}")
    try:
        path.mkdir()
    except FileExistsError:
        pass
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"archive path is not a real directory: {path}")


@dataclass(frozen=True)
class ArchivePaths:
    """档案库根目录 + 派生路径。"""

    root: Path

    @property
    def db_path(self) -> Path:
        return self.root / "db.sqlite"

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def runs_dir(self) -> Path:
        """每次 ingest/re-extract 独占的 staging 根目录。"""
        return self.tmp_dir / "runs"

    @property
    def locks_dir(self) -> Path:
        """SHA 前缀槽锁目录。锁文件一旦创建必须永久保留，不能 unlink 以免竞态。"""
        return self.root / "locks"

    @property
    def failed_logs_dir(self) -> Path:
        """已有成功档案重跑失败时保存的独立诊断日志。"""
        return self.root / "failed-logs"

    @property
    def ingest_log(self) -> Path:
        return self.root / "ingest.jsonl"

    @property
    def known_parties_path(self) -> Path:
        """主体身份基准库（known_parties.json）。含真实 PII，存档案库根、权限 0600。"""
        return self.root / "known_parties.json"

    def doc_dir(self, sha256: str) -> Path:
        """历史可变目录位置；仅为旧库兼容保留，绝不作为新发布目标。"""
        return self.documents_dir / sha256[:SHA_SHORT_LEN]

    def generation_dir(self, sha256: str, generation_id: Optional[str] = None) -> Path:
        """返回一个 generation 目录名；调用方以 rename 发布，不得覆盖既有目录。"""
        prefix = _sha_prefix(sha256)
        run_id = generation_id or uuid.uuid4().hex
        _validate_generation_id(run_id)
        return self.documents_dir / f"{prefix}{GENERATION_MARKER}{run_id}"

    def staging_dir(self, sha256: str, run_id: Optional[str] = None) -> Path:
        """返回一个尚未发布的 staging 路径（不创建目录）。"""
        prefix = _sha_prefix(sha256)
        token = run_id or uuid.uuid4().hex
        _validate_generation_id(token)
        return self.runs_dir / f"{prefix}-{token}"

    def resolve_document_dir(
        self,
        row: object | None = None,
        *,
        sha256: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
    ) -> Path:
        """解析一条记录当前 generation 的安全本地路径。

        ``output_dir`` 是数据库中的 pointer，但搬库后它可能仍含旧 archive 的绝对路径。
        因此只采纳经过校验的受控 basename（``<sha12>`` 旧格式或
        ``<sha12>.gen-<uuid>`` 新格式），并始终重新锚定到本 ``ArchivePaths`` 的
        ``documents/`` 下。非法 basename、穿越或指向 documents 外的 symlink 都拒绝。

        这也是 CLI 的 raw/delete 等需要使用的入口；它不查询 DB，也不猜测 generation。
        """
        if row is not None:
            if sha256 is not None or output_dir is not None:
                raise ValueError("pass either row or sha256/output_dir, not both")
            sha256 = getattr(row, "sha256", None)
            output_dir = getattr(row, "output_dir", None)
        prefix = _sha_prefix(sha256 or "")
        name = _document_dir_name(prefix, output_dir)
        raw_documents = self.documents_dir
        if raw_documents.is_symlink():
            raise ValueError(f"refusing symlinked documents root: {raw_documents}")
        documents = raw_documents.resolve()
        unresolved = documents / name
        # ``Path.resolve`` alone is insufficient: a symlink from one controlled
        # basename to another sibling would still have ``documents`` as parent.
        # A DB pointer must never make a read-only CLI command follow a symlink.
        if unresolved.is_symlink():
            raise ValueError(f"refusing symlinked document directory: {unresolved}")
        candidate = unresolved.resolve()
        if candidate.parent != documents:
            raise ValueError(f"document directory escapes archive root: {output_dir!r}")
        return candidate

    def document_dirs_for_sha(self, sha256: str) -> list[Path]:
        """Return every safely verifiable legacy/generation directory for ``sha256``.

        This is intentionally stricter than :meth:`resolve_document_dir` and is
        used only by destructive ``delete --purge-files``.  Directory names are
        merely a short-SHA routing hint: each candidate's regular ``source.pdf``
        must hash to the requested *full* SHA before it can be returned.  A
        sibling with the same 12-character prefix but a different full hash is
        left alone.  Invalid controlled-looking candidates or symlinks abort the
        purge before its database row is deleted.
        """
        prefix = _sha_prefix(sha256)
        documents = self.documents_dir
        if documents.is_symlink():
            raise ValueError(f"refusing symlinked documents root: {documents}")
        if not documents.exists():
            return []
        if not documents.is_dir():
            raise ValueError(f"documents root is not a directory: {documents}")

        matching: list[Path] = []
        generation_prefix = f"{prefix}{GENERATION_MARKER}"
        for candidate in documents.iterdir():
            name = candidate.name
            if name != prefix and not name.startswith(generation_prefix):
                continue
            # Validate the whole basename rather than treating a prefix glob as
            # authority.  This also rejects a malformed generation that could
            # otherwise be swapped in after planning a destructive operation.
            _document_dir_name(prefix, name)
            if candidate.is_symlink():
                raise ValueError(f"refusing symlinked document directory: {candidate}")
            if not candidate.is_dir():
                raise ValueError(f"document candidate is not a directory: {candidate}")
            source = candidate / "source.pdf"
            if source.is_symlink():
                raise ValueError(f"refusing symlinked archived source: {source}")
            if not source.is_file():
                raise ValueError(f"archived source missing for purge candidate: {source}")
            if sha256_of_file(source) == sha256.lower():
                matching.append(candidate)
        return matching

    def slot_lock_path(self, sha256: str) -> Path:
        """返回完整 SHA 前缀的长期槽锁文件路径（不创建/不加锁）。"""
        return self.locks_dir / f"{_sha_prefix(sha256)}.lock"

    @contextmanager
    def slot_lock(self, sha256: str) -> Iterator[None]:
        """持有 SHA 前缀的进程间排他锁，直到调用方完成 DB 提交。

        不能通过删除锁文件来"清理"：等待方可能已打开旧 inode，unlink 后新进程会锁
        新 inode，两个临界区随即并发。文件因此永久留在 ``locks/``，只释放 flock。
        """
        # Direct callers may acquire a slot lock without calling ``ensure``
        # first.  Validate every controlled parent before opening the lock file.
        self.ensure()
        path = self.slot_lock_path(sha256)
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def ensure(self) -> None:
        """Create/validate archive-controlled directories without following symlinks.

        ``documents/``, ``tmp/``, ``tmp/runs/`` and ``locks/`` are writable
        archive internals.  Existing symlinks are rejected before we create a
        child or open a file, so a poisoned archive layout cannot redirect a
        publish, staging write, or lock file outside the archive.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError(f"archive root is not a directory: {self.root}")
        _ensure_real_directory(self.documents_dir)
        _ensure_real_directory(self.tmp_dir)
        _ensure_real_directory(self.runs_dir)
        _ensure_real_directory(self.locks_dir)


def _sha_prefix(sha256: str) -> str:
    """校验完整 SHA-256，并返回规范化的 12 位前缀。"""
    if len(sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in sha256):
        raise ValueError(f"invalid sha256 for archive path: {sha256!r}")
    return sha256[:SHA_SHORT_LEN].lower()


def _validate_generation_id(generation_id: str) -> None:
    """允许 UUID 的 hex 或标准带横杠形式，拒绝任意文件名片段。"""
    try:
        uuid.UUID(generation_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"invalid generation id: {generation_id!r}") from exc


def _document_dir_name(prefix: str, output_dir: Optional[str | Path]) -> str:
    """从 DB pointer 提取并校验受控目录 basename；空 pointer 回退历史位置。"""
    if output_dir is None or not str(output_dir).strip():
        return prefix
    name = Path(output_dir).name
    if name == prefix:
        return name
    marker = f"{prefix}{GENERATION_MARKER}"
    if not name.startswith(marker):
        raise ValueError(f"untrusted document directory name: {name!r}")
    _validate_generation_id(name[len(marker):])
    return name


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """流式 SHA256（避免大文件全文件读入）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path, *, expected_sha256: Optional[str] = None) -> str:
    """
    把源 PDF 做独立的字节快照并原子放到 ``dst``。

    名字为了兼容旧调用保留，但不再硬链接：输入文件被用户原地修改后，硬链接会让
    档案内的 ``source.pdf`` 静默改变。若给出 ``expected_sha256``，在临时副本上校验，
    从而发现 hash 与复制之间的源文件竞态。返回值始终是 ``"copy"``。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.copying-{uuid.uuid4().hex}"
    try:
        shutil.copy2(src, tmp)
        if expected_sha256 is not None and sha256_of_file(tmp) != expected_sha256:
            raise ValueError(f"source changed while taking archive snapshot: {src}")
        os.replace(tmp, dst)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return "copy"


def safe_rmtree(path: Path) -> None:
    """删除目录（不存在则忽略）。仅用于已知是本工具创建的目录。"""
    if path.exists():
        shutil.rmtree(path)
