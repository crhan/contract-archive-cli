"""
档案库路径约定 + 文件操作小工具。

archive/
  db.sqlite (+ -wal, -shm)
  ingest.jsonl                # 总 log，每次 ingest 一行 JSON
  documents/
    <sha-short>/              # sha256 前 12 位
      source.pdf              # 硬链接源 PDF（跨盘 fallback copy）
      mineru/                 # MinerU 原始产物（markdown.md / layout.json / images/...）
      extracted.json          # 抽取结果 + 置信度（复跑 extract 命令的输入）
      ingest.log              # 单合同详细 stderr
  tmp/                        # ingest 过程暂存区，全成功后 os.rename 到 documents/
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


SHA_SHORT_LEN = 12


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
    def ingest_log(self) -> Path:
        return self.root / "ingest.jsonl"

    def doc_dir(self, sha256: str) -> Path:
        return self.documents_dir / sha256[:SHA_SHORT_LEN]

    def ensure(self) -> None:
        """启动时调用：建立根目录骨架。tmp/ 不立即建（按需创建并清理）。"""
        self.root.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(exist_ok=True)


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


def link_or_copy(src: Path, dst: Path) -> str:
    """
    优先硬链接（省空间，inode 共享），跨盘失败回退到 copy。
    返回实际使用的策略，"link" 或 "copy"。
    dst 已存在则先删除（reingest 场景）。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        # 跨盘 / 不支持硬链接的文件系统（exFAT 等）
        shutil.copy2(src, dst)
        return "copy"


def safe_rmtree(path: Path) -> None:
    """删除目录（不存在则忽略）。仅用于已知是本工具创建的目录。"""
    if path.exists():
        shutil.rmtree(path)
