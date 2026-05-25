"""
本地合同档案库（SQLite + 文件系统）。

模块结构：
- db.py         连接 / PRAGMA / migrations 引擎
- repository.py DAO（CRUD + search + stats）
- ingest.py     单 PDF 入库流水线（hash → MinerU → extract → rename → DB）
- paths.py      档案库路径约定 + 硬链接/拷贝工具
"""
from .db import checkpoint, open_archive_db, transaction, utc_now_iso
from .ingest import IngestResult, discover_pdfs, ingest_pdf, re_extract
from .paths import ArchivePaths, default_archive_root, link_or_copy, sha256_of_file
from .repository import (
    DocumentRow,
    SealRow,
    SearchFilter,
    Stats,
    TodoItem,
    collect_stats,
    delete_document,
    find_by_sha,
    find_by_sha_prefix,
    get_document,
    insert_document,
    list_documents,
    list_obligations,
    list_seals,
    replace_document,
    search_documents,
    update_extraction,
)

__all__ = [
    "open_archive_db",
    "transaction",
    "checkpoint",
    "utc_now_iso",
    "ArchivePaths",
    "default_archive_root",
    "link_or_copy",
    "sha256_of_file",
    "DocumentRow",
    "SealRow",
    "SearchFilter",
    "Stats",
    "TodoItem",
    "list_obligations",
    "list_seals",
    "find_by_sha",
    "find_by_sha_prefix",
    "get_document",
    "list_documents",
    "search_documents",
    "insert_document",
    "update_extraction",
    "replace_document",
    "delete_document",
    "collect_stats",
    "IngestResult",
    "ingest_pdf",
    "re_extract",
    "discover_pdfs",
]
