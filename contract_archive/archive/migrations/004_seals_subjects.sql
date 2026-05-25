-- 档案库 schema v4：印章(seals) + 主体(subjects) 可索引子表
--
-- 设计要点（沿用 002/003 的既定取向）：
-- 1. 一类一表：印章和主体各自独立子表，承载各自的可索引列。这是项目既有风格
--    （risk_clauses / obligations 同构），让数据结构贴合查询，不上泛化 entities 表。
-- 2. 双存合理冗余：seals/subjects 既进 details_json（envelope 整体 dump，供展示），
--    又进子表（供 EXISTS 过滤 / 聚合）。写库三处（insert/update/replace）先 DELETE
--    再批量 INSERT 保证一致——和 obligations/risk_clauses 完全一致。
-- 3. ON DELETE CASCADE：删主表行时子表自动清，依赖 connect() 里的 PRAGMA foreign_keys=ON。
-- 4. subjects 来源是信封 parties（合同另并入 party_a/b），让"按主体检索"覆盖所有文档类型，
--    补上"证明类主体此前搜不到"的缺口。
-- 5. 不加 CHECK constraint（沿用 v1-v3 理由：未来加值免重建表，校验下沉到 Pydantic/LLM）。

CREATE TABLE document_seals (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  owner     TEXT,                 -- 盖章主体（公司/机构全称），认不出为 NULL
  seal_type TEXT,                 -- "公章" / "合同专用章" / "财务专用章" ...
  raw_text  TEXT NOT NULL,        -- 印章 OCR 原文（可能残缺），可追溯
  ordering  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_seals_doc   ON document_seals(doc_id);
CREATE INDEX idx_seals_owner ON document_seals(owner);
CREATE INDEX idx_seals_type  ON document_seals(seal_type);

CREATE TABLE document_subjects (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  subject  TEXT NOT NULL,         -- 主体名（人/机构全称）
  ordering INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_subjects_doc  ON document_subjects(doc_id);
CREATE INDEX idx_subjects_name ON document_subjects(subject);

INSERT INTO schema_version(version, applied_at)
  VALUES(4, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
