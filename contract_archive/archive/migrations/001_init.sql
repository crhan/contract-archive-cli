-- 档案库初始 schema (version 1)
--
-- 设计要点（来自多轮 review 仲裁）：
-- 1. status / severity 不加 CHECK constraint：未来加状态要重建表（SQLite 限制），
--    校验下沉到 Pydantic 层更灵活
-- 2. amount 用 INTEGER 分（amount_cents），避免 REAL 累加精度漂移；
--    同时保留 amount_text 原文供人工核对
-- 3. 不用 FTS5：典型档案库规模千级，LIKE '%关键词%' 全表扫毫秒级；
--    trigram 要求 ≥3 字符匹配，2 字中文人名/词（"车位"/"张三"）会全部 miss，
--    unicode61 单字切分对中文精度太差。务实选择 LIKE。
-- 4. ingested_at 加 DESC 索引：list 命令默认排序走索引
-- 5. AUTOINCREMENT 保留：防止 delete 后 rowid 复用指向新合同

CREATE TABLE schema_version (
  version    INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE documents (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  sha256              TEXT NOT NULL UNIQUE,
  source_path         TEXT NOT NULL,            -- 原始 PDF 绝对路径（用户引用）
  output_dir          TEXT NOT NULL,            -- archive/documents/<sha-short>/ 绝对路径
  ingested_at         TEXT NOT NULL,            -- ISO8601 UTC 'YYYY-MM-DDTHH:MM:SSZ'
  mineru_duration_s   REAL,
  llm_duration_s      REAL,
  status              TEXT NOT NULL,            -- 'ok' | 'partial' | 'failed'
  error_message       TEXT,

  -- 合同字段（rule + LLM hybrid 抽取结果）
  contract_name       TEXT,
  party_a             TEXT,
  party_b             TEXT,
  amount_text         TEXT,                     -- "人民币壹佰万元整"，原文
  amount_cents        INTEGER,                  -- 1000000 * 100 = 100000000，精确分
  sign_date           TEXT,                     -- ISO 'YYYY-MM-DD'
  expire_date         TEXT,
  auto_renewal        INTEGER,                  -- 0/1/NULL
  overall_confidence  REAL                      -- [0, 1]
);

CREATE INDEX idx_doc_party_a   ON documents(party_a);
CREATE INDEX idx_doc_party_b   ON documents(party_b);
CREATE INDEX idx_doc_sign_date ON documents(sign_date);
CREATE INDEX idx_doc_expire    ON documents(expire_date);
CREATE INDEX idx_doc_amount    ON documents(amount_cents);
CREATE INDEX idx_doc_status    ON documents(status);
CREATE INDEX idx_doc_ingested  ON documents(ingested_at DESC);

CREATE TABLE risk_clauses (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  clause_text TEXT NOT NULL,
  severity    TEXT                              -- 'low' | 'med' | 'high' | NULL
);
CREATE INDEX idx_risk_doc ON risk_clauses(doc_id);

-- name / party 字符串字段加索引，加速 LIKE '%xxx%' 之外的等值/前缀查询。
-- LIKE '%xxx%' 本身无法走 B-tree 索引（前置通配），但合同档案库规模小，
-- 全表扫描完全可接受（1 万条 < 10ms）。

INSERT INTO schema_version(version, applied_at) VALUES(1, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
