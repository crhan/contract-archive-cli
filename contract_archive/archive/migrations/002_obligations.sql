-- 档案库 schema v2：合同双方义务/动作清单
--
-- 设计要点：
-- 1. 独立表：典型合同有 5-15 条义务，每条带 actor + deadline，独立表才能走
--    `WHERE deadline < ?` 索引做"近 30 天待办看板"
-- 2. 不加 actor/severity 的 CHECK constraint（v1 同样的理由：未来加新值
--    免重建表，Pydantic 层校验更灵活）
-- 3. ordering 列保留原文出现顺序，show 命令展示时按顺序好读

CREATE TABLE obligations (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  actor      TEXT NOT NULL,        -- 'party_a' | 'party_b' | 'both'
  action     TEXT NOT NULL,        -- "递交审贷资料"
  deadline   TEXT,                  -- ISO 'YYYY-MM-DD' 或 NULL
  evidence   TEXT,                  -- 原文片段
  ordering   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_obligations_doc      ON obligations(doc_id);
CREATE INDEX idx_obligations_deadline ON obligations(deadline);
CREATE INDEX idx_obligations_actor    ON obligations(actor);

INSERT INTO schema_version(version, applied_at)
  VALUES(2, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
