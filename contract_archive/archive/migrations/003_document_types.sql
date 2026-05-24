-- 档案库 schema v3：多文档类型支持（合同 → 通用文档档案库）
--
-- 设计要点：
-- 1. 加法式迁移——保留全部合同列（party_a/b、amount_*、sign/expire_date、
--    auto_renewal、risk_clauses、obligations 不动），合同的查询/统计照常工作。
-- 2. 新增"通用信封"列：任何文档类型都填这几列，list/show/搜索走通用列即可
--    跨类型统一展示，类型专属字段整体存 details_json。
-- 3. doc_type 不加 CHECK constraint（沿用 v1/v2 理由：未来加类型免重建表，
--    校验下沉到 Pydantic/LLM 层）。规范取值见 schemas.DOC_TYPES。
-- 4. 回填：现存行都是历史合同，把合同字段映射到通用列，保证老数据在新
--    list/show 里不空白。

ALTER TABLE documents ADD COLUMN doc_type TEXT NOT NULL DEFAULT '合同协议';
ALTER TABLE documents ADD COLUMN title TEXT;                  -- 通用标题（合同名/证明抬头/发票号…）
ALTER TABLE documents ADD COLUMN summary TEXT;                -- 一句话摘要（可追溯钩子）
ALTER TABLE documents ADD COLUMN details_json TEXT;          -- 类型专属字段（parties/amounts/fields/key_dates）整体 JSON
ALTER TABLE documents ADD COLUMN primary_date TEXT;          -- 主日期 ISO（合同=签订日，证明=出具日）
ALTER TABLE documents ADD COLUMN primary_amount_cents INTEGER;  -- 主金额（分），跨类型金额排序/过滤

-- 回填历史合同行 → 通用列
UPDATE documents SET title = contract_name WHERE title IS NULL;
UPDATE documents SET summary = contract_name WHERE summary IS NULL AND contract_name IS NOT NULL;
UPDATE documents SET primary_date = sign_date WHERE primary_date IS NULL;
UPDATE documents SET primary_amount_cents = amount_cents WHERE primary_amount_cents IS NULL;

CREATE INDEX idx_doc_type         ON documents(doc_type);
CREATE INDEX idx_doc_primary_date ON documents(primary_date);
CREATE INDEX idx_doc_primary_amt  ON documents(primary_amount_cents);

INSERT INTO schema_version(version, applied_at)
  VALUES(3, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
