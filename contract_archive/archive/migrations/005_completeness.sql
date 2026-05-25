-- 档案库 schema v5：合同完整性核查状态（可索引）
--
-- 设计要点（沿用 003/004 的既定取向）：
-- 1. 加法式：只新增一列，不动任何现有列/表，老数据照常工作。
-- 2. 双存合理冗余：完整性详情（status + issues）已随 envelope 整体进 details_json
--    （供 show 展示），这里只把 status 镜像成一列，让 `list --incomplete` 能走
--    WHERE 过滤——details_json 无法高效查询。和 003 的 primary_date 同构。
-- 3. 取值：NULL=未判定/非合同（老数据、证明发票等），'complete'|'incomplete'|'unknown'。
--    不加 CHECK constraint（沿用 v1-v4 理由：未来加值免重建表，校验下沉到 Pydantic/LLM）。
-- 4. 加索引：--incomplete 是等值过滤，索引命中即可；档案库规模虽小，但这是查询
--    入口列，加索引零代价（与 003 给 primary_date 加索引同理）。

ALTER TABLE documents ADD COLUMN completeness_status TEXT;  -- NULL=未判定/非合同, 'complete'|'incomplete'|'unknown'

CREATE INDEX idx_doc_completeness ON documents(completeness_status);

INSERT INTO schema_version(version, applied_at)
  VALUES(5, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
