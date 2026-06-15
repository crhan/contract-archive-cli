# 多源融合提取改进：实现状态与验收清单

> 本文档原本是一份“待实现需求 prompt”。截至当前代码状态，核心问题是真实的，改造也大部分已经落地。
> 后续不要按旧需求从头重做；应按本文的剩余验收项收尾。

## 结论

这项需求不是伪需求。原始问题成立：

- 单源文本层 / OCR 二选一会在混合版式 PDF 中丢页或丢表格结构。
- 纯文本抽取对保险计划表等复杂版式不稳。
- VL 看图能补表格，但也会混淆投保人 / 被保险人等字段。
- 高价值字段需要多源候选、冲突评判、置信度和低置信兜底。

但当前仓库已经完成阶段 1-3 的主要实现。剩余工作不是继续扩功能，而是完成真实数据验收：

1. 私有 evalset 的 6 个真实 PDF case 仍标为 DRAFT，需要人工核对定稿 gold。
2. 定稿后跑端到端 PDF eval gate，并保存结果。
3. runtime agent 兜底目前只有 no-op 接口，是否继续做要等低置信比例评估后再决定。

## 当前实现状态

| 项 | 状态 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| 页级分流 | 已完成 | `contract_archive/utils/page_router.py`, `contract_archive/pipelines/ocr_pipeline.py` | 逐页判 text / ocr；文本页用原生文本，扫描或表格页走 VL OCR，再按页序拼回。 |
| VL 看图抽字段 | 已完成 | `contract_archive/extraction/vl_extract.py` | 对关键页图按高价值字段定义抽候选，产出 `FieldCandidate(source="vision")`。 |
| 文本路抽字段 | 已完成 | `contract_archive/extraction/text_fields.py` | 从全文按同一组字段定义抽候选，产出 `FieldCandidate(source="text")`。 |
| 多源融合与评判 | 已完成 | `contract_archive/extraction/fusion.py` | A 文本 / C 看图两路候选一致则直接采信；冲突时据原图 LLM 评判。 |
| sidecar 输出 | 已完成 | `contract_archive/schemas/document.py` | 融合结果只写 `field_verdicts` / `fusion_overall_confidence`，不回写 `amounts` / `fields`。 |
| 保险类型接线 | 已完成 | `contract_archive/extraction/doc_type_handlers.py`, `contract_archive/extraction/insurance_extractor.py` | `保险凭证` 启用 vision fusion，并用独立概念键区分一般医疗、特定医疗、重疾、身故、被保险人等。 |
| ingest 编排 | 已完成 | `contract_archive/archive/ingest.py` | 入库和 `run_full_extraction` 都会按 doc_type handler 决定是否运行融合。 |
| 低置信 agent 兜底 | 延后 | `contract_archive/extraction/agent_fallback.py` | 当前是 no-op 标记接口；未来如果低置信长尾值得处理，再在这里接 runtime agent。 |
| 私有真实 PDF gate | 未完成 | `~/project/contract-archive-evalset/dataset` | 6 个真实 case 已存在，但 `meta.json` 仍标 DRAFT，未看到最终 gate 结果。 |

## 已覆盖的自动化测试

当前主仓库已有单测覆盖核心实现：

- `tests/test_page_router.py`
- `tests/test_ocr_pipeline.py`
- `tests/test_vl_extract.py`
- `tests/test_fusion.py`
- `tests/test_vision_fusion_integration.py`
- `tests/test_ingest_vision_fusion.py`
- `tests/test_evals_score.py`

最近一次本地完整测试：

```bash
uv run --no-sync python -m pytest
```

结果：`327 passed`。

## 真实样本与剩余验收

私有 evalset 中已有 6 个真实 PDF case：

| sha 前 12 | 原 doc id | 角色 |
| --- | --- | --- |
| `92ea0c803d6c` | 33 | 平安互联网长期(B)医疗，复杂表格，验证医疗保额 / 年度限额 / 免赔 / 赔付比例。 |
| `b78258551633` | 34 | 安盛卓越优选意外计划一，验证身故保额与凭证字段。 |
| `d0abccde2524` | 35 | 平安小玩童7号，验证被保险人是陈意而不是投保人陈若涵。 |
| `a6d60676d17c` | 36 | 安盛卓越优选意外计划四，验证身故保额与凭证字段。 |
| `eeb92d6e8ecf` | 31 | 瑞众常青树两全 + 附加重疾，扫描件夹文本页。 |
| `7504cdcdd4f7` | 32 | 瑞众护身福两全，扫描件夹文本页。 |

这些 case 的 `source.pdf` / `input.txt` / `gold.json` 已在：

```text
~/project/contract-archive-evalset/dataset/extraction/<sha前12>/
```

但当前 `meta.json` 仍写着：

```text
make_gold champion 抽取 DRAFT；私有数据集不脱敏；需人工核对后定稿
```

因此不能把“真实 PDF 全链路 gate 已过”算作完成。

## 完成定义

后续要把这项需求关掉，必须满足下面条件。

1. 人工核对并定稿私有 evalset 的 6 个 `gold.json`。
2. 重点核对 `field_verdicts`，不要只看 `amounts` / `fields`：
   - doc 33：一般医疗 200万、特定医疗 200万、重疾 400万、年度限额 400万、保证续保期总限额 800万、免赔额 1万、等待期 90天、保证续保 20年、社保内 100%、社保外 60%、除外乳腺疾病及并发症。
   - doc 35：被保险人必须是陈意，不是投保人陈若涵；意外身故 / 残疾 / 医疗 / 住院津贴 / 飞机意外等字段按原文核对。
   - doc 34 / 36：身故 / 残疾保额、应税薪金门槛、潜水除外等字段按原文核对。
   - doc 31 / 32：确认扫描件不能只依赖少量文本层，端到端 OCR / 混合提取不丢关键字段。
3. 跑端到端评测：

```bash
CONTRACT_ARCHIVE_EVALSET_DIR=~/project/contract-archive-evalset/dataset \
  uv run --no-sync python -m evals.run --models qwen3.7-max --out evals/results/fusion_gate

uv run --no-sync python -m evals.report evals/results/fusion_gate
```

4. gate 报告通过，并将结果目录保留用于追溯。
5. 如果仍存在低置信字段，统计低置信占比和类型，再决定是否实现 runtime agent 兜底。

## TDD / 回归策略

改融合逻辑、字段定义、页路由、OCR pipeline 或 prompt 时，优先按这个顺序工作：

1. 先补或更新单测，复现具体失败：
   - 分流问题放 `tests/test_page_router.py` / `tests/test_ocr_pipeline.py`。
   - 候选抽取问题放 `tests/test_vl_extract.py` / `tests/test_vision_fusion_integration.py`。
   - 融合裁决问题放 `tests/test_fusion.py`。
   - ingest 接线问题放 `tests/test_ingest_vision_fusion.py`。
   - gate 失明问题放 `tests/test_evals_score.py`。
2. 再改实现。
3. 跑相关单测。
4. 如果改动影响抽取 prompt、模型、页路由或融合策略，必须跑私有 PDF eval gate。

## 不要做的事

- 不要再把这项需求理解为“当前架构仍是整份二选一 + 纯文本抽取”。这已经过期。
- 不要把融合结论回写到 `amounts` / `fields`；只写 sidecar。
- 不要把保险保额合成一个大字段；一般医疗、特定医疗、重疾、身故、年度限额必须独立概念键。
- 不要把 runtime agent 兜底提前做成复杂系统；先用 gate 和低置信统计证明有必要。
- 不要把私有 evalset 的真实数据复制进公开主仓库。
