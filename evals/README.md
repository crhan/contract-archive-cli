# evals — 合同抽取换模型评测脚手架

判断能否用更便宜/更快的模型替换 `qwen3.7-max`，覆盖**文本抽取线**和 **VL 签章线**。
评测调项目自己的 `extract_document()` / `check_seals_on_images()` **整条链路**，对产出的
`DocumentExtraction` 逐字段打分——测的就是生产实际会产出什么。

> 开源测评栈调研、为何薄自建、候选模型对比、方法学引用，全在 **[RESEARCH.md](./RESEARCH.md)**。

## 目录

```
evals/
  RESEARCH.md     调研：开源框架横评 + 候选模型对比 + 方法学
  README.md       本文件：怎么跑 / 怎么加 case / 决策框架
  score.py        确定性逐字段打分（纯函数，可单测）
  run.py          跑 cases×模型（抽取线）
  report.py       gate 决策表（抽取线）
  seal.py         VL 签章线（gen/run/report/demo 一体）
  demo.py         无 key 演示（产 sample_report.md）
  sample_report.md / sample_seal_report.md   演示产物
  cases/extraction/<id>/{input.txt, gold.json, meta.json}
  cases/seal/<id>/{page_*.png, gold.json, meta.json[, private/]}
  results/        每次跑生成（已 gitignore）
```

## 快速开始

```bash
# 0) 看报告长什么样（无需 API key）
uv run --no-sync python -m evals.demo
uv run --no-sync python -m evals.seal demo

# 1) 真实跑（需 DASHSCOPE_API_KEY，走 .env / config）
uv run --no-sync python -m evals.run --models qwen3.7-max,qwen-plus,qwen-flash --suite extraction
uv run --no-sync python -m evals.report evals/results/<时间戳>

# 2) 签章线
uv run --no-sync python -m evals.seal gen                  # 合成 plumbing 图（首次）
uv run --no-sync python -m evals.seal run --models qwen-vl-max,qwen3-vl-flash
uv run --no-sync python -m evals.seal report evals/results/<seal_时间戳>
```

测试用 `uv run --no-sync python -m pytest`（见仓库 memory：`uv run pytest` 会落到 pyenv shim）。

## Step-0：先零脚手架冒烟，再做量化

别一上来就为可能根本不合格的模型搭打分系统。生产代码已支持换模型，先肉眼筛一遍：

```bash
DASHSCOPE_LLM_MODEL=qwen-plus contract-archive extract <已入库doc_id>   # 重抽
contract-archive show <doc_id>                                          # 肉眼对比 champion
```

连合法 JSON 都吐不出、doc_type 都判错的候选，到这步就该淘汰，不必进 harness。

## 决策框架：默认不可替换，候选须逐项非劣

替换是**风险问题**，不是精度问题。**不用"加权平均比大小"**——平均会抹平关键文档的塌方
（补充协议那格 F1 从 0.9 掉到 0.5，占比小则总分只动一两个点，你看不见）。

`report.py` 跑**多重门禁，全过才放行**（`score.py` 顶部 `DELTA`/`PARSE_FLOOR` 可调）：

1. **JSON 解析成功率 ≥ 98%**——便宜模型最常见退化是吐非法 JSON/枚举越界/嵌套崩，一条失败=整篇归零。
2. **签章缺陷召回 ≥ champion − δ**——漏报签章=合同蒙混过关，最致命（硬门槛，不进平均）。
3. **每个关键字段**（doc_type / parties / primary_amount / completeness_issues）：
   候选 per-case 指标 **bootstrap CI 下界 ≥ champion 均值 − δ**。
4. 以上全过 → **ELIGIBLE，才比成本/延迟**（约束优化：质量是闸门、成本是目标，不硬凑性价比综合分）。

> "未检测到下降" ≠ "可替换"——小样本 CI 很宽是 power 不足。举证责任在候选：证明不了非劣就是不过。

## 字段分流表（打分通道）

| 字段 | 通道 | 说明 |
|---|---|---|
| doc_type | 分类 | 正确/错误 + 误分类列表 |
| title / summary | 弱信号(+judge) | 归一化精确匹配；质量评判留 Phase 3 窄 judge |
| parties | 集合对齐 | 归一化串相等→P/R/F1 |
| primary_date / key_dates | 归一化日期 | 复用生产 normalize_date 比值 |
| primary_amount / amounts | **exact** | 数值+is_total_component 整体匹配，错一个算 FN+FP |
| fields | 集合对齐 | 按 label 配对，value 归一化比 |
| seals | 集合对齐 | owner+raw_text 模糊匹配（OCR 残缺） |
| obligations | 集合对齐 | actor 相等+action 相似 |
| sub_agreements | 递归对齐 | title 配对 + sign_date/印章数核对 |
| completeness issues | **F-beta(β=2)** | 偏召回；对齐 key=(category,页码)；签章召回单设硬门槛 |

## 怎么加 case（关键：避免偏袒 champion + PII 红线）

1. **PII 红线**：committed case 一律合成匿名（张三/李四/示例置业占位）。`input/` 里的示例苑等
   真实合同**绝不能**做 committed gold。真实脱敏样例放 `cases/<suite>/<id>/private/`（已 gitignore）。
2. **gold 不要用 champion 单源生成**——会继承 champion 盲区（它漏抽的，人工跟着漏，候选抽到反被判 FP）。
   - 高风险字段（parties / amounts / issues）**从原文盲标**，不看任何模型输出。
     用 **`python -m evals.review <case_id>`**：只显示 input.txt + 空白模板，填完
     `--diff` 自动列出"你标了 gold 没有 / gold 有你没标"，想偷看 gold 都看不到。
   - 要 draft 就用**两个异家族**模型各跑一遍取**并集**喂人工修正（make_gold 的 `--crosscheck`）。
   - judge 评分时候选与 champion 输出**匿名+随机左右顺序**（盲测）。
3. **分层 + 过采样**：`meta.json` 标 `stratum`/`difficulty`。稀有但致命的层（含补充协议/多落款/
   留白多选一）天然少，要**主动过采样**（每层 ≥ ~100），单独报指标，别被主分布淹没。
4. **样本量**：bootstrap CI 非劣检验，每分层单元约需 80-150 例，6 类×难度档总量 oom 在 1000-2000 例。
   当前种子仅几例，**只够验证 harness 跑通**，不够下替换结论。

## 用 make_gold 从真实合同批量起草（省人工）

`evals.make_gold` 把**已入库**的真实合同（archive 的 mineru 文本 + extraction_result.json）
转成**脱敏 draft gold**，落到 gitignored 的 `evals/cases_private/extraction/<id>/`：

```bash
uv run --no-sync python -m evals.make_gold                          # 全部已入库文档
uv run --no-sync python -m evals.make_gold --doc-id <id>            # 单个
uv run --no-sync python -m evals.make_gold --crosscheck deepseek-v4-pro  # 加异家族交叉抽取
```

- **数据源**：复用 `_load_document_text`（= 生产喂 extract_document 的同一文本），免重新 OCR/LLM。
- **脱敏**：LLM 为主（识别上下文人名/中英文机构名/各类号码）+ 正则兜底（身份证/手机/座机/邮编）。
  每份再跑残留启发式扫描，把可疑 token 写进 `SCAN.txt`。
- **脱敏不保证完整**（实测仍可能漏 14 位号码片段、罕见写法）。所以产物是 **DRAFT、只进 gitignored
  `cases_private/`**，每份带 `REVIEW.md`：人工①通读确认无残留真实 PII、②对照原文**盲标** parties/
  amounts/issues（破 champion 盲区），确认后才手动复制到可提交的 `cases/extraction/`。
- `--crosscheck <异家族模型>` 在脱敏文本上再抽一遍产 `crosscheck.json`，供人工对比补 champion 漏抽。

## 增强路线（Phase 3，按需建）

- 窄 LLM-judge：仅 title/summary，异家族+blind+rubric+多采样。
- 幻觉对抗负样本：原文故意不含某信息，看模型编不编（捏造金额比漏金额更危险）。
- 脏 OCR 分层：真实 OCR 错字/串行，便宜模型在噪声下退化更快。
- temperature=0 自一致性：`run --repeat N` 已支持收集，可扩展到方差报告。
- 回归门禁：`report` 出 `--gate` 退出码接 CI，防模型/prompt 回退。
