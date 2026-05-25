"""
合同抽取换模型离线评测脚手架。

目标：判断能否用更便宜/更快的模型替换 qwen3.7-max，覆盖文本抽取线 + VL 签章线，
不掉精度。评测调用项目自己的 extract_document() / check_seals_on_images() 整条链路
（含 prompt + 后处理归一化 + 求和 + completeness 纠正），对产出的 DocumentExtraction
逐字段打分——测的就是生产实际会产出什么。

模块：
- score：确定性逐字段打分（复用生产 normalize；列表贪心对齐→P/R/F1；金额 exact；
  完整性 issues F-beta 偏召回；格式合规）。纯函数，可脱离 API 单测。
- run：跑 cases × 候选模型，计时 + 取 usage，产出落 results/。
- report：聚合成 gate 决策表（默认不可替换，候选须逐项非劣才放行）。

方法学与决策框架见 evals/README.md，开源测评栈调研见 evals/RESEARCH.md。
"""
