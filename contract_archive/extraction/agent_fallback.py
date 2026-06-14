"""low_confidence 长尾的 agent 兜底接口（本期 no-op）。

确定性流水线（页级分流 + 多源融合）是主体，能覆盖绝大多数文档。融合后整体置信度低于阈值
的少数长尾——表格极乱、多份夹页、矛盾无法据图判定——留这个口子给未来接 agentic 兜底
（如 pi coding agent 反复看图/查证）。本期**只记日志、标记，不接任何 agent**：等私有评测
数据攒够、看清长尾占比与形态，再决定接什么、怎么接。届时只改这一个函数，主体流水线不动。
"""
from __future__ import annotations

import logging

from ..schemas import DocumentExtraction

logger = logging.getLogger(__name__)


def escalate_low_confidence(
    extraction: DocumentExtraction, *, source_pdf: str | None = None
) -> DocumentExtraction:
    """标记并放行一份低置信文档（no-op）。原样返回 extraction，不改抽取结果。

    未来在此插入 agentic 兜底：拿到 extraction（含 field_verdicts/低置信项）与 source_pdf，
    可反复看图核查、改写 verdict 后回填。当前仅落一条日志，便于事后统计长尾占比/形态。
    """
    low_keys = [v.key for v in extraction.field_verdicts if v.low_confidence]
    logger.info(
        "[agent-fallback] 低置信文档（overall=%s, pdf=%s）；低置信概念=%s；本期 no-op 放行",
        extraction.fusion_overall_confidence,
        source_pdf or "?",
        low_keys or "（无逐项标记，仅整体偏低）",
    )
    return extraction
