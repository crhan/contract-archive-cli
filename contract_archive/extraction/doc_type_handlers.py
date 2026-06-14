"""文档类型 → 处理器映射：先识别 doc_type，再按类型走特化抽取/后处理/融合。

取代散落在 ingest 里的 `if doc_type == "合同协议"`：加新类型只需在 DOC_TYPE_HANDLERS
注册一条，不必再翻遍代码找分支。每个 handler 声明：

- specialized_extractor：第二层特化抽取（合同→extract_contract；保险→insurance，见 commit 8）。
  可就地 enrich envelope（如把专属字段合回），返回 (合同抽取, 置信度) 供落库。
- post_processors：**类型专属**后处理（如合同的看落款页签章核查）。通用后处理
  （页码校正 correct_evidence_pages / 身份核对 PartyRegistry）类型无关，留在 ingest，不进 handler。
- enable_vision_fusion：是否开多源融合（保险默认开，见 commit 8）。

注：document_extractor 里"completeness 仅合同""金额自洽校验仅合同"是信封构造的内聚领域逻辑
（completeness 本是合同概念、与 computed_total 勾稽），留在原处，不外移成 post_processor——
硬拆会让 extract_document 自身产出不完整、分裂语义。这里只统一 ingest 层的类型路由。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..schemas import ContractExtraction, DocumentExtraction, ExtractionConfidence
from .contract_extractor import extract_contract
from .vision_seal import augment_completeness_with_vision

# 第二层特化抽取：据全文 + 已抽通用信封 + 是否开 LLM，返回 (合同抽取, 置信度)；可就地 enrich envelope。
SpecializedExtractor = Callable[
    [str, DocumentExtraction, bool], "tuple[ContractExtraction, ExtractionConfidence]"
]
# 类型专属后处理：就地改 envelope，返回是否生效（供日志）。签名对齐 augment_completeness_with_vision。
PostProcessor = Callable[[DocumentExtraction, Path], bool]


@dataclass(frozen=True)
class DocTypeHandler:
    """一个文档类型的处理声明。除 doc_type 外都可缺省（DEFAULT_HANDLER 即全缺省）。"""

    doc_type: str
    specialized_extractor: SpecializedExtractor | None = None
    post_processors: tuple[PostProcessor, ...] = ()
    enable_vision_fusion: bool = False


def _contract_specialized(
    document_text: str, envelope: DocumentExtraction, llm_enabled: bool
) -> tuple[ContractExtraction, ExtractionConfidence]:
    """合同第二层：跑合同特化抽取，把义务/标题合回信封（专属 prompt 对义务/罚则区分更细）。"""
    ext, conf = extract_contract(document_text, llm_enabled=llm_enabled)
    envelope.obligations = ext.obligations
    if not ext.contract_name and envelope.title:
        ext.contract_name = envelope.title
    return ext, conf


DOC_TYPE_HANDLERS: dict[str, DocTypeHandler] = {
    "合同协议": DocTypeHandler(
        "合同协议",
        specialized_extractor=_contract_specialized,
        post_processors=(augment_completeness_with_vision,),
    ),
}

# 未注册类型（证明/发票/旅行/证件/其他…）：无第二层、无专属后处理、无融合，只走通用信封 + 通用后处理。
DEFAULT_HANDLER = DocTypeHandler("其他")


def get_handler(doc_type: str) -> DocTypeHandler:
    """按 doc_type 查处理器；未注册类型回退 DEFAULT_HANDLER。"""
    return DOC_TYPE_HANDLERS.get(doc_type, DEFAULT_HANDLER)
