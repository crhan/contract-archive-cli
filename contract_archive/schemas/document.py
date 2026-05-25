"""
统一文档 schema：所有 OCR pipeline 必须把自己的输出归一化到这里。
Schema 的稳定性是这个项目的命脉——后续 compare.py 完全依赖它。

设计原则（Linus 的"好品味"）：
- 字段尽量扁平、不嵌过深
- 缺失字段统一用 None / 空列表，绝不报错
- bbox 坐标统一规约为 PDF 原始 point (1pt = 1/72inch)，render dpi 不影响 schema
- 多 pipeline 通过 pipeline_name 字段区分来源
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

# -------- 共用基本块 --------


class BBox(BaseModel):
    """
    版面坐标框。坐标系：PDF 原始坐标系 (point)，原点左上角，y 向下。
    所有 pipeline 在归一化时必须做坐标换算 (px → pt)，确保 layout.json 可跨 pipeline 比较。
    """

    page: int = Field(..., description="0-based 页码")
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class LayoutBlock(BaseModel):
    """单个版面块——layout.json 的元素。"""

    bbox: BBox
    text: str = ""
    block_type: Literal[
        "title",
        "paragraph",
        "table",
        "figure",
        "header",
        "footer",
        "list",
        "formula",
        "stamp",
        "signature",
        "other",
    ] = "other"
    confidence: Optional[float] = None
    reading_order: Optional[int] = None


# -------- structured.json --------


class TableCell(BaseModel):
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""


class Table(BaseModel):
    """归一化表格结构。同时保留 HTML 与 cell 矩阵，方便不同评估方式。"""

    page: int
    bbox: Optional[BBox] = None
    html: Optional[str] = None  # PP-StructureV3/MinerU 都能直接给
    cells: list[TableCell] = Field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    caption: Optional[str] = None


class Section(BaseModel):
    level: int = 1  # 1=H1, 2=H2 ...
    title: str
    text: str = ""
    page_start: int
    page_end: int


class ExtractedEntity(BaseModel):
    """structured.json 里的通用 entity。专用合同字段走 extraction_result.json。"""

    entity_type: str  # "person" / "org" / "money" / "date" / "address" ...
    text: str
    page: Optional[int] = None
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None


class StructuredDocument(BaseModel):
    """structured.json 主体。"""

    title: Optional[str] = None
    document_type: Optional[str] = None  # "contract" / "invoice" / "report" / ...
    language: str = "zh"
    pages: int = 0
    sections: list[Section] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)


# -------- 顶层 pipeline 输出 --------


class PipelineMeta(BaseModel):
    pipeline_name: Literal["mineru"]
    pipeline_version: str = ""
    model: str = ""
    device: str = "cpu"
    source_pdf: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    notes: str = ""


class PipelineOutput(BaseModel):
    """
    一个 pipeline 跑完一份 PDF 的全部产物（结构化部分）。
    raw_text.txt / markdown.md / preview_images/ 直接写文件，
    layout.json / structured.json / pipeline_meta.json 也直接落盘。
    本对象用于 in-memory 传递和单元测试。
    """

    meta: PipelineMeta
    raw_text: str = ""
    markdown: str = ""
    layout: list[LayoutBlock] = Field(default_factory=list)
    structured: StructuredDocument = Field(default_factory=StructuredDocument)
    preview_image_paths: list[str] = Field(default_factory=list)


# -------- Semantic Extraction --------


class ObligationItem(BaseModel):
    """
    合同义务/动作条款。
    与 risk_clauses 区别：
      - obligation = "X 方应该做什么"（动作 + 截止）
      - risk_clause = "违约后果"（罚则/赔偿/解除条件）
    """

    actor: Literal["party_a", "party_b", "both"]
    action: str                         # "递交审贷资料"
    deadline: Optional[str] = None      # ISO 'YYYY-MM-DD' 或 None
    evidence: str = ""                  # 原文片段


class ContractExtraction(BaseModel):
    """合同语义抽取的统一 schema。所有字段都允许 None（抽不到比硬塞更诚实）。"""

    contract_name: Optional[str] = None
    party_a: Optional[str] = None  # 甲方
    party_b: Optional[str] = None  # 乙方
    amount: Optional[str] = None  # 保留原文（含币种），不强制转 float
    amount_value: Optional[float] = None  # 解析后的数值（人民币元）
    sign_date: Optional[str] = None  # 签订日期 ISO 8601
    expire_date: Optional[str] = None  # 到期/失效日期 ISO 8601
    auto_renewal: Optional[bool] = None
    risk_clauses: list[str] = Field(default_factory=list)
    obligations: list[ObligationItem] = Field(default_factory=list)
    raw_evidence: dict[str, str] = Field(
        default_factory=dict,
        description="字段→原文证据片段，用于人工抽检",
    )


# -------- 通用文档抽取（LLM-first，跨类型） --------
#
# 设计（贴合"LLM-first、少死代码"）：不为每种文档类型写死 pydantic 字段表，
# 而是一个通用信封——可查询的公共核心 + 柔性键值/金额/日期列表。
# 加新文档类型 = 零代码：LLM 自行决定 fields/amounts/key_dates 抽哪些。


class LabeledValue(BaseModel):
    """类型专属字段的通用键值对。"""

    label: str   # "持证人" / "职位" / "身份证号" / "发票号" ...
    value: str   # 原文值（统一用字符串承载；数值/日期另见 amounts/key_dates）


class LabeledAmount(BaseModel):
    """带标签的金额。"""

    label: str                      # "年收入" / "月均收入" / "公积金(个人)" / "合同金额"
    text: str                       # 原文（含大写/币种）
    value: Optional[float] = None   # 归一化数值（人民币元）
    # 是否计入文档主合计：收入证明的"年度税前收入""年度股权收益"=True；
    # "月均收入""公积金"等不该重复累加的=False。供 computed_total_value 求和。
    is_total_component: bool = False
    # 该金额覆盖的时间区间（ISO）。如"上年度""近12个月"由 LLM 据出具日解析为具体起止。
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class LabeledDate(BaseModel):
    """带标签的日期。"""

    label: str                      # "出具日" / "签订日" / "到期日" / "入职日"
    date: Optional[str] = None      # ISO 8601 'YYYY-MM-DD'


class Seal(BaseModel):
    """
    印章（红章）。来源是 MinerU 检测+OCR 的盖章文字，质量参差：
    清晰的能给出主体+章类型（"XX有限公司 合同专用章"），残缺的可能只剩单字。
    不强求拆出编号——OCR 残字硬塞编号是幻觉，宁可只留 raw_text。
    """

    raw_text: str                   # 印章 OCR 原文（可能残缺/乱序），可追溯
    owner: Optional[str] = None     # 盖章主体（公司/机构全称），认不出留 None
    seal_type: Optional[str] = None  # "公章" / "合同专用章" / "财务专用章" / "发票专用章" ...


class CompletenessIssue(BaseModel):
    """单条完整性缺陷（缺签章 / 缺要素）。"""

    item: str                                          # 缺失/异常要素，如"甲方签章""签订日期""转让价款"
    category: Literal["signature", "field"] = "field"  # signature=签章类，field=要素类
    detail: str = ""                                   # 缺什么（简述），如"落款处空白无章"
    # 出处定位：页码 + 原文片段（签章类带落款页码），让人能翻回原文核对。
    # 审计性结论的底线——不可追溯的缺陷不合格，宁可不报。
    evidence: str = ""


class Completeness(BaseModel):
    """
    合同完整性核查（仅合同协议适用，LLM 判定）。

    两类缺陷：
      - signature：落款区应盖章/签字的主体空着（用户最初的痛点：甲方未签章）
      - field：该合同类型应具备的要素缺失（双方/标的/价款/签订日 等）

    两条诚实底线：
      1. 红章 OCR 不可靠——检测不到章可能是真没盖，也可能是淡红/模糊没被识别。
         故缺章一律表述为"疑似"、供人工复核，不作为终判。
      2. 必填要素由 LLM 据合同类型自行判断（车位转让无到期日、框架协议无具体金额
         都属正常），不套死清单——把"本就不该有"的当缺失是误报之源。
    """

    status: Literal["complete", "incomplete", "unknown"] = "unknown"
    issues: list[CompletenessIssue] = Field(default_factory=list)


class SubAgreement(BaseModel):
    """
    文档内的附属协议——主协议之后所附的《补充协议》等。

    依附主协议（修改/补充原协议条款），但常有自己独立的签章落款区与生效条件
    （如"自甲方加盖公章、乙方签字之日生效"），故单列：既体现"这份 PDF 其实含 N 份
    协议"，又让 completeness 能逐个协议单元分别核查签章（主协议缺章 ≠ 补充协议缺章）。
    依附主协议故不单列 amounts/obligations——关键变更写进 summary，可追溯片段进 evidence。
    """

    title: str                       # 协议标题，如 "补充协议" / "补充协议（二）"
    summary: str = ""                # 这份补充协议改了/补充了什么（关键变更，便于检索回忆）
    sign_date: Optional[str] = None  # 该补充协议自己的签订/生效日 ISO（可能空白→None）
    seals: list[Seal] = Field(default_factory=list)  # 补充协议落款上的印章（供完整性核查）
    evidence: str = ""               # 原文关键片段，可追溯


# 粗粒度规范类型（用于 --type 过滤）。LLM 从中择一，更细的归类放进 title/fields。
DOC_TYPES = ("合同协议", "证明", "发票票据", "报告", "证件", "其他")
DocType = str  # 存库用 str（保持柔性，不上 Literal 以免 LLM 新类型被卡死）


class DocumentExtraction(BaseModel):
    """
    通用文档抽取信封：任何文档类型都归一化到这里（LLM-first）。

    公共核心（doc_type/title/summary/primary_*/parties）落 documents 表列、可查询；
    柔性 fields/amounts/key_dates/obligations 整体存 details_json，承载类型专属信息。
    所有字段允许空——抽不到比硬塞更诚实。
    """

    doc_type: str = "其他"                  # 规范类型，取自 DOC_TYPES
    title: Optional[str] = None             # 文档标题/抬头
    summary: Optional[str] = None           # 一句话摘要（可追溯的关键钩子）
    parties: list[str] = Field(default_factory=list)   # 涉及主体（人/机构全称）
    primary_date: Optional[str] = None      # 主日期 ISO（合同=签订日，证明=出具日）
    primary_amount_text: Optional[str] = None
    primary_amount_value: Optional[float] = None
    # 计算值（非抽取）：amounts 中 is_total_component=True 项之和。
    # 例：收入证明 = 年度税前收入 + 年度股权收益。无可累加项则为 None。
    computed_total_value: Optional[float] = None
    key_dates: list[LabeledDate] = Field(default_factory=list)
    amounts: list[LabeledAmount] = Field(default_factory=list)
    seals: list[Seal] = Field(default_factory=list)   # 文档上的印章（有则可验真/索引）
    fields: list[LabeledValue] = Field(default_factory=list)
    obligations: list[ObligationItem] = Field(default_factory=list)
    # 附属协议（主协议之外的《补充协议》等）。一份 PDF 可能含主协议 + N 份补充协议，
    # 每份有独立签章落款；completeness 会逐个协议单元核查。无则空列表。
    sub_agreements: list[SubAgreement] = Field(default_factory=list)
    # 完整性核查：仅合同协议填，其他类型 None
    # （"该不该有甲乙签章/要素齐不齐"对证明/发票无意义，强判只会制造噪声）。
    completeness: Optional[Completeness] = None
    raw_evidence: dict[str, str] = Field(
        default_factory=dict, description="字段→原文证据片段，用于人工抽检"
    )
    # 抽取元数据（非文档内容）：本次实际调用的 LLM 模型名（如 qwen3.7-max）。
    # 随 extraction_result.json 留存，并镜像到 documents.llm_model，供 show 追溯抽取来源。
    # 仅成功调用 LLM 时填；--no-llm / 无 key / 调用失败为 None。
    llm_model: Optional[str] = None


class FieldConfidence(BaseModel):
    """单字段置信度。"""

    value_source: Literal["rule", "llm", "merged", "missing"] = "missing"
    confidence: float = 0.0  # [0, 1]
    rule_hit: bool = False
    llm_agreed: Optional[bool] = None  # None = 未交叉验证


class ExtractionConfidence(BaseModel):
    """extraction_confidence.json 主体。逐字段给出置信度。"""

    contract_name: FieldConfidence = Field(default_factory=FieldConfidence)
    party_a: FieldConfidence = Field(default_factory=FieldConfidence)
    party_b: FieldConfidence = Field(default_factory=FieldConfidence)
    amount: FieldConfidence = Field(default_factory=FieldConfidence)
    sign_date: FieldConfidence = Field(default_factory=FieldConfidence)
    expire_date: FieldConfidence = Field(default_factory=FieldConfidence)
    auto_renewal: FieldConfidence = Field(default_factory=FieldConfidence)
    risk_clauses: FieldConfidence = Field(default_factory=FieldConfidence)
    overall: float = 0.0


# -------- 文件名常量 --------

FILE_RAW_TEXT = "raw_text.txt"
FILE_MARKDOWN = "markdown.md"
FILE_STRUCTURED = "structured.json"
FILE_LAYOUT = "layout.json"
FILE_PIPELINE_META = "pipeline_meta.json"
FILE_EXTRACTION = "extraction_result.json"
FILE_EXTRACTION_CONF = "extraction_confidence.json"
PREVIEW_DIR = "preview_images"
