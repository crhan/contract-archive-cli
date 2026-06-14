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

from ..errors import ErrorInfo

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
    # 页级分流摘要：{"total","text_pages","ocr_pages","table_pages"}。混合提取/原生快路径
    # 才填；MinerU CLI 整份路径留空（其内部分流不归我们统计）。供评测/调试看一份文档怎么被分流。
    page_routing: dict[str, int] = Field(default_factory=dict)


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


class PersonIdentity(BaseModel):
    """
    单个主体（自然人/机构）精确绑定的固有标识。

    与扁平 fields 的区别：fields 是文档级零散键值，常把多人混在一条里
    （如"乙方身份证号: 330106…；420302…"分不清哪个号属于谁）；这里把每个标识
    精确绑定到具体的人/机构，是跨文档身份核对（known_parties 基准库）的基础——
    同一主体的身份证号/电话/银行账号本应稳定，OCR 读错或被改动即可比对告警。
    """

    name: str                                    # 主体名（须与 parties 中某项对应）
    role: Optional[str] = None                   # 本文档中的角色：甲方/乙方/买受人/持证人 等
    # 该主体的固有标识键值：身份证号/电话/银行账号/开户行/统一社会信用代码 等。
    # 复用 LabeledValue（label=标识名，value=标识值）。
    identifiers: list[LabeledValue] = Field(default_factory=list)


class LabeledAmount(BaseModel):
    """带标签的金额。"""

    label: str                      # "年收入" / "月均收入" / "公积金(个人)" / "合同金额"
    text: str                       # 原文（含大写/币种）
    value: Optional[float] = None   # 归一化数值（人民币元；unit 非空时为该单位下的单价数值）
    # 计量单位：None=绝对金额（人民币元，默认，与历史一致）；非 None=单价/费率，
    # value 是「每单位」的数值，量纲见此字段（如"元/月·㎡""元/个/月""元/日"）。
    # 用于区分"合同总价 1228 万元"(unit=None) 与"物业费 2.25 元/月·㎡"(unit 非空)——
    # 后者量纲不同，不可与绝对金额相加，也不参与 computed_total / 金额自洽校验。
    # 同 unit 的周期单价可由代码派生周期费用（如物业费 = Σ按㎡单价 × 建筑面积）。
    unit: Optional[str] = None
    # 是否计入文档主合计：收入证明的"年度税前收入""年度股权收益"=True；
    # "月均收入""公积金"等不该重复累加的=False。供 computed_total_value 求和。
    # 单价项（unit 非空）一律 False——单价不是合同总价的组成部分。
    is_total_component: bool = False
    # 是否为某总价的"分期/部分付款"项（首期款/余款/定金/尾款）。供金额自洽校验：
    # 同一总价的各分期项之和应≈总价(合计)，不符即疑似金额笔误
    # （如车位首期误填 500000 > 总价 200000）。一次性付款、单价(元/月)等非分期项=False。
    is_installment: bool = False
    # 该金额覆盖的时间区间（ISO）。如"上年度""近12个月"由 LLM 据出具日解析为具体起止。
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    # 出处定位：页码 + 原文片段，让人能翻回原文核对这笔金额从哪来（与签章缺陷出处同一原则）。
    evidence: str = ""


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
    # signature=签章类，field=要素缺失类，amount=金额自洽异常类（如分期之和≠总价，代码确定性判出），
    # identity=主体固有标识与基准不符（known_parties 跨文档核对，如身份证号被 OCR 读错/被改）
    category: Literal["signature", "field", "amount", "identity"] = "field"
    detail: str = ""                                   # 缺什么/异常什么（简述），如"落款处空白无章"
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
DOC_TYPES = ("合同协议", "保险凭证", "旅行资料", "证明", "发票票据", "报告", "证件", "其他")
DocType = str  # 存库用 str（保持柔性，不上 Literal 以免 LLM 新类型被卡死）


# -------- 多源融合 sidecar --------
# 命名避开已占用的 FieldConfidence/ExtractionConfidence（那是合同的规则×LLM 交叉验证置信度，
# 语义/字段都不同）。融合是"多抽取源对同一高价值概念给候选 → 评判选定值 + 置信度"。


class FieldCandidate(BaseModel):
    """某一路抽取源对某概念的候选值（融合前的原始提案，留作审计追溯）。"""

    source: str  # 候选来源："text"(文本抽取) | "vision"(看图抽取) | 其他源
    value: str  # 候选值（统一字符串承载）
    evidence: str = ""  # 支持该候选的原文/原图片段
    page: Optional[int] = None  # 看图源的来源页（1-based）；文本源 None


class FieldVerdict(BaseModel):
    """多源融合对某高价值概念的评判结论（**sidecar，绝不回写原字段**）。

    为什么不回写原字段：保额/免赔等概念的原字段（amounts/fields）带着
    evidence/unit/is_total_component 等不变量，与 computed_total_value 的勾稽关系，
    回写会破坏这些不变量。融合结论单独挂这里，消费方按需读，原字段保持自洽。

    每概念一条、概念键独立（一般/特定医疗/重疾各一键），避免不同概念互相覆盖。
    """

    key: str  # 高价值概念键（如 "保额_一般医疗" / "被保险人"），每概念独立
    value: Optional[str] = None  # 评判选定值（抽不到/无定论为 None）
    # 选定值来源："agreed"(各源一致) | "text" | "vision" | "adjudicated"(模型据原图评判)
    source: str = "adjudicated"
    confidence: float = 0.0  # [0,1]
    low_confidence: bool = False  # 置信不足/源间矛盾且依图仍难断 → 标记，供 agent 兜底关注
    rationale: str = ""  # 评判依据（为何选这个值），可追溯
    candidates: list[FieldCandidate] = Field(default_factory=list)  # 喂给评判的全部候选（审计）


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
    # 派生值（非抽取，代码确定性算）：每月物业费估算
    # = Σ(按建筑面积计价的物业类单价，元/月·㎡) × 建筑面积。
    # 合同只给单价（物业服务费 2.25 + 服务费 4.55 + 能耗费 0.8 元/月·㎡），
    # 买受人关心的是月实付额——由代码乘算（按㎡项才并入，车位"元/个/月"等不同量纲不并）。
    # 抽不到单价或建筑面积则为 None。_text 是可追溯的算式说明。
    monthly_property_fee_value: Optional[float] = None
    monthly_property_fee_text: Optional[str] = None
    key_dates: list[LabeledDate] = Field(default_factory=list)
    amounts: list[LabeledAmount] = Field(default_factory=list)
    seals: list[Seal] = Field(default_factory=list)   # 文档上的印章（有则可验真/索引）
    fields: list[LabeledValue] = Field(default_factory=list)
    # 精确绑定到人的固有标识（身份证/电话/银行账号…）。与扁平 fields 互补：
    # fields 易把多人号码混在一条，这里按人拆开，供 known_parties 基准库逐人核对。
    person_identities: list[PersonIdentity] = Field(default_factory=list)
    obligations: list[ObligationItem] = Field(default_factory=list)
    # 附属协议（主协议之外的《补充协议》等）。一份 PDF 可能含主协议 + N 份补充协议，
    # 每份有独立签章落款；completeness 会逐个协议单元核查。无则空列表。
    sub_agreements: list[SubAgreement] = Field(default_factory=list)
    # 完整性核查：仅合同协议填，其他类型 None
    # （"该不该有甲乙签章/要素齐不齐"对证明/发票无意义，强判只会制造噪声）。
    completeness: Optional[Completeness] = None
    # 身份基本信息核对结果（跨文档类型，不限合同）：person_identities 与 known_parties
    # 基准库比对的不一致项（category="identity"）。首见入库不产生 issue，再见冲突才报。
    # 独立于 completeness（后者专司合同签章/要素），避免污染其"仅合同适用"的语义。
    identity_issues: list[CompletenessIssue] = Field(default_factory=list)
    raw_evidence: dict[str, str] = Field(
        default_factory=dict, description="字段→原文证据片段，用于人工抽检"
    )
    # 抽取元数据（非文档内容）：本次实际调用的 LLM 模型名（如 qwen3.7-max）。
    # 随 extraction_result.json 留存，并镜像到 documents.llm_model，供 show 追溯抽取来源。
    # 仅成功调用 LLM 时填；--no-llm / 无 key / 调用失败为 None。
    llm_model: Optional[str] = None
    # 抽取元数据（非文档内容）：本次 LLM 调用的 token 用量（input/output/total_tokens）。
    # 来源 DashScope resp["usage"]；供评测算成本、生产侧成本追踪。读不到 / 未调用为 None。
    llm_usage: Optional[dict] = None
    # 抽取元数据（非文档内容）：本次抽取失败的结构化错误，含 retryable 供 Agent 判重试。
    # 成功 / --no-llm 为 None。随 extraction_result.json 留存，并由 ingest 读出填 IngestResult.error。
    extraction_error: Optional[ErrorInfo] = None
    # 多源融合 sidecar（保险等开启 vision fusion 的类型才填）：高价值概念的逐项评判结论。
    # **只读不回写原字段**——保护 amounts/fields 的 evidence/unit/computed_total 不变量。
    # 随 model_dump_json 进 details_json，零 DB 迁移。未融合为空列表。
    field_verdicts: list[FieldVerdict] = Field(default_factory=list)
    # 融合整体置信度 [0,1]：低于阈值触发 agent_fallback 关注。随 details_json 留存，并镜像到
    # documents.overall_confidence 列（复用现列，零迁移）。未融合为 None。
    fusion_overall_confidence: Optional[float] = None


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
