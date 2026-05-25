"""
通用文档抽取（LLM-first，跨类型）。

与合同专用抽取（自带一份调校过的合同 prompt + ContractExtraction schema）相对，
这里是面向任意类型的通用路径：一次调用完成「判类型 + 抽字段」，
结果归一化到 DocumentExtraction 信封。两者都是纯 LLM（Phase 2 起无 rule）。
死代码 rule 仅保留为确定性数值归一化（中文大写金额→数值、日期→ISO），
不参与字段抽取——加新文档类型只需扩 prompt 里的举例，无需写代码。

设计动机：用户要的是「整理各类文档让其可追溯」，核心吃 LLM 能力，
尽量少依赖死代码规则体系。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..schemas import (
    DOC_TYPES,
    Completeness,
    CompletenessIssue,
    DocumentExtraction,
    LabeledAmount,
    LabeledDate,
    LabeledValue,
    Seal,
    SubAgreement,
)
from ..config import load_settings
from .llm_extractor import _extract_text, _parse_json_loose
from .normalize import coerce_obligations, normalize_date, parse_money_value
from .amount_check import check_amount_consistency

logger = logging.getLogger(__name__)


DOC_EXTRACT_SYSTEM_PROMPT = f"""你是一名严谨的文档档案管理助理。给定一份文档的 OCR 文本，
判断它属于哪类文档，并抽取结构化字段，用于建立可检索、可追溯的个人文档档案库。

铁律：
1. 只输出 JSON，不要任何解释、前缀、Markdown 代码块标记。
2. 抽不到的字段填 null 或空数组，禁止猜测、禁止拼凑。
3. 日期统一 ISO 8601 (YYYY-MM-DD)；占位/空白日期（"___年__月__日"）填 null。
4. 金额保留原文（含大写与币种），不要自己换算成数字。
5. 照实抽取，包括身份证号、电话等个人信息（这是用户本人的私人档案，需完整留存）。

doc_type 从以下规范类型择一（更细的归类写进 title）：
{("、".join(DOC_TYPES))}

JSON 字段定义：
{{
  "doc_type": "上面六类之一",
  "title": "简短但能区分同类文档的标题——务必嵌入关键当事人/标的物/编号，不要只写泛化文档名。如『张三在职收入证明』『XX花园3幢102室商品房认购协议』『18号地下车位转让协议(乙方李四)』",
  "summary": "一句话摘要，含关键主体+金额/数字+日期+标的，便于日后检索回忆与区分同类",
  "parties": ["涉及的人或机构全称", "..."],
  "primary_date": "该文档最重要的日期 ISO（合同=签订日，证明=出具日，发票=开票日）或 null",
  "primary_amount": "该文档最重要的金额原文（合同=合同额，收入证明=年收入）或 null",
  "key_dates": [{{"label": "出具日/签订日/到期日/入职日 等", "date": "YYYY-MM-DD"}}],
  "amounts": [{{"label": "年收入/月均收入/合同金额/首期款/余款 等", "text": "金额原文", "is_total_component": true_or_false, "is_installment": true_or_false, "period_start": "YYYY-MM-DD 或 null", "period_end": "YYYY-MM-DD 或 null", "evidence": "出处：第X页 + 原文片段"}}],
  "fields": [{{"label": "字段名", "value": "字段值"}}],
  "seals": [{{"owner": "盖章主体全称或 null", "seal_type": "公章/合同专用章/财务专用章/发票专用章 等或 null", "raw_text": "印章上识别到的原文"}}],
  "obligations": [
    {{"actor": "party_a|party_b|both", "action": "动宾短语", "deadline": "YYYY-MM-DD 或 null", "evidence": "原文片段"}}
  ],
  "sub_agreements": [
    {{"title": "补充协议", "summary": "改了/补充了什么", "sign_date": "YYYY-MM-DD 或 null", "seals": [{{"owner": "或 null", "seal_type": "或 null", "raw_text": "印章原文"}}], "evidence": "原文片段"}}
  ],
  "completeness": {{
    "status": "complete|incomplete|unknown",
    "issues": [{{"item": "缺失要素名（缺签章请标明所属协议，如 主协议·甲方签章）", "category": "signature|field", "detail": "缺什么", "evidence": "出处：第X页 + 原文留白片段 + 条款号，让人能翻回核对"}}]
  }}
}}

字段抽取要点：
- fields 是该类型专属的键值对，由文档内容自行决定抽哪些。例如：
  · 收入证明 → 持证人、身份证号、用人单位、职位、入职日期、联系人、联系电话、单位地址
  · 发票 → 发票号、税号、开票方、购买方、税额
  · 证件 → 证件号、有效期、签发机关
  把不属于 parties/amounts/key_dates 的有价值信息都放进 fields。
- amounts 列出文档里**所有**金额（不止主金额），各带语义 label。每个金额还需给出：
  · is_total_component：该金额是否计入"文档主合计"。收入证明的【年度税前收入】【年度股权应税收益】
    等一次性年度收入项填 true；【月均收入】【公积金(个人/公司)】等会与年度项重复累加或非收入的填 false。
    宁缺勿错：拿不准一律 false。
  · is_installment：该金额是否为某总价的"分期/部分付款"项（首期款/定金/余款/尾款 等）。
    车位/房屋合同的【首期款】【余款】填 true；一次性付款总额、单价(元/月·个、元/日)、
    服务费、违约金等非分期项填 false。供代码校验"分期之和是否等于总价"以发现金额笔误。
  · evidence：这笔金额的出处——页码(据页脚"第X页共Y页")+ 原文片段，便于翻回原文核对。
  · period_start / period_end：该金额覆盖的时间区间（ISO）。把"上年度""近12个月"等相对表述
    按【出具日】解析成具体起止：
      - "上年度/上一年度" = 上一个完整自然年（出具于 2026 年 → 2025-01-01 ~ 2025-12-31）
      - "本年度/今年" = 当年 1月1日 ~ 出具日
      - "近N个月/过去N个月" = 出具日往前推 N 个月 ~ 出具日
    文档若明写具体起止日期，以原文为准；无区间概念的金额（如合同总额）两者填 null。
- seals 是文档上的印章（红章）。从盖章处识别到的文字（公司名/章类型/编号）填进来：
  · raw_text 照实填识别到的原文，OCR 可能残缺、乱序甚至只剩单字——有什么填什么，不要编造。
  · owner（盖章主体全称）/ seal_type（章类型）能判断就填，拿不准一律 null，禁止猜测编号。
  · 文档若没有任何印章痕迹，seals 填空数组 []。
- obligations 仅当文档含明确"谁该在何时做什么"的待办/义务（合同尤甚）；
  证明、发票等通常为空数组。actor 只能是 party_a|party_b|both。
- sub_agreements 是这份文档里主协议之外的**附属协议**（最常见是《补充协议》，可能多份）。
  很多合同 PDF 在主协议落款后还附了补充协议，它修改/补充原协议（如改期限、改费用承担），
  且通常有自己独立的签章落款区与生效条件。识别与抽取：
  · 触发信号："《XX》补充协议""补充协议""附件协议"等标题，或"鉴于…达成如下补充协议"。
  · 每份填：title（如"补充协议"）；summary（这份改了/补充了什么，一句话）；
    sign_date（该补充协议落款日期，空白填 null）；seals（这份补充协议落款上的章，规则同上层
    seals，没有填 []）；evidence（原文关键片段）。
  · 主协议本身的字段仍填在顶层（parties/amounts/obligations 等），不要塞进 sub_agreements。
  · 没有附属协议就填空数组 []。
- completeness 是合同完整性核查，**仅当 doc_type 为"合同协议"时填**，其他类型一律 null
  （证明/发票没有"甲乙双方签章齐不齐"的概念）。两步判断：
  (1) 先据**这份合同的类型**判断它应具备哪些要素——双方主体、标的物、价款/金额、
      签订日期、双方签章等。要素清单因类型而异，自行判断，**不要套死清单**：
      车位转让/买卖等一次性合同没有到期日属正常，框架协议没有具体金额属正常，
      把"本就不该有"的判成缺失是错误。
  (2) 逐项核查实际是否齐全，缺的或留空白占位的（如"___年__月__日""甲方（盖章）："后空白）
      列进 issues。每条：item=要素名；category=signature(签章/签字类) 或 field(其他要素)；
      detail=缺什么（简述）；evidence=**出处定位**——注明页码（据每页页脚"第X页共Y页"）+
      留白处的原文片段 + 条款号，让人能翻回原文核对。**定位不出出处的缺陷不要报**（宁缺毋滥）。
  (3) 多选一条款不算缺：若某要素是"多选一"（典型：付款方式＝一次性付款 或 银行贷款分期），
      当事人实际选用并填好其中一种即视为完整，**未选用方式的留白是正常的、不算缺失**，
      不要报。只核查当事人实际选用方式内部的留白。
  签章核查要点：本文档每个协议单元——主协议 + 每一份 sub_agreements——都有自己的落款区，
  必须**逐个**核查各自"X方（盖章/签字）："处后面是否有实际印章文字或签名；空着=疑似缺。
  缺章的 issue.item 必须标明所属协议，如"主协议·甲方签章""补充协议·甲方签章"。
  **红章 OCR 经常读不出（淡红/模糊）**，所以凡判定缺章，detail 里务必注明"疑似，可能 OCR
  漏识，需人工复核"——不要把"没读到章"当成"确认没盖章"。
  status：所有协议单元的要素与签章全齐=complete；存在任一缺项=incomplete；信息不足=unknown。
"""


def call_llm_document(
    document_text: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_chars: int = 24000,
) -> dict[str, Any]:
    """
    调 DashScope LLM 做通用文档抽取，返回解析后的 dict（失败返回 {}）。

    与 llm_extractor.call_llm_extract 同样的调用骨架，但用通用文档 prompt。
    刻意不复用其函数体——合同那条路保持不动，避免改动牵连。
    """
    import dashscope  # lazy import

    # 统一从 config 层取（env > 配置文件 > 默认）；显式传参仍优先（param or settings）。
    settings = load_settings()
    model = model or settings.dashscope_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip LLM document extraction")
        return {}

    dashscope.base_http_api_url = base_url

    if len(document_text) > max_chars:
        # 头 1/3 尾 2/3：落款/金额/日期等关键信息多在尾部，权重更高
        head_size = max_chars // 3
        document_text = (
            document_text[:head_size]
            + "\n\n[...省略中段...]\n\n"
            + document_text[-(max_chars - head_size):]
        )

    messages = [
        {"role": "system", "content": DOC_EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"以下是文档正文，请判类型并抽取字段：\n\n{document_text}"},
    ]
    try:
        resp = dashscope.Generation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format="message",
            temperature=0.1,
            top_p=0.5,
            response_format={"type": "json_object"},
        )
    except TypeError:
        # 老版本 SDK 不接受 response_format/temperature
        resp = dashscope.Generation.call(
            api_key=api_key, model=model, messages=messages, result_format="message"
        )
    except Exception as e:
        logger.exception("DashScope document LLM call failed: %s", e)
        return {}

    text = _extract_text(resp)
    if not text:
        logger.warning("LLM empty response (document extract)")
        return {}
    parsed = _parse_json_loose(text)
    if not parsed:
        logger.warning("LLM document response not parseable: %s", text[:200])
    return parsed


def _norm_period(raw: Any) -> Optional[str]:
    """区间端点 → ISO 日期；非字符串/空/无法解析返回 None。"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    return normalize_date(raw.strip())


def _coerce_labeled_amounts(raw: Any) -> list[LabeledAmount]:
    """LLM amounts 数组 → LabeledAmount，顺手算数值、归一化区间日期。跳过非法项。"""
    if not isinstance(raw, list):
        return []
    out: list[LabeledAmount] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("value") or "").strip()
        if not text:
            continue
        label = str(item.get("label", "")).strip() or "金额"
        out.append(LabeledAmount(
            label=label,
            text=text,
            value=parse_money_value(text),
            is_total_component=bool(item.get("is_total_component", False)),
            is_installment=bool(item.get("is_installment", False)),
            period_start=_norm_period(item.get("period_start")),
            period_end=_norm_period(item.get("period_end")),
            evidence=str(item.get("evidence") or "").strip(),
        ))
    return out


def _coerce_labeled_dates(raw: Any) -> list[LabeledDate]:
    """LLM key_dates 数组 → LabeledDate，日期归一化到 ISO。"""
    if not isinstance(raw, list):
        return []
    out: list[LabeledDate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        date = item.get("date")
        date = normalize_date(date) if isinstance(date, str) and date else None
        if not label and not date:
            continue
        out.append(LabeledDate(label=label or "日期", date=date))
    return out


def _coerce_labeled_values(raw: Any) -> list[LabeledValue]:
    """LLM fields 数组 → LabeledValue。跳过空值项。"""
    if not isinstance(raw, list):
        return []
    out: list[LabeledValue] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = item.get("value")
        value = "" if value is None else str(value).strip()
        if not label or not value:
            continue
        out.append(LabeledValue(label=label, value=value))
    return out


def _coerce_parties(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(p).strip() for p in raw if p and str(p).strip()]


def _coerce_seals(raw: Any) -> list[Seal]:
    """LLM seals 数组 → Seal。跳过 raw_text 与 owner 全空的垃圾项。"""
    if not isinstance(raw, list):
        return []
    out: list[Seal] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_text = str(item.get("raw_text") or "").strip()
        owner = str(item.get("owner") or "").strip() or None
        seal_type = str(item.get("seal_type") or "").strip() or None
        if not raw_text and not owner:
            continue
        out.append(Seal(raw_text=raw_text, owner=owner, seal_type=seal_type))
    return out


def _coerce_completeness(raw: Any, doc_type: str) -> Optional[Completeness]:
    """
    LLM completeness 字段 → Completeness。仅合同协议保留；其他类型返回 None
    （即便 LLM 误填了也丢弃，避免给证明/发票安上无意义的"缺签章"）。
    缺字段/非法结构时返回 None，不硬塞。
    """
    if doc_type != "合同协议" or not isinstance(raw, dict):
        return None
    status = str(raw.get("status", "")).strip()
    if status not in ("complete", "incomplete", "unknown"):
        status = "unknown"
    issues: list[CompletenessIssue] = []
    for item in raw.get("issues") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("item") or "").strip()
        if not name:
            continue
        category = str(item.get("category") or "").strip()
        if category not in ("signature", "field"):
            category = "field"
        issues.append(CompletenessIssue(
            item=name,
            category=category,
            detail=str(item.get("detail") or "").strip(),
            evidence=str(item.get("evidence") or "").strip(),
        ))
    # 有缺项却被 LLM 标 complete：以缺项为准纠正（issues 是更硬的证据）。
    if issues and status == "complete":
        status = "incomplete"
    return Completeness(status=status, issues=issues)


def _coerce_sub_agreements(raw: Any) -> list[SubAgreement]:
    """LLM sub_agreements 数组 → SubAgreement。跳过无 title 的垃圾项；seals 复用 _coerce_seals。"""
    if not isinstance(raw, list):
        return []
    out: list[SubAgreement] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        sign_date = item.get("sign_date")
        sign_date = normalize_date(sign_date) if isinstance(sign_date, str) and sign_date else None
        out.append(SubAgreement(
            title=title,
            summary=str(item.get("summary") or "").strip(),
            sign_date=sign_date,
            seals=_coerce_seals(item.get("seals")),
            evidence=str(item.get("evidence") or "").strip(),
        ))
    return out


def extract_document(
    document_text: str,
    llm_enabled: bool = True,
) -> DocumentExtraction:
    """
    通用文档抽取主入口：LLM 判类型 + 抽字段 → DocumentExtraction 信封。

    :param llm_enabled: False（或无 API key）时返回空信封（doc_type 留默认）。
                        通用路径不依赖 rule，关掉 LLM 就没有可抽的东西——诚实返回空。
    """
    if not llm_enabled:
        return DocumentExtraction()

    raw = call_llm_document(document_text)
    if not raw:
        return DocumentExtraction()

    doc_type = str(raw.get("doc_type", "")).strip()
    if doc_type not in DOC_TYPES:
        doc_type = "其他"

    primary_amount_text = (raw.get("primary_amount") or None)
    if isinstance(primary_amount_text, str):
        primary_amount_text = primary_amount_text.strip() or None

    primary_date = raw.get("primary_date")
    primary_date = normalize_date(primary_date) if isinstance(primary_date, str) and primary_date else None

    amounts = _coerce_labeled_amounts(raw.get("amounts"))
    # 计算值（非抽取）：把 LLM 标记为 is_total_component 的金额求和。
    # 由代码做算术（LLM 只负责语义分类），避免 LLM 加法出错。
    components = [a.value for a in amounts if a.is_total_component and a.value is not None]
    computed_total = round(sum(components), 2) if components else None

    # 完整性 = LLM 判的签章/要素 + 代码确定性判的金额自洽异常（分期之和≠总价 等）。
    # 金额异常只对合同挂（completeness 是合同概念）；有异常必判 incomplete。
    # 注：vision_seal.augment 重判签章时保留 category!="signature" 的 issue，amount 类不受影响。
    completeness = _coerce_completeness(raw.get("completeness"), doc_type)
    if doc_type == "合同协议":
        amount_issues = check_amount_consistency(amounts, computed_total)
        if amount_issues:
            base_issues = completeness.issues if completeness else []
            completeness = Completeness(status="incomplete", issues=base_issues + amount_issues)

    return DocumentExtraction(
        doc_type=doc_type,
        title=(str(raw["title"]).strip() if raw.get("title") else None),
        summary=(str(raw["summary"]).strip() if raw.get("summary") else None),
        parties=_coerce_parties(raw.get("parties")),
        primary_date=primary_date,
        primary_amount_text=primary_amount_text,
        primary_amount_value=parse_money_value(primary_amount_text),
        computed_total_value=computed_total,
        key_dates=_coerce_labeled_dates(raw.get("key_dates")),
        amounts=amounts,
        seals=_coerce_seals(raw.get("seals")),
        fields=_coerce_labeled_values(raw.get("fields")),
        obligations=coerce_obligations(raw.get("obligations")),
        sub_agreements=_coerce_sub_agreements(raw.get("sub_agreements")),
        completeness=completeness,
        raw_evidence={},
        # raw 非空=本次确实跑通了 LLM，记下实际模型（与 call_llm_document 内同一 settings）。
        llm_model=load_settings().dashscope_model,
    )
