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
import re
from typing import Any, Optional

from ..schemas import (
    DOC_TYPES,
    Completeness,
    CompletenessIssue,
    DocumentExtraction,
    LabeledAmount,
    LabeledDate,
    LabeledValue,
    PersonIdentity,
    Seal,
    SubAgreement,
)
from ..config import load_settings
from ..errors import classify_exception, config_missing, extract_empty
from .llm_extractor import (
    LlmResult,
    _call_openai_compat,
    _parse_json_loose,
    _truncate_middle,
)
from .normalize import coerce_obligations, normalize_date, parse_money_value
from .amount_check import check_amount_consistency
from .property_fee import estimate_monthly_property_fee

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
  "key_dates": [{{"label": "出具日/签订日/到期日/入职日 等（用规范名词，见下方约束）", "date": "YYYY-MM-DD"}}],
  "amounts": [{{"label": "年收入/月均收入/合同金额/首期款/余款/物业服务费 等", "text": "金额原文", "unit": "单价量纲或 null（绝对金额填 null；单价/费率填如『元/月·㎡』『元/个/月』『元/日』）", "is_total_component": true_or_false, "is_installment": true_or_false, "period_start": "YYYY-MM-DD 或 null", "period_end": "YYYY-MM-DD 或 null", "evidence": "第X页 + 原文片段"}}],
  "fields": [{{"label": "字段名", "value": "字段值"}}],
  "person_identities": [{{"name": "主体名（须与 parties 对应）", "role": "甲方/乙方/买受人/持证人 等或 null", "identifiers": [{{"label": "身份证号/电话/银行账号/开户行/统一社会信用代码 等", "value": "值"}}]}}],
  "seals": [{{"owner": "盖章主体全称或 null", "seal_type": "公章/合同专用章/财务专用章/发票专用章 等或 null", "raw_text": "印章上识别到的原文"}}],
  "obligations": [
    {{"actor": "party_a|party_b|both", "action": "动宾短语", "deadline": "YYYY-MM-DD 或 null", "evidence": "原文片段"}}
  ],
  "sub_agreements": [
    {{"title": "补充协议", "summary": "改了/补充了什么", "sign_date": "YYYY-MM-DD 或 null", "seals": [{{"owner": "或 null", "seal_type": "或 null", "raw_text": "印章原文"}}], "evidence": "原文片段"}}
  ],
  "completeness": {{
    "status": "complete|incomplete|unknown",
    "issues": [{{"item": "缺失要素名（缺签章请标明所属协议，如 主协议·甲方签章）", "category": "signature|field", "detail": "缺什么", "evidence": "第X页 + 原文留白片段 + 条款号，让人能翻回核对"}}]
  }}
}}

字段抽取要点：
- key_dates label 用**规范名词**，避免同义不同名造成下游检索断链：
  · 签订日 / 出具日 / 开票日 / 入职日 / 起租日 / 到期日 / 解除日；
  · 商品房买卖合同优先用：房屋交付日、首期房价款支付截止日、贷款申请材料提交截止日、
    贷款发放截止日、预售许可证取得日、土地使用权终止日期、抵押登记日期、
    债务履行期限起始日、债务履行期限截止日、不动产登记办理截止日、配套设施竣工验收日。
  · 不要用模糊词如"X 日"代替"X 截止日"——能区分"动作发生日"vs"动作截止日"就尽量区分。
- fields 是该类型专属的键值对，由文档内容自行决定抽哪些。例如：
  · 收入证明 → 持证人、身份证号、用人单位、职位、入职日期、联系人、联系电话、单位地址
  · 发票 → 发票号、税号、开票方、购买方、税额
  · 证件 → 证件号、有效期、签发机关
  · 商品房买卖合同（预售/现售/二手房）→ 凡文档含相应条款的，**必抽**：
    房屋坐落（完整地址）、房屋编号、房屋性质（毛坯/精装/全装修）、房屋类型（住宅/办公等）、
    规划用途、预测建筑面积、套内建筑面积、分摊共有建筑面积、计价方式（按建筑面积/套内/按套）、
    付款方式（一次性/商业贷款/公积金/组合贷）、绿色建筑等级；
    土地用途、土地使用权终止日期、不动产权证号、预售许可证号、不动产单元号；
    抵押状态（抵押中/无抵押）、抵押权人、抵押范围、抵押解除承诺；
    质量担保人及担保范围（按楼幢号分担连带责任的第三方公司）；
    保修期·地基主体结构、保修期·防水/外墙渗漏、保修期·电气管线给排水、保修期·供热供冷；
    前期物业服务企业、物业服务费、服务费、地下车位管理费、能耗费；
    预售资金监管银行、预售资金监管账户、监管机构；
    争议解决方式（仲裁/法院诉讼）、送达方式、合同份数、不动产登记办理期限。
  · 房屋租赁合同 → 租赁标的、房屋用途、租期起止、支付方式、押金、违约金比例、争议解决方式。
  把不属于 parties/amounts/key_dates 的有价值信息都放进 fields。
  **fields 与 key_dates/obligations 的边界**：纯日期点（如"交付日"）放 key_dates；
  "X 方应做 Y" 放 obligations；客观属性、第三方机构名、约定条款值放 fields。
  同一信息（如"房屋交付日"）若已在 obligations.deadline，仍可在 key_dates 里冗余存放——
  方便下游按时间检索。
- person_identities 是 fields 的"精确到人"版：把每个**具体的人/机构**与其固有标识精确绑定。
  fields 里"乙方身份证号: A；B"分不清谁是谁，这里必须按人拆开，供跨文档逐人核对。
  · 每个主体一个对象：name（与 parties 对应的姓名/全称）、role（其在本文档的角色）、
    identifiers（该主体的身份证号/电话/银行账号/开户行/税号等键值，label+value）。
  · name 必须**逐字摘自正文/parties 中的主体全称**，禁止改字、补字、规范化或自行翻译
    （如把『浙典』写成『浙奥』即为幻觉）；正文里找不到的名字一律不得编造。
  · 同一实体只出一个对象：哪怕它在正文有多个称谓（如"出卖人（以下简称甲方）"），
    也只用一个 name（取正文全称）、role 写其主要称谓，**禁止拆成"出卖人|X"和
    "甲方|Y"两条指向同一实体的记录**——拆开会让跨文档核对把一家公司当成两家。
  · 同一文档里多个自然人的身份证、电话**必须分别绑到各自名下，禁止混填或合并**。
    例：买受人张三→身份证A、电话X；李四→身份证B、电话Y，务必拆成两个对象。
  · 只放"主体固有"的稳定标识（身份证/电话/账号/税号），不放金额、日期、地址这类
    随文档变化的信息。一个标识都绑不出则填空数组 []。
- amounts 列出文档里**所有**金额（不止主金额），各带语义 label。每个金额还需给出：
  · unit：计量单位。**绝对金额**（合同总价、首期款、定金、年收入 等一笔确定的钱）填 null；
    **单价/费率**（每单位若干钱）按原文量纲填，如物业费"2.25 元/月·平方米"→ "元/月·㎡"、
    车位"100 元/个/月"→ "元/个/月"、违约金"每日万分之一点五"→ 不是金额不抽。
    商品房合同第物业管理条款的【物业服务费】【服务费】【能耗费】【地下车位管理费】等
    **都要各列一条**并填 unit——下游会按㎡单价 × 建筑面积派生月物业费。
    单价项的 is_total_component 与 is_installment **一律 false**（单价不是总价组成、也非分期）。
  · is_total_component：该金额是否计入"文档主合计"。收入证明的【年度税前收入】【年度股权应税收益】
    等一次性年度收入项填 true；【月均收入】【公积金(个人/公司)】等会与年度项重复累加或非收入的填 false。
    **铁律——合计项之间不可有包含关系，否则重复累加**：若文档已给出"总价款/合同总额"这类
    汇总金额，则**只有该汇总项**标 true，它的各分期子项（首期款/余款/尾款/定金）**一律 false**。
    例：房屋合同 总价款12279889 标 true；首期1849889、余款10430000 标 false（它们是总价的拆分，
    再标 true 会让合计变成 总价+首期+余款＝2×总价）。仅当文档**没有**单一汇总项、只有各独立
    组成项（如收入证明的年度收入+股权收益）时，才让各组成项都标 true。宁缺勿错：拿不准一律 false。
  · is_installment：该金额是否为某总价的"分期/部分付款"项（首期款/余款/尾款 等）。
    车位/房屋合同的【首期款】【余款】填 true；一次性付款总额、单价(元/月·个、元/日)、
    服务费、违约金等非分期项填 false。
    **定金/订金/预付款一律 false**：它通常签约时支付并抵作房款（已含在首期款内或另行抵扣），
    不与首期/余款并列再累加成总价；若误标 true，"分期之和"会虚高于总价、触发假的金额笔误告警
    （仅当合同明确约定定金是首期/余款之外、与之并列累加构成总价的独立一期时才标 true）。
    供代码校验"分期之和是否等于总价"以发现金额笔误。
    注意：标了 is_installment=true 的项，is_total_component 必为 false（分期不入合计，见上）。
  · evidence：这笔金额在原文的定位，页码(据页脚"第X页共Y页")+ 原文片段，便于翻回核对。
    值本身只填"第X页 + 片段"，勿带"出处"二字——展示时会自动加前缀。
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
  · 凡含 "X 方应/应当 在 Y 前 做 Z"、"X 方负责 Z"、"X 方承诺 Z" 的子句**全部抽进 obligations**，
    不要只挑头几个。商品房买卖典型条款（party_a=出卖人，party_b=买受人）：
    party_a 应当在 X 前向买受人交付商品房；party_a 应当退还买受人已付全部房款（解除合同情形）；
    party_a 负责修复房屋质量问题；party_b 应当于 X 前支付首期房价款；
    party_b 应当于 X 前向贷款机构提交贷款申请材料；party_b 自筹资金付清剩余房款；
    party_b 配合办理退房及注销备案手续；party_b 申请办理房屋交易和不动产登记；
    party_b 办理房屋交接手续。
  · deadline：动作的明确截止日期（ISO）。"自 X 日起 N 日内" 解析为 X+N 日；
    没有时间约束的填 null。
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
    max_chars: int = 60000,
) -> LlmResult:
    """
    调 DashScope LLM（OpenAI 兼容口）做通用文档抽取，返回 LlmResult（parsed/model/usage）。

    见 CLAUDE.md：DashScope 一律走兼容口（原生 Generation 不认部分模型 id）。
    用通用文档 prompt，传输/解析复用 llm_extractor 的兼容口 helper。
    失败时 parsed={}（调用方判 `if not res.parsed`），与历史"返回空 dict"语义一致。
    """
    # 统一从 config 层取（env > 配置文件 > 默认）；显式传参仍优先（param or settings）。
    settings = load_settings()
    model = model or settings.dashscope_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip LLM document extraction")
        return LlmResult(
            parsed={}, model=model,
            error=config_missing("DASHSCOPE_API_KEY 缺失，跳过 LLM 文档抽取"),
        )

    user_msg = f"以下是文档正文，请判类型并抽取字段：\n\n{_truncate_middle(document_text, max_chars)}"
    try:
        content, usage = _call_openai_compat(DOC_EXTRACT_SYSTEM_PROMPT, user_msg, model, api_key, base_url)
    except Exception as e:  # noqa: BLE001 — 外部调用降级返回空，但保留结构化 error 供上层判重试
        logger.exception("DashScope document LLM call failed: %s", e)
        return LlmResult(parsed={}, model=model, error=classify_exception(e))

    if not content:
        logger.warning("LLM empty response (document extract)")
        return LlmResult(parsed={}, model=model, usage=None)
    parsed = _parse_json_loose(content)
    if not parsed:
        logger.warning("LLM document response not parseable: %s", content[:200])
    return LlmResult(parsed=parsed, model=model, usage=usage)


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
        unit = str(item.get("unit") or "").strip() or None
        is_installment = bool(item.get("is_installment", False))
        # is_total_component（计入主合计）的两个**代码强制不变量**，纠正 LLM 误标：
        #  1) 单价项（unit 非空，如"2.25 元/月·㎡"）量纲不同，绝不入合计；
        #  2) 分期项（is_installment，如首期/余款/尾款）是某总价的部分付款，不是合计的
        #     独立组成——若与总价同时计入会重复累加（总价12279889 + 首期 + 余款 = 2×总价）。
        # 这两类金额无论 LLM 怎么标，is_total_component 一律压成 False。
        is_total_component = (
            bool(item.get("is_total_component", False))
            and not unit
            and not is_installment
        )
        # parse_money_value 对单价文本（"2.25元/月·㎡"）同样取得首个数值（2.25）。
        out.append(LabeledAmount(
            label=label,
            text=text,
            value=parse_money_value(text),
            unit=unit,
            is_total_component=is_total_component,
            is_installment=is_installment,
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


def _coerce_person_identities(raw: Any) -> list[PersonIdentity]:
    """
    LLM person_identities 数组 → PersonIdentity（精确到人的固有标识）。

    跳过无 name 或无任何有效 identifier 的项——光有名字没标识对核对无意义。
    identifiers 复用 _coerce_labeled_values 的清洗（去空 label/value）。
    """
    if not isinstance(raw, list):
        return []
    out: list[PersonIdentity] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        identifiers = _coerce_labeled_values(item.get("identifiers"))
        if not identifiers:
            continue
        role = str(item.get("role") or "").strip() or None
        out.append(PersonIdentity(name=name, role=role, identifiers=identifiers))
    return out


def _filter_identities_by_text(
    identities: list[PersonIdentity], document_text: str
) -> list[PersonIdentity]:
    """
    丢弃 name 在正文中根本不出现的 person_identity——确定性的幻觉护栏。

    prompt 已要求 name 逐字摘自正文，但 LLM 偶尔仍改字/编造（实测把『浙典』幻觉成
    正文不存在的『浙奥』），使同一实体在 known_parties 分裂。凡正文（去空白后）不含
    该 name 的，判为幻觉丢弃。比较去空白以容忍 OCR 在名字中夹空格。
    注：VL 落款章绑定的主体在抽取之后才追加（owner 未必在正文），不经此过滤。
    """
    if not document_text:
        return identities
    haystack = re.sub(r"\s+", "", document_text)
    kept: list[PersonIdentity] = []
    for person in identities:
        if re.sub(r"\s+", "", person.name) in haystack:
            kept.append(person)
        else:
            logger.info("丢弃疑似幻觉主体（正文未出现该名）: %s", person.name)
    return kept


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
    model: str | None = None,
) -> DocumentExtraction:
    """
    通用文档抽取主入口：LLM 判类型 + 抽字段 → DocumentExtraction 信封。

    :param llm_enabled: False（或无 API key）时返回空信封（doc_type 留默认）。
                        通用路径不依赖 rule，关掉 LLM 就没有可抽的东西——诚实返回空。
    :param model: 覆盖抽取所用 model（默认 None=走 settings.dashscope_model）。
                  评测换模型的唯一入口——实际跑的 model 即 res.model，回填 llm_model
                  保证"记录的模型=实际跑的模型"（单一真相源，不再二次读 settings）。
    """
    if not llm_enabled:
        return DocumentExtraction()

    res = call_llm_document(document_text, model=model)
    raw = res.parsed
    if not raw:
        # 抽取为空：**不设 llm_model（保持 None）**——schema 定义"调用失败 llm_model 为 None"，
        # 且 evals 的 parse_ok 一票否决依赖它（llm_model 非 None = 调用/解析成功）；这里若设非 None，
        # 会让产不出 JSON 的劣质模型在换模型评测里蒙混过 parse_ok gate（见 MEMORY「评测报告撒谎坑」）。
        # 只带结构化 error（缺 key→CONFIG_MISSING / API 异常分类 / 空 JSON→EXTRACT_EMPTY），供 ingest 判重试。
        return DocumentExtraction(
            extraction_error=res.error or extract_empty("LLM 返回空或无法解析为 JSON"),
        )

    doc_type = str(raw.get("doc_type", "")).strip()
    if doc_type not in DOC_TYPES:
        doc_type = "其他"

    primary_amount_text = (raw.get("primary_amount") or None)
    if isinstance(primary_amount_text, str):
        primary_amount_text = primary_amount_text.strip() or None

    primary_date = raw.get("primary_date")
    primary_date = normalize_date(primary_date) if isinstance(primary_date, str) and primary_date else None

    # person_identities 过幻觉护栏：name 须在正文出现，编造名（正文不存在）丢弃。
    person_identities = _filter_identities_by_text(
        _coerce_person_identities(raw.get("person_identities")), document_text
    )

    amounts = _coerce_labeled_amounts(raw.get("amounts"))
    # 计算值（非抽取）：把 LLM 标记为 is_total_component 的金额求和。
    # 由代码做算术（LLM 只负责语义分类），避免 LLM 加法出错。
    components = [a.value for a in amounts if a.is_total_component and a.value is not None]
    computed_total = round(sum(components), 2) if components else None

    # 派生值（非抽取）：月物业费 = Σ按㎡单价 × 建筑面积。同由代码乘算，LLM 只抽单价。
    fields = _coerce_labeled_values(raw.get("fields"))
    monthly_fee_value, monthly_fee_text = estimate_monthly_property_fee(amounts, fields)

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
        monthly_property_fee_value=monthly_fee_value,
        monthly_property_fee_text=monthly_fee_text,
        key_dates=_coerce_labeled_dates(raw.get("key_dates")),
        amounts=amounts,
        seals=_coerce_seals(raw.get("seals")),
        fields=fields,
        person_identities=person_identities,
        obligations=coerce_obligations(raw.get("obligations")),
        sub_agreements=_coerce_sub_agreements(raw.get("sub_agreements")),
        completeness=completeness,
        raw_evidence={},
        # 单一真相源：记下本次调用实际请求的 model（res.model），而非二次读 settings——
        # 后者在评测换模型时会张冠李戴（记 qwen3.7-max 实跑 qwen-plus）。
        llm_model=res.model,
        # token 用量（评测算成本的旁证）；生产侧也可用于成本追踪。读不到为 None。
        llm_usage=res.usage,
        # 结构化错误：成功为 None；用于 ingest 失败诊断与 Agent 重试决策。
        extraction_error=res.error,
    )
