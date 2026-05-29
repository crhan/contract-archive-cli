"""
CLI 渲染层：把 DocumentRow / IngestResult 等数据对象格式化成展示字符串 / JSON dict / rich Table。

这里只放与 typer/console 无关的纯函数（输入数据对象，输出字符串/dict/Table 对象，
不碰 stdout——打印交给 cli.py 的 console），便于单测、也让 cli.py 专注命令定义与参数解析。
函数对入参做鸭子类型，不依赖具体 model 类型，避免反向 import。
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from rich.table import Table


def status_color(s: str) -> str:
    """status 着色：ok 绿 / partial 黄 / failed 红。"""
    color = {"ok": "green", "partial": "yellow", "failed": "red"}.get(s, "white")
    return f"[{color}]{s}[/{color}]"


def subject_of(r) -> str:
    """list 用的『主体』列：优先信封 parties，回退合同甲乙方。截断防撑宽。"""
    parties = r.details().get("parties") or []
    if not parties:
        parties = [p for p in (r.party_a, r.party_b) if p]
    if not parties:
        return "-"
    s = "、".join(parties[:2])
    return s if len(s) <= 20 else s[:19] + "…"


def display_amount(r) -> str:
    """
    list 金额列：有计算合计（computed_total_value）优先显示并标 *，
    否则回退抽取的主金额（primary_amount_value），都没有则 '-'。
    """
    total = r.details().get("computed_total_value")
    if isinstance(total, (int, float)):
        return f"¥{total:,.0f}[cyan]*[/cyan]"
    if r.primary_amount_value is not None:
        return f"¥{r.primary_amount_value:,.0f}"
    return "-"


def completeness_mark(r) -> str:
    """
    list『完整』列：仅合同有值。疑似缺红色警示，其余从简。
    'incomplete'→红警；'complete'→绿勾；'unknown'→黄问号；None（非合同/未判）→灰横。
    """
    s = getattr(r, "completeness_status", None)
    return {
        "incomplete": "[red]⚠ 疑似缺[/red]",
        "complete": "[green]✓[/green]",
        "unknown": "[yellow]?[/yellow]",
    }.get(s, "[dim]-[/dim]")


def period_str(a: dict) -> str:
    """金额覆盖区间的展示标注，如 ' [2025-01-01~2025-12-31]'；无区间返回空串。"""
    start, end = a.get("period_start"), a.get("period_end")
    if not start and not end:
        return ""
    return f" [dim][{start or '?'}~{end or '?'}][/dim]"


def local_time(iso_utc: Optional[str]) -> str:
    """
    入库时间 UTC ISO（'2026-05-24T23:05:04Z'）→ 本地时区展示串。
    存储保持 UTC（可移植、可比较），仅展示时转本地。解析失败则原样返回。
    """
    if not iso_utc:
        return "-"
    try:
        dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return iso_utc


def ingest_result_to_dict(r) -> dict:
    """IngestResult → JSON 友好 dict（pdf_path 转字符串）。"""
    return {
        "pdf_path": str(r.pdf_path),
        "sha256": r.sha256,
        "status": r.status,
        "doc_id": r.doc_id,
        "mineru_duration_s": r.mineru_duration_s,
        "llm_duration_s": r.llm_duration_s,
        "error_message": r.error_message,
        # 结构化错误（code/category/retryable）；成功/跳过为 None。供 Agent 判可否重试。
        "error": r.error.model_dump() if r.error else None,
        "skipped_reason": r.skipped_reason,
    }


def seal_rows_to_dict(rows) -> list[dict]:
    """SealRow 列表 → JSON 友好 dict 列表（seals --format json 用）。"""
    return [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "owner": r.owner,
            "seal_type": r.seal_type,
            "raw_text": r.raw_text,
        }
        for r in rows
    ]


def row_to_dict(r) -> dict:
    """DocumentRow → JSON 友好 dict（list/search/show 的 --format json 用）。"""
    details = r.details()
    return {
        "id": r.id,
        "sha256": r.sha256,
        "status": r.status,
        "doc_type": r.doc_type,
        "title": r.title,
        "summary": r.summary,
        "primary_date": r.primary_date,
        "primary_amount_value": r.primary_amount_value,
        "computed_total_value": details.get("computed_total_value"),
        "seals": details.get("seals"),
        "sub_agreements": details.get("sub_agreements"),
        "completeness": details.get("completeness"),
        "completeness_status": r.completeness_status,
        "llm_model": details.get("llm_model"),
        "details": details,
        "contract_name": r.contract_name,
        "party_a": r.party_a,
        "party_b": r.party_b,
        "amount_text": r.amount_text,
        "amount_value": r.amount_value,
        "sign_date": r.sign_date,
        "expire_date": r.expire_date,
        "auto_renewal": bool(r.auto_renewal) if r.auto_renewal is not None else None,
        "risk_clauses": r.risk_clauses,
        "obligations": [
            {"actor": o.actor, "action": o.action,
             "deadline": o.deadline, "evidence": o.evidence}
            for o in r.obligations
        ],
        "overall_confidence": r.overall_confidence,
        "source_path": r.source_path,
        "output_dir": r.output_dir,
        "ingested_at": r.ingested_at,
    }


def build_show_table(row) -> Table:
    """
    show 命令的单文档详情表（纯函数：输入 DocumentRow，输出 rich Table，不打印）。

    从 cli.py 下沉至此让 cli.py 专注命令定义；按展示段拆成多个 _show_*_rows helper
    （守项目 50 行/函数铁律）。逻辑与原 show 命令的表格构建逐行等价。
    """
    table = Table(title=f"Document #{row.id} · {row.doc_type or '?'} ({status_color(row.status)})")
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    det = row.details()
    _show_header_rows(table, row, det)
    _show_amount_rows(table, row, det)
    _show_identity_rows(table, det)
    _show_seal_rows(table, row)
    _show_completeness_rows(table, row)
    _show_footer_rows(table, row)
    return table


def _show_header_rows(table: Table, row, det: dict) -> None:
    """元信息 + 通用信封 + 合同专属列（party/到期/续约）或非合同的主体/日期。"""
    table.add_row("sha256", row.sha256)
    table.add_row("source_path", row.source_path)
    table.add_row("output_dir", row.output_dir)
    table.add_row("ingested_at", local_time(row.ingested_at))
    # mineru_s/llm_s（执行耗时）是运维遥测，不属于档案内容——不在 show 展示。
    if row.error_message:
        table.add_row("[red]error[/red]", row.error_message)

    table.add_row("", "")
    table.add_row("[bold]doc_type[/bold]", row.doc_type or "-")
    table.add_row("[bold]title[/bold]", row.title or row.contract_name or "-")
    if row.summary:
        table.add_row("summary", row.summary)

    # 合同有专属列（party/到期/续约），日期走表列；其余类型走 details 的主体/日期。
    is_contract = bool(row.contract_name or row.party_a or row.party_b)
    if is_contract:
        table.add_row("", "")
        table.add_row("party_a", row.party_a or "-")
        table.add_row("party_b", row.party_b or "-")
        table.add_row("sign_date", row.sign_date or "-")
        table.add_row("expire_date", row.expire_date or "-")
        table.add_row(
            "auto_renewal",
            "是" if row.auto_renewal == 1 else ("否" if row.auto_renewal == 0 else "-"),
        )
    else:
        parties = det.get("parties") or []
        if parties:
            table.add_row("主体", "\n".join(f"• {p}" for p in parties))
        key_dates = det.get("key_dates") or []
        if key_dates:
            table.add_row(
                "日期",
                "\n".join(f"• {d.get('label', '')}: {d.get('date') or '-'}" for d in key_dates),
            )


def _show_amount_rows(table: Table, row, det: dict) -> None:
    """金额明细 / 计算合计 / 类型专属字段（所有文档类型通用）。"""
    amounts = det.get("amounts") or []
    if amounts:
        lines = []
        for a in amounts:
            v = a.get("value")
            unit = a.get("unit")
            if unit:  # 单价项：显示量纲（如 2.25 元/月·㎡），不套 ¥（非绝对金额）
                vs = f"（{v:g} {unit}）" if isinstance(v, (int, float)) else ""
            else:
                vs = f"（¥{v:,.2f}）" if isinstance(v, (int, float)) else ""
            mark = " [cyan]✓计入合计[/cyan]" if a.get("is_total_component") else ""
            if a.get("is_installment"):
                mark += " [magenta]分期[/magenta]"
            lines.append(
                f"• {a.get('label', '')}: {a.get('text', '')}{vs}{period_str(a)}{mark}"
            )
            ev = a.get("evidence") or ""
            if ev:
                lines.append(f"    [dim]↳ 出处：{ev}[/dim]")
        table.add_row("金额", "\n".join(lines))
    elif row.amount_text:  # details 无 amounts 的旧数据/回退，至少显示表列主金额
        table.add_row(
            "金额",
            f"{row.amount_text} (¥{row.amount_value:,.2f})" if row.amount_value is not None else row.amount_text,
        )
    total = det.get("computed_total_value")
    if isinstance(total, (int, float)):
        table.add_row(
            "[bold]合计(计算)[/bold]",
            f"[cyan]¥{total:,.2f}[/cyan] [dim](上方标✓项之和，非抽取值)[/dim]",
        )
    mfee = det.get("monthly_property_fee_value")
    if isinstance(mfee, (int, float)):
        mfee_text = det.get("monthly_property_fee_text") or ""
        detail = f" [dim]({mfee_text})[/dim]" if mfee_text else ""
        table.add_row(
            "[bold]月物业费(估算)[/bold]",
            f"[cyan]¥{mfee:,.2f}/月[/cyan]{detail}",
        )
    fields = det.get("fields") or []
    if fields:
        table.add_row(
            "字段",
            "\n".join(f"• {f.get('label', '')}: {f.get('value', '')}" for f in fields),
        )


def _show_identity_rows(table: Table, det: dict) -> None:
    """身份标识（精确到人）：person_identities，known_parties 基准库逐人核对的依据。"""
    pids = det.get("person_identities") or []
    if not pids:
        return
    lines = []
    for p in pids:
        role = p.get("role")
        head = f"[bold]{p.get('name', '?')}[/bold]" + (f" [dim]({role})[/dim]" if role else "")
        lines.append(head)
        for idv in p.get("identifiers") or []:
            lines.append(f"  • {idv.get('label', '')}: {idv.get('value', '')}")
    table.add_row("身份标识", "\n".join(lines))


def _show_seal_rows(table: Table, row) -> None:
    """印章 + 附属协议（补充协议，各有独立签章落款）。det 用 row.details() 现取避免分支差异。"""
    seals = row.details().get("seals") or []
    if seals:
        lines = []
        for s in seals:
            owner = s.get("owner") or "?"
            stype = s.get("seal_type")
            head = owner + (f" · {stype}" if stype else "")
            raw = s.get("raw_text") or ""
            lines.append(f"• {head}  [dim]{raw}[/dim]" if raw else f"• {head}")
        table.add_row("[bold]印章[/bold]", "\n".join(lines))

    subs = row.details().get("sub_agreements") or []
    if subs:
        lines = []
        for sub in subs:
            head = f"[bold]{sub.get('title') or '附属协议'}[/bold]"
            if sub.get("sign_date"):
                head += f"  [dim]{sub['sign_date']}[/dim]"
            lines.append(head)
            if sub.get("summary"):
                lines.append(f"  {sub['summary']}")
            sseals = sub.get("seals") or []
            if sseals:
                for s in sseals:
                    owner = s.get("owner") or "?"
                    stype = s.get("seal_type")
                    lines.append(f"  印章: {owner}" + (f" · {stype}" if stype else ""))
            else:
                lines.append("  [dim]印章: 无[/dim]")
        table.add_row("[bold]补充协议[/bold]", "\n".join(lines))


def _show_completeness_rows(table: Table, row) -> None:
    """合同完整性核查块（仅合同有；签章经落款页 VL 核查，要素/金额据原文）。"""
    comp = row.details().get("completeness")
    if not comp:
        return
    status = comp.get("status")
    issues = comp.get("issues") or []
    if status == "complete":
        table.add_row("[bold]完整性[/bold]", "[green]✓ 要素与签章齐全[/green]")
    elif status == "incomplete":
        lines = ["[red]⚠ 疑似不完整[/red] [dim](签章经落款页核查；要素/金额据原文，可翻回核对)[/dim]"]
        cat_label = {"signature": "签章", "amount": "金额", "field": "要素"}
        for it in issues:
            cat = cat_label.get(it.get("category"), "要素")
            detail = it.get("detail") or ""
            tail = f" — [dim]{detail}[/dim]" if detail else ""
            lines.append(f"• [{cat}] {it.get('item', '')}{tail}")
            evidence = it.get("evidence") or ""
            if evidence:
                lines.append(f"    [dim]↳ 出处：{evidence}[/dim]")
        table.add_row("[bold]完整性[/bold]", "\n".join(lines))
    else:  # unknown
        table.add_row("[bold]完整性[/bold]", "[yellow]? 信息不足，未能判定[/yellow]")


def _show_footer_rows(table: Table, row) -> None:
    """身份核对不一致 + 抽取元数据（llm_model/置信度）+ 双方义务动作 + 风险条款。"""
    # 身份核对：person_identities 与 known_parties 基准库比对的不一致项（跨文档类型）。
    id_issues = row.details().get("identity_issues") or []
    if id_issues:
        lines = ["[red]⚠ 与基准库不一致[/red] [dim](known_parties 跨文档核对，请人工确认)[/dim]"]
        for it in id_issues:
            detail = it.get("detail") or ""
            tail = f" — [dim]{detail}[/dim]" if detail else ""
            lines.append(f"• {it.get('item', '')}{tail}")
            ev = it.get("evidence") or ""
            if ev:
                lines.append(f"    [dim]↳ {ev}[/dim]")
        table.add_row("[bold]身份核对[/bold]", "\n".join(lines))
    table.add_row(
        "llm_model",
        row.details().get("llm_model") or "[dim]- (旧抽取未记录，重抽后显示)[/dim]",
    )
    table.add_row(
        "overall_confidence",
        f"{row.overall_confidence:.2f}" if row.overall_confidence is not None else "-",
    )
    if row.obligations:
        table.add_row("", "")
        for actor_key, label in (
            ("party_a", "[bold]甲方动作[/bold]"),
            ("party_b", "[bold]乙方动作[/bold]"),
            ("both",    "[bold]双方动作[/bold]"),
        ):
            items = [o for o in row.obligations if o.actor == actor_key]
            if not items:
                continue
            lines = []
            for o in items:
                dl = o.deadline or "[dim]无日期[/dim]"
                lines.append(f"• [{dl}] {o.action}")
            table.add_row(label, "\n".join(lines))
    if row.risk_clauses:
        table.add_row(
            "[bold]risk_clauses[/bold]",
            "\n".join(f"• {c}" for c in row.risk_clauses),
        )


def build_list_table(rows, root) -> Table:
    """list 命令的档案列表（纯函数：rows + 档案根目录 → rich Table，不打印）。"""
    table = Table(
        title=f"Archive · {root} ({len(rows)} of total)",
        caption="amount 带 * 为计算合计（如收入证明=年税前+股权），无 * 为抽取的主金额",
        caption_justify="left",
    )
    table.add_column("id", style="cyan", justify="right")
    table.add_column("status")
    table.add_column("type", style="magenta")
    table.add_column("完整")  # 合同完整性：⚠ 疑似缺 / ✓ / -（非合同）
    table.add_column("title", overflow="fold")
    table.add_column("主体", overflow="fold")  # 区分同类文档（谁的/和谁签的）
    table.add_column("date")
    table.add_column("amount", justify="right")
    table.add_column("ingested", style="dim")
    for r in rows:
        table.add_row(
            str(r.id),
            status_color(r.status),
            r.doc_type or "-",
            completeness_mark(r),
            r.title or r.contract_name or "-",
            subject_of(r),
            r.primary_date or "-",
            display_amount(r),
            local_time(r.ingested_at)[:10],  # 本地日期，与 show 一致
        )
    return table


# ---------- raw 命令：原文高亮（TTY 上色，标出 LLM 抽到的关键字）----------
#
# 终端着色用 ANSI 转义码。是否上色由 cli.py 按 --color + isatty 决定；这里只做
# "数据 → 带色字符串" 的纯转换，便于单测。按抽取来源分类着色，让"哪些被识别到、
# 识别成什么类别"一眼可见。

_HL_RESET = "\033[0m"
_HL_STYLES: dict[str, str] = {
    "party": "\033[1;36m",   # 加粗青：当事人 / 主体 / 印章 owner
    "amount": "\033[1;33m",  # 加粗黄：金额（原文串）
    "date": "\033[1;34m",    # 加粗蓝：日期（原文串；ISO 规范化值通常命不中）
    "risk": "\033[1;31m",    # 加粗红：风险条款
    "field": "\033[1;35m",   # 加粗紫：其他字段值 / 义务出处 / 印章原文
}
_HL_LABELS = [("party", "当事人"), ("amount", "金额"), ("date", "日期"),
              ("risk", "风险"), ("field", "字段")]


def extracted_terms(row) -> dict[str, str]:
    """
    收集 LLM 抽取的、可能在原文里**原样出现**的串 → 高亮类别 key。

    只收原文原样承载的值（主体名 / 原始金额串 / 字段值 / 出处片段）；日期 ISO、
    金额数值、摘要等是规范化或改写的，在原文里 substring 命不中 → 自然不高亮，
    诚实反映"原文里真出现且被抽到"的项。短串（<2 字）丢弃，避免单字满屏误命中。
    """
    terms: dict[str, str] = {}

    def add(value, style: str) -> None:
        if isinstance(value, str):
            v = value.strip()
            if len(v) >= 2:
                terms[v] = style

    # 合同专属顶层列
    add(row.contract_name, "field")
    add(row.party_a, "party")
    add(row.party_b, "party")
    add(row.amount_text, "amount")
    add(row.sign_date, "date")
    add(row.expire_date, "date")
    for rc in row.risk_clauses:
        add(rc, "risk")
    for o in row.obligations:
        add(o.evidence, "field")        # evidence 是原文片段
    # 通用信封柔性字段（details_json = DocumentExtraction）
    det = row.details()
    for p in det.get("parties") or []:
        add(p, "party")
    for a in det.get("amounts") or []:
        add(a.get("text"), "amount")    # 原文金额串（含大写 / 币种）
        add(a.get("evidence"), "amount")
    for d in det.get("key_dates") or []:
        add(d.get("date"), "date")
    for f in det.get("fields") or []:
        add(f.get("value"), "field")    # 字段原文值
    for s in det.get("seals") or []:
        add(s.get("owner"), "party")
        add(s.get("raw_text"), "field")
    return terms


def render_highlighted(text: str, terms: dict[str, str]) -> str:
    """
    给原文里命中的抽取串套 ANSI 着色，返回新串（纯函数，不碰 stdout）。

    长串优先排进正则 alternation：finditer 同位置优先吃长串、且天然从左到右
    不重叠——无需手动合并重叠区间（把特殊情况消成正常情况）。命不中的 term 自然忽略。
    """
    if not terms:
        return text
    ordered = sorted(terms, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(t) for t in ordered))
    out: list[str] = []
    last = 0
    for m in pattern.finditer(text):
        hit = m.group()
        style = _HL_STYLES.get(terms.get(hit, "field"), _HL_STYLES["field"])
        out.append(text[last:m.start()])
        out.append(f"{style}{hit}{_HL_RESET}")
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def color_legend(terms: dict[str, str]) -> str:
    """已命中的类别 → 一行 ANSI 图例，解释每种颜色代表的抽取类别。无命中返回空串。"""
    used = set(terms.values())
    parts = [f"{_HL_STYLES[k]}■{name}{_HL_RESET}"
             for k, name in _HL_LABELS if k in used]
    return "图例 " + "  ".join(parts) if parts else ""


def build_search_table(rows) -> Table:
    """search 命令的命中列表（纯函数：rows → rich Table，不打印）。"""
    table = Table(title=f"Search · {len(rows)} hit(s)")
    table.add_column("id", style="cyan", justify="right")
    table.add_column("name", overflow="fold")
    table.add_column("party_a", overflow="fold")
    table.add_column("party_b", overflow="fold")
    table.add_column("amount", justify="right")
    table.add_column("sign_date")
    table.add_column("expire_date")
    table.add_column("risks", justify="right")
    for r in rows:
        amount = f"¥{r.amount_value:,.0f}" if r.amount_value is not None else "-"
        table.add_row(
            str(r.id),
            r.contract_name or "-",
            r.party_a or "-",
            r.party_b or "-",
            amount,
            r.sign_date or "-",
            r.expire_date or "-",
            str(len(r.risk_clauses)),
        )
    return table
