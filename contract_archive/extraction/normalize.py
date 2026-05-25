"""
确定性数值/结构归一化工具。

这是项目里**唯一保留的"死代码 rule"**——但它做的不是字段抽取，而是把 LLM
吐出的原文值规整成可存储/可比较的形态（中文大写金额→数值、日期→ISO、
LLM 数组→强类型对象）。LLM 在精确数值/日期换算上不可靠，这类确定性转换
交给代码更稳。字段"抽什么"全归 LLM。
"""
from __future__ import annotations

import re
from typing import Any

from ..schemas import ObligationItem


# ---------- 日期 ----------


def normalize_date(value: str | None) -> str | None:
    """把原文日期粗糙归一化到 ISO 8601。失败原样返回。"""
    if not value:
        return None
    # 兼容 OCR 输出里数字与"年/月"之间出现的空格
    m = re.match(
        r"^((?:19|20)\d{2}|二[〇零]{1,3}[一二三四五六七八九十]{1,3})"
        r"\s*[年\-./]\s*(\d{1,2}|[一二三四五六七八九十]{1,3})"
        r"\s*[月\-./]\s*(\d{1,2}|[一二三四五六七八九十]{1,3})",
        value,
    )
    if not m:
        return value
    y, mo, d = m.group(1), m.group(2), m.group(3)
    if not y.isdigit():
        y = _cn_year_to_int(y)
    if not mo.isdigit():
        mo = _cn_num_to_int(mo)
    if not d.isdigit():
        d = _cn_num_to_int(d)
    try:
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    except (ValueError, TypeError):
        return value


_CN_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_year_to_int(s: str) -> str:
    digits = [str(_CN_DIGITS.get(ch, "")) for ch in s if ch in _CN_DIGITS]
    return "".join(digits) or s


def _cn_num_to_int(s: str) -> str:
    if "十" in s:
        parts = s.split("十")
        tens = _CN_DIGITS.get(parts[0], 1) if parts[0] else 1
        ones = _CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return str(tens * 10 + ones)
    return "".join(str(_CN_DIGITS.get(ch, "")) for ch in s) or s


# ---------- 金额 ----------


_CN_MONEY_DIGITS = {
    "零": 0, "壹": 1, "贰": 2, "叁": 3, "肆": 4, "伍": 5,
    "陆": 6, "柒": 7, "捌": 8, "玖": 9,
    # 小写也支持（偶尔混用）
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "两": 2,
}
_CN_MONEY_UNITS = {
    "拾": 10, "佰": 100, "仟": 1000, "万": 10000, "亿": 100000000,
    "十": 10, "百": 100, "千": 1000,
}


def _cn_money_to_value(s: str) -> float | None:
    """
    中文大写金额 → 数值（元）。
    "壹仟贰佰贰拾柒万玖仟捌佰捌拾玖元整" → 12279889.0
    "陆拾贰万壹仟壹佰零陆元柒角壹分" → 621106.71
    解析失败返回 None。
    """
    s = s.replace("人民币", "").replace("圆", "元").replace("整", "").strip()
    if not s:
        return None
    # 小数部分：元后的 角(0.1) / 分(0.01)，如"…元柒角壹分" = .71
    frac = 0.0
    if "元" in s:
        intpart, _, fracpart = s.partition("元")
        for unit_ch, unit_val in (("角", 0.1), ("分", 0.01)):
            idx = fracpart.find(unit_ch)
            if idx >= 1:
                d = _CN_MONEY_DIGITS.get(fracpart[idx - 1])
                if d is not None:
                    frac += d * unit_val
        s = intpart
    total = 0
    section = 0   # 当前"万"以下的累加
    digit = 0     # 上一个数字
    saw_any = False
    for ch in s:
        if ch in _CN_MONEY_DIGITS:
            digit = _CN_MONEY_DIGITS[ch]
            saw_any = True
        elif ch in _CN_MONEY_UNITS:
            unit = _CN_MONEY_UNITS[ch]
            if unit >= 10000:
                # 万 / 亿 触发段结算
                section = (section + digit) * unit
                total += section
                section = 0
            else:
                # 拾/佰/仟：digit 为 0 时按 1（"拾" 单独 = 10）
                section += (digit if digit else 1) * unit
            digit = 0
            saw_any = True
        elif ch.isspace():
            continue
        else:
            return None  # 不认识的字符（含混阿拉伯/中文），交给阿拉伯路径
    section += digit
    total += section
    result = float(total) + frac
    return result if (saw_any or frac) and result > 0 else None


def parse_money_value(value: str | None) -> float | None:
    """从原文金额抽出数值（人民币元）。先试阿拉伯数字，再试中文大写。"""
    if not value:
        return None
    # 阿拉伯数字优先（精度高，覆盖大多数情况）
    m = re.search(r"([0-9]+(?:[,，]\d{3})*(?:\.\d+)?)\s*(万元|万|千元|百元|元|圆)?", value)
    if m:
        num = float(m.group(1).replace(",", "").replace("，", ""))
        unit = m.group(2) or "元"
        multiplier = {"万元": 10000, "万": 10000, "千元": 1000, "百元": 100,
                      "元": 1, "圆": 1}.get(unit, 1)
        return num * multiplier
    # 退化到中文大写
    return _cn_money_to_value(value)


# ---------- LLM 数组 → 强类型 ----------


def coerce_bool(value: Any) -> bool | None:
    """LLM 的 auto_renewal 等布尔字段：兼容 true/false/是/否/null。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("true", "是", "yes", "1"):
        return True
    if s in ("false", "否", "no", "0"):
        return False
    return None


def coerce_str_list(value: Any, max_len: int = 200) -> list[str]:
    """LLM 的字符串数组（如 risk_clauses）：过滤空项、截断。"""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = str(item).strip() if item is not None else ""
        if s:
            out.append(s[:max_len])
    return out


def coerce_obligations(raw: Any) -> list[ObligationItem]:
    """
    把 LLM 返回的 obligations 数组（dict 列表）转 ObligationItem。
    actor 兼容"甲方"/"乙方"中文别名；跳过缺 actor/action 的非法项，不抛异常。
    """
    if not isinstance(raw, list):
        return []
    actor_alias = {
        "甲方": "party_a", "Party A": "party_a", "partyA": "party_a",
        "乙方": "party_b", "Party B": "party_b", "partyB": "party_b",
        "双方": "both", "Both": "both",
    }
    out: list[ObligationItem] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        actor = str(item.get("actor", "")).strip()
        actor = actor_alias.get(actor, actor)
        if actor not in ("party_a", "party_b", "both"):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        deadline = item.get("deadline")
        deadline = normalize_date(deadline) or None if isinstance(deadline, str) else None
        evidence = str(item.get("evidence", "")).strip()[:500]
        out.append(
            ObligationItem(
                actor=actor,  # type: ignore[arg-type]
                action=action[:200],
                deadline=deadline,
                evidence=evidence,
            )
        )
    return out
