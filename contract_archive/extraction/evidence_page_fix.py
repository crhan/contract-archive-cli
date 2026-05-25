"""
出处页码校正：用 MinerU content_list.json 的可靠 page_idx 覆盖 LLM 猜的页码。

为什么需要：LLM 抽取走扁平 raw_text（多页拼接、页边界已丢失），它填进 evidence
的页码靠估算，长文档常错位（实测 29 号占用费在 PDF 第6页，LLM 填了第5页）。而
content_list.json 每个文本块带准确 page_idx——拿 evidence 里的原文片段去反查，把
页码校正过来。这与签章核查（vision_seal 用 content_list 定位落款页）同一可靠来源。

只动"第X页 + 原文片段"这种带可定位片段的出处；签章类 evidence（VL 给的
"据落款页图：第X页"，无原文片段）正则不匹配，天然不受影响。

降级：无 content_list / 片段反查不到时，保留原页码不动——诚实，不瞎改。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..schemas import DocumentExtraction

logger = logging.getLogger(__name__)

# 匹配 evidence 里的"第X页 + 原文片段"对（片段取到分号/串尾止）。
# 拼接 evidence（多分期项以分号隔开）会逐对匹配各自校正；
# 签章式"据落款页图：第8页"没有"+ 片段"，不匹配 → 不动。
_PAGE_FRAG = re.compile(r"第\s*(\d+)\s*页\s*[+＋]\s*([^；;]*)")
_WS = re.compile(r"\s+")

# 反查用的最短/滑窗片段长度：太短易误命中多页，故要求 ≥8 字连续重叠。
_MIN_ANCHOR = 8
_WINDOW = 12


def _load_blocks(mineru_dir: Path) -> list[tuple[str, int]]:
    """content_list.json → [(去空白文本, page_idx)]。读不到/解析失败返回 []。"""
    content_lists = list(mineru_dir.glob("_mineru_raw/*/auto/*_content_list.json"))
    if not content_lists:
        return []
    try:
        items = json.loads(content_lists[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取 content_list 失败，跳过页码校正: %s", e)
        return []
    blocks: list[tuple[str, int]] = []
    for it in items:
        if isinstance(it, dict) and it.get("page_idx") is not None:
            text = _WS.sub("", str(it.get("text", "")))
            if text:
                blocks.append((text, int(it["page_idx"])))
    return blocks


def _find_page(fragment: str, blocks: list[tuple[str, int]]) -> int | None:
    """用原文片段在各文本块中反查 page_idx（0-based）。滑窗子串命中即返回，否则 None。"""
    frag = _WS.sub("", fragment)
    if len(frag) < _MIN_ANCHOR:
        return None
    for i in range(max(1, len(frag) - _MIN_ANCHOR + 1)):
        sub = frag[i:i + _WINDOW]
        if len(sub) < _MIN_ANCHOR:
            break
        for text, page_idx in blocks:
            if sub in text:
                return page_idx
    return None


def _correct_evidence(evidence: str, blocks: list[tuple[str, int]]) -> str:
    """校正一条 evidence 里所有"第X页 + 片段"对的页码；反查不到的对保持不动。"""
    def repl(m: "re.Match[str]") -> str:
        frag = m.group(2)
        page_idx = _find_page(frag, blocks)
        if page_idx is None:
            return m.group(0)
        return f"第{page_idx + 1}页 + {frag}"

    return _PAGE_FRAG.sub(repl, evidence)


def correct_evidence_pages(env: DocumentExtraction, mineru_dir: Path) -> bool:
    """
    用 content_list 的 page_idx 校正 env 中 amounts / completeness issues 的 evidence 页码。

    原地修改 env。有 content_list 可用返回 True；无则返回 False（调用方保留原页码）。
    amount 类 issue 的 evidence 是各分期项出处的拼接，_PAGE_FRAG 逐对匹配，一并校正。
    """
    blocks = _load_blocks(mineru_dir)
    if not blocks:
        return False
    for amount in env.amounts:
        if amount.evidence:
            amount.evidence = _correct_evidence(amount.evidence, blocks)
    if env.completeness:
        for issue in env.completeness.issues:
            if issue.evidence:
                issue.evidence = _correct_evidence(issue.evidence, blocks)
    return True
