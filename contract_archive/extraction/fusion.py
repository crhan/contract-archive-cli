"""多源融合：把文本抽取（A 路）与看图抽取（C 路）对高价值概念的候选评判成定值 + 置信度。

为什么要融合：单源对复杂表格/混合版式系统性丢数据——文本路把表格抹平、看图路偶有错位。
让两路各自给候选，**一致就直接采信（省一次 LLM）**，**矛盾才据原图评判**，比任一单源都稳。

铁规（均为正确性约束，非兼容性）：
- **只产出 FieldVerdict sidecar，绝不回写原字段**（attach_verdicts）：保额/免赔等原字段带着
  evidence/unit/is_total_component 等不变量与 computed_total 的勾稽，回写会破坏。
- **概念键独立**：一般/特定医疗/重疾各一键（由调用方保证），fusion 按键各管各，绝不跨键比对——
  这从根上消除"A 概念的值覆盖 B 概念"（治 ③ 的对齐错位）。
- **collect_candidates → {键:{源:[候选]}}**：每源一个列表，多页/多次候选都留住，绝不互相覆盖。
- **_agree 看归一化后的值**：相等才算无分歧、跳过评判；只要有一源不同就送评判。
- 评判 **独立依据原图、勿受候选主导、矛盾以图为准并标 low_confidence**；置信 < 阈值 → low_confidence。
- 评判键并发，client/proxy-env 外层一次性构造、max_retries=2（worker 只复用）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import get_timeout_s, load_settings
from ..errors import ErrorInfo, config_missing
from ..schemas import DocumentExtraction, FieldCandidate, FieldVerdict
from ..utils import encode_image_data_uri, map_concurrent, merge_usage
from ..utils.http_env import sanitized_httpx_proxy_env
from .llm_extractor import _parse_json_loose, _usage_from_openai

logger = logging.getLogger(__name__)

DEFAULT_FUSION_THRESHOLD = 0.6  # 评判置信 < 此值 → low_confidence（供 agent 兜底关注）

# 各源采信置信度（无需评判的路径）
_CONF_MULTI_SOURCE_AGREED = 0.95  # ≥2 源一致：最可信
_CONF_SINGLE_SOURCE = 0.7  # 仅一源给出（未经交叉验证）：可用但留余量
_CONF_NO_IMAGE_ADJUDICATION = 0.4  # 源间矛盾但无原图可依：只能挑一个，必标 low_confidence


ADJUDICATE_PROMPT = """你是严谨的保险/合同字段核查员。某字段的多路抽取给出了**互相矛盾**的候选值，下面附上相关页的原图。请**独立依据原图**判定该字段真实值。

字段：{key}
定义：{definition}
（提示：被保险人=保障对象本人；投保人=出钱投保的人。二者不同，切勿混淆。）

各路候选（仅作线索，**不要被候选主导**）：
{candidates}

要求：
1. 以原图为准。候选与原图矛盾时，以图为准，并把 low_confidence 设为 true。
2. 图上看不清/无法判定时，value 给最可能值或 null，confidence 给低分，low_confidence 设 true。
3. 表格里按行列对齐读，别错位。
4. 只输出 JSON，不要解释、不要 Markdown 代码块标记：
{{"value": <值或 null>, "confidence": <0~1 小数>, "low_confidence": <true/false>, "rationale": <一句依据>}}
"""


@dataclass
class FusionResult:
    """融合产物：高价值概念逐项评判结论 + 整体置信度（取最弱项，任一高价值字段不稳即拉低）。"""

    verdicts: list[FieldVerdict] = field(default_factory=list)
    overall_confidence: Optional[float] = None
    usage: Optional[dict] = None  # 评判（adjudicate）的 LLM 开销；看图/文本路开销由各自调用点记
    error: Optional[ErrorInfo] = None


def collect_candidates(
    text_by_key: dict[str, list[FieldCandidate]],
    vision_by_key: dict[str, list[FieldCandidate]],
) -> dict[str, dict[str, list[FieldCandidate]]]:
    """按 {键: {源: [候选]}} 归集。每源一个列表，多页/多次候选都留住，绝不互相覆盖。"""
    out: dict[str, dict[str, list[FieldCandidate]]] = {}
    for source_name, by_key in (("text", text_by_key), ("vision", vision_by_key)):
        for key, cands in (by_key or {}).items():
            if cands:
                out.setdefault(key, {}).setdefault(source_name, []).extend(cands)
    return out


def _normalize_value(v: str) -> str:
    """轻量归一化用于一致性比较：去空白/币种修饰、万元→万、大小写。不动语义。"""
    s = v.strip().lower().replace(" ", "").replace("　", "")
    for junk in ("元整", "圆整", "元", "圆", "￥", "¥", "rmb", "人民币", "整"):
        s = s.replace(junk, "")
    s = s.replace("万元", "万")
    return s


def _agree(cands_by_source: dict[str, list[FieldCandidate]]) -> bool:
    """所有源、所有候选归一化后的值是否全相等（相等才算无分歧、可跳过评判）。"""
    values = {
        _normalize_value(c.value)
        for cands in cands_by_source.values()
        for c in cands
        if c.value
    }
    return len(values) == 1


def _flatten(cands_by_source: dict[str, list[FieldCandidate]]) -> list[FieldCandidate]:
    out: list[FieldCandidate] = []
    for cands in cands_by_source.values():
        out.extend(cands)
    return out


def _agreed_verdict(key: str, cands_by_source: dict[str, list[FieldCandidate]]) -> FieldVerdict:
    """各源一致：直接采信，不调用 LLM。多源一致最可信，单源未经交叉验证留余量。"""
    flat = _flatten(cands_by_source)
    value = next((c.value for c in flat if c.value), None)
    multi = len(cands_by_source) >= 2
    return FieldVerdict(
        key=key,
        value=value,
        source="agreed" if multi else next(iter(cands_by_source)),
        confidence=_CONF_MULTI_SOURCE_AGREED if multi else _CONF_SINGLE_SOURCE,
        low_confidence=False,
        rationale="各源一致" if multi else "仅单源给出，未交叉验证",
        candidates=flat,
    )


def _pages_for(cands_by_source: dict[str, list[FieldCandidate]]) -> list[int]:
    """评判要看的页：候选里带的页号（看图候选有页、文本候选无）。去重保序。"""
    seen: list[int] = []
    for c in _flatten(cands_by_source):
        if c.page is not None and c.page not in seen:
            seen.append(c.page)
    return seen


def _render_candidates(cands_by_source: dict[str, list[FieldCandidate]]) -> str:
    lines = []
    for source, cands in cands_by_source.items():
        for c in cands:
            loc = f"（第{c.page}页）" if c.page is not None else ""
            ev = f" 证据：{c.evidence}" if c.evidence else ""
            lines.append(f"- [{source}{loc}] {c.value}{ev}")
    return "\n".join(lines)


def _no_image_verdict(key: str, cands_by_source: dict[str, list[FieldCandidate]]) -> FieldVerdict:
    """源间矛盾但无原图可依：只能挑看图源（其次文本源）的值，必标 low_confidence。"""
    flat = _flatten(cands_by_source)
    vision = next((c.value for c in flat if c.source == "vision" and c.value), None)
    value = vision or next((c.value for c in flat if c.value), None)
    return FieldVerdict(
        key=key,
        value=value,
        source="adjudicated",
        confidence=_CONF_NO_IMAGE_ADJUDICATION,
        low_confidence=True,
        rationale="源间矛盾且无原图可依，无法据图判定",
        candidates=flat,
    )


def adjudicate_field(
    client,
    model: str,
    key: str,
    cands_by_source: dict[str, list[FieldCandidate]],
    images: list[Path],
    field_def: str,
    threshold: float,
) -> tuple[FieldVerdict, Optional[dict]]:
    """源间矛盾 → 据原图独立评判。返回 (verdict, usage)。异常由调用方隔离。"""
    flat = _flatten(cands_by_source)
    prompt = ADJUDICATE_PROMPT.format(
        key=key,
        definition=field_def or "（无额外定义，按字面理解）",
        candidates=_render_candidates(cands_by_source),
    )
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": encode_image_data_uri(img)}})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    parsed = _parse_json_loose(resp.choices[0].message.content or "")
    value = parsed.get("value")
    value = str(value).strip() if value not in (None, "") else None
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    # 模型自报 low_confidence 或置信低于阈值，都判 low（阈值兜底，防模型乐观）
    low = bool(parsed.get("low_confidence")) or confidence < threshold
    verdict = FieldVerdict(
        key=key,
        value=value,
        source="adjudicated",
        confidence=confidence,
        low_confidence=low,
        rationale=str(parsed.get("rationale") or "")[:200],
        candidates=flat,
    )
    return verdict, _usage_from_openai(resp)


def fuse_sources(
    text_by_key: dict[str, list[FieldCandidate]],
    vision_by_key: dict[str, list[FieldCandidate]],
    *,
    images_by_page: Optional[dict[int, Path]] = None,
    field_defs: Optional[dict[str, str]] = None,
    threshold: float = DEFAULT_FUSION_THRESHOLD,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> FusionResult:
    """融合文本/看图两路候选 → 逐高价值概念评判。一致直接采信（不调 LLM），矛盾才据图评判。

    images_by_page：页号→页图，供评判看原图（缺则走无图降级、标 low_confidence）。
    field_defs：概念键→定义，喂评判 prompt（如被保险人 vs 投保人口径）。
    无矛盾键时**完全不构造 client / 不发请求**——省钱，也让无 key 环境照常产出"一致"结论。
    """
    candidates = collect_candidates(text_by_key, vision_by_key)
    if not candidates:
        return FusionResult()

    images_by_page = images_by_page or {}
    field_defs = field_defs or {}

    # 先分流：一致的键直接出 verdict（零 LLM）；矛盾的键攒起来批量并发评判。
    verdicts: dict[str, FieldVerdict] = {}
    disputed: list[str] = []
    for key in sorted(candidates):
        if _agree(candidates[key]):
            verdicts[key] = _agreed_verdict(key, candidates[key])
        else:
            disputed.append(key)

    usage: Optional[dict] = None
    error: Optional[ErrorInfo] = None
    if disputed:
        settings = load_settings()
        model = model or settings.dashscope_vl_extract_model
        api_key = api_key or settings.dashscope_api_key
        base_url = base_url or settings.dashscope_base_url
        if not api_key:
            # 无凭证：矛盾键退化为无图降级（挑看图值、标 low），并记 error。
            logger.warning("DASHSCOPE_API_KEY missing; adjudicate degraded to no-image verdicts")
            error = config_missing("DASHSCOPE_API_KEY 缺失，融合评判降级")
            for key in disputed:
                verdicts[key] = _no_image_verdict(key, candidates[key])
        else:
            usage = _adjudicate_disputed(
                disputed, candidates, images_by_page, field_defs, threshold,
                model, api_key, base_url, verdicts,
            )

    ordered = [verdicts[k] for k in sorted(verdicts)]
    overall = min((v.confidence for v in ordered), default=None)
    return FusionResult(verdicts=ordered, overall_confidence=overall, usage=usage, error=error)


def _adjudicate_disputed(
    disputed, candidates, images_by_page, field_defs, threshold,
    model, api_key, base_url, verdicts,
) -> Optional[dict]:
    """并发评判矛盾键。client/proxy-env 外层一次性构造，worker 只复用。回填 verdicts，返回合并 usage。"""
    from openai import OpenAI

    compat_url = base_url.replace("/api/v1", "/compatible-mode/v1")
    with sanitized_httpx_proxy_env():
        client = OpenAI(
            api_key=api_key,
            base_url=compat_url,
            timeout=get_timeout_s("DASHSCOPE_TIMEOUT_S", 300.0),
            max_retries=2,
        )

        def _one(key: str) -> tuple[str, FieldVerdict, Optional[dict]]:
            # client 直接闭包进来（不走模块级状态，多文档并发融合也互不干扰）。
            pages = _pages_for(candidates[key])
            images = [images_by_page[p] for p in pages if p in images_by_page]
            if not images:
                return key, _no_image_verdict(key, candidates[key]), None
            verdict, u = adjudicate_field(
                client, model, key, candidates[key], images, field_defs.get(key, ""), threshold
            )
            return key, verdict, u

        results = map_concurrent(_one, disputed)

    usages: list[Optional[dict]] = []
    for res in results:
        if not res:  # 整个 _one 抛异常被隔离（含原图编码失败等）——下面补无图降级
            continue
        key, verdict, u = res
        verdicts[key] = verdict
        usages.append(u)
    # 被隔离掉的矛盾键补无图降级，避免漏键
    for key in disputed:
        verdicts.setdefault(key, _no_image_verdict(key, candidates[key]))
    return merge_usage(usages)


def attach_verdicts(extraction: DocumentExtraction, result: FusionResult) -> DocumentExtraction:
    """把融合结论写入 sidecar，**绝不回写原字段**。usage 并入 llm_usage（评判开销进总账）。"""
    extraction.field_verdicts = result.verdicts
    extraction.fusion_overall_confidence = result.overall_confidence
    if result.usage:
        extraction.llm_usage = merge_usage([extraction.llm_usage, result.usage])
    return extraction


def run_vision_fusion(
    extraction: DocumentExtraction,
    document_text: str,
    images_by_page: dict[int, Path],
    *,
    fields: dict[str, str],
    threshold: float = DEFAULT_FUSION_THRESHOLD,
    text_model: str | None = None,
    vision_model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> bool:
    """端到端融合编排：A(文本)/C(看图) 两路**并发**按同一组 fields 抽候选 → fuse_sources 评判
    → attach 到 envelope sidecar。返回是否产出了 verdict。

    A、C 两路无依赖、各发各的 LLM，故并发跑（map_concurrent 2 workers）。两路抽取 + 评判的
    全部 token 并入 envelope.llm_usage，让成本核算看到融合总开销。无 fields/无候选 → False。
    """
    # 延迟导入，避免 fusion ←→ 抽取源模块的潜在环；也让无融合路径零额外加载。
    from .text_fields import read_fields_in_text
    from .vl_extract import read_fields_on_images

    if not fields:
        return False

    labels = sorted(images_by_page) if images_by_page else []
    image_paths = [images_by_page[p] for p in labels]

    def _a():  # A 路：文本看字段
        return read_fields_in_text(
            document_text, fields, model=text_model, api_key=api_key, base_url=base_url
        )

    def _c():  # C 路：看图看字段（无图则跳过）
        if not image_paths:
            return None
        return read_fields_on_images(
            image_paths, fields, page_labels=labels,
            model=vision_model, api_key=api_key, base_url=base_url,
        )

    a_res, c_res = map_concurrent(lambda f: f(), [_a, _c], max_workers=2)
    text_by_key = a_res.by_key if a_res else {}
    vision_by_key = c_res.by_key if c_res else {}
    if not text_by_key and not vision_by_key:
        return False

    result = fuse_sources(
        text_by_key, vision_by_key,
        images_by_page=images_by_page, field_defs=fields, threshold=threshold,
        model=vision_model, api_key=api_key, base_url=base_url,
    )
    attach_verdicts(extraction, result)  # 已并入评判开销
    extraction.llm_usage = merge_usage(
        [extraction.llm_usage, a_res.usage if a_res else None, c_res.usage if c_res else None]
    )
    return bool(result.verdicts)
