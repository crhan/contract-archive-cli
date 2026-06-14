"""多源融合的"看图抽字段"（C 路）：把关键页图直接喂给通用 VL 模型，抽高价值概念候选值。

与 vl_ocr（逐页转写全文）的分工不同：这里不要全文，只针对调用方给定的一组高价值概念键，
让 VL **据图**直接回 JSON 候选——表格/混合版式里文本抽取易丢的字段（保额/免赔/赔付比例），
看图比读 OCR 文本更稳。产出 FieldCandidate(source="vision")，交 fusion 与文本候选评判融合。

字段定义（含义、被保险人 vs 投保人之类的领域口径）由调用方经 fields_spec 传入——本模块只是
"看图抽这些被描述的字段"的通用工具，领域知识留在保险 handler，不写死在这里。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import get_timeout_s, load_settings
from ..errors import ErrorInfo, config_missing
from ..schemas import FieldCandidate
from ..utils import encode_image_data_uri, map_concurrent, merge_usage
from ..utils.http_env import sanitized_httpx_proxy_env
from .llm_extractor import _parse_json_loose, _usage_from_openai

logger = logging.getLogger(__name__)


VISION_EXTRACT_PROMPT = """你是严谨的看图抽取助理。下面是一张文档页图像。请**只看这张图**，据实抽取下列字段，以 JSON 返回。

铁律：
1. 只看图、据实抽取；这张图上看不到的字段一律填 null，禁止猜测、禁止跨页脑补。
2. 值保留原文（含币种/单位/百分比），如 "200万元" "1万元" "100%" "90天"。
3. 表格里的字段按行列对齐读，别错位；同一概念有多个数值时分别归到对应的 key。
4. 只输出 JSON，不要解释、不要 Markdown 代码块标记。

要抽取的字段（key: 含义）：
{fields}

返回 JSON（每个字段一个对象；看不到填 null）：
{{
{schema}
}}
"""


@dataclass
class VisionFieldsResult:
    """看图抽取产物：概念键 → 看图候选列表（按页序）。"""

    by_key: dict[str, list[FieldCandidate]] = field(default_factory=dict)
    model: str = ""
    usage: Optional[dict] = None
    error: Optional[ErrorInfo] = None


def read_fields_on_images(
    image_paths: list[Path],
    fields_spec: dict[str, str],
    *,
    page_labels: list[int] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> VisionFieldsResult:
    """并发看图抽取：每张图一次 VL 调用，汇总每个概念键的候选（source="vision"）。

    - page_labels 给每张图的真实页号（缺省 1..N），回填到候选 page，供评判/审计追溯。
    - 无凭证 → 空结果 + error（让融合只走文本路）。空输入/空 spec → 空结果。
    - 单图失败隔离（map_concurrent 降级 None），不拖垮其余页；保序：候选按页序累积。
    - client 与 proxy-env 在并发块外层一次性构造（同 vl_ocr），worker 只复用；max_retries=2。
    """
    settings = load_settings()
    model = model or settings.dashscope_vl_extract_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
    if not fields_spec or not image_paths:
        return VisionFieldsResult(model=model)
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip vision field extraction")
        return VisionFieldsResult(
            model=model, error=config_missing("DASHSCOPE_API_KEY 缺失，跳过看图抽取")
        )

    labels = page_labels if page_labels is not None else list(range(1, len(image_paths) + 1))
    prompt = _build_prompt(fields_spec)
    keys = list(fields_spec)

    from openai import OpenAI

    compat_url = base_url.replace("/api/v1", "/compatible-mode/v1")
    with sanitized_httpx_proxy_env():
        client = OpenAI(
            api_key=api_key,
            base_url=compat_url,
            timeout=get_timeout_s("DASHSCOPE_TIMEOUT_S", 300.0),
            max_retries=2,
        )
        per_image = map_concurrent(
            lambda item: _read_one(client, model, prompt, item[1]),
            list(zip(labels, image_paths)),
        )

    by_key: dict[str, list[FieldCandidate]] = {k: [] for k in keys}
    usages: list[Optional[dict]] = []
    for label, res in zip(labels, per_image):
        if not res:  # 该页失败（map_concurrent 降级 None）
            continue
        parsed, usage = res
        usages.append(usage)
        for k in keys:
            cand = candidate_from_raw(parsed.get(k), source="vision", page=label)
            if cand is not None:
                by_key[k].append(cand)

    return VisionFieldsResult(
        by_key={k: v for k, v in by_key.items() if v},  # 丢掉无候选的键
        model=model,
        usage=merge_usage(usages),
    )


def _build_prompt(fields_spec: dict[str, str]) -> str:
    fields = "\n".join(f'- "{k}": {desc}' for k, desc in fields_spec.items())
    schema = ",\n".join(
        f'  "{k}": {{"value": <值或 null>, "evidence": <原文片段或 null>}}' for k in fields_spec
    )
    return VISION_EXTRACT_PROMPT.format(fields=fields, schema=schema)


def _read_one(client, model: str, prompt: str, path: Path) -> tuple[dict, Optional[dict]]:
    """看一张图抽字段，返回 (parsed_json, usage)。异常由 map_concurrent 隔离为 None。"""
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": encode_image_data_uri(path)}},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    parsed = _parse_json_loose(resp.choices[0].message.content or "")
    return parsed, _usage_from_openai(resp)


_NULLISH = {"", "null", "none", "n/a", "无", "未知"}


def candidate_from_raw(
    raw: Any, *, source: str, page: Optional[int] = None
) -> Optional[FieldCandidate]:
    """把模型对某 key 的返回（str 或 {value,evidence}）规整为候选；空/null → None。

    文本路（source="text"，无页号）与看图路（source="vision"，带页号）共用同一规整逻辑。
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        value = raw.get("value")
        evidence = str(raw.get("evidence") or "")
    else:
        value, evidence = raw, ""
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in _NULLISH:
        return None
    return FieldCandidate(source=source, value=value, evidence=evidence, page=page)
