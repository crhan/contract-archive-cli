"""
多模态签章核查：对落款页图像调 qwen-vl，确证每个落款区甲/乙方的盖章/签字有无。

为什么要看图：MinerU 把落款签章区当 image 抠出，手写签字和红章都没被 OCR 成文字
（layout 也无 signature/stamp 类型）——纯文本判签章既会误报（签了但读不到）又会漏判。
只有看图才能确证。文本抽取负责要素核查，签章核查交这里。

降级：无落款页图 / 无 key / VL 调用失败时，调用方保留原文本签章判断（不破坏 --no-llm）。
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Optional

from ..config import load_settings
from ..schemas import Completeness, CompletenessIssue, DocumentExtraction
from .llm_extractor import _parse_json_loose

logger = logging.getLogger(__name__)


VL_PROMPT = """你是严谨的合同签章核查员。下面是合同的落款/签署页图像，请逐个落款区核查
每一方的签署情况，只看图、据实判断。

定义：
- 盖章(seal)：该方位置有红色印章图案。
- 签字(signature)：该方位置有手写笔迹姓名。
- 空白：该方位置既无红章也无手写签字。

只输出 JSON，不要解释、不要 markdown 代码块：
{
  "units": [
    {
      "agreement": "落款所属协议（如 主协议 / 补充协议）",
      "parties": [
        {"role": "甲方", "has_seal": true_or_false, "has_signature": true_or_false, "note": "看到的主体名或说明"},
        {"role": "乙方", "has_seal": true_or_false, "has_signature": true_or_false, "note": "..."}
      ]
    }
  ]
}

要点：
- 一份文档可能有多个落款区，不同页通常是不同协议（主协议、补充协议）的落款。
- 红章可能较淡或被文字压住，仔细看；拿不准 has_seal 填 false 并在 note 里说明。
- 手写签字哪怕潦草也算 has_signature=true。
- 只核查"甲方(签章)""乙方(签章)"这类落款签署位，不要把正文印章/骑缝章当落款。
"""


def locate_signature_pages(mineru_dir: Path, max_pages: int = 4) -> list[Path]:
    """从 MinerU content_list 找含'签章'的页，映射到 preview_images/page_NNN.png（1-based）。"""
    preview = mineru_dir / "preview_images"
    if not preview.is_dir():
        return []
    content_lists = list(mineru_dir.glob("_mineru_raw/*/auto/*_content_list.json"))
    if not content_lists:
        return []
    try:
        items = json.loads(content_lists[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取 content_list 失败: %s", e)
        return []
    page_idxs = sorted({
        it["page_idx"]
        for it in items
        if isinstance(it, dict)
        and it.get("page_idx") is not None
        and "签章" in str(it.get("text", ""))
    })
    out: list[Path] = []
    for idx in page_idxs[:max_pages]:
        img = preview / f"page_{idx + 1:03d}.png"
        if img.exists():
            out.append(img)
    return out


def _encode_image(path: Path) -> str:
    """本地图 → data URI。OpenAI 兼容接口不收 file://，用 base64 内联。"""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _call_vl(
    image_paths: list[Path], model: str, api_key: str, base_url: str
) -> Optional[str]:
    """
    走 DashScope 的 OpenAI 兼容接口调多模态模型看落款页图。失败返回 None。

    端点：把原生 base_url 的 /api/v1 换成 /compatible-mode/v1（DashScope OpenAI 兼容模式）。
    图：本地 PNG 转 base64 data URI（兼容接口不支持 file://）。
    """
    from openai import OpenAI

    compat_url = base_url.replace("/api/v1", "/compatible-mode/v1")
    content: list[dict] = [{"type": "text", "text": VL_PROMPT}]
    content.extend(
        {"type": "image_url", "image_url": {"url": _encode_image(p)}} for p in image_paths
    )
    content.append({"type": "text", "text": "请核查以上落款页的签章情况，按要求输出 JSON。"})
    try:
        client = OpenAI(api_key=api_key, base_url=compat_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
        )
    except Exception as e:  # noqa: BLE001 — 外部调用，任何异常都降级，不让它中断入库
        logger.exception("VL 签章核查调用失败: %s", e)
        return None
    return resp.choices[0].message.content


def _issues_from_vision(parsed: dict) -> list[CompletenessIssue]:
    """VL 结果 → 签章缺陷 issues：某方既无章又无签字即为缺。只列缺的。"""
    issues: list[CompletenessIssue] = []
    for unit in parsed.get("units") or []:
        if not isinstance(unit, dict):
            continue
        agreement = str(unit.get("agreement") or "协议").strip()
        for party in unit.get("parties") or []:
            if not isinstance(party, dict):
                continue
            role = str(party.get("role") or "").strip()
            if not role:
                continue
            if not bool(party.get("has_seal")) and not bool(party.get("has_signature")):
                issues.append(CompletenessIssue(
                    item=f"{agreement}·{role}签章",
                    category="signature",
                    detail="落款页图像显示该处空白，无红章也无手写签字",
                ))
    return issues


def augment_completeness_with_vision(env: DocumentExtraction, mineru_dir: Path) -> bool:
    """
    用 VL 看落款页重判签章，替换 env.completeness 的 signature 类 issues（保留 field 类）。

    仅对合同协议生效。成功返回 True；无图 / 无 key / VL 失败返回 False，
    由调用方保留原文本签章判断作降级。
    """
    if env.doc_type != "合同协议":
        return False
    images = locate_signature_pages(mineru_dir)
    if not images:
        logger.info("未定位到落款页图，跳过 VL 签章核查")
        return False
    settings = load_settings()
    if not settings.dashscope_api_key:
        return False
    text = _call_vl(
        images, settings.dashscope_vl_model, settings.dashscope_api_key, settings.dashscope_base_url
    )
    if not text:
        return False
    parsed = _parse_json_loose(text)
    if not parsed:
        logger.warning("VL 签章响应无法解析为 JSON: %s", text[:200])
        return False

    sig_issues = _issues_from_vision(parsed)
    # 保留文本判出的要素(field)缺陷，签章(signature)缺陷整体换成 VL 看图的结果。
    field_issues = [i for i in env.completeness.issues if i.category != "signature"] if env.completeness else []
    all_issues = field_issues + sig_issues
    env.completeness = Completeness(
        status="incomplete" if all_issues else "complete",
        issues=all_issues,
    )
    return True
