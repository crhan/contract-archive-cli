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
from ..schemas import (
    Completeness,
    CompletenessIssue,
    DocumentExtraction,
    LabeledValue,
    PersonIdentity,
)
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
      "page": 该落款区所在页码数字（看图前的【第X页】标注，必须填）,
      "parties": [
        {"role": "甲方", "has_seal": true_or_false, "has_signature": true_or_false, "seal_owner": "章上识别到的主体全称（无章填 null）", "seal_text": "章上完整文字，含章类型与编号，如 销售合同专用章 33010000000001（无章填 null）", "signature_name": "手写签字处的姓名（无签字填 null）", "note": "说明"},
        {"role": "乙方", "has_seal": true_or_false, "has_signature": true_or_false, "seal_owner": "...", "seal_text": "...", "signature_name": "...", "note": "..."}
      ]
    }
  ]
}

要点：
- 一份文档可能有多个落款区，不同页通常是不同协议（主协议、补充协议）的落款。
- 每张图前有【第X页】标注，每个 unit 的 page 必须填它所在的那一页，便于追溯出处。
- 红章可能较淡或被文字压住，仔细看；拿不准 has_seal 填 false 并在 note 里说明。
- 有红章时尽量读出章面文字：主体全称填 seal_owner，章类型(公章/合同专用章/财务专用章)
  与编号数字一并填 seal_text。章面模糊就读多少填多少，**禁止编造编号**；无章则两者填 null。
- 手写签字哪怕潦草也算 has_signature=true，并尽量辨识姓名填 signature_name；无签字填 null。
- 只核查"甲方(签章)""乙方(签章)"这类落款签署位，不要把正文印章/骑缝章当落款。
"""


# 落款页标志词：只在签署区出现、正文罕见。不用"盖章"——正文常提"加盖公章"会误判整页。
# "签章"覆盖"甲方(签章)"式落款；"委托代理人/经办人"覆盖认购等用"买受人/出卖人"的落款。
_SIGN_PAGE_MARKERS = ("签章", "委托代理人", "经办人")


def locate_signature_pages(mineru_dir: Path, max_pages: int = 4) -> list[Path]:
    """从 MinerU content_list 找落款页（含签章/委托代理人等标志词），映射到 preview_images/page_NNN.png。"""
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
        and any(m in str(it.get("text", "")) for m in _SIGN_PAGE_MARKERS)
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
    for p in image_paths:
        content.append({"type": "text", "text": f"【第 {_page_no(p)} 页】"})
        content.append({"type": "image_url", "image_url": {"url": _encode_image(p)}})
    content.append({"type": "text", "text": "请逐页核查落款签章，按要求输出 JSON（每个 unit 回填 page）。"})
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


def _page_no(image: Path) -> str:
    """从 preview 文件名 page_NNN.png 提取页码（去前导零）。"""
    return image.stem.replace("page_", "").lstrip("0") or "0"


def _issues_from_vision(parsed: dict, fallback_evidence: str = "") -> list[CompletenessIssue]:
    """
    VL 结果 → 签章缺陷 issues：某方既无章又无签字即为缺，只列缺的。

    出处优先用该 unit 自己回填的 page（各落款区各自归属的页，如主协议→第8页、
    补充协议→第9页）；unit 没给 page 才退回 fallback（所有落款页的笼统出处）。
    """
    issues: list[CompletenessIssue] = []
    for unit in parsed.get("units") or []:
        if not isinstance(unit, dict):
            continue
        agreement = str(unit.get("agreement") or "协议").strip()
        page = str(unit.get("page") or "").strip()
        evidence = f"据落款页图：第 {page} 页" if page else fallback_evidence
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
                    evidence=evidence,
                ))
    return issues


def _signature_evidence(images: list[Path]) -> str:
    """所有落款页的笼统出处（VL 未回填 unit.page 时的兜底）。"""
    return f"据落款页图：第 {'、'.join(_page_no(p) for p in images)} 页"


def read_seals_on_images(
    images: list[Path],
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Optional[dict]:
    """
    对落款页图像调 VL 模型，返回解析后的**完整结构**（units[].parties[] 含
    has_seal/has_signature 及章读数 seal_owner/seal_text/signature_name）。

    这是 VL 看图的单一原始来源：check_seals_on_images（缺陷列表）与
    augment_completeness_with_vision（绑章号 + 跨合同核对）都基于它，避免重复调 VL。

    :param model: 覆盖 VL model（默认 None=走 settings.dashscope_vl_model）。
    :return: 解析后的 dict（无图 → {"units": []}）；无 key / VL 调用失败 /
             响应无法解析返回 None——让调用方据此降级（保留原文本签章判断）。
    """
    if not images:
        return {"units": []}
    settings = load_settings()
    model = model or settings.dashscope_vl_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip VL seal check")
        return None
    text = _call_vl(images, model, api_key, base_url)
    if not text:
        return None
    parsed = _parse_json_loose(text)
    if not parsed:
        logger.warning("VL 签章响应无法解析为 JSON: %s", text[:200])
        return None
    return parsed


def check_seals_on_images(
    images: list[Path],
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Optional[list[CompletenessIssue]]:
    """
    对落款页图像调 VL 模型核查签章，返回签章缺陷 issues（只列缺的）。

    评测据此横向对比不同 VL 模型。内部走 read_seals_on_images（VL 看图的单一来源）。

    :return: 缺陷列表（[] 表示看图后未发现缺签章）；无 key / VL 调用失败 /
             响应无法解析返回 None——让调用方据此降级（保留原文本签章判断）。
    """
    parsed = read_seals_on_images(images, model, api_key, base_url)
    if parsed is None:
        return None
    return _issues_from_vision(parsed, _signature_evidence(images))


def _attach_seal_identities(env: DocumentExtraction, parsed: dict) -> None:
    """
    把 VL 读出的印章读数绑定到头部主体，作为 identifier(label="印章")追加进
    env.person_identities——后续 known_parties reconcile 即自动跨合同核对同一主体的章号一致性。

    用章主体全称(seal_owner)作主体名：章面印的就是主体全称，与头部声明的甲方通常一致，
    这正是"头部主体 ↔ 落款签章"的对应落点。无 owner / 无 seal_text 的不绑（没法核对）。
    """
    by_name = {p.name: p for p in env.person_identities}
    for unit in parsed.get("units") or []:
        if not isinstance(unit, dict):
            continue
        for party in unit.get("parties") or []:
            if not isinstance(party, dict) or not bool(party.get("has_seal")):
                continue
            owner = str(party.get("seal_owner") or "").strip()
            seal_text = str(party.get("seal_text") or "").strip()
            if not owner or not seal_text:
                continue
            pid = by_name.get(owner)
            if pid is None:
                pid = PersonIdentity(
                    name=owner,
                    role=str(party.get("role") or "").strip() or None,
                )
                env.person_identities.append(pid)
                by_name[owner] = pid
            if not any(i.label == "印章" and i.value == seal_text for i in pid.identifiers):
                pid.identifiers.append(LabeledValue(label="印章", value=seal_text))


def _signatory_mismatch_issues(env: DocumentExtraction, parsed: dict) -> list[CompletenessIssue]:
    """
    落款签字人 vs 当事人名单一致性核查：VL 读出的 signature_name 若不在 env.parties 中，
    报疑似代签/冒签/笔误。例：补充协议乙方落款"王五"，而乙方(买受人)是张三、李四。

    保守判定：手写签字 OCR 不可靠，一律标"疑似，需人工复核"；无 signature_name 的不判。
    委托代理人代签等也会触发——作为异常交人工核对是对的（宁可疑，不漏冒签）。
    名字匹配用双向子串（"张三" ↔ "张三（买受人）"算同一人）。
    """
    parties = [p for p in (env.parties or []) if p]
    if not parties:
        return []

    def norm(s: str) -> str:
        return "".join((s or "").split())

    def in_parties(name: str) -> bool:
        n = norm(name)
        return bool(n) and any(n in norm(p) or norm(p) in n for p in parties)

    issues: list[CompletenessIssue] = []
    for unit in parsed.get("units") or []:
        if not isinstance(unit, dict):
            continue
        agreement = str(unit.get("agreement") or "协议").strip()
        page = str(unit.get("page") or "").strip()
        for party in unit.get("parties") or []:
            if not isinstance(party, dict):
                continue
            signer = str(party.get("signature_name") or "").strip()
            if not signer or in_parties(signer):
                continue
            role = str(party.get("role") or "").strip()
            issues.append(CompletenessIssue(
                item=f"{agreement}·{role}落款人与当事人不符",
                category="signature",
                detail=f"落款签字「{signer}」不在当事人名单（{'、'.join(parties)}）中，"
                       "疑似代签/冒签/笔误，需人工复核",
                evidence=f"据落款页图：第 {page} 页" if page else "",
            ))
    return issues


def augment_completeness_with_vision(env: DocumentExtraction, mineru_dir: Path) -> bool:
    """
    用 VL 看落款页重判签章：替换 env.completeness 的 signature 类 issues（保留 field/amount），
    并把读出的印章绑定到头部主体（供 known_parties 跨合同核对章号一致性）。

    仅对合同协议生效。成功返回 True；无图 / 无 key / VL 失败返回 False，
    由调用方保留原文本签章判断作降级。
    """
    if env.doc_type != "合同协议":
        return False
    images = locate_signature_pages(mineru_dir)
    if not images:
        logger.info("未定位到落款页图，跳过 VL 签章核查")
        return False
    parsed = read_seals_on_images(images)
    if parsed is None:
        return False
    sig_issues = _issues_from_vision(parsed, _signature_evidence(images))
    # 落款人 vs 当事人交叉核对：签字人不在名单 → 疑似代签/冒签（也归 signature 类）。
    sig_issues += _signatory_mismatch_issues(env, parsed)
    # 保留文本判出的非签章缺陷（field/amount 等），签章(signature)缺陷整体换成 VL 看图的结果。
    field_issues = [i for i in env.completeness.issues if i.category != "signature"] if env.completeness else []
    all_issues = field_issues + sig_issues
    env.completeness = Completeness(
        status="incomplete" if all_issues else "complete",
        issues=all_issues,
    )
    # 印章读数绑定到头部主体，使 2.7 的 known_parties reconcile 自动跨合同核对章号一致性。
    _attach_seal_identities(env, parsed)
    return True
