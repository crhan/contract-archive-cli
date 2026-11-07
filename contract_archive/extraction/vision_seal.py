"""
多模态签章核查：对落款页图像调 qwen-vl，确证每个落款区甲/乙方的盖章/签字有无。

为什么要看图：手写签字和红章经常不会出现在 OCR 文本里。纯文本判签章既会误报
（签了但读不到）又会漏判。文本抽取负责要素核查，签章核查交这里。

降级：无落款页图 / 无 key / VL 调用失败时，调用方保留原文本签章判断。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from ..config import get_timeout_s, load_settings
from ..errors import ErrorInfo, config_missing, extract_empty
from ..schemas import (
    Completeness,
    CompletenessIssue,
    DocumentExtraction,
    FILE_RAW_TEXT,
    LabeledValue,
    PersonIdentity,
    PREVIEW_DIR,
    StageIssue,
)
from ..utils import (
    NO_THINKING,
    encode_image_data_uri,
    inspect_pdf_pages,
    render_pdf_to_images,
)
from ..utils.llm_runtime import observed_chat_completion
from .llm_extractor import (
    _parse_json_loose,
    output_truncated_error,
    safe_api_error,
)

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
      "agreement": "落款所属协议，**只能填「主协议」或「补充协议」**（多份补充协议依次「补充协议一」「补充协议二」）",
      "page": 该落款区所在页码数字（看图前的【第X页】标注，必须填）,
      "parties": [
        {"role": "甲方", "has_seal": true_or_false, "has_signature": true_or_false, "seal_owner": "章上识别到的主体全称（无章填 null）", "seal_text": "章上完整文字（无章填 null）", "seal_no": "章上的编号数字串，如 33010000000001（读不到填 null）", "signature_name": "手写签字处的姓名（无签字填 null）", "note": "说明"},
        {"role": "乙方", "has_seal": true_or_false, "has_signature": true_or_false, "seal_owner": "...", "seal_text": "...", "signature_name": "...", "note": "..."}
      ]
    }
  ]
}

要点：
- 一份文档可能有多个落款区，不同页通常是不同协议（主协议、补充协议）的落款。
- **agreement 只填「主协议」/「补充协议」这类固定角色名，禁止填文档标题全称**
  （如「地下车位使用权转让协议」是标题，不是角色名，应填「主协议」）。
  只有一个落款区时一律填「主协议」。角色名固定，缺陷条目才跨模型、跨批次可比。
- 每张图前有【第X页】标注，每个 unit 的 page 必须填它所在的那一页，便于追溯出处。
- 红章可能较淡或被文字压住，仔细看；拿不准 has_seal 填 false 并在 note 里说明。
- 有红章时尽量读出章面文字：主体全称填 seal_owner，章类型与全文填 seal_text，
  **章上的编号数字串单独填 seal_no（如 33010000000001）**——编号是跨合同核对的关键。
  章面模糊就读多少填多少，**禁止编造编号**；无章则三者填 null。
- 手写签字哪怕潦草也算 has_signature=true，并尽量辨识姓名填 signature_name；无签字填 null。
- 只核查"甲方(签章)""乙方(签章)"这类落款签署位，不要把正文印章/骑缝章当落款。
"""


# 落款页标志词：只在签署区出现、正文罕见。不用"盖章"——正文常提"加盖公章"会误判整页。
# "签章"覆盖"甲方(签章)"式落款；"委托代理人/经办人"覆盖认购等用"买受人/出卖人"的落款。
_SIGN_PAGE_MARKERS = ("签章", "委托代理人", "经办人")
_PAGE_HEADING_RE = re.compile(r"^##\s*第\s*(\d+)\s*页", re.MULTILINE)


def locate_signature_pages(ocr_dir: Path, max_pages: int = 4) -> list[Path]:
    """定位落款页预览图：优先文本页标题标记，退到文档尾页。"""
    preview = ocr_dir / PREVIEW_DIR
    preview.mkdir(parents=True, exist_ok=True)
    _ensure_tail_previews(ocr_dir, preview, max_pages)

    page_idxs = _signature_page_idxs_from_text(ocr_dir)
    out: list[Path] = []
    for idx in page_idxs:
        img = preview / f"page_{idx + 1:03d}.png"
        if img.exists():
            out.append(img)
    if out:
        return out[:max_pages]
    return sorted(preview.glob("page_*.png"))[-max_pages:]


def _ensure_tail_previews(ocr_dir: Path, preview: Path, max_pages: int) -> None:
    if any(preview.glob("page_*.png")):
        return
    source_pdf = ocr_dir.parent / "source.pdf"
    if not source_pdf.exists():
        return
    try:
        page_infos = inspect_pdf_pages(source_pdf)
        start = max(0, len(page_infos) - max_pages)
        render_pdf_to_images(source_pdf, preview, dpi=160, pages=list(range(start, len(page_infos))))
    except Exception:  # noqa: BLE001 - 签章预览是可选增强，失败即降级
        logger.warning("rendering candidate signature pages failed")


def _signature_page_idxs_from_text(ocr_dir: Path) -> list[int]:
    raw = ocr_dir / FILE_RAW_TEXT
    if not raw.exists():
        return []
    try:
        text = raw.read_text(encoding="utf-8")
    except OSError:
        return []
    matches = list(_PAGE_HEADING_RE.finditer(text))
    page_idxs: list[int] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        if any(marker in body for marker in _SIGN_PAGE_MARKERS):
            page_idxs.append(int(match.group(1)) - 1)
    return sorted(set(idx for idx in page_idxs if idx >= 0))


def _call_vl(
    image_paths: list[Path], model: str, api_key: str, base_url: str
) -> tuple[Optional[str], Optional[ErrorInfo]]:
    """
    走 DashScope 的 OpenAI 兼容接口调多模态模型看落款页图。失败返回安全错误。

    端点：把原生 base_url 的 /api/v1 换成 /compatible-mode/v1（DashScope OpenAI 兼容模式）。
    图：本地 PNG 转 base64 data URI（兼容接口不支持 file://）。
    """
    try:
        # 图像读取/编码、SDK 构造、请求和响应解包都属于可选增强阶段；任一处
        # 失败都必须留下安全 warning，而不能让后续错误地清空文本抽取的签章问题。
        from openai import OpenAI

        compat_url = base_url.replace("/api/v1", "/compatible-mode/v1")
        content: list[dict] = [{"type": "text", "text": VL_PROMPT}]
        for p in image_paths:
            content.append({"type": "text", "text": f"【第 {_page_no(p)} 页】"})
            content.append(
                {"type": "image_url", "image_url": {"url": encode_image_data_uri(p)}}
            )
        content.append(
            {"type": "text", "text": "请逐页核查落款签章，按要求输出 JSON（每个 unit 回填 page）。"}
        )
        # 显式 timeout（默认 300s，DASHSCOPE_TIMEOUT_S 可调）：VL 内联多张落款页大图、
        # 请求体大更易长挂，不设则吃 SDK 默认 ~600s。超时走下面 except 降级返回 None。
        client = OpenAI(
            api_key=api_key, base_url=compat_url,
            timeout=get_timeout_s("DASHSCOPE_TIMEOUT_S", 300.0),
        )
        resp = observed_chat_completion(
            client,
            stage="seal_vision",
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            extra_body=NO_THINKING,
        )
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            return None, output_truncated_error()
        return choice.message.content, None
    except Exception as exc:  # noqa: BLE001 — 外部调用，任何异常都降级，不让它中断入库
        logger.warning("VL seal verification call failed")
        return None, safe_api_error(exc, "VL seal verification request failed")


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
            # _read_seals_on_images_with_issue validates these as real bool values.
            # Keep exact checks here too so a direct/internal caller cannot turn the
            # string "false" into a truthy signed result.
            if party.get("has_seal") is False and party.get("has_signature") is False:
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
    parsed, _ = _read_seals_on_images_with_issue(images, model, api_key, base_url)
    return parsed


def _read_seals_on_images_with_issue(
    images: list[Path],
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[Optional[dict], Optional[ErrorInfo]]:
    """与 public read_seals_on_images 相同，但把可选阶段的安全错误返回给 handler。"""
    if not images:
        return {"units": []}, None
    settings = load_settings()
    model = model or settings.dashscope_vl_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip VL seal check")
        return None, config_missing("DASHSCOPE_API_KEY missing; skip VL seal verification")
    text, error = _call_vl(images, model, api_key, base_url)
    if error is not None:
        return None, error
    if not text:
        return None, extract_empty("LLM returned an empty response")
    parsed = _parse_json_loose(text)
    if not parsed:
        logger.warning("VL seal verification response was not parseable as JSON")
        return None, extract_empty("LLM returned invalid JSON")
    structure_error = _invalid_seal_response_error(parsed)
    if structure_error is not None:
        logger.warning("VL seal verification response had an invalid structure")
        return None, structure_error
    return parsed, None


def _invalid_seal_response_error(parsed: object) -> Optional[ErrorInfo]:
    """Validate the minimum structure needed to replace text-derived seal issues.

    The model may legitimately omit optional prompt fields such as ``agreement``,
    ``page`` and seal-reading metadata.  It may not omit the actual signing verdict:
    every returned unit must contain at least one party, and every party must have a
    non-empty role plus literal JSON booleans for both verdicts.  A partially valid
    response is deliberately rejected as a whole: accepting its good units would
    incorrectly clear issues for the unverified units.
    """
    if not isinstance(parsed, dict):
        return _seal_response_structure_error()
    units = parsed.get("units")
    if not isinstance(units, list) or not units:
        return _seal_response_structure_error()
    for unit in units:
        if not isinstance(unit, dict):
            return _seal_response_structure_error()
        parties = unit.get("parties")
        if not isinstance(parties, list) or not parties:
            return _seal_response_structure_error()
        for party in parties:
            if not isinstance(party, dict):
                return _seal_response_structure_error()
            role = party.get("role")
            if not isinstance(role, str) or not role.strip():
                return _seal_response_structure_error()
            if type(party.get("has_seal")) is not bool:
                return _seal_response_structure_error()
            if type(party.get("has_signature")) is not bool:
                return _seal_response_structure_error()
    return None


def _seal_response_structure_error() -> ErrorInfo:
    """Safe, machine-readable error for a syntactically valid but unusable result."""
    return ErrorInfo(
        code="INVALID_RESPONSE",
        category="validation",
        message="LLM returned an invalid seal verification structure",
        retryable=False,
    )


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


_SEAL_NO_RE = re.compile(r"\d{6,}")  # 章编号：连续 6+ 位数字


def _seal_number(party: dict) -> str:
    """
    章号核对值：优先用 VL 给的 seal_no，否则从 seal_text 提取最长数字串。

    为何用章号而非章面主体名/全文：章号是章上刻死的编号，比 VL 看红章读出的主体名稳定
    （实测同一枚章主体名会被读成"浙典"/"浙奥"，但编号一致）。用编号核对避开这种误差。
    """
    no = str(party.get("seal_no") or "").strip()
    if no:
        return no
    nums = _SEAL_NO_RE.findall(str(party.get("seal_text") or ""))
    return max(nums, key=len) if nums else ""


# 落款 role ↔ 头部 role 同义归组：VL 按 prompt 输出"甲方/乙方"，但头部主体的 role
# 可能是认购协议的"出卖人/买受人"、租赁的"出租方/承租方"等。按"甲方阵营/乙方阵营"
# 归组匹配，章号才能绑到正确头部主体——只死认"甲/乙"字会让认购协议(甲方=出卖人)漏匹配，
# 反被幻觉的"甲方|X"主体截走章号（实测的浙典/浙奥分裂成因之一）。
_ROLE_GROUP_A = ("甲", "出卖", "卖方", "出租", "转让", "出借", "供方", "发包")
_ROLE_GROUP_B = ("乙", "买受", "买方", "承租", "受让", "借款", "需方", "承包")
_SINGLE_PAGE_EVIDENCE_RE = re.compile(
    r"^\s*(?:据落款页图\s*[：:]?\s*)?第\s*(\d+)\s*页\s*$"
)
_SUPPLEMENT_AGREEMENT_RE = re.compile(
    r"^补充协议(?:[一二三四五六七八九十百千]+|\d+|[（(](?:[一二三四五六七八九十百千]+|\d+)[）)])?$"
)


def _role_group(role: str) -> str:
    """把 role 归到甲方阵营('A') / 乙方阵营('B')；两边都不沾返回 ''。"""
    text = role or ""
    if any(k in text for k in _ROLE_GROUP_A):
        return "A"
    if any(k in text for k in _ROLE_GROUP_B):
        return "B"
    return ""


def _coverage_role_group(role: object) -> str:
    """Return an unambiguous A/B role group for destructive issue replacement.

    ``_role_group`` intentionally has a permissive first-match behaviour for
    attaching optional seal identifiers.  Clearing a pre-existing issue needs a
    stricter proof: a phrase containing aliases from both camps (for example
    ``甲乙双方``) has no unique party slot and must not authorise a clear.
    """
    if not isinstance(role, str):
        return ""
    text = role.strip()
    if not text:
        return ""
    is_a = any(alias in text for alias in _ROLE_GROUP_A)
    is_b = any(alias in text for alias in _ROLE_GROUP_B)
    if is_a == is_b:
        return ""
    # Keep the existing role normaliser as the source of the returned group.
    return _role_group(text)


def _coverage_agreement(value: object) -> str:
    """Map only unambiguous model agreement labels to a stable slot kind."""
    if not isinstance(value, str):
        return ""
    label = "".join(value.split())
    if label in {"主协议", "主合同", "本协议"}:
        return "main"
    if _SUPPLEMENT_AGREEMENT_RE.fullmatch(label):
        return "supplement"
    return ""


def _coverage_page(value: object) -> int | None:
    """Parse a positive page number without accepting booleans or fractional data."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        page = int(value.strip())
        return page if page > 0 else None
    return None


def _image_pages(images: list[Path]) -> set[int]:
    """Return only valid positive page numbers represented by this VL request."""
    pages: set[int] = set()
    for image in images:
        page = _coverage_page(_page_no(image))
        if page is not None:
            pages.add(page)
    return pages


def _old_signature_slot(
    issue: CompletenessIssue, *, has_sub_agreements: bool
) -> tuple[str, int, str] | None:
    """Decode one existing signature issue into a trusted `(agreement, page, role)` slot.

    The initial text extraction can be wrong, so it only supplies an issue to
    replace when its evidence names exactly one page and its item identifies a
    party.  Agreement omission is safe to infer as the main agreement only if the
    envelope has no sub-agreements; otherwise it could refer to any signing unit.
    """
    evidence = _SINGLE_PAGE_EVIDENCE_RE.fullmatch(issue.evidence or "")
    if evidence is None:
        return None
    page = _coverage_page(evidence.group(1))
    if page is None:
        return None

    parts = [part.strip() for part in (issue.item or "").split("·")]
    if len(parts) == 1:
        role_text = parts[0]
        # A one-part item is only an omitted agreement, not a malformed embedded
        # agreement label.  Refuse text such as "主协议甲方签章" with no delimiter.
        if has_sub_agreements or "协议" in role_text or "合同" in role_text:
            return None
        agreement = "main"
    elif len(parts) == 2 and all(parts):
        agreement = _coverage_agreement(parts[0])
        role_text = parts[1]
        if not agreement:
            return None
    else:
        return None

    role = _coverage_role_group(role_text)
    if not role:
        return None
    return agreement, page, role


def _vision_slots_with_clearance(
    parsed: dict, image_pages: set[int]
) -> set[tuple[str, int, str]] | None:
    """Return uniquely represented, in-request VL signing slots.

    Every response unit and party must map to a trusted slot.  A single unknown
    agreement/role/page, out-of-request page, or repeated slot invalidates the
    whole response: a valid subset cannot authorise deleting any old issue.
    """
    slots: set[tuple[str, int, str]] = set()
    for unit in parsed.get("units") or []:
        if not isinstance(unit, dict):
            return None
        agreement = _coverage_agreement(unit.get("agreement"))
        page = _coverage_page(unit.get("page"))
        if not agreement or page is None or page not in image_pages:
            return None
        parties = unit.get("parties")
        if not isinstance(parties, list) or not parties:
            return None
        for party in parties:
            if not isinstance(party, dict):
                return None
            role = _coverage_role_group(party.get("role"))
            if not role:
                return None
            slot = agreement, page, role
            if slot in slots:
                return None
            slots.add(slot)
    return slots


def _has_complete_signature_coverage(
    env: DocumentExtraction, parsed: dict, images: list[Path]
) -> bool:
    """Whether every pre-existing signature issue has one trusted VL slot."""
    available = _vision_slots_with_clearance(parsed, _image_pages(images))
    if available is None:
        return False
    if env.completeness is None:
        return True
    old_issues = [issue for issue in env.completeness.issues if issue.category == "signature"]
    if not old_issues:
        return True

    has_sub_agreements = bool(env.sub_agreements)
    matched_old_slots: set[tuple[str, int, str]] = set()
    for issue in old_issues:
        slot = _old_signature_slot(issue, has_sub_agreements=has_sub_agreements)
        if slot is None or slot not in available or slot in matched_old_slots:
            return False
        matched_old_slots.add(slot)
    return True


def _incomplete_coverage_error() -> ErrorInfo:
    """Safe warning for a response that cannot replace all old signature issues."""
    return ErrorInfo(
        code="INCOMPLETE_COVERAGE",
        category="validation",
        message="Vision seal response did not cover every existing signature issue",
        retryable=False,
    )


def _match_head_party(env: DocumentExtraction, vl_role: str) -> Optional[PersonIdentity]:
    """
    VL 落款 role(甲方/乙方) → 头部 person_identity（按甲/乙阵营同义归组匹配）。

    头部主体名来自正文抽取（稳定、有明确指向），是"头部主体 ↔ 落款签章"对应关系的锚点——
    比 VL 看红章读出的章面主体名可靠得多。认购协议头部用"出卖人/买受人"而 VL 用"甲方/乙方"，
    故按阵营归组而非死认"甲/乙"字。匹配不到则返回 None，由调用方兜底。
    """
    group = _role_group(vl_role)
    if not group:
        return None
    for p in env.person_identities:
        if _role_group(p.role or "") == group:
            return p
    return None


def _attach_seal_identities(env: DocumentExtraction, parsed: dict) -> None:
    """
    把 VL 读出的章号绑定到**头部声明的对应主体**（按 role 匹配），作 identifier(label="印章")
    追加进 env.person_identities——后续 known_parties reconcile 即自动跨合同核对章号一致性。

    为何绑头部主体而非章面 owner：头部主体名来自正文 OCR（稳定、有明确指向），VL 看红章读出的
    章面主体名常误读（"浙典"→"浙奥"）。用稳定的头部主体名作锚点、章号编号作核对值，才是用户
    要的"头部主体 ↔ 落款签章"对应。头部匹配不到对应方时，退回用章面 owner 兜底（有总比丢好）。
    """
    for unit in parsed.get("units") or []:
        if not isinstance(unit, dict):
            continue
        for party in unit.get("parties") or []:
            if not isinstance(party, dict) or party.get("has_seal") is not True:
                continue
            seal_no = _seal_number(party)
            if not seal_no:
                continue
            role = str(party.get("role") or "").strip()
            pid = _match_head_party(env, role)
            if pid is None:
                owner = str(party.get("seal_owner") or "").strip()
                if not owner:
                    continue
                pid = PersonIdentity(name=owner, role=role or None)
                env.person_identities.append(pid)
            if not any(i.label == "印章" and i.value == seal_no for i in pid.identifiers):
                pid.identifiers.append(LabeledValue(label="印章", value=seal_no))


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


def augment_completeness_with_vision(env: DocumentExtraction, ocr_dir: Path) -> bool:
    """
    用 VL 看落款页重判签章：只在覆盖每一条旧签章 issue 后才替换它们（保留 field/amount），
    并把读出的印章绑定到头部主体（供 known_parties 跨合同核对章号一致性）。

    仅对合同协议生效。成功返回 True；无图 / 无 key / VL 失败返回 False，
    由调用方保留原文本签章判断作降级。
    """
    if env.doc_type != "合同协议":
        return False
    images = locate_signature_pages(ocr_dir)
    if not images:
        logger.info("未定位到落款页图，跳过 VL 签章核查")
        return False
    parsed, error = _read_seals_on_images_with_issue(images)
    if error is not None:
        # 签章看图是可选增强：绝不清掉文本抽取出的 completeness/原字段，只留可审计 warning。
        env.stage_issues.append(
            StageIssue(stage="seal_vision", error=error, severity="warning")
        )
    if parsed is None:
        return False

    # Replacing text-derived signature issues is destructive.  Do it only if the
    # response contains one unambiguous, requested-image slot for every old issue;
    # a partial response must not clear its covered subset either.
    if not _has_complete_signature_coverage(env, parsed, images):
        env.stage_issues.append(
            StageIssue(
                stage="seal_vision",
                error=_incomplete_coverage_error(),
                severity="warning",
            )
        )
        return False

    sig_issues = _issues_from_vision(parsed, _signature_evidence(images))
    # 落款人 vs 当事人交叉核对：签字人不在名单 → 疑似代签/冒签（也归 signature 类）。
    sig_issues += _signatory_mismatch_issues(env, parsed)

    old_completeness = env.completeness
    old_signature_issues = (
        [issue for issue in old_completeness.issues if issue.category == "signature"]
        if old_completeness
        else []
    )
    non_signature_issues = (
        [issue for issue in old_completeness.issues if issue.category != "signature"]
        if old_completeness
        else []
    )

    if old_signature_issues:
        # Full coverage authorises replacing all old signature issues as a unit.
        # Non-signature issues survive unchanged; only an issue-free result may
        # become complete.
        all_issues = non_signature_issues + sig_issues
        env.completeness = Completeness(
            status="incomplete" if all_issues else "complete",
            issues=all_issues,
        )
    elif sig_issues:
        # With no old signature issue there is nothing to clear.  A newly observed
        # defect can still make an unknown/empty completeness result incomplete.
        env.completeness = Completeness(
            status="incomplete",
            issues=non_signature_issues + sig_issues,
        )
    # If there were no old signature issues and vision found none, retain the
    # original status exactly.  In particular unknown/None must not become
    # complete just because the response happened to be all-positive.

    # 印章读数绑定到头部主体，使 2.7 的 known_parties reconcile 自动跨合同核对章号一致性。
    _attach_seal_identities(env, parsed)
    return True
