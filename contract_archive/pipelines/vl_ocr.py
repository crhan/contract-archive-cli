"""DashScope OCR：用专用 OCR 模型（qwen-vl-ocr）逐页转写 PDF 图片。

为什么逐页：qwen-vl-ocr 是专用 OCR 模型，maxInputTokens 仅 30000，一次塞不下多页
高分辨率图（旧实现把整份 PDF 全部页塞进一个请求，只有上下文极大的通用 VL 模型
qwen3.6-flash 才扛得住，且慢、易超时）。这里改为每页一次调用、拼接结果，既用上专用
OCR 模型，也消除了"页数超上限就回退 mineru"的硬限制。

模型取 settings.dashscope_ocr_model（默认 qwen-vl-ocr-latest）；签章核查仍用 vl_model，
互不影响。
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import NamedTuple, Optional

from ..config import get_timeout_s, load_settings
from ..utils import map_concurrent
from ..utils.http_env import sanitized_httpx_proxy_env

logger = logging.getLogger(__name__)


VL_OCR_PAGE_PROMPT = """你是严谨的 OCR 助理。请只转写这张图片中的全部文本，输出简洁 Markdown，不要总结、不要解释、不要编造。

要求：
- 表格尽量转成 Markdown 表格；复杂表格逐行保留字段名和值。
- 保留保险/合同/凭证中的编号、姓名、日期、金额、保障责任、电话、地址等关键字段。
- 看不清的地方写 `[看不清]`，不要猜。
- 只输出本页文本，不要自己加页码标题（调用方会统一加 `## 第 X 页`）。
"""

# 单页三种异常态各用独立标记，互不混淆 —— 关键是把"请求失败"和"模型识别不清"分开：
#   _MARK_FAILED    请求级失败（SDK 自动重试耗尽后仍抛错）。混进 [看不清] 就永久救不回、
#                   也无法事后审计/单页补跑，所以必须独立。
#   _MARK_TRUNCATED 输出触达模型 8192 token 上限被截断；已得内容保留，但显式标记残页。
#   _MARK_ILLEGIBLE 模型正常返回但本页无可识别文本。
_MARK_FAILED = "[本页 OCR 调用失败]"
_MARK_TRUNCATED = "[本页输出达模型上限被截断]"
_MARK_ILLEGIBLE = "[看不清]"


def _ocr_max_retries() -> int:
    """逐页 OCR 的 SDK 重试次数（CONTRACT_ARCHIVE_VL_OCR_RETRIES 可调，默认 4）。

    openai SDK 对 429/超时/5xx/连接错误本就会自动指数退避重试（且读 Retry-After），
    这里只是把默认的 2 调高 —— 一份文档逐页要发几十上百个请求，偶发限流/抖动的概率
    随页数累积，靠 SDK 多重试几次比手写循环干净，也避免单页偶发失败直接丢一整页内容。
    """
    raw = os.getenv("CONTRACT_ARCHIVE_VL_OCR_RETRIES")
    if not raw or not raw.strip():
        return 4
    try:
        val = int(raw.strip())
    except ValueError:
        logger.warning("CONTRACT_ARCHIVE_VL_OCR_RETRIES=%r 不是整数，回退默认 4", raw)
        return 4
    return val if val >= 0 else 4


class _PageResult(NamedTuple):
    """单页 OCR 结果：正文 + 三态统计标志。

    并发执行下不再在循环里 mutate 计数器（会 race），改为每页返回结构化结果，
    全部跑完后聚合计数——用数据结构消除竞态，而非加锁。
    """

    body: str
    ok: bool
    failed: bool
    truncated: bool


def _ocr_one_page(client, model: str, idx: int, path: Path, total: int) -> _PageResult:
    """转写单页。请求/编码异常隔离到本页（标 _MARK_FAILED），绝不拖垮整份。

    编码（_encode_image）也纳入 try：单张图损坏只标失败页，不再像旧实现那样
    在循环外抛出、崩掉整份 OCR。
    """
    try:
        content = [
            {"type": "image_url", "image_url": {"url": _encode_image(path)}},
            {"type": "text", "text": VL_OCR_PAGE_PROMPT},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        choice = resp.choices[0]
        page_text = (choice.message.content or "").strip()
        truncated = choice.finish_reason == "length"
    except Exception as e:  # noqa: BLE001 - 单页失败不能拖垮整份；全失败才回退 MinerU
        logger.warning("[vl-ocr] page %s/%s failed after retries: %s", idx, total, e)
        return _PageResult(_MARK_FAILED, ok=False, failed=True, truncated=False)

    if truncated:
        # qwen-vl-ocr 单页输出硬上限 8192 token，超了会被静默截断。
        # 保留已得内容（残页也有价值），但显式标记，避免下游把残页当完整页。
        logger.warning(
            "[vl-ocr] page %s/%s truncated at output cap (maxOutputTokens=8192)", idx, total
        )
        page_text = f"{page_text}\n\n{_MARK_TRUNCATED}".strip()

    if page_text:
        return _PageResult(page_text, ok=True, failed=False, truncated=truncated)
    return _PageResult(_MARK_ILLEGIBLE, ok=False, failed=False, truncated=False)


def ocr_pages(
    image_paths: list[Path],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    page_labels: list[int] | None = None,
) -> Optional[list[_PageResult]]:
    """逐页并发 OCR，返回 **与 image_paths 同序** 的 per-page 结果（_PageResult 列表）。

    无凭证 → None（让调用方回退）。空输入 → []。page_labels 仅用于日志页号（缺省 1..N），
    不影响返回顺序——页级混合提取据此把"只 OCR 扫描页"的结果按真实页号拼回全文。

    并发要点：openai SDK 同步阻塞、GIL 在网络等待时释放，线程池即可；client 与 proxy-env
    上下文在并发块外层一次性构造，worker 只复用 client（sanitized_httpx_proxy_env 改的是
    进程级 os.environ，多线程各自进退会竞态）。单页失败已在 _ocr_one_page 内隔离、不返回 None。
    """
    if not image_paths:
        return []

    settings = load_settings()
    model = model or settings.dashscope_ocr_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip VL OCR")
        return None

    from openai import OpenAI

    compat_url = base_url.replace("/api/v1", "/compatible-mode/v1")
    total = len(image_paths)
    labels = page_labels if page_labels is not None else list(range(1, total + 1))
    logger.info("[vl-ocr] %s page(s) via %s (逐页)", total, model)

    with sanitized_httpx_proxy_env():
        client = OpenAI(
            api_key=api_key,
            base_url=compat_url,
            timeout=get_timeout_s("DASHSCOPE_TIMEOUT_S", 300.0),
            max_retries=_ocr_max_retries(),
        )
        return map_concurrent(
            lambda item: _ocr_one_page(client, model, item[0], item[1], total),
            list(zip(labels, image_paths)),
        )


def ocr_pdf_images_with_vl(
    image_paths: list[Path],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Optional[str]:
    """
    用 DashScope 专用 OCR 模型（OpenAI 兼容口）逐页转写渲染好的 PDF 页图片。

    每页一次请求，拼成 `## 第 X 页` 分隔的 Markdown（整份页号 1..N）。单页异常不中断整份，
    三种异常态各记独立标记（调用失败 / 输出截断 / 看不清），只有无任何可用页时才返回 None 让
    调用方回退到原 MinerU 路径。429/超时/5xx 由 SDK 自动重试（见 _ocr_max_retries）。
    无凭证时返回 None。整份 OCR 路径用本函数；页级混合提取用 ocr_pages 拿 per-page 结果。
    """
    if not image_paths:
        return ""
    results = ocr_pages(image_paths, model=model, api_key=api_key, base_url=base_url)
    if results is None:
        return None  # 无凭证

    total = len(results)
    parts = [f"## 第 {i} 页\n\n{r.body}" for i, r in enumerate(results, 1)]
    ok_pages = sum(1 for r in results if r.ok)
    failed_pages = sum(1 for r in results if r.failed)
    truncated_pages = sum(1 for r in results if r.truncated)

    if ok_pages == 0:
        logger.warning(
            "[vl-ocr] no usable page (%s failed / %s total); caller will fall back",
            failed_pages,
            total,
        )
        return None
    logger.info(
        "[vl-ocr] done: %s/%s ok, %s failed, %s truncated",
        ok_pages,
        total,
        failed_pages,
        truncated_pages,
    )
    return "\n\n".join(parts).strip() or None


def _encode_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"
