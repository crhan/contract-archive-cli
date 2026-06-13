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
from pathlib import Path
from typing import Optional

from ..config import get_timeout_s, load_settings
from ..utils.http_env import sanitized_httpx_proxy_env

logger = logging.getLogger(__name__)


VL_OCR_PAGE_PROMPT = """你是严谨的 OCR 助理。请只转写这张图片中的全部文本，输出简洁 Markdown，不要总结、不要解释、不要编造。

要求：
- 表格尽量转成 Markdown 表格；复杂表格逐行保留字段名和值。
- 保留保险/合同/凭证中的编号、姓名、日期、金额、保障责任、电话、地址等关键字段。
- 看不清的地方写 `[看不清]`，不要猜。
- 只输出本页文本，不要自己加页码标题（调用方会统一加 `## 第 X 页`）。
"""


def ocr_pdf_images_with_vl(
    image_paths: list[Path],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Optional[str]:
    """
    用 DashScope 专用 OCR 模型（OpenAI 兼容口）逐页转写渲染好的 PDF 页图片。

    每页一次请求，拼成 `## 第 X 页` 分隔的 Markdown。单页失败不中断（记 `[看不清]`），
    只有全部页都失败才返回 None，让调用方回退到原 MinerU 路径。无凭证时返回 None。
    """
    if not image_paths:
        return ""

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
    logger.info("[vl-ocr] %s page(s) via %s (逐页)", total, model)

    parts: list[str] = []
    ok_pages = 0
    with sanitized_httpx_proxy_env():
        client = OpenAI(
            api_key=api_key,
            base_url=compat_url,
            timeout=get_timeout_s("DASHSCOPE_TIMEOUT_S", 300.0),
        )
        for idx, path in enumerate(image_paths, 1):
            content = [
                {"type": "image_url", "image_url": {"url": _encode_image(path)}},
                {"type": "text", "text": VL_OCR_PAGE_PROMPT},
            ]
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                )
                page_text = (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - 单页失败不能拖垮整份；全失败才回退 MinerU
                logger.warning("[vl-ocr] page %s/%s failed: %s", idx, total, e)
                page_text = ""
            if page_text:
                ok_pages += 1
            parts.append(f"## 第 {idx} 页\n\n{page_text or '[看不清]'}")

    if ok_pages == 0:
        logger.warning("[vl-ocr] all %s page(s) failed; caller will fall back", total)
        return None
    logger.info("[vl-ocr] done: %s/%s page(s) ok", ok_pages, total)
    return "\n\n".join(parts).strip() or None


def _encode_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"
