"""DashScope VL OCR for small PDFs."""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from ..config import get_timeout_s, load_settings
from ..utils.http_env import sanitized_httpx_proxy_env

logger = logging.getLogger(__name__)


VL_OCR_PROMPT = """你是严谨的 OCR 助理。下面是一份 PDF 的逐页图片。
请只根据图片内容转写全文，输出简洁 Markdown，不要总结、不要解释、不要编造。

要求：
- 保留每页页码，用 `## 第 X 页` 分隔。
- 表格尽量转成 Markdown 表格；如果表格很复杂，逐行保留字段名和值。
- 保留保险/合同/凭证中的编号、姓名、日期、金额、保障责任、电话、地址等关键字段。
- 看不清的地方写 `[看不清]`，不要猜。
"""


def ocr_pdf_images_with_vl(
    image_paths: list[Path],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Optional[str]:
    """
    Transcribe rendered PDF pages through DashScope's OpenAI-compatible VL API.

    Returns None when credentials are unavailable or the model call fails, so the
    caller can keep the original MinerU failure path.
    """
    if not image_paths:
        return ""

    settings = load_settings()
    model = model or settings.dashscope_vl_model
    api_key = api_key or settings.dashscope_api_key
    base_url = base_url or settings.dashscope_base_url
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY missing; skip VL OCR")
        return None

    from openai import OpenAI

    compat_url = base_url.replace("/api/v1", "/compatible-mode/v1")
    content: list[dict] = [{"type": "text", "text": VL_OCR_PROMPT}]
    for idx, path in enumerate(image_paths, 1):
        content.append({"type": "text", "text": f"【第 {idx} 页】"})
        content.append(
            {"type": "image_url", "image_url": {"url": _encode_image(path)}}
        )
    content.append({"type": "text", "text": "请按要求输出 Markdown 转写文本。"})

    try:
        with sanitized_httpx_proxy_env():
            client = OpenAI(
                api_key=api_key,
                base_url=compat_url,
                timeout=get_timeout_s("DASHSCOPE_TIMEOUT_S", 300.0),
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                top_p=0.5,
            )
    except Exception as e:  # noqa: BLE001 - fallback must never mask MinerU root cause
        logger.exception("VL OCR fallback failed: %s", e)
        return None

    text = resp.choices[0].message.content or ""
    return text.strip() or None


def _encode_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"
