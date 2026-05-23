"""
DashScope (阿里百炼) qwen-vl-ocr pipeline。

设计要点：
- qwen-vl-ocr 不收 PDF，必须先 PyMuPDF 拆页
- 每页**双调用**（实测后确认必须）：
    1) ocr_options.task = "document_parsing"
       返回真正的 markdown (含表格/标题结构)
    2) ocr_options.task = "advanced_recognition"
       返回 ocr_result.words_info (含 location 8 值四角点)，text 字段是 OCR JSON 不是 markdown
- model: qwen-vl-ocr-latest（model id 通过 .env 注入）
- API key: 仅从环境变量读取，绝不打印；异常时日志只打 resp 结构关键字段
"""
from __future__ import annotations

import base64
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas import (
    BBox,
    LayoutBlock,
    PipelineMeta,
    PipelineOutput,
    Section,
    StructuredDocument,
)
from ..utils import PageImage, render_pdf_to_images
from .base import BasePipeline

logger = logging.getLogger(__name__)


class DashScopePipeline(BasePipeline):
    name = "dashscope"

    # 渲染 DPI：扫描件源是 400 DPI，重渲染到 200 DPI 对 qwen-vl-ocr 已足够
    # （它会再内部缩放到 min/max_pixels 范围）
    DEFAULT_DPI = 200

    def __init__(
        self,
        device: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dpi: int | None = None,
    ) -> None:
        super().__init__(device=device or "cpu")  # 该路无本地 GPU 需求
        self.model = model or os.getenv("DASHSCOPE_OCR_MODEL", "qwen-vl-ocr-latest")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
        )
        self.dpi = dpi or self.DEFAULT_DPI

        if not self.api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY 未设置，无法调用 qwen-vl-ocr。请在 .env 中填入。"
            )

    # ---------- 主流程 ----------
    def _process(self, pdf_path: Path, work_dir: Path) -> PipelineOutput:
        import dashscope  # lazy import
        from ..schemas import PREVIEW_DIR

        dashscope.base_http_api_url = self.base_url

        # 1) 拆页 → PNG
        preview_dir = work_dir / PREVIEW_DIR
        pages = render_pdf_to_images(pdf_path, preview_dir, dpi=self.dpi)
        logger.info("[dashscope] rendered %d pages @ %d dpi", len(pages), self.dpi)

        # 2) 逐页双调用
        page_markdowns: list[str] = []
        page_raw_texts: list[str] = []
        layout_blocks: list[LayoutBlock] = []

        for p in pages:
            md = self._call_document_parsing(p.image_path)
            blocks = self._call_advanced_recognition(p)
            page_markdowns.append(md)
            page_raw_texts.append(_markdown_to_plain(md))
            layout_blocks.extend(blocks)

        # 3) 归一化
        markdown = "\n\n---\n\n".join(
            f"<!-- page {i + 1} -->\n{md}" for i, md in enumerate(page_markdowns)
        )
        raw_text = "\n".join(page_raw_texts)
        sections = _split_sections_from_markdown(markdown)

        meta = PipelineMeta(
            pipeline_name="dashscope",
            pipeline_version=getattr(dashscope, "__version__", "unknown"),
            model=self.model,
            device=self.device,
            source_pdf=str(pdf_path),
            started_at=datetime.now(),  # 占位，base 会覆盖
            finished_at=datetime.now(),
            duration_seconds=0.0,
            notes=f"dpi={self.dpi}",
        )
        structured = StructuredDocument(
            title=sections[0].title if sections else None,
            document_type=None,  # 后续 extraction 阶段才判定
            language="zh",
            pages=len(pages),
            sections=sections,
        )
        return PipelineOutput(
            meta=meta,
            raw_text=raw_text,
            markdown=markdown,
            layout=layout_blocks,
            structured=structured,
            preview_image_paths=[str(p.image_path) for p in pages],
        )

    # ---------- DashScope 调用 ----------
    def _call_document_parsing(self, image_path: Path) -> str:
        """
        task=document_parsing：返回真正的 markdown（含标题/段落/表格 HTML）。
        """
        import dashscope

        resp = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/png;base64,{_b64(image_path)}",
                            "min_pixels": 3072,
                            "max_pixels": 8388608,
                        },
                        {"text": "Read all the text in the image."},
                    ],
                }
            ],
            ocr_options={"task": "document_parsing"},
        )
        text, _ = _extract_content(resp)
        return text or ""

    def _call_advanced_recognition(self, page: PageImage) -> list[LayoutBlock]:
        """
        task=advanced_recognition：返回 ocr_result.words_info（含 location 8 值四角点）。
        坐标：图像像素 → 换算到 PDF point (× 72 / dpi)。
        """
        import dashscope

        resp = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/png;base64,{_b64(page.image_path)}",
                            "min_pixels": 3072,
                            "max_pixels": 8388608,
                        },
                        {"text": "Read all the text in the image."},
                    ],
                }
            ],
            ocr_options={"task": "advanced_recognition"},
        )

        text, ocr_result = _extract_content(resp)
        words: list[dict] = []
        if isinstance(ocr_result, dict):
            words = ocr_result.get("words_info", []) or []
        elif isinstance(text, str) and text.strip().startswith("["):
            # 部分场景 SDK 把 ocr 数组放到 text 字符串里，做一次解析兜底
            import json as _json
            try:
                words = _json.loads(text)
            except _json.JSONDecodeError:
                pass

        scale = 72.0 / page.dpi  # px → pt
        blocks: list[LayoutBlock] = []
        for w in words:
            location = w.get("location") or []
            rotate_rect = w.get("rotate_rect") or []
            if len(location) == 8:
                xs = location[0::2]
                ys = location[1::2]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            elif len(location) == 4:
                x0, y0, x1, y1 = location
            elif len(rotate_rect) == 5:
                # [cx, cy, w, h, angle]：忽略旋转，取外接矩形
                cx, cy, w_, h_, _angle = rotate_rect
                x0, y0, x1, y1 = cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2
            else:
                continue
            blocks.append(
                LayoutBlock(
                    bbox=BBox(
                        page=page.page_index,
                        x0=x0 * scale,
                        y0=y0 * scale,
                        x1=x1 * scale,
                        y1=y1 * scale,
                    ),
                    text=w.get("text", ""),
                    block_type="paragraph",
                    confidence=w.get("confidence"),
                )
            )
        return blocks


# ---------- 辅助函数 ----------


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_content(resp: Any) -> tuple[str, dict | None]:
    """
    安全地从 DashScope 响应里取 (text, ocr_result)。失败时只打 resp 结构关键字段，
    绝不把整个 resp dump 出来——request_id/headers 里可能带敏感信息。
    """
    try:
        choices = resp["output"]["choices"]
        content = choices[0]["message"]["content"]
        if isinstance(content, str):
            return content, None
        if isinstance(content, list):
            text = ""
            ocr_result: dict | None = None
            for item in content:
                if not isinstance(item, dict):
                    continue
                if "text" in item and not text:
                    text = item["text"]
                if "ocr_result" in item:
                    ocr_result = item["ocr_result"]
            return text, ocr_result
        return "", None
    except (KeyError, IndexError, TypeError):
        # 只打结构信息，不打完整内容
        shape = _shape(resp)
        logger.warning("[dashscope] unexpected response shape: %s", shape)
        return "", None


def _shape(obj: Any, depth: int = 0) -> Any:
    """递归脱敏：只返回类型/keys/列表长度，不保留任何 value。"""
    if depth > 4:
        return "..."
    if isinstance(obj, dict):
        return {k: _shape(v, depth + 1) for k, v in obj.items() if k.lower() not in {"api_key", "authorization", "token", "key"}}
    if isinstance(obj, list):
        return [f"list[{len(obj)}]"]
    return type(obj).__name__


def _markdown_to_plain(md: str) -> str:
    """
    剥离 markdown / LaTeX / HTML 标记，仅保留纯文本——用于 raw_text.txt。
    qwen-vl-ocr document_parsing 的 markdown 实测包含大量 LaTeX
    （\textbf, \begin{tabular}, \\ 换行符），必须一并处理。
    """
    text = md

    # 1) markdown 代码块（保留内容，去掉围栏；qwen 经常把整段包在 ```latex ... ```）
    text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"```", "", text)

    # 2) HTML 注释（我们自己加的 <!-- page N -->）
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 3) LaTeX 命令
    text = re.sub(r"\\begin\{[^}]*\}", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\underline\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\\\", "\n", text)  # LaTeX 换行
    text = re.sub(r"\\hline", "", text)
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?", "", text)  # 兜底：其它 LaTeX 命令
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"&", " ", text)  # 表格列分隔

    # 4) markdown 行内/段落标记
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|.*\|\s*$", lambda m: m.group(0).replace("|", " ").strip(), text, flags=re.MULTILINE)

    # 5) HTML 标签
    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sections_from_markdown(md: str) -> list[Section]:
    """按 # / ## 把 markdown 切成 Section 列表。简陋但够 playground 用。"""
    sections: list[Section] = []
    current_title: str | None = None
    current_level: int = 1
    current_buf: list[str] = []
    current_page_start = 0
    current_page = 0

    for line in md.splitlines():
        page_marker = re.match(r"<!--\s*page\s+(\d+)\s*-->", line)
        if page_marker:
            current_page = int(page_marker.group(1)) - 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            if current_title is not None:
                sections.append(
                    Section(
                        level=current_level,
                        title=current_title,
                        text="\n".join(current_buf).strip(),
                        page_start=current_page_start,
                        page_end=current_page,
                    )
                )
            current_title = heading.group(2)
            current_level = len(heading.group(1))
            current_buf = []
            current_page_start = current_page
        else:
            current_buf.append(line)

    if current_title is not None:
        sections.append(
            Section(
                level=current_level,
                title=current_title,
                text="\n".join(current_buf).strip(),
                page_start=current_page_start,
                page_end=current_page,
            )
        )
    return sections
