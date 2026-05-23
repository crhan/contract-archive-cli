"""
Pipeline 抽象基类。所有 OCR 实现遵循同一份契约：

    run(pdf_path, out_dir) -> PipelineOutput
    + 把 raw_text/markdown/structured/layout/preview_images 全部落盘到 out_dir

不做 "register decorator" 之类的过度设计。Linus 的话：
"我是个该死的实用主义者"。三路而已，直接 if/else 派发。
"""
from __future__ import annotations

import abc
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas import (
    FILE_LAYOUT,
    FILE_MARKDOWN,
    FILE_PIPELINE_META,
    FILE_RAW_TEXT,
    FILE_STRUCTURED,
    PREVIEW_DIR,
    PipelineMeta,
    PipelineOutput,
)
from ..utils import describe_device, select_device

logger = logging.getLogger(__name__)


class BasePipeline(abc.ABC):
    """所有 OCR pipeline 的统一入口。"""

    name: str = "base"

    def __init__(self, device: str | None = None) -> None:
        self.device = select_device(device)
        logger.info("[%s] device = %s", self.name, describe_device(self.device))

    # ---------- 抽象层 ----------
    @abc.abstractmethod
    def _process(self, pdf_path: Path, work_dir: Path) -> PipelineOutput:
        """
        子类实现：跑完整 OCR 流程，把临时文件写到 work_dir，返回内存结果。
        meta.started_at / finished_at / duration_seconds 不需子类填写，由 run() 包装。
        """

    # 已知 pipeline 产物文件，清空目录时只删这些 + preview 子目录
    _OWNED_FILES = {
        FILE_RAW_TEXT,
        FILE_MARKDOWN,
        FILE_STRUCTURED,
        FILE_LAYOUT,
        FILE_PIPELINE_META,
        "extraction_result.json",
        "extraction_confidence.json",
    }
    _OWNED_DIRS = {PREVIEW_DIR, "_paddle_raw", "_mineru_raw"}

    # ---------- 模板方法 ----------
    def run(self, pdf_path: str | Path, out_dir: str | Path) -> PipelineOutput:
        """
        统一入口。负责：
        - 创建输出目录（仅清理 *本 pipeline 写过的文件*，绝不递归删除未知内容）
        - 计时
        - 调用子类 _process
        - 把结果按统一文件名落盘
        """
        pdf_path = Path(pdf_path).resolve()
        out_dir = Path(out_dir).resolve()
        if out_dir.exists():
            # 只删自家产物：白名单文件 + 白名单子目录，绝不 rmtree 未知内容
            for item in out_dir.iterdir():
                if item.is_file() and item.name in self._OWNED_FILES:
                    item.unlink()
                elif item.is_dir() and item.name in self._OWNED_DIRS:
                    shutil.rmtree(item)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / PREVIEW_DIR).mkdir(exist_ok=True)

        started = datetime.now()
        t0 = time.perf_counter()
        try:
            result = self._process(pdf_path, out_dir)
        except Exception as e:
            logger.exception("[%s] pipeline failed", self.name)
            # 即使失败也写一份 meta，方便对比报告知道这一路挂了
            self._dump_failure(out_dir, pdf_path, started, str(e))
            raise
        duration = time.perf_counter() - t0
        finished = datetime.now()

        # 子类可能没填 meta，这里补齐
        result.meta.started_at = started
        result.meta.finished_at = finished
        result.meta.duration_seconds = duration
        if not result.meta.source_pdf:
            result.meta.source_pdf = str(pdf_path)
        if not result.meta.device:
            result.meta.device = self.device

        self._dump(out_dir, result)
        logger.info(
            "[%s] done in %.2fs, output=%s", self.name, duration, out_dir
        )
        return result

    # ---------- 落盘 ----------
    @staticmethod
    def _dump(out_dir: Path, result: PipelineOutput) -> None:
        (out_dir / FILE_RAW_TEXT).write_text(result.raw_text, encoding="utf-8")
        (out_dir / FILE_MARKDOWN).write_text(result.markdown, encoding="utf-8")
        (out_dir / FILE_STRUCTURED).write_text(
            result.structured.model_dump_json(indent=2, exclude_none=False),
            encoding="utf-8",
        )
        (out_dir / FILE_LAYOUT).write_text(
            json.dumps(
                [b.model_dump() for b in result.layout],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / FILE_PIPELINE_META).write_text(
            result.meta.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _dump_failure(
        self, out_dir: Path, pdf_path: Path, started: datetime, err: str
    ) -> None:
        """失败时仍写一份 meta，附错误信息。"""
        meta = PipelineMeta(
            pipeline_name=self.name,  # type: ignore[arg-type]
            source_pdf=str(pdf_path),
            started_at=started,
            finished_at=datetime.now(),
            duration_seconds=(datetime.now() - started).total_seconds(),
            notes=f"FAILED: {err}",
        )
        (out_dir / FILE_PIPELINE_META).write_text(
            meta.model_dump_json(indent=2), encoding="utf-8"
        )


def _safe_dump_json(path: Path, payload: Any) -> None:
    """工具：JSON dump 兼容 pydantic / dataclass / dict / list。"""
    if hasattr(payload, "model_dump_json"):
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    else:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
