"""
MinerU 3.x pipeline。

调用方式：subprocess 调 `mineru` CLI（Python API 在 3.x 不稳）。
mineru 3.x CLI 的输出目录约定（注意：和 2.x 不同！）：
    <out_dir>/<pdf_stem>/<auto|vlm>/
        ├── <stem>.md                     # 主 markdown（不是 full.md）
        ├── <stem>_content_list.json      # 结构化元素列表
        ├── <stem>_layout.pdf
        ├── <stem>_model.json
        ├── <stem>_middle.json
        └── images/

content_list.json 元素的 bbox 是 **归一化到 0-1000 整数**，不是 PDF point。
我们把它换算回 PDF point（× page_width_pt / 1000）以与其他 pipeline 对齐。

历史：原本与 DashScope/PaddleOCR pipeline 共享一个 BasePipeline 抽象基类。
重构后唯一具体实现，抽象基类已 inline 到本文件，避免一抽象一具体的反模式。
"""
from __future__ import annotations

import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from ..config import get_timeout_s
from ..schemas import (
    BBox,
    FILE_LAYOUT,
    FILE_MARKDOWN,
    FILE_PIPELINE_META,
    FILE_RAW_TEXT,
    FILE_STRUCTURED,
    LayoutBlock,
    PipelineMeta,
    PipelineOutput,
    PREVIEW_DIR,
    Section,
    StructuredDocument,
    Table,
)
from ..utils import (
    PdfPageInfo,
    TextLayerStats,
    analyze_text_layer,
    describe_device,
    extract_text_layer,
    inspect_pdf_pages,
    is_text_layer_usable,
    render_pdf_to_images,
    select_device,
)
from ..utils.http_env import sanitize_no_proxy_for_httpx
from .vl_ocr import ocr_pdf_images_with_vl

logger = logging.getLogger(__name__)


# MinerU 3.x content_list.json 中 "type" 字段到统一 schema 的映射
# 注意：MinerU 3.x 没有独立的 "title" 类型，标题是 type:"text" + text_level>=1
_MINERU_TYPE_MAP = {
    "text": "paragraph",
    "image": "figure",
    "table": "table",
    "equation": "formula",
    "list": "list",
    "code": "paragraph",
    "seal": "stamp",
    "chart": "figure",
    "header": "header",
    "footer": "footer",
    "page_number": "footer",
    "aside_text": "paragraph",
    "page_footnote": "footer",
}


# 已知 pipeline 产物文件，清空目录时只删这些 + 这些子目录，绝不 rmtree 未知内容
_OWNED_FILES = {
    FILE_RAW_TEXT,
    FILE_MARKDOWN,
    FILE_STRUCTURED,
    FILE_LAYOUT,
    FILE_PIPELINE_META,
    "extraction_result.json",
    "extraction_confidence.json",
}
_OWNED_DIRS = {PREVIEW_DIR, "_mineru_raw"}


class MinerUPipeline:
    """MinerU 3.x PDF 解析。单一职责：run(pdf, out_dir) → PipelineOutput。"""

    # Historical class name stays for API compatibility. User-facing logs use
    # "ocr" because this pipeline may run native text, DashScope VL, or MinerU.
    name = "ocr"

    def __init__(
        self,
        device: str | None = None,
        backend: str | None = None,
        dpi: int = 200,
        prefer_text_layer: bool = True,
        allow_vl_fallback: bool = True,
        prefer_vl_ocr: bool | None = None,
        lite_retry: bool | None = None,
        vl_ocr_max_pages: int | None = None,
        vl_ocr_dpi: int | None = None,
    ) -> None:
        self.device = select_device(device)
        logger.info("[%s] device = %s", self.name, describe_device(self.device))
        # MinerU 3.x backend 合法值（实测）：
        #   pipeline                 CPU 兜底，兼容性最好
        #   hybrid-auto-engine       3.x 默认，混合方案
        #   hybrid-http-client       走 http server
        #   vlm-auto-engine          GPU VLM 推理
        #   vlm-http-client          走 http server
        # 默认策略：CUDA → vlm-auto-engine，其它（CPU/MPS）→ pipeline
        self.backend = backend or ("vlm-auto-engine" if self.device == "cuda" else "pipeline")
        self.dpi = dpi
        self.prefer_text_layer = prefer_text_layer
        self.allow_vl_fallback = allow_vl_fallback
        self.prefer_vl_ocr = (
            _env_bool("CONTRACT_ARCHIVE_VL_OCR_FIRST", True)
            if prefer_vl_ocr is None
            else prefer_vl_ocr
        )
        self.lite_retry = (
            _env_bool("CONTRACT_ARCHIVE_MINERU_LITE_RETRY", True)
            if lite_retry is None
            else lite_retry
        )
        self.vl_ocr_max_pages = vl_ocr_max_pages or _env_int(
            "CONTRACT_ARCHIVE_VL_OCR_MAX_PAGES", 10
        )
        self.vl_ocr_dpi = vl_ocr_dpi or _env_int("CONTRACT_ARCHIVE_VL_OCR_DPI", 160)

    # ---------- 入口 ----------
    def run(self, pdf_path: str | Path, out_dir: str | Path) -> PipelineOutput:
        """
        统一入口：
        - 清理本 pipeline 写过的旧产物（白名单，绝不递归删未知内容）
        - 计时 + 调 _process
        - 按统一文件名落盘
        """
        pdf_path = Path(pdf_path).resolve()
        out_dir = Path(out_dir).resolve()
        if out_dir.exists():
            for item in out_dir.iterdir():
                if item.is_file() and item.name in _OWNED_FILES:
                    item.unlink()
                elif item.is_dir() and item.name in _OWNED_DIRS:
                    shutil.rmtree(item)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / PREVIEW_DIR).mkdir(exist_ok=True)

        started = datetime.now()
        t0 = time.perf_counter()
        try:
            result = self._process(pdf_path, out_dir)
        except Exception as e:
            logger.exception("[%s] pipeline failed", self.name)
            _dump_failure(out_dir, pdf_path, started, str(e))
            raise
        duration = time.perf_counter() - t0
        finished = datetime.now()

        result.meta.started_at = started
        result.meta.finished_at = finished
        result.meta.duration_seconds = duration
        if not result.meta.source_pdf:
            result.meta.source_pdf = str(pdf_path)
        if not result.meta.device:
            result.meta.device = self.device

        _dump(out_dir, result)
        logger.info("[%s] done in %.2fs, output=%s", self.name, duration, out_dir)
        return result

    # ---------- 实现 ----------
    def _process(self, pdf_path: Path, work_dir: Path) -> PipelineOutput:
        page_infos = inspect_pdf_pages(pdf_path)
        text_stats = analyze_text_layer(pdf_path)

        if self.prefer_text_layer and is_text_layer_usable(text_stats):
            logger.info(
                "[pdf] using native text layer: chars=%s printable=%.2f cjk=%.2f",
                text_stats.non_ws_chars,
                text_stats.printable_ratio,
                text_stats.cjk_ratio,
            )
            text = extract_text_layer(pdf_path)
            return _output_from_text(
                pdf_path=pdf_path,
                page_infos=page_infos,
                text=text,
                device=self.device,
                source="native-text-layer",
                notes=_text_stats_note("backend=native-text-layer", text_stats),
            )

        logger.info(
            "[pdf] native text layer unusable: chars=%s printable=%.2f cjk=%.2f control=%.2f",
            text_stats.non_ws_chars,
            text_stats.printable_ratio,
            text_stats.cjk_ratio,
            text_stats.control_ratio,
        )

        # 1) preview images（独立于 MinerU 内部产物，供下游审阅）
        # 旧实现会在 OCR 前无条件渲染整份 PDF。大 PDF 会先产生大量 PNG 和内存压力，
        # 即使后续 OCR 失败也白做。现在先跑 OCR；成功后再尽力渲染 preview。
        preview_dir = work_dir / PREVIEW_DIR

        if self.prefer_vl_ocr:
            reason = (
                "native text layer unusable"
                if self.prefer_text_layer
                else "native text layer skipped"
            )
            vl_first = self._try_vl_ocr(
                pdf_path,
                work_dir,
                page_infos,
                text_stats,
                f"{reason}; CONTRACT_ARCHIVE_VL_OCR_FIRST enabled",
                source="vl-ocr-first",
            )
            if vl_first is not None:
                return vl_first

        # 2) 调用 mineru CLI
        mineru_out = work_dir / "_mineru_raw"
        mineru_out.mkdir(exist_ok=True)
        env = _mineru_subprocess_env(os.environ)
        env.setdefault("MINERU_MODEL_SOURCE", "modelscope")  # 国内更快

        cmd = [
            _resolve_mineru(),
            "-p",
            str(pdf_path),
            "-o",
            str(mineru_out),
            "-b",
            self.backend,
        ]
        logger.info("[mineru-cli] running: %s", " ".join(cmd))
        # 显式 timeout（默认 1800s，CONTRACT_ARCHIVE_MINERU_TIMEOUT_S 可调）：MinerU 跑深度模型，
        # 畸形/超大 PDF 可能永久挂死——子进程不抛异常，subprocess.run 会无限阻塞，
        # 批量串行 ingest 就此冻死。TimeoutExpired 会自动 kill 该子进程（精确，不波及他人），
        # 转成 RuntimeError 走 run() 的失败落盘 + ingest 的 mineru_failed 分类。
        timeout_s = get_timeout_s("CONTRACT_ARCHIVE_MINERU_TIMEOUT_S", 1800.0)
        try:
            proc = _run_mineru_cli(cmd, env, timeout_s)
        except subprocess.TimeoutExpired as e:
            reason = _mineru_timeout_reason(pdf_path, timeout_s, text_stats)
            retry = self._try_mineru_lite_retry(
                pdf_path, mineru_out, env, timeout_s, reason
            )
            if retry is not None:
                proc = retry
            else:
                fallback = self._try_vl_ocr(
                    pdf_path, work_dir, page_infos, text_stats, reason
                )
                if fallback is not None:
                    return fallback
                raise RuntimeError(reason) from e
        if proc.returncode != 0:
            retry = self._try_mineru_lite_retry(
                pdf_path,
                mineru_out,
                env,
                timeout_s,
                _mineru_failure_reason(proc),
            )
            if retry is not None:
                proc = retry

        if proc.returncode != 0:
            logger.error("[mineru-cli] stdout=%s", proc.stdout[-2000:])
            logger.error("[mineru-cli] stderr=%s", proc.stderr[-2000:])
            failure = _mineru_failure_reason(proc)
            fallback = self._try_vl_ocr(
                pdf_path, work_dir, page_infos, text_stats, failure
            )
            if fallback is not None:
                return fallback
            raise RuntimeError(failure)

        # 3) 找到 MinerU 实际写入的目录
        result_dir = _locate_mineru_result(mineru_out, pdf_path.stem)
        if result_dir is None:
            raise RuntimeError(
                f"MinerU output not found under {mineru_out}; stdout={proc.stdout[-500:]}"
            )

        # 4) 读 markdown / content_list.json
        # MinerU 3.x 文件名是 {stem}.md / {stem}_content_list.json
        stem = pdf_path.stem
        candidates_md = [result_dir / f"{stem}.md", result_dir / "full.md"]
        candidates_cl = [
            result_dir / f"{stem}_content_list.json",
            result_dir / "content_list.json",
        ]
        md_path = next((p for p in candidates_md if p.exists()), candidates_md[0])
        cl_path = next((p for p in candidates_cl if p.exists()), candidates_cl[0])
        markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        content_list = (
            json.loads(cl_path.read_text(encoding="utf-8")) if cl_path.exists() else []
        )

        # 拿到每页 PDF point 尺寸用于 bbox 归一化反算。这里不需要整页渲染。
        page_dims = {p.page_index: (p.width_pt, p.height_pt) for p in page_infos}

        layout_blocks, tables, raw_text = _normalize_mineru(content_list, page_dims)
        # MinerU 的 markdown 会对 _/*/[]/() 等做反斜杠转义（如 "\_2027年\_6月"），
        # 喂给 rule/LLM 之前清掉这些转义，避免 extraction 抓不到
        raw_text = _unescape_markdown(raw_text)
        markdown_for_extract = _unescape_markdown(markdown)
        sections = _split_sections(markdown_for_extract)

        structured = StructuredDocument(
            title=sections[0].title if sections else None,
            document_type=None,
            language="zh",
            pages=len(page_infos),
            sections=sections,
            tables=tables,
        )

        meta = PipelineMeta(
            pipeline_name="mineru",
            pipeline_version=_mineru_version(),
            model="MinerU",
            device=self.device,
            source_pdf=str(pdf_path),
            started_at=datetime.now(),
            finished_at=datetime.now(),
            duration_seconds=0.0,
            notes=f"backend={self.backend}, model_source=modelscope",
        )

        # MinerU 成功后再尽力渲染 preview；渲染失败不应推翻已完成的 OCR 产物。
        pages = _render_previews_safe(pdf_path, preview_dir, self.dpi)

        # 5) 复制 MinerU 自己渲染的 images 到 preview 目录
        mineru_images = result_dir / "images"
        if mineru_images.exists():
            dst = preview_dir / "mineru_images"
            dst.mkdir(exist_ok=True)
            for f in mineru_images.iterdir():
                shutil.copy(f, dst / f.name)

        return PipelineOutput(
            meta=meta,
            raw_text=raw_text,
            markdown=markdown,
            layout=layout_blocks,
            structured=structured,
            preview_image_paths=[str(p.image_path) for p in pages],
        )

    def _try_mineru_lite_retry(
        self,
        pdf_path: Path,
        mineru_out: Path,
        env: dict[str, str],
        timeout_s: float,
        first_failure: str,
    ) -> subprocess.CompletedProcess | None:
        """Retry slow/fragile PDFs with a cheaper MinerU OCR profile before VL fallback."""
        if not self.lite_retry:
            return None
        if self.backend == "pipeline":
            logger.warning("[mineru-lite] retrying after failure: %s", first_failure)
            if mineru_out.exists():
                shutil.rmtree(mineru_out)
            mineru_out.mkdir(exist_ok=True)
            cmd = [
                _resolve_mineru(),
                "-p",
                str(pdf_path),
                "-o",
                str(mineru_out),
                "-b",
                "pipeline",
                "-m",
                "ocr",
                "-l",
                "ch_lite",
                "-f",
                "false",
                "-t",
                "false",
                "--image-analysis",
                "false",
            ]
            try:
                proc = _run_mineru_cli(cmd, env, timeout_s)
            except subprocess.TimeoutExpired:
                logger.warning("[mineru-lite] retry timed out after %.0fs", timeout_s)
                return None
            if proc.returncode == 0:
                logger.info("[mineru-lite] retry succeeded")
                return proc
            logger.warning("[mineru-lite] retry failed: %s", _mineru_failure_reason(proc))
        return None

    def _try_vl_ocr(
        self,
        pdf_path: Path,
        work_dir: Path,
        page_infos: list[PdfPageInfo],
        text_stats: TextLayerStats,
        reason: str,
        source: str = "vl-ocr-fallback",
    ) -> PipelineOutput | None:
        """Use DashScope VL OCR for small PDFs when remote OCR is allowed."""
        if not self.allow_vl_fallback:
            return None
        if len(page_infos) > self.vl_ocr_max_pages:
            logger.warning(
                "[vl-ocr] skip fallback: pages=%s exceeds max=%s",
                len(page_infos),
                self.vl_ocr_max_pages,
            )
            return None

        logger.info("[%s] trying DashScope VL OCR: %s", source, reason)
        preview_dir = work_dir / PREVIEW_DIR
        pages = _render_previews_safe(pdf_path, preview_dir, self.vl_ocr_dpi)
        if not pages:
            return None
        text = ocr_pdf_images_with_vl([p.image_path for p in pages])
        if not text:
            return None
        return _output_from_text(
            pdf_path=pdf_path,
            page_infos=page_infos,
            text=text,
            device=self.device,
            source=source,
            notes=_text_stats_note(
                f"backend={source}, reason={reason}",
                text_stats,
            ),
            preview_image_paths=[str(p.image_path) for p in pages],
        )


# ---------- 落盘工具 ----------


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
    out_dir: Path, pdf_path: Path, started: datetime, err: str
) -> None:
    """失败时仍写一份 meta，附错误信息。"""
    meta = PipelineMeta(
        pipeline_name="mineru",
        source_pdf=str(pdf_path),
        started_at=started,
        finished_at=datetime.now(),
        duration_seconds=(datetime.now() - started).total_seconds(),
        notes=f"FAILED: {err}",
    )
    (out_dir / FILE_PIPELINE_META).write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )


# ---------- 解析 / 归一化 ----------


def _locate_mineru_result(out_root: Path, stem: str) -> Path | None:
    """
    MinerU 3.x 输出位置约定：<out>/<stem>/<auto|vlm>/。
    主 markdown 文件名是 {stem}.md（不是 full.md）。
    """
    candidates = [
        out_root / stem / "auto",
        out_root / stem / "vlm",
        out_root / stem,
    ]
    expected_md = (f"{stem}.md", "full.md")  # 兼容 2.x 旧目录
    for c in candidates:
        if c.exists() and any((c / name).exists() for name in expected_md):
            return c
    # 兜底：递归找 {stem}.md
    for p in out_root.rglob(f"{stem}.md"):
        return p.parent
    return None


def _resolve_mineru() -> str:
    """
    定位 mineru 可执行文件的绝对路径。

    为什么不能直接用裸字符串 "mineru" 交给 subprocess：
    contract-archive 经 `uv tool install` 安装在隔离 venv 里，mineru 作为同一个 venv 的
    依赖（extra）一起装。但该 venv 的 bin/ 目录**不在**用户 shell 的 PATH 上——
    contract-archive 通过 ~/.local/bin 的 symlink 启动，子进程继承的是 shell PATH，
    于是靠 PATH 解析 "mineru" 必然 FileNotFoundError。

    策略（确定性优先，消除"PATH 里必须有 mineru"这个隐含前提）：
      1. 找与当前解释器同目录的兄弟可执行文件——uv tool / 已激活 venv 场景下，
         mineru 与 python 必然同在一个 bin/，这一步直接命中。
      2. 兜底 shutil.which，兼容用户把 mineru 手动放进 PATH 的开发环境。

    返回：mineru 可执行文件路径。
    抛出：FileNotFoundError（附安装指引），比裸的 [Errno 2] 可读得多。
    """
    sibling = Path(sys.executable).parent / "mineru"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("mineru")
    if found:
        return found
    raise FileNotFoundError(
        "找不到 mineru 可执行文件。它随 contract-archive-cli 的 mineru extra 一起安装：\n"
        "  uv tool install 'contract-archive-cli[mineru]'        # 首次安装\n"
        "  uv tool install 'contract-archive-cli[mineru]' --reinstall   # 已装过 contract-archive-cli 但缺 mineru\n"
        "开发环境：uv sync --extra mineru"
    )


def _mineru_version() -> str:
    try:
        # --version 探测给个短 timeout（30s）：仅是探测，挂住没意义；
        # TimeoutExpired 是 Exception 子类，被下面 except 兜成 "unknown"。
        proc = subprocess.run(
            [_resolve_mineru(), "--version"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        return (proc.stdout or proc.stderr).strip()
    except Exception:
        return "unknown"


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _mineru_subprocess_env(source: Mapping[str, str]) -> dict[str, str]:
    """Build the environment passed to the MinerU CLI subprocess."""
    # MinerU 子进程不需要 DashScope 凭证；过滤掉避免把 secret 无谓透传给子进程。
    env = {k: v for k, v in source.items() if not k.startswith("DASHSCOPE_")}

    # MinerU 3.x starts a local FastAPI server and the CLI talks to 127.0.0.1
    # through httpx. Broad NO_PROXY values with CIDR or IPv6 entries can make
    # httpx.URLPattern reject the environment before the local server is used.
    sanitize_no_proxy_for_httpx(env)
    return env


def _run_mineru_cli(
    cmd: list[str],
    env: dict[str, str],
    timeout_s: float,
) -> subprocess.CompletedProcess:
    """
    Run MinerU as its own process group.

    MinerU CLI starts a local mineru-api child process. subprocess.run(timeout=...)
    kills only the direct child, which can leave the local server alive. Killing
    the process group keeps retries and batch ingest from accumulating orphan
    model servers.
    """
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as stdout_f:
        stderr_f = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
        try:
            return _run_mineru_cli_with_files(cmd, env, timeout_s, stdout_f, stderr_f)
        finally:
            stderr_f.close()


def _run_mineru_cli_with_files(
    cmd: list[str],
    env: dict[str, str],
    timeout_s: float,
    stdout_f,
    stderr_f,
) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout_f,
        stderr=stderr_f,
        text=True,
        start_new_session=True,
    )
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        _terminate_process_tree(proc.pid)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)
            proc.wait()
        stdout = _read_tempfile(stdout_f)
        stderr = _read_tempfile(stderr_f)
        e.stdout = stdout
        e.stderr = stderr
        raise e
    stdout = _read_tempfile(stdout_f)
    stderr = _read_tempfile(stderr_f)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _terminate_process_tree(pid: int) -> None:
    _signal_process_tree(pid, signal.SIGTERM)


def _kill_process_tree(pid: int) -> None:
    _signal_process_tree(pid, signal.SIGKILL)


def _signal_process_tree(pid: int, sig: signal.Signals) -> None:
    # MinerU's local fast_api child may start a new process group. Walk /proc
    # children first, then signal each process group and pid.
    pids = _descendant_pids(pid)
    for child in reversed(pids):
        _signal_process_group(child, sig)
    for child in reversed(pids):
        _signal_pid(child, sig)


def _descendant_pids(pid: int) -> list[int]:
    out: list[int] = []
    stack = [pid]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        out.append(current)
        children_path = Path(f"/proc/{current}/task/{current}/children")
        try:
            children = [
                int(part)
                for part in children_path.read_text(encoding="ascii").split()
                if part.strip()
            ]
        except OSError:
            children = []
        stack.extend(children)
    return out


def _signal_process_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def _signal_pid(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def _read_tempfile(handle) -> str:
    handle.flush()
    handle.seek(0)
    return handle.read()


def _mineru_failure_reason(proc: subprocess.CompletedProcess) -> str:
    # 把 stderr 尾部带进异常——失败日志/DB error_message 才能看到真实原因，
    # 不必回头翻控制台（曾因失败日志没记 stderr 而难定位）。
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    reason = tail[-1] if tail else "no stderr captured"
    return f"mineru CLI failed (rc={proc.returncode}): {reason}"


def _render_previews_safe(pdf_path: Path, preview_dir: Path, dpi: int) -> list:
    try:
        return render_pdf_to_images(pdf_path, preview_dir, dpi=dpi)
    except Exception as e:  # noqa: BLE001 - preview images are useful but not primary OCR output
        logger.warning("[pdf] preview render failed; continue without previews: %s", e)
        return []


def _mineru_timeout_reason(
    pdf_path: Path, timeout_s: float, text_stats: TextLayerStats
) -> str:
    reason = (
        f"mineru 超时（>{timeout_s:.0f}s）: {pdf_path.name}；"
        "确需处理可调大 CONTRACT_ARCHIVE_MINERU_TIMEOUT_S"
    )
    if text_stats.non_ws_chars:
        reason += (
            "。PDF 有文字层但质量不可用"
            f"（chars={text_stats.non_ws_chars}, printable={text_stats.printable_ratio:.2f}, "
            f"cjk={text_stats.cjk_ratio:.2f}, control={text_stats.control_ratio:.2f}）"
        )
    return reason


def _text_stats_note(prefix: str, stats: TextLayerStats) -> str:
    return (
        f"{prefix}; pages={stats.pages}, chars={stats.non_ws_chars}, "
        f"printable_ratio={stats.printable_ratio:.3f}, cjk_ratio={stats.cjk_ratio:.3f}, "
        f"control_ratio={stats.control_ratio:.3f}"
    )


def _output_from_text(
    *,
    pdf_path: Path,
    page_infos: list[PdfPageInfo],
    text: str,
    device: str,
    source: str,
    notes: str,
    preview_image_paths: list[str] | None = None,
) -> PipelineOutput:
    markdown = _unescape_markdown(text)
    sections = _split_sections(markdown)
    if not sections:
        title = _guess_title(markdown) or pdf_path.stem
        sections = [
            Section(
                level=1,
                title=title,
                text=markdown.strip(),
                page_start=0,
                page_end=max(0, len(page_infos) - 1),
            )
        ]
    structured = StructuredDocument(
        title=sections[0].title if sections else None,
        document_type=None,
        language="zh",
        pages=len(page_infos),
        sections=sections,
        tables=[],
    )
    now = datetime.now()
    return PipelineOutput(
        meta=PipelineMeta(
            pipeline_name="mineru",
            pipeline_version=_mineru_version(),
            model=source,
            device=device,
            source_pdf=str(pdf_path),
            started_at=now,
            finished_at=now,
            duration_seconds=0.0,
            notes=notes,
        ),
        raw_text=markdown,
        markdown=markdown,
        layout=[],
        structured=structured,
        preview_image_paths=preview_image_paths or [],
    )


def _guess_title(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip(" #\t")
        if 4 <= len(line) <= 80:
            return line
    return None


def _normalize_mineru(
    content_list: list[dict],
    page_dims: dict[int, tuple[float, float]],
) -> tuple[list[LayoutBlock], list[Table], str]:
    """
    把 MinerU 3.x content_list.json 归一化到统一 schema。

    关键点（与 2.x 不同）：
    - **没有 "title" 这个 type**，标题是 type:"text" + "text_level" >= 1
    - bbox 是**归一化到 0-1000 整数**，必须乘以页面真实宽高换算回 PDF point
    - `table_caption` / `image_caption` 是 **list[str]**，要 join
    """
    blocks: list[LayoutBlock] = []
    tables: list[Table] = []
    raw_lines: list[str] = []

    for i, item in enumerate(content_list):
        page = item.get("page_idx", 0)
        bbox_raw = item.get("bbox") or []
        bbox = None
        if len(bbox_raw) == 4:
            page_w, page_h = page_dims.get(page, (595.0, 841.0))  # A4 兜底
            # 0-1000 归一化坐标 → PDF point
            bbox = BBox(
                page=page,
                x0=float(bbox_raw[0]) * page_w / 1000.0,
                y0=float(bbox_raw[1]) * page_h / 1000.0,
                x1=float(bbox_raw[2]) * page_w / 1000.0,
                y1=float(bbox_raw[3]) * page_h / 1000.0,
            )

        item_type = item.get("type", "text")
        text_level = item.get("text_level") or 0
        text = item.get("text", "") or ""

        # 标题识别：type=text + text_level>=1
        if item_type == "text" and text_level >= 1:
            block_type = "title"
        else:
            block_type = _MINERU_TYPE_MAP.get(item_type, "other")

        # caption list[str] → str
        caption = item.get("table_caption") or item.get("image_caption") or []
        if isinstance(caption, list):
            caption = " ".join(str(c) for c in caption if c)

        if bbox:
            blocks.append(
                LayoutBlock(
                    bbox=bbox,
                    text=text or caption or "",
                    block_type=block_type,  # type: ignore[arg-type]
                    reading_order=i,
                )
            )
        if text:
            raw_lines.append(text)
        elif caption:
            raw_lines.append(caption)

        if item_type == "table":
            tables.append(
                Table(
                    page=page,
                    bbox=bbox,
                    html=item.get("table_body", ""),
                    caption=caption or None,
                )
            )

    return blocks, tables, "\n".join(raw_lines)


def _unescape_markdown(text: str) -> str:
    """
    剥离 MinerU 在 markdown 里加的反斜杠转义（\\_/\\*/\\[/\\]/\\(/\\)/\\#）。
    再清掉常见的"数字两侧夹下划线"残留（合同填空符号被 OCR 当成下划线）：
    '甲方于_2027年_6月_30日' → '甲方于 2027 年 6 月 30 日'
    """
    import re as _re

    text = _re.sub(r"\\([_*\[\]()#+\-.!`])", r"\1", text)
    # 数字/中文之间的下划线一律视为空白
    text = _re.sub(r"_+", " ", text)
    return text


def _split_sections(md: str) -> list[Section]:
    import re

    sections: list[Section] = []
    current_title: str | None = None
    current_level = 1
    buf: list[str] = []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            if current_title is not None:
                sections.append(
                    Section(
                        level=current_level,
                        title=current_title,
                        text="\n".join(buf).strip(),
                        page_start=0,
                        page_end=0,
                    )
                )
            current_title = m.group(2)
            current_level = len(m.group(1))
            buf = []
        else:
            buf.append(line)
    if current_title is not None:
        sections.append(
            Section(
                level=current_level,
                title=current_title,
                text="\n".join(buf).strip(),
                page_start=0,
                page_end=0,
            )
        )
    return sections
