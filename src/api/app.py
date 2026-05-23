"""
FastAPI orchestrator —— 把三路 pipeline + extraction + compare 暴露成 HTTP。

非生产级。目的：让 playground 既能 CLI 用，也能 HTTP 调试。
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..compare import generate_report
from ..extraction import extract_contract
from ..pipelines import get_pipeline
from ..schemas import FILE_MARKDOWN

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Document Intelligence Playground",
    description="PDF → OCR → Structured Output 对比平台",
    version="0.1.0",
)

OUTPUT_ROOT = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
INPUT_ROOT = Path(os.getenv("INPUT_DIR", "./input")).resolve()
PIPELINES = ("dashscope", "paddleocr", "mineru")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "output_root": str(OUTPUT_ROOT)}


@app.post("/ocr/{pipeline_name}")
async def run_ocr(pipeline_name: str, file: UploadFile = File(...)) -> JSONResponse:
    if pipeline_name not in PIPELINES:
        raise HTTPException(404, f"unknown pipeline: {pipeline_name}")
    pdf_path = _save_upload(file)
    out_dir = OUTPUT_ROOT / pipeline_name
    try:
        result = get_pipeline(pipeline_name).run(pdf_path, out_dir)
    except Exception as e:
        logger.exception("ocr failed")
        raise HTTPException(500, str(e))
    finally:
        pdf_path.unlink(missing_ok=True)
    return JSONResponse(
        {
            "pipeline": pipeline_name,
            "duration_s": result.meta.duration_seconds,
            "pages": result.structured.pages,
            "layout_blocks": len(result.layout),
            "tables": len(result.structured.tables),
            "output_dir": str(out_dir),
        }
    )


@app.post("/extract")
async def run_extract(pipeline_name: str = "dashscope", llm: bool = True) -> JSONResponse:
    """基于已有 OCR 输出（OUTPUT_ROOT/<pipeline>/markdown.md）做合同字段抽取。"""
    md_path = OUTPUT_ROOT / pipeline_name / FILE_MARKDOWN
    if not md_path.exists():
        raise HTTPException(404, f"{md_path} not found; run /ocr/{pipeline_name} first")
    document_text = md_path.read_text(encoding="utf-8")
    extraction, conf = extract_contract(document_text, llm_enabled=llm)
    out_dir = md_path.parent
    (out_dir / "extraction_result.json").write_text(
        extraction.model_dump_json(indent=2), encoding="utf-8"
    )
    (out_dir / "extraction_confidence.json").write_text(
        conf.model_dump_json(indent=2), encoding="utf-8"
    )
    return JSONResponse(
        {
            "pipeline": pipeline_name,
            "extraction": extraction.model_dump(),
            "confidence_overall": conf.overall,
        }
    )


@app.post("/pipeline/all")
async def run_all(file: UploadFile = File(...), llm: bool = True) -> JSONResponse:
    """一键三路 + 抽取 + 对比。"""
    pdf_path = _save_upload(file)
    summary = {}
    try:
        for name in PIPELINES:
            try:
                res = get_pipeline(name).run(pdf_path, OUTPUT_ROOT / name)
                md = (OUTPUT_ROOT / name / FILE_MARKDOWN).read_text(encoding="utf-8")
                extraction, conf = extract_contract(md, llm_enabled=llm)
                (OUTPUT_ROOT / name / "extraction_result.json").write_text(
                    extraction.model_dump_json(indent=2), encoding="utf-8"
                )
                (OUTPUT_ROOT / name / "extraction_confidence.json").write_text(
                    conf.model_dump_json(indent=2), encoding="utf-8"
                )
                summary[name] = {
                    "duration_s": res.meta.duration_seconds,
                    "extraction_overall": conf.overall,
                    "status": "ok",
                }
            except Exception as e:
                summary[name] = {"status": f"failed: {e}"}
    finally:
        pdf_path.unlink(missing_ok=True)

    report = generate_report(OUTPUT_ROOT)
    (OUTPUT_ROOT / "compare_report.md").write_text(report, encoding="utf-8")
    return JSONResponse({"summary": summary, "report_path": str(OUTPUT_ROOT / "compare_report.md")})


@app.get("/compare")
def get_compare() -> dict:
    return {"report": generate_report(OUTPUT_ROOT)}


def _save_upload(file: UploadFile) -> Path:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir=INPUT_ROOT)
    os.close(fd)
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return Path(tmp)
