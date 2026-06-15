import os
from pathlib import Path

import fitz
import pytest

from contract_archive.pipelines import ocr_pipeline as op
from contract_archive.pipelines.ocr_pipeline import OCRPipeline
from contract_archive.pipelines.vl_ocr import _PageResult
from contract_archive.utils.http_env import sanitized_httpx_proxy_env


def _make_pdf(path: Path, text: str = "") -> None:
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    if text:
        page.insert_text((40, 60), text, fontsize=10)
    doc.save(path)
    doc.close()


def _ok_page(body: str) -> _PageResult:
    return _PageResult(body, ok=True, failed=False, truncated=False)


def test_native_text_layer_writes_ocr_meta(tmp_path, monkeypatch):
    text = (
        "This policy certificate has enough readable native text for routing. "
        "It should not call remote OCR for this page."
    )
    pdf = tmp_path / "native.pdf"
    _make_pdf(pdf, text)

    monkeypatch.setattr(
        op,
        "ocr_pages",
        lambda image_paths, **kw: (_ for _ in ()).throw(AssertionError("unexpected OCR")),
    )

    out = OCRPipeline(vl_ocr_dpi=72).run(pdf, tmp_path / "out")

    assert out.meta.pipeline_name == "ocr"
    assert out.meta.model == "native-text-layer"
    assert "readable native text" in out.raw_text
    assert out.meta.page_routing == {
        "total": 1,
        "text_pages": 1,
        "ocr_pages": 0,
        "table_pages": 0,
    }


def test_hybrid_extraction_mixes_native_text_and_vl(tmp_path, monkeypatch):
    pdf = tmp_path / "mixed.pdf"
    doc = fitz.open()
    p1 = doc.new_page(width=400, height=300)
    p1.insert_text(
        (40, 60),
        "This insurance policy certificate page carries plenty of readable english content here.",
        fontsize=10,
    )
    doc.new_page(width=400, height=300)
    doc.save(pdf)
    doc.close()

    seen: dict = {}

    def fake_ocr_pages(image_paths, **kw):
        seen["n_images"] = len(image_paths)
        seen["labels"] = kw.get("page_labels")
        return [_ok_page("扫描页 VL 抽取内容") for _ in image_paths]

    monkeypatch.setattr(op, "ocr_pages", fake_ocr_pages)

    out = OCRPipeline(
        prefer_text_layer=True,
        allow_vl_fallback=True,
        vl_ocr_max_pages=10,
        vl_ocr_dpi=72,
    ).run(pdf, tmp_path / "out")

    assert out.meta.model == "hybrid-text-vl"
    assert out.meta.page_routing == {
        "total": 2,
        "text_pages": 1,
        "ocr_pages": 1,
        "table_pages": 0,
    }
    assert seen["n_images"] == 1
    assert seen["labels"] == [2]
    assert "readable english content" in out.raw_text
    assert "扫描页 VL 抽取内容" in out.raw_text
    assert out.raw_text.index("## 第 1 页") < out.raw_text.index("## 第 2 页")


def test_image_ocr_required_but_remote_disabled_fails(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _make_pdf(pdf)

    with pytest.raises(RuntimeError, match="remote OCR is disabled"):
        OCRPipeline(
            prefer_text_layer=False,
            allow_vl_fallback=False,
            vl_ocr_dpi=72,
        ).run(pdf, tmp_path / "out")


def test_sanitized_httpx_proxy_env_restores_no_proxy(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7892")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1,10.0.0.0/8")
    monkeypatch.delenv("no_proxy", raising=False)

    with sanitized_httpx_proxy_env():
        assert os.environ["NO_PROXY"] == "localhost,127.0.0.1"
        assert os.environ["no_proxy"] == "localhost,127.0.0.1"

    assert os.environ["NO_PROXY"] == "localhost,127.0.0.1,::1,10.0.0.0/8"
    assert "no_proxy" not in os.environ
