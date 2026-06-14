import os
from pathlib import Path
import subprocess

import fitz

from contract_archive.pipelines import mineru_pipeline as mp
from contract_archive.pipelines.mineru_pipeline import MinerUPipeline
from contract_archive.pipelines.vl_ocr import _PageResult
from contract_archive.utils.http_env import sanitized_httpx_proxy_env


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((32, 64), "dummy insurance certificate")
    doc.save(path)
    doc.close()


def _ok_page(body: str) -> _PageResult:
    """构造一页成功 OCR 结果，给混合提取的 ocr_pages monkeypatch 用。"""
    return _PageResult(body, ok=True, failed=False, truncated=False)


def test_mineru_timeout_can_fall_back_to_vl_ocr(tmp_path, monkeypatch):
    pdf = tmp_path / "insurance.pdf"
    _make_pdf(pdf)

    monkeypatch.setattr(mp, "_resolve_mineru", lambda: "mineru")

    def timeout_run(cmd, env, timeout_s):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s)

    monkeypatch.setattr(mp, "_run_mineru_cli", timeout_run)
    monkeypatch.setattr(
        mp,
        "ocr_pages",
        lambda image_paths, **kw: [_ok_page("保险凭证\n被保险人：张三")],
    )

    out = MinerUPipeline(
        prefer_text_layer=False,
        allow_vl_fallback=True,
        prefer_vl_ocr=False,
        lite_retry=False,
        vl_ocr_max_pages=2,
        vl_ocr_dpi=72,
    ).run(pdf, tmp_path / "out")

    assert out.meta.model == "vl-ocr-fallback"
    assert "保险凭证" in out.raw_text
    assert out.preview_image_paths
    # 混合提取记录页级分流：该页文本 <50 字 → 判为 ocr 页
    assert out.meta.page_routing == {
        "total": 1,
        "text_pages": 0,
        "ocr_pages": 1,
        "table_pages": 0,
    }


def test_mineru_timeout_retries_lite_profile_before_vl(tmp_path, monkeypatch):
    pdf = tmp_path / "insurance.pdf"
    _make_pdf(pdf)

    monkeypatch.setattr(mp, "_resolve_mineru", lambda: "mineru")
    monkeypatch.setattr(mp, "_mineru_version", lambda: "mineru-test")

    calls: list[list[str]] = []

    def fake_run(cmd, env, timeout_s):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s)
        out_root = Path(cmd[cmd.index("-o") + 1])
        stem = pdf.stem
        result_dir = out_root / stem / "ocr"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"{stem}.md").write_text("# 保险凭证\n被保险人：张三", encoding="utf-8")
        (result_dir / f"{stem}_content_list.json").write_text(
            '[{"type":"text","text":"保险凭证\\n被保险人：张三","page_idx":0}]',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mp, "_run_mineru_cli", fake_run)
    monkeypatch.setattr(
        mp,
        "ocr_pages",
        lambda image_paths, **kw: (_ for _ in ()).throw(
            AssertionError("VL fallback should not run")
        ),
    )

    out = MinerUPipeline(
        prefer_text_layer=False,
        allow_vl_fallback=True,
        prefer_vl_ocr=False,
        lite_retry=True,
    ).run(pdf, tmp_path / "out")

    assert "保险凭证" in out.raw_text
    assert calls[1][calls[1].index("-m") + 1] == "ocr"
    assert calls[1][calls[1].index("-l") + 1] == "ch_lite"
    assert calls[1][calls[1].index("-f") + 1] == "false"
    assert calls[1][calls[1].index("-t") + 1] == "false"


def test_vl_ocr_first_skips_mineru_when_enabled(tmp_path, monkeypatch):
    pdf = tmp_path / "insurance.pdf"
    _make_pdf(pdf)

    monkeypatch.setattr(mp, "_mineru_version", lambda: "mineru-test")
    monkeypatch.setattr(
        mp,
        "_run_mineru_cli",
        lambda cmd, env, timeout_s: (_ for _ in ()).throw(
            AssertionError("MinerU should not run when VL OCR is first")
        ),
    )
    monkeypatch.setattr(
        mp,
        "ocr_pages",
        lambda image_paths, **kw: [_ok_page("保险凭证\n被保险人：张三")],
    )

    out = MinerUPipeline(
        prefer_text_layer=False,
        allow_vl_fallback=True,
        prefer_vl_ocr=True,
        vl_ocr_max_pages=2,
        vl_ocr_dpi=72,
    ).run(pdf, tmp_path / "out")

    assert out.meta.model == "vl-ocr-first"
    assert "保险凭证" in out.raw_text
    assert out.preview_image_paths


def test_hybrid_extraction_mixes_native_text_and_vl(tmp_path, monkeypatch):
    """混合提取：文本页走原生、扫描页走 VL，按页序拼回；**只把扫描页交给 VL**（省 token）。"""
    pdf = tmp_path / "mixed.pdf"
    doc = fitz.open()
    p1 = doc.new_page(width=400, height=300)
    p1.insert_text(
        (40, 60),
        "This insurance policy certificate page carries plenty of readable english content here.",
        fontsize=10,
    )
    doc.new_page(width=400, height=300)  # 第 2 页空白 → 扫描页
    doc.save(pdf)
    doc.close()

    monkeypatch.setattr(mp, "_mineru_version", lambda: "mineru-test")

    seen: dict = {}

    def fake_ocr_pages(image_paths, **kw):
        seen["n_images"] = len(image_paths)
        seen["labels"] = kw.get("page_labels")
        return [_ok_page("扫描页 VL 抽取内容") for _ in image_paths]

    monkeypatch.setattr(mp, "ocr_pages", fake_ocr_pages)

    out = MinerUPipeline(
        prefer_text_layer=True,
        allow_vl_fallback=True,
        prefer_vl_ocr=True,
        lite_retry=False,
        vl_ocr_max_pages=10,
        vl_ocr_dpi=72,
    ).run(pdf, tmp_path / "out")

    assert out.meta.model == "vl-ocr-first"
    assert out.meta.page_routing == {
        "total": 2,
        "text_pages": 1,
        "ocr_pages": 1,
        "table_pages": 0,
    }
    # 只把扫描页（第 2 页）交给 VL，文本页不浪费一次看图调用
    assert seen["n_images"] == 1
    assert seen["labels"] == [2]
    # 文本页原生内容 + 扫描页 VL 内容，按真实页序拼接
    assert "readable english content" in out.raw_text
    assert "扫描页 VL 抽取内容" in out.raw_text
    assert out.raw_text.index("## 第 1 页") < out.raw_text.index("## 第 2 页")


def test_mineru_subprocess_env_filters_secrets_and_sanitizes_no_proxy():
    env = mp._mineru_subprocess_env(
        {
            "DASHSCOPE_API_KEY": "secret",
            "HTTP_PROXY": "http://127.0.0.1:7892",
            "HTTPS_PROXY": "http://127.0.0.1:7892",
            "NO_PROXY": "localhost,127.0.0.1,::1,10.0.0.0/8",
            "PATH": "/bin",
        }
    )

    assert "DASHSCOPE_API_KEY" not in env
    assert env["HTTP_PROXY"] == "http://127.0.0.1:7892"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:7892"
    assert env["NO_PROXY"] == "localhost,127.0.0.1"
    assert env["no_proxy"] == "localhost,127.0.0.1"
    assert env["PATH"] == "/bin"


def test_sanitized_httpx_proxy_env_restores_no_proxy(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7892")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1,10.0.0.0/8")
    monkeypatch.delenv("no_proxy", raising=False)

    with sanitized_httpx_proxy_env():
        assert os.environ["NO_PROXY"] == "localhost,127.0.0.1"
        assert os.environ["no_proxy"] == "localhost,127.0.0.1"

    assert os.environ["NO_PROXY"] == "localhost,127.0.0.1,::1,10.0.0.0/8"
    assert "no_proxy" not in os.environ
