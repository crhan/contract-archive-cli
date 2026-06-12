import os
from pathlib import Path
import subprocess

import fitz

from contract_archive.pipelines import mineru_pipeline as mp
from contract_archive.pipelines.mineru_pipeline import MinerUPipeline
from contract_archive.utils.http_env import sanitized_httpx_proxy_env


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((32, 64), "dummy insurance certificate")
    doc.save(path)
    doc.close()


def test_mineru_timeout_can_fall_back_to_vl_ocr(tmp_path, monkeypatch):
    pdf = tmp_path / "insurance.pdf"
    _make_pdf(pdf)

    monkeypatch.setattr(mp, "_resolve_mineru", lambda: "mineru")

    def timeout_run(cmd, env, timeout_s):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s)

    monkeypatch.setattr(mp, "_run_mineru_cli", timeout_run)
    monkeypatch.setattr(
        mp,
        "ocr_pdf_images_with_vl",
        lambda image_paths: "## 第 1 页\n保险凭证\n被保险人：张三",
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
        "ocr_pdf_images_with_vl",
        lambda image_paths: (_ for _ in ()).throw(AssertionError("VL fallback should not run")),
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
        "ocr_pdf_images_with_vl",
        lambda image_paths: "## 第 1 页\n保险凭证\n被保险人：张三",
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
