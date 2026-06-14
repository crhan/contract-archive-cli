"""ingest 多源融合接线：_select_fusion_images 选页 + _maybe_run_vision_fusion 分派/兜底。

不跑真融合/网络：patch run_vision_fusion / escalate_low_confidence 验编排；选页用合成 PDF + 假 preview。
"""
from __future__ import annotations

from pathlib import Path

import fitz

from contract_archive.archive import ingest
from contract_archive.schemas import DocumentExtraction, FieldVerdict

_TEXT_BODY = "This insurance policy certificate page carries plenty of readable english content present."


def _build_doc(tmp_path, page_specs) -> Path:
    """造 doc_dir/{source.pdf, mineru/preview_images/page_NNN.png}。page_specs: 每页 'table'|'scan'|'text'。"""
    doc_dir = tmp_path / "doc"
    preview = doc_dir / "mineru" / "preview_images"
    preview.mkdir(parents=True)
    doc = fitz.open()
    for i, spec in enumerate(page_specs):
        page = doc.new_page()
        if spec == "text":
            page.insert_text((72, 100), _TEXT_BODY, fontsize=11)
        elif spec == "table":
            page.insert_text((72, 72), _TEXT_BODY, fontsize=10)
            x0, y0, cw, ch = 72, 120, 80, 30
            for r in range(4):
                page.draw_line((x0, y0 + r * ch), (x0 + 3 * cw, y0 + r * ch))
            for c in range(4):
                page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 3 * ch))
            for r in range(3):
                for c in range(3):
                    page.insert_text((x0 + c * cw + 5, y0 + r * ch + 18), f"c{r}{c}", fontsize=8)
        # 'scan' = 空白页（无文本层）
        (preview / f"page_{i + 1:03d}.png").write_bytes(b"\x89PNG\r\nFAKE")
    doc.save(str(doc_dir / "source.pdf"))
    doc.close()
    return doc_dir / "mineru"


# ---------- _select_fusion_images ----------


def test_select_prefers_table_and_ocr_skips_text(tmp_path):
    mineru_dir = _build_doc(tmp_path, ["table", "scan", "text"])
    out = ingest._select_fusion_images(mineru_dir)
    # 表格页(0→1-based 1) + 扫描页(1→2) 入选；文本页(2→3) 不选
    assert set(out) == {1, 2}
    assert all(p.exists() for p in out.values())


def test_select_empty_without_source_or_preview(tmp_path):
    bare = tmp_path / "m"
    (bare / "preview_images").mkdir(parents=True)
    assert ingest._select_fusion_images(bare) == {}  # 无 source.pdf


def test_select_respects_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTRACT_ARCHIVE_VISION_FUSION_MAX_PAGES", "1")
    mineru_dir = _build_doc(tmp_path, ["scan", "scan", "scan"])
    out = ingest._select_fusion_images(mineru_dir)
    assert len(out) == 1  # 上限封顶


# ---------- _maybe_run_vision_fusion ----------


def test_maybe_fusion_skips_non_fusion_type(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(ingest, "run_vision_fusion", lambda *a, **k: calls.__setitem__("n", 1) or True)
    env = DocumentExtraction(doc_type="合同协议")  # 无 enable_vision_fusion
    ingest._maybe_run_vision_fusion(env, "text", Path("/nope/mineru"), lambda m: None)
    assert calls["n"] == 0


def test_maybe_fusion_runs_and_escalates_low(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "_select_fusion_images", lambda d: {1: Path("p1")})

    def fake_fusion(env, text, images, *, fields, threshold):
        env.field_verdicts = [FieldVerdict(key="k", value="v", confidence=0.3, low_confidence=True)]
        env.fusion_overall_confidence = 0.3
        return True

    monkeypatch.setattr(ingest, "run_vision_fusion", fake_fusion)
    escalated = {}
    monkeypatch.setattr(
        ingest, "escalate_low_confidence",
        lambda env, source_pdf=None: escalated.__setitem__("pdf", source_pdf),
    )
    env = DocumentExtraction(doc_type="保险凭证")
    ingest._maybe_run_vision_fusion(env, "text", tmp_path / "mineru", lambda m: None)
    assert env.fusion_overall_confidence == 0.3
    assert "pdf" in escalated  # 低置信 → 触发 agent 兜底


def test_maybe_fusion_no_escalate_when_high(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "_select_fusion_images", lambda d: {1: Path("p1")})

    def fake_fusion(env, text, images, *, fields, threshold):
        env.fusion_overall_confidence = 0.92
        env.field_verdicts = [FieldVerdict(key="k", value="v", confidence=0.92)]
        return True

    monkeypatch.setattr(ingest, "run_vision_fusion", fake_fusion)
    escalated = {}
    monkeypatch.setattr(
        ingest, "escalate_low_confidence",
        lambda env, source_pdf=None: escalated.__setitem__("yes", True),
    )
    env = DocumentExtraction(doc_type="保险凭证")
    ingest._maybe_run_vision_fusion(env, "text", tmp_path / "mineru", lambda m: None)
    assert escalated == {}  # 高置信不兜底


def test_maybe_fusion_swallows_exceptions(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("fusion boom")

    monkeypatch.setattr(ingest, "_select_fusion_images", lambda d: {1: Path("p1")})
    monkeypatch.setattr(ingest, "run_vision_fusion", boom)
    env = DocumentExtraction(doc_type="保险凭证")
    logs = []
    # 不抛出，记一条跳过日志
    ingest._maybe_run_vision_fusion(env, "text", Path("/m"), logs.append)
    assert any("跳过" in m for m in logs)
