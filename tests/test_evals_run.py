"""evals.run：EVALSET_DIR 解析 + load_cases 识别 text/pdf case。不跑真抽取/网络。"""
from __future__ import annotations

import json

from evals.run import DEFAULT_CASES, evalset_dir, load_cases


def test_evalset_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_ARCHIVE_EVALSET_DIR", str(tmp_path / "private" / "dataset"))
    assert evalset_dir() == tmp_path / "private" / "dataset"


def test_evalset_dir_falls_back_to_builtin(monkeypatch):
    monkeypatch.delenv("CONTRACT_ARCHIVE_EVALSET_DIR", raising=False)
    assert evalset_dir() == DEFAULT_CASES


def _case(suite_dir, name, *, text=None, pdf=False, gold=True):
    d = suite_dir / name
    d.mkdir(parents=True)
    if gold:
        (d / "gold.json").write_text(json.dumps({"doc_type": "其他"}), encoding="utf-8")
    if text is not None:
        (d / "input.txt").write_text(text, encoding="utf-8")
    if pdf:
        (d / "source.pdf").write_bytes(b"%PDF fake")
    return d


def test_load_cases_recognizes_text_and_pdf(tmp_path):
    suite = tmp_path / "extraction"
    _case(suite, "c_text", text="纯文本 case")
    _case(suite, "c_pdf", pdf=True)
    _case(suite, "c_both", text="文本", pdf=True)
    _case(suite, "c_nogold", text="无金标准", gold=False)  # 应跳过
    _case(suite, "c_neither", gold=True)  # 既无 input 也无 pdf → 跳过

    cases = {c["case_id"]: c for c in load_cases(suite)}
    assert set(cases) == {"c_text", "c_pdf", "c_both"}
    assert cases["c_text"]["pdf"] is None and cases["c_text"]["input"] == "纯文本 case"
    assert cases["c_pdf"]["pdf"] is not None and cases["c_pdf"]["input"] is None
    assert cases["c_both"]["pdf"] is not None and cases["c_both"]["input"] == "文本"
