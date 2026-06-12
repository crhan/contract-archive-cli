"""
Code review 采纳项的回归测试——锁住 P0/P1/P2 修复，防回退。

对应 findings：
- P1 COMMAND_META 漂移守卫（未登记命令会 fail-open 误导 agent）
- P0 空抽取不设 llm_model（evals parse_ok 一票否决依赖它）
- P2 retry_after 解析 / 408+裸 httpx 异常分类 / build_show_table 拆分后渲染 / ndjson 失败事件带 error
"""
import json
from types import SimpleNamespace

import typer
from rich.table import Table
from typer.testing import CliRunner

import contract_archive.cli as climod
from contract_archive import errors as E
from contract_archive.archive.ingest import IngestResult
from contract_archive.cli import app
from contract_archive.cli_introspect import COMMAND_META
from contract_archive.cli_render import build_show_table
from contract_archive.extraction import document_extractor as de
from contract_archive.extraction.llm_extractor import LlmResult

runner = CliRunner()


# ---------- P1：COMMAND_META 漂移守卫 ----------

def test_command_meta_covers_all_registered_commands():
    """每个注册命令都必须有安全元数据；漏登记会让 capabilities fail-open 报『只读非破坏』误导 agent。"""
    group = typer.main.get_command(app)
    missing = set(group.commands) - set(COMMAND_META)
    assert not missing, f"命令缺 COMMAND_META 安全元数据: {missing}"


# ---------- P0：空抽取保持 llm_model=None（否则劣质模型蒙混过 evals parse_ok）----------

def test_empty_extraction_keeps_llm_model_none(monkeypatch):
    monkeypatch.setattr(
        de, "call_llm_document",
        lambda *a, **k: LlmResult(parsed={}, model="qwen3.7-max", error=E.config_missing("no key")),
    )
    env = de.extract_document("正文", llm_enabled=True)
    assert env.llm_model is None                       # evals score.py parse_ok 依赖
    assert env.extraction_error is not None
    assert env.extraction_error.code == "CONFIG_MISSING"


# ---------- P2：retry_after / 408 / 裸 httpx 异常分类 ----------

def test_retry_after_parsed_from_response_header():
    resp = SimpleNamespace(headers={"retry-after": "30"})
    exc = type("RateLimitError", (Exception,), {"status_code": 429, "response": resp})("rate")
    info = E.classify_exception(exc)
    assert info.code == "RATE_LIMITED" and info.retry_after_s == 30.0


def test_408_is_timeout_transient():
    exc = type("E408", (Exception,), {"status_code": 408})("request timeout")
    assert E.classify_exception(exc).code == "TIMEOUT"
    assert E.classify_exception(exc).retryable is True


def test_bare_httpx_connect_error_is_retryable():
    exc = type("ConnectError", (Exception,), {})("connection refused")
    assert E.classify_exception(exc).retryable is True


# ---------- P2：build_show_table 拆分后渲染冒烟（锁 L3 行为）----------

def _fake_row(**kw):
    base = dict(
        id=1, sha256="abc123", status="ok", source_path="/x.pdf", output_dir="/o",
        ingested_at="2026-01-01T00:00:00Z", error_message=None,
        doc_type="合同协议", title="测试合同", contract_name="测试合同", summary="一句话摘要",
        party_a="甲方", party_b="乙方", sign_date="2026-01-01", expire_date=None,
        auto_renewal=1, amount_text="¥100", amount_value=100.0, overall_confidence=0.7,
        obligations=[], risk_clauses=[],
    )
    details = kw.pop("_details", {})
    base.update(kw)
    row = SimpleNamespace(**base)
    row.details = lambda: details
    return row


def test_build_show_table_contract_renders():
    row = _fake_row(_details={
        "amounts": [{"label": "价款", "text": "¥100", "value": 100.0, "is_total_component": True}],
        "seals": [{"owner": "甲方", "seal_type": "公章", "raw_text": "甲方公章"}],
        "completeness": {"status": "incomplete",
                         "issues": [{"item": "主协议·甲方签章", "category": "signature",
                                     "detail": "落款空白", "evidence": "第3页"}]},
    })
    row.obligations = [SimpleNamespace(actor="party_a", action="付款", deadline="2026-02-01", evidence="")]
    row.risk_clauses = ["违约金 10%"]
    table = build_show_table(row)
    assert isinstance(table, Table)
    assert table.row_count > 5


def test_build_show_table_non_contract_renders():
    row = _fake_row(
        doc_type="证明", contract_name=None, party_a=None, party_b=None,
        _details={"parties": ["张三"], "key_dates": [{"label": "出具日", "date": "2026-01-01"}],
                  "fields": [{"label": "持证人", "value": "张三"}]},
    )
    table = build_show_table(row)
    assert isinstance(table, Table)
    assert table.row_count > 3


# ---------- P2：ndjson 失败事件携带结构化 error ----------

def test_ndjson_file_event_carries_structured_error(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"%PDF-1.4 x")
    arch = tmp_path / "arch"
    monkeypatch.setattr(climod, "MinerUPipeline", lambda **kw: object())

    def fake_ingest(pdf, paths, conn, **kw):
        return IngestResult(pdf_path=pdf, sha256="s", status="partial", doc_id=1,
                            error=E.config_missing("no key"), error_message="empty")

    monkeypatch.setattr(climod, "ingest_pdf", fake_ingest)
    result = runner.invoke(
        app, ["ingest", str(src), "--progress", "ndjson", "--archive", str(arch), "--no-llm"]
    )
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    file_events = [e for e in events if e.get("event") == "file_done"]
    assert len(file_events) == 1
    assert file_events[0]["error"]["code"] == "CONFIG_MISSING"
    assert file_events[0]["error"]["retryable"] is False
