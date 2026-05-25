"""
introspection 命令单测：capabilities / describe / schema 的输出合法性与关键安全元数据。

这些是 MCP/tool wrapper 自动生成 tool 定义的依据，字段错了下游静默崩——必须守住。
"""
import json

from typer.testing import CliRunner

from contract_archive.cli import app

runner = CliRunner()


def _json_out(args: list[str]) -> dict:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_capabilities_lists_commands_with_safety_meta():
    d = _json_out(["capabilities"])
    assert d["schema_version"] == "1"
    assert d["tool"] == "contract-archive"
    names = {c["name"] for c in d["commands"]}
    assert {"ingest", "list", "delete", "schema", "capabilities"} <= names
    delete = next(c for c in d["commands"] if c["name"] == "delete")
    assert delete["destructive"] is True
    assert delete["requires_confirmation"] is True
    ingest = next(c for c in d["commands"] if c["name"] == "ingest")
    assert "cost" in ingest["side_effects"]      # Agent 调用前须知此命令花钱
    assert ingest["idempotent"] is True


def test_describe_ingest_exposes_params_and_choices():
    d = _json_out(["describe", "ingest"])
    names = {p["name"] for p in d["params"]}
    assert "path" in names and "no_llm" in names
    fmt = next(p for p in d["params"] if p["name"] == "fmt")
    assert set(fmt["choices"]) == {"table", "json"}


def test_describe_unknown_command_exit_2():
    assert runner.invoke(app, ["describe", "nope"]).exit_code == 2


def test_schema_document_is_valid_jsonschema_with_new_field():
    d = _json_out(["schema", "document"])
    assert d["type"] == "object"
    assert "extraction_error" in d["properties"]


def test_schema_error_has_retryable_signal():
    d = _json_out(["schema", "error"])
    assert "retryable" in d["properties"]
    assert "code" in d["properties"]


def test_schema_unknown_type_exit_2():
    assert runner.invoke(app, ["schema", "nope"]).exit_code == 2
