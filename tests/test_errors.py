"""
errors 模块单测：异常分类（duck-typing）+ 构造器 + IngestResult 的 JSON 输出带 error。

重点验证 retryable 信号正确——这是 Agent 重试决策的依据，错了比没有更危险。
"""
from contract_archive import errors as E
from contract_archive.archive.ingest import IngestResult
from contract_archive.cli_render import ingest_result_to_dict


class _Status(Exception):
    """伪装 openai 异常：带 status_code 属性，供 duck-typing 分类。"""

    def __init__(self, msg: str, status: int):
        super().__init__(msg)
        self.status_code = status


def test_rate_limit_retryable():
    e = E.classify_exception(_Status("rate limit", 429))
    assert e.code == "RATE_LIMITED"
    assert e.category == "transient"
    assert e.retryable is True


def test_auth_is_config_not_retryable():
    e = E.classify_exception(_Status("invalid key", 401))
    assert e.code == "AUTH_FAILED"
    assert e.category == "config"
    assert e.retryable is False


def test_permission_denied():
    e = E.classify_exception(_Status("forbidden", 403))
    assert e.code == "PERMISSION_DENIED"
    assert e.retryable is False


def test_bad_request_not_retryable():
    e = E.classify_exception(_Status("bad model id", 400))
    assert e.code == "BAD_REQUEST"
    assert e.retryable is False


def test_5xx_retryable():
    e = E.classify_exception(_Status("upstream down", 503))
    assert e.code == "UPSTREAM_ERROR"
    assert e.retryable is True


def test_timeout_by_classname():
    class APITimeoutError(Exception):
        pass

    e = E.classify_exception(APITimeoutError("request timed out"))
    assert e.code == "TIMEOUT"
    assert e.retryable is True


def test_ratelimit_by_classname_without_status():
    class RateLimitError(Exception):
        pass

    e = E.classify_exception(RateLimitError("slow down"))
    assert e.code == "RATE_LIMITED"
    assert e.retryable is True


def test_unknown_is_conservative_not_retryable():
    e = E.classify_exception(ValueError("something weird"))
    assert e.code == "UNKNOWN"
    assert e.retryable is False


def test_message_truncated_to_cap():
    e = E.classify_exception(ValueError("x" * 2000))
    assert len(e.message) <= 500


def test_constructors():
    assert E.config_missing("no key").code == "CONFIG_MISSING"
    assert E.config_missing("no key").retryable is False
    assert E.extract_empty("empty").code == "EXTRACT_EMPTY"
    assert E.mineru_failed("crash").code == "MINERU_FAILED"
    assert E.mineru_failed("crash").category == "infra"


def test_ingest_result_to_dict_carries_structured_error():
    err = E.config_missing("DASHSCOPE_API_KEY 缺失")
    r = IngestResult(
        pdf_path="/x.pdf", sha256="abc123", status="partial", doc_id=1, error=err
    )
    d = ingest_result_to_dict(r)
    assert d["error"]["code"] == "CONFIG_MISSING"
    assert d["error"]["category"] == "config"
    assert d["error"]["retryable"] is False
    # 旧字段仍在，向后兼容
    assert "error_message" in d


def test_ingest_result_to_dict_error_none_when_ok():
    r = IngestResult(pdf_path="/x.pdf", sha256="abc", status="ok", doc_id=1)
    assert ingest_result_to_dict(r)["error"] is None
