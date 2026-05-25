"""
结构化错误模型——给机器（Agent / 自动化编排）一个可据以决策的错误信号。

为什么存在：历史上错误是自由文本（`f"mineru: {e}"`），退出码只有 0/1，
Agent 无法区分「限流（该退避重试）」与「缺 API key（重试无用，该停下改配置）」，
只能正则匹配错误串——供应商一改措辞就崩。这里把错误归一成
`code / category / retryable`，让上层（尤其 ingest 的 JSON 输出）携带可判定信号。

设计原则：
- 分类靠 duck-typing（异常类名 + status_code），**不 import openai**——
  errors 是底层模块，不该绑死某个 SDK 的异常类层级，也避免无谓的强依赖。
- ErrorInfo 是 pydantic 模型，可直接嵌进 DocumentExtraction 落盘、可 model_dump 进 JSON 输出。
- retryable 是给 Agent 的核心信号：transient 类（限流/超时/网络/5xx）为 True，
  config/permission/validation/user 类为 False。
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ErrorCategory(str, Enum):
    """错误大类。retryable 默认由类别决定（transient→可重试，其余不可）。"""

    user = "user"              # 用户输入错（不存在的 id / 非 PDF）
    validation = "validation"  # 入参/上游响应不合法（400、空抽取）
    config = "config"          # 配置缺失或无效（缺 API key、key 失效）
    permission = "permission"  # 鉴权通过但无权限（403）
    transient = "transient"    # 瞬时故障，重试可能成功（限流/超时/网络/5xx）
    infra = "infra"            # 基础设施/外部工具故障（MinerU 崩、DB 锁）
    unknown = "unknown"        # 未能归类


class ErrorInfo(BaseModel):
    """
    一条结构化错误。嵌进抽取信封落盘、并由 CLI 的 --format json 原样吐出。

    :param code: 稳定的机器可读错误码（如 RATE_LIMITED），供 Agent switch。
    :param category: 错误大类（见 ErrorCategory）。
    :param message: 人类可读详情（已截断，避免把超长 traceback 灌进 JSON）。
    :param retryable: 给 Agent 的核心信号——是否值得退避后重试。
    :param retry_after_s: 建议的重试等待秒数（限流场景），未知为 None。
    """

    code: str
    category: str
    message: str
    retryable: bool
    retry_after_s: Optional[float] = None


# message 截断长度：够定位问题，又不至于把整个 traceback/超长上游响应灌进 JSON。
_MAX_MESSAGE_LEN = 500


def _short(text: str) -> str:
    """错误文本归一：去首尾空白 + 截断，保证 JSON 体积可控。"""
    text = (text or "").strip()
    return text if len(text) <= _MAX_MESSAGE_LEN else text[: _MAX_MESSAGE_LEN - 1] + "…"


def classify_exception(exc: BaseException) -> ErrorInfo:
    """
    把一个（多半来自外部 API 调用的）异常归类成 ErrorInfo。

    用 duck-typing 而非 isinstance(openai.XxxError)：读异常类名 + status_code，
    既能识别 openai SDK 的异常（RateLimitError/AuthenticationError/...），
    也能兜住自建网关/其他 SDK 抛出的形似异常。无法识别时归 UNKNOWN（不可重试，保守）。
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None
    msg = _short(str(exc))

    # 限流：429 / RateLimitError。可重试，给默认退避建议。
    if name == "RateLimitError" or status == 429:
        return ErrorInfo(
            code="RATE_LIMITED", category=ErrorCategory.transient.value,
            message=msg, retryable=True, retry_after_s=_retry_after(exc),
        )
    # 认证失败 / key 无效：401。属配置问题，重试无用，应改配置。
    if name == "AuthenticationError" or status == 401:
        return ErrorInfo(
            code="AUTH_FAILED", category=ErrorCategory.config.value,
            message=msg, retryable=False,
        )
    # 鉴权通过但无权限：403。
    if name == "PermissionDeniedError" or status == 403:
        return ErrorInfo(
            code="PERMISSION_DENIED", category=ErrorCategory.permission.value,
            message=msg, retryable=False,
        )
    # 请求不合法：400（模型 id 错、参数非法）。重试无用。
    if name == "BadRequestError" or status == 400:
        return ErrorInfo(
            code="BAD_REQUEST", category=ErrorCategory.validation.value,
            message=msg, retryable=False,
        )
    # 超时：408 / openai 或 httpx 的超时异常 / 文本含 timeout。瞬时，可重试。
    if (
        name in ("APITimeoutError", "ReadTimeout", "ConnectTimeout")
        or status == 408
        or "timed out" in msg.lower()
        or "timeout" in msg.lower()
    ):
        return ErrorInfo(
            code="TIMEOUT", category=ErrorCategory.transient.value,
            message=msg, retryable=True,
        )
    # 连接错误 / 上游 5xx：瞬时，可重试（含 httpx 裸 ConnectError）。
    if name in ("APIConnectionError", "ConnectError", "InternalServerError") or (status is not None and status >= 500):
        return ErrorInfo(
            code="UPSTREAM_ERROR", category=ErrorCategory.transient.value,
            message=msg, retryable=True,
        )
    return ErrorInfo(
        code="UNKNOWN", category=ErrorCategory.unknown.value,
        message=msg, retryable=False,
    )


def _retry_after(exc: BaseException) -> Optional[float]:
    """尽力从异常/响应头取 Retry-After 秒数；取不到返回 None（由调用方自定退避）。"""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
    return None


# ---------- 非异常来源的常见错误构造器（语义清晰，省得各处手拼 ErrorInfo） ----------


def config_missing(detail: str) -> ErrorInfo:
    """缺必要配置（最常见：缺 DASHSCOPE_API_KEY）。重试无用，须先配置。"""
    return ErrorInfo(
        code="CONFIG_MISSING", category=ErrorCategory.config.value,
        message=_short(detail), retryable=False,
    )


def extract_empty(detail: str) -> ErrorInfo:
    """LLM 调用成功但抽取产出为空（非缺 key 场景）。保守标不可重试。"""
    return ErrorInfo(
        code="EXTRACT_EMPTY", category=ErrorCategory.validation.value,
        message=_short(detail), retryable=False,
    )


def mineru_failed(detail: str) -> ErrorInfo:
    """MinerU 解析失败（subprocess 非零/崩溃）。归基础设施类。"""
    return ErrorInfo(
        code="MINERU_FAILED", category=ErrorCategory.infra.value,
        message=_short(detail), retryable=False,
    )
