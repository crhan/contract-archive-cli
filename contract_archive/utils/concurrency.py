"""并发执行 helper：把同步的 LLM 调用并发化（多源融合/逐页 OCR/多字段评判共用）。

为什么线程池而非 asyncio：openai SDK 是同步阻塞 IO，GIL 在网络等待时释放，线程池足够；
引 asyncio 要把整条调用链（extract/ingest/sqlite）改 async + 重写所有 fake-openai 测试，
破坏面大。这里用 ThreadPoolExecutor 包同步调用——**调用方须在并发块外层一次性构造好
OpenAI client 与 proxy-env 上下文**（sanitized_httpx_proxy_env 改的是进程级 os.environ，
多线程各自进退会竞态），worker 只复用 client，保持最小改动、避免竞态。
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_LLM_CONCURRENCY = 4


def llm_concurrency() -> int:
    """并发度旋钮 CONTRACT_ARCHIVE_LLM_CONCURRENCY（默认 4）。

    运行时旋钮，env-only（同 get_timeout_s / VL_OCR_RETRIES 风格，不进 CONFIG_KEYS）。
    坏值（非整数 / <1 / 缺失）回退默认并 warning——坏配置不该让命令崩。
    """
    raw = os.getenv("CONTRACT_ARCHIVE_LLM_CONCURRENCY")
    if not raw or not raw.strip():
        return DEFAULT_LLM_CONCURRENCY
    try:
        val = int(raw.strip())
    except ValueError:
        logger.warning(
            "CONTRACT_ARCHIVE_LLM_CONCURRENCY=%r 不是整数，回退默认 %d", raw, DEFAULT_LLM_CONCURRENCY
        )
        return DEFAULT_LLM_CONCURRENCY
    if val < 1:
        logger.warning(
            "CONTRACT_ARCHIVE_LLM_CONCURRENCY=%r 必须 >=1，回退默认 %d", raw, DEFAULT_LLM_CONCURRENCY
        )
        return DEFAULT_LLM_CONCURRENCY
    return val


def _run_one(fn: Callable[[T], R], item: T, on_error: Callable[[T, Exception], R] | None) -> R:
    """跑单个 fn(item)，异常隔离：on_error 给降级值，否则降级为 None。"""
    try:
        return fn(item)
    except Exception as e:  # noqa: BLE001 - 单项失败隔离，不拖垮整批
        if on_error is None:
            logger.warning("并发任务失败，降级为 None: %s", e)
            return None  # type: ignore[return-value]
        return on_error(item, e)


def map_concurrent(
    fn: Callable[[T], R],
    items: list[T],
    *,
    max_workers: int | None = None,
    on_error: Callable[[T, Exception], R] | None = None,
) -> list[R]:
    """并发跑 fn(item)，**保序**返回 list（结果顺序 == 输入顺序，不随完成先后变化）。

    - 单项失败隔离：fn 抛异常时该位置取 on_error(item, exc) 的降级值（默认 None），其余照常
      完成——与逐页 OCR"单页失败标记、不拖垮整份"语义一致。
    - 并发度 = max_workers or min(llm_concurrency(), len(items))。
    - workers<=1 或 items<=1 退化为同序串行（测试设 CONCURRENCY=1 可让 fake 的 behaviors
      序列确定消费；也省掉单项开线程的开销）。
    """
    items = list(items)
    if not items:
        return []
    workers = max_workers if max_workers is not None else min(llm_concurrency(), len(items))
    if workers <= 1 or len(items) == 1:
        return [_run_one(fn, it, on_error) for it in items]

    out: list[R] = [None] * len(items)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {ex.submit(_run_one, fn, it, on_error): i for i, it in enumerate(items)}
        for fut in as_completed(fut_to_idx):
            out[fut_to_idx[fut]] = fut.result()
    return out


_USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def merge_usage(usages: list[dict | None]) -> dict | None:
    """多路 LLM 调用的 token usage 求和（input/output/total_tokens）。全 None → None。

    融合一份文档会发多路请求（文本抽取 + 看图 + N 个评判）。合并后挂到 env.llm_usage，
    让成本核算/评测能看到融合的**总**开销，而不是只记某一路。
    """
    acc = {k: 0 for k in _USAGE_KEYS}
    seen = False
    for u in usages:
        if not u:
            continue
        seen = True
        for k in _USAGE_KEYS:
            v = u.get(k)
            if isinstance(v, (int, float)):
                acc[k] += int(v)
    return acc if seen else None
