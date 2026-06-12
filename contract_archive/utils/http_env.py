"""Small environment guards for httpx/OpenAI-compatible clients."""
from __future__ import annotations

import os
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager


_PROXY_ENV_KEYS = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
}


def sanitize_no_proxy_for_httpx(env: MutableMapping[str, str]) -> None:
    """
    Keep proxy settings, but simplify no_proxy to values httpx always accepts.

    The local shell may set NO_PROXY with CIDR blocks or IPv6 literals. Some
    httpx versions parse those entries as URL patterns and can raise InvalidURL
    before any request is sent.
    """
    if _PROXY_ENV_KEYS.intersection(env):
        env["NO_PROXY"] = "localhost,127.0.0.1"
        env["no_proxy"] = "localhost,127.0.0.1"


@contextmanager
def sanitized_httpx_proxy_env() -> Iterator[None]:
    """Temporarily sanitize process env for OpenAI/httpx client construction."""
    old_no_proxy = os.environ.get("NO_PROXY")
    old_no_proxy_lower = os.environ.get("no_proxy")
    changed = bool(_PROXY_ENV_KEYS.intersection(os.environ))
    if changed:
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        os.environ["no_proxy"] = "localhost,127.0.0.1"
    try:
        yield
    finally:
        if changed:
            if old_no_proxy is None:
                os.environ.pop("NO_PROXY", None)
            else:
                os.environ["NO_PROXY"] = old_no_proxy
            if old_no_proxy_lower is None:
                os.environ.pop("no_proxy", None)
            else:
                os.environ["no_proxy"] = old_no_proxy_lower
