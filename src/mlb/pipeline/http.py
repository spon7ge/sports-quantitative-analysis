"""Rate-limited HTTP client for MLB ingest."""

from __future__ import annotations

import ssl
import sys
import time
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi

from src.mlb.config import MlbConfig

HttpFn = Callable[[str, dict | None], bytes]

_TIMEOUT_SECONDS = 30.0
_MAX_ATTEMPTS = 5
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def _ssl_context() -> ssl.SSLContext:
    """Use certifi CAs. python.org macOS builds ship with no default bundle."""
    return ssl.create_default_context(cafile=certifi.where())


def _is_transient(error: BaseException) -> bool:
    if isinstance(error, HTTPError):
        return error.code in _RETRY_STATUS
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, URLError):
        reason = error.reason
        if isinstance(reason, BaseException):
            return _is_transient(reason)
        message = str(reason).lower()
        return "timed out" in message or "timeout" in message
    if isinstance(error, OSError):
        message = str(error).lower()
        return "timed out" in message or "timeout" in message
    return False


class HttpClient:
    """Callable `(url, params) -> bytes` that sleeps and sends a User-Agent."""

    def __init__(self, config: MlbConfig) -> None:
        self.config = config

    def __call__(self, url: str, params: dict | None = None) -> bytes:
        full_url = url
        if params:
            full_url = f"{url}?{urlencode(params, doseq=True)}"
        request = Request(
            full_url,
            headers={"User-Agent": self.config.user_agent},
        )
        last_error: BaseException | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            time.sleep(self.config.rate_limit_seconds)
            try:
                with urlopen(
                    request,
                    context=_ssl_context(),
                    timeout=_TIMEOUT_SECONDS,
                ) as response:
                    return response.read()
            except (URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt >= _MAX_ATTEMPTS or not _is_transient(error):
                    raise
                delay = float(2 ** (attempt - 1))
                print(
                    f"mlb http: attempt {attempt}/{_MAX_ATTEMPTS} failed "
                    f"({error}); retrying in {delay:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error
