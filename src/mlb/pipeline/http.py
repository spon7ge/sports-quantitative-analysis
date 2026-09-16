"""Rate-limited HTTP client for MLB ingest."""

from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.mlb.config import MlbConfig

HttpFn = Callable[[str, dict | None], bytes]


class HttpClient:
    """Callable `(url, params) -> bytes` that sleeps and sends a User-Agent."""

    def __init__(self, config: MlbConfig) -> None:
        self.config = config

    def __call__(self, url: str, params: dict | None = None) -> bytes:
        time.sleep(self.config.rate_limit_seconds)
        full_url = url
        if params:
            full_url = f"{url}?{urlencode(params, doseq=True)}"
        request = Request(
            full_url,
            headers={"User-Agent": self.config.user_agent},
        )
        with urlopen(request) as response:
            return response.read()
