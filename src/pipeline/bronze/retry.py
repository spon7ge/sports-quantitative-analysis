"""Retry transient NBA API failures with exponential backoff."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

TRANSIENT_ERROR_MARKERS = (
    "rate limit",
    "too many requests",
    "429",
    "timeout",
    "timed out",
    "connection",
    "read timed out",
)

def is_transient_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in TRANSIENT_ERROR_MARKERS)

def call_with_retry(
    function: Callable[P, T],
    *args: P.args,
    label: str = "",
    max_retries: int = 5,
    base_delay: float = 8.0,
    max_delay: float = 60.0,
    **kwargs: P.kwargs,
) -> T:
    for attempt in range(max_retries + 1):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            if not is_transient_error(error) or attempt == max_retries:
                raise

            delay = min(max_delay, base_delay * (2**attempt))
            delay += random.uniform(0, delay * 0.5)

            logger.warning(
                "Transient error on %s (attempt %d/%d): "
                "%s; retrying in %.1fs",
                label or function.__name__,
                attempt + 1,
                max_retries,
                error,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("Retry loop ended unexpectedly")