"""Stage timings and truthful JSON/DB completion status for odds scrapers."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class StageReport:
    book: str
    league: str = "nba"
    json_ok: bool = False
    db_ok: bool | None = None
    stages_s: dict[str, float] = field(default_factory=dict)
    rows_json: int = 0
    rows_upserted: int = 0
    retries: int = 0
    rate_limits: int = 0
    error: str | None = None

    def mark_json(self, *, ok: bool, rows: int = 0) -> None:
        self.json_ok = ok
        self.rows_json = rows

    def mark_db(self, *, ok: bool, rows: int = 0) -> None:
        self.db_ok = ok
        self.rows_upserted = rows

    def add_retry(self, n: int = 1) -> None:
        self.retries += n

    def add_rate_limit(self, n: int = 1) -> None:
        self.rate_limits += n

    def set_stage(self, name: str, seconds: float) -> None:
        self.stages_s[name] = round(float(seconds), 3)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def log_and_exit_code(self) -> int:
        """Log a machine-readable line; return 0 when JSON ok.

        ``db_ok`` False fails the run (explicit load error). ``db_ok`` None means
        DB was skipped (JSON-only mode) and does not fail.
        """
        payload = self.to_dict()
        logger.info("SCRAPE_REPORT %s", json.dumps(payload, sort_keys=True))
        print(f"SCRAPE_REPORT {json.dumps(payload, sort_keys=True)}", flush=True)
        if not self.json_ok:
            return 1
        if self.db_ok is False:
            return 1
        return 0


@contextmanager
def stage_timer(report: StageReport, name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        report.set_stage(name, time.perf_counter() - start)


def retry_after_seconds(response: Any, *, default: float) -> float:
    """Parse Retry-After (seconds or HTTP-date); fall back to ``default``."""
    try:
        header = response.headers.get("Retry-After") if response is not None else None
    except Exception:
        header = None
    if not header:
        return default
    try:
        return max(float(header), 0.0)
    except (TypeError, ValueError):
        return default
