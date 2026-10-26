from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

_MISSING_RATE_COLUMNS = (
    "player",
    "prop",
    "side",
    "line",
    "american_price",
    "event_key",
)
_DUPLICATE_COLUMNS = (
    "book",
    "snapshot_at",
    "player_norm",
    "event_key",
    "prop",
    "period",
    "side",
    "line",
)


@dataclass
class QualityReport:
    missing_rates_by_book: dict[str, dict[str, float]]
    duplicate_count: int
    schema_flags: dict[str, Any]
    dropped_by_schedule: int
    snapshot_span_by_book: dict[str, dict[str, Any]]


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"quotes missing required columns: {', '.join(missing)}")


def _books_with_all_null(
    quotes: pd.DataFrame, column: str
) -> list[str]:
    return [
        str(book)
        for book, values in quotes.groupby("book", sort=True)[column]
        if values.isna().all()
    ]


def _books_without_event_key_overlap(quotes: pd.DataFrame) -> list[str]:
    event_keys_by_book = {
        str(book): set(values.dropna().astype(str))
        for book, values in quotes.groupby("book", sort=True)["event_key"]
    }
    isolated = []
    for book, event_keys in event_keys_by_book.items():
        other_event_keys = set().union(
            *(
                other_keys
                for other_book, other_keys in event_keys_by_book.items()
                if other_book != book
            )
        )
        if not event_keys.intersection(other_event_keys):
            isolated.append(book)
    return isolated


def validate_quotes(quotes: pd.DataFrame) -> QualityReport:
    """Summarize missingness, duplicates, and snapshot coverage by book."""
    required = set(_MISSING_RATE_COLUMNS) | set(_DUPLICATE_COLUMNS) | {
        "is_main",
        "start_time",
    }
    _require_columns(quotes, required)

    missing_rates = (
        quotes.groupby("book", sort=True)[list(_MISSING_RATE_COLUMNS)]
        .agg(lambda values: float(values.isna().mean()))
        .to_dict(orient="index")
    )
    duplicate_count = int(quotes.duplicated(list(_DUPLICATE_COLUMNS)).sum())
    schema_flags = {
        "all_null_side": _books_with_all_null(quotes, "side"),
        "all_null_is_main": _books_with_all_null(quotes, "is_main"),
        "all_null_american_price": _books_with_all_null(
            quotes, "american_price"
        ),
        "missing_start_time": int(quotes["start_time"].isna().sum()),
        "books_without_event_key_overlap": _books_without_event_key_overlap(quotes),
    }
    spans = quotes.groupby("book", sort=True)["snapshot_at"].agg(["min", "max"])
    snapshot_spans = {
        str(book): {"min": row["min"], "max": row["max"]}
        for book, row in spans.iterrows()
    }
    return QualityReport(
        missing_rates_by_book=missing_rates,
        duplicate_count=duplicate_count,
        schema_flags=schema_flags,
        dropped_by_schedule=0,
        snapshot_span_by_book=snapshot_spans,
    )


def filter_schedule(
    quotes: pd.DataFrame,
    *,
    date: str,
    start_hour: int = 12,
    end_hour: int = 19,
    tz: str = "America/Los_Angeles",
) -> tuple[pd.DataFrame, int]:
    """Keep unknown starts and games within the local half-open time window."""
    _require_columns(quotes, {"start_time"})
    if not 0 <= start_hour < end_hour <= 24:
        raise ValueError("schedule hours must satisfy 0 <= start < end <= 24")

    local_day = pd.Timestamp(date).date()
    local_start = pd.Timestamp(local_day, tz=tz) + pd.Timedelta(hours=start_hour)
    local_end = pd.Timestamp(local_day, tz=tz) + pd.Timedelta(hours=end_hour)
    starts = pd.to_datetime(quotes["start_time"], utc=True, errors="coerce")
    keep = starts.isna() | (
        (starts >= local_start.tz_convert("UTC"))
        & (starts < local_end.tz_convert("UTC"))
    )
    kept = quotes.loc[keep].copy()
    return kept, int((~keep).sum())
