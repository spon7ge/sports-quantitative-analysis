from __future__ import annotations

import logging
import math
import os
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text

from src.odds.quote_specs import QuoteSpec, get_quote_spec

logger = logging.getLogger(__name__)

# Sports slate calendar day — same line/odds on a new ET date still upserts.
_SCRAPE_DATE_TZ = ZoneInfo("America/New_York")


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def values_equal(left: Any, right: Any) -> bool:
    if _is_null(left) and _is_null(right):
        return True
    if _is_null(left) or _is_null(right):
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, bool) or isinstance(right, bool):
            return left == right
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    return left == right


def scraped_date_key(value: Any) -> date | None:
    """Calendar date (America/New_York) from scraped_at for change-filter identity."""
    if _is_null(value):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(_SCRAPE_DATE_TZ).date()


def _identity_key(row: pd.Series, identity_cols: tuple[str, ...]) -> tuple:
    # Partition by scrape calendar day so identical line+odds still insert once
    # per ET date (board readers keep date-free identity → absolute latest).
    return (
        *tuple(row.get(col) for col in identity_cols),
        scraped_date_key(row.get("scraped_at")),
    )


def filter_unchanged_quotes(
    df: pd.DataFrame,
    *,
    latest: pd.DataFrame,
    spec: QuoteSpec,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df.copy(), 0

    latest_map: dict[tuple, pd.Series] = {}
    if latest is not None and not latest.empty:
        for _, row in latest.iterrows():
            latest_map[_identity_key(row, spec.identity_cols)] = row

    keep_idx: list[int] = []
    skipped = 0
    for idx, row in df.iterrows():
        prior = latest_map.get(_identity_key(row, spec.identity_cols))
        if prior is None:
            keep_idx.append(idx)
            continue
        changed = any(
            not values_equal(row.get(col), prior.get(col)) for col in spec.compare_cols
        )
        if changed:
            keep_idx.append(idx)
        else:
            skipped += 1

    return df.loc[keep_idx].copy(), skipped


def _skip_change_filter() -> bool:
    return os.environ.get("ODDS_SKIP_CHANGE_FILTER", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def fetch_latest_quotes(table: str, *, league: str, spec: QuoteSpec) -> pd.DataFrame:
    from src.utils.db import get_engine

    # scraped_at needed so change filter can partition identity by ET calendar day.
    cols = list(
        dict.fromkeys([*spec.identity_cols, *spec.compare_cols, "scraped_at"])
    )
    col_sql = ", ".join(f'"{c}"' for c in cols)
    identity_sql = ", ".join(f'"{c}"' for c in spec.identity_cols)
    order_sql = identity_sql + ', "scraped_at" DESC'
    sql = text(
        f"""
SELECT DISTINCT ON ({identity_sql}) {col_sql}
FROM odds.{table}
WHERE league = :league
ORDER BY {order_sql}
"""
    )
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"league": league})


def apply_change_filter(table: str, df: pd.DataFrame, *, league: str) -> pd.DataFrame:
    if df.empty or _skip_change_filter():
        return df
    spec = get_quote_spec(table)
    latest = fetch_latest_quotes(table, league=league, spec=spec)
    kept, skipped = filter_unchanged_quotes(df, latest=latest, spec=spec)
    if skipped:
        logger.info(
            "odds change filter table=%s league=%s kept=%s skipped_unchanged=%s",
            table,
            league,
            len(kept),
            skipped,
        )
    return kept
