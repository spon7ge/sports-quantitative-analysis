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

# Cap VALUES/LATERAL batches so bind params stay reasonable.
_IDENTITY_LOOKUP_BATCH = 500


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


def identity_key_rows(df: pd.DataFrame, identity_cols: tuple[str, ...]) -> pd.DataFrame:
    """Unique identity tuples present in the incoming scrape (complete key)."""
    present = [c for c in identity_cols if c in df.columns]
    if not present:
        return pd.DataFrame(columns=list(identity_cols))
    return df.loc[:, present].drop_duplicates().reset_index(drop=True)


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


def _quote_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return f'"{name}"'


def _safe_table(table: str) -> str:
    if not table.replace("_", "").isalnum():
        raise ValueError(f"unsafe table name: {table!r}")
    return table


def _latest_for_identity_batch_sql(
    table: str,
    *,
    identity_cols: tuple[str, ...],
    select_cols: list[str],
    n_rows: int,
) -> str:
    """One latest row per incoming identity via indexed LATERAL lookup."""
    id_cols_sql = ", ".join(_quote_ident(c) for c in identity_cols)
    select_sql = ", ".join(f"latest.{_quote_ident(c)}" for c in select_cols)
    value_rows = []
    for i in range(n_rows):
        cells = ", ".join(f":id_{i}_{j}" for j in range(len(identity_cols)))
        value_rows.append(f"({cells})")
    values_sql = ", ".join(value_rows)
    join_preds = " AND ".join(
        f"t.{_quote_ident(c)} IS NOT DISTINCT FROM ids.{_quote_ident(c)}"
        for c in identity_cols
    )
    lateral_select = ", ".join(f"t.{_quote_ident(c)}" for c in select_cols)
    safe = _safe_table(table)
    return f"""
SELECT {select_sql}
FROM (VALUES {values_sql}) AS ids({id_cols_sql})
CROSS JOIN LATERAL (
  SELECT {lateral_select}
  FROM odds.{safe} t
  WHERE t.league = :league
    AND {join_preds}
  ORDER BY t."scraped_at" DESC
  LIMIT 1
) latest
"""


def fetch_latest_quotes(
    table: str,
    *,
    league: str,
    spec: QuoteSpec,
    incoming: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Latest stored quote per identity present in ``incoming``.

    Looks up only scrape identities (complete key, including book-specific
    fields like ``odds_type``). Avoids a full-history DISTINCT ON scan.
    """
    from src.utils.db import db_load_slot, get_engine

    if incoming is None:
        raise TypeError("incoming DataFrame is required for identity-scoped lookup")
    if incoming.empty:
        return pd.DataFrame()

    keys = identity_key_rows(incoming, spec.identity_cols)
    if keys.empty:
        return pd.DataFrame()

    select_cols = list(
        dict.fromkeys([*spec.identity_cols, *spec.compare_cols, "scraped_at"])
    )
    engine = get_engine()
    frames: list[pd.DataFrame] = []

    with db_load_slot():
        with engine.connect() as conn:
            for start in range(0, len(keys), _IDENTITY_LOOKUP_BATCH):
                batch = keys.iloc[start : start + _IDENTITY_LOOKUP_BATCH]
                sql = text(
                    _latest_for_identity_batch_sql(
                        table,
                        identity_cols=spec.identity_cols,
                        select_cols=select_cols,
                        n_rows=len(batch),
                    )
                )
                params: dict[str, Any] = {"league": league}
                for i, row in enumerate(batch.itertuples(index=False, name=None)):
                    for j, val in enumerate(row):
                        params[f"id_{i}_{j}"] = None if _is_null(val) else val
                frames.append(pd.read_sql(sql, conn, params=params))

    if not frames:
        return pd.DataFrame(columns=select_cols)
    return pd.concat(frames, ignore_index=True)


def apply_change_filter(table: str, df: pd.DataFrame, *, league: str) -> pd.DataFrame:
    if df.empty or _skip_change_filter():
        return df
    import time

    t0 = time.perf_counter()
    spec = get_quote_spec(table)
    latest = fetch_latest_quotes(table, league=league, spec=spec, incoming=df)
    kept, skipped = filter_unchanged_quotes(df, latest=latest, spec=spec)
    elapsed = time.perf_counter() - t0
    logger.info(
        "odds change filter table=%s league=%s kept=%s skipped_unchanged=%s "
        "latest_rows=%s filter_s=%.3f",
        table,
        league,
        len(kept),
        skipped,
        0 if latest is None else len(latest),
        elapsed,
    )
    return kept
