"""Postgres upsert helper for odds snapshot loads."""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import create_engine

from src.scrapers.mlb.paths import repo_root

logger = logging.getLogger(__name__)

# Default measured batch size for execute_values bulk upserts.
DEFAULT_UPSERT_BATCH_SIZE = 5000

_slot_depth = threading.local()


def _load_env() -> None:
    path = repo_root() / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


_load_env()


@lru_cache(maxsize=1)
def get_engine():
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. Add it to .env at the repo root."
        )
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=4,
        max_overflow=4,
        connect_args={"connect_timeout": 20, "sslmode": "require"},
    )


def _db_slot_count() -> int:
    raw = os.environ.get("ODDS_DB_LOAD_SLOTS", "2").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 2
    return max(n, 1)


def _db_lock_dir() -> Path:
    override = os.environ.get("ODDS_DB_LOCK_DIR", "").strip()
    if override:
        path = Path(override)
    else:
        path = repo_root() / "data" / ".odds_db_locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def db_load_slot():
    """Limit concurrent odds DB work across scraper subprocesses (default 2).

    Re-entrant in-process so change-filter lookup + upsert share one slot.
    """
    import fcntl

    depth = getattr(_slot_depth, "n", 0)
    if depth > 0:
        _slot_depth.n = depth + 1
        try:
            yield
        finally:
            _slot_depth.n = depth
        return

    slots = _db_slot_count()
    lock_dir = _db_lock_dir()
    handles: list = []
    acquired = None
    try:
        while acquired is None:
            for i in range(slots):
                fh = open(lock_dir / f"slot_{i}.lock", "a+", encoding="utf-8")
                handles.append(fh)
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = fh
                    break
                except BlockingIOError:
                    fh.close()
                    handles.pop()
            if acquired is None:
                time.sleep(0.05)
        _slot_depth.n = 1
        yield
    finally:
        _slot_depth.n = 0
        if acquired is not None:
            try:
                fcntl.flock(acquired.fileno(), fcntl.LOCK_UN)
            finally:
                acquired.close()
        for fh in handles:
            if fh is not acquired and not fh.closed:
                fh.close()


def _normalize_col(name: str) -> str:
    if "_" in name:
        return name.lower()
    s = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
        "_",
        name,
    ).lower()
    return re.sub(r"(\d)_([a-z])", r"\1\2", s)


def _clean_val(v):
    if v is pd.NaT or v is pd.NA:
        return None
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        if math.isnan(fv):
            return None
        if fv == int(fv):
            return int(fv)
        return fv
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return v


def _df_to_tuples(df: pd.DataFrame) -> tuple[list[str], list[tuple]]:
    cols = list(df.columns)
    rows = [
        tuple(_clean_val(v) for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    return cols, rows


def _dedupe_rows_by_conflict(
    cols: list[str],
    rows: list[tuple],
    conflict_cols: list[str],
) -> tuple[list[str], list[tuple]]:
    """Keep last row per conflict key so ON CONFLICT is not CardinalityViolation."""
    key_idx = [cols.index(c) for c in conflict_cols if c in cols]
    if not key_idx or not rows:
        return cols, rows
    seen: dict[tuple, tuple] = {}
    order: list[tuple] = []
    for row in rows:
        key = tuple(row[i] for i in key_idx)
        if key not in seen:
            order.append(key)
        seen[key] = row
    deduped = [seen[k] for k in order]
    dropped = len(rows) - len(deduped)
    if dropped:
        logger.warning(
            "upsert_df dropped %s duplicate row(s) on %s",
            dropped,
            ", ".join(conflict_cols),
        )
    return cols, deduped


@lru_cache(maxsize=32)
def _table_columns(schema: str, table: str) -> frozenset[str]:
    q = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %(schema)s AND table_name = %(table)s
    """
    cols = pd.read_sql(q, get_engine(), params={"schema": schema, "table": table})[
        "column_name"
    ]
    return frozenset(cols.astype(str).str.lower())


def _align_df_to_table(
    df: pd.DataFrame,
    *,
    schema: str,
    table: str,
) -> pd.DataFrame:
    table_cols = _table_columns(schema, table)
    if not table_cols:
        raise RuntimeError(
            f"No columns found for {schema}.{table} — table is missing."
        )
    out = df.copy()
    out.columns = [_normalize_col(c) for c in out.columns]
    keep = [c for c in out.columns if c in table_cols]
    return out[keep]


_NULLABLE_UPSERT_CONFLICT_COLS = frozenset({"points", "line_score", "event_id"})


def upsert_df(
    table: str,
    df: pd.DataFrame,
    schema: str = "odds",
    conflict_cols: list[str] | None = None,
    batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    *,
    lineage_col: str | None = "fetched_at",
) -> None:
    """Upsert a DataFrame into Postgres (``odds`` schema by default)."""
    if df.empty:
        return
    if not conflict_cols:
        raise ValueError(f"conflict_cols required for table '{table}'")

    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]
    if lineage_col:
        df[lineage_col] = datetime.now(UTC)

    pk_cols = [c for c in conflict_cols if c in df.columns]
    if pk_cols:
        required_pk = [c for c in pk_cols if c not in _NULLABLE_UPSERT_CONFLICT_COLS]
        if required_pk:
            df = df.dropna(subset=required_pk)

    aligned = _align_df_to_table(df, schema=schema, table=table)
    cols, rows = _df_to_tuples(aligned)
    cols, rows = _dedupe_rows_by_conflict(cols, rows, conflict_cols)
    if not rows:
        return

    col_list = ", ".join(f'"{c}"' for c in cols)
    conflict = ", ".join(f'"{c}"' for c in conflict_cols)
    update_cols = [c for c in cols if c not in conflict_cols]
    updates = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
    sql = (
        f"INSERT INTO {schema}.{table} ({col_list}) VALUES %s "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    )
    engine = get_engine()
    t0 = time.perf_counter()
    with db_load_slot():
        conn = engine.raw_connection()
        try:
            cur = conn.cursor()
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                execute_values(cur, sql, batch, page_size=len(batch))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()
    logger.info(
        "upsert_df table=%s.%s rows=%s batch_size=%s upsert_s=%.3f",
        schema,
        table,
        len(rows),
        batch_size,
        time.perf_counter() - t0,
    )
