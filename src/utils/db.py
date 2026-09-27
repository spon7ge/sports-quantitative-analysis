"""Postgres upsert helper for odds snapshot loads."""

from __future__ import annotations

import math
import os
import re
from datetime import UTC, date, datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import create_engine

from src.scrapers.mlb.paths import repo_root


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
        connect_args={"connect_timeout": 20, "sslmode": "require"},
    )


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
    batch_size: int = 2000,
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
    col_list = ", ".join(f'"{c}"' for c in cols)
    conflict = ", ".join(f'"{c}"' for c in conflict_cols)
    update_cols = [c for c in cols if c not in conflict_cols]
    updates = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
    sql = (
        f"INSERT INTO {schema}.{table} ({col_list}) VALUES %s "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    )
    engine = get_engine()
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
