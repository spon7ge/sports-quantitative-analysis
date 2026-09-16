"""Import and as-of selection of MLB strikeout market quotes."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.markets.odds import american_to_implied, decimal_to_implied
from src.mlb.schemas import MARKET_QUOTE_COLUMNS, coerce_frame

REQUIRED_QUOTE_FIELDS = (
    "game_pk",
    "pitcher_id",
    "sportsbook",
    "line",
    "over_price",
    "under_price",
    "fetched_at_utc",
    "quote_status",
)

_DEFAULT_PRICE_FORMAT = "american"
_DEFAULT_RULE_VERSION = "k_standard_v1"
_DEFAULT_JURISDICTION = "US"


def _read_quote_records(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            return pd.DataFrame(rows)
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("quotes", "data", "records"):
                if key in payload and isinstance(payload[key], list):
                    return pd.DataFrame(payload[key])
            return pd.DataFrame([payload])
        raise ValueError(f"Unsupported JSON payload in {path}")
    raise ValueError(f"Quotes file must be CSV or JSON, got {path.suffix}")


def import_quotes(path: str | Path, config: MlbConfig) -> pd.DataFrame:
    """Load CSV/JSON quotes and coerce append-only onto ``MARKET_QUOTE_COLUMNS``."""
    _ = config
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = _read_quote_records(source)
    missing = [name for name in REQUIRED_QUOTE_FIELDS if name not in frame.columns]
    if missing:
        raise ValueError(f"import_quotes missing required fields: {missing}")
    if "price_format" not in frame.columns:
        frame["price_format"] = _DEFAULT_PRICE_FORMAT
    else:
        frame["price_format"] = frame["price_format"].fillna(_DEFAULT_PRICE_FORMAT)
    if "rule_version" not in frame.columns:
        frame["rule_version"] = _DEFAULT_RULE_VERSION
    else:
        frame["rule_version"] = frame["rule_version"].fillna(_DEFAULT_RULE_VERSION)
    if "jurisdiction" not in frame.columns:
        frame["jurisdiction"] = _DEFAULT_JURISDICTION
    else:
        frame["jurisdiction"] = frame["jurisdiction"].fillna(_DEFAULT_JURISDICTION)
    if "quote_id" not in frame.columns:
        frame["quote_id"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    missing_ids = frame["quote_id"].isna() | (
        frame["quote_id"].astype(str).str.len() == 0
    )
    if missing_ids.any():
        generated = [
            f"q-{int(row.game_pk)}-{int(row.pitcher_id)}-{uuid4().hex[:12]}"
            for row in frame.loc[missing_ids].itertuples()
        ]
        frame.loc[missing_ids, "quote_id"] = generated
    return coerce_frame(frame, MARKET_QUOTE_COLUMNS)


def implied_over_price(over_price: float, price_format: str | None) -> float:
    fmt = str(price_format or _DEFAULT_PRICE_FORMAT).lower()
    if fmt == "decimal":
        return float(decimal_to_implied(float(over_price)))
    return float(american_to_implied(float(over_price)))


def implied_under_price(under_price: float, price_format: str | None) -> float:
    fmt = str(price_format or _DEFAULT_PRICE_FORMAT).lower()
    if fmt == "decimal":
        return float(decimal_to_implied(float(under_price)))
    return float(american_to_implied(float(under_price)))


def _target_line(row: pd.Series, default_line: float) -> float:
    for key in ("line", "market_line", "quoted_line"):
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return float(default_line)


def select_quotes_asof(
    predictions: pd.DataFrame,
    quotes: pd.DataFrame,
    config: MlbConfig,
) -> pd.DataFrame:
    """Attach at most one pre-cutoff quote per prediction row.

    Quotes with ``fetched_at_utc >= prediction_cutoff_utc`` are never used.
    When ``quote_latency_seconds > 0``, quotes that first appear after
    ``cutoff - latency`` are scored with the worst (highest implied) over
    price among quotes in ``(cutoff - latency, cutoff)``.
    """
    if predictions.empty or quotes.empty:
        return predictions.iloc[0:0].copy()

    pred = predictions.reset_index(drop=True).copy()
    pred["_pred_idx"] = np.arange(len(pred), dtype=np.int64)
    q = quotes.copy()
    quote_only = [c for c in q.columns if c not in ("game_pk", "pitcher_id")]
    pred = pred.drop(
        columns=[c for c in quote_only if c in pred.columns], errors="ignore"
    )
    if "fetched_at_utc" not in q.columns or "prediction_cutoff_utc" not in pred.columns:
        raise ValueError(
            "select_quotes_asof requires fetched_at_utc and prediction_cutoff_utc"
        )
    q["fetched_at_utc"] = pd.to_datetime(q["fetched_at_utc"], utc=True)
    pred["prediction_cutoff_utc"] = pd.to_datetime(
        pred["prediction_cutoff_utc"], utc=True
    )

    merged = pred.merge(
        q,
        on=["game_pk", "pitcher_id"],
        how="inner",
        suffixes=("", "_quote"),
    )
    if merged.empty:
        return merged

    fetched = pd.to_datetime(merged["fetched_at_utc"], utc=True)
    cutoff = pd.to_datetime(merged["prediction_cutoff_utc"], utc=True)
    merged = merged.loc[fetched < cutoff].copy()
    if merged.empty:
        return merged

    default_line = float(
        config.lines[1]
        if len(config.lines) > 1
        else (config.lines[0] if config.lines else 5.5)
    )
    latency = pd.Timedelta(seconds=float(config.quote_latency_seconds))
    quote_line_col = "line_quote" if "line_quote" in merged.columns else "line"

    picked_rows: list[pd.Series] = []
    for _, group in merged.groupby("_pred_idx", sort=True):
        row0 = group.iloc[0]
        target = _target_line(row0, default_line)
        line_values = pd.to_numeric(group[quote_line_col], errors="coerce")
        distance = (line_values - target).abs()
        nearest = distance.min()
        at_line = group.loc[distance == nearest]
        cutoff_i = pd.Timestamp(row0["prediction_cutoff_utc"])
        fetched_i = pd.to_datetime(at_line["fetched_at_utc"], utc=True)
        chosen: pd.Series
        if latency > pd.Timedelta(0):
            window_start = cutoff_i - latency
            in_window = (fetched_i > window_start) & (fetched_i < cutoff_i)
            if bool(in_window.any()):
                window = at_line.loc[in_window.to_numpy()]
                implied = [
                    implied_over_price(float(r["over_price"]), r.get("price_format"))
                    for _, r in window.iterrows()
                ]
                chosen = window.iloc[int(np.argmax(np.asarray(implied)))]
            else:
                known = at_line.loc[fetched_i <= window_start]
                if known.empty:
                    continue
                chosen = known.loc[
                    pd.to_datetime(known["fetched_at_utc"], utc=True).idxmax()
                ]
        else:
            chosen = at_line.loc[
                pd.to_datetime(at_line["fetched_at_utc"], utc=True).idxmax()
            ]
        picked_rows.append(chosen)

    if not picked_rows:
        return merged.iloc[0:0].copy()
    return pd.DataFrame(picked_rows).reset_index(drop=True)
