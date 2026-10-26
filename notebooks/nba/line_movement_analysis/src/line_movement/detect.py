"""Detect price and line movements in normalized quote snapshots.

Price and line continuity are independent series. When both move, this module
emits separate ``price`` and ``line`` rows instead of a combined ``both`` row.
"""

from __future__ import annotations

import pandas as pd

_CONTINUITY_COLS = ["book", "player_norm", "event_key", "prop", "period", "side"]
_PRICE_SERIES_COLS = [*_CONTINUITY_COLS, "line"]
_MOVEMENT_COLUMNS = [
    "book",
    "player_norm",
    "event_key",
    "prop",
    "period",
    "side",
    "series",
    "prev_line",
    "new_line",
    "prev_american_price",
    "new_american_price",
    "change_type",
    "line_up",
    "line_down",
    "price_shortened",
    "price_lengthened",
    "snapshot_at",
    "prev_snapshot_at",
    "lag_seconds",
    "line_choice",
]
_REQUIRED_COLUMNS = {
    *_PRICE_SERIES_COLS,
    "american_price",
    "is_main",
    "line_updated_at",
    "snapshot_at",
}
_EPSILON = 1e-9


def _changed(current: pd.Series, previous: pd.Series) -> pd.Series:
    one_missing = current.isna() ^ previous.isna()
    both_present = current.notna() & previous.notna()
    return one_missing | (both_present & current.sub(previous).abs().gt(_EPSILON))


def _base_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[_CONTINUITY_COLS].copy()


def _price_movements(quotes: pd.DataFrame) -> pd.DataFrame:
    ordered = quotes.sort_values(
        [*_PRICE_SERIES_COLS, "snapshot_at"], kind="stable", na_position="last"
    ).copy()
    grouped = ordered.groupby(_PRICE_SERIES_COLS, dropna=False, sort=False)
    previous_price = grouped["american_price"].shift()
    previous_snapshot = grouped["snapshot_at"].shift()
    has_previous = grouped.cumcount().gt(0)
    moved = has_previous & _changed(ordered["american_price"], previous_price)
    current = ordered.loc[moved]
    previous_price = previous_price.loc[moved]
    previous_snapshot = previous_snapshot.loc[moved]

    result = _base_rows(current)
    result["series"] = "price"
    result["prev_line"] = current["line"].to_numpy()
    result["new_line"] = current["line"].to_numpy()
    result["prev_american_price"] = previous_price.to_numpy()
    result["new_american_price"] = current["american_price"].to_numpy()
    result["change_type"] = "price"
    result["line_up"] = False
    result["line_down"] = False
    comparable = (
        current["american_price"].notna().to_numpy()
        & previous_price.notna().to_numpy()
    )
    result["price_shortened"] = comparable & (
        current["american_price"].to_numpy() < previous_price.to_numpy()
    )
    result["price_lengthened"] = comparable & (
        current["american_price"].to_numpy() > previous_price.to_numpy()
    )
    result["snapshot_at"] = current["snapshot_at"].to_numpy()
    result["prev_snapshot_at"] = previous_snapshot.to_numpy()
    result["lag_seconds"] = (
        current["snapshot_at"].to_numpy() - previous_snapshot.to_numpy()
    ) / pd.Timedelta(seconds=1)
    result["line_choice"] = pd.NA
    return result[_MOVEMENT_COLUMNS]


def _choose_line(group: pd.DataFrame) -> pd.Series:
    main = group.loc[group["is_main"].fillna(False).eq(True)]
    if not main.empty:
        selected = main.iloc[-1].copy()
        selected["line_choice"] = "main"
        return selected

    updated_at = pd.to_datetime(group["line_updated_at"], errors="coerce", utc=True)
    if updated_at.notna().any():
        selected = group.loc[updated_at.idxmax()].copy()
        selected["line_choice"] = "updated_at"
        return selected

    numeric_lines = pd.to_numeric(group["line"], errors="coerce")
    if numeric_lines.notna().any():
        median = numeric_lines.median()
        selected = group.loc[(numeric_lines - median).abs().idxmin()].copy()
    else:
        selected = group.iloc[-1].copy()
    selected["line_choice"] = "median"
    return selected


def _line_movements(quotes: pd.DataFrame) -> pd.DataFrame:
    numeric_lines = pd.to_numeric(quotes["line"], errors="coerce")
    quotes = quotes.loc[numeric_lines.isna() | numeric_lines.abs().gt(_EPSILON)]
    if quotes.empty:
        return pd.DataFrame(columns=_MOVEMENT_COLUMNS)

    snapshot_groups = [*_CONTINUITY_COLS, "snapshot_at"]
    chosen = (
        quotes.groupby(snapshot_groups, dropna=False, sort=False, group_keys=False)
        .apply(_choose_line, include_groups=False)
        .reset_index()
    )
    chosen = chosen.sort_values(
        [*_CONTINUITY_COLS, "snapshot_at"], kind="stable", na_position="last"
    )
    grouped = chosen.groupby(_CONTINUITY_COLS, dropna=False, sort=False)
    previous_line = grouped["line"].shift()
    previous_price = grouped["american_price"].shift()
    previous_snapshot = grouped["snapshot_at"].shift()
    has_previous = grouped.cumcount().gt(0)
    moved = has_previous & _changed(chosen["line"], previous_line)
    current = chosen.loc[moved]
    previous_line = previous_line.loc[moved]
    previous_price = previous_price.loc[moved]
    previous_snapshot = previous_snapshot.loc[moved]

    result = _base_rows(current)
    result["series"] = "line"
    result["prev_line"] = previous_line.to_numpy()
    result["new_line"] = current["line"].to_numpy()
    result["prev_american_price"] = previous_price.to_numpy()
    result["new_american_price"] = current["american_price"].to_numpy()
    result["change_type"] = "line"
    comparable = current["line"].notna().to_numpy() & previous_line.notna().to_numpy()
    result["line_up"] = comparable & (
        current["line"].to_numpy() > previous_line.to_numpy()
    )
    result["line_down"] = comparable & (
        current["line"].to_numpy() < previous_line.to_numpy()
    )
    result["price_shortened"] = False
    result["price_lengthened"] = False
    result["snapshot_at"] = current["snapshot_at"].to_numpy()
    result["prev_snapshot_at"] = previous_snapshot.to_numpy()
    result["lag_seconds"] = (
        current["snapshot_at"].to_numpy() - previous_snapshot.to_numpy()
    ) / pd.Timedelta(seconds=1)
    result["line_choice"] = current["line_choice"].to_numpy()
    return result[_MOVEMENT_COLUMNS]


def detect_movements(quotes: pd.DataFrame) -> pd.DataFrame:
    """Return price and selected-line changes between successive snapshots."""
    missing = sorted(_REQUIRED_COLUMNS.difference(quotes.columns))
    if missing:
        raise ValueError(f"quotes missing required columns: {', '.join(missing)}")
    if quotes.empty:
        return pd.DataFrame(columns=_MOVEMENT_COLUMNS)

    movements = pd.concat(
        [_price_movements(quotes), _line_movements(quotes)], ignore_index=True
    )
    if movements.empty:
        return pd.DataFrame(columns=_MOVEMENT_COLUMNS)
    return movements.sort_values(
        ["snapshot_at", "series"], kind="stable"
    ).reset_index(drop=True)
