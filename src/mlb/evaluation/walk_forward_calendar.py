from __future__ import annotations

from pathlib import Path

import pandas as pd

CALENDAR_COLUMNS = (
    "season",
    "regular_season_open_date",
    "regular_season_close_date",
)


def calendar_from_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    frame = schedule.copy()
    regular = frame["game_type"].astype(str).eq("R")
    final = frame["abstract_game_state"].astype(str).isin({"Final", "Completed"})
    kept = frame.loc[regular & final].copy()
    if kept.empty:
        return pd.DataFrame(columns=list(CALENDAR_COLUMNS))
    kept["official_date"] = kept["official_date"].astype(str)
    grouped = kept.groupby("season", as_index=False).agg(
        regular_season_open_date=("official_date", "min"),
        regular_season_close_date=("official_date", "max"),
    )
    grouped["season"] = grouped["season"].astype(int)
    return grouped.loc[:, list(CALENDAR_COLUMNS)].sort_values("season")


def validate_regular_season_calendar(
    frame: pd.DataFrame,
    seasons: range = range(2018, 2026),
) -> None:
    required = set(seasons)
    present = set(int(s) for s in frame["season"])
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"calendar missing seasons: {missing}")
    if frame["season"].duplicated().any():
        raise ValueError("calendar seasons must be unique")
    opened = pd.to_datetime(frame["regular_season_open_date"], errors="coerce")
    closed = pd.to_datetime(frame["regular_season_close_date"], errors="coerce")
    if opened.isna().any() or closed.isna().any():
        raise ValueError("calendar dates must parse")
    if (opened > closed).any():
        raise ValueError("calendar open must be on or before close")


def load_regular_season_calendar(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in CALENDAR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"calendar missing columns: {missing}")
    out = frame.loc[:, list(CALENDAR_COLUMNS)].copy()
    out["season"] = out["season"].astype(int)
    out["regular_season_open_date"] = out["regular_season_open_date"].astype(str)
    out["regular_season_close_date"] = out["regular_season_close_date"].astype(str)
    validate_regular_season_calendar(out)
    return out
