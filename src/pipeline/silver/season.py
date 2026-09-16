"""Assign NBA season labels in YYYY-YY form, e.g. 2025-26."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re

import pandas as pd

SEASON_COLUMN = "season"
_SEASON_LABEL = re.compile(r"^(19|20)\d{2}-\d{2}$")
_SEASON_START = re.compile(r"^(19|20)\d{2}")


def format_nba_season(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def assign_season(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    labels = pd.Series(pd.NA, index=result.index, dtype="string")

    if "season_year" in result:
        labels = result["season_year"].map(_normalize_season_label).astype("string")

    if "game_date" in result:
        from_dates = _season_from_game_date(result["game_date"])
        labels = labels.fillna(from_dates)

    if labels.isna().any():
        raise ValueError(
            "could not assign season; need season_year or game_date"
        )

    result[SEASON_COLUMN] = labels
    return result


def load_player_gamelogs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        suffix = source.suffix.lower()
        if suffix == ".parquet":
            frame = pd.read_parquet(source)
        elif suffix == ".csv":
            frame = pd.read_csv(source)
        else:
            raise ValueError(f"unsupported input format: {source}")

        frames.append(frame)

    if not frames:
        raise ValueError("at least one input path is required")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return assign_season(combined)


def _normalize_season_label(value: object) -> str | pd.NA:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NA

    if _SEASON_LABEL.fullmatch(text):
        return format_nba_season(int(text[:4]))

    match = _SEASON_START.match(text)
    if match:
        return format_nba_season(int(match.group(0)))

    return pd.NA


def _season_from_game_date(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce")
    start_year = parsed.dt.year.where(parsed.dt.month >= 8, parsed.dt.year - 1)
    return start_year.map(
        lambda year: format_nba_season(int(year)) if pd.notna(year) else pd.NA
    ).astype("string")
