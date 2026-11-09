"""Walk-forward dates for the game simulator.

A game on November 6 is scored with every model fit on November 5 and
earlier. Rows from the slate date are targets. They are not training rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def parse_slate_dates(dates: Sequence[str]) -> list[pd.Timestamp]:
    """Unique calendar dates, earliest first."""
    if isinstance(dates, str) or not isinstance(dates, Sequence) or not dates:
        raise ValueError("slate dates must be a non-empty list of dates")
    parsed = []
    for value in dates:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError(f"unreadable slate date {value!r}")
        parsed.append(stamp.normalize())
    return sorted(set(parsed))


def _dated(logs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(logs, pd.DataFrame) or "game_date" not in logs.columns:
        raise ValueError("logs need a game_date column")
    out = logs.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    if out["game_date"].isna().any():
        raise ValueError("logs have a missing game_date")
    return out


def history_before(logs: pd.DataFrame, slate) -> pd.DataFrame:
    """Rows strictly before ``slate``. The slate date itself is excluded."""
    frame = _dated(logs)
    day = pd.Timestamp(slate).normalize()
    if pd.isna(day):
        raise ValueError(f"unreadable slate date {slate!r}")
    kept = frame.loc[frame["game_date"] < day].reset_index(drop=True)
    if kept.empty:
        raise ValueError(f"no completed games before {day.date()}")
    if not (kept["game_date"] < day).all():
        raise AssertionError("history includes the slate date")
    return kept


def played_dates(logs: pd.DataFrame, start, end) -> list[pd.Timestamp]:
    """Dates in ``[start, end]`` that have at least one completed game.

    Calendar days with no rows, including league off days inside the window,
    are omitted. A window with no games at all is an error.
    """
    frame = _dated(logs)
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    if pd.isna(first) or pd.isna(last) or last < first:
        raise ValueError("end must be on or after the start")
    window = frame.loc[frame["game_date"].between(first, last, inclusive="both")]
    dates = sorted(pd.Timestamp(day) for day in window["game_date"].unique())
    if not dates:
        raise ValueError(
            f"no completed games from {first.date()} through {last.date()}"
        )
    return dates


def played_on(logs: pd.DataFrame, slate) -> pd.DataFrame:
    """Rows from the slate date. Use these as actuals, not as a fit."""
    frame = _dated(logs)
    day = pd.Timestamp(slate).normalize()
    if pd.isna(day):
        raise ValueError(f"unreadable slate date {slate!r}")
    return frame.loc[frame["game_date"].eq(day)].reset_index(drop=True)


def assert_before(cutoff, slate, label: str) -> pd.Timestamp:
    """The fitted cutoff has to be a calendar day before the game."""
    trained = pd.Timestamp(cutoff).normalize()
    day = pd.Timestamp(slate).normalize()
    if pd.isna(trained) or pd.isna(day):
        raise ValueError(f"{label} cutoff or slate date is missing")
    if trained >= day:
        raise ValueError(
            f"{label} cutoff {trained.date()} is not before {day.date()}"
        )
    return trained


def environment_cutoff(model: dict) -> pd.Timestamp:
    """Latest team date stored on a Layer 1 model."""
    teams = (model.get("state") or {}).get("teams") or {}
    dates = [
        pd.Timestamp(team["last_date"]).normalize()
        for team in teams.values()
        if team.get("last_date")
    ]
    if not dates:
        raise ValueError("environment model has no team dates")
    return max(dates)
