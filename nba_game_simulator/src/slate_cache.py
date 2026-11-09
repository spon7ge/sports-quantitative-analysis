"""Disk cache of one slate's simulated games.

A date already on disk is reused. A date that is not on disk is fit and
simulated. Changing the simulator does not invalidate a stored date.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

_FRAMES = (
    "game_sims",
    "player_sims",
    "game_actuals",
    "player_actuals",
    "stopped",
)


def _day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value).normalize()
    if pd.isna(stamp):
        raise ValueError(f"unreadable slate date {value!r}")
    return stamp


def cache_path(root, day) -> Path:
    """Pickle path for one slate date."""
    return Path(root) / f"{_day(day).date().isoformat()}.pkl"


def missing_slates(root, dates: Sequence) -> list:
    """Dates with no stored sims, in the caller's order."""
    if isinstance(dates, str) or not isinstance(dates, Sequence):
        raise ValueError("slate dates must be a list of dates")
    return [day for day in dates if not cache_path(root, day).is_file()]


def _check_frame(frame: pd.DataFrame, day: pd.Timestamp, label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} must be a DataFrame")
    if frame.empty:
        return
    if "slate_date" not in frame.columns:
        raise ValueError(f"{label} needs a slate_date column")
    stamps = pd.to_datetime(frame["slate_date"], errors="coerce").dt.normalize()
    if stamps.isna().any() or not stamps.eq(day).all():
        raise ValueError(f"{label} rows are not all from {day.date()}")


def save_slate(root, day, **frames: pd.DataFrame) -> Path:
    """Write one slate. Rows from any other date are rejected."""
    stamp = _day(day)
    missing = [name for name in _FRAMES if name not in frames]
    if missing:
        raise ValueError(f"slate cache needs {missing}")
    for name in _FRAMES:
        _check_frame(frames[name], stamp, name)
    path = cache_path(root, stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slate_date": stamp.date().isoformat(),
        **{name: frames[name] for name in _FRAMES},
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)
    return path


def load_slate(root, day) -> dict[str, pd.DataFrame]:
    """Load one slate and reject a file whose rows belong to another date."""
    stamp = _day(day)
    path = cache_path(root, stamp)
    if not path.is_file():
        raise FileNotFoundError(f"no cached sims for {stamp.date()}")
    payload = pickle.loads(path.read_bytes())
    if payload.get("slate_date") != stamp.date().isoformat():
        raise ValueError(f"cache file {path.name} is not {stamp.date()}")
    for name in _FRAMES:
        if name not in payload:
            raise ValueError(f"cache file {path.name} is missing {name}")
        _check_frame(payload[name], stamp, name)
    return {name: payload[name] for name in _FRAMES}


def read_slates(root, dates: Sequence) -> dict[str, pd.DataFrame]:
    """Concatenate stored slates in ``dates`` order."""
    absent = missing_slates(root, dates)
    if absent:
        labels = ", ".join(_day(day).date().isoformat() for day in absent)
        raise ValueError(f"no cached sims for {labels}")
    parts = [load_slate(root, day) for day in dates]
    stored = {}
    for name in _FRAMES:
        frames = [part[name] for part in parts if not part[name].empty]
        stored[name] = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
    return stored
