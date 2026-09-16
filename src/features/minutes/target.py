"""Canonical pregame minutes target."""

from __future__ import annotations

import pandas as pd

from src.pipeline.silver.columns import parse_minutes


def reconcile_target_minutes(
    frame: pd.DataFrame,
) -> pd.Series:
    """Prefer parsed ``min_sec``, then ``min``, then ``minutes``."""
    for column in ("min_sec", "min", "minutes"):
        if column not in frame:
            continue
        parsed = parse_minutes(frame[column])
        if parsed.notna().any():
            return parsed

    raise ValueError(
        "No usable minutes column found; "
        "expected min_sec, min, or minutes."
    )
