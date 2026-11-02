"""Assemble the frame the minutes notebook trains on."""

from __future__ import annotations

import pandas as pd

from minutes_xgb.paths import ensure_repo_on_path


def add_total_stars_active(frame: pd.DataFrame) -> pd.DataFrame:
    """Count how many of starter ranks 1–3 played in this team-game.

    Rank is prior starts, then prior minutes. A player with missing
    minutes counts as not having played. Every row in the team-game
    receives the same count, including players outside the top three.
    """
    ensure_repo_on_path()
    from src.features.minutes import add_total_stars_active as shared_star_count

    return shared_star_count(frame)


def build_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add starter, minutes, and star-count columns.

    Starter and minutes features come from the shared pipeline.
    ``total_stars_active`` is applied after those columns exist.
    """
    ensure_repo_on_path()
    from src.features.minutes import add_minutes_features, add_starter_features

    featured = add_starter_features(frame)
    featured = add_minutes_features(featured)
    return add_total_stars_active(featured)


def positive_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a positive parsed minute total."""
    return frame.loc[frame["minutes"] > 0].copy()
