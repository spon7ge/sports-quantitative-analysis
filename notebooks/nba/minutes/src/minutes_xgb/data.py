"""Load regular-season player gamelogs and summarize the raw frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from minutes_xgb.config import REGULAR_SEASONS


def load_regular_season(
    root: Path,
    seasons: tuple[str, ...] = REGULAR_SEASONS,
) -> pd.DataFrame:
    """Concatenate silver regular-season gamelogs and flag recorded starters."""
    frames = [
        pd.read_parquet(
            root
            / "data/silver/nba"
            / season
            / "regular_season/player_gamelogs.parquet"
        )
        for season in seasons
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["starting"] = (
        frame["start_position"]
        .astype("string")
        .str.strip()
        .isin(["G", "F", "C"])
        .astype(int)
    )
    return frame


def data_quality(frame: pd.DataFrame) -> dict[str, float | int]:
    """Counts and minute-distribution stats printed before feature building."""
    average = frame.groupby("player_id")["minutes"].mean()
    stats: dict[str, float | int] = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "players": int(frame["player_id"].nunique()),
        "games": int(frame["game_id"].nunique()),
        "duplicate_player_games": int(
            frame.duplicated(subset=["game_id", "player_id"]).sum()
        ),
        "missing_minutes": int(frame["minutes"].isna().sum()),
        "missing_starting": int(frame["starting"].isna().sum()),
        "median_minutes": float(frame["minutes"].median()),
        "mean_minutes": float(frame["minutes"].mean()),
        "max_minutes": float(frame["minutes"].max()),
        "std_minutes": float(frame["minutes"].std()),
        "skew_minutes": float(frame["minutes"].skew()),
        "kurtosis_minutes": float(frame["minutes"].kurt()),
    }
    for threshold in (10, 20, 30, 35):
        stats[f"pct_players_avg_ge_{threshold}"] = float(
            (average >= threshold).mean() * 100
        )
    return stats


def format_data_quality(stats: dict[str, float | int]) -> str:
    lines = [
        f"Number of rows : {stats['rows']}",
        f"Number of columns: {stats['columns']}",
        f"Number of Unique Players: {stats['players']}",
        f"Number of Unique Games: {stats['games']}",
        f"Total number of duplicates: {stats['duplicate_player_games']}",
        f"Missing Labels: {stats['missing_minutes']}",
        f"Missing Start Position: {stats['missing_starting']}",
        f"Median Minutes: {stats['median_minutes']}",
        f"Mean Minutes: {round(float(stats['mean_minutes']), 3)}",
        f"Max Minutes: {round(float(stats['max_minutes']), 3)}",
        f"Standard Deviation of Minutes: {round(float(stats['std_minutes']), 3)}",
        f"Skewness of Minutes: {round(float(stats['skew_minutes']), 3)}",
        f"Kurtosis of Minutes: {round(float(stats['kurtosis_minutes']), 3)}",
    ]
    for threshold in (10, 20, 30, 35):
        pct = float(stats[f"pct_players_avg_ge_{threshold}"])
        lines.append(
            f"Percentage of players averaging {threshold}+ minutes: {pct:.1f}%"
        )
    return "\n".join(lines)
