"""Load regular-season player gamelogs and summarize the raw points frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from points_xgb.config import REGULAR_SEASONS


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
    """Counts printed before feature building."""
    played = frame["minutes"] > 0
    rate = frame.loc[played, "pts"] / frame.loc[played, "minutes"]
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "players": int(frame["player_id"].nunique()),
        "games": int(frame["game_id"].nunique()),
        "duplicate_player_games": int(
            frame.duplicated(subset=["game_id", "player_id"]).sum()
        ),
        "missing_points": int(frame["pts"].isna().sum()),
        "missing_minutes": int(frame["minutes"].isna().sum()),
        "zero_minute_rows": int((frame["minutes"] == 0).sum()),
        "median_points": float(frame["pts"].median()),
        "mean_points": float(frame["pts"].mean()),
        "max_points": float(frame["pts"].max()),
        "median_rate": float(rate.median()) if played.any() else float("nan"),
        "mean_rate": float(rate.mean()) if played.any() else float("nan"),
        "max_rate": float(rate.max()) if played.any() else float("nan"),
    }


def format_data_quality(
    stats: dict[str, float | int],
    *,
    include_rate: bool = True,
) -> str:
    lines = [
        f"Number of rows : {stats['rows']}",
        f"Number of columns: {stats['columns']}",
        f"Number of Unique Players: {stats['players']}",
        f"Number of Unique Games: {stats['games']}",
        f"Total number of duplicates: {stats['duplicate_player_games']}",
        f"Missing points: {stats['missing_points']}",
        f"Missing minutes: {stats['missing_minutes']}",
        f"Zero-minute rows: {stats['zero_minute_rows']}",
        f"Median points: {stats['median_points']}",
        f"Mean points: {round(float(stats['mean_points']), 3)}",
    ]
    if include_rate:
        lines.extend(
            [
                (
                    "Median points per minute (appearances): "
                    f"{float(stats['median_rate']):.3f}"
                ),
                (
                    "Mean points per minute (appearances): "
                    f"{float(stats['mean_rate']):.3f}"
                ),
                f"Max raw points per minute: {float(stats['max_rate']):.3f}",
            ]
        )
    else:
        lines.append(f"Max points: {float(stats['max_points']):.3f}")
    return "\n".join(lines)
