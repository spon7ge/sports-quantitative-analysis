"""DataFrame validation and normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import RAW_DATASETS

ID_COLUMN_ALIASES = {
    "GAME_ID": "game_id",
    "PLAYER_ID": "player_id",
    "TEAM_ID": "team_id",
    "gameId": "game_id",
    "personId": "player_id",
    "teamId": "team_id",
}

def normalize_datasets(
    datasets: str | Iterable[str] | None,
) -> list[str]:
    if datasets is None:
        return list(RAW_DATASETS)

    requested = [datasets] if isinstance(datasets, str) else list(datasets)
    unknown = set(requested) - set(RAW_DATASETS)

    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {sorted(unknown)}. "
            f"Choose from: {', '.join(RAW_DATASETS)}"
        )

    return [name for name in RAW_DATASETS if name in requested]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identity columns without changing other API column names."""
    aliases = {
        source: target
        for source, target in ID_COLUMN_ALIASES.items()
        if source in df
    }
    result = df.rename(columns=aliases).copy()

    if "game_id" in result:
        result["game_id"] = result["game_id"].astype(str).str.zfill(10)

    for column in ("player_id", "team_id"):
        if column in result:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ).astype("Int64")

    return result

def rows_for_games(
    df: pd.DataFrame,
    game_ids: set[str],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result = normalize_columns(df)

    if "game_id" not in result:
        return result

    return result.loc[result["game_id"].isin(game_ids)].copy()