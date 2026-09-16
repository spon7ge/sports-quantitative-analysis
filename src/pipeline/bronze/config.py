"""League and dataset configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LeagueKey = Literal["nba", "wnba"]

RAW_DATASETS = (
    "player_base",
    "player_adv",
    "team_base",
    "team_adv",
    "start_positions",
)

LEAGUE_WIDE_DATASETS = frozenset(RAW_DATASETS[:4])

@dataclass(frozen=True)
class LeagueConfig:
    key: LeagueKey
    label: str
    league_id: str | None
    parquet_name_by_dataset: dict[str, str]
    default_season: str
    checkpoint_name: str

LEAGUES: dict[LeagueKey, LeagueConfig] = {
    "nba": LeagueConfig(
        key="nba",
        label="NBA",
        league_id=None,
        parquet_name_by_dataset={
            "player_base": "nba_player_base.parquet",
            "player_adv": "nba_player_adv.parquet",
            "team_base": "nba_team_base.parquet",
            "team_adv": "nba_team_adv.parquet",
            "start_positions": "nba_player_tracking.parquet",
        },
        default_season="2025-26",
        checkpoint_name="nba_start_positions_checkpoint.parquet",
    ),
    "wnba": LeagueConfig(
        key="wnba",
        label="WNBA",
        league_id="10",
        parquet_name_by_dataset={
            "player_base": "wnba_player_base.parquet",
            "player_adv": "wnba_player_adv.parquet",
            "team_base": "wnba_team_base.parquet",
            "team_adv": "wnba_team_adv.parquet",
            "start_positions": "wnba_start_positions.parquet",
        },
        default_season="2025",
        checkpoint_name="wnba_start_positions_checkpoint.parquet",
    ),
}

def checkpoint_path(config: LeagueConfig, output_dir: Path) -> Path:
    return output_dir / "cache" / config.checkpoint_name