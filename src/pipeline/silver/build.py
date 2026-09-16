"""Orchestrate the raw-Parquet to silver-Parquet pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd

from src.pipeline.bronze.config import LEAGUES
from src.pipeline.bronze.fetch import GameLogs

from .columns import normalize_columns, prepare_tracking
from .merge import merge_gamelogs
from .positions import enrich_positions, load_player_positions
from .rotowire import enrich_rotowire, load_rotowire

LeagueKey = Literal["nba", "wnba"]

logger = logging.getLogger(__name__)

DATASETS = (
    "player_base",
    "player_adv",
    "team_base",
    "team_adv",
    "start_positions",
)

def build_silver(
    season: str,
    season_type: str = "Regular Season",
    *,
    league: LeagueKey = "nba",
    raw_dir: str | Path = "data/bronze",
    silver_dir: str | Path = "data/silver",
    positions_dir: str | Path = "data/bronze/player_positions",
    rotowire_dir: str | Path = "data/bronze/rotowire",
    raw_frames: dict[str, pd.DataFrame] | None = None,
    auto_scrape_rotowire: bool = False,
) -> pd.DataFrame:
    if league not in LEAGUES:
        raise ValueError(f"Unknown league: {league!r}")

    frames = (
        prepare_raw_frames(raw_frames)
        if raw_frames is not None
        else read_raw_parquet(raw_dir, league)
    )

    _validate_required_frames(frames)

    silver = merge_gamelogs(
        frames["player_base"],
        frames.get("player_adv"),
        frames["team_base"],
        frames.get("team_adv"),
        frames.get("start_positions"),
    )

    silver["season_type"] = season_type

    positions = load_player_positions(
        season,
        league=league,
        positions_dir=positions_dir,
    )
    silver = enrich_positions(
        silver,
        positions,
        league=league,
    )

    rotowire = load_rotowire(
        season,
        league=league,
        rotowire_dir=rotowire_dir,
        auto_scrape=auto_scrape_rotowire,
    )
    silver = enrich_rotowire(silver, rotowire)

    output_path = silver_path(
        silver_dir,
        league,
        season,
        season_type,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(output_path, index=False)

    logger.info(
        "Saved %d silver rows and %d columns to %s",
        len(silver),
        len(silver.columns),
        output_path,
    )

    return silver

def fetch_and_build_silver(
    season: str,
    season_type: str = "Regular Season",
    *,
    league: LeagueKey = "nba",
    raw_dir: str | Path = "data/bronze",
    silver_dir: str | Path = "data/silver",
    positions_dir: str | Path = "data/bronze/player_positions",
    rotowire_dir: str | Path = "data/bronze/rotowire",
    auto_scrape_rotowire: bool = False,
    datasets: str | Iterable[str] | None = None,
    parallel: bool = True,
    checkpoint: str | Path | None = None,
    batch_size: int = 100,
    start_position_delay: float = 2.5,
    start_position_workers: int = 5,
    run_all_batches: bool = True,
) -> pd.DataFrame:
    logs = GameLogs(
        season=season,
        season_type=season_type,
        league=league,
        output_dir=raw_dir,
    )

    logs.fetch(
        datasets=datasets,
        parallel=parallel,
        checkpoint=checkpoint,
        batch_size=batch_size,
        start_position_delay=start_position_delay,
        start_position_workers=start_position_workers,
        run_all_batches=run_all_batches,
    )

    return build_silver(
        season,
        season_type,
        league=league,
        raw_dir=raw_dir,
        silver_dir=silver_dir,
        positions_dir=positions_dir,
        rotowire_dir=rotowire_dir,
        raw_frames=logs.data,
        auto_scrape_rotowire=auto_scrape_rotowire,
    )

def read_raw_parquet(
    raw_dir: str | Path,
    league: LeagueKey,
) -> dict[str, pd.DataFrame]:
    config = LEAGUES[league]
    frames = {}

    for dataset in DATASETS:
        filename = config.parquet_name_by_dataset[dataset]
        path = Path(raw_dir) / filename

        if not path.exists():
            logger.warning("Raw dataset not found: %s", path)
            continue

        frame = pd.read_parquet(path)
        frames[dataset] = (
            prepare_tracking(frame)
            if dataset == "start_positions"
            else normalize_columns(frame)
        )

        logger.info(
            "Loaded %d rows from %s",
            len(frame),
            path,
        )

    return frames

def prepare_raw_frames(
    raw_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        dataset: (
            prepare_tracking(frame)
            if dataset == "start_positions"
            else normalize_columns(frame)
        )
        for dataset, frame in raw_frames.items()
        if dataset in DATASETS
    }

def silver_path(
    silver_dir: str | Path,
    league: LeagueKey,
    season: str,
    season_type: str,
) -> Path:
    season_type_slug = (
        season_type.strip().lower().replace(" ", "_")
    )

    return (
        Path(silver_dir)
        / league
        / season
        / season_type_slug
        / "player_gamelogs.parquet"
    )

def _validate_required_frames(
    frames: dict[str, pd.DataFrame],
) -> None:
    for dataset in ("player_base", "team_base"):
        if dataset not in frames or frames[dataset].empty:
            raise ValueError(
                f"Missing or empty raw dataset: {dataset}"
            )