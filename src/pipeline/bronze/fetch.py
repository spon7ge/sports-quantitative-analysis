"""Fetch NBA or WNBA game logs and save one Parquet file per dataset."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelogs, teamgamelogs

from .config import (
    LEAGUES,
    LEAGUE_WIDE_DATASETS,
    RAW_DATASETS,
    LeagueConfig,
    LeagueKey,
    checkpoint_path,
)
from .dataframes import (
    normalize_columns,
    normalize_datasets,
    rows_for_games,
)
from .retry import call_with_retry
from .storage import ParquetStore
from .tracking import fetch_game_tracking

logger = logging.getLogger(__name__)

LeagueFetcher = Callable[[str], pd.DataFrame]

class GameLogs:
    """Fetch game-log datasets for one league and season."""

    def __init__(
        self,
        season: str,
        season_type: str = "Regular Season",
        *,
        league: LeagueKey = "nba",
        output_dir: str | Path = "data/bronze",
    ) -> None:
        self.season = season
        self.season_type = season_type
        self._config: LeagueConfig = LEAGUES[league]
        self._store = ParquetStore(output_dir, self._config)
        self._data: dict[str, pd.DataFrame] = {}

    @property
    def league(self) -> LeagueKey:
        return self._config.key

    @property
    def data(self) -> dict[str, pd.DataFrame]:
        return dict(self._data)

    def parquet_path(self, dataset: str) -> Path:
        return self._store.path_for(dataset)

    def get(self, dataset: str) -> pd.DataFrame:
        if dataset not in self._data:
            available = sorted(self._data) or "(none)"
            raise KeyError(
                f"Dataset {dataset!r} not fetched. "
                f"Available: {available}"
            )

        return self._data[dataset]

    def fetch(
        self,
        datasets: str | Iterable[str] | None = None,
        *,
        parallel: bool = True,
        start_position_delay: float = 2.5,
        batch_size: int = 100,
        checkpoint: str | Path | None = None,
        start_position_workers: int = 5,
        run_all_batches: bool = True,
        game_ids: Iterable[object] | None = None,
    ) -> GameLogs:
        """Fetch requested datasets and write them to Parquet files."""
        requested = normalize_datasets(datasets)
        checkpoint = checkpoint or checkpoint_path(
            self._config,
            self._store.output_dir,
        )

        logger.info(
            "Fetching %s %s for %s %s",
            self._config.label,
            ", ".join(requested),
            self.season,
            self.season_type,
        )

        league_wide = [
            name
            for name in requested
            if name in LEAGUE_WIDE_DATASETS
        ]

        if league_wide:
            self._fetch_league_wide(
                league_wide,
                parallel=parallel,
            )

        if "start_positions" in requested:
            self._fetch_start_positions(
                game_ids=self._resolve_game_ids(game_ids),
                delay=start_position_delay,
                batch_size=batch_size,
                checkpoint=checkpoint,
                workers=start_position_workers,
                run_all_batches=run_all_batches,
            )

        return self

    def _player_logs(self, measure: str) -> pd.DataFrame:
        arguments = self._endpoint_arguments(measure)

        return playergamelogs.PlayerGameLogs(
            **arguments
        ).get_data_frames()[0]

    def _team_logs(self, measure: str) -> pd.DataFrame:
        arguments = self._endpoint_arguments(measure)

        return teamgamelogs.TeamGameLogs(
            **arguments
        ).get_data_frames()[0]

    def _endpoint_arguments(
        self,
        measure: str,
    ) -> dict[str, str]:
        arguments = {
            "season_nullable": self.season,
            "season_type_nullable": self.season_type,
            "measure_type_player_game_logs_nullable": measure,
        }

        if self._config.league_id is not None:
            arguments["league_id_nullable"] = self._config.league_id

        return arguments

    def _fetch_league_wide(
        self,
        datasets: list[str],
        *,
        parallel: bool,
    ) -> None:
        specifications: dict[
            str,
            tuple[LeagueFetcher, str],
        ] = {
            "player_base": (self._player_logs, "Base"),
            "player_adv": (self._player_logs, "Advanced"),
            "team_base": (self._team_logs, "Base"),
            "team_adv": (self._team_logs, "Advanced"),
        }

        def fetch_one(
            name: str,
        ) -> tuple[str, pd.DataFrame]:
            fetcher, measure = specifications[name]
            frame = call_with_retry(
                fetcher,
                measure,
                label=name,
            )
            return name, normalize_columns(frame)

        if parallel and len(datasets) > 1:
            with ThreadPoolExecutor(
                max_workers=len(datasets)
            ) as executor:
                futures = [
                    executor.submit(fetch_one, name)
                    for name in datasets
                ]

                for future in as_completed(futures):
                    name, frame = future.result()
                    self._save(name, frame)

            return

        for name in datasets:
            _, frame = fetch_one(name)
            self._save(name, frame)

    def _save(
        self,
        dataset: str,
        frame: pd.DataFrame,
    ) -> None:
        self._data[dataset] = frame
        path = self._store.write(dataset, frame)

        logger.info(
            "Saved %s rows for %s to %s",
            f"{len(frame):,}",
            dataset,
            path,
        )

    def _resolve_game_ids(
        self,
        supplied_game_ids: Iterable[object] | None,
    ) -> list[object]:
        if supplied_game_ids is not None:
            return list(supplied_game_ids)

        if "player_base" in self._data:
            return (
                self._data["player_base"]["game_id"]
                .dropna()
                .unique()
                .tolist()
            )

        raise ValueError(
            "start_positions needs game IDs. Fetch player_base "
            "in the same call, fetch player_base first, or pass "
            "game_ids."
        )

    def _fetch_start_positions(
        self,
        *,
        game_ids: list[object],
        delay: float,
        batch_size: int,
        checkpoint: str | Path,
        workers: int,
        run_all_batches: bool,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if workers < 1:
            raise ValueError(
                "start_position_workers must be at least 1"
            )

        requested_ids = {
            str(game_id).zfill(10)
            for game_id in game_ids
        }

        saved_frames, completed_ids = self._load_checkpoint(
            checkpoint,
            requested_ids,
        )

        remaining = [
            game_id
            for game_id in game_ids
            if str(game_id).zfill(10) not in completed_ids
        ]

        failed: list[object] = []
        batches = [
            remaining[index : index + batch_size]
            for index in range(0, len(remaining), batch_size)
        ]

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):
            logger.info(
                "Fetching tracking batch %d/%d",
                batch_number,
                len(batches),
            )

            batch_frames = self._fetch_tracking_batch(
                batch,
                delay,
                workers,
                failed,
            )

            if batch_frames:
                saved_frames.extend(batch_frames)
                ParquetStore.write_checkpoint(
                    checkpoint,
                    pd.concat(saved_frames, ignore_index=True),
                )

            if not run_all_batches:
                break

        if failed:
            logger.warning(
                "Tracking failed for %d games: %s",
                len(failed),
                failed,
            )

        if saved_frames:
            combined = pd.concat(
                saved_frames,
                ignore_index=True,
            )
        else:
            combined = pd.DataFrame()

        self._save(
            "start_positions",
            rows_for_games(combined, requested_ids),
        )

    @staticmethod
    def _load_checkpoint(
        checkpoint: str | Path,
        requested_ids: set[str],
    ) -> tuple[list[pd.DataFrame], set[str]]:
        frame = ParquetStore.read_checkpoint(checkpoint)

        if frame is None:
            return [], set()

        frame = rows_for_games(frame, requested_ids)

        if "game_id" not in frame:
            raise ValueError(
                f"Checkpoint {checkpoint} has no game ID column"
            )

        completed = set(
            frame["game_id"].dropna().unique()
        )

        logger.info(
            "Loaded checkpoint with %d completed games",
            len(completed),
        )

        return [frame], completed

    @staticmethod
    def _fetch_tracking_batch(
        game_ids: list[object],
        delay: float,
        workers: int,
        failed: list[object],
    ) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            futures = {
                executor.submit(
                    fetch_game_tracking,
                    game_id,
                    delay,
                ): game_id
                for game_id in game_ids
            }

            for future in as_completed(futures):
                game_id = futures[future]

                try:
                    frame = future.result()
                except Exception as error:
                    logger.warning(
                        "Tracking failed for game %s: %s",
                        game_id,
                        error,
                    )
                    failed.append(game_id)
                    continue

                if frame is None:
                    failed.append(game_id)
                else:
                    frames.append(frame)

        return frames

class NBAGameLogs(GameLogs):
    def __init__(
        self,
        season: str,
        season_type: str = "Regular Season",
        *,
        output_dir: str | Path = "data/bronze",
    ) -> None:
        super().__init__(
            season,
            season_type,
            league="nba",
            output_dir=output_dir,
        )

class WNBAGameLogs(GameLogs):
    def __init__(
        self,
        season: str,
        season_type: str = "Regular Season",
        *,
        output_dir: str | Path = "data/bronze",
    ) -> None:
        super().__init__(
            season,
            season_type,
            league="wnba",
            output_dir=output_dir,
        )

def parse_cli_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch NBA/WNBA game logs into Parquet files"
        )
    )

    parser.add_argument(
        "--league",
        choices=tuple(LEAGUES),
        default="nba",
    )
    parser.add_argument("--season")
    parser.add_argument(
        "--season-type",
        default="Regular Season",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=RAW_DATASETS,
    )
    parser.add_argument(
        "--output-dir",
        default="data/bronze",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--start-position-delay",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--start-position-workers",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
    )

    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_cli_args(argv)
    season = (
        args.season
        or LEAGUES[args.league].default_season
    )

    logs = GameLogs(
        season,
        season_type=args.season_type,
        league=args.league,
        output_dir=args.output_dir,
    )

    logs.fetch(
        datasets=args.datasets,
        parallel=not args.sequential,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        start_position_delay=args.start_position_delay,
        start_position_workers=args.start_position_workers,
        run_all_batches=not args.one_batch,
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())