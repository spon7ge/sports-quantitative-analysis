"""Play-by-play fetch reads silver game IDs and resumes safely."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from src.pipeline.bronze.play_by_play import (
    fetch_season_play_by_play,
    play_by_play_path,
    silver_game_ids,
)


def _gamelog(folder: Path, season_type: str, game_ids: list[object]) -> None:
    destination = folder / season_type / "player_gamelogs.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "game_id": game_ids,
            "player_id": list(range(len(game_ids))),
        }
    ).to_parquet(destination, index=False)


def _actions(game_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gameId": [game_id, game_id],
            "actionNumber": [1, 2],
            "description": ["start", "jump ball"],
        }
    )


class SilverGameIdTests(unittest.TestCase):
    def test_reads_unique_ids_from_every_season_type(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _gamelog(
                root / "nba" / "2026-27",
                "regular_season",
                ["22600184", "0022600179", "0022600179"],
            )
            _gamelog(
                root / "nba" / "2026-27",
                "playoffs",
                ["0022600179", "0042600101"],
            )

            ids = silver_game_ids(root, "2026-27")

        self.assertEqual(
            ids,
            ["0022600179", "0022600184", "0042600101"],
        )

    def test_missing_season_raises(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                silver_game_ids(Path(folder), "2026-27")

    def test_rejects_a_season_label_that_is_not_consecutive(self) -> None:
        with self.assertRaises(ValueError):
            play_by_play_path("data/PlayByPlay", "2026-28")


class FetchSeasonPlayByPlayTests(unittest.TestCase):
    def test_writes_v3_rows_and_skips_games_already_saved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _gamelog(
                root / "silver" / "nba" / "2026-27",
                "regular_season",
                ["0022600001", "0022600002"],
            )
            output_dir = root / "PlayByPlay"
            output_dir.mkdir()
            _actions("0022600001").to_parquet(
                output_dir / "s27_pbp.parquet",
                index=False,
            )
            requested: list[str] = []

            def request(game_id: str) -> pd.DataFrame:
                requested.append(game_id)
                return _actions(game_id)

            result = fetch_season_play_by_play(
                "2026-27",
                silver_dir=root / "silver",
                output_dir=output_dir,
                delay_seconds=0,
                request=request,
            )
            saved = pd.read_parquet(result.path)

        self.assertEqual(requested, ["0022600002"])
        self.assertEqual(result.skipped, ("0022600001",))
        self.assertEqual(result.failed, ())
        self.assertEqual(
            saved["gameId"].tolist(),
            ["0022600001", "0022600001", "0022600002", "0022600002"],
        )
        self.assertEqual(result.path.name, "s27_pbp.parquet")

    def test_empty_or_failed_games_are_not_marked_complete(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _gamelog(
                root / "silver" / "nba" / "2026-27",
                "regular_season",
                ["0022600001", "0022600002", "0022600003"],
            )

            def request(game_id: str) -> pd.DataFrame:
                if game_id == "0022600002":
                    raise RuntimeError("stats timeout simulated")
                if game_id == "0022600003":
                    return _actions(game_id).iloc[0:0]
                return _actions(game_id)

            result = fetch_season_play_by_play(
                "2026-27",
                silver_dir=root / "silver",
                output_dir=root / "PlayByPlay",
                delay_seconds=0,
                request=request,
            )
            saved = pd.read_parquet(result.path)

        self.assertEqual(result.fetched, ("0022600001",))
        self.assertEqual(result.failed, ("0022600002", "0022600003"))
        self.assertEqual(set(saved["gameId"]), {"0022600001"})
