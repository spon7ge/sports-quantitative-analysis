"""NBA season labels are YYYY-YY, e.g. 2025-26."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import pandas as pd

from src.pipeline.silver.season import assign_season, load_player_gamelogs


class AssignSeasonTests(unittest.TestCase):
    def test_copies_season_year_in_nba_format(self) -> None:
        frame = pd.DataFrame(
            {
                "season_year": ["2025-26", "2025-26"],
                "game_date": ["2025-10-22", "2026-01-15"],
            }
        )

        result = assign_season(frame)

        self.assertListEqual(result["season"].tolist(), ["2025-26", "2025-26"])

    def test_derives_season_from_game_date_when_season_year_is_absent(self) -> None:
        frame = pd.DataFrame(
            {
                "game_date": ["2025-10-22", "2026-01-15", "2026-04-10"],
            }
        )

        result = assign_season(frame)

        self.assertListEqual(
            result["season"].tolist(),
            ["2025-26", "2025-26", "2025-26"],
        )

    def test_october_opens_a_new_season_and_july_stays_in_the_prior_season(self) -> None:
        frame = pd.DataFrame(
            {
                "game_date": ["2025-07-01", "2025-10-01"],
            }
        )

        result = assign_season(frame)

        self.assertListEqual(
            result["season"].tolist(),
            ["2024-25", "2025-26"],
        )


class LoadPlayerGamelogsTests(unittest.TestCase):
    def test_loads_parquet_and_adds_season(self) -> None:
        frame = pd.DataFrame(
            {
                "player_id": [1],
                "game_id": ["0022500001"],
                "game_date": ["2025-10-22"],
                "minutes": [32.0],
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "player_gamelogs.parquet"
            frame.to_parquet(path, index=False)

            loaded = load_player_gamelogs((path,))

        self.assertEqual(loaded.loc[0, "season"], "2025-26")
        self.assertEqual(len(loaded), 1)
