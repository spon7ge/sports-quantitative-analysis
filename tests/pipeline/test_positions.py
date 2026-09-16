"""Position enrichment should match nba_api names to Basketball Reference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.silver.positions import (
    canonical_position,
    enrich_positions,
    load_player_positions,
    normalize_player_name,
    FINE_POSITIONS,
)


class NormalizePlayerNameTests(unittest.TestCase):
    def test_strips_periods_from_initials(self) -> None:
        self.assertEqual(
            normalize_player_name("B.J. Johnson"),
            normalize_player_name("BJ Johnson"),
        )

    def test_strips_accents(self) -> None:
        self.assertEqual(normalize_player_name("Luka Dončić"), "luka doncic")

    def test_does_not_treat_jrue_as_a_suffix(self) -> None:
        self.assertEqual(normalize_player_name("Jrue Holiday"), "jrue holiday")


class LoadAndEnrichPositionTests(unittest.TestCase):
    def test_matches_generational_suffix_when_only_one_side_has_it(self) -> None:
        folder = Path(self._csv_dir())
        Path(folder / "nba_2022_players.csv").write_text(
            "name,pos,age\n"
            "Bobby Portis,C,28\n"
            "Xavier Tillman Sr.,PF,23\n"
            "Jimmy Butler,SF,32\n",
            encoding="utf-8",
        )
        lookup = load_player_positions(
            "2021-22",
            league="nba",
            positions_dir=folder,
        )
        games = pd.DataFrame(
            {
                "player_name": [
                    "Bobby Portis Jr.",
                    "Xavier Tillman",
                    "Jimmy Butler III",
                ]
            }
        )

        result = enrich_positions(games, lookup, league="nba")

        self.assertListEqual(result["pos"].tolist(), ["C", "PF", "SF"])

    def test_keeps_jr_and_sr_distinct_when_both_exist(self) -> None:
        folder = Path(self._csv_dir())
        Path(folder / "nba_2022_players.csv").write_text(
            "name,pos,age\n"
            "Marcus Morris Sr.,PF,32\n"
            "Marcus Morris Jr.,SF,22\n",
            encoding="utf-8",
        )
        lookup = load_player_positions(
            "2021-22",
            league="nba",
            positions_dir=folder,
        )
        games = pd.DataFrame(
            {
                "player_name": [
                    "Marcus Morris Sr.",
                    "Marcus Morris Jr.",
                ]
            }
        )

        result = enrich_positions(games, lookup, league="nba")

        self.assertListEqual(result["pos"].tolist(), ["PF", "SF"])

    def test_skips_league_average_rows(self) -> None:
        folder = Path(self._csv_dir())
        Path(folder / "nba_2022_players.csv").write_text(
            "name,pos,age\n"
            "Joel Embiid,C,27\n"
            "League Average,,\n",
            encoding="utf-8",
        )
        lookup = load_player_positions(
            "2021-22",
            league="nba",
            positions_dir=folder,
        )

        self.assertEqual(len(lookup), 1)
        self.assertEqual(lookup.iloc[0]["name_key"], "joel embiid")

    def test_tracking_fg_codes_do_not_replace_missing_fine_positions(self) -> None:
        games = pd.DataFrame(
            {
                "player_name": ["Trevon Scott"],
                "start_position": ["F"],
            }
        )

        result = enrich_positions(games, pd.DataFrame(), league="nba")

        self.assertTrue(pd.isna(result.loc[0, "pos"]))
        self.assertIsNone(canonical_position("F", FINE_POSITIONS))
        self.assertIsNone(canonical_position("G", FINE_POSITIONS))

    def _csv_dir(self) -> str:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return folder.name
