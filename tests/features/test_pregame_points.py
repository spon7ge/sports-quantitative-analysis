"""Pregame points features stay frozen at the quote as-of date."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.features.points.pregame import (
    appearances_before,
    build_pregame_points_features,
)
from src.models.xgboost_models.example_set import (
    filter_commence_after_as_of,
    load_player_points_quotes,
    map_quote_names,
    settle_priced_player_points,
    snapshot_as_of,
)


def _clock(minutes: float) -> str:
    whole = int(minutes)
    seconds = int(round((minutes - whole) * 60))
    return f"{whole}:{seconds:02d}"


def _row(
    player_id: int,
    game_date: str,
    minutes: float,
    *,
    pts: float,
    game_id: int,
    team_id: int = 100,
    opp_team_id: int = 200,
    player_name: str = "Cade Cunningham",
) -> dict:
    return {
        "player_id": player_id,
        "player_name": player_name,
        "team_id": team_id,
        "team_abbreviation": "DET",
        "opp_team_id": opp_team_id,
        "game_id": game_id,
        "game_date": game_date,
        "season_year": "2025-26",
        "season_type": "Regular Season",
        "matchup": "DET vs. CLE",
        "minutes": minutes,
        "min": minutes,
        "min_sec": _clock(minutes),
        "start_position": "G",
        "pts": pts,
        "fgm": 5,
        "fga": 12,
        "fg3_m": 2,
        "fg3_a": 4,
        "ftm": 2,
        "fta": 3,
        "usg_pct": 25.0,
        "tchs": 40,
        "pass": 20,
        "assists": 4,
        "reb": 5,
        "stl": 1,
        "blk": 1,
        "team_pace": 100.0,
        "team_net_rating": 1.0,
        "team_off_rating": 110.0,
        "team_def_rating": 108.0,
        "team_fga": 88,
        "opp_pace": 100.0,
        "opp_net_rating": -1.0,
        "opp_def_rating": 108.0,
        "player_team_spread": -3.0,
        "game_total": 220.0,
        "available_flag": 1,
        "comment": "",
        "wl": "W",
    }


class ExampleSetFilterTests(unittest.TestCase):
    def test_keeps_only_player_points(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "quotes.csv"
            pd.DataFrame(
                {
                    "BOOKMAKER": ["FanDuel", "FanDuel", "FanDuel"],
                    "CATEGORY": [
                        "player_points",
                        "player_rebounds",
                        "player_assists",
                    ],
                    "NAME": ["Cade Cunningham"] * 3,
                    "OVER/UNDER": ["Over", "Over", "Over"],
                    "LINE": [26.5, 6.5, 8.5],
                    "ODDS": [-110, -110, -110],
                    "COMMENCE_TIME": ["2026-05-09"] * 3,
                    "LAST_UPDATE": ["t"] * 3,
                    "DATA_PULLED_AT": ["2026-05-09 14:38:43"] * 3,
                }
            ).to_csv(path, index=False)
            quotes = load_player_points_quotes(path)
        self.assertEqual(len(quotes), 1)
        self.assertTrue((quotes["CATEGORY"] == "player_points").all())


class AsOfCutoffTests(unittest.TestCase):
    def test_drops_games_on_or_after_as_of(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "2026-05-08", 30, pts=20, game_id=1),
                _row(1, "2026-05-09", 40, pts=99, game_id=2),
            ]
        )
        prior = appearances_before(frame, "2026-05-09")
        self.assertEqual(len(prior), 1)
        self.assertEqual(int(prior.iloc[0]["game_id"]), 1)

    def test_candidate_features_ignore_same_day_box_scores(self) -> None:
        history = pd.DataFrame(
            [
                _row(1, "2026-04-01", 32, pts=18, game_id=1),
                _row(1, "2026-05-09", 40, pts=99, game_id=2),
            ]
        )
        candidate = pd.DataFrame(
            [
                {
                    **_row(1, "2026-05-09", 0, pts=0, game_id=99),
                    "minutes": np.nan,
                    "min": np.nan,
                    "min_sec": "",
                    "pts": np.nan,
                    "start_position": "",
                    "season_type": "Playoffs",
                }
            ]
        )
        featured = build_pregame_points_features(
            history,
            candidate,
            as_of="2026-05-09",
        )
        self.assertEqual(len(featured), 1)
        self.assertEqual(featured.iloc[0]["pts_lag_1"], 18)
        self.assertEqual(featured.iloc[0]["min_lag_1"], 32)
        self.assertNotEqual(featured.iloc[0]["pts_lag_1"], 99)

    def test_maps_quote_names_from_history(self) -> None:
        history = pd.DataFrame(
            [_row(1630595, "2026-04-01", 32, pts=18, game_id=1)]
        )
        mapping = map_quote_names(
            pd.Series(["Cade Cunningham", "Nobody"]),
            history,
        )
        self.assertEqual(mapping["Cade Cunningham"], 1630595)
        self.assertNotIn("Nobody", mapping)


class RegularSeasonContextTests(unittest.TestCase):
    def test_candidate_uses_scheduled_regular_season_game(self) -> None:
        history = pd.DataFrame(
            [_row(1, "2026-02-08", 32, pts=18, game_id=1, team_id=100)]
        )
        schedule = pd.DataFrame(
            [
                {
                    **_row(
                        1,
                        "2026-02-11",
                        40,
                        pts=99,
                        game_id=778,
                        team_id=100,
                        opp_team_id=200,
                    ),
                    "team_abbreviation": "NYK",
                    "matchup": "NYK @ PHI",
                    "season_type": "Regular Season",
                    "game_total": 225.0,
                    "player_team_spread": 4.5,
                }
            ]
        )
        quotes = pd.DataFrame(
            {
                "BOOKMAKER": ["FanDuel"],
                "CATEGORY": ["player_points"],
                "NAME": ["Cade Cunningham"],
                "OVER/UNDER": ["Over"],
                "LINE": [26.5],
                "ODDS": [-110],
                "COMMENCE_TIME": ["2026-02-11"],
                "LAST_UPDATE": ["t"],
                "DATA_PULLED_AT": ["2026-02-10 14:47:44"],
            }
        )
        from src.models.xgboost_models.example_set import (
            build_candidate_rows,
        )

        candidates = build_candidate_rows(
            quotes,
            history,
            schedule=pd.concat([history, schedule], ignore_index=True),
            as_of="2026-02-10",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0]["season_type"], "Regular Season")
        self.assertEqual(candidates.iloc[0]["matchup"], "NYK @ PHI")
        self.assertEqual(int(candidates.iloc[0]["game_id"]), 778)
        self.assertTrue(pd.isna(candidates.iloc[0]["pts"]))
        self.assertEqual(candidates.iloc[0]["game_total"], 225.0)


class SnapshotDisplayTests(unittest.TestCase):
    def test_as_of_is_the_pull_date(self) -> None:
        quotes = pd.DataFrame(
            {"DATA_PULLED_AT": ["2026-03-25 13:19:22", "2026-03-25 13:19:22"]}
        )
        self.assertEqual(
            snapshot_as_of(quotes),
            pd.Timestamp("2026-03-25"),
        )

    def test_drops_pull_date_and_keeps_later_commence(self) -> None:
        priced = pd.DataFrame(
            {
                "commence_time": ["2026-03-25", "2026-03-26", "2026-03-24"],
                "player_name": ["A", "B", "C"],
            }
        )
        shown = filter_commence_after_as_of(priced, "2026-03-25")
        self.assertEqual(shown["player_name"].tolist(), ["B"])

    def test_settle_voids_dnp_and_skips_pull_date(self) -> None:
        priced = pd.DataFrame(
            {
                "status": ["PRICED", "PRICED", "PRICED"],
                "selected_side": ["over", "over", "over"],
                "player_name": ["OnDate", "Later", "DNP"],
                "player_id": [1, 2, 3],
                "book": ["FanDuel", "FanDuel", "FanDuel"],
                "line": [10.5, 10.5, 8.5],
                "commence_time": ["2026-03-25", "2026-03-26", "2026-03-26"],
            }
        )
        quotes = pd.DataFrame(
            {
                "BOOKMAKER": ["FanDuel"] * 6,
                "NAME": ["OnDate", "OnDate", "Later", "Later", "DNP", "DNP"],
                "OVER/UNDER": ["Over", "Under"] * 3,
                "LINE": [10.5, 10.5, 10.5, 10.5, 8.5, 8.5],
                "ODDS": [-110, -110, -110, -110, -110, -110],
                "COMMENCE_TIME": [
                    "2026-03-25",
                    "2026-03-25",
                    "2026-03-26",
                    "2026-03-26",
                    "2026-03-26",
                    "2026-03-26",
                ],
                "DATA_PULLED_AT": ["2026-03-25 13:19:22"] * 6,
            }
        )
        panel = pd.DataFrame(
            [
                _row(1, "2026-03-24", 30, pts=12, game_id=1, player_name="OnDate"),
                _row(2, "2026-03-24", 28, pts=11, game_id=2, player_name="Later"),
                _row(3, "2026-03-24", 22, pts=9, game_id=3, player_name="DNP"),
                _row(1, "2026-03-25", 30, pts=20, game_id=10, player_name="OnDate"),
                _row(2, "2026-03-26", 28, pts=14, game_id=11, player_name="Later"),
                _row(3, "2026-03-26", 0, pts=0, game_id=12, player_name="DNP"),
            ]
        )
        graded = settle_priced_player_points(
            priced,
            quotes,
            panel,
            as_of="2026-03-25",
        )
        self.assertEqual(graded["player_name"].tolist(), ["Later", "DNP"])
        later = graded.loc[graded["player_name"].eq("Later")].iloc[0]
        self.assertEqual(later["grade"], "win")
        dnp = graded.loc[graded["player_name"].eq("DNP")].iloc[0]
        self.assertEqual(dnp["grade"], "void")

    def test_hides_pull_date_game_even_if_commence_is_next_utc_day(self) -> None:
        priced = pd.DataFrame(
            {
                "status": ["PRICED"],
                "selected_side": ["over"],
                "player_name": ["Kawhi Leonard"],
                "player_id": [202695],
                "book": ["FanDuel"],
                "line": [20.5],
                "commence_time": ["2026-03-26"],
            }
        )
        quotes = pd.DataFrame(
            {
                "BOOKMAKER": ["FanDuel", "FanDuel"],
                "NAME": ["Kawhi Leonard", "Kawhi Leonard"],
                "OVER/UNDER": ["Over", "Under"],
                "LINE": [20.5, 20.5],
                "ODDS": [-110, -110],
                "COMMENCE_TIME": ["2026-03-26", "2026-03-26"],
                "DATA_PULLED_AT": ["2026-03-25 13:19:22"] * 2,
            }
        )
        history = pd.DataFrame(
            [
                _row(
                    202695,
                    "2026-03-24",
                    32,
                    pts=18,
                    game_id=1,
                    team_id=100,
                    player_name="Kawhi Leonard",
                )
            ]
        )
        same_day = pd.DataFrame(
            [
                {
                    **_row(
                        202695,
                        "2026-03-25",
                        30,
                        pts=22,
                        game_id=25,
                        team_id=100,
                        player_name="Kawhi Leonard",
                    ),
                }
            ]
        )
        panel = pd.concat([history, same_day], ignore_index=True)
        graded = settle_priced_player_points(
            priced,
            quotes,
            panel,
            as_of="2026-03-25",
        )
        self.assertTrue(graded.empty)
