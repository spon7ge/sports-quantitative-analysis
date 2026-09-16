"""Rotowire tipoff dates must parse without format inference."""

from __future__ import annotations

import unittest
import warnings

import pandas as pd

from src.pipeline.silver.rotowire import (
    _parse_tipoff_dates,
    enrich_rotowire,
)


class ParseTipoffDatesTests(unittest.TestCase):
    def test_october_stays_in_season_start_year(self) -> None:
        frame = pd.DataFrame(
            {
                "Tipoff": ["Oct 19 7:30 PM"],
                "Season": [2021],
            }
        )

        parsed = _parse_tipoff_dates(frame, 2021)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2021-10-19"))

    def test_january_rolls_into_the_next_calendar_year(self) -> None:
        frame = pd.DataFrame(
            {
                "Tipoff": ["Jan 28 8:00 PM"],
                "Season": [2021],
            }
        )

        parsed = _parse_tipoff_dates(frame, 2021)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2022-01-28"))

    def test_february_29_uses_the_leap_year_in_the_season(self) -> None:
        frame = pd.DataFrame(
            {
                "Tipoff": ["Feb 29 7:00 PM"],
                "Season": [2023],
            }
        )

        parsed = _parse_tipoff_dates(frame, 2023)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2024-02-29"))

    def test_does_not_warn_about_inferred_datetime_formats(self) -> None:
        frame = pd.DataFrame(
            {
                "Tipoff": ["Nov 3 7:00 PM", "Mar 18 7:30 PM"],
                "Season": [2021, 2021],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parsed = _parse_tipoff_dates(frame, 2021)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2021-11-03"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2022-03-18"))


class EnrichRotowireTests(unittest.TestCase):
    def test_signs_home_line_for_the_player_team(self) -> None:
        games = pd.DataFrame(
            {
                "game_date": ["2021-10-20", "2021-10-20"],
                "matchup": ["PHX @ CHA", "CHA vs. PHX"],
                "team_abbreviation": ["PHX", "CHA"],
            }
        )
        lines = pd.DataFrame(
            {
                "rw_date": [pd.Timestamp("2021-10-20")],
                "rw_away": ["PHX"],
                "rw_home": ["CHO"],
                "game_total": [223.0],
                "team_spread": [1.5],
            }
        )

        result = enrich_rotowire(games, lines)

        self.assertEqual(result.loc[0, "game_total"], 223.0)
        self.assertEqual(result.loc[0, "player_team_spread"], -1.5)
        self.assertEqual(result.loc[1, "player_team_spread"], 1.5)

    def test_keeps_lines_missing_when_rotowire_has_no_game(self) -> None:
        games = pd.DataFrame(
            {
                "game_date": ["2021-12-31"],
                "matchup": ["LAL vs. POR"],
                "team_abbreviation": ["LAL"],
            }
        )
        lines = pd.DataFrame(
            {
                "rw_date": [pd.Timestamp("2021-12-30")],
                "rw_away": ["PHI"],
                "rw_home": ["BKN"],
                "game_total": [221.5],
                "team_spread": [-3.5],
            }
        )

        result = enrich_rotowire(games, lines)

        self.assertTrue(pd.isna(result.loc[0, "game_total"]))
        self.assertTrue(pd.isna(result.loc[0, "player_team_spread"]))
