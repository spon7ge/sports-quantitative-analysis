"""Lineup features use the starters on that game date and their prior stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from game_enviroment import build_features
from lineup import (
    LINEUP_GAME_FEATURES,
    build_lineup_features,
    fill_lineup_gaps,
    lineup_game_features,
    merge_lineup_features,
)


def _logs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _appearance(**overrides) -> dict:
    row = {
        "season_year": "2023-24",
        "season_type": "Regular Season",
        "game_id": "g1",
        "game_date": "2024-01-01",
        "team_id": 10,
        "team_abbreviation": "HOM",
        "matchup": "HOM vs. AWY",
        "player_id": 1,
        "start_position": "G",
        "min": 30.0,
        "pace": 100.0,
        "off_rating": 110.0,
        "def_rating": 110.0,
    }
    row.update(overrides)
    return row


def _fit(frame: pd.DataFrame, **kwargs):
    params = {
        "halflife": 12.0,
        "prior_weight": 0.0,
        "season_carry": 0.6,
        "window": 300,
    }
    params.update(kwargs)
    return build_lineup_features(frame, **params)


def _cell(table: pd.DataFrame, game_id: str, team_id: int, column: str) -> float:
    row = table.loc[table["game_id"].eq(game_id) & table["team_id"].eq(team_id)]
    assert len(row) == 1
    return float(row[column].iloc[0])


def test_lineup_is_that_games_starters_and_uses_prior_stats_only():
    # Game 2 starts a different five than game 1. Same-night ratings are 999.
    # PG is not a start_position code, so that player stays out of the five.
    logs = _logs(
        [
            _appearance(player_id=1, start_position="G", pace=110, off_rating=120, def_rating=100),
            _appearance(player_id=2, start_position="F", pace=90, off_rating=100, def_rating=110),
            _appearance(player_id=3, start_position="", pace=40, off_rating=40, def_rating=140),
            _appearance(player_id=7, start_position="PG", pace=1, off_rating=1, def_rating=1),
            _appearance(
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=4,
                start_position="C",
                pace=98,
                off_rating=105,
                def_rating=108,
            ),
            _appearance(
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=5,
                start_position=" G ",
                pace=102,
                off_rating=115,
                def_rating=112,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                player_id=1,
                start_position="G",
                pace=999,
                off_rating=999,
                def_rating=999,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                player_id=3,
                start_position="C",
                pace=999,
                off_rating=999,
                def_rating=999,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                player_id=2,
                start_position="",
                pace=1,
                off_rating=1,
                def_rating=1,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                player_id=7,
                start_position="PG",
                pace=1,
                off_rating=1,
                def_rating=1,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=4,
                start_position="C",
                pace=999,
                off_rating=999,
                def_rating=1,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=5,
                start_position="G",
                pace=999,
                off_rating=999,
                def_rating=1,
            ),
        ]
    )

    table, _state = _fit(logs)

    assert np.isnan(_cell(table, "g1", 10, "lineup_pace"))
    assert _cell(table, "g2", 10, "lineup_pace") == pytest.approx(75.0)
    assert _cell(table, "g2", 10, "lineup_off_rating") == pytest.approx(80.0)
    assert _cell(table, "g2", 10, "lineup_def_rating") == pytest.approx(120.0)
    assert _cell(table, "g2", 20, "lineup_pace") == pytest.approx(100.0)
    assert _cell(table, "g2", 20, "lineup_off_rating") == pytest.approx(110.0)
    assert _cell(table, "g2", 20, "lineup_def_rating") == pytest.approx(110.0)


def test_a_starter_with_no_history_takes_the_league_prior():
    logs = _logs(
        [
            _appearance(player_id=1, pace=110, off_rating=120, def_rating=100),
            _appearance(
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=4,
                pace=90,
                off_rating=100,
                def_rating=120,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                player_id=1,
                pace=999,
                off_rating=999,
                def_rating=999,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                player_id=6,
                start_position="F",
                pace=999,
                off_rating=999,
                def_rating=999,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=4,
                pace=999,
                off_rating=999,
                def_rating=999,
            ),
        ]
    )

    table, _state = _fit(logs)

    assert _cell(table, "g2", 10, "lineup_pace") == pytest.approx(105.0)
    assert _cell(table, "g2", 10, "lineup_off_rating") == pytest.approx(115.0)
    assert _cell(table, "g2", 10, "lineup_def_rating") == pytest.approx(105.0)


def test_playoff_games_stay_out_of_regular_season_priors():
    logs = _logs(
        [
            _appearance(player_id=1, pace=110, off_rating=120, def_rating=100),
            _appearance(
                game_id="po",
                game_date="2024-01-02",
                season_type="Playoffs",
                player_id=1,
                pace=999,
                off_rating=999,
                def_rating=999,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                player_id=1,
                pace=50,
                off_rating=50,
                def_rating=50,
            ),
        ]
    )

    table, _state = _fit(logs, include_playoffs=False)

    assert _cell(table, "g2", 10, "lineup_pace") == pytest.approx(110.0)
    assert _cell(table, "g2", 10, "lineup_off_rating") == pytest.approx(120.0)
    assert _cell(table, "g2", 10, "lineup_def_rating") == pytest.approx(100.0)


def test_home_and_away_lineups_land_on_one_game_row():
    logs = _logs(
        [
            _appearance(player_id=1, pace=110, off_rating=120, def_rating=100),
            _appearance(
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=4,
                pace=90,
                off_rating=100,
                def_rating=120,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                player_id=1,
                pace=999,
                off_rating=999,
                def_rating=999,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=4,
                pace=999,
                off_rating=999,
                def_rating=999,
            ),
        ]
    )
    table, _state = _fit(logs)
    games = pd.DataFrame(
        {
            "game_id": ["g2"],
            "home_id": [10],
            "away_id": [20],
        }
    )

    merged = merge_lineup_features(games, table)

    assert list(LINEUP_GAME_FEATURES) == [
        "lineup_pace_h",
        "lineup_pace_a",
        "lineup_off_rating_h",
        "lineup_off_rating_a",
        "lineup_def_rating_h",
        "lineup_def_rating_a",
    ]
    assert merged.loc[0, "lineup_pace_h"] == pytest.approx(110.0)
    assert merged.loc[0, "lineup_pace_a"] == pytest.approx(90.0)
    assert merged.loc[0, "lineup_off_rating_h"] == pytest.approx(120.0)
    assert merged.loc[0, "lineup_def_rating_a"] == pytest.approx(120.0)

    # State frozen after game 1 is what game 2's lineup feature is allowed to see.
    _earlier, state = _fit(logs.loc[logs["game_id"].eq("g1")].copy())
    served = lineup_game_features(state, "2023-24", home_ids=[1], away_ids=[4])
    assert served["lineup_pace_h"] == pytest.approx(merged.loc[0, "lineup_pace_h"])
    assert served["lineup_pace_a"] == pytest.approx(merged.loc[0, "lineup_pace_a"])
    assert served["lineup_off_rating_h"] == pytest.approx(120.0)
    assert served["lineup_def_rating_a"] == pytest.approx(120.0)


def test_build_features_keeps_lineup_columns():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "game_date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "season_year": ["2023-24", "2023-24"],
            "home_id": [10, 10],
            "away_id": [20, 20],
            "home_abbr": ["HOM", "HOM"],
            "away_abbr": ["AWY", "AWY"],
            "poss48": [100.0, 102.0],
            "pts_total": [220.0, 230.0],
            "total": [225.0, 228.0],
            "home_spread": [-3.0, -2.5],
            "lineup_pace_h": [98.0, 99.0],
            "lineup_pace_a": [101.0, 97.0],
            "lineup_off_rating_h": [112.0, 114.0],
            "lineup_off_rating_a": [108.0, 110.0],
            "lineup_def_rating_h": [109.0, 107.0],
            "lineup_def_rating_a": [111.0, 113.0],
        }
    )

    built, _state = build_features(
        games,
        halflife=12,
        prior_w=4,
        carry=0.6,
        window=300,
        min_league=1,
    )

    for column in LINEUP_GAME_FEATURES:
        assert built[column].tolist() == games[column].tolist()


def test_a_missing_lineup_is_filled_from_earlier_games_only():
    games = pd.DataFrame(
        {
            "game_date": ["2024-01-03", "2024-01-01", "2024-01-05"],
            "lineup_pace_h": [np.nan, 100.0, 120.0],
            "lineup_pace_a": [90.0, 110.0, np.nan],
        }
    )

    filled = fill_lineup_gaps(games)

    assert filled["lineup_pace_h"].tolist() == pytest.approx([100.0, 100.0, 120.0])
    assert filled["lineup_pace_a"].tolist() == pytest.approx([90.0, 110.0, 100.0])
