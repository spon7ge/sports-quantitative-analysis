"""Layer 1 draws one game from the player's team point of view."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from game_enviroment import (
    fit_player_context,
    load_latest_slate,
    load_live_slate,
    simulate_game,
)


def _team(abbr: str, pace_num: float) -> dict:
    return {
        "abbr": abbr,
        "num": pace_num,
        "den": 0.0,
        "last_date": "2026-04-10",
        "last_season": "2026-27",
    }


def _model(**overrides) -> dict:
    model = {
        "config": {"season_carry": 1.0, "prior_weight": 4.0},
        "state": {
            "league_poss": 100.0,
            "league_pts": 220.0,
            "teams": {"1": _team("LAL", 40.0), "2": _team("BOS", -40.0)},
        },
        "sides": {
            "features": ["own_pace", "opp_pace", "is_home"],
            "beta": [0.0, 1.0, 0.0, 0.0],
            "residual": {"nu": 30.0, "scale": 0.0, "rho": 1.0},
        },
        "scoring": {
            "league_ppp": 1.1,
            "off_center": 110.0,
            "def_center": 110.0,
            "beta": [0.0, 0.0, 0.0, 0.0],
            "residual": {"nu": 30.0, "scale": 0.0},
            "market_weight": 1.0,
        },
        "ot_given_margin": {
            "edges": [0.0, 3.0, 6.0, 10.0, 1e9],
            "p_ot": [1.0, 0.0, 0.0, 0.0],
            "p_more_given_ot": 0.0,
        },
    }
    model.update(overrides)
    return model


def _draw(**kwargs):
    args = dict(
        model=_model(),
        team="LAL",
        opponent="BOS",
        date="2026-11-05",
        is_home=True,
        total=225.0,
        spread=-6.0,
        n=20,
        seed=1,
    )
    args.update(kwargs)
    return simulate_game(**args)


def test_draw_is_oriented_to_the_players_team():
    home = _draw()
    away = _draw(team="BOS", opponent="LAL", is_home=False, spread=6.0)

    assert list(home.columns) == [
        "team_poss",
        "opp_poss",
        "pace",
        "team_pts",
        "opp_pts",
        "game_total",
        "is_home",
        "n_ot",
        "game_minutes",
        "team_margin",
    ]
    # Posted 225 and Lakers -6. Market weight is 1, so every draw is that price.
    assert np.allclose(home["team_pts"], 115.5)
    assert np.allclose(home["opp_pts"], 109.5)
    assert np.allclose(home["game_total"], home["team_pts"] + home["opp_pts"])
    assert np.allclose(home["team_margin"], home["team_pts"] - home["opp_pts"])
    assert np.all(home["is_home"] == 1.0)
    # Same game from Boston's side: Boston is the dog, so its points are 109.5.
    assert np.allclose(away["team_pts"], 109.5)
    assert np.allclose(away["opp_pts"], 115.5)
    assert np.all(away["is_home"] == 0.0)


def test_each_side_keeps_its_own_pace_and_one_shared_shock():
    noisy = _model()
    noisy["sides"]["residual"] = {"nu": 30.0, "scale": 3.0, "rho": 1.0}
    noisy["ot_given_margin"]["p_ot"] = [0.0, 0.0, 0.0, 0.0]
    drawn = _draw(model=noisy, n=300, seed=2, spread=-15.0)

    # Lakers pace feature is 40/4 = 10, Boston is -10, coefficient is 1.
    assert abs(float(drawn["team_poss"].mean()) - 110.0) < 0.5
    assert abs(float(drawn["opp_poss"].mean()) - 90.0) < 0.5
    assert np.allclose(drawn["team_poss"] - 110.0, drawn["opp_poss"] - 90.0)
    assert np.allclose(drawn["pace"], (drawn["team_poss"] + drawn["opp_poss"]) / 2.0)


def test_overtime_follows_the_drawn_margin_and_lengthens_possessions():
    close = _draw(spread=0.0, total=220.0)
    blowout = _draw(spread=-20.0, total=220.0)

    assert np.all(close["n_ot"] == 1)
    assert np.all(close["game_minutes"] == 53.0)
    assert np.allclose(close["team_poss"], 110.0 * 53.0 / 48.0)
    assert np.allclose(close["opp_poss"], 90.0 * 53.0 / 48.0)
    assert np.all(blowout["n_ot"] == 0)
    assert np.allclose(blowout["game_minutes"], 48.0)
    assert np.allclose(blowout["team_poss"], 110.0)


def test_points_follow_possessions_when_no_market_is_posted():
    model = _model()
    model["scoring"]["market_weight"] = 0.0
    model["ot_given_margin"]["p_ot"] = [0.0, 0.0, 0.0, 0.0]
    drawn = _draw(model=model, total=None, spread=None)

    assert np.allclose(drawn["team_pts"], 110.0 * 1.1)
    assert np.allclose(drawn["opp_pts"], 90.0 * 1.1)
    assert np.allclose(drawn["team_margin"], drawn["team_pts"] - drawn["opp_pts"])


def test_fit_learns_team_pace_scoring_and_overtime_from_history():
    rng = np.random.default_rng(0)
    n = 80
    pace_h = rng.normal(0.0, 1.0, n)
    pace_a = rng.normal(0.0, 1.0, n)
    home_poss = 100.0 + 3.0 * pace_h
    away_poss = 100.0 + 3.0 * pace_a
    games = pd.DataFrame(
        {
            "lg_poss": 100.0,
            "nominal_min": 48.0,
            "pace_h": pace_h,
            "pace_a": pace_a,
            "b2b_h": 0.0,
            "b2b_a": 0.0,
            "rested_h": 0.0,
            "rested_a": 0.0,
            "total_dev": 0.0,
            "abs_spread": 0.0,
            "total_missing": 0.0,
            "spread_missing": 1.0,
            "lineup_pace_h": 100.0,
            "lineup_pace_a": 100.0,
            "lineup_off_rating_h": 110.0,
            "lineup_off_rating_a": 110.0,
            "lineup_def_rating_h": 110.0,
            "lineup_def_rating_a": 110.0,
            "home_poss": home_poss,
            "away_poss": away_poss,
            "home_pts": 1.1 * home_poss,
            "away_pts": 1.1 * away_poss,
            "margin": np.array([1.0] * 40 + [20.0] * 40),
            "n_ot": np.array([1] * 40 + [0] * 40),
        }
    )
    fitted = fit_player_context(games)

    own_pace = fitted["sides"]["beta"][1 + fitted["sides"]["features"].index("own_pace")]
    assert own_pace == pytest.approx(3.0, abs=0.05)
    assert fitted["scoring"]["league_ppp"] == pytest.approx(1.1)
    assert fitted["ot_given_margin"]["p_ot"][0] == pytest.approx(1.0)
    assert fitted["ot_given_margin"]["p_ot"][-1] == pytest.approx(0.0)


def test_live_slate_uses_pinnacle_lines_and_rotowire_starters(tmp_path):
    pinnacle = tmp_path / "pinnacle"
    lineups = tmp_path / "lineups"
    pinnacle.mkdir()
    lineups.mkdir()
    (pinnacle / "pinnacle_nba_2026-11-05_190000_team.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "participants": ["Boston Celtics", "Los Angeles Lakers"],
                        "team_markets": {
                            "spread": [
                                {
                                    "period": 0,
                                    "lines": [
                                        {"team": "Los Angeles Lakers", "points": -4.5, "side": "home"},
                                        {"team": "Boston Celtics", "points": 4.5, "side": "away"},
                                    ],
                                }
                            ],
                            "total": [
                                {
                                    "period": 0,
                                    "lines": [
                                        {"side": "over", "points": 226.5},
                                        {"side": "under", "points": 226.5},
                                    ],
                                }
                            ],
                        },
                    }
                ]
            }
        )
    )
    (lineups / "rotowire_nba_2026-11-05_180000.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "home": {"team": "LAL", "lineup": [{"name": "LeBron James"}]},
                        "away": {"team": "BOS", "lineup": [{"name": "Jayson Tatum"}]},
                    }
                ]
            }
        )
    )
    roster = pd.DataFrame(
        [
            {"player_name": "LeBron James", "player_id": 2544, "team_abbreviation": "LAL"},
            {"player_name": "Jayson Tatum", "player_id": 1628369, "team_abbreviation": "BOS"},
        ]
    )

    slate = load_live_slate(
        "LAL",
        "2026-11-05",
        roster,
        pinnacle_dir=pinnacle,
        lineup_dir=lineups,
    )

    assert slate["opponent"] == "BOS"
    assert slate["is_home"] is True
    assert slate["spread"] == -4.5
    assert slate["total"] == 226.5
    assert slate["team_lineup"] == [2544]
    assert slate["opponent_lineup"] == [1628369]
    assert slate["team_out"] == []
    assert slate["opponent_out"] == []


def test_live_slate_marks_rotowire_out_players_unavailable(tmp_path):
    pinnacle = tmp_path / "pinnacle"
    lineups = tmp_path / "lineups"
    pinnacle.mkdir()
    lineups.mkdir()
    (pinnacle / "pinnacle_nba_2026-11-05_190000_team.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "participants": ["Boston Celtics", "Los Angeles Lakers"],
                        "team_markets": {
                            "spread": [
                                {
                                    "period": 0,
                                    "lines": [
                                        {
                                            "team": "Los Angeles Lakers",
                                            "points": -4.5,
                                            "side": "home",
                                        },
                                        {
                                            "team": "Boston Celtics",
                                            "points": 4.5,
                                            "side": "away",
                                        },
                                    ],
                                }
                            ],
                            "total": [
                                {
                                    "period": 0,
                                    "lines": [{"side": "over", "points": 226.5}],
                                }
                            ],
                        },
                    }
                ]
            }
        )
    )
    (lineups / "rotowire_nba_2026-11-05_180000.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "home": {
                            "team": "LAL",
                            "lineup": [{"name": "LeBron James"}],
                            "may_not_play": [
                                {"name": "Anthony Davis", "badge": "Out"},
                                {"name": "Austin Reaves", "badge": "Ques"},
                                {"name": "Rui Hachimura", "badge": "Probable"},
                            ],
                        },
                        "away": {
                            "team": "BOS",
                            "lineup": [{"name": "Jayson Tatum"}],
                            "may_not_play": [
                                {"name": "Kristaps Porzingis", "badge": "OFS"},
                            ],
                        },
                    }
                ]
            }
        )
    )
    roster = pd.DataFrame(
        [
            {"player_name": "LeBron James", "player_id": 2544, "team_abbreviation": "LAL"},
            {"player_name": "Anthony Davis", "player_id": 203076, "team_abbreviation": "LAL"},
            {"player_name": "Austin Reaves", "player_id": 1630559, "team_abbreviation": "LAL"},
            {"player_name": "Rui Hachimura", "player_id": 1629060, "team_abbreviation": "LAL"},
            {"player_name": "Jayson Tatum", "player_id": 1628369, "team_abbreviation": "BOS"},
            {
                "player_name": "Kristaps Porzingis",
                "player_id": 204001,
                "team_abbreviation": "BOS",
            },
        ]
    )

    slate = load_live_slate(
        "LAL",
        "2026-11-05",
        roster,
        pinnacle_dir=pinnacle,
        lineup_dir=lineups,
    )

    assert slate["team_out"] == [203076]
    assert slate["opponent_out"] == [204001]
    assert slate["team_probable"] == [1629060]
    assert slate["opponent_probable"] == []
    assert slate["team_out_absent"] == []
    assert slate["opponent_out_absent"] == []
    assert slate["team_lineup"] == [2544]


def test_live_slate_leaves_off_an_out_player_with_no_game(tmp_path):
    pinnacle = tmp_path / "pinnacle"
    lineups = tmp_path / "lineups"
    pinnacle.mkdir()
    lineups.mkdir()
    (pinnacle / "pinnacle_nba_2026-11-05_190000_team.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "participants": [
                            "Portland Trail Blazers",
                            "Los Angeles Lakers",
                        ],
                        "team_markets": {
                            "spread": [
                                {
                                    "period": 0,
                                    "lines": [
                                        {
                                            "team": "Los Angeles Lakers",
                                            "points": -6.5,
                                            "side": "home",
                                        },
                                        {
                                            "team": "Portland Trail Blazers",
                                            "points": 6.5,
                                            "side": "away",
                                        },
                                    ],
                                }
                            ],
                            "total": [
                                {
                                    "period": 0,
                                    "lines": [{"side": "over", "points": 224.5}],
                                }
                            ],
                        },
                    }
                ]
            }
        )
    )
    (lineups / "rotowire_nba_2026-11-05_180000.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "home": {
                            "team": "LAL",
                            "lineup": [{"name": "LeBron James"}],
                        },
                        "away": {
                            "team": "POR",
                            "lineup": [{"name": "Scoot Henderson"}],
                            "may_not_play": [
                                {"name": "Shaedon Sharpe", "badge": "Out"},
                            ],
                        },
                    }
                ]
            }
        )
    )
    roster = pd.DataFrame(
        [
            {"player_name": "LeBron James", "player_id": 2544, "team_abbreviation": "LAL"},
            {
                "player_name": "Scoot Henderson",
                "player_id": 1630703,
                "team_abbreviation": "POR",
            },
        ]
    )

    slate = load_live_slate(
        "LAL",
        "2026-11-05",
        roster,
        pinnacle_dir=pinnacle,
        lineup_dir=lineups,
    )

    assert slate["opponent"] == "POR"
    assert slate["opponent_lineup"] == [1630703]
    assert slate["opponent_out"] == []
    assert slate["opponent_out_absent"] == ["Shaedon Sharpe"]
    assert slate["team_out_absent"] == []


def test_live_slate_rejects_a_team_missing_from_the_board(tmp_path):
    pinnacle = tmp_path / "pinnacle"
    lineups = tmp_path / "lineups"
    pinnacle.mkdir()
    lineups.mkdir()
    (pinnacle / "pinnacle_nba_2026-11-05_190000_team.json").write_text(
        json.dumps({"games": []})
    )
    roster = pd.DataFrame(
        [{"player_name": "LeBron James", "player_id": 2544, "team_abbreviation": "LAL"}]
    )

    with pytest.raises(ValueError, match="LAL"):
        load_live_slate(
            "LAL",
            "2026-11-05",
            roster,
            pinnacle_dir=pinnacle,
            lineup_dir=lineups,
        )


def _board(spread: float) -> dict:
    return {
        "games": [
            {
                "participants": ["Boston Celtics", "Los Angeles Lakers"],
                "team_markets": {
                    "spread": [
                        {
                            "period": 0,
                            "lines": [
                                {"team": "Los Angeles Lakers", "points": spread, "side": "home"},
                                {"team": "Boston Celtics", "points": -spread, "side": "away"},
                            ],
                        }
                    ],
                    "total": [
                        {
                            "period": 0,
                            "lines": [{"side": "over", "points": 220.0}],
                        }
                    ],
                },
            }
        ]
    }


def test_latest_slate_uses_the_newest_odds_and_lineup_files(tmp_path):
    pinnacle = tmp_path / "pinnacle"
    lineups = tmp_path / "lineups"
    pinnacle.mkdir()
    lineups.mkdir()
    (pinnacle / "pinnacle_nba_2026-11-04_230000_team.json").write_text(
        json.dumps(_board(-20.0))
    )
    (pinnacle / "pinnacle_nba_2026-11-05_120000_team.json").write_text(
        json.dumps(_board(-1.0))
    )
    (pinnacle / "pinnacle_nba_2026-11-05_190000_team.json").write_text(
        json.dumps(_board(-4.5))
    )
    (lineups / "rotowire_nba_2026-11-05_120000.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "home": {"team": "LAL", "lineup": [{"name": "LeBron James"}]},
                        "away": {"team": "BOS", "lineup": [{"name": "Old Celtic"}]},
                    }
                ]
            }
        )
    )
    (lineups / "rotowire_nba_2026-11-05_190000.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "away": {"team": "BOS", "lineup": [{"name": "Jayson Tatum"}]},
                        "home": {"team": "LAL", "lineup": [{"name": "LeBron James"}]},
                    }
                ]
            }
        )
    )
    roster = pd.DataFrame(
        [
            {"player_name": "LeBron James", "player_id": 2544, "team_abbreviation": "LAL"},
            {"player_name": "Jayson Tatum", "player_id": 1628369, "team_abbreviation": "BOS"},
            {"player_name": "Old Celtic", "player_id": 99, "team_abbreviation": "BOS"},
        ]
    )

    slate = load_latest_slate("LAL", roster, pinnacle_dir=pinnacle, lineup_dir=lineups)

    assert slate["date"] == "2026-11-05"
    assert slate["spread"] == -4.5
    assert slate["total"] == 220.0
    assert slate["team_lineup"] == [2544]
    assert slate["opponent_lineup"] == [1628369]
    assert slate["team_out"] == []
    assert slate["opponent_out"] == []
    assert slate["pinnacle_file"].endswith("pinnacle_nba_2026-11-05_190000_team.json")
