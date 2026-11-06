"""Layer 4 turns possessions and minutes into one team's shots and turnovers.

Training rates stop before the game being scored. Live rates keep the latest
completed game. The possession count and the minutes are inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from possession_allocation import (
    allocate_opportunities,
    fit_opportunities,
    live_opportunities,
    training_opportunities,
)


def _row(**overrides) -> dict:
    row = {
        "player_id": 1,
        "game_id": "g1",
        "game_date": "2024-01-01",
        "team_id": 10,
        "min": 20.0,
        "fga": 10.0,
        "fgm": 5.0,
        "fg3_a": 4.0,
        "fg3_m": 2.0,
        "fta": 4.0,
        "ftm": 3.0,
        "tov": 2.0,
        "team_fga": 40.0,
        "team_poss": 100.0,
        "team_tov": 12.0,
    }
    row.update(overrides)
    return row


def _logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(),
            _row(
                game_id="g2",
                game_date="2024-01-15",
                min=30.0,
                fga=30.0,
                fgm=12.0,
                fg3_a=12.0,
                fg3_m=3.0,
                fta=6.0,
                ftm=6.0,
                tov=6.0,
                team_fga=80.0,
                team_poss=100.0,
                team_tov=20.0,
            ),
        ]
    )


def _slate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": "2024-02-01",
                "team_id": 10,
                "opp_team_id": 20,
                "is_home": 1,
                "is_starter": 1,
            }
        ]
    )


def test_training_rates_exclude_the_game_being_scored():
    trained = training_opportunities(_logs())
    game2 = trained.loc[trained["game_id"].eq("g2")].iloc[0]
    assert game2["fga_per_min"] == pytest.approx(10.0 / 20.0)
    assert game2["fga_per_poss"] == pytest.approx(40.0 / 100.0)
    assert game2["tov_per_poss"] == pytest.approx(12.0 / 100.0)
    assert game2["p_3pa"] == pytest.approx(4.0 / 10.0)
    assert game2["p_3p"] == pytest.approx(2.0 / 4.0)
    assert game2["p_2p"] == pytest.approx(3.0 / 6.0)
    assert game2["fta_per_fga"] == pytest.approx(4.0 / 10.0)
    assert game2["ft_pct"] == pytest.approx(3.0 / 4.0)
    assert game2["tov_per_min"] == pytest.approx(2.0 / 20.0)
    opening = trained.loc[trained["game_id"].eq("g1")].iloc[0]
    assert np.isnan(opening["fga_per_min"])
    assert np.isnan(opening["fga_per_poss"])


def test_live_rates_keep_the_latest_completed_game():
    live = live_opportunities(_logs(), _slate()).iloc[0]
    assert live["fga_per_min"] == pytest.approx((0.5 + 1.0) / 2.0)
    assert live["fga_per_poss"] == pytest.approx((0.4 + 0.8) / 2.0)
    assert live["tov_per_poss"] == pytest.approx((0.12 + 0.20) / 2.0)
    assert live["p_3pa"] == pytest.approx((0.4 + 12.0 / 30.0) / 2.0)


def test_live_rates_stop_before_the_slate_date():
    logs = _logs()
    future = _row(game_id="g3", game_date="2024-02-01", fga=99.0, min=10.0)
    live = live_opportunities(pd.concat([logs, pd.DataFrame([future])]), _slate())
    assert live.iloc[0]["fga_per_min"] == pytest.approx((0.5 + 1.0) / 2.0)


def test_allocation_uses_the_supplied_possessions_and_minutes():
    profile = pd.DataFrame(
        [
            {
                "player_id": 1,
                "fga_per_min": 0.5,
                "p_3pa": 0.0,
                "p_3p": 0.0,
                "p_2p": 1.0,
                "fta_per_fga": 0.0,
                "ft_pct": 0.0,
                "tov_per_min": 0.2,
                "fga_per_poss": 0.8,
                "tov_per_poss": 0.1,
            },
            {
                "player_id": 2,
                "fga_per_min": 0.5,
                "p_3pa": 0.0,
                "p_3p": 0.0,
                "p_2p": 1.0,
                "fta_per_fga": 0.0,
                "ft_pct": 0.0,
                "tov_per_min": 0.0,
                "fga_per_poss": 0.8,
                "tov_per_poss": 0.1,
            },
        ]
    )
    model = fit_opportunities(profile, alpha=0.0)
    minutes = np.tile(np.array([[36.0, 12.0]]), (2000, 1))
    short = allocate_opportunities(
        model, profile, minutes, np.full(2000, 100.0), np.random.default_rng(0)
    )
    long = allocate_opportunities(
        model, profile, minutes, np.full(2000, 200.0), np.random.default_rng(0)
    )
    assert np.array_equal(short["FGA"].sum(axis=1), short["team_fga"])
    assert np.array_equal(short["TOV"].sum(axis=1), short["team_tov"])
    assert np.array_equal(short["PTS"], 2 * short["FGA"])
    assert np.all(short["TOV"][:, 1] == 0)
    assert short["FGA"][:, 0].mean() / short["team_fga"].mean() == pytest.approx(0.75, abs=0.03)
    assert short["team_fga"].mean() == pytest.approx(80.0, abs=3.0)
    assert long["team_fga"].mean() == pytest.approx(160.0, abs=4.0)


def test_a_player_with_no_minutes_takes_no_shots():
    profile = pd.DataFrame(
        [
            {
                "player_id": 1,
                "fga_per_min": 0.5,
                "p_3pa": 0.0,
                "p_3p": 0.0,
                "p_2p": 1.0,
                "fta_per_fga": 0.0,
                "ft_pct": 0.8,
                "tov_per_min": 0.1,
                "fga_per_poss": 0.8,
                "tov_per_poss": 0.1,
            },
            {
                "player_id": 2,
                "fga_per_min": 0.5,
                "p_3pa": 0.0,
                "p_3p": 0.0,
                "p_2p": 1.0,
                "fta_per_fga": 0.0,
                "ft_pct": 0.8,
                "tov_per_min": 0.1,
                "fga_per_poss": 0.8,
                "tov_per_poss": 0.1,
            },
        ]
    )
    model = fit_opportunities(profile, alpha=0.0)
    minutes = np.tile(np.array([[30.0, 0.0]]), (20, 1))
    drawn = allocate_opportunities(
        model, profile, minutes, np.full(20, 100.0), np.random.default_rng(1)
    )
    assert np.all(drawn["FGA"][:, 1] == 0)
    assert np.all(drawn["TOV"][:, 1] == 0)


def test_missing_history_uses_the_fitted_fallback():
    trained = training_opportunities(_logs())
    model = fit_opportunities(trained, alpha=0.0)
    assert model.fga_per_min == pytest.approx(0.5)
    assert model.fga_per_poss == pytest.approx(0.4)
    blank = pd.DataFrame(
        [
            {"player_id": 9, "team_id": 10},
            {"player_id": 8, "team_id": 10},
        ]
    )
    minutes = np.tile(np.array([[24.0, 24.0]]), (1500, 1))
    drawn = allocate_opportunities(
        model, blank, minutes, np.full(1500, 100.0), np.random.default_rng(2)
    )
    assert drawn["team_fga"].mean() == pytest.approx(40.0, abs=2.0)
    assert drawn["FGA"][:, 0].mean() / drawn["team_fga"].mean() == pytest.approx(0.5, abs=0.04)


def test_conflicting_team_attempts_raise():
    logs = pd.DataFrame(
        [
            _row(player_id=1, team_fga=40.0),
            _row(player_id=2, team_fga=50.0),
        ]
    )
    with pytest.raises(ValueError, match="team_fga"):
        training_opportunities(logs)


def test_player_points_sum_to_the_scenario_score():
    profile = pd.DataFrame(
        [
            {
                "player_id": 1,
                "fga_per_min": 0.5,
                "p_3pa": 0.0,
                "p_3p": 0.0,
                "p_2p": 1.0,
                "fta_per_fga": 0.0,
                "ft_pct": 0.0,
                "tov_per_min": 0.1,
                "fga_per_poss": 0.8,
                "tov_per_poss": 0.1,
            },
            {
                "player_id": 2,
                "fga_per_min": 0.5,
                "p_3pa": 0.0,
                "p_3p": 0.0,
                "p_2p": 1.0,
                "fta_per_fga": 0.0,
                "ft_pct": 0.0,
                "tov_per_min": 0.1,
                "fga_per_poss": 0.8,
                "tov_per_poss": 0.1,
            },
            {
                "player_id": 3,
                "fga_per_min": 0.5,
                "p_3pa": 0.0,
                "p_3p": 0.0,
                "p_2p": 1.0,
                "fta_per_fga": 0.0,
                "ft_pct": 0.0,
                "tov_per_min": 0.0,
                "fga_per_poss": 0.8,
                "tov_per_poss": 0.1,
            },
        ]
    )
    model = fit_opportunities(profile, alpha=0.0)
    minutes = np.tile(np.array([[36.0, 12.0, 0.0]]), (30, 1))
    team_points = np.array([110.4, 200.6] * 15)
    drawn = allocate_opportunities(
        model,
        profile,
        minutes,
        np.full(30, 100.0),
        np.random.default_rng(4),
        team_points=team_points,
    )

    scored = 2 * drawn["2PM"] + 3 * drawn["3PM"] + drawn["FTM"]
    assert np.array_equal(drawn["PTS"].sum(axis=1), np.array([110, 201] * 15))
    assert np.array_equal(drawn["PTS"], scored)
    assert np.array_equal(drawn["FGM"], drawn["2PM"] + drawn["3PM"])
    assert np.array_equal(drawn["FGA"], drawn["2PA"] + drawn["3PA"])
    assert np.array_equal(drawn["FGA"].sum(axis=1), drawn["team_fga"])
    assert np.all(drawn["2PM"] <= drawn["2PA"])
    assert np.all(drawn["3PM"] <= drawn["3PA"])
    assert np.all(drawn["FTM"] <= drawn["FTA"])
    assert np.all(drawn["PTS"][:, 2] == 0)
    assert np.all(drawn["FGA"][:, 2] == 0)
    assert np.all(drawn["FTA"][:, 2] == 0)
    assert drawn["team_fga"].mean() == pytest.approx(80.0, abs=8.0)
    assert drawn["PTS"][:, 0].mean() > drawn["PTS"][:, 1].mean()


def test_scenario_points_must_match_the_simulation_rows():
    profile = pd.DataFrame(
        [
            {
                "player_id": 1,
                "fga_per_min": 0.5,
                "p_3pa": 0.0,
                "p_3p": 0.0,
                "p_2p": 1.0,
                "fta_per_fga": 0.0,
                "ft_pct": 0.0,
                "tov_per_min": 0.1,
                "fga_per_poss": 0.8,
                "tov_per_poss": 0.1,
            }
        ]
    )
    model = fit_opportunities(profile, alpha=0.0)
    minutes = np.ones((2, 1))
    with pytest.raises(ValueError, match="team points"):
        allocate_opportunities(
            model,
            profile,
            minutes,
            np.ones(2),
            np.random.default_rng(5),
            team_points=np.ones(1),
        )
    with pytest.raises(ValueError, match="team points"):
        allocate_opportunities(
            model,
            profile,
            minutes,
            np.ones(2),
            np.random.default_rng(5),
            team_points=np.array([10.0, -1.0]),
        )


def test_allocation_rejects_a_possession_vector_of_the_wrong_length():
    profile = pd.DataFrame([_row(player_id=1), _row(player_id=2)])
    trained = training_opportunities(_logs())
    model = fit_opportunities(trained, alpha=0.0)
    minutes = np.ones((4, 2))
    with pytest.raises(ValueError, match="possessions"):
        allocate_opportunities(
            model, profile, minutes, np.ones(3), np.random.default_rng(3)
        )
