"""Layer 4 allocates supplied possessions inside Layer 3 lineups.

Events in a segment belong to the five players on the court. Conditioning
on a score changes makes, not the sampled attempts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from game_opportunities import (
    OpportunityModel,
    allocate_game_opportunities,
    allocate_opportunities,
    fit_opportunities,
)


def _model(**overrides) -> OpportunityModel:
    values = {
        "fga_per_min": 0.5,
        "p_3pa": 0.0,
        "p_3p": 0.0,
        "p_2p": 1.0,
        "fta_per_fga": 0.0,
        "ft_pct": 0.0,
        "tov_per_min": 0.1,
        "fga_per_poss": 0.5,
        "tov_per_poss": 0.1,
        "fta_per_min": 0.0,
        "fta_per_poss": 0.0,
        "alpha": 0.0,
    }
    values.update(overrides)
    return OpportunityModel(**values)


def _profile(player_ids) -> pd.DataFrame:
    return pd.DataFrame({"player_id": list(player_ids)})


def _quarters(players) -> list[dict]:
    return [
        {"period": period, "start_min": 0.0, "end_min": 12.0, "players": list(players)}
        for period in range(1, 5)
    ]


def test_model_carries_assist_and_rebound_rates():
    model = _model()

    assert model.version == 3
    assert model.assist_rate == pytest.approx(0.55)
    assert model.oreb_rate == pytest.approx(0.25)
    assert model.assist_rho == pytest.approx(0.0)
    assert model.oreb_rho == pytest.approx(0.0)


def test_segment_events_stay_with_the_players_on_the_court():
    players = [1, 2, 3, 4, 5, 6]
    starters = players[:5]
    second = [6, 2, 3, 4, 5]
    schedule = [
        {"period": 1, "start_min": 0.0, "end_min": 6.0, "players": starters},
        {"period": 1, "start_min": 6.0, "end_min": 12.0, "players": second},
        *[
            {"period": period, "start_min": 0.0, "end_min": 12.0, "players": starters}
            for period in range(2, 5)
        ],
    ]
    # Player 1 sits the last six minutes of the first quarter. Player 6 plays them.
    minutes = np.array([[42.0, 48.0, 48.0, 48.0, 48.0, 6.0]])
    drawn = allocate_opportunities(
        _model(),
        _profile(players),
        minutes,
        np.array([100.0]),
        np.random.default_rng(0),
        game_minutes=np.array([48.0]),
        segments=[schedule],
        return_intervals=True,
    )

    assert drawn["FGA"].sum() == drawn["team_fga"][0]
    assert drawn["metadata"]["segments_used"] is True
    for interval in drawn["intervals"]:
        active = interval["active"]
        for values in interval["stats"].values():
            assert np.all(values[~active] == 0)


def test_conditioned_score_uses_the_sampled_attempts():
    players = [1, 2, 3, 4, 5]
    minutes = np.full((8, 5), 48.0)
    points = np.full(8, 4.0)
    drawn = allocate_opportunities(
        _model(),
        _profile(players),
        minutes,
        np.full(8, 4.0),
        np.random.default_rng(1),
        team_points=points,
        game_minutes=np.full(8, 48.0),
        segments=[_quarters(players)] * 8,
    )

    assert np.array_equal(drawn["PTS"].sum(axis=1), np.full(8, 4))
    assert np.array_equal(drawn["FGA"], drawn["FGM"])
    assert np.all(drawn["team_fga"] == 2)


def test_both_teams_share_one_timeline_and_their_scores():
    home_players = [1, 2, 3, 4, 5]
    away_players = [11, 12, 13, 14, 15]
    environment = pd.DataFrame(
        {
            "is_home": [1],
            "team_poss": [4.0],
            "opp_poss": [4.0],
            "game_minutes": [48.0],
            "n_ot": [0],
            "team_pts": [4.0],
            "opp_pts": [2.0],
        }
    )
    home_minutes = pd.DataFrame(
        np.full((1, 5), 48.0),
        columns=home_players,
        index=environment.index,
    )
    away_minutes = pd.DataFrame(
        np.full((1, 5), 48.0),
        columns=away_players,
        index=environment.index,
    )

    # Away makes are certain twos as well, so two points means one made attempt.
    result = allocate_game_opportunities(
        _model(),
        _profile(home_players),
        _profile(away_players),
        home_minutes,
        away_minutes,
        environment,
        np.random.default_rng(2),
        home_segments=[_quarters(home_players)],
        away_segments=[_quarters(away_players)],
        max_score_resamples=80,
    )

    assert result["home"]["PTS"].sum() == 4
    assert result["away"]["PTS"].sum() == 2
    assert result["simulation_index"].equals(environment.index)


def test_fit_reads_assists_from_a_complete_roster():
    rows = []
    for player in range(5):
        rows.append(
            {
                "player_id": player + 1,
                "team_id": 10,
                "game_id": "0022400001",
                "game_date": "2024-01-01",
                "minutes": 48.0,
                "fga": 10,
                "fgm": 5,
                "fg3_a": 0,
                "fg3_m": 0,
                "fta": 0,
                "ftm": 0,
                "tov": 1,
                "ast": 2,
                "oreb": 1,
                "team_fga": 50,
                "team_poss": 100.0,
                "team_tov": 10,
            }
        )
    model = fit_opportunities(pd.DataFrame(rows), alpha=0.0)
    assert model.assist_rate == pytest.approx(0.4)
