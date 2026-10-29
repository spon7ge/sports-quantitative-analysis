from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.dynamic_points import (
    DynamicPointsConfig,
    build_causal_frame,
    kalman_update,
    predictive_scale,
)


def _games() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, start in (
        ("2022-23", "2022-10-01"),
        ("2023-24", "2023-10-01"),
        ("2024-25", "2024-10-01"),
    ):
        for game in range(3):
            for player_id, team_id, opponent_id, position in (
                (1, 10, 20, "PG"),
                (2, 20, 10, "C"),
            ):
                rows.append(
                    {
                        "season_year": season,
                        "season_type": "Regular Season",
                        "game_date": pd.Timestamp(start)
                        + pd.Timedelta(days=game * 2),
                        "game_id": f"{season}-{game}",
                        "player_id": player_id,
                        "player_name": f"Player {player_id}",
                        "team_id": team_id,
                        "opp_team_id": opponent_id,
                        "pos": position,
                        "matchup": "AAA vs. BBB"
                        if team_id == 10
                        else "BBB @ AAA",
                        "minutes": 30.0 + player_id + game,
                        "fga": 10.0 + player_id + game,
                        "fta": 2.0 + game,
                        "pts": 15.0 + 2 * player_id + game,
                        "team_def_rating": 110.0 + team_id / 10 + game,
                    }
                )
    return pd.DataFrame(rows)


def test_kalman_update_matches_closed_form() -> None:
    result = kalman_update(
        mean=3.0,
        variance=4.0,
        observation=20.0,
        base_mean=12.0,
        observation_variance=16.0,
    )
    assert result.gain == pytest.approx(0.2)
    assert result.innovation == pytest.approx(5.0)
    assert result.mean == pytest.approx(4.0)
    assert result.variance == pytest.approx(3.2)


def test_predictive_scale_adds_variances() -> None:
    assert predictive_scale(9.0, 16.0) == pytest.approx(5.0)


def test_causal_features_use_only_prior_games() -> None:
    frame, audit = build_causal_frame(_games())
    player = frame.loc[frame["player_id"].eq(1)].sort_values("game_date")
    assert np.isnan(player.iloc[0]["min_lag_1"])
    assert player.iloc[1]["min_lag_1"] == pytest.approx(
        player.iloc[0]["minutes"]
    )
    assert player.iloc[1]["fga_lag_1"] == pytest.approx(player.iloc[0]["fga"])
    assert player.iloc[1]["fta_lag_1"] == pytest.approx(player.iloc[0]["fta"])
    assert audit["input_rows"] == len(_games())
    assert audit["output_rows"] == len(_games())


def test_team_and_model_rookie_flags_follow_prior_history() -> None:
    games = _games()
    mask = games["player_id"].eq(1) & games["season_year"].eq("2024-25")
    games.loc[mask, ["team_id", "opp_team_id"]] = [30, 40]
    frame, _ = build_causal_frame(games)
    player = frame.loc[frame["player_id"].eq(1)].sort_values("game_date")
    first_2024 = player.loc[player["season_year"].eq("2024-25")].iloc[0]
    assert bool(first_2024["new_team_flag"])
    assert not bool(first_2024["rookie_flag"])
    first_2022 = player.loc[player["season_year"].eq("2022-23")].iloc[0]
    assert bool(first_2022["rookie_flag"])


def test_duplicate_player_game_is_rejected() -> None:
    duplicate = pd.concat([_games(), _games().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate player/game"):
        build_causal_frame(duplicate)


def test_config_rejects_nonpositive_uncertainty() -> None:
    with pytest.raises(ValueError, match="positive"):
        DynamicPointsConfig(state_drift_sd=0.0)
