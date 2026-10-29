from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.dynamic_points import (
    DynamicPointsConfig,
    build_causal_frame,
    kalman_update,
    predictive_scale,
    walk_forward_trace,
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
                        + pd.Timedelta(days=game * (game + 1)),
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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mean": np.nan}, "state inputs"),
        ({"observation": np.inf}, "state inputs"),
        ({"base_mean": -np.inf}, "state inputs"),
        ({"variance": np.nan}, "variances"),
        ({"observation_variance": np.inf}, "variances"),
    ],
)
def test_kalman_update_rejects_nonfinite_inputs(
    kwargs: dict[str, float],
    match: str,
) -> None:
    inputs = {
        "mean": 3.0,
        "variance": 4.0,
        "observation": 20.0,
        "base_mean": 12.0,
        "observation_variance": 16.0,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=match):
        kalman_update(**inputs)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_predictive_scale_rejects_nonfinite_variances(value: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        predictive_scale(value, 16.0)


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


def test_opponent_defense_prior_excludes_current_game() -> None:
    games = _games()
    current_game = games["game_id"].eq("2023-24-0")
    games.loc[current_game & games["team_id"].eq(20), "team_def_rating"] = 999.0

    frame, _ = build_causal_frame(games)

    player_game = frame.loc[
        frame["player_id"].eq(1) & frame["game_id"].eq("2023-24-0")
    ].iloc[0]
    assert player_game["opp_def_prior"] == pytest.approx(113.0)


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


@pytest.mark.parametrize(
    "column",
    ["minutes", "fga", "fta", "team_def_rating"],
)
def test_malformed_numeric_source_values_are_rejected(column: str) -> None:
    games = _games()
    games[column] = games[column].astype(object)
    games.loc[games.index[0], column] = "not-a-number"

    with pytest.raises(ValueError, match=rf"{column}.*1"):
        build_causal_frame(games)


def test_legitimate_numeric_missing_values_are_audited() -> None:
    games = _games()
    games.loc[games.index[0], "minutes"] = np.nan
    games.loc[games.index[1], "fga"] = None
    games.loc[games.index[2], "fta"] = pd.NA
    games.loc[games.index[3], "team_def_rating"] = np.nan

    _, audit = build_causal_frame(games)

    assert audit["minutes_missing_values"] == 1
    assert audit["fga_missing_values"] == 1
    assert audit["fta_missing_values"] == 1
    assert audit["team_def_rating_missing_values"] == 1


def test_non_regular_season_removal_is_truthfully_audited() -> None:
    games = _games()
    games.loc[games.index[0], "season_type"] = "Playoffs"

    _, audit = build_causal_frame(games)

    assert audit["non_regular_season_rows_removed"] == 1
    assert "postseason_rows_removed" not in audit


def test_config_rejects_nonpositive_uncertainty() -> None:
    with pytest.raises(ValueError, match="positive"):
        DynamicPointsConfig(state_drift_sd=0.0)


def test_walk_forward_is_out_of_sample_and_predicts_before_update() -> None:
    frame, _ = build_causal_frame(_games())
    trace, audits = walk_forward_trace(
        frame,
        evaluation_seasons=("2024-25",),
        config=DynamicPointsConfig(train_window_seasons=2),
    )
    assert set(trace["season_year"]) == {"2024-25"}
    assert audits[0]["train_seasons"] == ("2022-23", "2023-24")
    assert audits[0]["evaluation_season"] == "2024-25"
    first = trace.sort_values(["game_date", "game_id", "player_id"]).iloc[0]
    expected_innovation = first["points"] - first["predictive_mean"]
    assert first["innovation"] == pytest.approx(expected_innovation)
    expected_after = (
        first["state_before"] + first["kalman_gain"] * first["innovation"]
    )
    assert first["state_after"] == pytest.approx(expected_after)


def test_offseason_variance_is_added_once_without_moving_mean() -> None:
    frame, _ = build_causal_frame(_games())
    config = DynamicPointsConfig(
        state_drift_sd=0.5,
        offseason_drift_sd=2.0,
        train_window_seasons=2,
    )
    trace, _ = walk_forward_trace(
        frame,
        evaluation_seasons=("2024-25",),
        config=config,
    )
    player = trace.loc[trace["player_id"].eq(1)].sort_values("game_date")
    first, second = player.iloc[0], player.iloc[1]
    assert bool(first["season_boundary"])
    assert first["state_before"] == pytest.approx(first["state_end_training"])
    assert first["state_var_before"] == pytest.approx(
        first["state_var_end_training"] + config.offseason_drift_sd**2
    )
    assert not bool(second["season_boundary"])
    assert second["state_var_before"] == pytest.approx(
        first["state_var_after"] + config.state_drift_sd**2
    )


def test_trace_contains_finite_variance_and_identity_columns() -> None:
    frame, _ = build_causal_frame(_games())
    trace, _ = walk_forward_trace(
        frame,
        evaluation_seasons=("2024-25",),
        config=DynamicPointsConfig(),
    )
    required = {
        "team_id",
        "opp_team_id",
        "position",
        "state_var_before",
        "obs_var",
        "predictive_var",
        "games_in_model_before",
        "returning_player",
        "new_team_flag",
        "rookie_flag",
        "z",
        "pit",
        "df",
    }
    assert required.issubset(trace.columns)
    finite_numeric = [
        "team_id",
        "opp_team_id",
        "state_var_before",
        "obs_var",
        "predictive_var",
        "games_in_model_before",
        "z",
        "pit",
        "df",
    ]
    assert np.isfinite(trace[finite_numeric]).all().all()
    assert (trace["predictive_var"] > 0).all()
    assert np.allclose(
        trace["predictive_var"],
        trace["state_var_before"] + trace["obs_var"],
    )


def test_unknown_evaluation_season_is_rejected() -> None:
    frame, _ = build_causal_frame(_games())
    with pytest.raises(ValueError, match="evaluation season"):
        walk_forward_trace(
            frame,
            evaluation_seasons=("2030-31",),
            config=DynamicPointsConfig(),
        )
