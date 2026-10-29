from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from src.models import dynamic_points_diagnostics
from src.models.dynamic_points_diagnostics import (
    core_calibration,
    gain_summary,
    games_played_tier,
    group_calibration,
    offseason_summary,
    posterior_draw_summary,
    shrinkage_summary,
    variance_decomposition,
)


def _high_resolution_student_t_crps(
    outcome: float,
    mean: float,
    scale: float,
    df: float,
) -> float:
    """Independent midpoint-quantile reference for Student-t CRPS."""
    probabilities = (np.arange(400_000, dtype=float) + 0.5) / 400_000
    quantiles = mean + scale * stats.t.ppf(probabilities, df)
    errors = outcome - quantiles
    losses = np.where(
        errors >= 0,
        probabilities * errors,
        (probabilities - 1.0) * errors,
    )
    return float(2.0 * losses.mean())


def _trace() -> pd.DataFrame:
    n = 80
    games = np.arange(n)
    state = np.linspace(-4.0, 4.0, n)
    innovation = -0.25 * state + np.sin(games)
    state_var = np.where(games < 10, 9.0, 1.0)
    obs_var = np.full(n, 16.0)
    scale = np.sqrt(state_var + obs_var)
    return pd.DataFrame(
        {
            "season_year": np.where(games < 40, "2024-25", "2025-26"),
            "player_id": games % 4,
            "team_id": games % 2,
            "opp_team_id": (games + 1) % 2,
            "position": np.where(games % 2, "G", "C"),
            "points": 20.0 + innovation,
            "predictive_mean": np.full(n, 20.0),
            "innovation": innovation,
            "z": innovation / scale,
            "pit": np.linspace(0.01, 0.99, n),
            "state_before": state,
            "state_var_before": state_var,
            "obs_var": obs_var,
            "predictive_var": state_var + obs_var,
            "kalman_gain": state_var / (state_var + obs_var),
            "games_in_model_before": games,
            "season_game_number": games % 20 + 1,
            "returning_player": np.ones(n, dtype=bool),
            "season_boundary": (games % 20) == 0,
            "new_team_flag": (games % 17) == 0,
            "rookie_flag": (games % 19) == 0,
            "df": np.full(n, 100.0),
        }
    )


def test_games_played_tiers_have_requested_boundaries() -> None:
    tier = games_played_tier(pd.Series([0, 9, 10, 29, 30, 59, 60]))
    assert tier.astype(str).tolist() == [
        "0-9",
        "0-9",
        "10-29",
        "10-29",
        "30-59",
        "30-59",
        "60+",
    ]


def test_variance_shares_sum_to_one_and_keep_coverage() -> None:
    result = variance_decomposition(_trace())
    assert np.allclose(
        result["state_plugin_scale_squared_share"]
        + result["observation_plugin_scale_squared_share"],
        1.0,
    )
    assert {
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "coverage_95",
    }.issubset(result.columns)


def test_shrinkage_detects_regression_toward_zero() -> None:
    by_tier, by_decile = shrinkage_summary(_trace())
    assert {"mean_z", "mean_innovation"}.issubset(by_tier.columns)
    assert by_decile.iloc[0]["mean_innovation"] > 0
    assert by_decile.iloc[-1]["mean_innovation"] < 0
    assert bool(by_decile.iloc[0]["toward_zero"])
    assert bool(by_decile.iloc[-1]["toward_zero"])
    assert by_decile["extreme_decile"].sum() == 2


def test_offseason_summary_compares_early_and_late() -> None:
    windows, relationship = offseason_summary(_trace())
    assert set(windows["window"]) == {5, 10, 20}
    assert set(windows["period"]) == {"early", "rest"}
    assert relationship["n"] > 0
    assert relationship["slope"] < 0


def test_group_flags_respect_minimum_sample() -> None:
    result = group_calibration(_trace(), "position", min_n=50)
    assert {
        "mean_z_se",
        "sd_z_se",
        "mean_flag",
        "sd_flag",
        "low_sample",
    }.issubset(result.columns)
    assert result["low_sample"].all()
    assert not result["actionable_flag"].any()


def test_group_calibration_defaults_to_30_minimum_samples() -> None:
    result = group_calibration(_trace().iloc[:20], "position")
    assert (result["n"] < 30).all()
    assert result["low_sample"].all()
    assert not result["actionable_flag"].any()


@pytest.mark.parametrize("min_n", [True, 1, 2.5])
def test_group_calibration_rejects_invalid_minimum_samples(
    min_n: object,
) -> None:
    with pytest.raises(ValueError, match="integer at least 2"):
        group_calibration(_trace(), "position", min_n=min_n)  # type: ignore[arg-type]


def test_core_metrics_name_sample_and_seasons() -> None:
    result = core_calibration(_trace())
    assert result["n"] == 80
    assert result["seasons"] == ("2024-25", "2025-26")
    assert 0 <= result["coverage_80"] <= 1
    assert int(np.asarray(result["pit_histogram"]).sum()) == 80


def test_gain_summary_suppresses_boundary_jump_flags() -> None:
    trace = _trace()
    trace["kalman_gain"] = np.where(trace["season_boundary"], 1.0, 0.1)
    by_tier, by_games, by_boundary = gain_summary(trace)
    assert {"n", "mean_gain", "gain_se"}.issubset(by_tier.columns)
    assert len(by_games) == 20
    assert by_games["games_quantile"].nunique() == 20
    assert not by_games["unexplained_jump_flag"].any()
    assert set(by_boundary["season_boundary"]) == {False, True}


def test_draw_metrics_skip_cleanly_without_draws() -> None:
    result = posterior_draw_summary(_trace(), draws=None)
    assert result == {
        "status": "skipped",
        "reason": "posterior draws unavailable",
    }


def test_draw_metrics_compare_empirical_and_gaussian_approximations() -> None:
    trace = _trace().iloc[:4].copy()
    draws = np.array(
        [
            [19.0, 20.0, 21.0],
            [20.0, 20.5, 21.0],
            [18.5, 20.0, 21.5],
            [19.5, 20.0, 20.5],
        ]
    )
    result = posterior_draw_summary(trace, draws)
    assert result["n"] == 4
    assert result["n_draws"] == 3
    assert result["crps_difference"] == pytest.approx(
        result["draw_crps"] - result["gaussian_crps"]
    )
    assert result["pit_mean_difference"] == pytest.approx(
        result["draw_pit_mean"] - result["gaussian_pit_mean"]
    )
    assert result["draw_vs_student_t_crps_difference"] == pytest.approx(
        result["draw_crps"] - result["student_t_crps"]
    )
    assert result["draw_vs_student_t_pit_mean_difference"] == pytest.approx(
        result["draw_pit_mean"] - result["student_t_pit_mean"]
    )


def test_student_t_pit_uses_each_rows_exact_df_and_predictive_scale() -> None:
    trace = _trace().iloc[:3].copy()
    trace["df"] = [3.0, 7.0, 30.0]
    trace["points"] = trace["predictive_mean"] + [0.0, 2.0, -3.0]
    trace["innovation"] = trace["points"] - trace["predictive_mean"]
    trace["z"] = trace["innovation"] / np.sqrt(trace["predictive_var"])
    draws = np.tile(trace["predictive_mean"].to_numpy()[:, None], (1, 3))

    result = posterior_draw_summary(trace, draws)

    expected = stats.t.cdf(
        trace["z"].to_numpy(),
        trace["df"].to_numpy(),
    )
    assert result["student_t_pit_mean"] == pytest.approx(expected.mean(), abs=1e-12)
    assert result["student_t_pit_sd"] == pytest.approx(
        np.std(expected, ddof=1),
        abs=1e-12,
    )


def test_student_t_crps_matches_high_resolution_quantile_reference() -> None:
    trace = _trace().iloc[:1].copy()
    trace["df"] = 5.0
    trace["points"] = 23.0
    trace["innovation"] = trace["points"] - trace["predictive_mean"]
    trace["z"] = trace["innovation"] / np.sqrt(trace["predictive_var"])
    draws = np.array([[19.0, 20.0, 21.0]])

    result = posterior_draw_summary(trace, draws)

    expected = _high_resolution_student_t_crps(
        outcome=23.0,
        mean=20.0,
        scale=float(np.sqrt(trace.iloc[0]["predictive_var"])),
        df=5.0,
    )
    assert result["student_t_crps"] == pytest.approx(expected, abs=2e-6)


def test_student_t_crps_processes_rows_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_limit = dynamic_points_diagnostics._STUDENT_T_CRPS_CHUNK_ROWS
    n_rows = 2 * chunk_limit + 17
    processed_sizes: list[int] = []
    evaluate_chunk = dynamic_points_diagnostics._student_t_crps_chunk

    def record_chunk(
        points: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray,
        dfs: np.ndarray,
    ) -> np.ndarray:
        assert len(points) == len(means) == len(scales) == len(dfs)
        processed_sizes.append(len(points))
        return evaluate_chunk(points, means, scales, dfs)

    monkeypatch.setattr(
        dynamic_points_diagnostics,
        "_student_t_crps_chunk",
        record_chunk,
    )
    points = np.linspace(10.0, 30.0, n_rows)
    means = np.linspace(12.0, 28.0, n_rows)
    scales = np.linspace(2.0, 8.0, n_rows)
    dfs = np.linspace(3.0, 50.0, n_rows)

    result = dynamic_points_diagnostics._student_t_row_crps(
        points,
        means,
        scales,
        dfs,
    )
    reference_chunk_rows = 7
    reference = np.concatenate(
        [
            evaluate_chunk(
                points[start : start + reference_chunk_rows],
                means[start : start + reference_chunk_rows],
                scales[start : start + reference_chunk_rows],
                dfs[start : start + reference_chunk_rows],
            )
            for start in range(0, n_rows, reference_chunk_rows)
        ]
    )

    assert result.shape == (n_rows,)
    assert processed_sizes == [chunk_limit, chunk_limit, 17]
    assert max(processed_sizes) <= chunk_limit
    np.testing.assert_allclose(result, reference, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("function", "column"),
    [
        (core_calibration, "z"),
        (variance_decomposition, "predictive_var"),
        (shrinkage_summary, "state_before"),
        (offseason_summary, "season_game_number"),
        (gain_summary, "kalman_gain"),
    ],
)
def test_diagnostics_reject_non_finite_trace_values(
    function: object,
    column: str,
) -> None:
    trace = _trace()
    trace.loc[0, column] = np.nan
    with pytest.raises(ValueError, match=column):
        function(trace)  # type: ignore[operator]


def test_group_calibration_rejects_missing_group_values() -> None:
    trace = _trace()
    trace.loc[0, "position"] = None
    with pytest.raises(ValueError, match="position"):
        group_calibration(trace, "position", min_n=10)


def test_empty_offseason_slices_keep_n_and_explicit_nan() -> None:
    trace = _trace()
    trace["returning_player"] = False
    windows, relationship = offseason_summary(trace)
    assert (windows["n"] == 0).all()
    assert windows["mean_z"].isna().all()
    assert windows["sd_z"].isna().all()
    assert relationship["n"] == 0
    assert np.isnan(relationship["slope"])


def test_numeric_columns_reject_coercible_strings() -> None:
    trace = _trace()
    trace["z"] = trace["z"].astype(str)
    with pytest.raises(ValueError, match="z.*numeric"):
        core_calibration(trace)


def test_public_boundaries_name_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing core_calibration columns.*z"):
        core_calibration(_trace().drop(columns="z"))


@pytest.mark.parametrize(
    "draws",
    [
        np.ones((80,)),
        np.ones((79, 3)),
        np.full((80, 3), np.nan),
        np.empty((80, 0)),
    ],
)
def test_draw_metrics_reject_invalid_draw_arrays(draws: np.ndarray) -> None:
    with pytest.raises(ValueError, match="draws"):
        posterior_draw_summary(_trace(), draws)
