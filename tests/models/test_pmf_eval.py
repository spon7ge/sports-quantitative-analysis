"""Points PMF scores: PIT, lines, and naive comparisons."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from models.shared.metrics import (
    OVER_RATE_TOLERANCE,
    PIT_MEAN_TOLERANCE,
    _flag_gap,
    _pit_summary,
    binary_brier,
    binary_log_loss,
    evaluate_points_pmfs,
    fit_l0_base_rates,
    incremental_slope,
    logistic_calibration,
    paired_date_bootstrap,
    probability_over,
    pseudo_lines,
    seeded_unit_uniform,
    wilson_interval,
)


def test_pseudo_lines_floor_the_ewma_and_clamp_at_a_half_point():
    lines = pseudo_lines([21.9, 0.2, 8.0, np.nan])
    assert lines[0].tolist() == [13.5, 17.5, 21.5, 25.5, 29.5]
    assert lines[1].tolist() == [0.5, 0.5, 0.5, 4.5, 8.5]
    assert lines[2].tolist() == [0.5, 4.5, 8.5, 12.5, 16.5]
    assert np.isnan(lines[3]).all()


def test_probability_over_is_the_mass_above_the_floor():
    pmf = np.array([0.2, 0.3, 0.5])
    assert probability_over(pmf, 0.5) == pytest.approx(0.8)
    assert probability_over(pmf, 1.5) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="line"):
        probability_over(pmf, 1.0)


def test_wilson_interval_for_five_successes_in_twenty_trials():
    low, high = wilson_interval(5, 20)
    assert low == pytest.approx(0.1118, abs=1e-3)
    assert high == pytest.approx(0.4687, abs=1e-3)
    empty_low, empty_high = wilson_interval(0, 0)
    assert math.isnan(empty_low) and math.isnan(empty_high)


def test_logistic_calibration_recovers_a_logit_slope():
    rng = np.random.default_rng(0)
    logit_p = rng.normal(size=800)
    probability = 1.0 / (1.0 + np.exp(-logit_p))
    eta = -0.4 + 1.5 * np.log(probability / (1.0 - probability))
    chance = 1.0 / (1.0 + np.exp(-eta))
    outcome = rng.random(800) < chance
    intercept, slope = logistic_calibration(probability, outcome)
    assert intercept == pytest.approx(-0.4, abs=0.35)
    assert slope == pytest.approx(1.5, abs=0.35)
    assert math.isnan(logistic_calibration(np.full(5, 0.4), np.array([0, 1, 0, 1, 0]))[0])


def test_incremental_slope_is_zero_when_the_probability_is_the_base_rate():
    outcome = np.array([0.0, 1.0, 0.0, 1.0])
    base = np.full(4, 0.5)
    assert math.isnan(incremental_slope(outcome, base, base))
    probability = np.array([0.2, 0.8, 0.2, 0.8])
    assert incremental_slope(outcome, probability, base) == pytest.approx(5.0 / 3.0)


def test_date_bootstrap_keeps_a_game_date_together():
    dates = ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"]
    score_b = np.array([0.0, 1.0, 1.0, 1.0])
    score_a = np.zeros(4)
    result = paired_date_bootstrap(score_a, score_b, dates, n_boot=400, seed=1)
    assert result["diff"] == pytest.approx(0.75)
    assert result["n"] == 4
    assert result["n_dates"] == 2
    assert 0.0 <= result["ci_low"] <= result["diff"] <= result["ci_high"] <= 1.0
    flat = paired_date_bootstrap(np.zeros(3), np.ones(3), ["2024-01-01"] * 3, n_boot=20)
    assert flat["ci_low"] == pytest.approx(1.0)
    assert flat["ci_high"] == pytest.approx(1.0)


def test_base_rates_ignore_the_scored_period():
    history = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2024-11-01", "2024-11-01", "2024-12-01", "2024-12-01"]
            ),
            "tier": ["<15", "<15", "<15", "31+"],
            "pts": [0, 2, 30, 30],
            "pts_ewm_hl_3": [1.2, 1.2, 1.2, 10.2],
        }
    )
    rates = fit_l0_base_rates(history, before="2024-12-01")
    assert rates.set_index("tier").loc["<15", "rate"] == pytest.approx(0.5)
    assert rates.set_index("tier").loc["<15", "n"] == 2
    assert "31+" not in set(rates["tier"])


def test_flag_requires_both_the_tolerance_and_two_standard_errors():
    assert _flag_gap(0.51, 0.50, PIT_MEAN_TOLERANCE, 0.0001) is False
    assert _flag_gap(0.52, 0.50, PIT_MEAN_TOLERANCE, 0.02) is False
    assert _flag_gap(0.52, 0.50, PIT_MEAN_TOLERANCE, 0.001) is True
    pit = np.full(200, 0.9)
    summary = _pit_summary(pit, pit, np.array(["<15"] * 200))
    row = summary.loc[summary["tier"].eq("all")].iloc[0]
    assert row["mean"] == pytest.approx(0.9)
    assert bool(row["mean_flag"])
    assert bool(row["var_scale_flag"])
    noisy = np.array([0.9, 0.9, 0.1])
    small = _pit_summary(noisy, noisy, np.array(["15-24"] * 3))
    small_row = small.loc[small["tier"].eq("all")].iloc[0]
    assert abs(small_row["mean"] - 0.5) > PIT_MEAN_TOLERANCE
    assert bool(small_row["mean_flag"]) is False


def test_each_line_counts_player_games_once():
    scored, history = _frames()
    pmf = _pmfs()
    report = evaluate_points_pmfs(scored, pmf, pmf, history=history, n_boot=50, seed=0)
    lines = report["lines"].loc[report["lines"]["tier"].eq("all")]
    assert list(lines["line"]) == ["L0-8", "L0-4", "L0", "L0+4", "L0+8"]
    assert (lines["n"] == 2).all()
    assert lines["n"].sum() == 10
    histogram = report["pit_histogram"].loc[
        report["pit_histogram"]["tier"].eq("all")
        & report["pit_histogram"]["model"].eq("B")
    ]
    assert histogram["share"].sum() == pytest.approx(1.0)
    assert len(histogram) == 10
    scores = report["scores"].loc[report["scores"]["tier"].eq("all")].iloc[0]
    assert scores["log_score_diff"] == pytest.approx(0.0)
    assert scores["rps_diff"] == pytest.approx(0.0)


def test_history_on_the_scored_date_does_not_change_the_base_rate():
    scored, history = _frames()
    poisoned = pd.concat(
        [
            history,
            pd.DataFrame(
                {
                    "game_date": [pd.Timestamp("2024-12-01")],
                    "tier": ["<15"],
                    "pts": [40],
                    "pts_ewm_hl_3": [1.2],
                }
            ),
        ],
        ignore_index=True,
    )
    clean = evaluate_points_pmfs(
        scored, _pmfs(), _pmfs(), history=history, n_boot=20, seed=0
    )
    dirty = evaluate_points_pmfs(
        scored, _pmfs(), _pmfs(), history=poisoned, n_boot=20, seed=0
    )
    left = clean["naive_line"].set_index("tier")
    right = dirty["naive_line"].set_index("tier")
    assert left.loc["<15", "log_loss_constant"] == pytest.approx(
        right.loc["<15", "log_loss_constant"]
    )
    assert left.loc["all", "n_fit"] == 4


def test_over_rate_flag_stays_down_when_the_gap_is_inside_two_se():
    scored = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "player_id": [1, 2, 3, 4],
            "game_date": pd.Timestamp("2024-12-01"),
            "tier": ["<15", "<15", "<15", "<15"],
            "pts": [2, 2, 2, 0],
            "pts_ewm_hl_3": [1.2, 1.2, 1.2, 1.2],
            "pts_lag_1": [1.0, 1.0, 1.0, 1.0],
            "season_pts_mean": [1.0, 1.0, 1.0, 1.0],
        }
    )
    # L0 = 1.5. Three of four go over. P(Over) is 0.5, so the gap is 0.25.
    pmf = np.tile(np.array([0.2, 0.3, 0.5, 0.0]), (4, 1))
    history = pd.DataFrame(
        {
            "game_date": pd.Timestamp("2024-11-01"),
            "tier": ["<15", "<15"],
            "pts": [0, 2],
            "pts_ewm_hl_3": [1.2, 1.2],
        }
    )
    report = evaluate_points_pmfs(scored, pmf, pmf, history=history, n_boot=10, seed=0)
    line = report["lines"].loc[
        report["lines"]["tier"].eq("all") & report["lines"]["line"].eq("L0")
    ].iloc[0]
    assert line["n"] == 4
    assert line["observed"] == pytest.approx(0.75)
    assert line["p_over_b"] == pytest.approx(0.5)
    assert abs(line["gap_b"]) > OVER_RATE_TOLERANCE
    assert bool(line["flag"]) is False
    assert line["log_loss_b"] == pytest.approx(
        float(np.mean(binary_log_loss(np.full(4, 0.5), np.array([1, 1, 1, 0]))))
    )
    assert line["brier_b"] == pytest.approx(
        float(np.mean(binary_brier(np.full(4, 0.5), np.array([1, 1, 1, 0]))))
    )


def test_seeded_pit_uniform_repeats_and_stays_inside_the_unit_interval():
    first = seeded_unit_uniform([10, "10"], ["g", 3], seed=4, label="points_pit")
    second = seeded_unit_uniform([10, "10"], ["g", 3], seed=4, label="points_pit")
    other = seeded_unit_uniform([10, "10"], ["g", 3], seed=4, label="other")
    assert first.tolist() == pytest.approx(second.tolist())
    assert not np.allclose(first, other)
    assert np.all((first >= 0.0) & (first < 1.0))


def _frames():
    scored = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "player_id": [1, 2],
            "game_date": pd.Timestamp("2024-12-01"),
            "tier": ["<15", "31+"],
            "pts": [0, 2],
            "pts_ewm_hl_3": [1.2, 10.4],
            "pts_lag_1": [2.0, 9.0],
            "season_pts_mean": [1.5, 11.0],
        }
    )
    history = pd.DataFrame(
        {
            "game_date": pd.Timestamp("2024-11-01"),
            "tier": ["<15", "<15", "31+", "31+"],
            "pts": [0, 2, 12, 8],
            "pts_ewm_hl_3": [1.2, 1.2, 10.2, 10.2],
        }
    )
    return scored, history


def _pmfs():
    return np.array(
        [
            [0.5, 0.2, 0.2, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.4, 0.3],
        ]
    )
