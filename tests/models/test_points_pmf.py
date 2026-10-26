"""Grid points PMF: conditional PPM quantiles, copula map, integer histogram."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from models.shared.minutes_sampler import QUANTILE_LEVELS

LEVELS = np.asarray(QUANTILE_LEVELS, dtype=float)


def _tails():
    from models.shared.points_pmf import PpmTails

    return PpmTails(
        arrays={
            ("lower", 0): np.array([0.5, 1.0]),
            ("lower", 1): np.array([0.25, 1.0]),
            ("upper", 0): np.array([0.0, 0.4]),
            ("upper", 1): np.array([0.0, 0.2]),
        }
    )


def test_flat_zero_knots_keep_probability_at_zero():
    from models.shared.points_pmf import PpmTails, ppm_quantile_function

    knots = np.zeros((1, 11))
    tails = PpmTails(
        arrays={
            ("lower", 1): np.array([1.0]),
            ("upper", 1): np.array([0.0, 0.0]),
        }
    )
    u = np.array([[0.0, 0.05, 0.40, 0.95, 1.0]])
    got = ppm_quantile_function(u, knots, LEVELS, tails, np.array([1]))
    np.testing.assert_allclose(got, 0.0)


def test_flat_prefix_of_zeros_ends_at_the_last_zero_knot():
    """Knots through 0.40 are 0, so Q(u) = 0 on [0, 0.40] and rises after."""
    from models.shared.points_pmf import ppm_quantile_function

    knots = np.array([[0, 0, 0, 0, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    u = np.array([[0.0, 0.40, 0.45, 0.50]])
    got = ppm_quantile_function(u, knots, LEVELS, _tails(), np.array([1]))
    np.testing.assert_allclose(got[0, :2], 0.0)
    assert got[0, 2] > 0
    np.testing.assert_allclose(got[0, 3], 0.5)


def test_knot_levels_return_sorted_knots():
    from models.shared.points_pmf import ppm_quantile_function

    raw = np.array([0.9, 0.0, 0.2, 0.1, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0])
    u = LEVELS.reshape(1, -1)
    got = ppm_quantile_function(u, raw.reshape(1, -1), LEVELS, _tails(), np.array([0]))
    np.testing.assert_allclose(got[0], np.sort(raw))


def test_lower_tail_scales_q05_by_the_empirical_ratio():
    from models.shared.points_pmf import ppm_quantile_function

    knots = np.linspace(0.40, 1.40, 11).reshape(1, -1)
    u = np.array([[0.0, 0.05]])
    got = ppm_quantile_function(u, knots, LEVELS, _tails(), np.array([1]))
    # Group 1 lower table is [0.25, 1]. u=0 reads the first entry.
    np.testing.assert_allclose(got[0, 0], 0.40 * 0.25)
    np.testing.assert_allclose(got[0, 1], 0.40)


def test_points_pmf_rows_sum_to_one():
    from models.shared.points_pmf import points_pmf

    minutes = np.array([[10.0, 22.0, 35.0], [0.0, 8.0, 16.0]])
    ppm = np.array(
        [
            [[0.2, 0.5, 1.1], [0.0, 0.4, 0.9], [0.3, 0.7, 1.4]],
            [[0.0, 1.0, 2.0], [-0.2, 0.5, 3.0], [0.1, 0.1, 0.8]],
        ]
    )
    for rounding in ("rint", "floor"):
        pmf = points_pmf(minutes, ppm, k_max=40, rounding=rounding)
        assert pmf.shape == (2, 41)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0)


def test_points_pmf_degenerate_half_point_split():
    """Every cell is 12.5. rint puts it in 13; floor leaves it in 12."""
    from models.shared.points_pmf import points_pmf

    minutes = np.full((1, 4), 10.0)
    ppm = np.full((1, 4, 2), 1.25)
    rint = points_pmf(minutes, ppm, k_max=20, rounding="rint")
    floor = points_pmf(minutes, ppm, k_max=20, rounding="floor")
    assert rint[0, 13] == pytest.approx(1.0)
    assert floor[0, 12] == pytest.approx(1.0)
    assert rint[0, 12] == pytest.approx(0.0)
    assert floor[0, 13] == pytest.approx(0.0)


def test_points_pmf_clips_to_the_support():
    from models.shared.points_pmf import points_pmf

    minutes = np.array([[10.0, 10.0]])
    ppm = np.array([[[20.0], [-1.0]]])
    pmf = points_pmf(minutes, ppm, k_max=5, rounding="floor")
    assert pmf[0, 5] == pytest.approx(0.5)
    assert pmf[0, 0] == pytest.approx(0.5)


def test_prediction_frame_sets_minutes_given_from_the_grid():
    from models.shared.points_pmf import ppm_prediction_frame

    frame = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "starting": [1, 0],
            "minutes": [33.0, 12.0],
            "pts_per_min_ewm_hl_10": [0.4, 0.2],
        }
    )
    grid = np.array([[10.0, 20.0], [5.0, 15.0]])
    batched = ppm_prediction_frame(
        frame,
        grid,
        features=["starting", "pts_per_min_ewm_hl_10", "minutes_given"],
    )
    assert len(batched) == 4
    np.testing.assert_allclose(batched["minutes_given"], [10.0, 20.0, 5.0, 15.0])
    assert list(batched["player_id"]) == ["a", "a", "b", "b"]
    assert "minutes" not in batched.columns or (
        batched["minutes_given"].to_numpy() != batched["minutes"].to_numpy()
    ).any()


def test_gaussian_copula_independent_and_comonotonic():
    from models.shared.points_pmf import gaussian_copula_uniform

    u_m = np.array([0.2, 0.4, 0.8])
    u_p = np.array([0.1, 0.7, 0.3])
    independent = gaussian_copula_uniform(u_m, u_p, rho=0.0)
    np.testing.assert_allclose(independent, u_p)
    comonotonic = gaussian_copula_uniform(u_m, u_p, rho=1.0)
    np.testing.assert_allclose(comonotonic, u_m)
    mid = gaussian_copula_uniform(np.array([0.5]), np.array([0.5]), rho=0.6)
    expected = norm.cdf(0.6 * norm.ppf(0.5) + np.sqrt(1 - 0.6**2) * norm.ppf(0.5))
    np.testing.assert_allclose(mid, expected)


def test_feature_list_rejects_game_outcomes_and_requires_minutes_given():
    from models.shared.points_pmf import check_feature_list

    check_feature_list(["starting", "minutes_given"], conditional=True)
    check_feature_list(["starting", "fga_per_min_10"], conditional=False)
    with pytest.raises(ValueError, match="minutes"):
        check_feature_list(["minutes", "minutes_given"], conditional=True)
    with pytest.raises(ValueError, match="minutes_given"):
        check_feature_list(["starting"], conditional=True)
    with pytest.raises(ValueError, match="minutes_given"):
        check_feature_list(["starting", "minutes_given"], conditional=False)


def test_build_ppm_tails_accepts_a_later_holdout_start():
    from models.shared.points_pmf import build_ppm_tails

    levels = [f"q_{level:.2f}" for level in LEVELS]
    row = {column: 0.4 for column in levels}
    row.update(
        {
            "q_0.05": 0.20,
            "q_0.95": 0.40,
            "ppm": 0.10,
            "starting": 1,
            "fold_id": 1,
            "game_date": "2025-10-21",
        }
    )
    build_ppm_tails(pd.DataFrame([row, row]), folds=(1,), holdout_start="2026-10-20")


def test_build_ppm_tails_rejects_holdout_dates_and_uses_ratios():
    from models.shared.points_pmf import build_ppm_tails

    levels = [f"q_{level:.2f}" for level in LEVELS]
    row = {column: 0.4 for column in levels}
    row.update(
        {
            "q_0.05": 0.20,
            "q_0.95": 0.40,
            "ppm": 0.10,
            "starting": 1,
            "fold_id": 1,
            "game_date": "2024-01-01",
        }
    )
    holdout = dict(row)
    holdout["game_date"] = "2026-10-20"
    holdout["ppm"] = 0.05
    oof = pd.DataFrame([row, row])
    oof.loc[1, "ppm"] = 0.90
    with pytest.raises(ValueError):
        build_ppm_tails(pd.DataFrame([holdout]), folds=(1,))
    tails = build_ppm_tails(oof, folds=(1,))
    np.testing.assert_allclose(tails.arrays[("lower", 1)], [0.10 / 0.20, 1.0])
    np.testing.assert_allclose(tails.arrays[("upper", 1)], [0.0, 0.90 - 0.40])
