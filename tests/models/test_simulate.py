"""Integer points PMF from joint minutes and rate draws."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, spearmanr

from models.shared.copula import CopulaSample, IndependentCopula
from models.shared.metrics import integer_log_score, randomized_pit, rps
from models.shared.minutes_sampler import (
    QUANTILE_LEVELS,
    MinuteTailTables,
    ppf as minutes_ppf,
)
from models.shared.ppm_sampler import RateTailTables, ppf as rate_ppf
from models.shared.simulate import line_probs, points_pmf

LEVELS = np.asarray(QUANTILE_LEVELS, dtype=float)


def _flat_tables(kind):
    arrays = {
        ("lower", 0): np.array([1.0, 1.0]),
        ("upper", 0): np.array([0.0, 0.0]),
        ("lower", 1): np.array([1.0, 1.0]),
        ("upper", 1): np.array([0.0, 0.0]),
    }
    grouping = {"lower": "starting", "upper": "starting"}
    if kind == "minutes":
        return MinuteTailTables(arrays=arrays, grouping=grouping)
    return RateTailTables(arrays=arrays, grouping=grouping)


def _spread_tables():
    minutes = MinuteTailTables(
        arrays={
            ("lower", 0): np.array([0.2, 1.0]),
            ("upper", 0): np.array([0.0, 4.0]),
            ("lower", 1): np.array([0.5, 1.0]),
            ("upper", 1): np.array([0.0, 8.0]),
        },
        grouping={"lower": "starting", "upper": "starting"},
    )
    rate = RateTailTables(
        arrays={
            ("lower", 0): np.array([0.4, 1.0]),
            ("upper", 0): np.array([0.0, 0.3]),
            ("lower", 1): np.array([0.4, 1.0]),
            ("upper", 1): np.array([0.0, 0.4]),
        },
        grouping={"lower": "starting", "upper": "starting"},
    )
    return minutes, rate


def _knots(value):
    return np.full(LEVELS.shape, float(value))


def _row(game_id, player_id, minutes, rate, starting=1):
    row = {
        "game_id": game_id,
        "player_id": player_id,
        "starting": starting,
    }
    for level, minute, scoring_rate in zip(LEVELS, minutes, rate, strict=True):
        row[f"minutes_q_{level:.2f}"] = float(minute)
        row[f"rate_q_{level:.2f}"] = float(scoring_rate)
    return row


def _independent():
    pairs = pd.DataFrame(
        {
            "u_m": [0.4],
            "u_r": [0.6],
            "tier": ["15-24"],
            "game_date": [pd.Timestamp("2024-01-01")],
            "is_holdout": [False],
        }
    )
    return IndependentCopula(pairs, before_date="2025-10-21")


class GaussianCopula:
    def __init__(self, rho):
        self.rho = float(rho)

    def sample(self, tier, n, rng):
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        z = rng.multivariate_normal([0.0, 0.0], cov, size=int(n))
        uniforms = norm.cdf(z)
        pairs = pd.DataFrame(
            {"u_m": uniforms[:, 0], "u_r": uniforms[:, 1], "tier": tier}
        )
        return CopulaSample(pairs, False)


class SharedUniformCopula:
    def sample(self, tier, n, rng):
        uniform = rng.random(int(n))
        pairs = pd.DataFrame({"u_m": uniform, "u_r": uniform.copy(), "tier": tier})
        return CopulaSample(pairs, False)


def _capture(monkeypatch):
    captured = {}

    def spy_minutes(u, grids, lower, upper, tables):
        out = minutes_ppf(u, grids, lower, upper, tables)
        captured["minutes"] = np.asarray(out, dtype=float).copy()
        return out

    def spy_rate(u, grids, lower, upper, tables):
        out = rate_ppf(u, grids, lower, upper, tables)
        captured["rate"] = np.asarray(out, dtype=float).copy()
        return out

    monkeypatch.setattr("models.shared.simulate.minutes_ppf", spy_minutes)
    monkeypatch.setattr("models.shared.simulate.rate_ppf", spy_rate)
    return captured


def _monotone_row():
    minutes = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float)
    rate = np.array(
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
    )
    return pd.DataFrame([_row("g-joint", 7, minutes, rate)])


def test_half_point_over_is_the_mass_at_or_above_the_next_integer():
    pmf = np.zeros(31)
    pmf[24] = 0.2
    pmf[25] = 0.5
    pmf[26] = 0.3
    over, push, under = line_probs(pmf, 24.5)
    assert push == 0.0
    assert over == pytest.approx(pmf[25:].sum())
    assert over == pytest.approx(0.8)
    assert under == pytest.approx(pmf[:25].sum())
    assert over + push + under == pytest.approx(1.0)


def test_whole_number_line_pushes_on_that_integer():
    pmf = np.zeros(31)
    pmf[23] = 0.25
    pmf[24] = 0.40
    pmf[25] = 0.35
    over, push, under = line_probs(pmf, 24)
    assert push == pytest.approx(pmf[24])
    assert over == pytest.approx(pmf[25:].sum())
    assert under == pytest.approx(pmf[:24].sum())
    assert over + push + under == pytest.approx(1.0)
    over_half, push_half, _under_half = line_probs(pmf, 24.5)
    assert push_half == 0.0
    assert over_half == pytest.approx(pmf[25:].sum())
    assert over_half == pytest.approx(over)


def test_reruns_with_the_same_ids_are_identical():
    minutes, rate = _spread_tables()
    rows = pd.DataFrame(
        [
            _row("g1", 10, _knots(22), np.linspace(0.3, 1.1, 11)),
            _row("g1", 11, _knots(28), np.linspace(0.4, 1.4, 11)),
            _row("g2", 10, np.linspace(32, 42, 11), _knots(0.8)),
        ]
    )
    kwargs = dict(
        minutes_tables=minutes,
        rate_tables=rate,
        n_draws=400,
        kmax=60,
    )
    first = points_pmf(rows, _independent(), **kwargs)
    second = points_pmf(rows, _independent(), **kwargs)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])

    knots_m = np.linspace(18, 30, 11)
    knots_r = np.linspace(0.4, 1.2, 11)
    twins = pd.DataFrame(
        [
            _row("same-game", 1, knots_m, knots_r),
            _row("same-game", 2, knots_m, knots_r),
        ]
    )
    twin_pmf, _, _ = points_pmf(twins, _independent(), **kwargs)
    assert not np.array_equal(twin_pmf[0], twin_pmf[1])
    duplicate = pd.DataFrame(
        [
            _row("same-game", 1, knots_m, knots_r),
            _row("same-game", 1, knots_m, knots_r),
        ]
    )
    duplicated, _, _ = points_pmf(duplicate, _independent(), **kwargs)
    np.testing.assert_array_equal(duplicated[0], duplicated[1])
    np.testing.assert_array_equal(duplicated[0], twin_pmf[0])


def test_constant_knots_are_a_point_mass_at_round_minutes_times_rate():
    rows = pd.DataFrame(
        [
            _row("g-a", 1, _knots(20), _knots(0.5)),
            _row("g-b", 2, _knots(16), _knots(1.0)),
            _row("g-c", 3, _knots(36), _knots(0.5)),
            _row("g-d", 4, _knots(10), _knots(8.0)),
        ]
    )
    pmf, mean, median = points_pmf(
        rows,
        _independent(),
        n_draws=200,
        minutes_tables=_flat_tables("minutes"),
        rate_tables=_flat_tables("rate"),
    )
    # A rate knot of 8 is past the training cap, so the draw uses 6.
    expected = [round(20 * 0.5), round(16 * 1.0), round(36 * 0.5), round(10 * 6.0)]
    assert pmf.shape == (4, 91)
    for index, point in enumerate(expected):
        assert pmf[index].sum() == pytest.approx(1.0)
        assert pmf[index, point] == pytest.approx(1.0)
        assert mean[index] == pytest.approx(point)
        assert median[index] == point
    assert rows.loc[0, "minutes_q_0.50"] < 24
    assert rows.loc[2, "minutes_q_0.50"] >= 31


def test_pmf_mean_is_not_the_product_of_q50s_when_the_row_is_skewed():
    minutes = np.array([1, 1.5, 2, 2.5, 3, 4, 12, 22, 34, 48, 58], dtype=float)
    rate = _knots(1.0)
    rows = pd.DataFrame([_row("g-skew", 4, minutes, rate)])
    _pmf, mean, _median = points_pmf(
        rows,
        _independent(),
        n_draws=4_000,
        minutes_tables=_flat_tables("minutes"),
        rate_tables=_flat_tables("rate"),
    )
    q50_product = float(rows.loc[0, "minutes_q_0.50"] * rows.loc[0, "rate_q_0.50"])
    assert q50_product == pytest.approx(4.0)
    assert abs(float(mean[0]) - q50_product) > 1.0


def test_gaussian_copula_passes_dependence_through_the_quantile_functions(monkeypatch):
    captured = _capture(monkeypatch)
    minutes_tables, rate_tables = _spread_tables()
    rows = _monotone_row()
    points_pmf(
        rows,
        GaussianCopula(0.3),
        n_draws=10_000,
        minutes_tables=minutes_tables,
        rate_tables=rate_tables,
    )
    rho = float(spearmanr(captured["minutes"][0], captured["rate"][0]).statistic)
    expected = (6.0 / math.pi) * math.asin(0.3 / 2.0)
    assert rho == pytest.approx(expected, abs=0.03)
    assert rho == pytest.approx(0.29, abs=0.03)


def test_independent_copula_draws_have_no_rank_correlation(monkeypatch):
    captured = _capture(monkeypatch)
    minutes_tables, rate_tables = _spread_tables()
    points_pmf(
        _monotone_row(),
        _independent(),
        n_draws=10_000,
        minutes_tables=minutes_tables,
        rate_tables=rate_tables,
    )
    rho = float(spearmanr(captured["minutes"][0], captured["rate"][0]).statistic)
    assert rho == pytest.approx(0.0, abs=0.05)


def test_shared_uniforms_keep_perfect_rank_correlation(monkeypatch):
    captured = _capture(monkeypatch)
    minutes_tables, rate_tables = _spread_tables()
    points_pmf(
        _monotone_row(),
        SharedUniformCopula(),
        n_draws=2_000,
        minutes_tables=minutes_tables,
        rate_tables=rate_tables,
    )
    rho = float(spearmanr(captured["minutes"][0], captured["rate"][0]).statistic)
    assert rho == pytest.approx(1.0, abs=1e-6)


def test_integer_log_score_is_the_log_probability_of_the_outcome():
    pmf = np.zeros(8)
    pmf[4] = 0.25
    pmf[5] = 0.75
    assert integer_log_score(pmf, 5) == pytest.approx(math.log(0.75))
    point = np.zeros(8)
    point[3] = 1.0
    assert integer_log_score(point, 3) == pytest.approx(0.0)


def test_rps_is_the_squared_cdf_error():
    point = np.zeros(6)
    point[3] = 1.0
    assert rps(point, 3) == pytest.approx(0.0)
    spread = np.array([0.25, 0.25, 0.25, 0.25])
    # F = [0.25, 0.5, 0.75, 1], y = 1
    # squares: 0.25^2 + 0.5^2 + 0.25^2 + 0^2
    assert rps(spread, 1) == pytest.approx(0.0625 + 0.25 + 0.0625)


def test_randomized_pit_is_uniform_inside_the_outcome_atom():
    pmf = np.array([0.2, 0.8])
    assert randomized_pit(pmf, 0, 0.25) == pytest.approx(0.05)
    point = np.array([0.0, 1.0])
    assert randomized_pit(point, 1, 0.4) == pytest.approx(0.4)
    assert randomized_pit(point, 1, np.array([0.2, 0.8])).shape == (2,)
