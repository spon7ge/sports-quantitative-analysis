"""PIT pairs, empirical resampling, and the independence benchmarks."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from models.shared.copula import (
    MIN_TIER_PAIRS,
    EmpiricalCopula,
    IndependentCopula,
    build_pit_pairs,
    minutes_tier,
    pair_dependence,
    render_dependence_report,
)
from models.shared.minutes_sampler import QUANTILE_LEVELS, MinuteTailTables, cdf
from models.shared.ppm_sampler import RATE_CAP, RateTailTables, cdf_left

MINUTES_KNOTS = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float)
RATE_KNOTS = np.array(
    [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20],
    dtype=float,
)


def _minute_tables():
    return MinuteTailTables(
        arrays={
            ("lower", 0): np.array([0.2, 1.0]),
            ("upper", 0): np.array([0.0, 4.0]),
            ("lower", 1): np.array([0.5, 1.0]),
            ("upper", 1): np.array([0.0, 10.0]),
        },
        grouping={"lower": "starting", "upper": "starting"},
    )


def _rate_tables():
    return RateTailTables(
        arrays={
            ("lower", 0): np.array([0.0, 0.0, 0.0, 0.5, 1.0]),
            ("upper", 0): np.array([0.0, 0.4]),
            ("lower", 1): np.array([0.4, 1.0]),
            ("upper", 1): np.array([0.0, 0.5]),
        },
        grouping={"lower": "starting", "upper": "starting"},
    )


def _oos_row(
    *,
    game_id,
    player_id,
    minutes_knots,
    rate_knots,
    minutes,
    pts,
    game_date="2024-01-10",
    is_holdout=False,
):
    row = {
        "game_id": game_id,
        "player_id": player_id,
        "game_date": pd.Timestamp(game_date),
        "window_id": 1,
        "is_holdout": is_holdout,
        "minutes": minutes,
        "pts": pts,
    }
    for level, value in zip(QUANTILE_LEVELS, minutes_knots, strict=True):
        row[f"minutes_q_{level:.2f}"] = float(value)
    for level, value in zip(QUANTILE_LEVELS, rate_knots, strict=True):
        row[f"rate_q_{level:.2f}"] = float(value)
    return row


def _low_minute_knots():
    return np.array(
        [6, 7, 8, 9, 10, 12, 13, 13.5, 14, 14.5, 14.8],
        dtype=float,
    )


def _capped_rate_knots():
    knots = RATE_KNOTS.copy()
    knots[-1] = 7.5
    return knots


def test_build_pit_pairs_uses_sampler_cdfs_and_predicted_tier():
    low = _low_minute_knots()
    capped = _capped_rate_knots()
    frame = pd.DataFrame(
        [
            _oos_row(
                game_id="g-low",
                player_id=1,
                minutes_knots=low,
                rate_knots=RATE_KNOTS,
                minutes=40.0,
                pts=7,
            ),
            _oos_row(
                game_id="g-mid",
                player_id=2,
                minutes_knots=MINUTES_KNOTS,
                rate_knots=RATE_KNOTS,
                minutes=20.0,
                pts=14,
            ),
            _oos_row(
                game_id="g-cap",
                player_id=3,
                minutes_knots=MINUTES_KNOTS,
                rate_knots=capped,
                minutes=3.0,
                pts=30,
            ),
            _oos_row(
                game_id="g-tail",
                player_id=4,
                minutes_knots=MINUTES_KNOTS,
                rate_knots=RATE_KNOTS,
                minutes=7.5,
                pts=4,
            ),
        ]
    )
    starting = pd.DataFrame(
        {
            "game_id": ["g-low", "g-mid", "g-cap", "g-tail"],
            "player_id": [1, 2, 3, 4],
            "starting": [1, 1, 1, 1],
        }
    )
    pairs = build_pit_pairs(
        frame,
        minutes_tables=_minute_tables(),
        rate_tables=_rate_tables(),
        starting=starting,
    )
    again = build_pit_pairs(
        frame,
        minutes_tables=_minute_tables(),
        rate_tables=_rate_tables(),
        starting=starting,
    )
    pd.testing.assert_series_equal(pairs["u_m"], again["u_m"])
    pd.testing.assert_series_equal(pairs["u_r"], again["u_r"])

    # Realized 40 minutes would be 31+. The tier follows predicted q50.
    assert frame.loc[0, "minutes"] == 40
    assert pairs.loc[0, "tier"] == "<15"
    assert pairs.loc[1, "tier"] == "15-24"
    assert pairs.loc[1, "u_m"] == pytest.approx(0.5)
    assert pairs.loc[1, "u_r"] == pytest.approx(0.5)

    rate_grids = capped.reshape(1, -1)
    groups = np.array([1])
    left = cdf_left(np.array([RATE_CAP]), rate_grids, groups, groups, _rate_tables())
    assert left[0] < 1.0
    assert left[0] <= pairs.loc[2, "u_r"] < 1.0

    # 7.5 sits in the lower tail, between q05 * 0.5 and q05.
    assert 0.0 < pairs.loc[3, "u_m"] < 0.05
    minute_grids = MINUTES_KNOTS.reshape(1, -1)
    body = cdf(np.array([20.0]), minute_grids, groups, groups, _minute_tables())
    assert body[0] == pytest.approx(pairs.loc[1, "u_m"])


def test_minutes_tier_cuts_match_predicted_q50():
    q50 = np.array([14.9, 15.0, 23.9, 24.0, 30.9, 31.0])
    assert list(minutes_tier(q50)) == [
        "<15",
        "15-24",
        "15-24",
        "24-31",
        "24-31",
        "31+",
    ]


def test_independent_pairs_match_independence_benchmarks():
    rng = np.random.default_rng(0)
    n = 80_000
    pairs = pd.DataFrame({"u_m": rng.random(n), "u_r": rng.random(n)})
    stats = pair_dependence(pairs)
    assert stats["spearman"] == pytest.approx(0.0, abs=0.015)
    assert stats["lower_lower"] == pytest.approx(1.0, abs=0.15)
    assert stats["lower_upper"] == pytest.approx(1.0, abs=0.15)
    for decile in stats["deciles"].itertuples(index=False):
        assert decile.mean_u_r == pytest.approx(0.500, abs=0.015)
        assert decile.sd_u_r == pytest.approx(0.289, abs=0.015)
        assert decile.sd_u_r == pytest.approx(math.sqrt(1.0 / 12.0), abs=0.015)


def test_gaussian_copula_pairs_have_spearman_near_0_29():
    rho = 0.3
    rng = np.random.default_rng(1)
    n = 50_000
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    u = norm.cdf(z)
    stats = pair_dependence(pd.DataFrame({"u_m": u[:, 0], "u_r": u[:, 1]}))
    # (6/pi) * arcsin(rho/2) ≈ 0.288.
    assert stats["spearman"] == pytest.approx(0.29, abs=0.015)


def _stored_pairs():
    return pd.DataFrame(
        {
            "game_id": ["early", "cutoff", "later", "held"],
            "player_id": [1, 2, 3, 4],
            "game_date": pd.to_datetime(
                ["2024-01-01", "2024-03-01", "2024-06-01", "2024-02-01"]
            ),
            "is_holdout": [False, False, False, True],
            "tier": ["<15", "<15", "<15", "<15"],
            "u_m": [0.21, 0.22, 0.23, 0.24],
            "u_r": [0.31, 0.32, 0.33, 0.34],
        }
    )


def test_no_pair_on_or_after_before_date_is_returned():
    cutoff = pd.Timestamp("2024-03-01")
    copula = EmpiricalCopula(_stored_pairs(), cutoff)
    assert list(copula.pairs["game_id"]) == ["early"]
    drawn, pooled = copula.sample("<15", 40, np.random.default_rng(2))
    assert pooled
    assert (pd.to_datetime(drawn["game_date"]) < cutoff).all()
    assert set(drawn["game_id"]) == {"early"}


def test_holdout_pairs_are_excluded_by_default():
    pairs = _stored_pairs()
    copula = EmpiricalCopula(pairs, "2024-05-01")
    assert set(copula.pairs["game_id"]) == {"early", "cutoff"}
    assert not copula.pairs["is_holdout"].any()
    drawn, _pooled = copula.sample("<15", 40, np.random.default_rng(3))
    assert set(drawn["game_id"]) <= {"early", "cutoff"}
    included = EmpiricalCopula(pairs, "2024-05-01", include_holdout=True)
    assert set(included.pairs["game_id"]) == {"early", "cutoff", "held"}
    held_draw, _pooled = included.sample("<15", 80, np.random.default_rng(3))
    assert "held" in set(held_draw["game_id"])
    assert "later" not in set(held_draw["game_id"])


def test_sample_returns_stored_uniforms_without_jitter():
    pairs = _stored_pairs().iloc[:1].copy()
    extra = pd.concat([pairs] * 20, ignore_index=True)
    extra["u_m"] = np.linspace(0.11, 0.83, len(extra))
    extra["u_r"] = np.linspace(0.17, 0.71, len(extra))
    extra["game_date"] = pd.Timestamp("2024-01-01")
    extra["is_holdout"] = False
    copula = EmpiricalCopula(extra, "2024-02-01")
    drawn, _pooled = copula.sample("<15", 100, np.random.default_rng(4))
    assert np.isin(drawn["u_m"].to_numpy(), copula.pairs["u_m"].to_numpy()).all()
    assert np.isin(drawn["u_r"].to_numpy(), copula.pairs["u_r"].to_numpy()).all()


def test_small_tier_pools_all_tiers_and_flags():
    small = pd.DataFrame(
        {
            "game_id": ["s0", "s1"],
            "player_id": [1, 2],
            "game_date": pd.Timestamp("2024-01-01"),
            "is_holdout": False,
            "tier": "<15",
            "u_m": [0.11, 0.12],
            "u_r": [0.21, 0.22],
        }
    )
    other = pd.DataFrame(
        {
            "game_id": [f"o{i}" for i in range(5)],
            "player_id": np.arange(5),
            "game_date": pd.Timestamp("2024-01-02"),
            "is_holdout": False,
            "tier": "31+",
            "u_m": np.full(5, 0.77),
            "u_r": np.full(5, 0.66),
        }
    )
    copula = EmpiricalCopula(pd.concat([small, other], ignore_index=True), "2025-01-01")
    drawn, pooled = copula.sample("<15", 200, np.random.default_rng(5))
    assert pooled
    assert set(drawn["u_m"]) == {0.11, 0.12, 0.77}


def test_tier_with_3000_pairs_is_not_pooled():
    n = MIN_TIER_PAIRS
    frame = pd.DataFrame(
        {
            "game_id": np.arange(n).astype(str),
            "player_id": np.arange(n),
            "game_date": pd.Timestamp("2024-01-01"),
            "is_holdout": False,
            "tier": "<15",
            "u_m": np.linspace(0.01, 0.99, n),
            "u_r": np.linspace(0.02, 0.98, n),
        }
    )
    other = frame.iloc[:10].copy()
    other["tier"] = "31+"
    other["u_m"] = 0.42
    copula = EmpiricalCopula(pd.concat([frame, other], ignore_index=True), "2025-01-01")
    drawn, pooled = copula.sample("<15", 30, np.random.default_rng(6))
    assert pooled is False
    assert set(drawn["tier"]) == {"<15"}
    short = frame.iloc[: MIN_TIER_PAIRS - 1]
    pooled_copula = EmpiricalCopula(short, "2025-01-01")
    _drawn, pooled = pooled_copula.sample("<15", 5, np.random.default_rng(7))
    assert pooled is True


def test_independent_copula_draws_fresh_uniforms():
    stored = _stored_pairs()
    copula = IndependentCopula(stored, "2024-05-01")
    assert set(copula.pairs["game_id"]) == {"early", "cutoff"}
    assert not copula.pairs["is_holdout"].any()
    drawn, pooled = copula.sample("15-24", 400, np.random.default_rng(8))
    assert pooled is False
    assert len(drawn) == 400
    assert set(drawn["tier"]) == {"15-24"}
    assert not np.allclose(drawn["u_m"], 0.21)
    assert drawn["u_m"].between(0, 1).all()
    assert drawn["u_r"].between(0, 1).all()


def test_report_describes_preholdout_pairs_only():
    pre = pd.DataFrame(
        {
            "is_holdout": False,
            "tier": "<15",
            "u_m": [0.2, 0.4, 0.6],
            "u_r": [0.3, 0.5, 0.7],
        }
    )
    held = pre.copy()
    held["is_holdout"] = True
    text = render_dependence_report(pd.concat([pre, held], ignore_index=True))
    assert "not a calibration slice of the points PMF" in text
    assert "| <15 | 3 |" in text


def test_u_m_of_one_lands_in_the_last_decile():
    stats = pair_dependence(pd.DataFrame({"u_m": [0.0, 1.0], "u_r": [0.2, 0.4]}))
    counts = stats["deciles"].set_index("decile")["n"]
    assert counts.loc[1] == 1
    assert counts.loc[10] == 1
