"""Tests for empirical-Bayes points mean model with baseline-linked variance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor, cho_solve

from src.models.points_empirical_bayes import (
    CONTEXT_COLS,
    PRIOR_SD_RAW,
    fit_points_model,
    observation_variance,
    predict_points,
    shared_noise_posterior_moments,
)


REQUIRED_CONTEXT = (
    "starter_surprise",
    "is_home",
    "back2back",
    "opp_def_rating_prior",
    "team_spread_pregame",
    "implied_team_total_pregame",
    "usage_trend",
    "minutes_trend",
    "fga_trend",
    "fta_trend",
)


def _synthetic_train(
    n_players: int = 8,
    games: int = 12,
    seed: int = 0,
    context_cols: tuple[str, ...] = REQUIRED_CONTEXT,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for player in range(n_players):
        level = float(rng.normal(0.0, 1.5))
        for game in range(games):
            p5 = float(rng.uniform(5.0, 30.0))
            p10 = float(rng.uniform(5.0, 30.0))
            p20 = float(rng.uniform(5.0, 30.0))
            features = {
                "starter_surprise": float(rng.choice([-0.5, 0.0, 0.5])),
                "is_home": float(rng.integers(0, 2)),
                "back2back": float(rng.integers(0, 2)),
                "opp_def_rating_prior": float(rng.normal(112.0, 3.0)),
                "team_spread_pregame": float(rng.normal(0.0, 6.0)),
                "implied_team_total_pregame": float(rng.normal(114.0, 5.0)),
                "usage_trend": float(rng.normal(0.0, 2.0)),
                "minutes_trend": float(rng.normal(0.0, 3.0)),
                "fga_trend": float(rng.normal(0.0, 2.0)),
                "fta_trend": float(rng.normal(0.0, 1.0)),
            }
            baseline = 0.4 * p5 + 0.3 * p10 + 0.3 * p20
            mean = (
                baseline
                + level
                + 0.15 * features["usage_trend"]
                + 0.05 * features["minutes_trend"]
                + 1.2 * features["starter_surprise"]
                + 0.04 * features["is_home"]
                - 0.3 * features["back2back"]
            )
            noise = float(rng.normal(0.0, np.sqrt(8.0 + 0.15 * baseline)))
            row: dict[str, float | int] = {
                "player_id": player + 1,
                "pts": mean + noise,
                "p5": p5,
                "p10": p10,
                "p20": p20,
                "start_rate": 0.5,
            }
            for column in context_cols:
                row[column] = features[column]
            rows.append(row)
    return pd.DataFrame(rows)


def test_default_context_and_prior_keys_match() -> None:
    assert CONTEXT_COLS == REQUIRED_CONTEXT
    assert set(PRIOR_SD_RAW) >= {"level", *REQUIRED_CONTEXT, "game_total_pregame"}


def test_gamma1_zero_observation_variance_is_shared() -> None:
    baseline = np.array([2.0, 10.0, 25.0])
    sigma2 = observation_variance(
        baseline,
        gamma0=np.log(36.0),
        gamma1=0.0,
        log_b_center=2.0,
    )
    assert sigma2 == pytest.approx(np.full(3, 36.0))


def test_gamma1_zero_recovers_shared_posterior_moments() -> None:
    train = _synthetic_train()
    shared = fit_points_model(train, variance_mode="shared")

    matched = dict(shared)
    matched["gamma1"] = 0.0
    matched["variance_mode"] = "baseline"

    assert observation_variance(
        np.array([1.0, 12.0, 28.0]),
        shared["gamma0"],
        0.0,
        shared["log_b_center"],
    ) == pytest.approx(np.full(3, shared["noise_var"]))

    pred_shared = predict_points(train.head(5), shared)
    pred_matched = predict_points(train.head(5), matched)
    assert pred_matched["mean"] == pytest.approx(
        pred_shared["mean"], rel=1e-12, abs=1e-12
    )
    assert pred_matched["variance"] == pytest.approx(
        pred_shared["variance"], rel=1e-12, abs=1e-12
    )
    assert pred_matched["observation_variance"] == pytest.approx(
        np.full(5, shared["noise_var"]), rel=1e-12, abs=1e-12
    )

    refit = fit_points_model(
        train,
        variance_mode="baseline",
        fixed_gamma1=0.0,
        init_gamma0=np.log(shared["noise_var"]),
        init_player_var=max(float(shared["player_var"]), 1e-8),
    )
    assert refit["gamma1"] == pytest.approx(0.0)
    assert refit["nll"] == pytest.approx(shared["nll"], rel=1e-8, abs=1e-6)
    assert refit["noise_var"] == pytest.approx(shared["noise_var"], rel=1e-5, abs=1e-6)
    assert refit["beta"] == pytest.approx(shared["beta"], rel=1e-5, abs=1e-6)


def test_unequal_variance_posterior_matches_direct_gaussian() -> None:
    """Small direct Gaussian with diag observation noise and player intercepts."""
    rng = np.random.default_rng(7)
    n_players = 3
    games = 4
    p = 2
    design = rng.normal(size=(n_players * games, p))
    codes = np.repeat(np.arange(n_players), games)
    target = rng.normal(size=n_players * games)
    sigma2 = np.exp(rng.uniform(1.0, 3.0, size=n_players * games))
    player_var = 1.7
    prior_sd = np.array([4.0, 2.5])

    moments = shared_noise_posterior_moments(
        design=design,
        target=target,
        codes=codes,
        sigma2=sigma2,
        player_var=player_var,
        prior_sd=prior_sd,
    )

    n_obs = len(target)
    dim = p + n_players
    precision = np.zeros((dim, dim))
    precision[:p, :p] = np.diag(1.0 / prior_sd**2)
    for player in range(n_players):
        precision[p + player, p + player] = 1.0 / player_var

    rhs = np.zeros(dim)
    for i in range(n_obs):
        weight = 1.0 / sigma2[i]
        row = np.zeros(dim)
        row[:p] = design[i]
        row[p + codes[i]] = 1.0
        precision += weight * np.outer(row, row)
        rhs += weight * target[i] * row

    chol = cho_factor(precision, lower=True)
    mean = cho_solve(chol, rhs)
    cov = cho_solve(chol, np.eye(dim))

    assert moments["beta"] == pytest.approx(mean[:p], rel=1e-10, abs=1e-10)
    assert moments["beta_cov"] == pytest.approx(cov[:p, :p], rel=1e-10, abs=1e-10)
    assert moments["intercept"] == pytest.approx(mean[p:], rel=1e-10, abs=1e-10)
    assert moments["intercept_marginal_var"] == pytest.approx(
        np.diag(cov[p:, p:]), rel=1e-10, abs=1e-10
    )


def test_baseline_mode_fits_nonzero_gamma1_on_heteroscedastic_data() -> None:
    train = _synthetic_train(n_players=12, games=15, seed=3)
    fitted = fit_points_model(train, variance_mode="baseline")
    assert fitted["optimizer_success"] is True
    assert np.isfinite(fitted["gamma0"])
    assert np.isfinite(fitted["gamma1"])
    assert fitted["gamma1"] != pytest.approx(0.0, abs=1e-3)
    assert fitted["player_var"] >= 0.0


def test_prior_sd_uses_raw_values_times_feature_scale() -> None:
    train = _synthetic_train()
    fitted = fit_points_model(train, variance_mode="shared")
    expected = np.array(
        [
            PRIOR_SD_RAW["level"],
            *[
                PRIOR_SD_RAW[column] * fitted["scale"][index]
                for index, column in enumerate(CONTEXT_COLS)
            ],
        ]
    )
    assert fitted["prior_sd"] == pytest.approx(expected)
    assert fitted["prior_precision"] == pytest.approx(
        np.diag(1.0 / expected**2)
    )


def test_prior_sd_scale_multiplies_all_coefficient_priors() -> None:
    train = _synthetic_train()
    base = fit_points_model(train, variance_mode="shared", prior_sd_scale=1.0)
    tight = fit_points_model(train, variance_mode="shared", prior_sd_scale=0.5)
    assert tight["prior_sd"] == pytest.approx(0.5 * base["prior_sd"])


def test_standardized_and_original_unit_fits_match_predictions() -> None:
    train = _synthetic_train(seed=11)
    standardized = fit_points_model(
        train, variance_mode="shared", standardize=True
    )
    # Refit original-unit design at the same EB variance plugs so only the
    # design/prior reparameterization differs.
    original = fit_points_model(
        train,
        variance_mode="shared",
        standardize=False,
        init_gamma0=np.log(standardized["noise_var"]),
        init_player_var=max(float(standardized["player_var"]), 1e-8),
        fixed_gamma1=0.0,
    )
    # Shared mode already fixes gamma1; force matching variance plugs via
    # one-shot objective minimum should land at the same noise/player values.
    assert original["noise_var"] == pytest.approx(
        standardized["noise_var"], rel=1e-5, abs=1e-6
    )
    assert original["player_var"] == pytest.approx(
        standardized["player_var"], rel=1e-5, abs=1e-6
    )
    pred_std = predict_points(train.head(8), standardized)
    pred_raw = predict_points(train.head(8), original)
    assert pred_std["mean"] == pytest.approx(pred_raw["mean"], rel=1e-8, abs=1e-8)
    assert pred_std["variance"] == pytest.approx(
        pred_raw["variance"], rel=1e-8, abs=1e-8
    )
    conversion = np.r_[1.0, standardized["scale"]]
    assert standardized["beta"] / conversion == pytest.approx(
        original["beta"], rel=1e-8, abs=1e-8
    )
    assert standardized["prior_sd"] == pytest.approx(
        original["prior_sd"] * conversion, rel=1e-12, abs=1e-12
    )


def test_context_cols_subset_is_respected() -> None:
    cols = (
        "starter_surprise",
        "is_home",
        "back2back",
        "opp_def_rating_prior",
        "team_spread_pregame",
        "implied_team_total_pregame",
        "usage_trend",
        "minutes_trend",
    )
    train = _synthetic_train(context_cols=cols)
    fitted = fit_points_model(
        train, variance_mode="shared", context_cols=cols
    )
    assert fitted["context_cols"] == cols
    assert len(fitted["beta"]) == 1 + len(cols)
    pred = predict_points(train.head(3), fitted)
    assert np.isfinite(pred["mean"]).all()


def test_player_var_may_approach_zero() -> None:
    train = _synthetic_train(n_players=6, games=10, seed=21)
    # Strongly shrink player effects by starting near the lower bound.
    fitted = fit_points_model(
        train,
        variance_mode="shared",
        init_player_var=1e-8,
    )
    assert fitted["player_var"] >= 0.0
    assert np.isfinite(fitted["player_var"])
