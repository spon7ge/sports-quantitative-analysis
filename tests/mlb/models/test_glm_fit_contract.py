"""Fail-closed GLM fitting, usable-feature selection, and rest encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.mlb.models.preprocess import (
    BINARY_STRIKEOUT_FEATURES,
    encode_rest_features,
    prepare_strikeout_frame,
    select_usable_features,
)
from src.mlb.models.strikeouts import fit_strikeouts, predict_strikeout_pmf
from src.mlb.models.workload import GlmFitError, design_matrix, fit_nb2
from src.mlb.schemas import STRIKEOUT_FEATURE_COLUMNS


def test_singular_design_raises_glm_fit_error() -> None:
    y = np.array([3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 8.0, 4.0])
    x = np.column_stack(
        [
            np.ones(len(y)),
            np.zeros(len(y)),
            np.arange(len(y), dtype=float),
        ]
    )
    with pytest.raises(GlmFitError) as exc:
        fit_nb2(
            y,
            x,
            l2=2.0,
            default_mu=5.0,
            feature_names=("zero_col", "x1"),
            fold="holdout_2023",
        )
    message = str(exc.value)
    assert "holdout_2023" in message
    assert "zero_col" in message
    assert "rank" in message.lower()
    assert str(len(y)) in message


def test_fit_nb2_does_not_masquerade_as_glm_via_moments() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x = np.ones((len(y), 4))
    with pytest.raises(GlmFitError):
        fit_nb2(y, x, l2=1.0, default_mu=3.0)


def test_moments_fallback_only_when_explicitly_requested() -> None:
    y = np.array([4.0])
    x = np.ones((1, 1))
    fit = fit_nb2(y, x, l2=0.0, default_mu=5.0, fail_closed=False)
    assert fit.method == "moments"
    assert fit.coef[0] == pytest.approx(np.log(4.0))
    assert fit.alpha > 0


def test_fit_nb2_records_optimizer_diagnostics() -> None:
    rng = np.random.default_rng(4)
    n = 40
    x1 = rng.normal(size=n)
    mu = np.exp(0.4 + 0.3 * x1)
    y = rng.poisson(mu)
    x = np.column_stack([np.ones(n), x1])
    fit = fit_nb2(y, x, l2=0.0, default_mu=1.0, fold="diag_ok")
    assert fit.method == "glm"
    assert fit.converged is True
    assert fit.iterations is not None and int(fit.iterations) > 0
    assert fit.coef.shape == (2,)


def test_nonconverged_glm_raises_instead_of_returning_glm(monkeypatch) -> None:
    y = np.array([3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 8.0, 4.0])
    x = np.column_stack([np.ones(len(y)), np.arange(len(y), dtype=float)])

    class FakeResult:
        params = np.array([1.6, 0.0, 0.1])
        mle_retvals = {"converged": False, "iterations": 200}

        def cov_params(self) -> np.ndarray:
            return np.eye(3)

    class FakeNB:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def fit(self, *args, **kwargs):
            _ = args, kwargs
            return FakeResult()

    monkeypatch.setattr("src.mlb.models.workload.NegativeBinomial", FakeNB)
    with pytest.raises(GlmFitError, match="did not converge") as exc:
        fit_nb2(
            y, x, l2=0.0, default_mu=5.0, fold="eval_2023", require_convergence=True
        )
    message = str(exc.value)
    assert "eval_2023" in message
    assert "200" in message
    assert "intercept-only" in message


def test_nonconverged_nonzero_slopes_stay_labeled_glm(monkeypatch) -> None:
    y = np.array([3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 8.0, 4.0])
    x = np.column_stack([np.ones(len(y)), np.arange(len(y), dtype=float)])

    class FakeResult:
        params = np.array([1.6, 0.35, 0.1])
        mle_retvals = {"converged": False, "iterations": 12}

        def cov_params(self) -> np.ndarray:
            return np.eye(3)

    class FakeNB:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def fit(self, *args, **kwargs):
            _ = args, kwargs
            return FakeResult()

    monkeypatch.setattr("src.mlb.models.workload.NegativeBinomial", FakeNB)
    fit = fit_nb2(
        y, x, l2=0.0, default_mu=5.0, fold="eval_2022", require_convergence=True
    )
    assert fit.method == "glm"
    assert fit.converged is False
    assert fit.iterations == 12


def test_select_usable_features_drops_missing_and_constant() -> None:
    frame = pd.DataFrame(
        {
            "k_bf_shrunk_365": [0.20, 0.22, 0.24, 0.26],
            "opp_k_rate_vs_hand_shrunk": [np.nan, np.nan, np.nan, np.nan],
            "is_opener": [0.0, 0.0, 0.0, 0.0],
            "pitcher_throws_L": [0.0, 1.0, 0.0, 1.0],
        }
    )
    selection = select_usable_features(
        frame,
        ("k_bf_shrunk_365", "opp_k_rate_vs_hand_shrunk", "is_opener", "pitcher_throws_L"),
    )
    assert selection.retained == ("k_bf_shrunk_365", "pitcher_throws_L")
    assert selection.dropped["opp_k_rate_vs_hand_shrunk"] == "all_missing"
    assert selection.dropped["is_opener"] == "zero_variance"


def test_default_strikeout_schema_excludes_unpopulated_placeholders() -> None:
    forbidden = {
        "opp_k_rate_vs_hand_shrunk",
        "lineup_k_rate_shrunk",
        "csw_750",
        "whiff_750",
        "fb_velo_delta",
        "ff_share_delta",
        "is_opener",
        "is_restricted",
        "expected_bf_oof",
        "bf_sd_oof",
        "p_early_exit_oof",
    }
    assert forbidden.isdisjoint(STRIKEOUT_FEATURE_COLUMNS)
    required = {
        "k_bf_shrunk_365",
        "k_bf_shrunk_60",
        "rest_days_capped",
        "pitcher_throws_L",
        "is_home",
    }
    assert required.issubset(STRIKEOUT_FEATURE_COLUMNS)
    assert "bf_mean_5" not in STRIKEOUT_FEATURE_COLUMNS
    assert "predicted_bf_oof" not in STRIKEOUT_FEATURE_COLUMNS


def test_encode_rest_features_caps_and_flags_absence() -> None:
    frame = pd.DataFrame(
        {
            "rest_days": [np.nan, 5.0, 9.0, 21.0, 2500.0],
        }
    )
    encoded = encode_rest_features(frame)
    assert encoded.loc[0, "first_start_or_missing_history"] == 1.0
    assert encoded.loc[1, "standard_rest"] == 1.0
    assert encoded.loc[2, "extended_rest"] == 1.0
    assert encoded.loc[3, "long_absence"] == 1.0
    assert encoded.loc[4, "long_absence"] == 1.0
    assert encoded["rest_days_capped"].max() == 14.0
    assert pd.isna(encoded.loc[0, "rest_days_capped"])


def test_fit_strikeouts_stores_retained_schema_and_applies_it(
    mlb_config,
) -> None:
    rng = np.random.default_rng(7)
    n = 80
    k_rate = np.clip(0.18 + 0.08 * rng.normal(size=n), 0.10, 0.35)
    bf = np.clip(22.0 + 4.0 * rng.normal(size=n), 12.0, 32.0)
    train = pd.DataFrame(
        {
            "strikeouts": rng.poisson(k_rate * bf),
            "bf_mean_5": bf,
            "bf_sd_5": 2.0 + rng.random(n),
            "early_exit_rate_5": rng.uniform(0.05, 0.35, n),
            "k_bf_shrunk_365": k_rate,
            "k_bf_shrunk_60": k_rate + 0.01 * rng.normal(size=n),
            "predicted_bf_oof": bf,
            "rest_days": rng.choice([4.0, 5.0, 6.0, 8.0, 20.0, np.nan], size=n),
            "pitcher_throws_L": rng.integers(0, 2, size=n).astype(float),
            "is_home": rng.integers(0, 2, size=n).astype(float),
            "csw_750": np.nan,
            "game_date": pd.date_range("2021-04-01", periods=n, freq="D").astype(str),
        }
    )
    model = fit_strikeouts(train, mlb_config, fold="eval_2022")
    assert model.method == "glm"
    assert "csw_750" not in model.feature_names
    assert "k_bf_shrunk_365" in model.feature_names
    assert "is_home" in model.feature_names
    assert model.dropped_features.get("csw_750") in {None, "all_missing"}
    assert model.fold == "eval_2022"
    assert model.n_train == n
    assert model.matrix_rank == 1 + len(model.feature_names)
    assert model.centers
    assert model.scales
    pred = predict_strikeout_pmf(model, train, mlb_config)
    assert pred["expected_k"].std() > 0.05
    league = float(train["strikeouts"].mean())
    assert not np.allclose(pred["expected_k"].to_numpy(), league, atol=0.05)
    diag = model.extra["glm_diagnostics"]
    assert list(diag["retained_features"]) == list(model.feature_names)
    assert diag["method"] == "glm"
    assert diag["converged"] is True
    assert int(diag["fit_iterations"]) > 0
    assert len(diag["coef"]) == 1 + len(model.feature_names)
    assert diag["coef_names"][0] == "intercept"
    assert np.isfinite(diag["offset_min"])
    assert np.isfinite(diag["offset_max"])
    assert diag["offset_nan_count"] == 0


def test_continuous_predictors_are_standardized_before_penalty(mlb_config) -> None:
    rng = np.random.default_rng(11)
    n = 60
    k_rate = np.clip(0.20 + 0.05 * rng.normal(size=n), 0.12, 0.32)
    bf = np.clip(23.0 + 3.0 * rng.normal(size=n), 14.0, 30.0)
    train = pd.DataFrame(
        {
            "strikeouts": rng.poisson(k_rate * bf),
            "bf_mean_5": bf,
            "bf_sd_5": 2.0 + rng.random(n),
            "early_exit_rate_5": rng.uniform(0.05, 0.30, n),
            "k_bf_shrunk_365": k_rate,
            "k_bf_shrunk_60": np.clip(k_rate + 0.03 * rng.normal(size=n), 0.10, 0.40),
            "predicted_bf_oof": bf,
            "rest_days": rng.choice([4.0, 5.0, 6.0, 8.0, 16.0], size=n),
            "pitcher_throws_L": rng.integers(0, 2, size=n).astype(float),
            "is_home": rng.integers(0, 2, size=n).astype(float),
        }
    )
    model = fit_strikeouts(train, mlb_config)
    prepared = prepare_strikeout_frame(train)
    x = design_matrix(
        prepared,
        model.feature_names,
        model.medians,
        centers=model.centers,
        scales=model.scales,
    )
    for i, name in enumerate(model.feature_names):
        col = x[:, i + 1]
        if name in BINARY_STRIKEOUT_FEATURES:
            assert set(np.unique(np.round(col, 8))).issubset({0.0, 1.0})
        else:
            assert abs(col.mean()) < 1e-6
            assert abs(col.std(ddof=0) - 1.0) < 1e-6


def test_fit_strikeouts_counts_raw_offset_nans(mlb_config) -> None:
    rng = np.random.default_rng(5)
    n = 50
    k_rate = np.clip(0.20 + 0.06 * rng.normal(size=n), 0.12, 0.32)
    bf = np.clip(22.0 + 3.0 * rng.normal(size=n), 14.0, 30.0)
    predicted = bf.copy()
    predicted[:7] = np.nan
    train = pd.DataFrame(
        {
            "strikeouts": rng.poisson(k_rate * bf),
            "bf_mean_5": bf,
            "bf_sd_5": 2.0 + rng.random(n),
            "early_exit_rate_5": rng.uniform(0.05, 0.30, n),
            "k_bf_shrunk_365": k_rate,
            "k_bf_shrunk_60": k_rate,
            "predicted_bf_oof": predicted,
            "rest_days": rng.choice([4.0, 5.0, 6.0, 8.0], size=n),
            "pitcher_throws_L": rng.integers(0, 2, size=n).astype(float),
            "is_home": rng.integers(0, 2, size=n).astype(float),
        }
    )
    model = fit_strikeouts(train, mlb_config, fold="offset_nans")
    diag = model.extra["glm_diagnostics"]
    assert diag["offset_nan_count"] == 7
    assert np.isfinite(diag["offset_min"])
    assert np.isfinite(diag["offset_max"])


def test_zero_variance_after_fill_is_reported_not_fit() -> None:
    frame = pd.DataFrame(
        {
            "k_bf_shrunk_365": [0.2, 0.2, 0.2, 0.2],
            "is_home": [1.0, 0.0, 1.0, 0.0],
        }
    )
    selection = select_usable_features(frame, ("k_bf_shrunk_365", "is_home"))
    assert "k_bf_shrunk_365" in selection.dropped
    assert selection.dropped["k_bf_shrunk_365"] == "zero_variance"
    x = design_matrix(frame, selection.retained, selection.medians)
    assert x.shape[1] == 1 + len(selection.retained)
    assert np.linalg.matrix_rank(x) == x.shape[1]


def test_select_usable_features_drops_collinear_columns() -> None:
    frame = pd.DataFrame(
        {
            "k_bf_shrunk_365": [0.18, 0.22, 0.26, 0.30],
            "k_bf_shrunk_60": [0.18, 0.22, 0.26, 0.30],
            "is_home": [0.0, 1.0, 0.0, 1.0],
        }
    )
    selection = select_usable_features(
        frame, ("k_bf_shrunk_365", "k_bf_shrunk_60", "is_home")
    )
    assert "k_bf_shrunk_365" in selection.retained
    assert selection.dropped["k_bf_shrunk_60"] == "collinear"
    assert "is_home" in selection.retained
