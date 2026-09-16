"""Strikeout NB fit, NLL ranking, and prediction schema."""

from __future__ import annotations

import numpy as np
from src.mlb.fixtures import build_synthetic_tables
from src.mlb.models.metrics import pmf_nll
from src.mlb.models.pmf import negative_binomial_pmf
from src.mlb.models.strikeouts import fit_strikeouts, predict_strikeout_pmf
from src.mlb.models.workload import fit_workload, predict_workload
from src.mlb.schemas import PMF_COLUMNS, PREDICTION_COLUMNS, STRIKEOUT_FEATURE_COLUMNS
from tests.mlb.models.test_workload import dummy_feature_rows


def test_nb_nll_lower_for_true_mu() -> None:
    y = np.array([6, 6, 5, 7])
    true_pmf = negative_binomial_pmf(np.full(4, 6.0), 0.12, 15, 0.001)
    far_pmf = negative_binomial_pmf(np.full(4, 20.0), 0.12, 15, 0.001)
    assert pmf_nll(true_pmf, y).mean() < pmf_nll(far_pmf, y).mean()


def test_fit_strikeouts_on_synthetic_starts(mlb_config) -> None:
    tables = build_synthetic_tables(mlb_config)
    starts = tables["pitcher_starts"]
    features = dummy_feature_rows(starts, np.random.default_rng(mlb_config.seed))
    train = starts.merge(features, on=["pitcher_id", "game_pk"], suffixes=("", "_feat"))
    workload = fit_workload(train, mlb_config)
    wl = predict_workload(workload, train)
    train = train.copy()
    train["expected_bf_oof"] = wl["expected_bf"].to_numpy()
    train["bf_sd_oof"] = wl["bf_sd"].to_numpy()
    train["p_early_exit_oof"] = wl["p_early_exit"].to_numpy()
    train["expected_pitches_oof"] = wl["expected_pitches"].to_numpy()
    train["expected_outs_oof"] = wl["expected_outs"].to_numpy()
    model = fit_strikeouts(train, mlb_config)
    pred = predict_strikeout_pmf(model, train, mlb_config)
    assert list(pred.columns) == list(PREDICTION_COLUMNS)
    assert pred["expected_k"].notna().all()
    np.testing.assert_allclose(pred[list(PMF_COLUMNS)].sum(axis=1), 1.0, atol=1e-8)
    assert pred["p_over_4_5"].notna().all()
    assert np.all(pred["expected_innings"] == pred["expected_outs"] / 3.0)
    for column in STRIKEOUT_FEATURE_COLUMNS:
        assert column in train.columns
