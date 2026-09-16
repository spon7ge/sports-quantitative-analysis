"""Workload fit/predict on synthetic starts plus dummy features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.mlb.fixtures import build_synthetic_tables
from src.mlb.models.workload import fit_workload, predict_workload
from src.mlb.schemas import STRIKEOUT_FEATURE_COLUMNS, WORKLOAD_FEATURE_COLUMNS


def dummy_feature_rows(starts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(starts)
    data: dict[str, np.ndarray | pd.Series] = {
        "pitcher_id": starts["pitcher_id"].to_numpy(),
        "game_pk": starts["game_pk"].to_numpy(),
        "rest_days": rng.integers(4, 8, size=n).astype(float),
        "bf_per_start_5": 22.0 + rng.normal(0.0, 2.0, n),
        "pitches_per_start_5": 90.0 + rng.normal(0.0, 8.0, n),
        "outs_per_start_5": 16.0 + rng.normal(0.0, 2.0, n),
        "is_opener": (starts["role"] == "opener").astype(float).to_numpy(),
        "is_restricted": starts["is_home"].to_numpy() * 0.0,
        "is_il_return": np.zeros(n, dtype=float),
        "is_home": starts["is_home"].astype(float).to_numpy(),
        "expected_bf_oof": starts["batters_faced"].astype(float).to_numpy(),
        "bf_sd_oof": np.full(n, 3.0),
        "p_early_exit_oof": np.full(n, 0.12),
        "k_bf_shrunk_365": (
            starts["strikeouts"].astype(float) / starts["batters_faced"].clip(lower=1)
        ).to_numpy(),
        "k_bf_shrunk_60": (
            starts["strikeouts"].astype(float) / starts["batters_faced"].clip(lower=1)
        ).to_numpy(),
        "opp_k_rate_vs_hand_shrunk": np.full(n, 0.22),
        "lineup_k_rate_shrunk": np.full(n, 0.23),
        "csw_750": 0.27 + rng.normal(0.0, 0.02, n),
        "whiff_750": 0.24 + rng.normal(0.0, 0.02, n),
        "fb_velo_delta": rng.normal(0.0, 0.4, n),
        "ff_share_delta": rng.normal(0.0, 0.03, n),
        "pitcher_throws_L": (starts["pitcher_hand"] == "L").astype(float).to_numpy(),
    }
    frame = pd.DataFrame(data)
    for column in WORKLOAD_FEATURE_COLUMNS + STRIKEOUT_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def test_fit_workload_on_synthetic_starts(mlb_config) -> None:
    tables = build_synthetic_tables(mlb_config)
    starts = tables["pitcher_starts"]
    features = dummy_feature_rows(starts, np.random.default_rng(mlb_config.seed))
    train = starts.merge(features, on=["pitcher_id", "game_pk"], suffixes=("", "_feat"))
    model = fit_workload(train, mlb_config)
    pred = predict_workload(model, features)
    assert set(pred.columns) >= {
        "expected_bf",
        "bf_sd",
        "expected_pitches",
        "expected_outs",
        "p_early_exit",
    }
    assert pred["expected_bf"].notna().all()
    assert np.all(pred["expected_bf"] > 0)
    assert np.all(pred["bf_sd"] > 0)
    assert np.all((pred["p_early_exit"] >= 0) & (pred["p_early_exit"] <= 1))
    np.testing.assert_allclose(
        pred["expected_pitches"],
        pred["expected_bf"] * model.mean_pitches_per_bf,
    )
    np.testing.assert_allclose(
        pred["bf_sd"],
        np.sqrt(
            pred["expected_bf"]
            + model.alpha * pred["expected_bf"] ** 2
        ),
    )
