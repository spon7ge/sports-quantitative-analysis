"""Model bundle save/load round-trip."""

from __future__ import annotations

import numpy as np
from src.mlb.config import load_config
from src.mlb.fixtures import build_synthetic_tables
from src.mlb.models.serialize import (
    as_strikeout_model,
    load_model,
    save_model,
)
from src.mlb.models.strikeouts import fit_strikeouts, predict_strikeout_pmf
from src.mlb.models.workload import add_oof_workload_features
from src.mlb.pipeline.features import build_feature_rows


def test_save_load_round_trip(tmp_path) -> None:
    bundle = {
        "model_version": "nb_k_v1",
        "feature_names": ["rest_days", "k_bf_shrunk_365"],
        "params": np.array([1.2, -0.1, 0.4]),
        "alpha": 0.18,
        "medians": {"rest_days": 5.0, "k_bf_shrunk_365": 0.22},
    }
    path = tmp_path / "strikeouts.joblib"
    save_model(path, bundle)
    loaded = load_model(path)
    assert loaded["model_version"] == "nb_k_v1"
    assert loaded["feature_names"] == bundle["feature_names"]
    np.testing.assert_allclose(loaded["params"], bundle["params"])
    assert loaded["alpha"] == bundle["alpha"]
    assert loaded["medians"] == bundle["medians"]


def test_saved_strikeout_model_rehydrates_and_predicts(tmp_path) -> None:
    config = load_config()
    tables = build_synthetic_tables(config)
    features = build_feature_rows(tables, config)
    features = add_oof_workload_features(tables["pitcher_starts"], features, config)
    train = features.merge(
        tables["pitcher_starts"][["pitcher_id", "game_pk", "strikeouts"]],
        on=["pitcher_id", "game_pk"],
        how="inner",
    )
    train = train.dropna(subset=["strikeouts"])
    model = fit_strikeouts(train, config)
    path = tmp_path / "nb_k_v1.pkl"
    save_model(path, model)
    restored = as_strikeout_model(load_model(path))
    preds = predict_strikeout_pmf(restored, train.head(3), config)
    assert len(preds) == 3
    pmf_cols = [c for c in preds.columns if c.startswith("pmf_")]
    np.testing.assert_allclose(preds[pmf_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-6)

