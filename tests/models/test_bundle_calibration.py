"""Calibration travels with the boosters it was stored against."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from models.shared.artifacts import (
    booster_fingerprint,
    load_model_bundle,
    predict_calibrated_quantiles,
    save_model_bundle,
)
from models.shared.metrics import MINUTE_Q50_TIERS, RATE_Q50_TIERS
from xgboost import XGBRegressor


def _tiny_model(shift: float) -> XGBRegressor:
    frame = pd.DataFrame({"starting": np.arange(8, dtype=float)})
    target = np.arange(8, dtype=float) + shift
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        n_estimators=2,
        max_depth=1,
        random_state=0,
    )
    model.fit(frame, target)
    return model


def _frames():
    train = pd.DataFrame(
        {"game_date": pd.to_datetime(["2024-04-01"]), "starting": [1.0]}
    )
    holdout = pd.DataFrame(
        {"game_date": pd.to_datetime(["2025-10-21"]), "starting": [1.0]}
    )
    return train, holdout


def _calibration() -> dict:
    return {
        "offsets": pd.DataFrame(
            {"q_0.50": [0.1]}, index=pd.Index(["<15"], name="group")
        ),
        "group_definition": {
            "cold_start": "non-finite min_lag_1",
            "tiers": tuple(MINUTE_Q50_TIERS),
        },
        "min_n": 300,
        "date_start": "2023-01-01",
        "date_end": "2024-04-01",
        "row_counts": {"<15": 400, "cold_start": 10},
        "role_quantiles": pd.DataFrame(
            {"q_0.50": [20.0, 8.0]}, index=["starter", "bench"]
        ),
        "max_minutes": 63.0,
    }


def test_saves_increment_past_numbered_files_and_ignore_dates(tmp_path):
    train, holdout = _frames()
    (tmp_path / "pts_nba_model_2026-11-02.joblib").write_bytes(b"dated")
    (tmp_path / "pts_nba_model_01.joblib").write_bytes(b"first")
    path = save_model_bundle(
        {"q_0.50": _tiny_model(0.0)},
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"pinball": 1.0}],
        ho_metrics={"pinball": 1.0},
        features=["starting"],
        artifact_stem="pts_nba_model",
        save_dir=tmp_path,
    )
    assert path.name == "pts_nba_model_02.joblib"


def test_load_rejects_a_booster_that_does_not_match_the_fingerprint(tmp_path):
    train, holdout = _frames()
    model = _tiny_model(0.0)
    path = save_model_bundle(
        {"q_0.50": model},
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"pinball": 1.0}],
        ho_metrics={"pinball": 1.0},
        features=["starting"],
        artifact_stem="min_test",
        calibration=_calibration(),
        save_dir=tmp_path,
    )
    loaded = load_model_bundle(path)
    assert loaded["calibration"]["fingerprint"] == booster_fingerprint(
        loaded["quantile_models"], loaded["feature_names"]
    )

    swapped = joblib.load(path)
    swapped["quantile_models"]["q_0.50"] = _tiny_model(5.0)
    joblib.dump(swapped, path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_model_bundle(path)


def test_load_rejects_a_changed_group_definition(tmp_path):
    train, holdout = _frames()
    path = save_model_bundle(
        {"q_0.50": _tiny_model(0.0)},
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"pinball": 1.0}],
        ho_metrics={"pinball": 1.0},
        features=["starting"],
        artifact_stem="min_test",
        calibration=_calibration(),
        save_dir=tmp_path,
    )
    edited = joblib.load(path)
    edited["calibration"]["group_definition"]["tiers"] = (("nope", None, None),)
    joblib.dump(edited, path)
    with pytest.raises(ValueError, match="tiers"):
        load_model_bundle(path)


def test_save_rejects_a_score_table_in_place_of_the_offset_payload(tmp_path):
    train, holdout = _frames()
    scores = pd.DataFrame(
        {"quantile": [0.5], "empirical": [0.51], "delta": [0.01]}
    )
    with pytest.raises(ValueError, match="group_definition"):
        save_model_bundle(
            {"q_0.50": _tiny_model(0.0)},
            train_df=train,
            holdout_df=holdout,
            wf_results=[{"pinball": 1.0}],
            ho_metrics={"pinball": 1.0},
            features=["starting"],
            artifact_stem="min_test",
            calibration=scores,
            save_dir=tmp_path,
        )
    assert list(tmp_path.glob("*.joblib")) == []


def _rate_calibration() -> dict:
    payload = _calibration()
    payload["target"] = "pts_per_min"
    payload["lag_col"] = "pts_per_min_lag_1"
    payload["clip_low"] = 0.0
    payload["clip_high"] = 6.0
    payload["max_minutes"] = 6.0
    payload["group_definition"] = {
        "cold_start": "non-finite pts_per_min_lag_1",
        "tiers": tuple(RATE_Q50_TIERS),
    }
    return payload


def test_a_rate_bundle_loads_only_with_rate_tiers(tmp_path):
    train, holdout = _frames()
    path = save_model_bundle(
        {"q_0.50": _tiny_model(0.0)},
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"pinball": 1.0}],
        ho_metrics={"pinball": 1.0},
        features=["starting"],
        artifact_stem="pts_test",
        calibration=_rate_calibration(),
        save_dir=tmp_path,
    )
    loaded = load_model_bundle(path)
    assert loaded["calibration"]["target"] == "pts_per_min"

    edited = joblib.load(path)
    edited["calibration"]["group_definition"]["tiers"] = tuple(MINUTE_Q50_TIERS)
    joblib.dump(edited, path)
    with pytest.raises(ValueError, match="tiers"):
        load_model_bundle(path)


def test_predict_reads_a_rate_lag_beside_the_model_features_and_clips_at_the_cap():
    frame = pd.DataFrame(
        {
            "starting": [0, 0],
            "pts_per_min_lag_1": [0.4, 0.4],
        }
    )
    bundle = {
        "quantile_models": {"q_0.50": _Column([-1.0, 9.0])},
        "feature_names": ["starting"],
        "calibration": {
            "offsets": pd.DataFrame(
                {"q_0.50": [0.0]}, index=pd.Index(["<0.30"], name="group")
            ),
            "role_quantiles": pd.DataFrame(
                {"q_0.50": [0.4, 0.2]}, index=["starter", "bench"]
            ),
            "group_definition": {
                "cold_start": "non-finite pts_per_min_lag_1",
                "tiers": tuple(RATE_Q50_TIERS),
            },
            "lag_col": "pts_per_min_lag_1",
            "clip_low": 0.0,
            "clip_high": 6.0,
            "max_minutes": 6.0,
            "target": "pts_per_min",
        },
    }
    out = predict_calibrated_quantiles(bundle, frame)
    np.testing.assert_allclose(out["q_0.50"].to_numpy(), [0.0, 6.0])


def test_scaled_payload_loads_without_a_group_table(tmp_path):
    train, holdout = _frames()
    path = save_model_bundle(
        {"q_0.50": _tiny_model(0.0)},
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"pinball": 1.0}],
        ho_metrics={"pinball": 1.0},
        features=["starting"],
        artifact_stem="pts_scaled",
        calibration={
            "method": "scaled_quantile",
            "offsets": np.array([0.1, -0.1]),
            "taus": np.array([0.10, 0.90]),
            "clip_low": 0.0,
            "clip_high": 100.0,
            "target": "pts",
        },
        save_dir=tmp_path,
    )
    loaded = load_model_bundle(path)
    assert loaded["calibration"]["method"] == "scaled_quantile"
    np.testing.assert_allclose(loaded["calibration"]["offsets"], [0.1, -0.1])


def test_a_bundle_without_calibration_still_loads(tmp_path):
    train, holdout = _frames()
    path = save_model_bundle(
        {"q_0.50": _tiny_model(0.0)},
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"pinball": 1.0}],
        ho_metrics={"pinball": 1.0},
        features=["starting"],
        artifact_stem="min_test",
        save_dir=tmp_path,
    )
    loaded = load_model_bundle(path)
    assert loaded.get("calibration") is None


class _Column:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, frame):
        return self.values.copy()


def test_predict_sorts_replaces_cold_starts_then_adds_the_offset_and_clips():
    frame = pd.DataFrame(
        {
            "starting": [1, 0, 0],
            "min_lag_1": [np.nan, 5.0, 5.0],
        }
    )
    bundle = {
        "quantile_models": {
            "q_0.90": _Column([50.0, 5.0, 30.0]),
            "q_0.10": _Column([100.0, 9.0, 10.0]),
            "q_0.50": _Column([0.0, 1.0, 20.0]),
        },
        "feature_names": ["starting", "min_lag_1"],
        "calibration": {
            "offsets": pd.DataFrame(
                {"q_0.10": [-5.0], "q_0.50": [0.0], "q_0.90": [100.0]},
                index=pd.Index(["<15"], name="group"),
            ),
            "role_quantiles": pd.DataFrame(
                {
                    "q_0.10": [2.0, 1.0],
                    "q_0.50": [4.0, 3.0],
                    "q_0.90": [6.0, 5.0],
                },
                index=["starter", "bench"],
            ),
            "max_minutes": 63.0,
        },
    }

    out = predict_calibrated_quantiles(bundle, frame)

    assert list(out.columns) == ["q_0.10", "q_0.50", "q_0.90"]
    np.testing.assert_allclose(out.iloc[0].to_numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(out.iloc[1].to_numpy(), [1e-3, 5.0, 63.0])
    np.testing.assert_allclose(out.iloc[2].to_numpy(), [10.0, 20.0, 30.0])


def test_scaled_predict_sorts_then_clips():
    bundle = {
        "quantile_models": {
            "q_0.10": _Column([2.0, 0.0]),
            "q_0.90": _Column([1.0, 4.0]),
        },
        "feature_names": ["starting"],
        "calibration": {
            "method": "scaled_quantile",
            "offsets": np.array([0.0, 0.0]),
            "taus": np.array([0.10, 0.90]),
            "clip_low": 0.0,
            "clip_high": 1.5,
            "target": "pts",
        },
    }
    out = predict_calibrated_quantiles(
        bundle, pd.DataFrame({"starting": [1.0, 0.0]})
    )
    # Row 0 sorts to (1, 2) then clips at 1.5. Row 1 sorts to (0, 4) then clips.
    np.testing.assert_allclose(out.to_numpy(), [[1.0, 1.5], [0.0, 1.5]])
