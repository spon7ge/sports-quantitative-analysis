"""Pricing entry point: fingerprints, PMF, and P(over the line)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from models.shared.artifacts import booster_fingerprint, save_model_bundle
from models.shared.points_pmf import PpmTails


def _tiny(shift: float, features: list[str]) -> XGBRegressor:
    frame = pd.DataFrame({name: np.arange(8, dtype=float) for name in features})
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


def _bundle(tmp_path, stem: str, features: list[str], shift: float):
    train = pd.DataFrame({"game_date": pd.to_datetime(["2024-04-01"])})
    holdout = pd.DataFrame({"game_date": pd.to_datetime(["2025-10-21"])})
    model = _tiny(shift, features)
    path = save_model_bundle(
        {"q_0.50": model},
        train_df=train,
        holdout_df=holdout,
        wf_results=[{"pinball": 1.0}],
        ho_metrics={"pinball": 1.0},
        features=features,
        artifact_stem=stem,
        save_dir=tmp_path,
    )
    return path, {"q_0.50": model}


def _tails() -> PpmTails:
    return PpmTails(
        arrays={
            ("lower", 0): np.array([1.0, 1.0]),
            ("lower", 1): np.array([1.0, 1.0]),
            ("upper", 0): np.array([0.0, 0.0]),
            ("upper", 1): np.array([0.0, 0.0]),
        }
    )


def test_load_rejects_a_swapped_minutes_booster(tmp_path):
    from models.shared.points_pricing import load_points_pricing, save_points_pricing

    minutes_path, minutes_models = _bundle(tmp_path, "min_price", ["starting"], 0.0)
    ppm_path, ppm_models = _bundle(tmp_path, "ppm_price", ["starting", "minutes_given"], 0.0)
    payload_path = save_points_pricing(
        tmp_path / "pricing.joblib",
        minutes_bundle=minutes_path,
        ppm_bundle=ppm_path,
        minutes_models=minutes_models,
        minutes_features=["starting"],
        ppm_models=ppm_models,
        ppm_features=["starting", "minutes_given"],
        tails=_tails(),
        g_m=2,
        g_p=2,
        rounding="rint",
        k_max=20,
    )
    loaded = load_points_pricing(payload_path)
    assert loaded.minutes_fingerprint == booster_fingerprint(minutes_models, ["starting"])

    import joblib

    swapped = joblib.load(minutes_path)
    swapped["quantile_models"]["q_0.50"] = _tiny(4.0, ["starting"])
    joblib.dump(swapped, minutes_path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_points_pricing(payload_path)


def test_supplied_grids_round_half_up_and_clear_the_line():
    """10 * 1.25 = 12.5 lands on 13, which is over 12.5 and under 13.5."""
    from models.shared.points_pricing import price_points

    frame = pd.DataFrame({"starting": [1], "game_id": ["g"], "player_id": ["p"]})
    pmf, over = price_points(
        frame,
        np.array([[12.5, 13.5]]),
        minutes_q=np.array([[10.0]]),
        ppm_q=np.array([[[1.25]]]),
        rounding="rint",
        k_max=20,
    )
    assert pmf.shape == (1, 21)
    assert pmf[0, 13] == pytest.approx(1.0)
    np.testing.assert_allclose(over, [[1.0, 0.0]])
