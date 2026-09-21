"""Strikeouts use log(predicted_bf_oof) as exposure, never Game N BF."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from src.mlb.evaluation.backtest import _model_feature_rows
from src.mlb.models.strikeouts import fit_strikeouts, predict_strikeout_pmf
from src.mlb.pipeline.hf_tables import pregame_from_starts
from src.mlb.schemas import STRIKEOUT_FEATURE_COLUMNS, WORKLOAD_FEATURE_COLUMNS


def _k_frame(n: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k_rate = np.clip(0.18 + 0.08 * rng.normal(size=n), 0.10, 0.35)
    predicted_bf = np.clip(22.0 + 4.0 * rng.normal(size=n), 12.0, 32.0)
    return pd.DataFrame(
        {
            "strikeouts": rng.poisson(k_rate * predicted_bf),
            "batters_faced": np.full(n, 999.0),
            "predicted_bf_oof": predicted_bf,
            "expected_bf_oof": predicted_bf,
            "bf_mean_5": predicted_bf,
            "bf_sd_5": 2.0 + rng.random(n),
            "early_exit_rate_5": rng.uniform(0.05, 0.35, n),
            "k_bf_shrunk_365": k_rate,
            "k_bf_shrunk_60": k_rate + 0.01 * rng.normal(size=n),
            "rest_days": rng.choice([4.0, 5.0, 6.0, 8.0, 20.0], size=n),
            "pitcher_throws_L": rng.integers(0, 2, size=n).astype(float),
            "is_home": rng.integers(0, 2, size=n).astype(float),
            "game_date": pd.date_range("2021-04-01", periods=n, freq="D").astype(str),
        }
    )


def test_strikeout_schema_uses_offset_not_raw_bf() -> None:
    assert "predicted_bf_oof" not in STRIKEOUT_FEATURE_COLUMNS
    assert "batters_faced" not in STRIKEOUT_FEATURE_COLUMNS
    assert "bf_mean_5" not in STRIKEOUT_FEATURE_COLUMNS
    required = {
        "k_bf_shrunk_365",
        "k_bf_shrunk_60",
        "rest_days_capped",
        "pitcher_throws_L",
        "is_home",
    }
    assert required.issubset(STRIKEOUT_FEATURE_COLUMNS)
    workload = {
        "bf_mean_5",
        "bf_sd_5",
        "early_exit_rate_5",
        "pitches_per_start_5",
        "outs_per_start_5",
        "pitches_per_bf_5",
        "rest_days_capped",
        "long_absence",
        "first_start_or_missing_history",
    }
    assert workload.issubset(WORKLOAD_FEATURE_COLUMNS)


def test_expected_k_scales_with_predicted_bf_oof(mlb_config) -> None:
    train = _k_frame()
    model = fit_strikeouts(train, mlb_config)
    base = train.copy()
    base["predicted_bf_oof"] = 20.0
    doubled = base.copy()
    doubled["predicted_bf_oof"] = 40.0
    mu_base = predict_strikeout_pmf(model, base, mlb_config)["expected_k"].to_numpy()
    mu_doubled = predict_strikeout_pmf(model, doubled, mlb_config)["expected_k"].to_numpy()
    ratio = mu_doubled / np.clip(mu_base, 1e-8, None)
    assert float(np.median(ratio)) == pytest.approx(2.0, rel=0.02)
    assert float(np.mean(np.abs(ratio - 2.0) < 0.1)) > 0.9


def test_game_n_batters_faced_is_not_exposure(mlb_config) -> None:
    train = _k_frame()
    model_a = fit_strikeouts(train, mlb_config)
    leaked = train.copy()
    leaked["batters_faced"] = 1.0
    model_b = fit_strikeouts(leaked, mlb_config)
    np.testing.assert_allclose(model_a.coef, model_b.coef)
    assert model_a.alpha == model_b.alpha

    scored = train.copy()
    scored["batters_faced"] = 1.0
    mu_a = predict_strikeout_pmf(model_a, train, mlb_config)["expected_k"].to_numpy()
    mu_b = predict_strikeout_pmf(model_a, scored, mlb_config)["expected_k"].to_numpy()
    np.testing.assert_allclose(mu_a, mu_b)


def test_gamelog_path_fills_predicted_bf_oof(mlb_config) -> None:
    rows = []
    for day in range(8):
        date = pd.Timestamp("2024-04-01") + pd.Timedelta(days=7 * day)
        for i in range(6):
            bf = 12.0 if day < 6 else 40.0
            rows.append(
                {
                    "pitcher_id": 2000 + i,
                    "game_pk": 80000 + day * 10 + i,
                    "game_date": date.strftime("%Y-%m-%d"),
                    "season": 2024,
                    "strikeouts": 5,
                    "batters_faced": bf,
                    "pitches": bf * 4,
                    "outs": max(bf - 7, 1),
                    "role": "starter",
                    "is_home": i % 2,
                    "opponent_team_id": 2,
                    "team_id": 3,
                    "venue_id": 1,
                    "pitcher_hand": "R",
                    "scheduled_start_utc": date.tz_localize("UTC") + pd.Timedelta(hours=23),
                    "event_time_utc": date.tz_localize("UTC") + pd.Timedelta(hours=23),
                    "ingested_at_utc": date.tz_localize("UTC") + pd.Timedelta(hours=23),
                    "doubleheader": 0,
                }
            )
    starts = pd.DataFrame(rows)
    config = replace(mlb_config, workload_min_train_starts=8)
    tables = {
        "pitcher_starts": starts,
        "pregame_snapshots": pregame_from_starts(starts, config),
        "pitch_events": pd.DataFrame(),
    }
    features = _model_feature_rows(tables, config)
    later = features.merge(
        starts[["pitcher_id", "game_pk", "game_date", "batters_faced"]],
        on=["pitcher_id", "game_pk"],
    )
    later = later.loc[later["game_date"] == "2024-05-20"]
    assert later["predicted_bf_oof"].notna().all()
    assert later["predicted_bf_oof"].max() < 30
    assert np.all(later["batters_faced"] == 40)
