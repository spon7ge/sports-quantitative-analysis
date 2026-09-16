"""OOF workload features must not train on the row being predicted."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from src.mlb.config import MlbConfig
from src.mlb.models.workload import (
    add_oof_workload_features,
    fit_workload,
    predict_workload,
)
from src.mlb.schemas import WORKLOAD_FEATURE_COLUMNS


def _config(**overrides) -> MlbConfig:
    params = {
        "data_dir": Path("data/mlb"),
        "artifact_dir": Path("artifacts/mlb"),
        "seed": 42,
        "workload_min_train_starts": 4,
        "workload_l2": 1.0,
        "early_exit_bf": 15,
    }
    params.update(overrides)
    return MlbConfig(**params)


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pitcher = 111001
    game_pk = 700000
    for date, n, bf in (
        ("2018-04-01", 4, 10),
        ("2018-04-08", 4, 10),
        ("2018-04-15", 2, 1000),
    ):
        for i in range(n):
            game_pk += 1
            rows.append(
                {
                    "pitcher_id": pitcher + i,
                    "game_pk": game_pk,
                    "game_date": date,
                    "batters_faced": bf,
                    "pitches": bf * 4,
                    "outs": max(bf - 7, 1),
                    "rest_days": 5.0,
                    "bf_per_start_5": 10.0,
                    "pitches_per_start_5": 40.0,
                    "outs_per_start_5": 7.0,
                    "is_opener": 0.0,
                    "is_restricted": 0.0,
                    "is_il_return": 0.0,
                    "is_home": float(i % 2),
                }
            )
    starts = pd.DataFrame(rows)
    features = starts[["pitcher_id", "game_pk", *WORKLOAD_FEATURE_COLUMNS]].copy()
    return starts, features


def test_oof_workload_excludes_same_row_batters_faced() -> None:
    starts, features = _panel()
    config = _config()
    out = add_oof_workload_features(starts, features, config)
    keyed = out.merge(starts[["pitcher_id", "game_pk", "game_date", "batters_faced"]], on=["pitcher_id", "game_pk"])

    first = keyed.loc[keyed["game_date"] == "2018-04-01", "expected_bf_oof"]
    assert first.isna().all()

    third = keyed.loc[keyed["game_date"] == "2018-04-15"]
    assert third["expected_bf_oof"].notna().all()
    assert third["expected_bf_oof"].max() < 40
    assert np.all(third["batters_faced"] == 1000)

    leaked = starts.merge(features, on=["pitcher_id", "game_pk"], suffixes=("", "_dup"))
    in_sample = predict_workload(fit_workload(leaked, config), features)
    in_sample = in_sample.assign(
        pitcher_id=features["pitcher_id"].to_numpy(),
        game_pk=features["game_pk"].to_numpy(),
    )
    leaked_mu = in_sample.merge(
        third[["pitcher_id", "game_pk"]],
        on=["pitcher_id", "game_pk"],
    )["expected_bf"]
    assert leaked_mu.min() > third["expected_bf_oof"].max()
