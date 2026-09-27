"""Out-of-sample minutes/rate knots are fold predictions, written once."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.shared.artifacts import quantile_training_params
from models.shared.oos import (
    HOLDOUT_WINDOW_ID,
    QUANTILE_LEVELS,
    assemble_oos_predictions,
    capped_rate,
    holdout_prediction_frame,
    oof_prediction_frame,
    oos_columns,
    write_oos_parquet,
)
from models.shared.train import run_walk_forward


def _quantiles() -> list[float]:
    return list(QUANTILE_LEVELS)


def _knots(rows: int, start: float) -> dict[str, np.ndarray]:
    grid = np.vstack(
        [
            np.arange(rows, dtype=float)[:, None] + start + np.arange(11)
            for _ in range(1)
        ]
    )
    # Cross two columns so assembly has to sort along tau.
    grid = np.column_stack(
        [np.full(rows, start + offset, dtype=float) for offset in range(11)]
    )
    grid[:, 0], grid[:, 1] = grid[:, 1].copy(), grid[:, 0].copy()
    return {f"q_{level:.2f}": grid[:, index] for index, level in enumerate(_quantiles())}


def _identity(n: int, *, game_offset: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"g{game_offset + i}" for i in range(n)],
            "player_id": np.arange(n),
            "game_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "minutes": np.full(n, 20.0),
            "pts": np.full(n, 10.0),
        }
    )


def _oof(identity: pd.DataFrame, preds: dict[str, np.ndarray], fold: int) -> pd.DataFrame:
    frame = pd.DataFrame(preds, index=identity.index)
    frame["fold_id"] = fold
    frame["minutes"] = 99.0  # training target; not realized minutes
    return frame


def test_training_params_omit_quantile_alpha():
    class _Model:
        def get_params(self):
            return {
                "objective": "reg:quantileerror",
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "min_child_weight": 1,
                "n_jobs": 1,
                "random_state": 0,
                "early_stopping_rounds": 5,
                "quantile_alpha": 0.5,
            }

    params = quantile_training_params({"quantile_models": {"q_0.50": _Model()}})
    assert "quantile_alpha" not in params
    assert params["n_estimators"] == 10


def test_capped_rate_clips_at_six_and_rejects_zero_minutes():
    got = capped_rate([12.0, 30.0, 0.0], [20.0, 4.0, 10.0])
    np.testing.assert_allclose(got, [0.6, 6.0, 0.0])
    with pytest.raises(ValueError, match="minutes > 0"):
        capped_rate([1.0], [0.0])


def test_oof_frame_keeps_realized_minutes_not_the_training_target():
    identity = _identity(3)
    preds = _knots(3, start=1.0)
    oof = _oof(identity, preds, fold=2)
    out = oof_prediction_frame(oof, identity)
    np.testing.assert_allclose(out["minutes"], identity["minutes"])
    assert (out["window_id"] == 2).all()
    assert not out["is_holdout"].any()
    assert "q_0.50" in out.columns


def test_assemble_sorts_knots_and_stacks_both_targets():
    identity = _identity(2)
    minutes = oof_prediction_frame(_oof(identity, _knots(2, start=10.0), 1), identity)
    rate = oof_prediction_frame(_oof(identity, _knots(2, start=0.1), 1), identity)
    holdout_id = _identity(1, game_offset=10)
    holdout_minutes = holdout_prediction_frame(holdout_id, _knots(1, start=15.0))
    holdout_rate = holdout_prediction_frame(holdout_id, _knots(1, start=0.4))
    table = assemble_oos_predictions(minutes, rate, holdout_minutes, holdout_rate)

    assert list(table.columns) == list(oos_columns())
    assert len(table) == 3
    pre = table.loc[~table["is_holdout"]]
    held = table.loc[table["is_holdout"]]
    assert (pre["window_id"] == 1).all()
    assert (held["window_id"] == HOLDOUT_WINDOW_ID).all()
    minute_grid = pre[[f"minutes_q_{q:.2f}" for q in QUANTILE_LEVELS]].to_numpy()
    rate_grid = held[[f"rate_q_{q:.2f}" for q in QUANTILE_LEVELS]].to_numpy()
    assert np.all(np.diff(minute_grid, axis=1) >= 0)
    assert np.all(np.diff(rate_grid, axis=1) >= 0)
    # Crossed inputs were swapped back into tau order.
    assert minute_grid[0, 0] < minute_grid[0, 1]


def test_assemble_rejects_minutes_rate_key_mismatch():
    identity = _identity(2)
    minutes = oof_prediction_frame(_oof(identity, _knots(2, start=10.0), 1), identity)
    other = _identity(2, game_offset=5)
    rate = oof_prediction_frame(_oof(other, _knots(2, start=0.1), 1), other)
    holdout_id = _identity(1, game_offset=10)
    holdout = holdout_prediction_frame(holdout_id, _knots(1, start=1.0))
    with pytest.raises(ValueError, match="same player-games"):
        assemble_oos_predictions(minutes, rate, holdout, holdout)


def test_assemble_rejects_shared_preholdout_and_holdout_keys():
    identity = _identity(1)
    pre = oof_prediction_frame(_oof(identity, _knots(1, start=10.0), 1), identity)
    holdout = holdout_prediction_frame(identity, _knots(1, start=10.0))
    with pytest.raises(ValueError, match="in-sample"):
        assemble_oos_predictions(pre, pre, holdout, holdout)


def test_assemble_rejects_non_fold_window_on_preholdout_rows():
    identity = _identity(1)
    pre = oof_prediction_frame(_oof(identity, _knots(1, start=10.0), 1), identity)
    pre["window_id"] = HOLDOUT_WINDOW_ID
    other = _identity(1, game_offset=3)
    holdout = holdout_prediction_frame(other, _knots(1, start=1.0))
    with pytest.raises(ValueError, match="walk-forward fold"):
        assemble_oos_predictions(pre, pre.copy(), holdout, holdout.copy())


def test_write_oos_parquet_refuses_to_revise(tmp_path):
    identity = _identity(1)
    pre = oof_prediction_frame(_oof(identity, _knots(1, start=10.0), 1), identity)
    other = _identity(1, game_offset=4)
    holdout = holdout_prediction_frame(other, _knots(1, start=1.0))
    table = assemble_oos_predictions(pre, pre.copy(), holdout, holdout.copy())
    path = tmp_path / "minutes_rate_oos.parquet"
    write_oos_parquet(table, path)
    with pytest.raises(FileExistsError, match="never revised"):
        write_oos_parquet(table, path)


def test_walk_forward_oof_contains_only_validation_rows():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.date_range("2020-01-01", periods=20, freq="D"), 4)
    frame = pd.DataFrame(
        {
            "game_date": dates,
            "x": rng.normal(size=len(dates)),
            "minutes": rng.uniform(10, 30, size=len(dates)),
            "starting": 1,
        }
    )
    params = dict(
        objective="reg:quantileerror",
        n_estimators=8,
        max_depth=2,
        learning_rate=0.1,
        n_jobs=1,
        random_state=0,
        early_stopping_rounds=2,
    )
    result = run_walk_forward(
        frame[["x"]],
        frame["minutes"],
        frame,
        xgb_params=params,
        quantiles=[0.50],
        n_folds=2,
        train_frac=0.50,
        step_frac=0.25,
    )
    allowed: set[int] = set()
    for fold in result["wf_results"]:
        allowed.update(frame.index[fold["val_mask"]].tolist())
    assert set(result["oof"].index.tolist()) == allowed
    train_only = set(frame.index.tolist()) - allowed
    assert train_only.isdisjoint(result["oof"].index)
