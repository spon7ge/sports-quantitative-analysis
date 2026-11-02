"""Exploratory metadata and the out-of-sample prediction table."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from points_xgb.config import (
    ARTIFACT_STEM,
    HOLDOUT_SEASON,
    QUANTILES,
    ROLE_COL,
    SEED,
    TARGET_COL,
    XGB_PARAMS,
)
from points_xgb.paths import ensure_repo_on_path

ensure_repo_on_path()
from models.shared.metrics import interval_coverage, pinball_loss  # noqa: E402
from models.shared.oos import (  # noqa: E402
    holdout_prediction_frame,
    oof_prediction_frame,
)

_PARAM_KEYS = (
    "objective",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "min_child_weight",
    "early_stopping_rounds",
    "random_state",
)


def _iso_date(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def exploratory_metadata(
    ppm_df: pd.DataFrame,
    ppm_holdout: pd.DataFrame,
    ho_metrics: Mapping,
    wf_results: list[dict],
    *,
    features: list[str] | None = None,
    holdout_season: str = HOLDOUT_SEASON,
    label: str = TARGET_COL,
    label_rule: str = "min(pts / minutes, 6.0), minutes > 0",
) -> dict:
    if features is None:
        raise ValueError("pass features; the notebook owns PTS_FEATURES")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "exploratory",
        "model_name": "quantile_xgboost",
        "artifact_stem": ARTIFACT_STEM,
        "features": list(features),
        "label": label,
        "label_rule": label_rule,
        "quantiles": list(QUANTILES),
        "point_forecast": 0.50,
        "interval": [0.10, 0.90],
        "params": {key: XGB_PARAMS[key] for key in _PARAM_KEYS},
        "seed": SEED,
        "train_range": [
            _iso_date(ppm_df["game_date"].min()),
            _iso_date(ppm_df["game_date"].max()),
        ],
        "holdout_season": holdout_season,
        "holdout_range": [
            _iso_date(ppm_holdout["game_date"].min()),
            _iso_date(ppm_holdout["game_date"].max()),
        ],
        "primary_metric": "q50 pinball",
        "test_metrics": {
            "n": ho_metrics["n"],
            "pinball_q50": ho_metrics["pinball"],
            "coverage_80": ho_metrics.get("coverage_80pct"),
            "coverage_90": ho_metrics.get("coverage_90pct"),
            "walk_forward_pinball_q50": float(
                np.mean([row["pinball"] for row in wf_results])
            ),
        },
    }


def loaded_holdout_scores(
    models: Mapping,
    frame: pd.DataFrame,
    y_ho,
) -> dict[str, float]:
    keys = sorted(models, key=lambda key: float(str(key).split("_", 1)[1]))
    preds = {key: models[key].predict(frame) for key in keys}
    return {
        "pinball_q50": pinball_loss(y_ho, preds["q_0.50"], 0.50),
        "coverage_80": interval_coverage(y_ho, preds["q_0.10"], preds["q_0.90"]),
        "quantiles": keys,
    }


def build_oos_predictions(
    wf: Mapping,
    ppm_df: pd.DataFrame,
    ppm_holdout: pd.DataFrame,
    preds_ho: Mapping[str, np.ndarray],
    *,
    role_col: str = ROLE_COL,
    oof_cal: pd.DataFrame | None = None,
    preds_ho_cal: Mapping[str, np.ndarray] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Walk-forward and holdout rows for ``scripts/build_oos_table.py``.

    ``q_*`` stays the raw booster output. ``cal_q_*`` is the sorted,
    cold-start, offset knot when both calibrated frames are passed.
    """
    wf_oof = wf["oof"]
    folds_found = sorted(int(fold) for fold in wf_oof["fold_id"].unique())
    # The date splitter stops when the next validation window would run past
    # the last game date, so a requested fifth fold can be absent.
    if folds_found != list(range(1, len(folds_found) + 1)) or len(folds_found) < 4:
        raise ValueError(
            f"expected walk-forward folds 1..n with n >= 4, got {folds_found}"
        )
    walk_forward_rows = oof_prediction_frame(wf_oof, ppm_df).assign(
        starting=wf_oof["starting"].to_numpy(),
        early_stop=wf_oof["early_stop"].to_numpy(),
        train_tail_frac=wf_oof["train_tail_frac"].to_numpy(),
    )
    holdout_rows = holdout_prediction_frame(ppm_holdout, preds_ho).assign(
        starting=ppm_holdout[role_col].to_numpy(),
    )
    frame = pd.concat([walk_forward_rows, holdout_rows], ignore_index=True)
    if oof_cal is None and preds_ho_cal is None:
        return frame
    if oof_cal is None or preds_ho_cal is None:
        raise ValueError("raw and calibrated knots are written together")
    columns = [f"q_{level:.2f}" for level in QUANTILES]
    missing = [column for column in columns if column not in oof_cal]
    if missing:
        raise KeyError(f"calibrated walk-forward frame missing {missing}")
    if len(oof_cal) != len(walk_forward_rows):
        raise ValueError("calibrated walk-forward rows do not match raw rows")
    for column in columns:
        walk = np.asarray(oof_cal[column], dtype=float)
        if isinstance(preds_ho_cal, pd.DataFrame):
            held = preds_ho_cal[column].to_numpy(dtype=float)
        else:
            held = np.asarray(preds_ho_cal[column], dtype=float)
        if len(held) != len(holdout_rows):
            raise ValueError("calibrated holdout rows do not match raw rows")
        frame[f"cal_{column}"] = np.concatenate([walk, held])
    return frame


def write_oos_predictions(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
