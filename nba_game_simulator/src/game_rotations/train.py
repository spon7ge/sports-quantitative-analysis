"""Fit the conditional minutes model on separate dates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from role_context import OT_MIN, REG_MIN

from game_rotations.checks import (
    _as_float,
    _dates,
    _frame,
    _normalize_keys,
    _positive_integer,
    _required,
)
from game_rotations.constants import MODEL_VERSION, PREGAME, _FEATURES, _TOL
from game_rotations.model import (
    _ConditionalMinutes,
    _RecentMinutes,
    _choose_alpha,
    _recent_mean,
    _residual_banks,
    _sigma_table,
    _split_dates,
)
from game_rotations.script import _attach_script
from game_rotations.simulator import RotationSimulator

def fit_rotation(
    roles: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    calibration_fraction: float = 0.15,
    holdout_fraction: float = 0.20,
    min_residual_samples: int = 40,
) -> RotationSimulator:
    """Tune, calibrate and evaluate on separate dates, then refit on all rows.

    Holdout metrics use observed game state and positive appearances. They
    are not pregame distribution metrics or metrics after team allocation.
    """
    _frame(roles, "roles")
    _required(roles, ["minutes", "game_date", "player_id", "team_id", "game_id"])
    _required(roles, PREGAME)
    _positive_integer(min_residual_samples, "min_residual_samples")

    usable = roles.copy()
    _normalize_keys(usable, ["player_id", "team_id", "game_id"])

    if usable.duplicated(["player_id", "team_id", "game_id"]).any():
        raise ValueError("training roles contain repeated player-games")

    usable["game_date"] = _dates(usable["game_date"])
    usable["minutes"] = _as_float(usable["minutes"])
    _required(usable, ["team_final_margin", "game_minutes"])

    if "n_ot" not in usable:
        usable["n_ot"] = (_as_float(usable["game_minutes"]) - REG_MIN) / OT_MIN

    _attach_script(usable, validate_existing=True)

    if usable.groupby("game_id")["game_date"].nunique().gt(1).any():
        raise ValueError("one game_id has multiple game dates")

    script = ["team_final_margin", "game_minutes", "n_ot"]
    if usable.groupby(["game_id", "team_id"])[script].nunique().gt(1).any().any():
        raise ValueError("teammates have conflicting historical game states")

    labels = usable["minutes"].to_numpy()
    finite = np.isfinite(labels)

    if np.any(finite & ((labels < 0) | (labels > usable["game_minutes"] + _TOL))):
        raise ValueError("official minutes must be between zero and game duration")

    excluded_unknown = int((~finite).sum())
    excluded_zero = int((finite & (labels == 0)).sum())

    usable = usable.loc[finite & (labels > 0)].copy()
    usable = usable.sort_values(["game_date", "game_id", "player_id"], kind="stable")

    if len(usable) < 4:
        raise ValueError("not enough positive appearances for chronological fitting")

    train, tune, calibration, holdout = _split_dates(
        usable, validation_fraction, calibration_fraction, holdout_fraction
    )

    if len(train) < 2:
        raise ValueError("the training period needs at least two player-games")

    alpha = _choose_alpha(train, tune)
    learned = pd.concat([train, tune], ignore_index=True)
    probe = _ConditionalMinutes.fit(learned, alpha)

    fallback_mu = float(learned["minutes"].median())
    fallback_sd = max(float(learned["minutes"].std(ddof=0)), 1.0)

    pre_mu = _recent_mean(calibration, fallback_mu)
    cond_mu = probe.predict(calibration)

    residual_pre = _residual_banks(
        pre_mu,
        _as_float(calibration["minutes"]) - pre_mu,
        min_residual_samples,
    )
    residual_cond = _residual_banks(
        pre_mu,
        _as_float(calibration["minutes"]) - cond_mu,
        min_residual_samples,
    )

    test_target = _as_float(holdout["minutes"])
    pre_test = _recent_mean(holdout, fallback_mu)
    cond_test = np.clip(
        probe.predict(holdout),
        0,
        _as_float(holdout["game_minutes"]),
    )

    metadata = {
        "version": MODEL_VERSION,
        "alpha": alpha,
        "feature_columns": list(_FEATURES),
        "training_cutoff": usable["game_date"].max().date().isoformat(),
        "regulation_minutes": float(REG_MIN),
        "ot_minutes": float(OT_MIN),
        "excluded_unknown_labels": excluded_unknown,
        "excluded_zero_labels": excluded_zero,
        "split": {
            name: {
                "start": frame["game_date"].min().date().isoformat(),
                "end": frame["game_date"].max().date().isoformat(),
                "rows": len(frame),
                "dates": frame["game_date"].nunique(),
            }
            for name, frame in (
                ("train", train),
                ("validation", tune),
                ("calibration", calibration),
                ("holdout", holdout),
            )
        },
        "holdout": {
            "label": "positive appearances; observed state; before allocation",
            "recent_minutes_mae": float(np.mean(np.abs(pre_test - test_target))),
            "conditional_observed_state_mae": float(
                np.mean(np.abs(cond_test - test_target))
            ),
            "conditional_observed_state_rmse": float(
                np.sqrt(np.mean((cond_test - test_target) ** 2))
            ),
        },
        "calibration_role_counts": residual_cond["counts"].tolist(),
        "min_residual_samples": int(min_residual_samples),
    }

    return RotationSimulator(
        pre_model=_RecentMinutes(),
        cond_model=_ConditionalMinutes.fit(usable, alpha),
        sigma_pre=_sigma_table(
            pre_mu,
            _as_float(calibration["minutes"]) - pre_mu,
        ),
        sigma_cond=_sigma_table(
            pre_mu,
            _as_float(calibration["minutes"]) - cond_mu,
        ),
        residual_pre=residual_pre,
        residual_cond=residual_cond,
        fallback_mu=float(usable["minutes"].median()),
        fallback_sd=fallback_sd,
        metadata=metadata,
    )
