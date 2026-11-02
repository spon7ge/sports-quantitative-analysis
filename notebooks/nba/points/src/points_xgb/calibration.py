"""Walk-forward conformal checks for capped points per minute.

Offsets stay on out-of-fold rows dated before the 2026-27 opener.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from points_xgb.config import (
    HOLDOUT_START,
    LAG_COL,
    QUANTILES,
    RATE_CAP,
    TARGET_COL,
)
from points_xgb.paths import ensure_repo_on_path

ensure_repo_on_path()
from models.shared.conformal import (  # noqa: E402
    apply_offsets,
    base_knots,
    calibration_group,
    calibration_payload,
    fit_offsets,
    role_quantile_table,
)
from models.shared.conformal_report import (  # noqa: E402
    ConformalSpec,
    calibration_table as _calibration_table,
    nested_offset_backtest as _nested_offset_backtest,
)
from models.shared.metrics import RATE_Q50_TIERS  # noqa: E402

RATE_SPEC = ConformalSpec(
    target_col=TARGET_COL,
    lag_col=LAG_COL,
    tiers=RATE_Q50_TIERS,
    clip_low=0.0,
    clip_high=float(RATE_CAP),
    min_n=300,
)
TAIL_FOLDS = (1, 2, 3, 4)


def _active_spec(spec: ConformalSpec | None) -> ConformalSpec:
    return RATE_SPEC if spec is None else spec


def oof_player_games(oof: pd.DataFrame, ppm_df: pd.DataFrame) -> pd.DataFrame:
    """Player-games for the OOF rows. Duplicate rows are an error."""
    if oof.index.duplicated().any():
        raise RuntimeError("OOF player-games are duplicated; not deduping")
    ids = ppm_df.loc[oof.index, ["game_id", "player_id"]]
    if len(ids) != len(oof) or ids.duplicated(["game_id", "player_id"]).any():
        raise RuntimeError("OOF player-games are duplicated; not deduping")
    return ids


def oof_leakage_table(
    oof: pd.DataFrame,
    ppm_df: pd.DataFrame,
    folds: Sequence[Mapping],
) -> pd.DataFrame:
    """One row per fold. A validation date inside train or the stop tail raises."""
    tail_frac = float(pd.unique(oof["train_tail_frac"])[0])
    rows = []
    for fold in folds:
        part = oof.loc[oof["fold_id"].eq(fold["fold"])]
        oof_dates = set(pd.to_datetime(part["game_date"]))
        train_dates = set(pd.to_datetime(np.asarray(fold["train_dates"])))
        val_dates = set(pd.to_datetime(np.asarray(fold["val_dates"])))
        unique = pd.unique(pd.to_datetime(ppm_df.loc[fold["train_mask"], "game_date"]))
        n_stop = max(1, int(np.floor(len(unique) * tail_frac)))
        stop = set(pd.to_datetime(unique[-n_stop:]))
        rows.append(
            {
                "fold": fold["fold"],
                "train_end": pd.Timestamp(fold["train_dates"][-1]).date(),
                "val_start": pd.Timestamp(fold["val_dates"][0]).date(),
                "val_end": pd.Timestamp(fold["val_dates"][-1]).date(),
                "n": int(len(part)),
                "val_in_train": len(oof_dates & train_dates),
                "val_in_stop": len(oof_dates & stop),
                "dates_match": oof_dates == val_dates,
            }
        )
        if oof_dates & train_dates or oof_dates & stop or oof_dates != val_dates:
            raise RuntimeError(f"fold {fold['fold']} leaks or does not match its dates")
    return pd.DataFrame(rows)


def require_preholdout(oof: pd.DataFrame) -> None:
    """Raise when a calibration row is on or after opening night 2026-27."""
    dates = pd.to_datetime(oof["game_date"])
    if bool((dates >= pd.Timestamp(HOLDOUT_START)).any()):
        raise ValueError("holdout row in conformal sample")


def _knot_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (column for column in frame.columns if str(column).startswith("q_")),
        key=lambda column: float(str(column).split("_", 1)[1]),
    )


def fold_base_knots(
    oof: pd.DataFrame,
    ppm_df: pd.DataFrame,
    train_masks: Sequence,
    *,
    spec: ConformalSpec | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sort each fold, then fill cold starts from that fold's training rows.

    The default spec is the points-per-minute contract. A points run
    passes its own spec so the history column and lag match the label.
    """
    chosen = _active_spec(spec)
    require_preholdout(oof)
    fold_ids = sorted(int(value) for value in oof["fold_id"].unique())
    if fold_ids != list(range(1, len(train_masks) + 1)):
        raise RuntimeError("walk-forward fold ids are not 1..n in wf_results order")
    knots = _knot_columns(oof)
    parts = []
    for fold_id, part in oof.groupby("fold_id", sort=True):
        history = ppm_df.loc[
            train_masks[int(fold_id) - 1], ["starting", chosen.target_col]
        ]
        built = base_knots(
            part[knots],
            part["starting"].to_numpy(),
            ppm_df.loc[part.index, chosen.lag_col].to_numpy(),
            history,
            target_col=chosen.target_col,
        )
        parts.append(built)
    base = pd.concat(parts).reindex(oof.index)
    if not np.isfinite(base.to_numpy(dtype=float)).all():
        raise RuntimeError("base knots are not finite")
    group = calibration_group(
        base["q_0.50"].to_numpy(),
        ppm_df.loc[oof.index, chosen.lag_col].to_numpy(),
        tiers=chosen.tiers,
    )
    return base, group


def calibration_table(
    y,
    dates,
    base,
    group,
    *,
    starting,
    n_boot: int,
    seed: int,
    spec: ConformalSpec | None = None,
):
    """Per-knot gap on the given spec, or the rate spec when omitted."""
    return _calibration_table(
        y,
        dates,
        base,
        group,
        spec=_active_spec(spec),
        starting=starting,
        n_boot=n_boot,
        seed=seed,
    )


def nested_offset_backtest(
    oof: pd.DataFrame,
    base: pd.DataFrame,
    group,
    *,
    folds: Sequence[int] = (3, 4, 5),
    n_boot: int,
    seed: int,
    spec: ConformalSpec | None = None,
) -> dict:
    """Fit offsets on earlier folds and score fold k."""
    require_preholdout(oof)
    return _nested_offset_backtest(
        oof,
        base,
        group,
        _active_spec(spec),
        folds=folds,
        n_boot=n_boot,
        seed=seed,
    )


def holdout_role_quantiles(
    ppm_df: pd.DataFrame,
    *,
    spec: ConformalSpec | None = None,
) -> pd.DataFrame:
    """Starter and bench quantiles from the pre-holdout pool."""
    chosen = _active_spec(spec)
    return role_quantile_table(
        ppm_df[["starting", chosen.target_col]],
        QUANTILES,
        target_col=chosen.target_col,
    )


def rate_payload(
    offsets,
    groups,
    dates,
    role_quantiles,
    *,
    spec: ConformalSpec | None = None,
) -> dict:
    """Payload stored on the points bundle."""
    chosen = _active_spec(spec)
    return calibration_payload(
        offsets,
        groups,
        dates,
        min_n=chosen.min_n,
        role_quantiles=role_quantiles,
        max_minutes=chosen.clip_high,
        target=chosen.target_col,
        lag_col=chosen.lag_col,
        tiers=chosen.tiers,
        clip_low=chosen.clip_low,
    )


def holdout_calibrated(
    preds_ho: Mapping[str, np.ndarray] | pd.DataFrame,
    ppm_holdout: pd.DataFrame,
    offsets: pd.DataFrame,
    role_quantiles: pd.DataFrame,
    *,
    spec: ConformalSpec | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Base knots, groups, and clipped offsets for the holdout. No fit."""
    chosen = _active_spec(spec)
    base = base_knots(
        preds_ho,
        ppm_holdout["starting"],
        ppm_holdout[chosen.lag_col],
        role_quantiles=role_quantiles,
        target_col=chosen.target_col,
    )
    group = calibration_group(
        base["q_0.50"],
        ppm_holdout[chosen.lag_col],
        tiers=chosen.tiers,
    )
    calibrated = apply_offsets(
        base,
        group,
        offsets,
        chosen.clip_high,
        lower=chosen.clip_low,
    )
    return base, group, calibrated


def fit_rate_offsets(
    base: pd.DataFrame,
    oof: pd.DataFrame,
    group,
    *,
    spec: ConformalSpec | None = None,
) -> pd.DataFrame:
    """One offset table on every pre-holdout out-of-fold row."""
    chosen = _active_spec(spec)
    require_preholdout(oof)
    knots = _knot_columns(base)
    levels = {key: float(key.split("_", 1)[1]) for key in knots}
    return fit_offsets(
        base,
        oof[chosen.target_col],
        group,
        levels,
        min_n=chosen.min_n,
        group_order=(
            "cold_start",
            *[name for name, _low, _high in chosen.tiers],
        ),
    )


def calibrated_tail_frame(
    oof: pd.DataFrame,
    calibrated: pd.DataFrame,
    *,
    folds: Sequence[int] = TAIL_FOLDS,
    spec: ConformalSpec | None = None,
) -> pd.DataFrame:
    """Calibrated knots with the outcome column renamed to ``rate``.

    Fold 5 stays in the notebook's miss-rate table and is left out of
    the sidecar ``ppm_sampler.load_tail_sidecar`` can load.
    """
    active = _active_spec(spec)
    require_preholdout(oof)
    keep = oof["fold_id"].isin(list(folds))
    chosen = oof.loc[keep].copy()
    knots = _knot_columns(calibrated)
    for column in knots:
        chosen[column] = calibrated.loc[keep, column].to_numpy(dtype=float)
    chosen["rate"] = chosen[active.target_col].to_numpy(dtype=float)
    return chosen.drop(columns=[active.target_col]).reset_index(drop=True)
