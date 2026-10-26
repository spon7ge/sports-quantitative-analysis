"""Walk-forward conformal checks for the minutes notebook.

Offsets stay on out-of-fold rows. These helpers build the base knots,
the go/no-go table, and the nested backtest the notebook displays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from minutes_xgb.paths import ensure_repo_on_path

ensure_repo_on_path()
from models.shared.conformal import base_knots, calibration_group  # noqa: E402
from models.shared.conformal_report import (  # noqa: E402
    ConformalSpec,
    calibration_table as _calibration_table,
    nested_offset_backtest as _nested_offset_backtest,
    offset_stability_flags as _offset_stability_flags,
)
from models.shared.metrics import MINUTE_Q50_TIERS  # noqa: E402


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
            raise RuntimeError(
                f"fold {fold['fold']} leaks or does not match its dates"
            )
    return pd.DataFrame(rows)


def fold_base_knots(
    oof: pd.DataFrame,
    ppm_df: pd.DataFrame,
    train_masks: Sequence,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sort each fold, then fill cold starts from that fold's training rows."""
    fold_ids = sorted(int(value) for value in oof["fold_id"].unique())
    if fold_ids != list(range(1, len(train_masks) + 1)):
        raise RuntimeError("walk-forward fold ids are not 1..n in wf_results order")
    knots = _knot_columns(oof)
    parts = []
    for fold_id, part in oof.groupby("fold_id", sort=True):
        history = ppm_df.loc[train_masks[int(fold_id) - 1], ["starting", "minutes"]]
        built = base_knots(
            part[knots],
            part["starting"].to_numpy(),
            ppm_df.loc[part.index, "min_lag_1"].to_numpy(),
            history,
        )
        parts.append(built)
    base = pd.concat(parts).reindex(oof.index)
    if not np.isfinite(base.to_numpy(dtype=float)).all():
        raise RuntimeError("base knots are not finite")
    group = calibration_group(
        base["q_0.50"].to_numpy(),
        ppm_df.loc[oof.index, "min_lag_1"].to_numpy(),
    )
    return base, group



def _knot_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (column for column in frame.columns if str(column).startswith("q_")),
        key=lambda column: float(str(column).split("_", 1)[1]),
    )


def _minute_spec(clip_high: float, min_n: int = 300) -> ConformalSpec:
    return ConformalSpec(
        target_col="minutes",
        lag_col="min_lag_1",
        tiers=MINUTE_Q50_TIERS,
        clip_low=1e-3,
        clip_high=float(clip_high),
        min_n=int(min_n),
    )


def calibration_table(
    y,
    dates,
    base: pd.DataFrame,
    group,
    *,
    starting,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Per-knot gap and 80/90 coverage, with a date-clustered interval."""
    return _calibration_table(
        y,
        dates,
        base,
        group,
        spec=_minute_spec(1.0),
        starting=starting,
        n_boot=n_boot,
        seed=seed,
    )


def nested_offset_backtest(
    oof: pd.DataFrame,
    base: pd.DataFrame,
    group,
    *,
    folds=(3, 4, 5),
    min_n: int = 300,
    max_minutes: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Fit offsets on folds before k and score fold k.

    ``scores['wis_diff']`` and ``scores['crps_diff']`` are calibrated minus
    raw. Lower is better.
    """
    return _nested_offset_backtest(
        oof,
        base,
        group,
        _minute_spec(max_minutes, min_n=min_n),
        folds=folds,
        n_boot=n_boot,
        seed=seed,
    )


def offset_stability_flags(offset_tables):
    """Knots whose offset range exceeds the mean absolute offset."""
    return _offset_stability_flags(offset_tables)
