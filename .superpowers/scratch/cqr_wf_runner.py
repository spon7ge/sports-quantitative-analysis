"""Build the notebook's train pool, check OOF row identity, fit walk-forward.

Writes ``.superpowers/scratch/cqr_wf.joblib``. Does not touch the holdout
fit or ``data/oos``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks/nba/minutes/src"))

from minutes_xgb.config import (  # noqa: E402
    HOLDOUT_SEASON,
    ID_COLS,
    MIN_FEATURES as CONFIG_FEATURES,
    MIN_TIERS,
    QUANTILES,
    ROLE_COL,
    TARGET_COL,
    XGB_PARAMS,
)
from minutes_xgb.data import load_regular_season  # noqa: E402
from minutes_xgb.features import build_model_frame, positive_minutes  # noqa: E402
from models.shared.splits import date_walk_forward_folds, prepare_splits  # noqa: E402
from models.shared.train import run_walk_forward  # noqa: E402

OUT = ROOT / ".superpowers/scratch/cqr_wf.joblib"
PARKED = ("min_mean_vs_opp", "games_vs_opp")
LOCKED = [name for name in CONFIG_FEATURES if name not in PARKED]
PLAY = ["player_team_spread", "min_mean_vs_opp", "games_vs_opp"]
MIN_FEATURES = [*LOCKED, *PLAY]


def _stop_dates(train_dates: pd.Series, train_tail_frac: float = 0.10) -> set:
    dates = pd.to_datetime(np.asarray(train_dates))
    unique = pd.unique(dates)
    n_stop = int(np.floor(len(unique) * train_tail_frac))
    if n_stop < 1:
        n_stop = 1
    return set(pd.to_datetime(unique[-n_stop:]))


def main() -> None:
    print(f"features ({len(MIN_FEATURES)}): {MIN_FEATURES}", flush=True)
    frame = positive_minutes(build_model_frame(load_regular_season(ROOT)))
    splits = prepare_splits(
        frame,
        holdout_season=HOLDOUT_SEASON,
        features=MIN_FEATURES,
        target_col=TARGET_COL,
        id_cols=ID_COLS,
        role_col=ROLE_COL,
        extra_keep=["minutes", "pts", "min_ewm_hl_10"],
    )
    ppm = splits["ppm_df"]
    print(f"train rows {len(ppm):,}", flush=True)

    pieces = []
    leakage = []
    for fold in date_walk_forward_folds(ppm, n_folds=5):
        val_idx = ppm.index[fold["val_mask"]]
        part = ppm.loc[val_idx, ["game_id", "player_id", "game_date"]].copy()
        part["fold_id"] = fold["fold"]
        pieces.append(part)
        train_dates = set(pd.to_datetime(fold["train_dates"]))
        val_dates = set(pd.to_datetime(fold["val_dates"]))
        stop = _stop_dates(ppm.loc[fold["train_mask"], "game_date"])
        oof_dates = set(pd.to_datetime(part["game_date"]))
        leakage.append(
            {
                "fold": fold["fold"],
                "label": fold["label"],
                "train_start": pd.Timestamp(fold["train_dates"][0]).date().isoformat(),
                "train_end": pd.Timestamp(fold["train_dates"][-1]).date().isoformat(),
                "val_start": pd.Timestamp(fold["val_dates"][0]).date().isoformat(),
                "val_end": pd.Timestamp(fold["val_dates"][-1]).date().isoformat(),
                "stop_start": pd.Timestamp(min(stop)).date().isoformat(),
                "stop_end": pd.Timestamp(max(stop)).date().isoformat(),
                "n_val": int(len(part)),
                "n_stop_dates": len(stop),
                "val_in_train": len(oof_dates & train_dates),
                "val_in_stop": len(oof_dates & stop),
                "val_matches_declared": oof_dates == val_dates,
            }
        )
    identity = pd.concat(pieces, axis=0)
    dup = int(identity.duplicated(["game_id", "player_id"]).sum())
    print("PRECHECK duplicated game_id, player_id:", dup, flush=True)
    print(pd.DataFrame(leakage).to_string(index=False), flush=True)
    print("SANITY_DONE", flush=True)

    wf = run_walk_forward(
        splits["X"],
        splits["y"],
        ppm,
        xgb_params=XGB_PARAMS,
        role_col=ROLE_COL,
        tiers=MIN_TIERS,
        quantiles=QUANTILES,
        n_folds=5,
    )
    oof = wf["oof"]
    print("oof columns:", list(oof.columns), flush=True)
    try:
        direct = int(oof.duplicated(["game_id", "player_id"]).sum())
        print("direct duplicated:", direct, flush=True)
    except KeyError as exc:
        print("direct duplicated KeyError:", exc, flush=True)
    aligned = oof.copy()
    aligned[["game_id", "player_id"]] = ppm.loc[
        oof.index, ["game_id", "player_id"]
    ].to_numpy()
    print(
        "aligned duplicated game_id, player_id:",
        int(aligned.duplicated(["game_id", "player_id"]).sum()),
        flush=True,
    )
    print("oof index duplicated:", int(oof.index.duplicated().sum()), flush=True)

    expected = []
    for fold in date_walk_forward_folds(ppm, n_folds=5):
        expected.append(ppm.index[fold["val_mask"]].to_numpy())
    same_index = np.array_equal(oof.index.to_numpy(), np.concatenate(expected))
    print("oof index matches concatenated val masks:", same_index, flush=True)

    payload = {
        "oof": oof,
        "fold_ranges": wf["fold_ranges"],
        "train_masks": {
            index + 1: np.asarray(result["train_mask"], dtype=bool)
            for index, result in enumerate(wf["wf_results"])
        },
        "ppm": ppm[
            ["game_id", "player_id", "game_date", "minutes", "starting", "min_lag_1"]
        ].copy(),
        "features": MIN_FEATURES,
        "leakage": leakage,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, OUT)
    print("WALK_FORWARD_DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
