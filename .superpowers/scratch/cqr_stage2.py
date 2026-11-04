"""Date-tail holdout, calibrated knots, and the OOS table.

Does not write ``models/saved_models/min_nba_model_2026-04-12.joblib``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks/nba/minutes/src"))

from minutes_xgb.artifacts import (  # noqa: E402
    build_oos_predictions,
    write_oos_predictions,
)
from minutes_xgb.config import (  # noqa: E402
    HOLDOUT_SEASON,
    ID_COLS,
    MIN_TIERS,
    QUANTILES,
    ROLE_COL,
    TARGET_COL,
    XGB_PARAMS,
)
from minutes_xgb.data import load_regular_season  # noqa: E402
from minutes_xgb.features import build_model_frame, positive_minutes  # noqa: E402
from models.shared.splits import prepare_splits  # noqa: E402
from models.shared.train import evaluate_holdout  # noqa: E402

PAYLOAD = ROOT / ".superpowers/scratch/cqr_wf.joblib"
HOLDOUT_CACHE = ROOT / ".superpowers/scratch/cqr_holdout.joblib"
NB_PATH = ROOT / "notebooks/nba/minutes/02_minutes_xgboost.ipynb"


def display(obj) -> None:
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


def main() -> None:
    payload = joblib.load(PAYLOAD)
    features = list(payload["features"])
    print(f"features ({len(features)}): {features}", flush=True)
    frame = positive_minutes(build_model_frame(load_regular_season(ROOT)))
    splits = prepare_splits(
        frame,
        holdout_season=HOLDOUT_SEASON,
        features=features,
        target_col=TARGET_COL,
        id_cols=ID_COLS,
        role_col=ROLE_COL,
        extra_keep=["minutes", "pts", "min_ewm_hl_10"],
    )
    ppm_df = splits["ppm_df"]
    ppm_holdout = splits["ppm_holdout"]
    oof = payload["oof"]
    if not oof.index.isin(ppm_df.index).all():
        raise RuntimeError("cached OOF index is not in this train pool")
    if int(oof.index.duplicated().sum()):
        raise RuntimeError("cached OOF index is duplicated")
    cached_ids = payload["ppm"].loc[oof.index, ["game_id", "player_id"]]
    fresh_ids = ppm_df.loc[oof.index, ["game_id", "player_id"]]
    if not cached_ids.reset_index(drop=True).equals(fresh_ids.reset_index(drop=True)):
        raise RuntimeError("cached OOF player-games do not match the train pool")

    wf_results = []
    for fold_id in range(1, 6):
        mask = np.asarray(payload["train_masks"][fold_id], dtype=bool)
        if len(mask) != len(ppm_df):
            raise RuntimeError(f"fold {fold_id} train mask length does not match")
        wf_results.append(
            {"train_mask": pd.Series(mask, index=ppm_df.index)}
        )
    wf = {"oof": oof, "fold_ranges": payload["fold_ranges"], "wf_results": wf_results}

    if HOLDOUT_CACHE.exists():
        print("loading cached date-tail holdout", flush=True)
        cached = joblib.load(HOLDOUT_CACHE)
        preds_ho = cached["preds_ho"]
        ho_metrics = cached["ho_metrics"]
        y_ho = cached["y_ho"]
    else:
        holdout = evaluate_holdout(
            ppm_df,
            ppm_holdout,
            features=features,
            target_col=TARGET_COL,
            xgb_params=XGB_PARAMS,
            role_col=ROLE_COL,
            tiers=MIN_TIERS,
            quantiles=QUANTILES,
            fold_label=f"{HOLDOUT_SEASON} Blind Holdout",
        )
        preds_ho = holdout["preds_ho"]
        ho_metrics = holdout["ho_metrics"]
        y_ho = holdout["y_ho"]
        joblib.dump(
            {"preds_ho": preds_ho, "ho_metrics": ho_metrics, "y_ho": y_ho},
            HOLDOUT_CACHE,
        )
        print("HOLDOUT_CACHED", flush=True)

    nb = json.loads(NB_PATH.read_text())
    cell = "".join(nb["cells"][27]["source"])
    write = "".join(nb["cells"][47]["source"])
    if not cell.startswith("# 8c."):
        raise RuntimeError("notebook cell 27 is not the calibrated holdout cell")
    if "oof_cal=oof_cal" not in write:
        raise RuntimeError("notebook cell 47 does not write calibrated knots")
    namespace = {
        "wf": wf,
        "wf_results": wf_results,
        "ppm_df": ppm_df,
        "ppm_holdout": ppm_holdout,
        "preds_ho": preds_ho,
        "y_ho": y_ho,
        "ho_metrics": ho_metrics,
        "ROOT": ROOT,
        "np": np,
        "pd": pd,
        "display": display,
        "build_oos_predictions": build_oos_predictions,
        "write_oos_predictions": write_oos_predictions,
        "QUANTILES": QUANTILES,
        "MIN_TIERS": MIN_TIERS,
    }
    exec(cell, namespace)
    exec(write, namespace)
    print("STAGE2_DONE", flush=True)


if __name__ == "__main__":
    main()
