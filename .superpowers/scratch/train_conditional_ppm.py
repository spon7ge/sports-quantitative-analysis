"""Walk-forward conditional PPM. Does not write existing bundles or OOS files.

The label is min(points / minutes, 6), matching the saved 2026-04-12 model.
``minutes_given`` is realized minutes in training and is the only new feature.
The 2026-27 season is the holdout: it is predicted by the final fit and is
not in any walk-forward training row. 2025-26 stays in the training pool.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "nba" / "points" / "src"))

from models.shared.metrics import pinball_loss  # noqa: E402
from models.shared.oos import capped_rate  # noqa: E402
from models.shared.splits import date_walk_forward_folds, season_holdout_split  # noqa: E402
from models.shared.train import fit_quantile_models  # noqa: E402
from points_xgb.data import load_regular_season  # noqa: E402
from points_xgb.features import build_model_frame, positive_minutes  # noqa: E402

OUT = ROOT / ".superpowers" / "scratch" / "stage2"
SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
)
HOLDOUT_SEASON = "2026-27"
HOLDOUT_START = np.datetime64("2026-10-20")
LABEL_CAP = 6.0
QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
FEATURES = [
    "pts_per_min_ewm_hl_10",
    "player_fga_share_10",
    "ppm_trim_mean_20",
    "current_team_pts_per_min",
    "usg_wmean_10",
    "season_pts_per_min",
    "starting",
    "fta_per_min_10",
    "fga_per_min_10",
    "fg3a_per_min_10",
    "ppm_unstable",
    "opp_def_rating_mean_10",
    "min_mean_10",
    "minutes_given",
]
XGB_PARAMS = {
    "objective": "reg:quantileerror",
    "n_estimators": 1695,
    "max_depth": 3,
    "learning_rate": 0.052852647946832816,
    "subsample": 0.5916232263322159,
    "colsample_bytree": 0.6284413523882679,
    "reg_alpha": 0.022871045941195746,
    "reg_lambda": 2.0960269065270034,
    "min_child_weight": 8,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
}
ID_KEEP = [
    "game_id",
    "player_id",
    "game_date",
    "season_year",
    "starting",
    "minutes",
    "pts",
    "pts_per_min",
    "minutes_given",
]


def prepare_frame() -> pd.DataFrame:
    cached = OUT / "frame.parquet"
    if cached.exists():
        frame = pd.read_parquet(cached)
        seasons = set(frame["season_year"].astype(str))
        if HOLDOUT_SEASON in seasons and "2025-26" in seasons:
            print(f"loaded {cached} rows={len(frame):,}", flush=True)
            return frame
        print("cached frame does not include 2025-26 and the 2026-27 holdout; rebuilding", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_regular_season(ROOT, SEASONS)
    print(f"loaded seasons {SEASONS[0]}–{SEASONS[-1]}: {len(raw):,} rows", flush=True)
    featured = build_model_frame(raw)
    before = len(featured)
    featured = positive_minutes(featured)
    print(f"dropped minutes <= 0: {before - len(featured):,}", flush=True)
    featured = featured.sort_values(
        ["game_date", "game_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    rate = featured["pts"].to_numpy(dtype=float) / featured["minutes"].to_numpy(dtype=float)
    featured["pts_per_min"] = capped_rate(
        featured["pts"], featured["minutes"], cap=LABEL_CAP
    )
    capped = int((rate > LABEL_CAP).sum())
    print(f"capped pts/minutes at {LABEL_CAP}: {capped:,}", flush=True)
    featured["minutes_given"] = featured["minutes"].to_numpy(dtype=float)
    if featured.duplicated(["game_id", "player_id"]).any():
        raise AssertionError("duplicate game_id, player_id")
    missing = [column for column in FEATURES if column not in featured.columns]
    if missing:
        raise KeyError(f"missing features {missing}")
    keep = list(dict.fromkeys([*ID_KEEP, *FEATURES]))
    frame = featured.loc[:, keep].copy()
    frame.to_parquet(cached, index=False)
    print(f"wrote {cached} rows={len(frame):,}", flush=True)
    return frame


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, holdout = season_holdout_split(frame, holdout_season=HOLDOUT_SEASON)
    train_dates = np.asarray(pd.to_datetime(train["game_date"]), dtype="datetime64[ns]")
    holdout_dates = np.asarray(pd.to_datetime(holdout["game_date"]), dtype="datetime64[ns]")
    if np.any(train_dates >= HOLDOUT_START):
        raise AssertionError("holdout date in the training pool")
    if holdout_dates.size and np.any(holdout_dates < HOLDOUT_START):
        raise AssertionError("pre-holdout date in the holdout")
    print(
        f"train {len(train):,} {train['game_date'].min().date()}–{train['game_date'].max().date()}  "
        f"holdout {len(holdout):,}",
        flush=True,
    )
    return train, holdout


def _assert_same_windows(oof: pd.DataFrame) -> None:
    minutes = pd.read_parquet(
        ROOT / "data" / "oos" / "nba" / "minutes_predictions.parquet",
        columns=["game_date", "window_id", "is_holdout"],
    )
    minutes = minutes.loc[~minutes["is_holdout"].astype(bool)]
    for window, group in oof.groupby("fold_id"):
        got = set(pd.to_datetime(group["game_date"]).dt.normalize())
        expected = set(
            pd.to_datetime(
                minutes.loc[minutes["window_id"] == window, "game_date"]
            ).dt.normalize()
        )
        if got != expected:
            raise AssertionError(
                f"fold {window} dates differ: missing {len(expected - got)} extra {len(got - expected)}"
            )


def _oof_frame(val: pd.DataFrame, preds: dict, fold_id: int) -> pd.DataFrame:
    out = val.loc[:, ID_KEEP].copy()
    out["fold_id"] = fold_id
    out["early_stop"] = "train_tail"
    out["train_tail_frac"] = 0.10
    for key, values in preds.items():
        out[key] = np.asarray(values, dtype=float)
    return out


def _fit_fold(train: pd.DataFrame, fold_info: dict, params: dict) -> pd.DataFrame:
    train_mask = fold_info["train_mask"]
    val_mask = fold_info["val_mask"]
    train_rows = train.loc[train_mask]
    val_rows = train.loc[val_mask]
    overlap = set(pd.to_datetime(train_rows["game_date"]).dt.normalize()) & set(
        pd.to_datetime(val_rows["game_date"]).dt.normalize()
    )
    if overlap:
        raise AssertionError(f"fold {fold_info['fold']} shares dates with its training rows")
    models, preds = fit_quantile_models(
        train_rows.loc[:, FEATURES],
        train_rows["pts_per_min"],
        val_rows.loc[:, FEATURES],
        val_rows["pts_per_min"],
        quantiles=QUANTILES,
        xgb_params=params,
        early_stop="train_tail",
        train_dates=train_rows["game_date"],
        train_tail_frac=0.10,
        X_predict=val_rows.loc[:, FEATURES],
    )
    pinball = pinball_loss(val_rows["pts_per_min"], preds["q_0.50"], 0.50)
    print(f"  fold {fold_info['fold']} pinball q50 {pinball:.4f} n={len(val_rows):,}", flush=True)
    destination = OUT / "folds" / f"fold_{fold_info['fold']}.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"models": models, "features": FEATURES, "params": params, "pinball_q50": pinball},
        destination,
    )
    part = _oof_frame(val_rows, preds, fold_info["fold"])
    part_path = OUT / "folds" / f"oof_{fold_info['fold']}.parquet"
    part.to_parquet(part_path, index=False)
    del models
    return part


def train(frame: pd.DataFrame, params: dict | None = None) -> None:
    params = dict(XGB_PARAMS if params is None else params)
    train_pool, holdout = _split(frame)
    parts = []
    for fold_info in date_walk_forward_folds(train_pool, n_folds=5):
        fold_id = fold_info["fold"]
        part_path = OUT / "folds" / f"oof_{fold_id}.parquet"
        model_path = OUT / "folds" / f"fold_{fold_id}.joblib"
        if part_path.exists() and model_path.exists():
            print(f"fold {fold_id} already fit", flush=True)
            parts.append(pd.read_parquet(part_path))
            continue
        started = time.perf_counter()
        print(fold_info["label"], flush=True)
        parts.append(_fit_fold(train_pool, fold_info, params))
        print(f"  fold {fold_id} elapsed {time.perf_counter() - started:.0f}s", flush=True)
    oof = pd.concat(parts, ignore_index=True)
    if oof.duplicated(["game_id", "player_id"]).any():
        raise AssertionError("duplicate OOF keys")
    _assert_same_windows(oof)
    oof_path = OUT / "oof.parquet"
    oof.to_parquet(oof_path, index=False)
    print(f"wrote {oof_path} rows={len(oof):,}", flush=True)

    holdout_path = OUT / "holdout_models.joblib"
    if not holdout_path.exists():
        print("holdout fit", flush=True)
        models, preds = fit_quantile_models(
            train_pool.loc[:, FEATURES],
            train_pool["pts_per_min"],
            train_pool.loc[:, FEATURES],
            train_pool["pts_per_min"],
            quantiles=QUANTILES,
            xgb_params=params,
            early_stop="train_tail",
            train_dates=train_pool["game_date"],
            train_tail_frac=0.10,
            X_predict=holdout.loc[:, FEATURES],
        )
        pinball = pinball_loss(holdout["pts_per_min"], preds["q_0.50"], 0.50)
        print(f"  holdout pinball q50 {pinball:.4f} (not used to choose)", flush=True)
        joblib.dump(
            {"models": models, "features": FEATURES, "params": params, "pinball_q50": pinball},
            holdout_path,
        )
        predicted = _oof_frame(holdout, preds, fold_id=-1)
        predicted.to_parquet(OUT / "holdout_predictions.parquet", index=False)
    summary = {
        "features": FEATURES,
        "params": params,
        "label_cap": LABEL_CAP,
        "holdout_season": HOLDOUT_SEASON,
        "oof_rows": int(len(oof)),
        "pinball_q50_by_fold": {
            int(fold): float(pinball_loss(group["pts_per_min"], group["q_0.50"], 0.50))
            for fold, group in oof.groupby("fold_id")
        },
    }
    (OUT / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["pinball_q50_by_fold"], indent=2), flush=True)


def smoke(frame: pd.DataFrame) -> None:
    train_pool, _holdout = _split(frame)
    fold_info = next(date_walk_forward_folds(train_pool, n_folds=5))
    sample = train_pool.loc[fold_info["train_mask"]].head(1500)
    val = train_pool.loc[fold_info["val_mask"]].head(200)
    params = dict(XGB_PARAMS)
    params["n_estimators"] = 30
    params["early_stopping_rounds"] = 5
    started = time.perf_counter()
    _models, preds = fit_quantile_models(
        sample.loc[:, FEATURES],
        sample["pts_per_min"],
        val.loc[:, FEATURES],
        val["pts_per_min"],
        quantiles=[0.50],
        xgb_params=params,
        early_stop="train_tail",
        train_dates=sample["game_date"],
        train_tail_frac=0.10,
        X_predict=val.loc[:, FEATURES],
    )
    if not np.isfinite(preds["q_0.50"]).all():
        raise AssertionError("non-finite smoke prediction")
    print(
        f"smoke pinball {pinball_loss(val['pts_per_min'], preds['q_0.50'], 0.50):.4f} "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    prepared = prepare_frame()
    if mode == "smoke":
        smoke(prepared)
    elif mode == "train":
        train(prepared)
    else:
        raise SystemExit(f"unknown mode {mode}")
