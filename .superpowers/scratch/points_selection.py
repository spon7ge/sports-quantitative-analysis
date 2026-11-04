"""Check the points feature set on season folds. The 2026-27 holdout is never fit.

Stages:
  keep    five-seed q50 loss, full 22 features versus the old keep=True seven
  shap    q10/q50/q90 SHAP on the last season fold, including q10-q90 width
  groups  drop-one-correlated-group ablation, mean pinball, four season folds
  new     implied total, early-season context, and teammates-out, same rule
  all     the four stages in that order

Finished fits are skipped from .superpowers/scratch/points_selection/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(__file__).resolve().parent
ROOT = SCRATCH.parents[1]
PKG = ROOT / "notebooks" / "nba" / "points" / "src"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT))

from points_xgb.config import (  # noqa: E402
    HOLDOUT_SEASON,
    ID_COLS,
    QUANTILES,
    ROLE_COL,
    XGB_PARAMS,
)
from points_xgb.data import load_regular_season  # noqa: E402
from points_xgb.features import (  # noqa: E402
    apply_train_tukey,
    build_model_frame,
    positive_minutes,
)
from points_xgb.importance import shap_center_and_width  # noqa: E402
from points_xgb.paths import repo_root  # noqa: E402
from models.shared.metrics import pinball_loss  # noqa: E402
from models.shared.selection import (  # noqa: E402
    decide_groups,
    feature_groups,
    group_ablation,
)
from models.shared.splits import (  # noqa: E402
    prepare_splits,
    regular_season_mask,
    season_walk_forward_folds,
)
from models.shared.train import fit_quantile_models  # noqa: E402

OUT = SCRATCH / "points_selection"
CACHE = OUT / "fits"
TARGET = "pts"
VAL_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
SEED_FLOOR = (0, 1, 2, 3, 4)
GROUP_SEEDS = (0, 1, 2)

# The points notebook's direct-points list. This is not the rate model's
# PTS_FEATURES, and it is not filtered by the old keep column.
PTS_FEATURES = [
    "pts_ewm_hl_10",
    "fga_ewm_hl_10",
    "fta_ewm_hl_10",
    "fg3a_ewm_hl_10",
    "poss_ewm_hl_10",
    "tchs_ewm_hl_10",
    "minutes_ewm_10",
    "pts_lag_1",
    "min_lag_1",
    "fga_lag_1",
    "min_sd_10",
    "fga_sd_10",
    "player_fga_share_10",
    "usg_wmean_10",
    "usg_x_ts_pct_10",
    "starting",
    "starter_rank",
    "start_rate_10",
    "player_team_spread",
    "game_total",
    "opp_def_rating_mean_10",
    "opp_pace_10",
]

# The seven columns the old rule kept. pts_ewm_hl_10 is absent on purpose.
KEEP_FEATURES = [
    "starting",
    "player_fga_share_10",
    "start_rate_10",
    "min_lag_1",
    "starter_rank",
    "player_team_spread",
    "opp_def_rating_mean_10",
]

MARKET_GROUP = [
    "implied_team_total",
    "abs_spread",
    "fga_share_x_implied_total",
]
EARLY_GROUP = [
    "games_played_this_season",
    "changed_team",
    "prior_season_pts_mean",
]
TEAMMATES_GROUP = [
    "main_starters_out",
    "starter_min_missing_teammates",
]
EXTRA_FEATURES = MARKET_GROUP + EARLY_GROUP + TEAMMATES_GROUP


def _stamp(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_train_pool() -> tuple[pd.DataFrame, list[dict]]:
    """Featured train pool with the holdout season removed, plus season folds."""
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "train_pool.parquet"
    if cache.exists():
        pool = pd.read_parquet(cache)
        _stamp(f"loaded cached train pool: {len(pool):,} rows")
    else:
        _stamp("building features")
        frame = build_model_frame(load_regular_season(repo_root()))
        before = len(frame)
        frame = positive_minutes(frame)
        _stamp(f"dropped {before - len(frame):,} zero-minute rows, {len(frame):,} left")
        frame = frame.sort_values(
            ["game_date", "game_id", "player_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        missing = [column for column in PTS_FEATURES + EXTRA_FEATURES if column not in frame.columns]
        if missing:
            raise KeyError(f"features missing from the panel: {missing}")
        splits = prepare_splits(
            frame,
            holdout_season=HOLDOUT_SEASON,
            features=PTS_FEATURES + EXTRA_FEATURES,
            target_col=TARGET,
            id_cols=ID_COLS,
            role_col=ROLE_COL,
            extra_keep=["season_type"],
        )
        pool, low, high, dropped = apply_train_tukey(
            splits["train_df"],
            target_col=TARGET,
        )
        _stamp(
            f"Tukey fence on train-pool points [{low:.2f}, {high:.2f}] "
            f"dropped {dropped:,} rows; {len(pool):,} remain"
        )
        if pool["season_year"].astype(str).eq(HOLDOUT_SEASON).any():
            raise RuntimeError("holdout rows entered the train pool")
        pool.to_parquet(cache, index=False)
    if pool["season_year"].astype(str).eq(HOLDOUT_SEASON).any():
        raise RuntimeError("holdout rows entered the train pool")
    if not np.isfinite(pool[TARGET].to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite points in the train pool")
    folds = list(
        season_walk_forward_folds(
            pool,
            season_col="season_year",
            val_seasons=VAL_SEASONS,
            season_type_col="season_type",
        )
    )
    for fold in folds:
        _stamp(
            f"{fold['label']}: train {int(fold['train_mask'].sum()):,}  "
            f"val {int(fold['val_mask'].sum()):,}"
        )
    return pool, folds


def _fit_quantiles(train, val, columns, seed, quantiles) -> np.ndarray:
    params = dict(XGB_PARAMS)
    params["random_state"] = int(seed)
    _models, preds = fit_quantile_models(
        train.loc[:, list(columns)],
        train[TARGET],
        val.loc[:, list(columns)],
        val[TARGET],
        quantiles=list(quantiles),
        xgb_params=params,
        early_stop="train_tail",
        train_dates=train["game_date"],
        regular_mask=regular_season_mask(train["season_type"]),
        X_predict=val.loc[:, list(columns)],
    )
    keys = [f"q_{float(level):.2f}" for level in quantiles]
    grid = np.column_stack([np.asarray(preds[key], dtype=float) for key in keys])
    if grid.shape != (len(val), len(quantiles)) or not np.isfinite(grid).all():
        raise RuntimeError("quantile predictions are the wrong shape or non-finite")
    return np.sort(grid, axis=1)


def cached_fit(train, val, columns, seed, quantiles) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    identity = "|".join(
        [
            str(pd.Timestamp(train["game_date"].min()).date()),
            str(pd.Timestamp(train["game_date"].max()).date()),
            str(len(train)),
            str(pd.Timestamp(val["game_date"].min()).date()),
            str(pd.Timestamp(val["game_date"].max()).date()),
            str(len(val)),
            str(int(seed)),
            ",".join(columns),
            ",".join(f"{float(level):.2f}" for level in quantiles),
        ]
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    path = CACHE / f"{digest}.npy"
    if path.exists():
        grid = np.load(path)
        if grid.shape == (len(val), len(quantiles)) and np.isfinite(grid).all():
            return grid
    started = time.perf_counter()
    grid = _fit_quantiles(train, val, columns, seed, quantiles)
    np.save(path, grid)
    _stamp(
        f"fit {len(columns)} cols x {len(quantiles)} quantiles, "
        f"seed {seed}, val {len(val):,} in {time.perf_counter() - started:.0f}s"
    )
    return grid


def last_season_fold(folds: list[dict]) -> dict:
    chosen = [fold for fold in folds if str(fold["label"]).endswith(VAL_SEASONS[-1])]
    if len(chosen) != 1:
        raise RuntimeError(f"expected one {VAL_SEASONS[-1]} fold, found {len(chosen)}")
    return chosen[0]


def q50_loss(pool, fold, columns, seed) -> float:
    train = pool.loc[fold["train_mask"]]
    val = pool.loc[fold["val_mask"]]
    grid = cached_fit(train, val, columns, seed, [0.50])
    loss = pinball_loss(val[TARGET].to_numpy(dtype=float), grid[:, 0], 0.50)
    if not np.isfinite(loss):
        raise RuntimeError("non-finite q50 pinball")
    return float(loss)


def run_keep(pool, folds) -> dict:
    fold = last_season_fold(folds)
    _stamp(f"q50 seed check on {fold['label']}")
    rows = []
    for seed in SEED_FLOOR:
        full = q50_loss(pool, fold, PTS_FEATURES, seed)
        kept = q50_loss(pool, fold, KEEP_FEATURES, seed)
        rows.append({"seed": int(seed), "full": full, "keep": kept, "keep_minus_full": kept - full})
        _stamp(f"seed {seed}: full {full:.4f}  keep {kept:.4f}  worse by {kept - full:.4f}")
        if not 0.5 <= full <= 8.0:
            raise RuntimeError(f"full-model q50 pinball {full:.3f} is outside the expected range")
    table = pd.DataFrame(rows)
    diff = table["keep_minus_full"].to_numpy(dtype=float)
    summary = {
        "fold": fold["label"],
        "loss": "q50 pinball",
        "early_stop": "train_tail",
        "seeds": list(SEED_FLOOR),
        "full_mean": float(table["full"].mean()),
        "full_sd": float(table["full"].std(ddof=1)),
        "keep_mean": float(table["keep"].mean()),
        "keep_sd": float(table["keep"].std(ddof=1)),
        "paired_mean": float(diff.mean()),
        "paired_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "rows": rows,
    }
    path = OUT / "keep_vs_full.json"
    path.write_text(json.dumps(summary, indent=2))
    _stamp(
        f"seed sd of the full model {summary['full_sd']:.4f}; "
        f"keep set is worse by {summary['paired_mean']:.4f} "
        f"(SE {summary['paired_se']:.4f})"
    )
    return summary


def _shap_matrix(model, frame: pd.DataFrame) -> np.ndarray:
    import shap

    values = shap.TreeExplainer(model).shap_values(frame)
    if hasattr(values, "values"):
        values = values.values
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(frame), frame.shape[1]) or not np.isfinite(matrix).all():
        raise RuntimeError("SHAP matrix is the wrong shape or non-finite")
    return matrix


def run_shap(pool, folds) -> pd.DataFrame:
    fold = last_season_fold(folds)
    train = pool.loc[fold["train_mask"]]
    val = pool.loc[fold["val_mask"]]
    levels = [0.10, 0.50, 0.90]
    params = dict(XGB_PARAMS)
    params["random_state"] = 0
    _stamp(f"width SHAP on {fold['label']}, seed 0")
    models, _preds = fit_quantile_models(
        train.loc[:, PTS_FEATURES],
        train[TARGET],
        val.loc[:, PTS_FEATURES],
        val[TARGET],
        quantiles=levels,
        xgb_params=params,
        early_stop="train_tail",
        train_dates=train["game_date"],
        regular_mask=regular_season_mask(train["season_type"]),
        X_predict=val.loc[:, PTS_FEATURES],
    )
    sample = val.loc[:, PTS_FEATURES]
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=0)
    explained = {
        f"q_{level:.2f}": _shap_matrix(models[f"q_{level:.2f}"], sample)
        for level in levels
    }
    table = shap_center_and_width(explained, sample.columns)
    table.to_csv(OUT / "shap_center_width.csv")
    _stamp("width SHAP, highest first:")
    print(table.round(4).to_string(), flush=True)
    return table


def _run_groups(pool, folds, features, groups, stem: str) -> pd.DataFrame:
    levels = list(QUANTILES)

    def fit_predict(train, val, columns, seed):
        return cached_fit(train, val, columns, seed, levels)

    _stamp(f"{stem}: {len(groups)} groups, {len(folds)} folds, seeds {list(GROUP_SEEDS)}")
    for group in groups:
        _stamp("  " + "+".join(group))
    records = group_ablation(
        pool,
        features,
        groups,
        folds,
        fit_predict,
        levels,
        seeds=GROUP_SEEDS,
        target_col=TARGET,
    )
    decision = decide_groups(records)
    records.to_csv(OUT / f"{stem}_folds.csv", index=False)
    decision.to_csv(OUT / f"{stem}_decision.csv", index=False)
    _stamp(stem)
    print(decision.round(5).to_string(index=False), flush=True)
    return decision


def run_groups(pool, folds) -> pd.DataFrame:
    # Correlation is fit on the train pool only. The holdout is already gone.
    groups = feature_groups(pool.loc[:, PTS_FEATURES], min_abs_rho=0.8)
    (OUT / "feature_groups.json").write_text(json.dumps(groups, indent=2))
    return _run_groups(pool, folds, PTS_FEATURES, groups, "groups")


def run_new(pool, folds) -> pd.DataFrame:
    features = PTS_FEATURES + EXTRA_FEATURES
    return _run_groups(
        pool,
        folds,
        features,
        [MARKET_GROUP, EARLY_GROUP, TEAMMATES_GROUP],
        "new_groups",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=("keep", "shap", "groups", "new", "all"),
    )
    args = parser.parse_args()
    pool, folds = load_train_pool()
    if args.stage in ("keep", "all"):
        run_keep(pool, folds)
    if args.stage in ("shap", "all"):
        run_shap(pool, folds)
    if args.stage in ("groups", "all"):
        run_groups(pool, folds)
    if args.stage in ("new", "all"):
        run_new(pool, folds)


if __name__ == "__main__":
    main()
