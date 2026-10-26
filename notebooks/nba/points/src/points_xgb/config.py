"""Settings for the NBA points-per-minute quantile XGBoost notebook."""

from __future__ import annotations

import numpy as np

from points_xgb.paths import ensure_repo_on_path

ensure_repo_on_path()
from models.shared.metrics import RATE_Q50_TIERS  # noqa: E402
from models.shared.oos import RATE_CAP  # noqa: E402

HOLDOUT_SEASON = "2026-27"
ID_COLS = ["game_id", "player_id", "season_year", "player_name", "game_date"]
TARGET_COL = "pts_per_min"
LAG_COL = "pts_per_min_lag_1"
ROLE_COL = "starting"
SEED = 42
N_BOOT = 2_000
RATE_CAP = RATE_CAP

QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

#Best params: {'n_estimators': 1828, 'max_depth': 3, 'learning_rate': 0.08724457596180102, 
# 'subsample': 0.6342688186052617, 'colsample_bytree': 0.8968128251007528, 
# 'reg_alpha': 2.7388114185469976, 'reg_lambda': 0.3192032055207369, 'min_child_weight': 10}

XGB_PARAMS = {
    "objective": "reg:quantileerror",
    "n_estimators": 1828,
    "max_depth": 3,
    "learning_rate": 0.08724457596180102,
    "subsample": 0.5916232263322159,
    "colsample_bytree": 0.8968128251007528,
    "reg_alpha": 2.7388114185469976,
    "reg_lambda": 0.3192032055207369,
    "min_child_weight": 10,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
}

EXCLUDED_FEATURES = frozenset({"minutes", "pts", "predicted_minutes_oof"})

PTS_FEATURES = [
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
    "pts_per_min_mean_vs_opp",
    "fga_per_min_mean_vs_opp",
    "fta_per_min_mean_vs_opp",
    "fg3a_per_min_mean_vs_opp",
    "ppm_unstable",
    "opp_def_rating_mean_10",
    "min_mean_10",
    "min_sd_10",
    "minutes_ewm_10",
    "ts_pct_10",
    "opp_pace_10",
    "ppm_sd_10",
]

ARTIFACT_STEM = "pts_nba_model"
XGB_NAME = "Quantile XGBoost"
PREDICTIONS_PATH = "data/oos/nba/rate_predictions.parquet"
TAIL_SIDECAR = "models/saved_models/pts_nba_tails_cal_2026-04-12.joblib"
HOLDOUT_START = "2026-10-20"

REGULAR_SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27"
)


def _tier_mask(low: float | None, high: float | None):
    def mask(values):
        array = np.asarray(values)
        chosen = np.ones(array.shape[0], dtype=bool)
        if low is not None:
            chosen &= array >= low
        if high is not None:
            chosen &= array < high
        return chosen

    return mask


RATE_TIERS = {
    name: _tier_mask(low, high) for name, low, high in RATE_Q50_TIERS
}
