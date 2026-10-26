"""Settings for the NBA minutes quantile XGBoost notebook."""

from __future__ import annotations

HOLDOUT_SEASON = "2026-27"
ID_COLS = ["game_id", "player_id", "season_year", "player_name", "game_date"]
TARGET_COL = "minutes"
ROLE_COL = "starting"
SEED = 42
N_BOOT = 2_000

QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

XGB_PARAMS = {
    "objective": "reg:quantileerror",
    "n_estimators": 1324,
    "max_depth": 4,
    "learning_rate": 0.10721421777577989,
    "subsample": 0.822275368040345,
    "colsample_bytree": 0.8426714380124101,
    "reg_alpha": 3.5136107989594274,
    "reg_lambda": 0.08159719988366734,
    "min_child_weight": 4,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
    "seed": SEED,
}

MIN_TIERS = {
    "<15 min": lambda values: values < 15,
    "15-24 min": lambda values: (values >= 15) & (values < 24),
    "24-31 min": lambda values: (values >= 24) & (values < 31),
    "31+ min": lambda values: values >= 31,
}

ARTIFACT_STEM = "min_nba_model"
XGB_NAME = "Quantile XGBoost"

# Selected columns for this notebook. Live scoring still uses the saved
# model's own list until a new bundle is trained.
MIN_FEATURES = [
    "starting",
    "starter_rank",
    "start_rate_10",
    "season_start_rate",
    "starter_min_missing_teammates",
    "total_stars_active",
    "min_avg_lN",
    "min_lag_1",
    "min_ewm_hl_10",
    "player_days_since_appearance",
    "player_appearances_prev_7d",
    "min_p80_10",
    "min_p80_minus_p20_10",
    "is_home",
    "min_mean_vs_opp",
    "games_vs_opp",
]

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
