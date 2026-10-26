"""Notebook entry points for the NBA minutes quantile model.

Import this package from a notebook after putting
``notebooks/nba/minutes/src`` on ``sys.path``. Training and the shared
feature pipeline stay in ``models.shared`` and ``src.features.minutes``.
"""

from minutes_xgb.artifacts import (
    build_oos_predictions,
    exploratory_metadata,
    loaded_holdout_scores,
    write_oos_predictions,
)
from minutes_xgb.baseline import compare_to_baselines
from minutes_xgb.calibration import (
    calibration_table,
    fold_base_knots,
    nested_offset_backtest,
    offset_stability_flags,
    oof_leakage_table,
    oof_player_games,
)
from minutes_xgb.config import (
    ARTIFACT_STEM,
    HOLDOUT_SEASON,
    ID_COLS,
    MIN_FEATURES,
    MIN_TIERS,
    N_BOOT,
    QUANTILES,
    ROLE_COL,
    SEED,
    TARGET_COL,
    XGB_NAME,
    XGB_PARAMS,
)
from minutes_xgb.data import data_quality, format_data_quality, load_regular_season
from minutes_xgb.eda import (
    feature_missingness,
    feature_outlier_table,
    format_minutes_outliers,
    minutes_outlier_summary,
    missingness_summary,
    plot_minutes,
    plot_missingness,
    season_coverage,
    season_missingness,
    top_minute_rows,
)
from minutes_xgb.errors import build_miss_frame, miss_view, rotation_team_games
from minutes_xgb.features import (
    add_total_stars_active,
    build_model_frame,
    positive_minutes,
)
from minutes_xgb.importance import (
    feature_consensus,
    mean_absolute_shap,
    permutation_importance_table,
    plot_shap,
)
from minutes_xgb.paths import ensure_repo_on_path, minutes_root, repo_root
from minutes_xgb.scoring import (
    holdout_point_distribution,
    interval_sharpness,
    predicted_minute_tiers,
    quantile_calibration,
    quantile_crossing,
    score_predictions,
    tail_calibration_rows,
)
from minutes_xgb.slices import (
    monotone_shift,
    regular_season_blocks,
    role_quantile_fallback,
    role_reading,
    slice_score,
)

__all__ = [
    "ARTIFACT_STEM",
    "HOLDOUT_SEASON",
    "ID_COLS",
    "MIN_FEATURES",
    "MIN_TIERS",
    "N_BOOT",
    "QUANTILES",
    "ROLE_COL",
    "SEED",
    "TARGET_COL",
    "XGB_NAME",
    "XGB_PARAMS",
    "add_total_stars_active",
    "build_miss_frame",
    "build_model_frame",
    "build_oos_predictions",
    "calibration_table",
    "compare_to_baselines",
    "data_quality",
    "ensure_repo_on_path",
    "exploratory_metadata",
    "feature_consensus",
    "fold_base_knots",
    "feature_missingness",
    "feature_outlier_table",
    "format_data_quality",
    "format_minutes_outliers",
    "holdout_point_distribution",
    "interval_sharpness",
    "load_regular_season",
    "loaded_holdout_scores",
    "mean_absolute_shap",
    "minutes_outlier_summary",
    "minutes_root",
    "nested_offset_backtest",
    "miss_view",
    "missingness_summary",
    "monotone_shift",
    "offset_stability_flags",
    "oof_leakage_table",
    "oof_player_games",
    "permutation_importance_table",
    "plot_minutes",
    "plot_missingness",
    "plot_shap",
    "positive_minutes",
    "predicted_minute_tiers",
    "quantile_calibration",
    "quantile_crossing",
    "regular_season_blocks",
    "repo_root",
    "role_quantile_fallback",
    "role_reading",
    "rotation_team_games",
    "score_predictions",
    "season_coverage",
    "season_missingness",
    "slice_score",
    "tail_calibration_rows",
    "top_minute_rows",
    "write_oos_predictions",
]
