"""XGBoost models for minutes and player props."""

from .core import XGBoostConfig
from .minutes import (
    DEFAULT_MINUTES_BINS,
    DEFAULT_MINUTES_FEATURES,
    HierarchicalMinutePools,
    MINUTES_DISTRIBUTION_ARTIFACT,
    MINUTES_FEATURE_CONTRACTS,
    XGBoostMinutesModel,
    build_hierarchical_minute_pools,
)
from .points import (
    DEFAULT_POINTS_FEATURES,
    DIRECT_POINTS_FEATURES,
    POINTS_FEATURE_CONTRACTS,
    PointsResidualPools,
    XGBoostPointsModel,
    add_predicted_minutes_oof,
    add_predicted_points_oof,
    build_points_residual_pools,
)
from .joint_calibration import (
    JOINT_VARIANTS,
    PRODUCTION_JOINT_VARIANT,
    distribution_gate,
    fit_joint_variant,
    load_joint_calibration,
    save_joint_calibration,
    select_joint_variant,
    train_joint_calibration,
)
from .joint_variant_eval import (
    PROMOTION_CANDIDATE,
    decide_promotion,
    score_variant_folds,
)

__all__ = [
    "DEFAULT_MINUTES_BINS",
    "DEFAULT_MINUTES_FEATURES",
    "MINUTES_DISTRIBUTION_ARTIFACT",
    "MINUTES_FEATURE_CONTRACTS",
    "DEFAULT_POINTS_FEATURES",
    "DIRECT_POINTS_FEATURES",
    "POINTS_FEATURE_CONTRACTS",
    "JOINT_VARIANTS",
    "PRODUCTION_JOINT_VARIANT",
    "XGBoostConfig",
    "XGBoostMinutesModel",
    "XGBoostPointsModel",
    "HierarchicalMinutePools",
    "PointsResidualPools",
    "add_predicted_minutes_oof",
    "add_predicted_points_oof",
    "build_hierarchical_minute_pools",
    "build_points_residual_pools",
    "distribution_gate",
    "fit_joint_variant",
    "select_joint_variant",
    "train_joint_calibration",
    "save_joint_calibration",
    "load_joint_calibration",
    "PROMOTION_CANDIDATE",
    "decide_promotion",
    "score_variant_folds",
]