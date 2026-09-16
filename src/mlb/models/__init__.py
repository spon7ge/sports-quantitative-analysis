"""Package inits for MLB models."""

from __future__ import annotations

from src.mlb.models.calibration import (
    Recalibrator,
    apply_recalibration,
    calibration_slope_intercept,
    fit_pit_recalibration,
    randomized_pit,
)
from src.mlb.models.metrics import discrete_crps, pmf_nll
from src.mlb.models.pmf import negative_binomial_pmf, pmf_cdf
from src.mlb.models.props import prop_probabilities
from src.mlb.models.serialize import (
    as_strikeout_model,
    as_workload_model,
    load_model,
    save_model,
)
from src.mlb.models.shrinkage import shrink_rate
from src.mlb.models.strikeouts import (
    StrikeoutModel,
    fit_strikeouts,
    predict_strikeout_pmf,
)
from src.mlb.models.workload import (
    WorkloadModel,
    add_oof_workload_features,
    fit_workload,
    predict_workload,
)

__all__ = [
    "Recalibrator",
    "StrikeoutModel",
    "WorkloadModel",
    "add_oof_workload_features",
    "apply_recalibration",
    "as_strikeout_model",
    "as_workload_model",
    "calibration_slope_intercept",
    "discrete_crps",
    "fit_pit_recalibration",
    "fit_strikeouts",
    "fit_workload",
    "load_model",
    "negative_binomial_pmf",
    "pmf_cdf",
    "pmf_nll",
    "predict_strikeout_pmf",
    "predict_workload",
    "prop_probabilities",
    "randomized_pit",
    "save_model",
    "shrink_rate",
]
