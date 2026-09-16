"""Penalized NB2 strikeout count model with optional parameter averaging."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.models.pmf import negative_binomial_pmf, pmf_cdf
from src.mlb.models.props import prop_probabilities
from src.mlb.models.workload import design_matrix, feature_medians, fit_nb2
from src.mlb.schemas import (
    PMF_COLUMNS,
    PREDICTION_COLUMNS,
    STRIKEOUT_FEATURE_COLUMNS,
    UTC_DTYPE,
)


@dataclass
class StrikeoutModel:
    """Fitted strikeout NB2 on ``STRIKEOUT_FEATURE_COLUMNS``."""

    feature_names: tuple[str, ...]
    coef: np.ndarray
    alpha: float
    medians: dict[str, float]
    method: str = "glm"
    cov: np.ndarray | None = None
    model_version: str = "nb_k_v1"
    extra: dict[str, float] = field(default_factory=dict)


def fit_strikeouts(train: pd.DataFrame, config: MlbConfig) -> StrikeoutModel:
    """Penalized NB2 on strikeout counts. ``expected_bf_oof`` is a covariate."""
    np.random.seed(config.seed)
    feature_names = STRIKEOUT_FEATURE_COLUMNS
    medians = feature_medians(train, feature_names)
    x = design_matrix(train, feature_names, medians)
    if "strikeouts" not in train.columns:
        y = np.zeros(len(train), dtype=float)
    else:
        y = pd.to_numeric(train["strikeouts"], errors="coerce").fillna(0).to_numpy()
    coef, alpha, cov, method = fit_nb2(
        y,
        x,
        l2=float(config.strikeout_l2),
        default_mu=5.0,
    )
    return StrikeoutModel(
        feature_names=feature_names,
        coef=np.asarray(coef, dtype=float),
        alpha=float(alpha),
        medians=medians,
        method=method,
        cov=cov,
        model_version=str(config.model_version),
    )


def _psd(cov: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (cov + cov.T)
    eigval, eigvec = np.linalg.eigh(symmetric)
    eigval = np.clip(eigval, 0.0, None)
    rebuilt = eigvec @ np.diag(eigval) @ eigvec.T
    jitter = 1e-8 * np.eye(rebuilt.shape[0])
    return rebuilt + jitter


def _linear_predictor(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(np.clip(x @ coef, -20.0, 8.0)), 1e-6, 80.0)


def _average_pmfs(
    model: StrikeoutModel,
    x: np.ndarray,
    config: MlbConfig,
) -> np.ndarray:
    mu_hat = _linear_predictor(x, model.coef)
    k_max = int(config.k_max)
    tail = float(config.tail_mass_threshold)
    draws = int(config.posterior_draws)
    cov = model.cov
    if cov is None or draws <= 1:
        return negative_binomial_pmf(mu_hat, model.alpha, k_max, tail)
    cov_arr = np.asarray(cov, dtype=float)
    if (
        cov_arr.shape != (model.coef.size, model.coef.size)
        or not np.all(np.isfinite(cov_arr))
        or np.any(np.diag(cov_arr) > 50.0)
    ):
        return negative_binomial_pmf(mu_hat, model.alpha, k_max, tail)
    try:
        cov_psd = _psd(cov_arr)
        rng = np.random.Generator(np.random.PCG64(int(config.seed)))
        betas = rng.multivariate_normal(model.coef, cov_psd, size=draws)
    except Exception:
        return negative_binomial_pmf(mu_hat, model.alpha, k_max, tail)
    stacked = [
        negative_binomial_pmf(_linear_predictor(x, beta), model.alpha, k_max, tail)
        for beta in betas
    ]
    return np.mean(np.stack(stacked, axis=0), axis=0)


def _interval_bounds(pmf: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    cdf = pmf_cdf(pmf)
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[:, -1] = 1.0
    lower_q = (1.0 - float(level)) / 2.0
    upper_q = 1.0 - lower_q
    lower = (cdf >= lower_q).argmax(axis=1).astype(float)
    upper = (cdf >= upper_q).argmax(axis=1).astype(float)
    return lower, upper


def _pmf_moments(pmf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_max = pmf.shape[1] - 2
    support = np.arange(k_max + 1, dtype=float)
    mean_body = pmf[:, :-1] @ support
    tail_k = float(k_max + 1)
    mean = mean_body + pmf[:, -1] * tail_k
    second = pmf[:, :-1] @ (support ** 2) + pmf[:, -1] * (tail_k ** 2)
    var = np.clip(second - mean ** 2, 0.0, None)
    return mean, var


def _copy_optional(
    out: pd.DataFrame,
    frame: pd.DataFrame,
    dest: str,
    sources: tuple[str, ...],
) -> None:
    for source in sources:
        if source in frame.columns:
            out[dest] = pd.to_numeric(frame[source], errors="coerce")
            return


def predict_strikeout_pmf(
    model: StrikeoutModel,
    frame: pd.DataFrame,
    config: MlbConfig,
) -> pd.DataFrame:
    """Predict the K PMF and fill contract prediction columns."""
    x = design_matrix(frame, model.feature_names, model.medians)
    pmf = _average_pmfs(model, x, config)
    expected_k, variance_k = _pmf_moments(pmf)
    pi_lower, pi_upper = _interval_bounds(pmf, config.prediction_interval)

    data: dict[str, object] = {}
    for column, dtype in PREDICTION_COLUMNS.items():
        if dtype == "float64":
            data[column] = np.full(len(frame), np.nan, dtype=float)
        elif dtype == "int64":
            data[column] = np.zeros(len(frame), dtype=np.int64)
        elif dtype == "string":
            data[column] = np.array([""] * len(frame), dtype=object)
        else:
            data[column] = pd.Series(pd.NaT, index=frame.index, dtype=UTC_DTYPE)
    out = pd.DataFrame(data, index=frame.index)
    if "pitcher_id" in frame.columns:
        out["pitcher_id"] = frame["pitcher_id"].astype("int64")
    if "game_pk" in frame.columns:
        out["game_pk"] = frame["game_pk"].astype("int64")
    if "prediction_cutoff_utc" in frame.columns:
        out["prediction_cutoff_utc"] = pd.to_datetime(
            frame["prediction_cutoff_utc"], utc=True
        )
    if "forecast_horizon_hours" in frame.columns:
        out["forecast_horizon_hours"] = pd.to_numeric(
            frame["forecast_horizon_hours"], errors="coerce"
        )
    else:
        out["forecast_horizon_hours"] = float(config.forecast_horizon_hours)

    _copy_optional(out, frame, "expected_bf", ("expected_bf_oof", "expected_bf"))
    _copy_optional(
        out,
        frame,
        "expected_pitches",
        ("expected_pitches_oof", "expected_pitches"),
    )
    _copy_optional(out, frame, "expected_outs", ("expected_outs_oof", "expected_outs"))
    if out["expected_outs"].notna().any():
        out["expected_innings"] = out["expected_outs"] / 3.0

    out["expected_k"] = expected_k
    out["variance_k"] = variance_k
    out["pi_lower"] = pi_lower
    out["pi_upper"] = pi_upper
    for i, name in enumerate(PMF_COLUMNS):
        if i < pmf.shape[1]:
            out[name] = pmf[:, i]

    for line in config.lines:
        props = prop_probabilities(pmf, float(line))
        tag = str(line).replace(".", "_")
        out[f"p_over_{tag}"] = np.atleast_1d(np.asarray(props["p_over"], dtype=float))
        out[f"p_under_{tag}"] = np.atleast_1d(np.asarray(props["p_under"], dtype=float))

    out["model_version"] = str(config.model_version)
    out["feature_version"] = str(config.feature_set_version)
    out["workload_model_version"] = str(config.workload_model_version)
    if "source_snapshot_ids_json" in frame.columns:
        out["source_snapshot_ids_json"] = frame["source_snapshot_ids_json"].astype(str)
    return out.loc[:, list(PREDICTION_COLUMNS)]
