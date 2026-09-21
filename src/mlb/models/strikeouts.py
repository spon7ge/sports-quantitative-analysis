"""Penalized NB2 strikeout count model with optional parameter averaging."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.models.pmf import negative_binomial_pmf, pmf_cdf
from src.mlb.models.preprocess import (
    bf_exposure,
    bf_exposure_offset,
    compute_standardization,
    prepare_strikeout_frame,
    select_usable_features,
)
from src.mlb.models.props import prop_probabilities
from src.mlb.models.workload import (
    GlmFitError,
    _design_diagnostics,
    design_matrix,
    fit_nb2,
)
from src.mlb.schemas import (
    PMF_COLUMNS,
    PREDICTION_COLUMNS,
    STRIKEOUT_FEATURE_COLUMNS,
    UTC_DTYPE,
)

LOGGER = logging.getLogger(__name__)


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
    extra: dict[str, Any] = field(default_factory=dict)
    dropped_features: dict[str, str] = field(default_factory=dict)
    centers: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    train_start: str = ""
    train_end: str = ""
    n_train: int = 0
    matrix_rank: int = 0
    condition_number: float = float("nan")
    fold: str = ""
    uses_bf_offset: bool = True


def offset_fit_diagnostics(frame: pd.DataFrame, offset: np.ndarray) -> dict[str, Any]:
    """Min/max of the used offset and NaN-count of raw ``log(predicted_bf_oof)``."""
    used = np.asarray(offset, dtype=float).reshape(-1)
    finite_used = used[np.isfinite(used)]
    offset_min = float(np.min(finite_used)) if finite_used.size else float("nan")
    offset_max = float(np.max(finite_used)) if finite_used.size else float("nan")
    raw = None
    for name in ("predicted_bf_oof", "expected_bf_oof"):
        if name in frame.columns:
            raw = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            break
    if raw is None:
        nan_count = int(used.size)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_raw = np.log(raw)
        nan_count = int((~np.isfinite(log_raw)).sum())
    return {
        "offset_min": offset_min,
        "offset_max": offset_max,
        "offset_nan_count": nan_count,
    }


def build_glm_diagnostics(
    *,
    coef: np.ndarray | list[float],
    feature_names: tuple[str, ...],
    method: str,
    converged: bool | None,
    iterations: int | None,
    retained: tuple[str, ...],
    dropped: dict[str, str] | None,
    offset_stats: dict[str, Any],
) -> dict[str, Any]:
    coef_list = [float(v) for v in np.asarray(coef, dtype=float).reshape(-1)]
    names = ("intercept", *tuple(feature_names))
    coef_names = list(names[: len(coef_list)])
    beta = {
        name: coef_list[i]
        for i, name in enumerate(coef_names)
        if i < len(coef_list)
    }
    return {
        "coef": coef_list,
        "coef_names": coef_names,
        "beta": beta,
        "retained_features": list(retained),
        "dropped_features": dict(dropped or {}),
        "method": str(method),
        "converged": None if converged is None else bool(converged),
        "fit_iterations": None if iterations is None else int(iterations),
        **offset_stats,
    }


def format_glm_diagnostics(fold: str, diag: dict[str, Any]) -> str:
    beta = diag.get("beta") or {}
    if isinstance(beta, dict) and beta:
        beta_str = ", ".join(
            f"{name}={float(value):.6g}" for name, value in beta.items()
        )
    else:
        coef = diag.get("coef") or []
        names = diag.get("coef_names") or []
        beta_str = ", ".join(
            f"{name}={float(value):.6g}"
            for name, value in zip(names, coef, strict=False)
        )
    return (
        f"fold={fold} method={diag.get('method')} converged={diag.get('converged')} "
        f"iterations={diag.get('fit_iterations')} "
        f"retained_features={diag.get('retained_features')} "
        f"beta=[{beta_str}] "
        f"offset_min={diag.get('offset_min')} offset_max={diag.get('offset_max')} "
        f"offset_nan_count={diag.get('offset_nan_count')}"
    )


def _log_glm_diagnostics(fold: str, diag: dict[str, Any], *, error: bool = False) -> None:
    message = format_glm_diagnostics(fold, diag)
    if error:
        LOGGER.error(message)
    else:
        LOGGER.info(message)


def fit_strikeouts(
    train: pd.DataFrame,
    config: MlbConfig,
    *,
    fold: str = "",
) -> StrikeoutModel:
    """Penalized NB2 on strikeout counts using the retained usable schema."""
    np.random.seed(config.seed)
    prepared = prepare_strikeout_frame(train)
    selection = select_usable_features(prepared, STRIKEOUT_FEATURE_COLUMNS)
    offset = bf_exposure_offset(prepared)
    offset_stats = offset_fit_diagnostics(prepared, offset)
    if not selection.retained:
        diag = build_glm_diagnostics(
            coef=[],
            feature_names=(),
            method="",
            converged=None,
            iterations=None,
            retained=(),
            dropped=selection.dropped,
            offset_stats=offset_stats,
        )
        _log_glm_diagnostics(fold, diag, error=True)
        raise GlmFitError(
            "no usable strikeout features",
            n_train=len(prepared),
            unavailable_columns=tuple(selection.dropped),
            fold=fold,
            diagnostics=diag,
        )
    centers, scales = compute_standardization(
        prepared, selection.retained, selection.medians
    )
    x = design_matrix(
        prepared,
        selection.retained,
        selection.medians,
        centers=centers,
        scales=scales,
    )
    if "strikeouts" not in prepared.columns:
        y = np.zeros(len(prepared), dtype=float)
    else:
        y = pd.to_numeric(prepared["strikeouts"], errors="coerce").fillna(0).to_numpy()
    dropped = tuple(
        name for name, reason in selection.dropped.items() if reason == "all_missing"
    )
    try:
        fit = fit_nb2(
            y,
            x,
            l2=float(config.strikeout_l2),
            default_mu=5.0,
            feature_names=selection.retained,
            fold=fold,
            unavailable_columns=dropped,
            offset=offset,
            require_convergence=True,
        )
    except GlmFitError as exc:
        coef = np.asarray((exc.diagnostics or {}).get("coef", []), dtype=float)
        diag = build_glm_diagnostics(
            coef=coef,
            feature_names=selection.retained,
            method=str((exc.diagnostics or {}).get("method", "glm")),
            converged=(exc.diagnostics or {}).get("converged"),
            iterations=(exc.diagnostics or {}).get("fit_iterations"),
            retained=selection.retained,
            dropped=selection.dropped,
            offset_stats=offset_stats,
        )
        _log_glm_diagnostics(fold, diag, error=True)
        exc.diagnostics = diag
        raise
    coef, alpha, cov, method = fit.coef, fit.alpha, fit.cov, fit.method
    rank, condition, constant = _design_diagnostics(x, selection.retained)
    if constant:
        raise GlmFitError(
            "fitted design still contains unreported zero-variance columns",
            n_train=len(prepared),
            shape=x.shape,
            rank=rank,
            constant_columns=constant,
            fold=fold,
        )
    dates = None
    if "game_date" in prepared.columns:
        dates = pd.to_datetime(prepared["game_date"], errors="coerce")
    train_start = ""
    train_end = ""
    if dates is not None and dates.notna().any():
        train_start = str(dates.min().date())
        train_end = str(dates.max().date())
    diag = build_glm_diagnostics(
        coef=coef,
        feature_names=selection.retained,
        method=method,
        converged=fit.converged,
        iterations=fit.iterations,
        retained=selection.retained,
        dropped=selection.dropped,
        offset_stats=offset_stats,
    )
    _log_glm_diagnostics(fold, diag)
    extra = {
        "dropped_features": dict(selection.dropped),
        "centers": dict(centers),
        "scales": dict(scales),
        "train_start": train_start,
        "train_end": train_end,
        "n_train": int(len(prepared)),
        "matrix_rank": int(rank),
        "condition_number": float(condition),
        "method": method,
        "fold": fold,
        "feature_names": list(selection.retained),
        "uses_bf_offset": True,
        "glm_diagnostics": diag,
        "converged": fit.converged,
        "fit_iterations": fit.iterations,
    }
    return StrikeoutModel(
        feature_names=selection.retained,
        coef=np.asarray(coef, dtype=float),
        alpha=float(alpha),
        medians=selection.medians,
        method=method,
        cov=cov,
        model_version=str(config.model_version),
        extra=extra,
        dropped_features=dict(selection.dropped),
        centers=centers,
        scales=scales,
        train_start=train_start,
        train_end=train_end,
        n_train=int(len(prepared)),
        matrix_rank=int(rank),
        condition_number=float(condition),
        fold=fold,
        uses_bf_offset=True,
    )


def _psd(cov: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (cov + cov.T)
    eigval, eigvec = np.linalg.eigh(symmetric)
    eigval = np.clip(eigval, 0.0, None)
    rebuilt = eigvec @ np.diag(eigval) @ eigvec.T
    jitter = 1e-8 * np.eye(rebuilt.shape[0])
    return rebuilt + jitter


def _linear_predictor(
    x: np.ndarray,
    coef: np.ndarray,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    eta = x @ coef
    if offset is not None:
        eta = eta + np.asarray(offset, dtype=float)
    return np.clip(np.exp(np.clip(eta, -20.0, 8.0)), 1e-6, 80.0)


def _average_pmfs(
    model: StrikeoutModel,
    x: np.ndarray,
    config: MlbConfig,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    mu_hat = _linear_predictor(x, model.coef, offset=offset)
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
        negative_binomial_pmf(
            _linear_predictor(x, beta, offset=offset), model.alpha, k_max, tail
        )
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
    prepared = prepare_strikeout_frame(frame)
    x = design_matrix(
        prepared,
        model.feature_names,
        model.medians,
        centers=model.centers,
        scales=model.scales,
    )
    offset = bf_exposure_offset(prepared) if model.uses_bf_offset else None
    pmf = _average_pmfs(model, x, config, offset=offset)
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

    if model.uses_bf_offset:
        out["expected_bf"] = bf_exposure(prepared)
    else:
        _copy_optional(
            out,
            frame,
            "expected_bf",
            ("predicted_bf_oof", "expected_bf_oof", "expected_bf", "bf_mean_5"),
        )
    _copy_optional(
        out,
        frame,
        "expected_pitches",
        ("pitches_per_start_5", "expected_pitches_oof", "expected_pitches"),
    )
    _copy_optional(
        out, frame, "expected_outs", ("outs_per_start_5", "expected_outs_oof", "expected_outs")
    )
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
