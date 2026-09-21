"""Regularized NB2 workload model and expanding-window OOF features."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import Logit, NegativeBinomial

from src.mlb.config import MlbConfig
from src.mlb.schemas import WORKLOAD_FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)


class GlmFitError(RuntimeError):
    """Raised when a penalized GLM cannot be fit honestly."""

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        n_train: int | None = None,
        shape: tuple[int, int] | None = None,
        rank: int | None = None,
        constant_columns: tuple[str, ...] = (),
        unavailable_columns: tuple[str, ...] = (),
        condition_number: float | None = None,
        fold: str = "",
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        parts = [message]
        if cause is not None:
            parts.append(f"{type(cause).__name__}: {cause}")
        if fold:
            parts.append(f"fold={fold}")
        if n_train is not None:
            parts.append(f"n_train={n_train}")
        if shape is not None:
            parts.append(f"shape={shape[0]}x{shape[1]}")
        if rank is not None:
            parts.append(f"rank={rank}")
        if condition_number is not None and np.isfinite(condition_number):
            parts.append(f"cond={condition_number:.4g}")
        if constant_columns:
            parts.append("constant_columns=" + ",".join(constant_columns))
        if unavailable_columns:
            parts.append("unavailable_columns=" + ",".join(unavailable_columns))
        super().__init__("; ".join(parts))
        self.cause = cause
        self.n_train = n_train
        self.shape = shape
        self.rank = rank
        self.constant_columns = constant_columns
        self.unavailable_columns = unavailable_columns
        self.condition_number = condition_number
        self.fold = fold
        self.diagnostics = diagnostics or {}

_OOF_MAP = {
    "expected_bf_oof": "expected_bf",
    "predicted_bf_oof": "expected_bf",
    "bf_sd_oof": "bf_sd",
    "expected_pitches_oof": "expected_pitches",
    "expected_outs_oof": "expected_outs",
    "p_early_exit_oof": "p_early_exit",
}


@dataclass
class Nb2Fit:
    """Penalized NB2 coefficients plus the optimizer status that produced them."""

    coef: np.ndarray
    alpha: float
    cov: np.ndarray | None
    method: str
    converged: bool | None = None
    iterations: int | None = None


@dataclass
class WorkloadModel:
    """Fitted batters-faced NB2 plus a regularized early-exit logit."""

    feature_names: tuple[str, ...]
    coef: np.ndarray
    alpha: float
    medians: dict[str, float]
    mean_pitches_per_bf: float
    mean_outs_per_bf: float
    logit_coef: np.ndarray
    early_exit_bf: int
    method: str = "glm"
    cov: np.ndarray | None = None
    model_version: str = "nb_bf_v1"
    extra: dict[str, Any] = field(default_factory=dict)
    dropped_features: dict[str, str] = field(default_factory=dict)
    centers: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    matrix_rank: int = 0
    condition_number: float = float("nan")


def feature_medians(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for column in columns:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            median = series.median()
            medians[column] = float(median) if pd.notna(median) else 0.0
        else:
            medians[column] = 0.0
        if not np.isfinite(medians[column]):
            medians[column] = 0.0
    return medians


def design_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    medians: dict[str, float],
    *,
    centers: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> np.ndarray:
    cols = []
    n = len(frame)
    centers = centers or {}
    scales = scales or {}
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
        else:
            values = pd.Series(np.nan, index=frame.index)
        filled = values.fillna(medians.get(column, 0.0)).to_numpy(dtype=float)
        filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
        if column in centers:
            scale = float(scales.get(column, 1.0))
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            filled = (filled - float(centers[column])) / scale
        cols.append(filled)
    features = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
    intercept = np.ones((n, 1), dtype=float)
    return np.hstack([intercept, features])


def _moments_nb(
    y: np.ndarray,
    n_params: int,
    default_mu: float,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, float, None]:
    n = y.shape[0]
    if n == 0:
        mu = float(default_mu)
        var = mu
    else:
        mu = float(np.mean(y))
        var = float(np.var(y, ddof=1)) if n > 1 else mu
    mu = max(mu, 1e-6)
    alpha = max((var - mu) / (mu * mu), 1e-4)
    coef = np.zeros(n_params, dtype=float)
    if n_params:
        intercept = np.log(mu)
        if offset is not None and np.size(offset):
            intercept -= float(np.mean(np.asarray(offset, dtype=float)))
        coef[0] = intercept
    return coef, float(alpha), None


def _design_diagnostics(
    x: np.ndarray, feature_names: tuple[str, ...]
) -> tuple[int, float, tuple[str, ...]]:
    rank = int(np.linalg.matrix_rank(x, tol=1e-8))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        condition = float(np.linalg.cond(x))
    constant: list[str] = []
    names = feature_names or tuple(f"col_{j}" for j in range(1, x.shape[1]))
    for i, name in enumerate(names):
        j = i + 1
        if j >= x.shape[1]:
            break
        if float(np.std(x[:, j], ddof=0)) < 1e-12:
            constant.append(name)
    for j in range(len(names) + 1, x.shape[1]):
        if float(np.std(x[:, j], ddof=0)) < 1e-12:
            constant.append(f"col_{j}")
    return rank, condition, tuple(constant)


def _optimizer_status(result: Any) -> tuple[bool | None, int | None]:
    retvals = getattr(result, "mle_retvals", None)
    if not isinstance(retvals, dict):
        retvals = {}
    converged = retvals.get("converged")
    if converged is None and hasattr(result, "converged"):
        converged = result.converged
    iterations = None
    for key in ("iterations", "nit", "niter", "fcalls", "gcalls"):
        if retvals.get(key) is None:
            continue
        try:
            iterations = int(retvals[key])
            break
        except (TypeError, ValueError):
            continue
    conv_flag = None if converged is None else bool(converged)
    return conv_flag, iterations


def _extract_nb2_params(
    result: Any, p: int
) -> tuple[np.ndarray, float, np.ndarray]:
    params = np.asarray(result.params, dtype=float).reshape(-1)
    if params.size == p + 1:
        return params[:-1], float(params[-1]), params
    if params.size == p:
        return params, 1e-4, params
    raise ValueError(f"unexpected GLM parameter count {params.size}")


def fit_nb2(
    y: np.ndarray,
    x: np.ndarray,
    l2: float,
    default_mu: float = 1.0,
    *,
    fail_closed: bool = True,
    require_convergence: bool = False,
    feature_names: tuple[str, ...] = (),
    fold: str = "",
    unavailable_columns: tuple[str, ...] = (),
    offset: np.ndarray | None = None,
) -> Nb2Fit:
    """Penalized NB2 MLE. Fail closed unless ``fail_closed=False``.

    Non-convergence is recorded on the result. Strikeout fits pass
    ``require_convergence=True`` so a failed optimizer cannot be labeled ``glm``.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n, p = x_arr.shape
    y_arr = np.clip(np.rint(y_arr), 0, None)
    offset_arr = None
    if offset is not None:
        offset_arr = np.asarray(offset, dtype=float).reshape(-1)
        if offset_arr.shape[0] != n:
            raise ValueError(
                f"offset length {offset_arr.shape[0]} does not match rows {n}"
            )
        offset_arr = np.nan_to_num(offset_arr, nan=0.0, posinf=0.0, neginf=0.0)
    rank, condition, constant = _design_diagnostics(x_arr, feature_names)

    def _fail(
        message: str,
        cause: BaseException | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        raise GlmFitError(
            message,
            cause=cause,
            n_train=n,
            shape=(n, p),
            rank=rank,
            constant_columns=constant,
            unavailable_columns=unavailable_columns,
            condition_number=condition,
            fold=fold,
            diagnostics=diagnostics,
        ) from cause

    if n < 3 or p == 0:
        if fail_closed:
            _fail("insufficient rows or empty design for GLM")
        coef, alpha, cov = _moments_nb(y_arr, p, default_mu, offset=offset_arr)
        return Nb2Fit(coef, alpha, cov, "moments")

    if rank < p or constant:
        if fail_closed:
            _fail("rank-deficient or constant design matrix")
        coef, alpha, cov = _moments_nb(y_arr, p, default_mu, offset=offset_arr)
        return Nb2Fit(coef, alpha, cov, "moments")

    try:
        model = NegativeBinomial(y_arr, x_arr, offset=offset_arr)
        result = None
        use_regularized = l2 > 0 and n < 1000
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if use_regularized:
                    penalty = np.full(p + 1, float(l2))
                    penalty[0] = 0.0
                    penalty[-1] = 0.0
                    try:
                        result = model.fit_regularized(
                            alpha=penalty,
                            L1_wt=0.0,
                            disp=False,
                            maxiter=200,
                        )
                    except (ValueError, np.linalg.LinAlgError, RuntimeError, TypeError):
                        try:
                            result = model.fit_regularized(
                                alpha=float(l2),
                                L1_wt=0.0,
                                disp=False,
                                maxiter=200,
                            )
                        except TypeError:
                            result = model.fit_regularized(
                                alpha=float(l2),
                                L1_wt=0.0,
                                disp=False,
                            )
                if result is None:
                    result = model.fit(disp=0, maxiter=200, warn_convergence=False)
        try:
            coef, alpha, params = _extract_nb2_params(result, p)
        except ValueError as exc:
            _fail(str(exc), cause=exc)
        if not np.all(np.isfinite(coef)) or not np.isfinite(alpha):
            result = model.fit(disp=0, maxiter=200, warn_convergence=False)
            try:
                coef, alpha, params = _extract_nb2_params(result, p)
            except ValueError as exc:
                _fail(str(exc), cause=exc)
        if not np.all(np.isfinite(coef)) or not np.isfinite(alpha):
            _fail("GLM produced non-finite coefficients or dispersion")
        alpha = float(max(alpha, 1e-4))
        converged, iterations = _optimizer_status(result)
        slopes = np.asarray(coef[1:], dtype=float) if coef.size > 1 else np.zeros(0)
        intercept_only = coef.size <= 1 or (
            slopes.size > 0 and float(np.max(np.abs(slopes))) < 1e-8
        )
        if converged is False:
            LOGGER.warning(
                "GLM optimizer did not converge after %s iterations; "
                "fold=%s intercept_only=%s method=glm",
                iterations,
                fold or "?",
                intercept_only,
            )
        if require_convergence and converged is False and intercept_only:
            _fail(
                f"GLM optimizer did not converge after {iterations} iterations "
                "and produced intercept-only coefficients",
                diagnostics={
                    "coef": [float(v) for v in np.asarray(coef, dtype=float)],
                    "method": "glm",
                    "converged": False,
                    "fit_iterations": iterations,
                },
            )
        cov_mat: np.ndarray | None = None
        try:
            cov_full = np.asarray(result.cov_params(), dtype=float)
            if (
                cov_full.ndim == 2
                and cov_full.shape[0] == params.size
                and np.all(np.isfinite(cov_full))
            ):
                cov_mat = cov_full[:p, :p] if params.size == p + 1 else cov_full
        except (ValueError, np.linalg.LinAlgError, AttributeError):
            cov_mat = None
        return Nb2Fit(
            coef.astype(float),
            float(max(alpha, 1e-4)),
            cov_mat,
            "glm",
            converged=converged,
            iterations=iterations,
        )
    except GlmFitError:
        raise
    except Exception as exc:
        if fail_closed:
            _fail("GLM fit failed", cause=exc)
        coef, alpha, cov = _moments_nb(y_arr, p, default_mu, offset=offset_arr)
        return Nb2Fit(coef, alpha, cov, "moments")


def _logit_moments(y: np.ndarray, n_params: int) -> np.ndarray:
    rate = float(np.mean(y)) if y.size else 0.0
    rate = float(np.clip(rate, 1e-6, 1.0 - 1e-6))
    coef = np.zeros(n_params, dtype=float)
    if n_params:
        coef[0] = np.log(rate / (1.0 - rate))
    return coef


def fit_logit(y: np.ndarray, x: np.ndarray, l2: float) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n, p = x_arr.shape
    if n < 3 or p == 0 or np.unique(y_arr).size < 2:
        return _logit_moments(y_arr, p)
    try:
        model = Logit(y_arr, x_arr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if l2 > 0 and n < 1000:
                penalty = np.full(p, float(l2))
                penalty[0] = 0.0
                try:
                    result = model.fit_regularized(
                        alpha=penalty,
                        L1_wt=0.0,
                        disp=False,
                        maxiter=200,
                    )
                except Exception:
                    result = model.fit_regularized(
                        alpha=float(l2),
                        L1_wt=0.0,
                        disp=False,
                    )
            else:
                result = model.fit(disp=0, maxiter=200, warn_convergence=False)
        coef = np.asarray(result.params, dtype=float).reshape(-1)
        if coef.size != p or not np.all(np.isfinite(coef)):
            return _logit_moments(y_arr, p)
        return coef
    except Exception:
        return _logit_moments(y_arr, p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _mean_ratio(numer: pd.Series, denom: pd.Series, default: float) -> float:
    denom_f = pd.to_numeric(denom, errors="coerce").clip(lower=1e-6)
    numer_f = pd.to_numeric(numer, errors="coerce")
    ratio = (numer_f / denom_f).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return default
    return float(ratio.mean())


def fit_workload(train: pd.DataFrame, config: MlbConfig) -> WorkloadModel:
    """Fit regularized NB2 on batters faced plus an early-exit logit."""
    np.random.seed(config.seed)
    from src.mlb.models.preprocess import (
        compute_standardization,
        prepare_workload_frame,
        select_usable_features,
    )

    prepared = prepare_workload_frame(train)
    selection = select_usable_features(prepared, WORKLOAD_FEATURE_COLUMNS)
    if not selection.retained:
        raise GlmFitError(
            "no usable workload features",
            n_train=len(prepared),
            unavailable_columns=tuple(selection.dropped),
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
    if "batters_faced" not in prepared.columns:
        y = np.zeros(len(prepared), dtype=float)
    else:
        y = pd.to_numeric(prepared["batters_faced"], errors="coerce").fillna(0).to_numpy()
    dropped = tuple(selection.dropped)
    fit = fit_nb2(
        y,
        x,
        l2=float(config.workload_l2),
        default_mu=22.0,
        feature_names=selection.retained,
        unavailable_columns=dropped,
    )
    coef, alpha, cov, method = fit.coef, fit.alpha, fit.cov, fit.method
    rank, condition, _constant = _design_diagnostics(x, selection.retained)
    pitches = (
        prepared["pitches"]
        if "pitches" in prepared.columns
        else pd.Series(np.nan, index=prepared.index)
    )
    outs = (
        prepared["outs"]
        if "outs" in prepared.columns
        else pd.Series(np.nan, index=prepared.index)
    )
    bf = (
        prepared["batters_faced"]
        if "batters_faced" in prepared.columns
        else pd.Series(np.nan, index=prepared.index)
    )
    mean_pitches = _mean_ratio(pitches, bf, default=3.85)
    mean_outs = _mean_ratio(outs, bf, default=0.70)
    early = (y < float(config.early_exit_bf)).astype(float)
    logit_coef = fit_logit(early, x, l2=float(config.workload_l2))
    extra = {
        "dropped_features": dict(selection.dropped),
        "centers": dict(centers),
        "scales": dict(scales),
        "n_train": int(len(prepared)),
        "matrix_rank": int(rank),
        "condition_number": float(condition),
        "method": method,
    }
    return WorkloadModel(
        feature_names=selection.retained,
        coef=np.asarray(coef, dtype=float),
        alpha=float(alpha),
        medians=selection.medians,
        mean_pitches_per_bf=mean_pitches,
        mean_outs_per_bf=mean_outs,
        logit_coef=np.asarray(logit_coef, dtype=float),
        early_exit_bf=int(config.early_exit_bf),
        method=method,
        cov=cov,
        model_version=str(config.workload_model_version),
        extra=extra,
        dropped_features=dict(selection.dropped),
        centers=centers,
        scales=scales,
        n_train=int(len(prepared)),
        matrix_rank=int(rank),
        condition_number=float(condition),
    )


def predict_workload(model: WorkloadModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Return expected_bf, bf_sd, expected_pitches, expected_outs, p_early_exit."""
    from src.mlb.models.preprocess import prepare_workload_frame

    prepared = prepare_workload_frame(frame)
    x = design_matrix(
        prepared,
        model.feature_names,
        model.medians,
        centers=model.centers,
        scales=model.scales,
    )
    eta = x @ model.coef
    mu = np.clip(np.exp(np.clip(eta, -20.0, 8.0)), 1e-6, 60.0)
    alpha = max(float(model.alpha), 1e-12)
    bf_sd = np.sqrt(mu + alpha * mu * mu)
    expected_pitches = mu * float(model.mean_pitches_per_bf)
    expected_outs = mu * float(model.mean_outs_per_bf)
    p_early = _sigmoid(x @ model.logit_coef)
    return pd.DataFrame(
        {
            "expected_bf": mu,
            "bf_sd": bf_sd,
            "expected_pitches": expected_pitches,
            "expected_outs": expected_outs,
            "p_early_exit": p_early,
        },
        index=frame.index,
    )


def add_oof_workload_features(
    starts: pd.DataFrame,
    feature_rows: pd.DataFrame,
    config: MlbConfig,
) -> pd.DataFrame:
    """Expanding OOF workload predictions. Never in-sample BF.

    Refits at most weekly on small panels and every 28 days once history
    exceeds 500 starts.
    """
    np.random.seed(config.seed)
    from src.mlb.models.preprocess import prepare_workload_frame

    out = prepare_workload_frame(feature_rows)
    for column in _OOF_MAP:
        out[column] = np.nan

    if starts.empty or out.empty or "game_date" not in starts.columns:
        return out

    keep_start = [
        column
        for column in (
            "pitcher_id",
            "game_pk",
            "game_date",
            "batters_faced",
            "pitches",
            "outs",
            *WORKLOAD_FEATURE_COLUMNS,
        )
        if column in starts.columns
    ]
    keep_feat = [
        column
        for column in ("pitcher_id", "game_pk", *WORKLOAD_FEATURE_COLUMNS)
        if column in out.columns
    ]
    panel = starts[keep_start].merge(
        out[keep_feat],
        on=["pitcher_id", "game_pk"],
        how="left",
        suffixes=("", "_feat"),
    )
    for column in WORKLOAD_FEATURE_COLUMNS:
        feat_col = f"{column}_feat"
        if feat_col in panel.columns:
            panel[column] = panel[feat_col].where(
                panel[feat_col].notna(),
                panel[column] if column in panel.columns else np.nan,
            )
            panel = panel.drop(columns=[feat_col])
    panel = prepare_workload_frame(panel)
    panel["_oof_time"] = pd.to_datetime(panel["game_date"])
    panel = panel.sort_values("_oof_time").reset_index(drop=True)
    times = panel["_oof_time"].to_numpy()
    dates = np.unique(times)
    min_train = int(config.workload_min_train_starts)
    model = None
    last_fit_stamp: pd.Timestamp | None = None
    lookup = {
        (int(pitcher_id), int(game_pk)): idx
        for idx, pitcher_id, game_pk in zip(
            range(len(out)),
            out["pitcher_id"].to_numpy(),
            out["game_pk"].to_numpy(),
        )
    }
    oof_cols = {
        oof_name: out.columns.get_loc(oof_name)
        for oof_name in _OOF_MAP
        if oof_name in out.columns
    }

    for date in dates:
        hist_end = int(np.searchsorted(times, date, side="left"))
        if hist_end < min_train:
            continue
        stamp = pd.Timestamp(date)
        refit_days = 7 if hist_end < 500 else 28
        if (
            model is None
            or last_fit_stamp is None
            or (stamp - last_fit_stamp).days >= refit_days
        ):
            model = fit_workload(panel.iloc[:hist_end], config)
            last_fit_stamp = stamp
        day_start = hist_end
        day_end = int(np.searchsorted(times, date, side="right"))
        day = panel.iloc[day_start:day_end]
        if day.empty:
            continue
        pos = [
            lookup[(int(pitcher_id), int(game_pk))]
            for pitcher_id, game_pk in zip(
                day["pitcher_id"].to_numpy(),
                day["game_pk"].to_numpy(),
            )
            if (int(pitcher_id), int(game_pk)) in lookup
        ]
        if not pos:
            continue
        preds = predict_workload(model, out.iloc[pos])
        values = preds.to_numpy()
        pred_index = {name: i for i, name in enumerate(preds.columns)}
        for oof_name, pred_name in _OOF_MAP.items():
            col = oof_cols.get(oof_name)
            if col is None:
                continue
            out.iloc[pos, col] = values[:, pred_index[pred_name]]
    return out
