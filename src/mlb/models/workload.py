"""Regularized NB2 workload model and expanding-window OOF features."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import Logit, NegativeBinomial

from src.mlb.config import MlbConfig
from src.mlb.schemas import WORKLOAD_FEATURE_COLUMNS

_OOF_MAP = {
    "expected_bf_oof": "expected_bf",
    "bf_sd_oof": "bf_sd",
    "expected_pitches_oof": "expected_pitches",
    "expected_outs_oof": "expected_outs",
    "p_early_exit_oof": "p_early_exit",
}


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
    extra: dict[str, float] = field(default_factory=dict)


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
) -> np.ndarray:
    cols = []
    n = len(frame)
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
        else:
            values = pd.Series(np.nan, index=frame.index)
        filled = values.fillna(medians.get(column, 0.0)).to_numpy(dtype=float)
        filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(filled)
    features = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
    intercept = np.ones((n, 1), dtype=float)
    return np.hstack([intercept, features])


def _moments_nb(
    y: np.ndarray,
    n_params: int,
    default_mu: float,
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
        coef[0] = np.log(mu)
    return coef, float(alpha), None


def fit_nb2(
    y: np.ndarray,
    x: np.ndarray,
    l2: float,
    default_mu: float = 1.0,
) -> tuple[np.ndarray, float, np.ndarray | None, str]:
    """Penalized NB2 MLE with method-of-moments intercept-only fallback."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n, p = x_arr.shape
    y_arr = np.clip(np.rint(y_arr), 0, None)

    if n < 3 or p == 0:
        coef, alpha, cov = _moments_nb(y_arr, p, default_mu)
        return coef, alpha, cov, "moments"

    try:
        model = NegativeBinomial(y_arr, x_arr)
        result = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if l2 > 0:
                    penalty = np.full(p + 1, float(l2))
                    penalty[0] = 0.0
                    penalty[-1] = 0.0
                    try:
                        result = model.fit_regularized(
                            alpha=penalty,
                            L1_wt=0.0,
                            disp=False,
                        )
                    except Exception:
                        result = model.fit_regularized(
                            alpha=float(l2),
                            L1_wt=0.0,
                            disp=False,
                        )
                if result is None:
                    result = model.fit(disp=0, maxiter=200, warn_convergence=False)
        params = np.asarray(result.params, dtype=float).reshape(-1)
        if params.size == p + 1:
            coef = params[:-1]
            alpha = float(params[-1])
        elif params.size == p:
            coef = params
            alpha = 1e-4
        else:
            coef, alpha, cov = _moments_nb(y_arr, p, default_mu)
            return coef, alpha, cov, "moments"
        if (
            not np.all(np.isfinite(coef))
            or not np.isfinite(alpha)
            or alpha <= 0
            or np.any(np.abs(coef) > 15.0)
        ):
            coef, alpha, _cov = _moments_nb(y_arr, p, default_mu)
            return coef, alpha, _cov, "moments"
        cov_mat: np.ndarray | None = None
        try:
            cov_full = np.asarray(result.cov_params(), dtype=float)
            if (
                cov_full.ndim == 2
                and cov_full.shape[0] == params.size
                and np.all(np.isfinite(cov_full))
            ):
                cov_mat = cov_full[:p, :p] if params.size == p + 1 else cov_full
        except Exception:
            cov_mat = None
        return coef.astype(float), float(max(alpha, 1e-4)), cov_mat, "glm"
    except Exception:
        coef, alpha, cov = _moments_nb(y_arr, p, default_mu)
        return coef, alpha, cov, "moments"


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
            if l2 > 0:
                penalty = np.full(p, float(l2))
                penalty[0] = 0.0
                try:
                    result = model.fit_regularized(
                        alpha=penalty,
                        L1_wt=0.0,
                        disp=False,
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
    feature_names = WORKLOAD_FEATURE_COLUMNS
    medians = feature_medians(train, feature_names)
    x = design_matrix(train, feature_names, medians)
    if "batters_faced" not in train.columns:
        y = np.zeros(len(train), dtype=float)
    else:
        y = pd.to_numeric(train["batters_faced"], errors="coerce").fillna(0).to_numpy()
    coef, alpha, cov, method = fit_nb2(
        y,
        x,
        l2=float(config.workload_l2),
        default_mu=22.0,
    )
    pitches = (
        train["pitches"]
        if "pitches" in train.columns
        else pd.Series(np.nan, index=train.index)
    )
    outs = (
        train["outs"]
        if "outs" in train.columns
        else pd.Series(np.nan, index=train.index)
    )
    bf = (
        train["batters_faced"]
        if "batters_faced" in train.columns
        else pd.Series(np.nan, index=train.index)
    )
    mean_pitches = _mean_ratio(pitches, bf, default=3.85)
    mean_outs = _mean_ratio(outs, bf, default=0.70)
    early = (y < float(config.early_exit_bf)).astype(float)
    logit_coef = fit_logit(early, x, l2=float(config.workload_l2))
    return WorkloadModel(
        feature_names=feature_names,
        coef=np.asarray(coef, dtype=float),
        alpha=float(alpha),
        medians=medians,
        mean_pitches_per_bf=mean_pitches,
        mean_outs_per_bf=mean_outs,
        logit_coef=np.asarray(logit_coef, dtype=float),
        early_exit_bf=int(config.early_exit_bf),
        method=method,
        cov=cov,
        model_version=str(config.workload_model_version),
    )


def predict_workload(model: WorkloadModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Return expected_bf, bf_sd, expected_pitches, expected_outs, p_early_exit."""
    x = design_matrix(frame, model.feature_names, model.medians)
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


def _train_frame_for_date(
    starts: pd.DataFrame,
    feature_rows: pd.DataFrame,
    date: object,
) -> pd.DataFrame:
    hist = starts.loc[starts["game_date"] < date].copy()
    if hist.empty:
        return hist
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
        if column in hist.columns
    ]
    keep_feat = [
        column
        for column in ("pitcher_id", "game_pk", *WORKLOAD_FEATURE_COLUMNS)
        if column in feature_rows.columns
    ]
    merged = hist[keep_start].merge(
        feature_rows[keep_feat],
        on=["pitcher_id", "game_pk"],
        how="left",
        suffixes=("", "_feat"),
    )
    for column in WORKLOAD_FEATURE_COLUMNS:
        feat_col = f"{column}_feat"
        if feat_col in merged.columns:
            merged[column] = merged[feat_col].where(
                merged[feat_col].notna(),
                merged[column] if column in merged.columns else np.nan,
            )
            merged = merged.drop(columns=[feat_col])
    return merged


def add_oof_workload_features(
    starts: pd.DataFrame,
    feature_rows: pd.DataFrame,
    config: MlbConfig,
) -> pd.DataFrame:
    """Expanding OOF workload predictions. Refits at most weekly; never in-sample BF."""
    np.random.seed(config.seed)
    out = feature_rows.copy()
    for column in _OOF_MAP:
        out[column] = np.nan

    if starts.empty or out.empty or "game_date" not in starts.columns:
        return out

    dates = np.sort(np.asarray(starts["game_date"].to_numpy()))
    dates = np.unique(dates)
    min_train = int(config.workload_min_train_starts)
    model = None
    last_fit_stamp: pd.Timestamp | None = None
    pos_frame = pd.DataFrame(
        {
            "pitcher_id": out["pitcher_id"].to_numpy(),
            "game_pk": out["game_pk"].to_numpy(),
            "_oof_pos": np.arange(len(out), dtype=np.int64),
        }
    )
    for date in dates:
        hist = _train_frame_for_date(starts, feature_rows, date)
        if len(hist) < min_train:
            continue
        stamp = pd.Timestamp(str(date))
        if (
            model is None
            or last_fit_stamp is None
            or (stamp - last_fit_stamp).days >= 7
        ):
            model = fit_workload(hist, config)
            last_fit_stamp = stamp
        day_keys = starts.loc[
            starts["game_date"] == date,
            ["pitcher_id", "game_pk"],
        ].drop_duplicates()
        if day_keys.empty:
            continue
        aligned = pos_frame.merge(day_keys, on=["pitcher_id", "game_pk"], how="inner")
        if aligned.empty:
            continue
        pos = aligned["_oof_pos"].to_numpy()
        preds = predict_workload(model, out.iloc[pos])
        labels = out.index.to_numpy()[pos]
        preds.index = labels
        for oof_name, pred_name in _OOF_MAP.items():
            out.loc[labels, oof_name] = preds[pred_name].to_numpy()
    return out
