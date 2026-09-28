from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.mlb.models.preprocess import compute_standardization
from src.mlb.models.workload import design_matrix, fit_nb2

MIN_FEATURE_STD = 1e-12


class UnestimatedDispersion(RuntimeError):
    """Raised when statsmodels returned no negative-binomial dispersion."""


@dataclass
class WalkForwardFit:
    feature_names: tuple[str, ...]
    coef: np.ndarray
    alpha: float
    method: str
    centers: dict[str, float]
    scales: dict[str, float]
    medians: dict[str, float]
    dispersion_estimated: bool
    dropped_features: tuple[str, ...] = field(default_factory=tuple)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def fit_walk_forward_nb2(
    train: pd.DataFrame,
    target: str,
    features: tuple[str, ...],
    binary: tuple[str, ...],
    l2: float,
    offset: np.ndarray | None = None,
    require_convergence: bool = False,
) -> WalkForwardFit:
    retained: list[str] = []
    dropped: list[str] = []
    medians: dict[str, float] = {}
    for column in features:
        values = _numeric(train, column)
        median = float(values.median()) if values.notna().any() else 0.0
        if not np.isfinite(median):
            median = 0.0
        filled = np.nan_to_num(
            values.fillna(median).to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if float(np.std(filled, ddof=0)) < MIN_FEATURE_STD:
            dropped.append(column)
            continue
        retained.append(column)
        medians[column] = median

    binary_set = set(binary)
    continuous = tuple(column for column in retained if column not in binary_set)
    centers, scales = compute_standardization(train, continuous, medians)
    for column in continuous:
        if column in centers:
            continue
        filled = _numeric(train, column).fillna(medians[column]).to_numpy(dtype=float)
        centers[column] = float(np.mean(filled))
        scales[column] = float(np.std(filled, ddof=0))

    names = tuple(retained)
    x = design_matrix(train, names, medians, centers=centers, scales=scales)
    y = _numeric(train, target).to_numpy(dtype=float)
    fit = fit_nb2(
        y,
        x,
        l2=float(l2),
        feature_names=names,
        offset=offset,
        require_convergence=require_convergence,
    )
    if fit.dispersion_estimated is False:
        raise UnestimatedDispersion(
            "negative-binomial dispersion was not estimated; "
            f"method={fit.method} n_train={len(train)} features={names}"
        )
    return WalkForwardFit(
        feature_names=names,
        coef=np.asarray(fit.coef, dtype=float),
        alpha=float(fit.alpha),
        method=fit.method,
        centers=centers,
        scales=scales,
        medians=medians,
        dispersion_estimated=True,
        dropped_features=tuple(dropped),
    )


def predict_walk_forward_mean(
    model: WalkForwardFit,
    frame: pd.DataFrame,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    x = design_matrix(
        frame,
        model.feature_names,
        model.medians,
        centers=model.centers,
        scales=model.scales,
    )
    eta = x @ model.coef
    if offset is not None:
        eta = eta + np.asarray(offset, dtype=float).reshape(-1)
    return np.exp(eta)
