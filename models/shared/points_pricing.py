"""Load the minutes and conditional PPM bundles and price a points line.

The stored booster fingerprints are checked on load. A replaced bundle
raises before any probability is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from models.shared.artifacts import booster_fingerprint
from models.shared.points_pmf import points_pmf


@dataclass
class PointsPricing:
    minutes_models: dict[str, Any]
    minutes_features: list[str]
    ppm_models: dict[str, Any]
    ppm_features: list[str]
    tails: Any
    g_m: int
    g_p: int
    rounding: str
    k_max: int
    calibrator: Any
    minutes_fingerprint: str
    ppm_fingerprint: str
    rho: None
    minutes_bundle: str
    ppm_bundle: str
    minutes_tails: str | None = None


def save_points_pricing(
    path,
    *,
    minutes_bundle,
    ppm_bundle,
    minutes_models,
    minutes_features,
    ppm_models,
    ppm_features,
    tails,
    g_m: int,
    g_p: int,
    rounding: str,
    k_max: int,
    calibrator=None,
    minutes_tails=None,
) -> Path:
    """Write the pricing payload. The bundles themselves are not copied."""
    destination = Path(path)
    payload = {
        "minutes_bundle": str(minutes_bundle),
        "ppm_bundle": str(ppm_bundle),
        "minutes_features": list(minutes_features),
        "ppm_features": list(ppm_features),
        "minutes_fingerprint": booster_fingerprint(minutes_models, minutes_features),
        "ppm_fingerprint": booster_fingerprint(ppm_models, ppm_features),
        "tails": tails,
        "g_m": int(g_m),
        "g_p": int(g_p),
        "rounding": str(rounding),
        "k_max": int(k_max),
        "calibrator": calibrator,
        "rho": None,
        "minutes_tails": None if minutes_tails is None else str(minutes_tails),
        "minutes_tails_fingerprint": None
        if minutes_tails is None
        else sha256(Path(minutes_tails).read_bytes()).hexdigest(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, destination)
    return destination


def load_points_pricing(path) -> PointsPricing:
    """Load bundles and refuse them when a fingerprint does not match."""
    payload = joblib.load(path)
    minutes = joblib.load(payload["minutes_bundle"])
    ppm = joblib.load(payload["ppm_bundle"])
    minutes_found = booster_fingerprint(minutes["quantile_models"], minutes["feature_names"])
    ppm_found = booster_fingerprint(ppm["quantile_models"], ppm["feature_names"])
    if minutes_found != payload["minutes_fingerprint"]:
        raise ValueError("fingerprint")
    if ppm_found != payload["ppm_fingerprint"]:
        raise ValueError("fingerprint")
    tails_path = payload.get("minutes_tails")
    if tails_path is not None:
        found_tails = sha256(Path(tails_path).read_bytes()).hexdigest()
        if found_tails != payload.get("minutes_tails_fingerprint"):
            raise ValueError("fingerprint")
    return PointsPricing(
        minutes_models=minutes["quantile_models"],
        minutes_features=list(minutes["feature_names"]),
        ppm_models=ppm["quantile_models"],
        ppm_features=list(ppm["feature_names"]),
        tails=payload["tails"],
        g_m=int(payload["g_m"]),
        g_p=int(payload["g_p"]),
        rounding=str(payload["rounding"]),
        k_max=int(payload["k_max"]),
        calibrator=payload.get("calibrator"),
        minutes_fingerprint=payload["minutes_fingerprint"],
        ppm_fingerprint=payload["ppm_fingerprint"],
        rho=payload.get("rho"),
        minutes_bundle=payload["minutes_bundle"],
        ppm_bundle=payload["ppm_bundle"],
        minutes_tails=tails_path,
    )


def price_points(
    frame: pd.DataFrame,
    lines,
    *,
    minutes_q,
    ppm_q,
    rounding: str,
    k_max: int,
    calibrator=None,
) -> tuple[np.ndarray, np.ndarray]:
    """PMF and P(points > line) for grids that are already on the quadrature.

    ``lines`` is broadcast to one row per PMF row. A half-point line such as
    12.5 is cleared by 13 or more.
    """
    del frame
    pmf = points_pmf(minutes_q, ppm_q, int(k_max), rounding)
    if calibrator is not None:
        from models.shared.points_calibration import apply_points_cdf_calibrator

        pmf = apply_points_cdf_calibrator(np.cumsum(pmf, axis=1), calibrator)
    thresholds = np.asarray(lines, dtype=float)
    if thresholds.ndim == 1:
        thresholds = np.broadcast_to(thresholds, (pmf.shape[0], thresholds.shape[0])).copy()
    if thresholds.shape[0] != pmf.shape[0]:
        raise ValueError("lines")
    return pmf, _over_probability(pmf, thresholds)


def predict_distribution(pricing: PointsPricing, frame: pd.DataFrame, lines, minutes_tables):
    """Predict minutes, condition PPM on that grid, and return the points PMF."""
    from models.shared.minutes_sampler import QUANTILE_LEVELS, groups_for_frame, ppf
    from models.shared.points_pmf import ppm_prediction_frame, ppm_quantile_function

    levels = [float(level) for level in QUANTILE_LEVELS]
    names = [f"q_{level:.2f}" for level in levels]
    missing = [name for name in names if name not in pricing.minutes_models or name not in pricing.ppm_models]
    if missing:
        raise KeyError(f"missing quantiles {missing}")
    minute_knots = np.column_stack(
        [
            pricing.minutes_models[name].predict(frame.loc[:, pricing.minutes_features])
            for name in names
        ]
    )
    uniforms = np.tile((np.arange(pricing.g_m) + 0.5) / pricing.g_m, (len(frame), 1))
    lower, upper = groups_for_frame(frame["starting"], minutes_tables)
    minutes_q = ppf(uniforms, minute_knots, lower, upper, minutes_tables)
    batched = ppm_prediction_frame(frame, minutes_q, pricing.ppm_features)
    predictors = batched.loc[:, pricing.ppm_features]
    knots = np.empty((len(frame), pricing.g_m, len(levels)), dtype=float)
    for index, name in enumerate(names):
        predicted = pricing.ppm_models[name].predict(predictors)
        knots[:, :, index] = np.asarray(predicted, dtype=float).reshape(len(frame), pricing.g_m)
    flat = knots.reshape(len(frame) * pricing.g_m, len(levels))
    ppm_u = np.tile((np.arange(pricing.g_p) + 0.5) / pricing.g_p, (flat.shape[0], 1))
    groups = np.repeat(frame["starting"].to_numpy(dtype=int), pricing.g_m)
    ppm = ppm_quantile_function(ppm_u, flat, levels, pricing.tails, groups)
    return price_points(
        frame,
        lines,
        minutes_q=minutes_q,
        ppm_q=ppm.reshape(len(frame), pricing.g_m, pricing.g_p),
        rounding=pricing.rounding,
        k_max=pricing.k_max,
        calibrator=pricing.calibrator,
    )


def _over_probability(pmf: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """P(points > line) = 1 - F(floor(line))."""
    cdf = np.cumsum(pmf, axis=1)
    floor_line = np.floor(lines).astype(int)
    index = np.clip(floor_line, 0, pmf.shape[1] - 1)
    rows = np.arange(pmf.shape[0])[:, None]
    probability = 1.0 - cdf[rows, index]
    probability = np.where(floor_line < 0, 1.0, probability)
    probability = np.where(floor_line >= pmf.shape[1] - 1, 0.0, probability)
    return probability
