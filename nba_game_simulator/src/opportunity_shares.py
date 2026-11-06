"""Layer 4 opportunity-share adjustment.

``base_weights`` are baseline shares *before* the minutes adjustment.
"""

from __future__ import annotations

import numpy as np


def _as_nonnegative_finite_vector(
    value: np.ndarray,
    field: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain nonnegative finite values") from exc
    if arr.ndim != 1 or (expected_shape is not None and arr.shape != expected_shape):
        raise ValueError(f"{field} must be a one-dimensional player-aligned array")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{field} must contain nonnegative finite values")
    return arr


def opportunity_shares(
    base_weights: np.ndarray,
    available: np.ndarray,
    projected_minutes: np.ndarray,
    reference_minutes: np.ndarray,
    role_factors: np.ndarray | None = None,
    lineup_factors: np.ndarray | None = None,
    minutes_exponent: float = 1.0,
) -> np.ndarray:
    base_weights = _as_nonnegative_finite_vector(base_weights, "base_weights")
    shape = base_weights.shape
    available = _as_nonnegative_finite_vector(available, "available", shape)
    if np.any((available != 0) & (available != 1)):
        raise ValueError("available must contain only 0 or 1")
    projected_minutes = _as_nonnegative_finite_vector(
        projected_minutes, "projected_minutes", shape
    )
    reference_minutes = _as_nonnegative_finite_vector(
        reference_minutes, "reference_minutes", shape
    )
    try:
        minutes_exponent = float(minutes_exponent)
    except (TypeError, ValueError) as exc:
        raise ValueError("minutes_exponent must be finite and nonnegative") from exc
    if not np.isfinite(minutes_exponent) or minutes_exponent < 0:
        raise ValueError("minutes_exponent must be finite and nonnegative")

    n = shape[0]
    if role_factors is None:
        role_factors = np.ones(n, dtype=float)
    else:
        role_factors = _as_nonnegative_finite_vector(
            role_factors, "role_factors", shape
        )
    if lineup_factors is None:
        lineup_factors = np.ones(n, dtype=float)
    else:
        lineup_factors = _as_nonnegative_finite_vector(
            lineup_factors, "lineup_factors", shape
        )

    minute_ratio = np.divide(
        projected_minutes,
        reference_minutes,
        out=np.zeros_like(projected_minutes, dtype=float),
        where=reference_minutes > 0,
    )
    minute_factor = np.where(
        reference_minutes > 0,
        np.power(minute_ratio, minutes_exponent),
        0.0,
    )
    adjusted = (
        base_weights
        * available
        * minute_factor
        * role_factors
        * lineup_factors
    )
    total = float(adjusted.sum())
    if total <= 0:
        raise ValueError("No available players with positive opportunity weight")
    return adjusted / total
