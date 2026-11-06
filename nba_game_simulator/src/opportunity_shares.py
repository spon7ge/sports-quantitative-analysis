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
    allow_empty: bool = False,
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
        # Callers that can treat "nobody is eligible" as a zero allocation
        # (for example assists, when every make is then unassisted) opt in.
        # Everyone else still fails instead of inventing a uniform split.
        if allow_empty:
            return np.zeros(shape, dtype=float)
        raise ValueError("No available players with positive opportunity weight")
    return adjusted / total


def opportunity_share_rows(
    base_weights: np.ndarray,
    available: np.ndarray,
    projected_minutes: np.ndarray,
    reference_minutes: np.ndarray,
    role_factors: np.ndarray | None = None,
    lineup_factors: np.ndarray | None = None,
    minutes_exponent: float = 1.0,
    allow_empty: bool = False,
) -> np.ndarray:
    """Per-simulation shares. ``projected_minutes`` has shape (n_sims, players)."""
    projected = np.asarray(projected_minutes, dtype=float)
    if projected.ndim != 2:
        raise ValueError("projected_minutes must have shape (n_sims, n_players)")
    n_sims, n_players = projected.shape
    # Validate the player-level inputs once, on a single legal minute row.
    # The per-sim variation is applied below with the same formula.
    probe = np.ones(n_players, dtype=float)
    opportunity_shares(
        base_weights=base_weights,
        available=available,
        projected_minutes=probe,
        reference_minutes=reference_minutes,
        role_factors=role_factors,
        lineup_factors=lineup_factors,
        minutes_exponent=minutes_exponent,
        allow_empty=True,
    )
    base = _as_nonnegative_finite_vector(base_weights, "base_weights")
    if base.shape != (n_players,):
        raise ValueError("base_weights must align with projected_minutes")
    available_row = _as_nonnegative_finite_vector(available, "available", base.shape)
    if np.any((available_row != 0) & (available_row != 1)):
        raise ValueError("available must contain only 0 or 1")
    reference = _as_nonnegative_finite_vector(
        reference_minutes, "reference_minutes", base.shape
    )
    if np.any(~np.isfinite(projected)) or np.any(projected < 0):
        raise ValueError("projected_minutes must contain nonnegative finite values")
    try:
        minutes_exponent = float(minutes_exponent)
    except (TypeError, ValueError) as exc:
        raise ValueError("minutes_exponent must be finite and nonnegative") from exc
    if not np.isfinite(minutes_exponent) or minutes_exponent < 0:
        raise ValueError("minutes_exponent must be finite and nonnegative")
    if role_factors is None:
        role = np.ones(n_players, dtype=float)
    else:
        role = _as_nonnegative_finite_vector(role_factors, "role_factors", base.shape)
    if lineup_factors is None:
        lineup = np.ones(n_players, dtype=float)
    else:
        lineup = _as_nonnegative_finite_vector(
            lineup_factors, "lineup_factors", base.shape
        )

    ratio = np.divide(
        projected,
        reference[None, :],
        out=np.zeros((n_sims, n_players), dtype=float),
        where=reference[None, :] > 0,
    )
    minute_factor = np.where(
        reference[None, :] > 0,
        np.power(ratio, minutes_exponent),
        0.0,
    )
    adjusted = (
        base[None, :]
        * available_row[None, :]
        * minute_factor
        * role[None, :]
        * lineup[None, :]
    )
    row_sum = adjusted.sum(axis=1)
    shares = np.zeros_like(adjusted)
    positive = row_sum > 0
    shares[positive] = adjusted[positive] / row_sum[positive, None]
    if not allow_empty and not bool(np.all(positive)):
        raise ValueError("No available players with positive opportunity weight")
    return shares
