"""Sportsbook probability helpers for MLB strikeout quotes.

Wraps `src.models.odds.american_to_implied_probability` when the American
formula matches, and adds decimal / no-vig / EV utilities used by market
comparison. Does not modify the NBA odds module.
"""

from __future__ import annotations

import numpy as np

from src.models.odds import american_to_implied_probability

ArrayLike = float | int | np.ndarray | list[float] | list[int]


def _as_float_array(odds: ArrayLike) -> tuple[np.ndarray, bool]:
    arr = np.asarray(odds, dtype=float)
    return arr, arr.ndim == 0


def american_to_implied(odds: ArrayLike) -> float | np.ndarray:
    """Convert American odds to raw implied probability.

    A > 0: p = 100 / (A + 100). A < 0: p = -A / (-A + 100).
    """
    arr, scalar = _as_float_array(odds)
    if scalar:
        return float(american_to_implied_probability(float(arr)))
    out = np.empty(arr.shape, dtype=float)
    for idx, value in np.ndenumerate(arr):
        out[idx] = american_to_implied_probability(float(value))
    return out


def decimal_to_implied(odds: ArrayLike) -> float | np.ndarray:
    """Convert decimal odds to raw implied probability: p = 1 / d for d > 1."""
    arr, scalar = _as_float_array(odds)
    if np.any(arr <= 1.0):
        raise ValueError("Decimal odds must be greater than 1.")
    implied = 1.0 / arr
    return float(implied) if scalar else implied


def implied_to_decimal(p: ArrayLike) -> float | np.ndarray:
    """Invert an implied probability to decimal odds: d = 1 / p."""
    arr, scalar = _as_float_array(p)
    if np.any(arr <= 0.0):
        raise ValueError("Implied probability must be positive.")
    decimal = 1.0 / arr
    return float(decimal) if scalar else decimal


def no_vig(
    p_over: ArrayLike,
    p_under: ArrayLike,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Remove multiplicative overround: q_i = p_i / (p_over + p_under)."""
    over_arr, over_scalar = _as_float_array(p_over)
    under_arr, under_scalar = _as_float_array(p_under)
    total = over_arr + under_arr
    if np.any(total <= 0.0):
        raise ValueError("p_over + p_under must be positive to remove vig.")
    q_over = over_arr / total
    q_under = under_arr / total
    if over_scalar and under_scalar:
        return float(q_over), float(q_under)
    return q_over, q_under


def expected_profit(
    p_win: float,
    p_loss: float,
    decimal_odds: float,
    *,
    p_push: float = 0.0,
) -> float:
    """Expected profit per dollar staked at decimal odds ``d``.

    EV = p_win * (d - 1) - p_loss. A push returns the stake, so the
    p_push term is zero and does not change EV.
    """
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.")
    _ = p_push  # stake returned; contribution is identically 0
    return float(p_win) * (float(decimal_odds) - 1.0) - float(p_loss)
