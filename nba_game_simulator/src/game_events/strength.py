"""Predictive shrinkage for one rate.

Per-minute counts use a gamma prior on the rate, so the next game is
negative binomial. Makes out of attempts use a beta prior, so the next
game is beta-binomial. The strength is the prior's exposure, in minutes
or in attempts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from game_events.checks import _positive_float
from game_events.constants import PLAYER_EVENT_RATES, PLAYER_PRIOR_MINUTES
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

# log(strength) search. exp(-2) is about 0.14; exp(10) is about 22,000.
_LOG_STRENGTH_BOUNDS = (-2.0, 10.0)
_EXPOSURE_EDGES = (200.0, 1000.0)


def prior_minutes_map(value: Any) -> dict[str, float]:
    """One strength per rate. A single number is that strength for every rate."""
    if isinstance(value, bool):
        raise ValueError("player_prior_minutes must be a positive number or a rate map")
    if isinstance(value, (int, float, np.integer, np.floating)):
        strength = _positive_float(value, "player_prior_minutes")
        return {rate: strength for rate in PLAYER_EVENT_RATES}
    if isinstance(value, dict):
        mapped = {}
        for rate in PLAYER_EVENT_RATES:
            raw = value[rate] if rate in value else PLAYER_PRIOR_MINUTES
            mapped[rate] = _positive_float(raw, f"player_prior_minutes[{rate}]")
        return mapped
    raise ValueError("player_prior_minutes must be a positive number or a rate map")


def predictive_nll(log_k, n, d, t, x_f, e_f):
    """Gamma-Poisson posterior predictive for the next game.

    ``n`` and ``d`` are prior-window events and exposure. ``t`` is the prior
    target rate. ``x_f`` and ``e_f`` are the target game.
    """
    k = np.exp(log_k)
    alpha = np.asarray(n, dtype=float) + k * np.asarray(t, dtype=float)
    beta = np.asarray(d, dtype=float) + k
    exposure = np.asarray(e_f, dtype=float)
    events = np.asarray(x_f, dtype=float)
    probability = beta / (beta + exposure)
    log_likelihood = (
        gammaln(events + alpha)
        - gammaln(alpha)
        - gammaln(events + 1.0)
        + alpha * np.log(probability)
        + events * np.log1p(-probability)
    )
    return -float(np.sum(log_likelihood))


def predictive_attempt_nll(log_k, n, d, t, x_f, e_f):
    """Beta-binomial posterior predictive for the next game's makes."""
    k = np.exp(log_k)
    events = np.asarray(n, dtype=float)
    trials = np.asarray(d, dtype=float)
    target = np.asarray(t, dtype=float)
    makes = np.asarray(x_f, dtype=float)
    attempts = np.asarray(e_f, dtype=float)
    alpha = events + k * target
    beta = (trials - events) + k * (1.0 - target)
    log_likelihood = (
        gammaln(attempts + 1.0)
        - gammaln(makes + 1.0)
        - gammaln(attempts - makes + 1.0)
        + gammaln(makes + alpha)
        + gammaln(attempts - makes + beta)
        - gammaln(attempts + alpha + beta)
        - (gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))
    )
    return -float(np.sum(log_likelihood))


def tune_strength(n, d, t, x_f, e_f) -> float:
    """Minutes of shrinkage that minimize the next-game negative binomial loss."""
    return _tune(predictive_nll, n, d, t, x_f, e_f, _count_rows)


def tune_attempt_strength(n, d, t, x_f, e_f) -> float:
    """Attempts of shrinkage that minimize the next-game beta-binomial loss."""
    return _tune(predictive_attempt_nll, n, d, t, x_f, e_f, _attempt_rows)


def deviance_by_exposure(n, d, t, x_f, e_f, log_k) -> dict[str, float | None]:
    """Predictive deviance, 2 * nll, in prior-exposure buckets.

    Buckets are under 200, 200 through 1000, and over 1000. An empty bucket
    is ``None``.
    """
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    t = np.asarray(t, dtype=float)
    x_f = np.asarray(x_f, dtype=float)
    e_f = np.asarray(e_f, dtype=float)
    usable = _count_rows(n, d, t, x_f, e_f)
    k = float(np.exp(log_k))
    alpha = n + k * t
    beta = d + k
    probability = beta / (beta + e_f)
    row_deviance = -2.0 * (
        gammaln(x_f + alpha)
        - gammaln(alpha)
        - gammaln(x_f + 1.0)
        + alpha * np.log(probability)
        + x_f * np.log1p(-probability)
    )
    low, high = _EXPOSURE_EDGES
    groups = {
        "under_200": usable & (d < low),
        "200_to_1000": usable & (d >= low) & (d <= high),
        "over_1000": usable & (d > high),
    }
    return {
        name: float(row_deviance[mask].sum()) if np.any(mask) else None
        for name, mask in groups.items()
    }


def _tune(loss, n, d, t, x_f, e_f, usable) -> float:
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    t = np.asarray(t, dtype=float)
    x_f = np.asarray(x_f, dtype=float)
    e_f = np.asarray(e_f, dtype=float)
    ok = usable(n, d, t, x_f, e_f)
    if not np.any(ok):
        return float(PLAYER_PRIOR_MINUTES)
    result = minimize_scalar(
        loss,
        bounds=_LOG_STRENGTH_BOUNDS,
        method="bounded",
        args=(n[ok], d[ok], t[ok], x_f[ok], e_f[ok]),
    )
    if not result.success or not np.isfinite(result.x):
        return float(PLAYER_PRIOR_MINUTES)
    return float(np.exp(result.x))


def _count_rows(n, d, t, x_f, e_f) -> np.ndarray:
    return (
        np.isfinite(n)
        & np.isfinite(d)
        & np.isfinite(t)
        & np.isfinite(x_f)
        & np.isfinite(e_f)
        & (t >= 0.0)
        & (e_f > 0.0)
        & ((n > 0.0) | (t > 0.0))
    )


def _attempt_rows(n, d, t, x_f, e_f) -> np.ndarray:
    return (
        _count_rows(n, d, t, x_f, e_f)
        & (t < 1.0)
        & (x_f >= 0.0)
        & (x_f <= e_f)
        & ((d - n) > 0.0)
    )
