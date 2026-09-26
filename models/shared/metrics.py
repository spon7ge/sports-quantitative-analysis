"""Scoring helpers for quantile prop models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

# Default minutes bands (override per league/prop if needed).
DEFAULT_MIN_TIERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "<10 min": lambda a: a < 10,
    "10-20 min": lambda a: (a >= 10) & (a < 20),
    "20-30 min": lambda a: (a >= 20) & (a < 30),
    "30+ min": lambda a: a >= 30,
}

# Nominal central intervals that can be read off trained quantiles.
_INTERVALS: tuple[tuple[float, float, str], ...] = (
    (0.25, 0.75, "50"),
    (0.10, 0.90, "80"),
    (0.05, 0.95, "90"),
)


def pinball_loss(y_true, y_pred, alpha: float = 0.50) -> float:
    """Mean pinball (quantile) loss. Lower is better."""
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual)
    return float(np.mean(loss))


def pinball_50(y_true, y_pred, alpha: float = 0.50) -> float:
    """Negative pinball loss at ``alpha`` (higher is better for sklearn scorers)."""
    return -pinball_loss(y_true, y_pred, alpha)


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of outcomes inside ``[lower, upper]``."""
    actual = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((actual >= lo) & (actual <= hi)))


def _alpha_from_key(key: str) -> float | None:
    try:
        return float(str(key).split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _quantile_key(alpha: float) -> str:
    return f"q_{alpha:.2f}"


def score_quantile_fold(
    actual,
    preds: Mapping[str, np.ndarray],
    *,
    fold_label: str,
    starting=None,
    models: Mapping[str, Any] | None = None,
    tiers: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    lower_key: str = "q_0.10",
    median_key: str = "q_0.50",
    upper_key: str = "q_0.90",
    verbose: bool = True,
) -> dict[str, Any]:
    """Score one fold with pinball loss and interval coverage."""
    actual = np.asarray(actual, dtype=float)
    metrics: dict[str, Any] = {
        "fold": fold_label,
        "n": int(len(actual)),
    }

    pinball_bits: list[str] = []
    for key, pred in preds.items():
        alpha = _alpha_from_key(key)
        if alpha is None:
            continue
        loss = pinball_loss(actual, pred, alpha)
        metrics[f"pinball_{key}"] = loss
        pinball_bits.append(f"q{alpha:.2f}={loss:.3f}")
    metrics["pinball"] = metrics.get(f"pinball_{median_key}", float("nan"))

    coverage_bits: list[str] = []
    for lo, hi, label in _INTERVALS:
        lo_key, hi_key = _quantile_key(lo), _quantile_key(hi)
        if lo_key not in preds or hi_key not in preds:
            continue
        coverage = interval_coverage(actual, preds[lo_key], preds[hi_key])
        nominal = hi - lo
        metrics[f"coverage_{label}pct"] = coverage
        coverage_bits.append(
            f"{label}%={coverage:.1%} (target {nominal:.0%})"
        )
    if "coverage_80pct" not in metrics and lower_key in preds and upper_key in preds:
        metrics["coverage_80pct"] = interval_coverage(
            actual, preds[lower_key], preds[upper_key]
        )

    if models is not None:
        metrics["best_iters"] = {
            k: getattr(models[k], "best_iteration", None) for k in models
        }

    if verbose:
        print(f"\n{fold_label}")
        print(f"  n={len(actual):5d}")
        if pinball_bits:
            print(f"  pinball  | {'  '.join(pinball_bits)}")
        if coverage_bits:
            print(f"  coverage | {'  '.join(coverage_bits)}")

    median = (
        np.asarray(preds[median_key], dtype=float)
        if median_key in preds
        else None
    )
    has_interval = lower_key in preds and upper_key in preds
    lower = np.asarray(preds[lower_key], dtype=float) if has_interval else None
    upper = np.asarray(preds[upper_key], dtype=float) if has_interval else None

    def _slice(mask: np.ndarray, name: str) -> None:
        if mask.sum() == 0:
            return
        if median is not None:
            sliced_pinball = pinball_loss(actual[mask], median[mask], 0.50)
            metrics[f"pinball_{name}"] = sliced_pinball
        else:
            sliced_pinball = float("nan")
        if has_interval:
            sliced_coverage = interval_coverage(
                actual[mask], lower[mask], upper[mask]
            )
            metrics[f"coverage_{name}"] = sliced_coverage
        else:
            sliced_coverage = float("nan")
        if verbose:
            print(
                f"  {name:10s} | n={int(mask.sum()):5d} | "
                f"pinball q50: {sliced_pinball:.3f} | "
                f"80% coverage: {sliced_coverage:.1%}"
            )

    if starting is not None:
        starting = np.asarray(starting)
        for role, mask in (("Starters", starting == 1), ("Bench", starting == 0)):
            _slice(np.asarray(mask, dtype=bool), role)

    # Tiers describe the pregame minute projection, not the realized outcome.
    tier_values = median if median is not None else actual
    if verbose and median is not None:
        print("  minute tiers by predicted q50")
    for tier, fn in (tiers or DEFAULT_MIN_TIERS).items():
        _slice(np.asarray(fn(tier_values), dtype=bool), tier)

    return metrics
