"""Quantile scores, calibration, crossing, and interval sharpness."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from points_xgb.paths import ensure_repo_on_path

ensure_repo_on_path()
from models.shared.metrics import interval_coverage, pinball_loss  # noqa: E402
from models.shared.minutes_sampler import row_crps  # noqa: E402

TAIL_KNOTS = (
    ("lower", 0.05),
    ("lower", 0.10),
    ("upper", 0.90),
    ("upper", 0.95),
)

# Alpha is the excluded probability of the central interval.
WIS_INTERVALS = (
    (0.80, "q_0.40", "q_0.60"),
    (0.60, "q_0.30", "q_0.70"),
    (0.40, "q_0.20", "q_0.80"),
    (0.20, "q_0.10", "q_0.90"),
    (0.10, "q_0.05", "q_0.95"),
)


def _alpha(key: str) -> float:
    return float(str(key).split("_", 1)[1])


def row_pinball(actual, pred, alpha: float = 0.50) -> np.ndarray:
    """Per-row pinball. Positive residual uses ``alpha``; negative uses ``alpha - 1``."""
    residual = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    return np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual)


def score_predictions(
    y_true,
    preds: Mapping[str, np.ndarray],
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pinball loss per quantile and central interval coverage."""
    actual = np.asarray(y_true, dtype=float)
    pinball_rows = []
    for key, pred in preds.items():
        alpha = _alpha(key)
        pinball_rows.append(
            {
                "split": split,
                "quantile": alpha,
                "pinball": pinball_loss(actual, pred, alpha),
            }
        )
    coverage_rows = []
    for low, high, nominal in ((0.10, 0.90, 0.80), (0.05, 0.95, 0.90)):
        low_key, high_key = f"q_{low:.2f}", f"q_{high:.2f}"
        if low_key not in preds or high_key not in preds:
            continue
        coverage_rows.append(
            {
                "split": split,
                "interval": f"{nominal:.0%}",
                "coverage": interval_coverage(actual, preds[low_key], preds[high_key]),
                "target": nominal,
            }
        )
    return pd.DataFrame(pinball_rows), pd.DataFrame(coverage_rows)


def quantile_calibration(y_true, preds: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Share of outcomes at or below each predicted quantile."""
    actual = np.asarray(y_true, dtype=float)
    rows = []
    for key, pred in preds.items():
        alpha = _alpha(key)
        empirical = float(np.mean(actual <= np.asarray(pred, dtype=float)))
        rows.append(
            {
                "quantile": alpha,
                "target": alpha,
                "empirical": empirical,
                "delta": empirical - alpha,
            }
        )
    return pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True)


def quantile_crossing(
    preds: Mapping[str, np.ndarray],
) -> tuple[int, int, pd.DataFrame]:
    """Count rows whose predicted quantiles are not increasing."""
    keys = sorted(preds, key=_alpha)
    grid = np.column_stack([np.asarray(preds[key], dtype=float) for key in keys])
    crossed = np.any(np.diff(grid, axis=1) < 0, axis=1)
    n_rows = int(len(crossed))
    n_crossed = int(crossed.sum())
    pair_rows = []
    for left, right in zip(keys, keys[1:]):
        pair_rows.append(
            {
                "lower": left,
                "upper": right,
                "crossing_rate": float(
                    np.mean(
                        np.asarray(preds[left], dtype=float)
                        > np.asarray(preds[right], dtype=float)
                    )
                ),
            }
        )
    return n_crossed, n_rows, pd.DataFrame(pair_rows)


def predicted_rate_tiers(
    y_true,
    preds: Mapping[str, np.ndarray],
    tiers: Mapping[str, Callable],
) -> pd.DataFrame:
    """Holdout pinball, coverage, and width inside predicted-q50 tiers."""
    predicted = np.asarray(preds["q_0.50"], dtype=float)
    actual = np.asarray(y_true, dtype=float)
    low_80 = np.asarray(preds["q_0.10"], dtype=float)
    high_80 = np.asarray(preds["q_0.90"], dtype=float)
    low_90 = np.asarray(preds["q_0.05"], dtype=float)
    high_90 = np.asarray(preds["q_0.95"], dtype=float)
    width_80 = high_80 - low_80
    width_90 = high_90 - low_90
    rows = []
    for name, rule in tiers.items():
        mask = np.asarray(rule(predicted), dtype=bool)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "tier": name,
                "n": n,
                "share": n / len(predicted),
                "pinball_q50": pinball_loss(actual[mask], predicted[mask], 0.50),
                "empirical_q50": float(np.mean(actual[mask] <= predicted[mask])),
                "coverage_80": interval_coverage(
                    actual[mask], low_80[mask], high_80[mask]
                ),
                "coverage_90": interval_coverage(
                    actual[mask], low_90[mask], high_90[mask]
                ),
                "mean_width_80": float(np.mean(width_80[mask])),
                "mean_width_90": float(np.mean(width_90[mask])),
            }
        )
    return pd.DataFrame(rows)


def holdout_point_distribution(name: str, y, preds: Mapping[str, np.ndarray]) -> dict:
    """q50 RMSE and CRPS from the trained quantile knots.

    CRPS is twice the mean pinball across those knots.
    """
    actual = np.asarray(y, dtype=float)
    levels = sorted(_alpha(key) for key in preds)
    grid = np.column_stack(
        [np.asarray(preds[f"q_{level:.2f}"], dtype=float) for level in levels]
    )
    q50 = np.asarray(preds["q_0.50"], dtype=float)
    return {
        "model": name,
        "rmse": float(np.sqrt(mean_squared_error(actual, q50))),
        "crps": float(np.mean(row_crps(actual, grid, np.asarray(levels)))),
    }


def tail_calibration_rows(name: str, y, preds: Mapping[str, np.ndarray]) -> list[dict]:
    """Lower tail is outcomes at or below the knot. Upper tail is above it."""
    actual = np.asarray(y, dtype=float)
    rows = []
    for side, level in TAIL_KNOTS:
        knot = np.asarray(preds[f"q_{level:.2f}"], dtype=float)
        if side == "lower":
            nominal = level
            empirical = float(np.mean(actual <= knot))
        else:
            nominal = 1.0 - level
            empirical = float(np.mean(actual > knot))
        rows.append(
            {
                "model": name,
                "side": side,
                "quantile": level,
                "nominal": nominal,
                "empirical": empirical,
                "delta": empirical - nominal,
            }
        )
    return rows


def interval_sharpness(y_true, preds: Mapping[str, np.ndarray]) -> pd.DataFrame:
    actual = np.asarray(y_true, dtype=float)
    rows = []
    for low, high, nominal in ((0.10, 0.90, 0.80), (0.05, 0.95, 0.90)):
        low_key, high_key = f"q_{low:.2f}", f"q_{high:.2f}"
        width = np.asarray(preds[high_key], dtype=float) - np.asarray(
            preds[low_key], dtype=float
        )
        rows.append(
            {
                "interval": f"{nominal:.0%}",
                "coverage": interval_coverage(actual, preds[low_key], preds[high_key]),
                "target": nominal,
                "mean_width": float(np.mean(width)),
                "median_width": float(np.median(width)),
            }
        )
    return pd.DataFrame(rows)


def interval_score(y, lower, upper, alpha: float) -> np.ndarray:
    """Gneiting interval score. ``alpha`` is the probability outside the band."""
    actual = np.asarray(y, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    width = hi - lo
    below = actual < lo
    above = actual > hi
    return (
        width
        + (2.0 / alpha) * (lo - actual) * below
        + (2.0 / alpha) * (actual - hi) * above
    )


def wis_values(y, qmap: Mapping[str, np.ndarray]) -> np.ndarray:
    """Weighted interval score using the trained knots that are present."""
    actual = np.asarray(y, dtype=float)
    present = [
        (alpha, qmap[low], qmap[high])
        for alpha, low, high in WIS_INTERVALS
        if low in qmap and high in qmap
    ]
    total = 0.5 * np.abs(actual - np.asarray(qmap["q_0.50"], dtype=float))
    for alpha, lower, upper in present:
        total = total + (alpha / 2.0) * interval_score(actual, lower, upper, alpha)
    return total / (len(present) + 0.5)


def interval_stats(y, qmap: Mapping[str, np.ndarray], low: float, high: float) -> dict:
    low_key, high_key = f"q_{low:.2f}", f"q_{high:.2f}"
    if low_key not in qmap or high_key not in qmap:
        return {"cov": np.nan, "width_mean": np.nan, "width_median": np.nan}
    width = np.asarray(qmap[high_key], dtype=float) - np.asarray(
        qmap[low_key], dtype=float
    )
    return {
        "cov": interval_coverage(y, qmap[low_key], qmap[high_key]),
        "width_mean": float(np.mean(width)),
        "width_median": float(np.median(width)),
    }
