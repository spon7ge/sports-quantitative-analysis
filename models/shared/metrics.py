"""Scoring helpers for quantile prop models."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from models.shared.minutes_sampler import canonical_id
from models.shared.oos import QUANTILE_LEVELS, capped_rate

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
    tier_source = DEFAULT_MIN_TIERS if tiers is None else tiers
    if verbose and median is not None and tier_source:
        print("  minute tiers by predicted q50")
    for tier, fn in tier_source.items():
        _slice(np.asarray(fn(tier_values), dtype=bool), tier)

    return metrics


# Cuts on predicted minutes q50. Realized minutes are not a tier.
MINUTE_Q50_TIERS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<15", None, 15.0),
    ("15-24", 15.0, 24.0),
    ("24-31", 24.0, 31.0),
    ("31+", 31.0, None),
)

_COVERAGE_BANDS: tuple[tuple[str, float, float, float], ...] = (
    ("Q20-Q80", 0.20, 0.80, 0.60),
    ("Q10-Q90", 0.10, 0.90, 0.80),
    ("Q05-Q95", 0.05, 0.95, 0.90),
)


def calibration_by_minutes_tier(
    frame: pd.DataFrame,
    *,
    tolerance: float = 0.02,
    rate_cap: float = 6.0,
) -> pd.DataFrame:
    """Knot shares and interval coverage by predicted minutes q50.

    Pre-holdout OOS and holdout are separate. A check is flagged when
    ``abs(empirical - ideal)`` is greater than ``tolerance`` (2 percentage
    points). The rate target is the capped points-per-minute label.
    """
    predicted_q50 = frame["minutes_q_0.50"].to_numpy(dtype=float)
    is_holdout = frame["is_holdout"].astype(bool).to_numpy()
    outcomes = {
        "minutes": frame["minutes"].to_numpy(dtype=float),
        "rate": capped_rate(frame["pts"], frame["minutes"], cap=rate_cap),
    }
    splits = (
        ("pre-holdout OOS", ~is_holdout),
        ("holdout", is_holdout),
    )
    rows: list[dict[str, Any]] = []
    for target, outcome in outcomes.items():
        knots = {
            level: frame[f"{target}_q_{level:.2f}"].to_numpy(dtype=float)
            for level in QUANTILE_LEVELS
        }
        for split, split_mask in splits:
            for tier, low, high in MINUTE_Q50_TIERS:
                mask = split_mask & _tier_mask(predicted_q50, low, high)
                n = int(mask.sum())
                if n == 0:
                    continue
                y = outcome[mask]
                for level in QUANTILE_LEVELS:
                    empirical = float(np.mean(y <= knots[level][mask]))
                    rows.append(
                        _calibration_row(
                            target=target,
                            split=split,
                            tier=tier,
                            n=n,
                            check=f"q{level:.2f}",
                            ideal=float(level),
                            empirical=empirical,
                            tolerance=tolerance,
                        )
                    )
                for name, low_q, high_q, ideal in _COVERAGE_BANDS:
                    inside = (y >= knots[low_q][mask]) & (y <= knots[high_q][mask])
                    rows.append(
                        _calibration_row(
                            target=target,
                            split=split,
                            tier=tier,
                            n=n,
                            check=name,
                            ideal=ideal,
                            empirical=float(np.mean(inside)),
                            tolerance=tolerance,
                        )
                    )
    return pd.DataFrame(rows)


def _tier_mask(
    predicted_q50: np.ndarray,
    low: float | None,
    high: float | None,
) -> np.ndarray:
    mask = np.ones(len(predicted_q50), dtype=bool)
    if low is not None:
        mask &= predicted_q50 >= low
    if high is not None:
        mask &= predicted_q50 < high
    return mask


def _calibration_row(
    *,
    target: str,
    split: str,
    tier: str,
    n: int,
    check: str,
    ideal: float,
    empirical: float,
    tolerance: float,
) -> dict[str, Any]:
    gap = empirical - ideal
    # A gap that lands on the tolerance (2pp) is not a miss. Float noise
    # around that boundary stays unflagged.
    return {
        "target": target,
        "split": split,
        "tier": tier,
        "n": n,
        "check": check,
        "ideal": ideal,
        "empirical": empirical,
        "gap": gap,
        "flag": abs(gap) - tolerance > 1e-9,
    }


_LOG_SCORE_FLOOR = 1e-12


def integer_log_score(pmf, y):
    """Log score of an integer outcome, ``log(p_y)``.

    Higher is better. Probability is floored at 1e-12 so an outcome
    outside the PMF support scores as ``log(1e-12)``.
    """
    mat, y_int, scalar = _broadcast_pmf(pmf, y)
    probability = np.maximum(_pmf_at(mat, y_int), _LOG_SCORE_FLOOR)
    scores = np.log(probability)
    if scalar:
        return float(scores[0])
    return scores


def rps(pmf, y):
    """Ranked probability score ``sum_k (F(k) - 1{y <= k})^2``.

    Lower is better. A point mass on ``y`` inside the support scores 0.
    """
    mat, y_int, scalar = _broadcast_pmf(pmf, y)
    ks = np.arange(mat.shape[1])
    cdf = np.cumsum(mat, axis=1)
    indicator = (y_int[:, None] <= ks[None, :]).astype(float)
    scores = np.sum((cdf - indicator) ** 2, axis=1)
    if scalar:
        return float(scores[0])
    return scores


def randomized_pit(pmf, y, atom_u):
    """Randomized PIT ``F(y - 1) + U (F(y) - F(y - 1))``.

    ``F(-1) = 0``. ``atom_u`` is caller-supplied. An outcome above the
    last bin scores 1.
    """
    mat, y_int, _scalar = _broadcast_pmf(pmf, y)
    draw = np.asarray(atom_u, dtype=float)
    if draw.ndim == 0:
        draw = np.full(mat.shape[0], float(draw))
    else:
        draw = draw.reshape(-1)
        if draw.shape[0] == 1 and mat.shape[0] > 1:
            draw = np.full(mat.shape[0], float(draw[0]))
        elif mat.shape[0] == 1 and draw.shape[0] > 1:
            mat = np.repeat(mat, draw.shape[0], axis=0)
            y_int = np.repeat(y_int, draw.shape[0])
        elif draw.shape[0] != mat.shape[0]:
            raise ValueError("u")
    if not np.isfinite(draw).all() or np.any(draw < 0) or np.any(draw > 1):
        raise ValueError("u outside [0, 1]")
    cdf = np.cumsum(mat, axis=1)
    n, width = mat.shape
    rows = np.arange(n)
    f_y = np.zeros(n, dtype=float)
    f_left = np.zeros(n, dtype=float)
    inside = (y_int >= 0) & (y_int < width)
    f_y[inside] = cdf[rows[inside], y_int[inside]]
    positive = inside & (y_int > 0)
    f_left[positive] = cdf[rows[positive], y_int[positive] - 1]
    above = y_int >= width
    f_y[above] = 1.0
    f_left[above] = 1.0
    pit = f_left + draw * np.clip(f_y - f_left, 0.0, None)
    if np.ndim(atom_u) == 0 and np.asarray(pmf).ndim == 1 and np.ndim(y) == 0:
        return float(pit[0])
    return pit


def _broadcast_pmf(pmf, y):
    mat = np.asarray(pmf, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    elif mat.ndim != 2:
        raise ValueError("pmf")
    y_int = _integer_outcomes(y)
    if mat.shape[0] == 1 and y_int.shape[0] > 1:
        mat = np.repeat(mat, y_int.shape[0], axis=0)
    if y_int.shape[0] == 1 and mat.shape[0] > 1:
        y_int = np.repeat(y_int, mat.shape[0])
    if mat.shape[0] != y_int.shape[0]:
        raise ValueError("pmf rows and y length must match")
    scalar = np.asarray(pmf).ndim == 1 and np.ndim(y) == 0
    return mat, y_int, scalar


def _integer_outcomes(y) -> np.ndarray:
    values = np.asarray(y, dtype=float).reshape(-1)
    if values.size and (
        not np.isfinite(values).all()
        or np.any(np.abs(values - np.rint(values)) > 1e-6)
    ):
        raise ValueError("y")
    return np.rint(values).astype(int)


def _pmf_at(mat: np.ndarray, y_int: np.ndarray) -> np.ndarray:
    n, width = mat.shape
    out = np.zeros(n, dtype=float)
    inside = (y_int >= 0) & (y_int < width)
    rows = np.arange(n)
    out[inside] = mat[rows[inside], y_int[inside]]
    return out


# Points PMF evaluation. A gap that lands on a tolerance is not a miss.
PIT_MEAN_IDEAL = 0.5
PIT_MEAN_TOLERANCE = 0.01
PIT_VAR_SCALE_IDEAL = 1.0
PIT_VAR_SCALE_TOLERANCE = 0.05
PIT_TAIL_IDEAL = 0.05
PIT_TAIL_TOLERANCE = 0.01
OVER_RATE_TOLERANCE = 0.015
FLAG_SE = 2.0
N_BOOT_DATES = 2_000
WILSON_Z = 1.96
PSEUDO_LINE_OFFSETS = (-8.0, -4.0, 0.0, 4.0, 8.0)
PSEUDO_LINE_NAMES = ("L0-8", "L0-4", "L0", "L0+4", "L0+8")
FAVORED_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("50-55", 0.50, 0.55),
    ("55-60", 0.55, 0.60),
    ("60-65", 0.60, 0.65),
    ("65-70", 0.65, 0.70),
    ("70-75", 0.70, 0.75),
    ("75-80", 0.75, 0.80),
    ("80+", 0.80, None),
)
_PIT_COVERAGE = (
    ("coverage_50", 0.25, 0.75, 0.50),
    ("coverage_80", 0.10, 0.90, 0.80),
    ("coverage_90", 0.05, 0.95, 0.90),
)
_NAIVE_POINT_COLS = (
    ("last game", "pts_lag_1"),
    ("season-to-date", "season_pts_mean"),
    ("EWMA-hl3", "pts_ewm_hl_3"),
)


def seeded_unit_uniform(
    player_ids,
    game_ids,
    *,
    seed: int = 42,
    label: str = "points_pit",
):
    """One ``U(0, 1)`` per player-game, stable across reruns."""
    players = list(player_ids)
    games = list(game_ids)
    if len(players) != len(games):
        raise ValueError("id")
    out = np.empty(len(players), dtype=float)
    for index, (player_id, game_id) in enumerate(zip(players, games, strict=True)):
        material = (
            f"{int(seed)}|{label}|{canonical_id(player_id)}|{canonical_id(game_id)}"
        ).encode()
        digest = sha256(material).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        out[index] = float(rng.random())
    return out


def pseudo_lines(ewma_points) -> np.ndarray:
    """``L0 = floor(EWMA) + 0.5``, then ``L0`` ± 4 and ± 8, floored at 0.5.

    Returns shape ``(n, 5)`` in the order ``L0-8, L0-4, L0, L0+4, L0+8``.
    A missing EWMA leaves that row missing.
    """
    ewma = np.asarray(ewma_points, dtype=float).reshape(-1)
    l0 = np.floor(ewma) + 0.5
    offsets = np.asarray(PSEUDO_LINE_OFFSETS, dtype=float)
    lines = np.maximum(l0[:, None] + offsets[None, :], 0.5)
    lines[~np.isfinite(ewma)] = np.nan
    return lines


def probability_over(pmf, lines):
    """Mass on integers strictly above ``floor(line)`` for half-point lines."""
    mat = np.asarray(pmf, dtype=float)
    squeeze = mat.ndim == 1 and np.ndim(lines) == 0
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    elif mat.ndim != 2:
        raise ValueError("pmf")
    level = np.asarray(lines, dtype=float).reshape(-1)
    if mat.shape[0] == 1 and level.shape[0] > 1:
        mat = np.repeat(mat, level.shape[0], axis=0)
    if level.shape[0] == 1 and mat.shape[0] > 1:
        level = np.repeat(level, mat.shape[0])
    if mat.shape[0] != level.shape[0]:
        raise ValueError("line")
    finite = np.isfinite(level)
    whole = finite & np.isclose(level, np.rint(level), atol=1e-8)
    if np.any(whole):
        raise ValueError("line")
    width = mat.shape[1]
    cdf = np.cumsum(mat, axis=1)
    out = np.full(mat.shape[0], np.nan)
    floor_k = np.zeros(mat.shape[0], dtype=int)
    floor_k[finite] = np.floor(level[finite]).astype(int)
    below = finite & (floor_k < 0)
    above = finite & (floor_k >= width - 1)
    inside = finite & ~below & ~above
    rows = np.arange(mat.shape[0])
    out[inside] = 1.0 - cdf[rows[inside], floor_k[inside]]
    out[below] = 1.0
    out[above] = 0.0
    if squeeze:
        return float(out[0])
    return out


def binary_log_loss(probability, outcome) -> np.ndarray:
    """Bernoulli log loss. Lower is better."""
    p = np.clip(
        np.asarray(probability, dtype=float),
        _LOG_SCORE_FLOOR,
        1.0 - _LOG_SCORE_FLOOR,
    )
    y = np.asarray(outcome, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def binary_brier(probability, outcome) -> np.ndarray:
    """Bernoulli Brier score. Lower is better."""
    p = np.asarray(probability, dtype=float)
    y = np.asarray(outcome, dtype=float)
    return (p - y) ** 2


def wilson_interval(successes, n: int, *, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    count = float(successes)
    size = int(n)
    if size <= 0 or not np.isfinite(count):
        return float("nan"), float("nan")
    p = count / size
    z2 = float(z) ** 2
    denom = 1.0 + z2 / size
    center = (p + z2 / (2.0 * size)) / denom
    half = z * math.sqrt(p * (1.0 - p) / size + z2 / (4.0 * size * size)) / denom
    return float(center - half), float(center + half)


def logistic_calibration(probability, outcome) -> tuple[float, float]:
    """Intercept and slope of ``logit P(over)`` in a Bernoulli logit.

    Returns ``(nan, nan)`` when the outcome or the logit has no variation.
    """
    p = np.asarray(probability, dtype=float).reshape(-1)
    y = np.asarray(outcome, dtype=float).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("outcome")
    finite = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[finite], 1e-6, 1.0 - 1e-6)
    y = y[finite]
    if p.size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    logit = np.log(p / (1.0 - p))
    if np.unique(np.round(logit, 8)).size < 2:
        return float("nan"), float("nan")
    beta = np.zeros(2, dtype=float)
    for _ in range(25):
        eta = np.clip(beta[0] + beta[1] * logit, -20.0, 20.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(mu * (1.0 - mu), 1e-8)
        working = eta + (y - mu) / weight
        sw = np.sqrt(weight)
        design = np.column_stack([sw, sw * logit])
        nxt, *_ = np.linalg.lstsq(design, sw * working, rcond=None)
        if np.max(np.abs(nxt - beta)) < 1e-10:
            beta = nxt
            break
        beta = nxt
    else:
        return float("nan"), float("nan")
    if not np.isfinite(beta).all():
        return float("nan"), float("nan")
    return float(beta[0]), float(beta[1])


def incremental_slope(outcome, probability, base_rate) -> float:
    """OLS slope of ``(outcome - base rate)`` on ``(P(Over) - base rate)``.

    An intercept is included. Near 0, ``P(Over)`` adds nothing past the
    tier's constant over-rate. No slope when the probability does not vary.
    """
    base = np.asarray(base_rate, dtype=float).reshape(-1)
    y = np.asarray(outcome, dtype=float).reshape(-1) - base
    x = np.asarray(probability, dtype=float).reshape(-1) - base
    if y.shape != x.shape:
        raise ValueError("outcome")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    x_center = x - x.mean()
    denom = float(np.dot(x_center, x_center))
    if denom <= 0.0:
        return float("nan")
    y_center = y - y.mean()
    return float(np.dot(x_center, y_center) / denom)


def paired_date_bootstrap(
    score_a,
    score_b,
    dates,
    *,
    n_boot: int = N_BOOT_DATES,
    seed: int = 42,
) -> dict[str, Any]:
    """95% percentile interval for ``mean(score_b - score_a)``.

    Resamples game dates. Every player-game on a drawn date is kept
    together, with the date's row count.
    """
    left = np.asarray(score_a, dtype=float).reshape(-1)
    right = np.asarray(score_b, dtype=float).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("score")
    diff = right - left
    finite = np.isfinite(diff)
    diff = diff[finite]
    if diff.size == 0:
        raise ValueError("empty")
    date_values = _naive_dates(dates).to_numpy()[finite]
    codes, _uniques = pd.factorize(date_values, sort=False)
    n_clusters = int(codes.max()) + 1
    cluster_n = np.bincount(codes, minlength=n_clusters).astype(float)
    cluster_sum = np.bincount(codes, weights=diff, minlength=n_clusters)
    point = float(diff.mean())
    if n_clusters == 1 or int(n_boot) <= 0:
        return _bootstrap_row(diff.size, n_clusters, point, point, point)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n_clusters, size=(int(n_boot), n_clusters))
    weights = np.zeros((int(n_boot), n_clusters), dtype=np.int32)
    np.add.at(
        weights,
        (np.repeat(np.arange(int(n_boot)), n_clusters), draws.ravel()),
        1,
    )
    denom = weights @ cluster_n
    boot = (weights @ cluster_sum) / denom
    low, high = np.quantile(boot, [0.025, 0.975])
    return _bootstrap_row(diff.size, n_clusters, point, float(low), float(high))


def fit_l0_base_rates(history: pd.DataFrame, *, before) -> pd.DataFrame:
    """Constant ``P(points > L0)`` by predicted-minutes tier before ``before``.

    Rows on or after ``before`` are ignored. ``L0`` is each row's own
    EWMA pseudo-line.
    """
    empty = pd.DataFrame(columns=["tier", "rate", "n"])
    if history is None or len(history) == 0:
        return empty
    cutoff = pd.Timestamp(before)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    dates = _naive_dates(history["game_date"])
    keep = dates < cutoff
    frame = history.loc[keep]
    if frame.empty:
        return empty
    ewma = frame["pts_ewm_hl_3"].to_numpy(dtype=float)
    lines = pseudo_lines(ewma)[:, PSEUDO_LINE_NAMES.index("L0")]
    points = frame["pts"].to_numpy(dtype=float)
    tiers = _tier_labels(frame)
    finite = np.isfinite(lines) & np.isfinite(points)
    rows = []
    for name, _low, _high in MINUTE_Q50_TIERS:
        mask = finite & (tiers == name)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "tier": name,
                "rate": float(np.mean(points[mask] > lines[mask])),
                "n": n,
            }
        )
    if not rows:
        return empty
    return pd.DataFrame(rows)


def evaluate_points_pmfs(
    scored: pd.DataFrame,
    pmf_a,
    pmf_b,
    *,
    history: pd.DataFrame,
    n_boot: int = N_BOOT_DATES,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Score independent PMF A against empirical PMF B on one period.

    ``history`` may contain the scored rows. Base rates use only rows
    with ``game_date`` before the first scored date. Each pseudo-line is
    a separate player-game sample: offsets from one row are not stacked.
    """
    frame = scored.reset_index(drop=True)
    if frame.empty:
        raise ValueError("empty")
    mat_a = _as_pmf_matrix(pmf_a, len(frame))
    mat_b = _as_pmf_matrix(pmf_b, len(frame))
    if mat_a.shape != mat_b.shape:
        raise ValueError("pmf")
    y = frame["pts"].to_numpy(dtype=float)
    tiers = _tier_labels(frame)
    dates = _naive_dates(frame["game_date"])
    atom = seeded_unit_uniform(
        frame["player_id"],
        frame["game_id"],
        seed=seed,
        label="points_pit",
    )
    pit_a = np.asarray(randomized_pit(mat_a, y, atom), dtype=float)
    pit_b = np.asarray(randomized_pit(mat_b, y, atom), dtype=float)
    log_a = np.asarray(integer_log_score(mat_a, y), dtype=float)
    log_b = np.asarray(integer_log_score(mat_b, y), dtype=float)
    rps_a = np.asarray(rps(mat_a, y), dtype=float)
    rps_b = np.asarray(rps(mat_b, y), dtype=float)
    before = pd.Timestamp(dates.min())
    base_rates = fit_l0_base_rates(history, before=before)
    return {
        "pit": _pit_summary(pit_a, pit_b, tiers),
        "pit_histogram": _pit_histogram(pit_a, pit_b, tiers),
        "zero": _zero_table(mat_a, mat_b, y, tiers),
        "scores": _score_table(
            log_a, log_b, rps_a, rps_b, dates, tiers, n_boot=n_boot, seed=seed
        ),
        "lines": _line_table(frame, mat_a, mat_b, tiers),
        "calibration": _calibration_table(frame, mat_b, tiers),
        "favored": _favored_table(frame, mat_b, tiers),
        "naive_points": _naive_points_table(frame, mat_b, tiers),
        "naive_line": _naive_line_table(frame, mat_b, tiers, base_rates),
    }


def _flag_gap(estimate, ideal, tolerance, se) -> bool:
    if not np.isfinite(estimate) or not np.isfinite(ideal) or not np.isfinite(se):
        return False
    gap = abs(float(estimate) - float(ideal))
    outside = gap - float(tolerance) > 1e-9
    far = gap - FLAG_SE * float(se) > 1e-9
    return bool(outside and far)


def _mean_se(values: np.ndarray) -> float:
    n = int(values.shape[0])
    if n < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / math.sqrt(n))


def _proportion_se(share: float, n: int) -> float:
    if n <= 0 or not np.isfinite(share):
        return float("nan")
    return float(math.sqrt(share * (1.0 - share) / n))


def _var_scale_se(values: np.ndarray) -> float:
    """SE of ``12 * s^2`` with ``s^2`` the unbiased sample variance."""
    n = int(values.shape[0])
    if n < 4:
        return float("nan")
    centered = values - values.mean()
    m2 = float(np.dot(centered, centered) / n)
    m4 = float(np.dot(centered**2, centered**2) / n)
    var_s2 = m4 / n - (m2**2) * (n - 3) / (n * (n - 1))
    if not np.isfinite(var_s2) or var_s2 < 0.0:
        var_s2 = 0.0
    return float(12.0 * math.sqrt(var_s2))


def _as_pmf_matrix(pmf, n_rows: int) -> np.ndarray:
    mat = np.asarray(pmf, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != n_rows:
        raise ValueError("pmf")
    return mat


def _naive_dates(values) -> pd.Series:
    if isinstance(values, pd.Series):
        dates = pd.to_datetime(values)
    else:
        dates = pd.to_datetime(pd.Series(np.asarray(values)))
    if getattr(dates.dt, "tz", None) is not None:
        return dates.dt.tz_localize(None)
    return dates


def _tier_labels(frame: pd.DataFrame) -> np.ndarray:
    if "tier" in frame.columns:
        labels = np.asarray(frame["tier"].to_numpy(), dtype=object)
    else:
        if "minutes_q_0.50" not in frame.columns:
            raise KeyError("tier")
        labels = _labels_from_q50(frame["minutes_q_0.50"].to_numpy(dtype=float))
    known = {name for name, _low, _high in MINUTE_Q50_TIERS}
    if labels.size and not set(np.unique(labels)).issubset(known):
        raise ValueError("tier")
    return labels


def _labels_from_q50(predicted_q50: np.ndarray) -> np.ndarray:
    q50 = np.asarray(predicted_q50, dtype=float)
    labels = np.empty(q50.shape[0], dtype=object)
    seen = np.zeros(q50.shape[0], dtype=bool)
    for name, low, high in MINUTE_Q50_TIERS:
        mask = _tier_mask(q50, low, high)
        labels[mask] = name
        seen |= mask
    if not seen.all() or not np.isfinite(q50).all():
        raise ValueError("tier")
    return labels


def _iter_slices(tiers: np.ndarray):
    yield "all", np.ones(tiers.shape[0], dtype=bool)
    for name, _low, _high in MINUTE_Q50_TIERS:
        yield name, tiers == name


def _bootstrap_row(n, n_dates, diff, low, high) -> dict[str, Any]:
    return {
        "n": int(n),
        "n_dates": int(n_dates),
        "diff": float(diff),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _pit_summary(pit_a, pit_b, tiers) -> pd.DataFrame:
    rows = []
    for model, pit in (("A", pit_a), ("B", pit_b)):
        for tier, mask in _iter_slices(tiers):
            values = np.asarray(pit[mask], dtype=float)
            n = int(values.shape[0])
            if n == 0:
                continue
            mean = float(np.mean(values))
            mean_se = _mean_se(values)
            if n < 2:
                var_scale = float("nan")
            else:
                var_scale = float(12.0 * np.var(values, ddof=1))
            var_se = _var_scale_se(values)
            below = float(np.mean(values < 0.05))
            above = float(np.mean(values > 0.95))
            below_se = _proportion_se(below, n)
            above_se = _proportion_se(above, n)
            row = {
                "model": model,
                "tier": tier,
                "n": n,
                "mean": mean,
                "mean_se": mean_se,
                "mean_flag": _flag_gap(
                    mean, PIT_MEAN_IDEAL, PIT_MEAN_TOLERANCE, mean_se
                ),
                "var_scale": var_scale,
                "var_scale_se": var_se,
                "var_scale_flag": _flag_gap(
                    var_scale,
                    PIT_VAR_SCALE_IDEAL,
                    PIT_VAR_SCALE_TOLERANCE,
                    var_se,
                ),
                "share_below_05": below,
                "share_below_05_se": below_se,
                "share_below_05_flag": _flag_gap(
                    below, PIT_TAIL_IDEAL, PIT_TAIL_TOLERANCE, below_se
                ),
                "share_above_95": above,
                "share_above_95_se": above_se,
                "share_above_95_flag": _flag_gap(
                    above, PIT_TAIL_IDEAL, PIT_TAIL_TOLERANCE, above_se
                ),
            }
            for name, low, high, _ideal in _PIT_COVERAGE:
                share = float(np.mean((values >= low) & (values <= high)))
                row[name] = share
                row[f"{name}_se"] = _proportion_se(share, n)
            rows.append(row)
    return pd.DataFrame(rows)


def _pit_histogram(pit_a, pit_b, tiers) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0.0, 1.0, 11)
    for model, pit in (("A", pit_a), ("B", pit_b)):
        for tier, mask in _iter_slices(tiers):
            values = np.asarray(pit[mask], dtype=float)
            n = int(values.shape[0])
            if n == 0:
                continue
            counts, _edges = np.histogram(values, bins=edges)
            shares = counts / n
            for bin_id, share in enumerate(shares):
                rows.append(
                    {
                        "model": model,
                        "tier": tier,
                        "bin": bin_id,
                        "lo": float(edges[bin_id]),
                        "hi": float(edges[bin_id + 1]),
                        "share": float(share),
                        "n": n,
                        "ideal": 0.1,
                    }
                )
    return pd.DataFrame(rows)


def _zero_table(mat_a, mat_b, y, tiers) -> pd.DataFrame:
    rows = []
    observed_flag = np.asarray(y == 0, dtype=float)
    for model, mat in (("A", mat_a), ("B", mat_b)):
        predicted = mat[:, 0]
        for tier, mask in _iter_slices(tiers):
            if int(mask.sum()) == 0:
                continue
            pred = predicted[mask]
            obs = observed_flag[mask]
            n = int(obs.shape[0])
            pred_mean = float(np.mean(pred))
            obs_mean = float(np.mean(obs))
            se = _mean_se(pred - obs)
            rows.append(
                {
                    "model": model,
                    "tier": tier,
                    "n": n,
                    "predicted": pred_mean,
                    "observed": obs_mean,
                    "se": se,
                    "flag": _flag_gap(
                        pred_mean, obs_mean, OVER_RATE_TOLERANCE, se
                    ),
                }
            )
    return pd.DataFrame(rows)


def _score_table(
    log_a,
    log_b,
    rps_a,
    rps_b,
    dates,
    tiers,
    *,
    n_boot,
    seed,
) -> pd.DataFrame:
    rows = []
    for tier, mask in _iter_slices(tiers):
        if int(mask.sum()) == 0:
            continue
        log_ci = paired_date_bootstrap(
            log_a[mask], log_b[mask], dates[mask], n_boot=n_boot, seed=seed
        )
        rps_ci = paired_date_bootstrap(
            rps_a[mask], rps_b[mask], dates[mask], n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "tier": tier,
                "n": int(mask.sum()),
                "n_dates": log_ci["n_dates"],
                "log_score_a": float(np.mean(log_a[mask])),
                "log_score_b": float(np.mean(log_b[mask])),
                "log_score_diff": log_ci["diff"],
                "log_score_ci_low": log_ci["ci_low"],
                "log_score_ci_high": log_ci["ci_high"],
                "rps_a": float(np.mean(rps_a[mask])),
                "rps_b": float(np.mean(rps_b[mask])),
                "rps_diff": rps_ci["diff"],
                "rps_ci_low": rps_ci["ci_low"],
                "rps_ci_high": rps_ci["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def _line_table(frame, mat_a, mat_b, tiers) -> pd.DataFrame:
    # One row per line. A player-game's five offsets are not one sample.
    ewma = frame["pts_ewm_hl_3"].to_numpy(dtype=float)
    lines = pseudo_lines(ewma)
    points = frame["pts"].to_numpy(dtype=float)
    rows = []
    for tier, mask in _iter_slices(tiers):
        for offset, name in enumerate(PSEUDO_LINE_NAMES):
            level = lines[:, offset]
            use = mask & np.isfinite(level)
            n = int(use.sum())
            if n == 0:
                continue
            p_a = probability_over(mat_a[use], level[use])
            p_b = probability_over(mat_b[use], level[use])
            outcome = (points[use] > level[use]).astype(float)
            observed = float(np.mean(outcome))
            predicted_b = float(np.mean(p_b))
            gap_se = _mean_se(p_b - outcome)
            rows.append(
                {
                    "tier": tier,
                    "line": name,
                    "n": n,
                    "p_over_a": float(np.mean(p_a)),
                    "p_over_b": predicted_b,
                    "observed": observed,
                    "observed_se": _proportion_se(observed, n),
                    "gap_b": predicted_b - observed,
                    "gap_se": gap_se,
                    "flag": _flag_gap(
                        predicted_b, observed, OVER_RATE_TOLERANCE, gap_se
                    ),
                    "log_loss_a": float(np.mean(binary_log_loss(p_a, outcome))),
                    "log_loss_b": float(np.mean(binary_log_loss(p_b, outcome))),
                    "brier_a": float(np.mean(binary_brier(p_a, outcome))),
                    "brier_b": float(np.mean(binary_brier(p_b, outcome))),
                }
            )
    return pd.DataFrame(rows)


def _l0_over(frame, mat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = pseudo_lines(frame["pts_ewm_hl_3"].to_numpy(dtype=float))
    level = lines[:, PSEUDO_LINE_NAMES.index("L0")]
    finite = np.isfinite(level)
    probability = np.full(len(frame), np.nan)
    if finite.any():
        probability[finite] = probability_over(mat[finite], level[finite])
    outcome = np.full(len(frame), np.nan)
    points = frame["pts"].to_numpy(dtype=float)
    outcome[finite] = (points[finite] > level[finite]).astype(float)
    return level, probability, outcome


def _calibration_table(frame, mat_b, tiers) -> pd.DataFrame:
    _level, probability, outcome = _l0_over(frame, mat_b)
    rows = []
    for tier, mask in _iter_slices(tiers):
        use = mask & np.isfinite(probability) & np.isfinite(outcome)
        n = int(use.sum())
        if n == 0:
            continue
        intercept, slope = logistic_calibration(probability[use], outcome[use])
        rows.append(
            {
                "tier": tier,
                "n": n,
                "intercept": intercept,
                "slope": slope,
            }
        )
    return pd.DataFrame(rows)


def _favored_table(frame, mat_b, tiers) -> pd.DataFrame:
    _level, probability, outcome = _l0_over(frame, mat_b)
    favored = np.maximum(probability, 1.0 - probability)
    hit = np.where(probability >= 0.5, outcome, 1.0 - outcome)
    rows = []
    finite = np.isfinite(probability) & np.isfinite(outcome)
    for tier, mask in _iter_slices(tiers):
        for label, low, high in FAVORED_BUCKETS:
            bucket = finite & mask & (favored >= low)
            if high is None:
                bucket &= favored <= 1.0
            else:
                bucket &= favored < high
            n = int(bucket.sum())
            if n == 0:
                continue
            successes = float(np.sum(hit[bucket]))
            observed = successes / n
            predicted = float(np.mean(favored[bucket]))
            low_ci, high_ci = wilson_interval(successes, n)
            se = _mean_se(favored[bucket] - hit[bucket])
            rows.append(
                {
                    "tier": tier,
                    "line": "L0",
                    "bucket": label,
                    "n": n,
                    "predicted": predicted,
                    "observed": observed,
                    "wilson_low": low_ci,
                    "wilson_high": high_ci,
                    "flag": _flag_gap(
                        predicted, observed, OVER_RATE_TOLERANCE, se
                    ),
                }
            )
    return pd.DataFrame(rows)


def _naive_points_table(frame, mat_b, tiers) -> pd.DataFrame:
    ks = np.arange(mat_b.shape[1], dtype=float)
    pmf_mean = mat_b @ ks
    cdf = np.cumsum(mat_b, axis=1)
    pmf_median = np.argmax(cdf >= 0.5 - 1e-12, axis=1).astype(float)
    y = frame["pts"].to_numpy(dtype=float)
    available = np.isfinite(y) & np.isfinite(pmf_mean) & np.isfinite(pmf_median)
    baselines = {"PMF": (pmf_median, pmf_mean)}
    for label, column in _NAIVE_POINT_COLS:
        values = frame[column].to_numpy(dtype=float)
        baselines[label] = (values, values)
        available &= np.isfinite(values)
    rows = []
    for tier, mask in _iter_slices(tiers):
        use = mask & available
        n = int(use.sum())
        if n == 0:
            continue
        actual = y[use]
        for label, (mae_pred, rmse_pred) in baselines.items():
            mae_error = mae_pred[use] - actual
            rmse_error = rmse_pred[use] - actual
            rows.append(
                {
                    "tier": tier,
                    "predictor": label,
                    "n": n,
                    "mae": float(np.mean(np.abs(mae_error))),
                    "rmse": float(np.sqrt(np.mean(rmse_error**2))),
                }
            )
    return pd.DataFrame(rows)


def _naive_line_table(frame, mat_b, tiers, base_rates: pd.DataFrame) -> pd.DataFrame:
    _level, probability, outcome = _l0_over(frame, mat_b)
    rate_map = {
        str(row.tier): float(row.rate) for row in base_rates.itertuples(index=False)
    }
    fit_n = {
        str(row.tier): int(row.n) for row in base_rates.itertuples(index=False)
    }
    base = np.full(len(frame), np.nan)
    for name, rate in rate_map.items():
        base[tiers == name] = rate
    finite = np.isfinite(probability) & np.isfinite(outcome) & np.isfinite(base)
    rows = []
    for tier, mask in _iter_slices(tiers):
        use = mask & finite
        n = int(use.sum())
        if tier == "all":
            n_fit = int(sum(fit_n.values()))
        else:
            n_fit = int(fit_n.get(tier, 0))
        if n == 0:
            continue
        constant = np.clip(base[use], _LOG_SCORE_FLOOR, 1.0 - _LOG_SCORE_FLOOR)
        rows.append(
            {
                "tier": tier,
                "n": n,
                "n_fit": n_fit,
                "log_loss_b": float(
                    np.mean(binary_log_loss(probability[use], outcome[use]))
                ),
                "log_loss_constant": float(
                    np.mean(binary_log_loss(constant, outcome[use]))
                ),
                "slope": incremental_slope(outcome[use], probability[use], base[use]),
            }
        )
    return pd.DataFrame(rows)
