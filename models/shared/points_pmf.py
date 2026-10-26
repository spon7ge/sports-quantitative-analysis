"""Deterministic points PMF from a minutes grid and a conditional PPM grid.

No draws. Minutes values and PPM values are already on the quadrature
grid. Tails follow the minutes sampler's empirical ratios, with no floor
on the rate, so a run of knots at 0 keeps that probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from models.shared.minutes_sampler import QUANTILE_LEVELS, empirical_quantile

HOLDOUT_START = np.datetime64("2026-10-20")
_ROUNDING = ("rint", "floor")
_LEAKED_FEATURES = frozenset({"minutes", "pts", "predicted_minutes_oof"})


def check_feature_list(features, *, conditional: bool) -> list[str]:
    """Pregame names, plus ``minutes_given`` only on the conditional model.

    The list is whatever the caller trained. Game-N minutes and points are
    refused either way, so a later feature set cannot leak the label.
    """
    names = [str(column) for column in features]
    leaked = _LEAKED_FEATURES.intersection(names)
    if leaked:
        raise ValueError(f"leaked {sorted(leaked)}")
    given = "minutes_given" in names
    if conditional and not given:
        raise ValueError("minutes_given")
    if not conditional and given:
        raise ValueError("minutes_given")
    return names


@dataclass
class PpmTails:
    arrays: dict[tuple[str, int], np.ndarray]


def ppm_quantile_function(u, knots, levels, tails: PpmTails, groups) -> np.ndarray:
    """Quantile function. Knots are sorted. Interior is linear. Tails are empirical.

    ``u`` is (rows, points). ``knots`` is (rows, n_levels). ``groups`` selects
    the tail table for each row. A non-positive q05 maps the lower tail to 0
    when q05 is 0, because the scale is ``q05`` times the ratio table.
    """
    uniforms = np.asarray(u, dtype=float)
    prepared = np.sort(np.asarray(knots, dtype=float), axis=-1)
    if prepared.ndim == 1:
        prepared = prepared.reshape(1, -1)
    if uniforms.ndim == 1:
        uniforms = uniforms.reshape(1, -1)
    level_values = np.asarray(levels, dtype=float)
    if prepared.shape[1] != level_values.shape[0]:
        raise ValueError("knot count")
    if uniforms.shape[0] != prepared.shape[0]:
        raise ValueError("rows")
    group_ids = np.asarray(groups)
    if group_ids.shape[0] != prepared.shape[0]:
        raise ValueError("groups")
    out = np.empty(uniforms.shape, dtype=float)
    lower = uniforms < level_values[0]
    upper = uniforms > level_values[-1]
    middle = ~lower & ~upper
    if np.any(middle):
        out[middle] = _interp_shared_knots(uniforms, level_values, prepared)[middle]
    for group in np.unique(group_ids):
        mask = group_ids == int(group)
        if np.any(lower[mask]):
            table = _lookup(tails, "lower", int(group))
            ratios = _table_quantile(uniforms[mask] / float(level_values[0]), table)
            scaled = prepared[mask, 0][:, None] * ratios
            block = out[mask]
            block[lower[mask]] = scaled[lower[mask]]
            out[mask] = block
        if np.any(upper[mask]):
            table = _lookup(tails, "upper", int(group))
            width = 1.0 - float(level_values[-1])
            residual = _table_quantile(
                (uniforms[mask] - float(level_values[-1])) / width, table
            )
            shifted = prepared[mask, -1][:, None] + residual
            block = out[mask]
            block[upper[mask]] = shifted[upper[mask]]
            out[mask] = block
    return out


def _interp_shared_knots(u, levels, knots) -> np.ndarray:
    """Row-wise linear interpolation on one shared probability grid."""
    slots = np.searchsorted(levels, u, side="left")
    slots = np.clip(slots, 1, len(levels) - 1)
    left = slots - 1
    x0 = levels[left]
    x1 = levels[slots]
    rows = np.arange(knots.shape[0])[:, None]
    y0 = knots[rows, left]
    y1 = knots[rows, slots]
    width = x1 - x0
    weight = np.divide(u - x0, width, out=np.zeros_like(u), where=width != 0)
    return y0 + weight * (y1 - y0)


def _table_quantile(v, table: np.ndarray) -> np.ndarray:
    flat = empirical_quantile(np.asarray(v, dtype=float).ravel(), table)
    return flat.reshape(np.shape(v))


def _lookup(tails: PpmTails, tail: str, group: int) -> np.ndarray:
    try:
        return tails.arrays[(tail, int(group))]
    except KeyError as exc:
        raise ValueError(f"unknown group {group} for {tail}") from exc


def points_pmf(minutes_q, ppm_q, k_max: int, rounding: str) -> np.ndarray:
    """Equal-weight histogram of ``convert(minutes * ppm)``, clipped to ``[0, k_max]``.

    ``minutes_q`` is (n, G_M). ``ppm_q`` is (n, G_M, G_P). ``rint`` sends
    ``[k - 0.5, k + 0.5)`` to ``k``. ``floor`` drops the fraction.
    """
    minutes = np.asarray(minutes_q, dtype=float)
    ppm = np.asarray(ppm_q, dtype=float)
    if minutes.ndim != 2 or ppm.ndim != 3:
        raise ValueError("shape")
    if ppm.shape[0] != minutes.shape[0] or ppm.shape[1] != minutes.shape[1]:
        raise ValueError("shape")
    limit = int(k_max)
    if limit < 0:
        raise ValueError("k_max")
    if rounding not in _ROUNDING:
        raise ValueError("rounding")
    product = minutes[:, :, None] * ppm
    if rounding == "rint":
        counts = np.floor(product + 0.5)
    else:
        counts = np.floor(product)
    counts = np.clip(counts, 0, limit).astype(np.int64)
    n_rows, g_m, g_p = counts.shape
    width = limit + 1
    cells = g_m * g_p
    offsets = (np.arange(n_rows, dtype=np.int64) * width)[:, None]
    flat = np.bincount(
        (counts.reshape(n_rows, cells) + offsets).ravel(),
        minlength=n_rows * width,
    )
    return flat.reshape(n_rows, width).astype(float) / float(cells)


def ppm_prediction_frame(
    frame: pd.DataFrame,
    minutes_grid,
    features: list[str],
) -> pd.DataFrame:
    """Repeat each row once per minutes-grid value and set ``minutes_given``.

    Realized minutes are not copied into ``minutes_given``. Row ``i`` occupies
    ``G_M`` consecutive output rows, grid axis last, so predictions reshape
    to ``(n, G_M)``.
    """
    grid = np.asarray(minutes_grid, dtype=float)
    if grid.ndim != 2 or grid.shape[0] != len(frame):
        raise ValueError("minutes grid")
    g_m = grid.shape[1]
    base = frame.reset_index(drop=True)
    repeated = base.loc[base.index.repeat(g_m)].reset_index(drop=True)
    repeated["minutes_given"] = grid.reshape(-1)
    missing = [column for column in features if column not in repeated.columns]
    if missing:
        raise KeyError(f"missing {missing}")
    identifiers = [
        column
        for column in ("game_id", "player_id", "game_date", "starting")
        if column in repeated.columns and column not in features
    ]
    return repeated.loc[:, [*identifiers, *features]].copy()


def gaussian_copula_uniform(u_m, u_p, rho: float) -> np.ndarray:
    """``Φ(ρ Φ^{-1}(u_m) + √(1-ρ²) Φ^{-1}(u_p))``. ``rho`` is one number."""
    correlation = float(rho)
    if not np.isfinite(correlation) or correlation < -1.0 or correlation > 1.0:
        raise ValueError("rho")
    minutes_u = np.clip(np.asarray(u_m, dtype=float), 1e-12, 1.0 - 1e-12)
    ppm_u = np.clip(np.asarray(u_p, dtype=float), 1e-12, 1.0 - 1e-12)
    independent = np.sqrt(1.0 - correlation**2)
    latent = correlation * norm.ppf(minutes_u) + independent * norm.ppf(ppm_u)
    return norm.cdf(latent)


def build_ppm_tails(oof: pd.DataFrame, *, folds, holdout_start=HOLDOUT_START) -> PpmTails:
    """Lower ratios ``y / q05`` and upper residuals ``y - q95``, by starting.

    Knots are sorted and not floored. A holdout date is rejected. The lower
    pin is 1 and the upper pin is 0, matching the minutes tail tables.
    """
    dates = np.asarray(pd.to_datetime(oof["game_date"]), dtype="datetime64[ns]")
    if np.any(dates >= np.datetime64(holdout_start)):
        raise ValueError("holdout")
    present = set(int(fold) for fold in folds)
    chosen = oof.loc[oof["fold_id"].isin(present)].copy()
    if chosen.empty:
        raise ValueError("fold")
    columns = [f"q_{level:.2f}" for level in QUANTILE_LEVELS]
    grid = np.sort(chosen.loc[:, columns].to_numpy(dtype=float), axis=1)
    q05 = grid[:, 0]
    q95 = grid[:, -1]
    outcome = chosen["ppm"].to_numpy(dtype=float)
    starting = chosen["starting"].to_numpy(dtype=int)
    buckets: dict[tuple[str, int], list[float]] = {}
    for index in range(len(chosen)):
        group = int(starting[index])
        if outcome[index] < q05[index]:
            if q05[index] <= 0:
                raise ValueError("q05")
            buckets.setdefault(("lower", group), []).append(outcome[index] / q05[index])
        if outcome[index] > q95[index]:
            buckets.setdefault(("upper", group), []).append(outcome[index] - q95[index])
    arrays: dict[tuple[str, int], np.ndarray] = {}
    groups = sorted(set(int(value) for value in starting))
    for group in groups:
        lower = [*buckets.get(("lower", group), []), 1.0]
        upper = [*buckets.get(("upper", group), []), 0.0]
        arrays[("lower", group)] = np.sort(np.asarray(lower, dtype=float))
        arrays[("upper", group)] = np.sort(np.asarray(upper, dtype=float))
    return PpmTails(arrays=arrays)
