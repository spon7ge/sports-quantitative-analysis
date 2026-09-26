"""Inverse-transform minutes quantile function. Draws are not a price."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.settlement import maximum_minutes

QUANTILE_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
KNOT_FLOOR = 1e-3


@dataclass
class MinuteTailTables:
    arrays: dict[tuple[str, int], np.ndarray]


def prepare_quantile_grid(grids: np.ndarray) -> np.ndarray:
    """Sort once, then floor every knot at 1e-3. No second sort."""
    values = np.asarray(grids, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != len(QUANTILE_LEVELS):
        raise ValueError(f"knot count must be {len(QUANTILE_LEVELS)}")
    if not np.isfinite(values).all():
        raise ValueError("non-finite knot")
    ordered = np.sort(values, axis=1)
    return np.maximum(ordered, KNOT_FLOOR)


def empirical_quantile(v: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Linear empirical quantile. Probability 0 is table[0], probability 1 is table[-1]."""
    xp = np.linspace(0.0, 1.0, len(table))
    return np.interp(np.asarray(v, dtype=float), xp, np.asarray(table, dtype=float))


def _lookup(tables: MinuteTailTables, tail: str, group: int) -> np.ndarray:
    try:
        return tables.arrays[(tail, int(group))]
    except KeyError as exc:
        raise ValueError(f"unknown group {group} for {tail}") from exc


def _row_quantile(u, prepared, q05, q95, lower_group, upper_group, tables):
    out = np.empty(len(u), dtype=float)
    lower = u < 0.05
    upper = u > 0.95
    middle = ~lower & ~upper
    if np.any(lower):
        table = _lookup(tables, "lower", lower_group)
        out[lower] = q05 * empirical_quantile(u[lower] / 0.05, table)
    if np.any(upper):
        table = _lookup(tables, "upper", upper_group)
        out[upper] = q95 + empirical_quantile((u[upper] - 0.95) / 0.05, table)
    if np.any(middle):
        out[middle] = np.interp(u[middle], QUANTILE_LEVELS, prepared)
    return out


def quantile_minutes(u, grids, lower_groups, upper_groups, tables) -> np.ndarray:
    """Map uniforms of shape (rows, draws) through each row's Q."""
    uniforms = np.asarray(u, dtype=float)
    if uniforms.ndim != 2:
        raise ValueError("u must have shape (rows, draws)")
    if not np.isfinite(uniforms).all() or np.any(uniforms < 0) or np.any(uniforms > 1):
        raise ValueError("u outside [0, 1]")
    raw = np.asarray(grids, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    prepared = prepare_quantile_grid(grids)
    if prepared.shape[0] != uniforms.shape[0]:
        raise ValueError("grids and u must have the same number of rows")
    floored = np.maximum(raw, KNOT_FLOOR)
    lower_ids = np.asarray(lower_groups)
    upper_ids = np.asarray(upper_groups)
    out = np.empty_like(uniforms, dtype=float)
    for index in range(prepared.shape[0]):
        out[index] = _row_quantile(
            uniforms[index],
            prepared[index],
            floored[index, 0],
            floored[index, -1],
            int(lower_ids[index]),
            int(upper_ids[index]),
            tables,
        )
    cap = maximum_minutes("nba")
    return np.clip(out, 0.0, cap)
