"""One-sided conformal offsets for a fixed set of quantile knots.

Offsets are fit on scores ``y - q_τ(x)`` from a calibration set the
booster did not train on. Each group has its own offset. A group with
fewer than ``min_n`` rows is left out of the table and receives no
shift at apply time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from models.shared.artifacts import monotonize_quantiles
from models.shared.metrics import MINUTE_Q50_TIERS

# Same floor the minutes sampler uses before it builds a quantile function.
_KNOT_FLOOR = 1e-3
_GROUP_ORDER = ("cold_start", *[name for name, _low, _high in MINUTE_Q50_TIERS])


def conformal_offset(scores, tau: float) -> float:
    """k-th smallest score, with k on the finite-sample side of ``tau``.

    ``k = ceil((n+1)τ)`` for τ ≥ 0.5 and ``k = floor((n+1)τ)`` for
    τ < 0.5. ``k`` is 1-based. A rank outside ``1..n`` is an error:
    the sample is too small for that knot.
    """
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be finite and non-empty")
    level = float(tau)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("tau must be in (0, 1)")
    n = int(values.size)
    rank = (n + 1) * level
    nearest = round(rank)
    if abs(rank - nearest) <= 1e-8 * max(1.0, abs(rank)):
        rank = float(nearest)
    # ceil keeps an upper knot from undercovering. floor keeps a lower
    # knot from overcovering. The two rules meet at τ = 0.5.
    if level >= 0.5:
        k = int(math.ceil(rank))
    else:
        k = int(math.floor(rank))
    if k < 1 or k > n:
        raise ValueError(f"conformal rank k={k} is outside 1..{n}")
    ordered = np.sort(values)
    return float(ordered[k - 1])


def calibration_group(q50, min_lag_1, tiers=MINUTE_Q50_TIERS) -> np.ndarray:
    """Group label for each row, from base q50 unless the lag is missing.

    ``tiers`` are ``(name, low, high)`` cuts, left-closed. The default
    is the minute tiers. A missing lag is ``cold_start`` even when q50
    falls in a tier. Those rows were replaced by role quantiles, so a
    tier offset fit on tree scores would not belong to them.
    """
    predicted = pd.to_numeric(
        pd.Series(np.asarray(q50).reshape(-1)), errors="coerce"
    ).to_numpy(dtype=float)
    lag = pd.to_numeric(
        pd.Series(np.asarray(min_lag_1).reshape(-1)), errors="coerce"
    ).to_numpy(dtype=float)
    if predicted.shape[0] != lag.shape[0]:
        raise ValueError("q50 and min_lag_1 lengths differ")
    cold = ~np.isfinite(lag)
    if np.any(~cold & ~np.isfinite(predicted)):
        raise ValueError("q50")
    labels = np.empty(predicted.shape[0], dtype=object)
    labels[cold] = "cold_start"
    assigned = cold.copy()
    for name, low, high in tiers:
        mask = ~cold
        if low is not None:
            mask &= predicted >= low
        if high is not None:
            mask &= predicted < high
        labels[mask] = name
        assigned |= mask
    if not bool(assigned.all()):
        raise ValueError("q50")
    return labels


def fit_offsets(
    base: pd.DataFrame,
    y,
    group,
    levels: dict[str, float],
    min_n: int = 300,
    group_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One offset per group and knot. Groups under ``min_n`` are omitted.

    Rows are groups. Columns are the knot keys in ``levels`` order.
    """
    if not levels:
        raise ValueError("levels")
    if int(min_n) < 1:
        raise ValueError("min_n")
    keys = list(levels)
    missing = [key for key in keys if key not in base.columns]
    if missing:
        raise ValueError(f"base is missing {missing}")
    actual = np.asarray(y, dtype=float).reshape(-1)
    labels = np.asarray(group, dtype=object).reshape(-1)
    if actual.shape[0] != len(base) or labels.shape[0] != len(base):
        raise ValueError("length")
    grid = base.loc[:, keys].to_numpy(dtype=float)
    if not np.isfinite(grid).all() or not np.isfinite(actual).all():
        raise ValueError("non-finite")
    seen: list[object] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    preferred = list(_GROUP_ORDER if group_order is None else group_order)
    order = [name for name in preferred if name in seen]
    order.extend(name for name in seen if name not in order)
    rows: list[list[float]] = []
    index: list[object] = []
    threshold = int(min_n)
    for label in order:
        mask = labels == label
        if int(mask.sum()) < threshold:
            continue
        scores = actual[mask]
        rows.append(
            [
                conformal_offset(scores - grid[mask, column], float(tau))
                for column, tau in enumerate(levels.values())
            ]
        )
        index.append(label)
    out = pd.DataFrame(rows, index=pd.Index(index, name="group"), columns=keys)
    return out


def apply_offsets(
    base: pd.DataFrame,
    group,
    offsets: pd.DataFrame,
    max_minutes: float,
    *,
    lower: float = _KNOT_FLOOR,
) -> pd.DataFrame:
    """Add the group offset, sort each row, then clip.

    The default clip is ``[1e-3, max_minutes]``. ``lower`` may be 0 for
    a rate. The fourth argument is the inclusive upper clip. A group
    with no row in ``offsets`` gets a zero offset. Knots come back in
    increasing level order.
    """
    floor = float(lower)
    limit = float(max_minutes)
    if not np.isfinite(floor) or not np.isfinite(limit) or limit < floor:
        raise ValueError("max_minutes")
    keys = list(base.columns)
    if not keys or any(key not in offsets.columns for key in keys):
        raise ValueError("offsets")
    labels = np.asarray(group, dtype=object).reshape(-1)
    if labels.shape[0] != len(base):
        raise ValueError("length")
    grid = base.to_numpy(dtype=float)
    if not np.isfinite(grid).all():
        raise ValueError("non-finite")
    table = offsets.loc[:, keys]
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError("offsets")
    lookup = {str(label): row for row, label in enumerate(table.index)}
    add = np.zeros_like(grid)
    matched = np.fromiter(
        (lookup.get(str(label), -1) for label in labels),
        dtype=int,
        count=labels.shape[0],
    )
    known = matched >= 0
    add[known] = table.to_numpy(dtype=float)[matched[known]]
    shifted = {key: grid[:, column] + add[:, column] for column, key in enumerate(keys)}
    mono = monotonize_quantiles(shifted)
    ordered = sorted(mono, key=lambda key: float(str(key).split("_", 1)[1]))
    clipped = np.clip(
        np.column_stack([mono[key] for key in ordered]),
        floor,
        limit,
    )
    return pd.DataFrame(clipped, index=base.index, columns=ordered)


def role_quantile_table(
    history: pd.DataFrame,
    levels,
    target_col: str = "minutes",
) -> pd.DataFrame:
    """Empirical starter and bench quantiles. Both roles must have outcomes."""
    if not {target_col, "starting"}.issubset(history.columns):
        raise ValueError("history")
    taus = [float(level) for level in levels]
    keys = [f"q_{level:.2f}" for level in taus]
    rows = {}
    for name, flag in (("starter", 1), ("bench", 0)):
        rows[name] = _empirical_quantiles(history, flag, taus, target_col)
    return pd.DataFrame.from_dict(rows, orient="index", columns=keys)


def calibration_payload(
    offsets: pd.DataFrame,
    groups,
    dates,
    *,
    min_n: int,
    role_quantiles: pd.DataFrame,
    max_minutes: float,
    target: str = "minutes",
    lag_col: str = "min_lag_1",
    tiers=MINUTE_Q50_TIERS,
    clip_low: float = _KNOT_FLOOR,
) -> dict:
    """Offset table plus the facts a later load uses to refuse a mismatch.

    The booster fingerprint is not included. ``save_model_bundle`` hashes
    the boosters it actually writes. ``max_minutes`` is the inclusive
    upper clip. Minute defaults keep a previously saved call valid.
    """
    labels = np.asarray(groups, dtype=object).reshape(-1)
    when = pd.to_datetime(np.asarray(dates).reshape(-1))
    if labels.shape[0] != when.shape[0]:
        raise ValueError("groups and dates lengths differ")
    if len(when) == 0 or bool(when.isna().any()):
        raise ValueError("dates")
    names = ["cold_start", *[name for name, _low, _high in tiers]]
    for label in labels:
        if label not in names:
            names.append(label)
    return {
        "offsets": offsets,
        "group_definition": {
            "cold_start": f"non-finite {lag_col}",
            "tiers": tuple(tuple(item) for item in tiers),
        },
        "min_n": int(min_n),
        "date_start": pd.Timestamp(when.min()).date().isoformat(),
        "date_end": pd.Timestamp(when.max()).date().isoformat(),
        "row_counts": {str(name): int((labels == name).sum()) for name in names},
        "role_quantiles": role_quantiles,
        "max_minutes": float(max_minutes),
        "target": str(target),
        "lag_col": str(lag_col),
        "clip_low": float(clip_low),
        "clip_high": float(max_minutes),
    }


def _empirical_quantiles(
    history: pd.DataFrame,
    role_value: int,
    taus,
    target_col: str = "minutes",
) -> np.ndarray:
    outcome = history.loc[history["starting"] == role_value, target_col].to_numpy(
        dtype=float
    )
    outcome = outcome[np.isfinite(outcome)]
    if outcome.size == 0:
        raise ValueError(f"no history minutes for role {role_value}")
    return np.quantile(outcome, np.asarray(taus, dtype=float))


def base_knots(
    preds: Mapping[str, np.ndarray] | pd.DataFrame,
    starting,
    min_lag_1,
    history: pd.DataFrame | None = None,
    *,
    role_quantiles: pd.DataFrame | None = None,
    target_col: str = "minutes",
) -> pd.DataFrame:
    """Sort each row, then replace cold starts with role quantiles.

    ``history`` is the only minutes the fallback may see. A stored
    ``role_quantiles`` table is that same summary, kept so a later
    predict does not need the training rows. A cold row is one whose
    ``min_lag_1`` is missing.
    """
    if isinstance(preds, pd.DataFrame):
        index = preds.index
        mapping = {
            column: preds[column].to_numpy(dtype=float) for column in preds.columns
        }
    else:
        index = None
        mapping = {key: np.asarray(value, dtype=float) for key, value in preds.items()}
    if not mapping:
        raise ValueError("preds")
    lengths = {len(value) for value in mapping.values()}
    if len(lengths) != 1:
        raise ValueError("preds")
    n = lengths.pop()
    if any(not np.isfinite(value).all() for value in mapping.values()):
        raise ValueError("preds")
    ordered = monotonize_quantiles(mapping)
    keys = sorted(ordered, key=lambda key: float(str(key).split("_", 1)[1]))
    role = pd.to_numeric(
        pd.Series(np.asarray(starting).reshape(-1)), errors="coerce"
    ).to_numpy(dtype=float)
    lag = pd.to_numeric(
        pd.Series(np.asarray(min_lag_1).reshape(-1)), errors="coerce"
    ).to_numpy(dtype=float)
    if role.shape[0] != n or lag.shape[0] != n:
        raise ValueError("length")
    out = {key: ordered[key].copy() for key in keys}
    cold = ~np.isfinite(lag)
    if not cold.any():
        return pd.DataFrame(out, index=index)[keys]
    starter = role == 1
    bench = role == 0
    if np.any(cold & ~(starter | bench)):
        raise ValueError("cold-start role")
    taus = np.array([float(str(key).split("_", 1)[1]) for key in keys])
    if role_quantiles is None and (
        history is None or not {target_col, "starting"}.issubset(history.columns)
    ):
        raise ValueError("history")
    for role_value, role_name, role_mask in (
        (1, "starter", starter),
        (0, "bench", bench),
    ):
        need = cold & role_mask
        if not need.any():
            continue
        if role_quantiles is not None:
            if role_name not in role_quantiles.index:
                raise ValueError(f"no history minutes for role {role_value}")
            quantiles = role_quantiles.loc[role_name, keys].to_numpy(dtype=float)
        else:
            quantiles = _empirical_quantiles(history, role_value, taus, target_col)
        if quantiles.shape != (len(keys),) or not np.isfinite(quantiles).all():
            raise ValueError(f"no history minutes for role {role_value}")
        for key, value in zip(keys, quantiles, strict=True):
            out[key][need] = float(value)
    return pd.DataFrame(out, index=index)[keys]
