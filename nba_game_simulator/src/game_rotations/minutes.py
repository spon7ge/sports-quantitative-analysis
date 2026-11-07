"""Team minute budgets and bounded minute draws."""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import truncnorm

from game_rotations.checks import (
    _as_player_matrix,
    _positive_integer,
    _vector,
)
from game_rotations.constants import _SIGMA_FLOOR, _TOL
from game_rotations.roster import _active_shape

def scale_mu(mu, total, *, availability=None):
    """Normalize nonnegative weights over available players; caps are separate."""
    mu = np.atleast_2d(np.asarray(mu, dtype=float)).copy()

    if mu.size == 0 or not np.isfinite(mu).all() or (mu < 0).any():
        raise ValueError("minute weights must be finite, nonnegative and nonempty")

    totals = _vector(total, mu.shape[0], "total")

    if (totals < 0).any():
        raise ValueError("total must be nonnegative")

    active = _active_shape(availability, mu.shape)
    weights = np.where(active, mu, 0)
    maximum = weights.max(axis=1, keepdims=True)

    weights = np.divide(
        weights,
        maximum,
        out=np.zeros_like(weights),
        where=maximum > 0,
    )

    empty = weights.sum(axis=1) == 0
    weights[empty] = active[empty]
    sums = weights.sum(axis=1)

    if ((sums == 0) & (totals > 0)).any():
        raise ValueError("positive budgets require available players")

    factor = np.divide(
        totals,
        sums,
        out=np.zeros_like(totals),
        where=sums > 0,
    )

    return weights * factor[:, None]


def constrain(raw, w, total, cap, iters=100, *, availability=None):
    """Weighted projection onto a team budget, caps, and fixed zero minutes."""
    raw = np.atleast_2d(np.asarray(raw, dtype=float))

    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("raw minutes must be finite and nonempty")

    _positive_integer(iters, "iters")

    n, _ = raw.shape
    total = _vector(total, n, "total")
    cap = _vector(cap, n, "cap")
    weight = _as_player_matrix(w, raw.shape)

    if not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("projection weights must be finite and nonnegative")

    if (cap < 0).any() or (total < 0).any():
        raise ValueError("cap and budget must be nonnegative")

    active = _active_shape(availability, raw.shape)
    ceiling = cap * active.sum(axis=1)

    if np.any(total > ceiling + _TOL):
        raise ValueError("team minute budget exceeds available player capacity")

    total = np.minimum(total, ceiling)
    maximum = weight.max(axis=1, keepdims=True)

    weight = np.divide(
        weight,
        maximum,
        out=np.ones_like(weight),
        where=maximum > 0,
    )
    weight = np.maximum(weight, 1e-8)

    bound = np.where(active, cap[:, None], 0.0)
    raw = np.where(active, raw, 0.0)
    lo = np.min(-raw / weight, axis=1)
    hi = np.max((bound - raw) / weight, axis=1)

    for _ in range(int(iters)):
        mid = lo / 2 + hi / 2
        projected = np.clip(raw + mid[:, None] * weight, 0, bound)
        over = projected.sum(axis=1) > total
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)

    minutes = np.clip(
        raw + (lo / 2 + hi / 2)[:, None] * weight,
        0,
        bound,
    )

    if np.any(np.abs(minutes.sum(axis=1) - total) > _TOL):
        raise ValueError("projection failed to conserve the team minute budget")

    return minutes


def draw_minutes(mean, sd, cap, rng: np.random.Generator) -> np.ndarray:
    """Sample a stable truncated normal with the requested bounded mean.

    sd is the underlying normal's scale, not the bounded draw's standard
    deviation. Team projection changes individual means and intervals.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")

    mean, sd, cap = np.broadcast_arrays(
        np.asarray(mean, dtype=float),
        np.asarray(sd, dtype=float),
        np.asarray(cap, dtype=float),
    )

    if not all(np.isfinite(value).all() for value in (mean, sd, cap)):
        raise ValueError("mean, sd and cap must be finite")

    if (sd < 0).any() or (cap < 0).any():
        raise ValueError("sd and cap must be nonnegative")

    target = np.clip(mean, 0, cap)
    result = target.copy()
    free = (target > 0) & (target < cap) & (sd > 0)

    if not free.any():
        return result

    pairs = np.column_stack([target[free], sd[free], cap[free]])
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    locations = _location_for_mean(
        unique[:, 0],
        unique[:, 1],
        unique[:, 2],
    )[inverse]

    scale = sd[free]
    result[free] = _bounded_normal_draw(locations, scale, cap[free], rng)

    return np.clip(result, 0, cap)


def sim_gaussian(mean_mat, sd, total, cap, rng, *, availability=None):
    mean_mat = np.atleast_2d(np.asarray(mean_mat, dtype=float))
    total = np.asarray(total, dtype=float).reshape(-1)
    cap = _vector(cap, len(total), "cap")

    if mean_mat.shape[0] == 1:
        mean_mat = np.repeat(mean_mat, len(total), axis=0)

    if mean_mat.shape[0] != len(total):
        raise ValueError("expected minutes and team totals do not align")

    spread = _as_player_matrix(sd, mean_mat.shape)
    raw = draw_minutes(mean_mat, spread, cap[:, None], rng)

    return constrain(
        raw,
        np.maximum(spread, _SIGMA_FLOOR) ** 2,
        total,
        cap,
        availability=availability,
    )

def _location_for_mean(target, scale, upper, iters=60):
    target, scale, upper = np.broadcast_arrays(target, scale, upper)
    result = np.empty(target.shape)

    for i in np.ndindex(target.shape):
        wanted = float(target[i])
        sd = float(scale[i])
        cap = float(upper[i])

        def objective(loc, sd=sd, cap=cap, wanted=wanted):
            return _bounded_normal_mean(loc, sd, cap) - wanted

        step = max(sd, cap, 1.0)
        low, high = -step, cap + step

        for _ in range(100):
            if objective(low) <= 0 and objective(high) >= 0:
                break

            step *= 2
            low, high = -step, cap + step
        else:
            raise ValueError("could not bracket the truncated-normal mean")

        result[i] = brentq(
            objective,
            low,
            high,
            xtol=1e-8,
            maxiter=max(100, iters),
        )

    return result


def _truncated_mean(loc, scale, upper):
    loc, scale, upper = np.broadcast_arrays(loc, scale, upper)

    return np.array(
        [
            _bounded_normal_mean(float(a), float(b), float(c))
            for a, b, c in zip(
                loc.ravel(),
                scale.ravel(),
                upper.ravel(),
                strict=True,
            )
        ]
    ).reshape(loc.shape)


def _bounded_normal_mean(loc, scale, upper):
    if loc > upper + 8 * scale:
        return upper - _bounded_normal_mean(upper - loc, scale, upper)

    if loc >= -8 * scale:
        return float(
            truncnorm.mean(
                -loc / scale,
                (upper - loc) / scale,
                loc=loc,
                scale=scale,
            )
        )

    tail_scale = scale * (scale / -loc)
    limit = min(upper / tail_scale, 60.0)

    if limit < 1:
        def density(y):
            return np.exp(-limit * y - 0.5 * (upper / scale * y) ** 2)

        mass = quad(density, 0, 1, epsabs=1e-12)[0]
        moment = quad(
            lambda y: y * density(y),
            0,
            1,
            epsabs=1e-12,
        )[0]

        return upper * moment / mass

    def density(y):
        return np.exp(-y - 0.5 * (tail_scale / scale * y) ** 2)

    mass = quad(density, 0, limit, epsabs=1e-12)[0]
    moment = quad(
        lambda y: y * density(y),
        0,
        limit,
        epsabs=1e-12,
    )[0]

    return tail_scale * moment / mass


def _bounded_normal_draw(loc, scale, upper, rng):
    result = np.empty(loc.shape)
    left = loc < -8 * scale
    right = loc > upper + 8 * scale
    central = ~(left | right)

    if central.any():
        a = -loc[central] / scale[central]
        b = (upper[central] - loc[central]) / scale[central]

        result[central] = truncnorm.rvs(
            a,
            b,
            loc=loc[central],
            scale=scale[central],
            random_state=rng,
        )

    for mask, reflected in ((left, False), (right, True)):
        if not mask.any():
            continue

        position = np.flatnonzero(mask)
        local_loc = upper[mask] - loc[mask] if reflected else loc[mask]
        local_scale = scale[mask]
        local_upper = upper[mask]
        tail_scale = local_scale * (local_scale / -local_loc)
        values = np.empty(len(position))
        pending = np.arange(len(position))

        while len(pending):
            mass = -np.expm1(
                -local_upper[pending] / tail_scale[pending]
            )
            candidate = -tail_scale[pending] * np.log1p(
                -rng.random(len(pending)) * mass
            )
            accepted = rng.random(len(pending)) <= np.exp(
                -0.5 * (candidate / local_scale[pending]) ** 2
            )

            values[pending[accepted]] = candidate[accepted]
            pending = pending[~accepted]

        result[position] = local_upper - values if reflected else values

    return result
