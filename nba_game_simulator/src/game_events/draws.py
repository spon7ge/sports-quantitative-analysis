"""Independent random streams and overdispersed allocations."""

from __future__ import annotations

from typing import Any

import numpy as np
from game_events.checks import _invariant


def _streams(rng: np.random.Generator, keys: list) -> dict:
    """One independent generator per key, in a fixed key order.

    Changing whether one stat is drawn does not shift any other stat's draws.
    """
    root = np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
    return {
        key: np.random.default_rng(child)
        for key, child in zip(keys, root.spawn(len(keys)))
    }


def _overdispersed_binomial(
    n: np.ndarray, p: Any, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Binomial, or beta-binomial with intra-pool correlation ``rho``."""
    n = np.asarray(n, dtype=np.int64)
    p = np.broadcast_to(np.asarray(p, dtype=float), n.shape).copy()
    if rho > 0.0:
        strength = (1.0 - rho) / rho
        inner = (p > 0.0) & (p < 1.0)
        p[inner] = rng.beta(p[inner] * strength, (1.0 - p[inner]) * strength)
    return rng.binomial(n, p).astype(np.int64)


def _multinomial_rows(
    counts: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    stat: str,
    concentration: float | None = None,
) -> np.ndarray:
    """Draw ``counts[s]`` events over the row-``s`` weights.

    A finite ``concentration`` draws each row's shares from a Dirichlet
    around the weights first, which widens player totals.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = weights.sum(axis=1)
    usable = total > 0.0
    blocked = (counts > 0) & ~usable
    if np.any(blocked):
        raise ValueError(
            f"No eligible players to allocate {stat} "
            f"in {int(blocked.sum())} simulations"
        )
    out = np.zeros(weights.shape, dtype=np.int64)
    rows = np.flatnonzero(counts > 0)
    if rows.size == 0:
        return out
    probs = weights[rows] / total[rows, None]
    if concentration is not None:
        gamma = rng.standard_gamma(concentration * probs)
        gamma_total = gamma.sum(axis=1)
        drawn = gamma_total > 0.0
        probs[drawn] = gamma[drawn] / gamma_total[drawn, None]
    out[rows] = rng.multinomial(counts[rows], probs).astype(np.int64)
    return out


def _allocate(
    pool: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    stat: str,
    concentration: float | None = None,
) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    allocated = _multinomial_rows(pool, weights, rng, stat, concentration)
    _invariant(
        bool(np.array_equal(allocated.sum(axis=1), pool)),
        f"{stat} allocation does not match its team pool",
    )
    return allocated


def _log5(p_for: Any, p_against: Any, league: float) -> Any:
    """Combine a team's rate with its opponent's allowed rate.

    Equal rates at the league average return the league average. A league
    average opponent returns the team's own rate. A league rate of 0 or 1
    has no odds, so the team's own rate is returned unchanged.
    """
    eps = 1e-9
    p_for = np.asarray(p_for, dtype=float)
    if not eps < league < 1.0 - eps:
        return p_for if p_for.ndim else float(p_for)
    a = np.clip(p_for, eps, 1.0 - eps)
    b = np.clip(np.asarray(p_against, dtype=float), eps, 1.0 - eps)
    odds = (a / (1.0 - a)) * (b / (1.0 - b)) / (league / (1.0 - league))
    out = odds / (1.0 + odds)
    return out if out.ndim else float(out)
