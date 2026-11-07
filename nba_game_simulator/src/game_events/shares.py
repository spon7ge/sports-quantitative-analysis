"""Event shares, joint assists, and rebound pools."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from game_events.checks import (
    _as_context,
    _as_count_matrix,
    _as_unit_interval_vector,
    _invariant,
)
from game_events.draws import _multinomial_rows, _overdispersed_binomial


def event_shares(
    rates: np.ndarray,
    available: np.ndarray,
    projected_minutes: np.ndarray,
    reference_minutes: np.ndarray,
    role_factors: np.ndarray | None = None,
    lineup_factors: np.ndarray | None = None,
    context_factors: np.ndarray | None = None,
    minutes_exponent: float = 1.0,
    allow_empty: bool = False,
) -> np.ndarray:
    """Normalize event weights through the external ``opportunity_shares``.

    ``rates`` are the pre-minutes baseline. ``context_factors`` is the
    event-specific context: opponent shot profile for rebounds, or matchup
    context for steals and blocks. It multiplies lineup composition; it does
    not replace it. Only ``simulate_shared_events`` uses this helper.
    """
    from opportunity_shares import opportunity_share_rows, opportunity_shares

    lineup = lineup_factors
    if context_factors is not None:
        context = _as_context(context_factors, len(np.asarray(rates)))
        if lineup_factors is None:
            lineup = context
        else:
            # Validate lineup before multiplying. A negative lineup and a
            # negative context would otherwise cancel and look legal.
            try:
                lineup_values = np.asarray(lineup_factors, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "lineup_factors must contain nonnegative finite values"
                ) from exc
            if (
                lineup_values.shape != context.shape
                or np.any(~np.isfinite(lineup_values))
                or np.any(lineup_values < 0.0)
            ):
                raise ValueError(
                    "lineup_factors must contain nonnegative finite values"
                )
            lineup = lineup_values * context
    projected = np.asarray(projected_minutes, dtype=float)
    share = opportunity_share_rows if projected.ndim == 2 else opportunity_shares
    return share(
        base_weights=rates,
        available=available,
        projected_minutes=projected_minutes,
        reference_minutes=reference_minutes,
        role_factors=role_factors,
        lineup_factors=lineup,
        minutes_exponent=minutes_exponent,
        allow_empty=allow_empty,
    )


def allocate_joint_assists(
    fgm: np.ndarray,
    assisted_rate: np.ndarray,
    creation_weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Credit each assisted make to a teammate of the scorer.

    ``assisted_rate`` is ``(n_players,)`` or per simulation. Creation weights
    are ``(n_players,)``, ``(n_sims, n_players)``, or
    ``(n_sims, scorer, assister)`` when eligibility depends on the scorer,
    such as teammates who shared the floor.

    Returns ``(assists, pairs)`` where ``pairs[s, i, j]`` is the number of
    sim-``s`` baskets scored by ``j`` and assisted by ``i``. The scorer's
    own weight is removed before the draw, so the diagonal is zero. A make
    with no eligible teammate stays unassisted.
    """
    made = np.asarray(fgm)
    if made.ndim != 2:
        raise ValueError("FGM must be a two-dimensional count matrix")
    n_sims, n_players = made.shape
    made = _as_count_matrix(fgm, n_sims, n_players, "FGM")
    rate = np.asarray(assisted_rate, dtype=float)
    if rate.shape == (n_sims, n_players):
        if np.any(~np.isfinite(rate)) or np.any((rate < 0.0) | (rate > 1.0)):
            raise ValueError("assisted_rate must contain values in [0, 1]")
    else:
        rate = np.broadcast_to(
            _as_unit_interval_vector(rate, n_players, "assisted_rate"),
            (n_sims, n_players),
        )
    creation = np.asarray(creation_weights, dtype=float)
    if (
        creation.shape
        not in ((n_players,), (n_sims, n_players), (n_sims, n_players, n_players))
        or np.any(~np.isfinite(creation))
        or np.any(creation < 0.0)
    ):
        raise ValueError("creation_weights must contain nonnegative finite values")
    if creation.ndim < 3:
        creation = np.broadcast_to(creation, (n_sims, n_players))[:, None, :]
        creation = np.broadcast_to(creation, (n_sims, n_players, n_players))

    pairs = np.zeros((n_sims, n_players, n_players), dtype=np.int64)
    for scorer in range(n_players):
        eligible = creation[:, scorer, :].copy()
        eligible[:, scorer] = 0.0
        has_teammate = eligible.sum(axis=1) > 0.0
        assisted = rng.binomial(made[:, scorer], rate[:, scorer])
        assisted = np.where(has_teammate, assisted, 0).astype(np.int64)
        if np.any(assisted > 0):
            pairs[:, :, scorer] = _multinomial_rows(assisted, eligible, rng, "AST")

    assists = pairs.sum(axis=2)
    diagonal = pairs[:, np.arange(n_players), np.arange(n_players)]
    _invariant(bool(np.all(diagonal == 0)), "a player assisted their own field goal")
    _invariant(
        bool(np.all(pairs.sum(axis=1) <= made)),
        "assists exceed the scorer's made field goals",
    )
    return assists, pairs


def _rebound_pools(
    misses: np.ndarray,
    upstream_oreb: np.ndarray | None,
    p_rebound: float,
    p_oreb: float,
    rng: np.random.Generator,
    dispersion: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split missed field goals into offensive, defensive, and dead balls.

    Each miss is offensive with probability ``p_rebound * p_oreb``,
    defensive with ``p_rebound * (1 - p_oreb)``, and dead otherwise. When
    Layer 4 already fixed the offensive count, defensive rebounds are drawn
    from the remaining misses with the matching conditional probability.
    """
    dispersion = dispersion or {}
    p_off = p_rebound * p_oreb
    p_def = p_rebound * (1.0 - p_oreb)
    if upstream_oreb is None:
        live = _overdispersed_binomial(
            misses, p_rebound, dispersion.get("p_rebound", 0.0), rng
        )
        offensive = _overdispersed_binomial(
            live, p_oreb, dispersion.get("p_oreb", 0.0), rng
        )
        defensive = live - offensive
    else:
        offensive = np.asarray(upstream_oreb, dtype=np.int64)
        remaining_p = p_def / (1.0 - p_off) if p_off < 1.0 else 0.0
        defensive = _overdispersed_binomial(
            misses - offensive,
            remaining_p,
            dispersion.get("dreb_given_oreb", 0.0),
            rng,
        )
    _invariant(
        bool(np.all(offensive + defensive <= misses)),
        "rebounds exceed missed shots",
    )
    return offensive, defensive
