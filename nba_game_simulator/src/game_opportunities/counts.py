"""Count draws, multinomial pools, and score-conditioned makes."""

from __future__ import annotations

import numpy as np
from game_opportunities.checks import (
    _integer_matrix,
    _normalize_rows,
    _number,
    _player_vector,
    _rng,
    as_nonneg_int_vector,
)
from game_opportunities.constants import _MAX_POOL
from scipy.special import expit, logit
from scipy.stats import binom


def sample_negbin(mu, alpha, rng, size=None):
    """NB2 counts: variance = mean + alpha * mean**2."""
    _rng(rng)
    alpha = _number(alpha, "alpha", nonnegative=True)
    mean = np.asarray(mu, dtype=float)

    if not np.isfinite(mean).all() or (mean < 0).any() or (mean > _MAX_POOL).any():
        raise ValueError("count means must be finite and in [0, 10000]")

    if size is not None:
        mean = np.broadcast_to(mean, size)

    if alpha <= 1e-12:
        result = rng.poisson(mean)
    else:
        intensity = np.zeros(mean.shape)
        positive = mean > 0

        intensity[positive] = rng.gamma(
            1 / alpha,
            alpha * mean[positive],
        )

        if not np.isfinite(intensity).all() or (intensity > 1e8).any():
            raise ValueError("count dispersion produced an excessive intensity")

        result = rng.poisson(intensity)

    return np.asarray(result, dtype=np.int64)


def multinomial_rows(totals, probs, rng):
    _rng(rng)

    raw = np.asarray(totals)

    if raw.ndim != 1 or not len(raw):
        raise ValueError("totals must be a nonempty vector")

    totals = as_nonneg_int_vector(
        raw,
        len(raw),
        "totals",
    )
    weights = np.asarray(probs, dtype=float)

    if weights.ndim == 1:
        weights = np.broadcast_to(
            weights,
            (len(totals), len(weights)),
        )

    if weights.ndim != 2 or weights.shape[0] != len(totals) or not weights.shape[1]:
        raise ValueError("probs must be (players,) or (simulations, players)")

    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("probs must be finite and nonnegative")

    sums = weights.sum(axis=1)
    unit_sum = np.isclose(
        sums,
        1,
        rtol=0,
        atol=1e-8,
    )

    if np.any((totals > 0) & ~unit_sum):
        raise ValueError("positive pools need probabilities summing to one")

    if np.any((sums != 0) & ~unit_sum):
        raise ValueError("probabilities must sum to zero or one")

    result = np.zeros(
        weights.shape,
        dtype=np.int64,
    )

    for i in np.flatnonzero(totals):
        result[i] = rng.multinomial(
            int(totals[i]),
            weights[i] / sums[i],
        )

    return result


def renormalize_props(weights):
    values = np.asarray(weights, dtype=float)

    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be a finite nonnegative vector")

    if values.sum() <= 0:
        raise ValueError("weights must sum to a positive value")

    return values / values.sum()


def allocate_and_shoot(
    team_fga,
    shares,
    p_3pa,
    p_3p,
    p_2p,
    rng,
):
    fga = multinomial_rows(team_fga, shares, rng)
    k = fga.shape[1]

    split = _player_vector(
        p_3pa,
        k,
        "p_3pa",
        probability=True,
    )
    p3 = _player_vector(
        p_3p,
        k,
        "p_3p",
        probability=True,
    )
    p2 = _player_vector(
        p_2p,
        k,
        "p_2p",
        probability=True,
    )

    three_pa = rng.binomial(fga, split[None, :])
    two_pa = fga - three_pa
    three_pm = rng.binomial(three_pa, p3[None, :])
    two_pm = rng.binomial(two_pa, p2[None, :])

    return {
        "FGA": fga,
        "2PA": two_pa,
        "2PM": two_pm,
        "3PA": three_pa,
        "3PM": three_pm,
        "FGM": two_pm + three_pm,
    }


def apply_zone_shots(fga, zone_share, zone_make, rng):
    """Split each player's attempts into rim, mid, corner, and above the break.

    ``zone_share`` and ``zone_make`` are ``(players, 4)`` in that zone order.
    Twos are rim plus mid. Threes are the two zones beyond the arc.
    """

    _rng(rng)
    attempts = np.asarray(fga, dtype=np.int64)
    if attempts.ndim != 2:
        raise ValueError("fga must be (rows, players)")
    share = np.asarray(zone_share, dtype=float)
    make = np.asarray(zone_make, dtype=float)
    rows, players = attempts.shape
    if share.shape != (players, 4) or make.shape != (players, 4):
        raise ValueError("zone rates must have shape (players, 4)")
    if not np.isfinite(share).all() or (share < 0).any():
        raise ValueError("zone shares must be finite and nonnegative")
    if not np.isfinite(make).all() or (make < 0).any() or (make > 1).any():
        raise ValueError("zone make rates must be probabilities")

    zones = np.zeros((rows, players, 4), dtype=np.int64)
    for player in range(players):
        total = float(share[player].sum())
        probs = share[player] / total if total > 0 else np.array([1.0, 0.0, 0.0, 0.0])
        for row in np.flatnonzero(attempts[:, player]):
            zones[row, player] = rng.multinomial(int(attempts[row, player]), probs)

    made = np.zeros_like(zones)
    for zone in range(4):
        made[:, :, zone] = rng.binomial(zones[:, :, zone], make[None, :, zone])

    two_pa = zones[:, :, 0] + zones[:, :, 1]
    two_pm = made[:, :, 0] + made[:, :, 1]
    three_pa = zones[:, :, 2] + zones[:, :, 3]
    three_pm = made[:, :, 2] + made[:, :, 3]
    return two_pa, two_pm, three_pa, three_pm, zones, made


def zones_from_point_value(two_pa, two_pm, three_pa, three_pm):
    """Put twos at the rim and threes above the break.

    The four-zone mix is unknown on this path. A two is the short-range
    attempt, and a three is not. Rim is the only short zone the model records.
    """

    two_pa = np.asarray(two_pa, dtype=np.int64)
    attempts = np.zeros((*two_pa.shape, 4), dtype=np.int64)
    makes = np.zeros_like(attempts)
    attempts[..., 0] = two_pa
    makes[..., 0] = np.asarray(two_pm, dtype=np.int64)
    attempts[..., 3] = np.asarray(three_pa, dtype=np.int64)
    makes[..., 3] = np.asarray(three_pm, dtype=np.int64)
    return attempts, makes


def align_zone_makes(zone_fga, two_pm, three_pm, rng):
    """Place conditioned twos and threes back onto the zone attempts.

    Attempts stay fixed. The two makes are split across rim and mid, and the
    three makes across the two zones beyond the arc.
    """

    _rng(rng)
    zone_fga = np.asarray(zone_fga, dtype=np.int64)
    two_pm = np.asarray(two_pm, dtype=np.int64)
    three_pm = np.asarray(three_pm, dtype=np.int64)
    made = np.zeros_like(zone_fga)
    rows, players, _zones = zone_fga.shape
    for row in range(rows):
        for player in range(players):
            made[row, player, 0] = _share_successes(
                int(zone_fga[row, player, 0]),
                int(zone_fga[row, player, 1]),
                int(two_pm[row, player]),
                rng,
            )
            made[row, player, 1] = int(two_pm[row, player]) - int(made[row, player, 0])
            made[row, player, 2] = _share_successes(
                int(zone_fga[row, player, 2]),
                int(zone_fga[row, player, 3]),
                int(three_pm[row, player]),
                rng,
            )
            made[row, player, 3] = int(three_pm[row, player]) - int(
                made[row, player, 2]
            )
    return made


def _share_successes(first, second, successes, rng):
    total = first + second
    if successes < 0 or successes > total:
        raise ValueError("makes exceed zone attempts")
    if first == 0 or successes == 0:
        return 0
    if second == 0 or successes == total:
        return min(successes, first)
    return int(rng.hypergeometric(first, second, successes))


def split_turnovers(turnovers, live_share, rng):
    """Split each player's turnovers into live-ball and dead-ball."""

    _rng(rng)
    turnovers = np.asarray(turnovers, dtype=np.int64)
    share = np.asarray(live_share, dtype=float)
    if np.any(~np.isfinite(share)) or np.any((share < 0.0) | (share > 1.0)):
        raise ValueError("live-ball turnover share must be in [0, 1]")
    live = rng.binomial(turnovers, share).astype(np.int64)
    return live, turnovers - live


def simulate_fta_ftm(
    fga,
    fta_per_fga,
    ft_pct,
    alpha_fta,
    rng,
):
    """Legacy FGA-based helper.

    Main APIs use independent free-throw pools.
    """
    fga = _integer_matrix(fga, "fga")

    ratio = _player_vector(
        fta_per_fga,
        fga.shape[1],
        "fta_per_fga",
    )
    pct = _player_vector(
        ft_pct,
        fga.shape[1],
        "ft_pct",
        probability=True,
    )

    fta = sample_negbin(
        fga * ratio[None, :],
        alpha_fta,
        rng,
    )

    return fta, rng.binomial(fta, pct[None, :])


def allocate_from_pool(pool_totals, alloc_probs, rng):
    return multinomial_rows(
        pool_totals,
        alloc_probs,
        rng,
    )


def _beta_binomial(
    trials,
    probability,
    rho,
    rng,
):
    rho = _number(
        rho,
        "bounded-count dispersion",
        nonnegative=True,
    )

    if rho >= 1:
        raise ValueError("bounded-count dispersion must be below one")

    n = np.asarray(trials, dtype=np.int64)
    p = np.broadcast_to(
        np.asarray(probability, dtype=float),
        n.shape,
    ).copy()

    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("bounded-count probability must be in [0, 1]")

    free = (p > 0) & (p < 1)

    if rho > 1e-12 and free.any():
        concentration = 1 / rho - 1

        p[free] = rng.beta(
            p[free] * concentration,
            (1 - p[free]) * concentration,
        )

    return rng.binomial(n, p)


def _conditional_makes(
    shot,
    fta,
    p2,
    p3,
    pft,
    target,
    rng,
):
    """Binomial makes conditional on fixed attempts and team points."""
    result = dict(shot)
    two_pm = np.zeros_like(fta)
    three_pm = np.zeros_like(fta)
    ftm = np.zeros_like(fta)

    for row in range(len(target)):
        groups = []
        forced = 0

        for output, attempts, probability, points in (
            (two_pm, shot["2PA"], p2, 2),
            (three_pm, shot["3PA"], p3, 3),
            (ftm, fta, pft, 1),
        ):
            for player in range(fta.shape[1]):
                count = int(attempts[row, player])
                p = float(probability[player])

                if p == 1:
                    output[row, player] = count
                    forced += count * points
                elif count and p > 0:
                    groups.append((output, player, count, p, points))

        remainder = int(target[row]) - forced
        maximum = sum(count * points for _, _, count, _, points in groups)

        if remainder < 0 or remainder > maximum:
            raise ValueError(
                f"score in row {row} is infeasible for "
                "the supplied attempts and probabilities"
            )

        if remainder == 0:
            continue

        if remainder == maximum:
            for output, player, count, _, _ in groups:
                output[row, player] = count
            continue

        count = np.array([g[2] for g in groups])
        probability = np.array([g[3] for g in groups])
        points = np.array([g[4] for g in groups])
        logits = logit(probability)

        low, high = -1.0, 1.0

        while np.sum(count * points * expit(logits + low * points)) > remainder:
            low *= 2

        while np.sum(count * points * expit(logits + high * points)) < remainder:
            high *= 2

        for _ in range(80):
            middle = (low + high) / 2
            expected = np.sum(count * points * expit(logits + middle * points))

            if expected > remainder:
                high = middle
            else:
                low = middle

        tilted = expit(logits + ((low + high) / 2) * points)

        prefix = [np.array([1.0])]
        probabilities = []

        for group, p in zip(
            groups,
            tilted,
            strict=True,
        ):
            trials, score = group[2], group[4]
            successes = np.arange(min(trials, remainder // score) + 1)
            pmf = binom.pmf(successes, trials, p)

            polynomial = np.zeros(int(successes[-1] * score) + 1)
            polynomial[successes * score] = pmf

            accumulated = np.convolve(
                prefix[-1],
                polynomial,
            )[: remainder + 1]

            mass = accumulated.sum()

            if not np.isfinite(mass) or mass <= 0:
                raise ValueError(
                    f"score in row {row} has zero or numerically unresolved probability"
                )

            prefix.append(accumulated / mass)
            probabilities.append(pmf)

        if remainder >= len(prefix[-1]) or prefix[-1][remainder] <= 0:
            raise ValueError(f"score in row {row} is unreachable for these shot types")

        remaining = remainder

        for i in range(len(groups) - 1, -1, -1):
            output, player, trials, _, score = groups[i]
            successes = np.arange(min(trials, remaining // score) + 1)
            indices = remaining - score * successes
            weights = np.zeros(len(successes))
            valid = indices < len(prefix[i])

            weights[valid] = (
                prefix[i][indices[valid]] * probabilities[i][successes[valid]]
            )

            if weights.sum() <= 0:
                raise ValueError("conditional score sampling failed")

            made = int(
                rng.choice(
                    successes,
                    p=weights / weights.sum(),
                )
            )
            output[row, player] = made
            remaining -= made * score

        if remaining:
            raise ValueError("conditional score sampling left unmatched points")

    result["2PM"] = two_pm
    result["3PM"] = three_pm
    result["FGM"] = two_pm + three_pm

    return result, ftm


def _shoot_pools(
    team_fga,
    team_fta,
    shares,
    ft_shares,
    p3a,
    p3,
    p2,
    pft,
    rng,
    target=None,
    zone_share=None,
    zone_make=None,
):
    shot = allocate_and_shoot(
        team_fga,
        shares,
        p3a,
        p3,
        p2,
        rng,
    )
    if zone_share is not None:
        two_pa, two_pm, three_pa, three_pm, zone_fga, zone_fgm = apply_zone_shots(
            shot["FGA"],
            zone_share,
            zone_make,
            rng,
        )
        shot["2PA"] = two_pa
        shot["2PM"] = two_pm
        shot["3PA"] = three_pa
        shot["3PM"] = three_pm
        shot["FGM"] = two_pm + three_pm
    else:
        zone_fga, zone_fgm = zones_from_point_value(
            shot["2PA"],
            shot["2PM"],
            shot["3PA"],
            shot["3PM"],
        )
    fta = multinomial_rows(team_fta, ft_shares, rng)

    if target is None:
        ftm = rng.binomial(fta, pft[None, :])
    else:
        shot, ftm = _conditional_makes(
            shot,
            fta,
            p2,
            p3,
            pft,
            target,
            rng,
        )
        zone_fgm = align_zone_makes(zone_fga, shot["2PM"], shot["3PM"], rng)

    return {
        **shot,
        "FTA": fta,
        "FTM": ftm,
        "PTS": 2 * shot["2PM"] + 3 * shot["3PM"] + ftm,
        "ZONE_FGA": zone_fga,
        "ZONE_FGM": zone_fgm,
    }


def _assists(
    fgm,
    weights,
    probability,
    rho,
    rng,
):
    n, k = fgm.shape
    assist_probability = _beta_probability(
        n,
        probability,
        rho,
        rng,
    )
    result = np.zeros_like(fgm)

    for shooter in range(k):
        eligible = weights.copy()
        eligible[:, shooter] = 0

        probabilities = _normalize_rows(eligible)
        totals = rng.binomial(
            fgm[:, shooter],
            assist_probability,
        )

        totals[eligible.sum(axis=1) == 0] = 0

        result += multinomial_rows(
            totals,
            probabilities,
            rng,
        )

    return result


def _beta_probability(n, p, rho, rng):
    p = _number(
        p,
        "assist_rate",
        nonnegative=True,
    )
    rho = _number(
        rho,
        "assist dispersion",
        nonnegative=True,
    )

    if p > 1 or rho >= 1:
        raise ValueError("assist probability must be <= 1 and dispersion < 1")

    if rho <= 1e-12 or p in (0, 1):
        return np.full(n, p)

    concentration = 1 / rho - 1

    return rng.beta(
        p * concentration,
        (1 - p) * concentration,
        size=n,
    )
