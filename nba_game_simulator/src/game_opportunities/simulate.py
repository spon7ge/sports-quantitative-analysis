"""Allocate caller-supplied team pools across a roster."""

from __future__ import annotations

import numpy as np
from game_opportunities.allocate import _pool, _round_possessions, _whole_score
from game_opportunities.checks import (
    _availability,
    _check_minutes,
    _key,
    _matrix,
    _normalize_rows,
    _number,
    _player_vector,
    _positive_int,
    _rng,
)
from game_opportunities.counts import (
    _assists,
    _beta_binomial,
    _shoot_pools,
    multinomial_rows,
    sample_negbin,
)
from game_opportunities.model import SimulationResult, TeamSimResult


def simulate_game(
    team_inputs,
    player_inputs,
    n_sims,
    rng,
):
    """Allocate supplied team pools.

    AST/OREB use approximate box-score models.

    Dispersion fta is NB2 alpha.
    Dispersion ast/oreb/tov is beta-binomial correlation in [0, 1).
    Player assist and rebound propensities are rates per minute.
    """
    _positive_int(n_sims, "n_sims")
    _rng(rng)

    if not isinstance(team_inputs, dict) or not team_inputs:
        raise ValueError("team_inputs must be a nonempty dictionary")

    if set(team_inputs) != set(player_inputs):
        raise ValueError("team_inputs and player_inputs must describe the same teams")

    seen = set()
    result = {}

    for name, team in team_inputs.items():
        players = player_inputs[name]
        ids = tuple(players["player_ids"])
        keys = [_key(value) for value in ids]

        if not keys or len(set(keys)) != len(keys) or seen.intersection(keys):
            raise ValueError("player IDs must be unique within and across rosters")

        seen.update(keys)
        k = len(ids)

        minutes = _matrix(
            players["projected_minutes"],
            n_sims,
            k,
            "projected_minutes",
        )
        available = _availability(
            players.get("available"),
            n_sims,
            k,
        )

        _check_minutes(
            minutes,
            available,
            team.get("game_minutes"),
            strict_duration=False,
        )

        eligible = available & (minutes > 0)

        base = _player_vector(
            players["base_weights"],
            k,
            "base_weights",
        )
        reference = _player_vector(
            players["reference_minutes"],
            k,
            "reference_minutes",
        )

        if (reference <= 0).any():
            raise ValueError("reference_minutes must be positive")

        exponent = _number(
            players.get("minutes_exponent", 1),
            "minutes_exponent",
            nonnegative=True,
        )

        shot_weights = base[None, :] * (minutes / reference[None, :]) ** exponent

        for field_name in ("role_factors", "lineup_factors"):
            if players.get(field_name) is not None:
                shot_weights *= _player_vector(
                    players[field_name],
                    k,
                    field_name,
                )[None, :]

        shot_weights = np.where(
            eligible,
            shot_weights,
            0,
        )
        shares = _normalize_rows(shot_weights)

        p3a = _player_vector(
            players["p_3pa"],
            k,
            "p_3pa",
            probability=True,
        )
        p3 = _player_vector(
            players["p_3p"],
            k,
            "p_3p",
            probability=True,
        )
        p2 = _player_vector(
            players["p_2p"],
            k,
            "p_2p",
            probability=True,
        )
        pft = _player_vector(
            players["ft_pct"],
            k,
            "ft_pct",
            probability=True,
        )

        dispersion = players.get("dispersion", {})
        alpha = _number(
            dispersion.get("fta", 0.05),
            "FTA dispersion",
            nonnegative=True,
        )

        fga = _pool(
            team["team_fga"],
            n_sims,
            "team_fga",
        )
        ratio = _player_vector(
            players["fta_per_fga"],
            k,
            "fta_per_fga",
        )
        fta_rate = players.get("fta_per_min")

        if fta_rate is None:
            ft_weights = np.where(
                eligible,
                shot_weights * ratio[None, :],
                0,
            )
        else:
            ft_weights = (
                minutes
                * _player_vector(
                    fta_rate,
                    k,
                    "fta_per_min",
                )[None, :]
            )

        ft_shares = _normalize_rows(ft_weights)

        if "team_fta" in team:
            fta = _pool(
                team["team_fta"],
                n_sims,
                "team_fta",
            )
        elif fta_rate is not None:
            fta = sample_negbin(
                ft_weights.sum(axis=1),
                alpha,
                rng,
            )
        else:
            fta = sample_negbin(
                fga
                * np.sum(
                    shares * ratio[None, :],
                    axis=1,
                ),
                alpha,
                rng,
            )

        target = (
            _whole_score(team["team_points"], n_sims) if "team_points" in team else None
        )

        stats = _shoot_pools(
            fga,
            fta,
            shares,
            ft_shares,
            p3a,
            p3,
            p2,
            pft,
            rng,
            target,
        )

        ast_weights = (
            minutes
            * _player_vector(
                players["assist_propensity"],
                k,
                "assist_propensity",
            )[None, :]
        )

        stats["AST"] = _assists(
            stats["FGM"],
            ast_weights,
            players.get("assist_rate", 0.55),
            dispersion.get("ast", 0.05),
            rng,
        )

        misses = fga - stats["FGM"].sum(axis=1)

        oreb_pool = _beta_binomial(
            misses,
            players.get("oreb_rate", 0.25),
            dispersion.get("oreb", 0.05),
            rng,
        )
        oreb_weights = (
            minutes
            * _player_vector(
                players["oreb_propensity"],
                k,
                "oreb_propensity",
            )[None, :]
        )

        stats["OREB"] = multinomial_rows(
            oreb_pool,
            _normalize_rows(oreb_weights),
            rng,
        )

        tov_rate = _player_vector(
            players["tov_rate"],
            k,
            "tov_rate",
            probability=True,
        )
        tov_weights = shares * tov_rate[None, :]
        tov_probs = _normalize_rows(tov_weights)

        if "team_tov" in team:
            turnovers = _pool(
                team["team_tov"],
                n_sims,
                "team_tov",
            )
        else:
            possessions = _round_possessions(
                team["possessions"],
                n_sims,
            )
            turnovers = _beta_binomial(
                possessions,
                np.sum(
                    shares * tov_rate[None, :],
                    axis=1,
                ),
                dispersion.get("tov", 0.05),
                rng,
            )

        stats["TOV"] = multinomial_rows(
            turnovers,
            tov_probs,
            rng,
        )

        result[name] = TeamSimResult(
            player_ids=ids,
            opportunity_shares=shares,
            stats=stats,
            metadata={
                "score_conditioned": target is not None,
                "oreb_uses_field_goal_misses_only": True,
                "assists_use_minute_overlap_proxy": True,
                "opportunity_pools_approximate": True,
            },
        )

    return SimulationResult(teams=result)
