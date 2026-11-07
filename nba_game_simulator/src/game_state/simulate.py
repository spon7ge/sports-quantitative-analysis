"""Shared simulations and team minute budgets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from game_state.constants import BLOWOUT_MARGIN, PRE, SCRIPT_EDGES, STATE
from game_state.train import role_index


def allocate_minutes(raw, active, duration):
    raw = np.asarray(raw, float)
    active = np.asarray(active, bool)
    duration = np.asarray(duration, float)

    if (
        raw.shape != active.shape
        or not np.isfinite(raw).all()
        or not np.isfinite(duration).all()
    ):
        raise ValueError(
            "Minute weights/masks must align and be finite"
        )
    if (active.sum(axis=1) < 5).any():
        raise ValueError(
            "Every simulation needs at least five available players"
        )

    weights = np.maximum(raw, 0)
    free = active.copy()
    result = np.zeros_like(raw)
    remaining = 5 * duration.copy()

    for _ in range(raw.shape[1] + 1):
        w = weights * free
        empty = w.sum(axis=1) <= 1e-12
        w[empty] = free[empty]

        proposal = (
            w / np.maximum(
                w.sum(axis=1, keepdims=True), 1e-12
            )
            * remaining[:, None]
        )
        hit = free & (proposal > duration[:, None])
        done = ~hit.any(axis=1)

        result[done] += proposal[done]
        remaining[done] = 0
        free[done] = False

        result += hit * duration[:, None]
        remaining -= (
            hit * duration[:, None]
        ).sum(axis=1)
        free &= ~hit

        if not free.any():
            break

    if not np.allclose(
        result.sum(axis=1), 5 * duration, atol=1e-6
    ):
        raise ValueError("Could not satisfy the team minute budget")
    return result


def simulate_minutes(
    bundle,
    home_players,
    away_players,
    game_env,
    availability=None,
    seed=0,
):
    """Return roster-wide minutes using existing Layer 1 rows.

    availability:
        {
            "home": bool[n_simulations, n_home_players],
            "away": bool[n_simulations, n_away_players],
        }

    Pass the same masks used for Layer 1 availability scenarios.
    Player columns follow the corresponding roster's row order.
    """
    needed = {
        "team_margin", "game_minutes", "n_ot", "is_home"
    }
    if not needed.issubset(game_env):
        raise ValueError(
            f"Layer 1 environments need {sorted(needed)}"
        )
    if set(home_players.personId) & set(away_players.personId):
        raise ValueError("A player cannot belong to both rosters")

    if (
        len(game_env) == 0
        or not np.isfinite(
            game_env[list(needed)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
        ).all()
    ):
        raise ValueError("Empty or non-numeric Layer 1 environments")

    if "game_date" in game_env:
        dates = pd.to_datetime(
            game_env.game_date, errors="coerce"
        )
        if (
            dates.isna().any()
            or dates.le(pd.Timestamp(bundle["cutoff"])).any()
        ):
            raise ValueError(
                "Environment dates must be after this model's "
                "training cutoff"
            )

    duration = game_env.game_minutes.to_numpy(float)
    n_ot = game_env.n_ot.to_numpy(float)
    orientation = game_env.is_home.to_numpy(float)

    if (
        not np.isin(orientation, [0, 1]).all()
        or not np.equal(n_ot, np.floor(n_ot)).all()
        or (n_ot < 0).any()
    ):
        raise ValueError("Invalid orientation or overtime count")
    if not np.allclose(duration, 48 + 5 * n_ot):
        raise ValueError("Layer 1 duration and overtime disagree")

    home_margin = (
        game_env.team_margin.to_numpy(float)
        * np.where(orientation == 1, 1, -1)
    )

    rng = np.random.default_rng(seed)
    bucket = np.searchsorted(
        SCRIPT_EDGES, home_margin, side="right"
    )
    script = np.empty((len(game_env), 2))

    for i, bank in enumerate(bundle["script"]):
        mask = bucket == i
        script[mask] = bank[
            rng.integers(len(bank), size=mask.sum())
        ]

    output = {
        "script": pd.DataFrame(
            script,
            columns=[
                "home_late_lead_min",
                "home_late_trail_min",
            ],
            index=game_env.index,
        ),
        "p_blowout": float(np.mean(
            np.abs(home_margin) >= BLOWOUT_MARGIN
        )),
    }

    for side, players, sign in [
        ("home", home_players, 1),
        ("away", away_players, -1),
    ]:
        players = players.reset_index(drop=True)
        if (
            players.personId.duplicated().any()
            or not set(PRE).issubset(players)
        ):
            raise ValueError(
                "Player rows must have unique IDs "
                "and all pregame features"
            )
        if not players.is_home.eq(side == "home").all():
            raise ValueError(
                "Pregame player orientation does not match roster side"
            )

        n, p = len(game_env), len(players)
        x = pd.DataFrame(
            np.tile(players[PRE].to_numpy(float), (n, 1)),
            columns=PRE,
        )
        x["team_final_margin"] = np.repeat(
            sign * home_margin, p
        )
        x["abs_final_margin"] = np.repeat(
            np.abs(home_margin), p
        )
        x["game_minutes"] = np.repeat(duration, p)
        x["late_lead_min"] = np.repeat(
            script[:, 0 if side == "home" else 1], p
        )
        x["late_trail_min"] = np.repeat(
            script[:, 1 if side == "home" else 0], p
        )

        raw = bundle["model"].predict(
            x[PRE + STATE]
        ).reshape(n, p)

        for j, role in enumerate(role_index(players.roll_min)):
            bank = bundle["residuals"][role]
            raw[:, j] += rng.choice(bank, size=n)

        if availability is not None:
            active = np.asarray(availability[side])
            if (
                active.shape != (n, p)
                or not np.isin(active, [0, 1]).all()
            ):
                raise ValueError(
                    "Availability must match environment rows "
                    "and roster columns"
                )
            active = active.astype(bool)
        else:
            flags = (
                players["available"].to_numpy()
                if "available" in players
                else np.ones(p, bool)
            )
            if not np.isin(flags, [0, 1]).all():
                raise ValueError(
                    "Available flags must be boolean or 0/1"
                )
            active = np.broadcast_to(flags, (n, p))

        result = allocate_minutes(raw, active, duration)
        output[side] = pd.DataFrame(
            result,
            columns=players.personId.to_list(),
            index=game_env.index,
        )

    return output
