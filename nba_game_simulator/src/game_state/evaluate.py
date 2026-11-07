"""End-to-end evaluation using saved Layer 1 forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from game_state.constants import BLOWOUT_MARGIN
from game_state.load import game_ids
from game_state.roster import historical_roster, pregame_players
from game_state.simulate import simulate_minutes


def evaluate_pipeline(bundle, labels, games, archive):
    """Archive must contain out-of-sample historical Layer 1 draws."""
    archive = archive.copy()
    archive["gameId"] = game_ids(archive.gameId)

    records, overtime_scores = [], []
    missing_targets = compared = skipped = 0

    for game in games.to_dict("records"):
        env = archive[
            archive.gameId.eq(game["gameId"])
        ].copy()
        if env.empty:
            skipped += 1
            continue

        if "home_margin" in env:
            env["team_margin"] = env.home_margin
            env["is_home"] = 1

        for column in (
            "team_margin", "is_home", "n_ot", "game_minutes"
        ):
            env[column] = pd.to_numeric(
                env[column], errors="coerce"
            )

        if "game_date" in env:
            dates = pd.to_datetime(
                env.game_date, errors="coerce"
            ).dt.normalize()
            if not dates.eq(game["game_date"]).all():
                raise ValueError(
                    "Archived environment dates do not match "
                    "the historical game"
                )

        if "model_cutoff" in env:
            cutoffs = pd.to_datetime(
                env.model_cutoff, errors="coerce"
            )
            if (
                cutoffs.isna().any()
                or cutoffs.ge(game["game_date"]).any()
            ):
                raise ValueError(
                    "Layer 1 archive includes models fitted "
                    "on future/current outcomes"
                )

        env["game_date"] = game["game_date"]
        history = labels[
            labels.game_date < game["game_date"]
        ]

        rosters = [
            historical_roster(history, game[side])
            for side in ("home", "away")
        ]
        if min(map(len, rosters)) < 5:
            skipped += 1
            continue

        players = [
            pregame_players(
                history, roster, game, game[side]
            )
            for roster, side in zip(
                rosters, ("home", "away")
            )
        ]
        draws = simulate_minutes(
            bundle,
            *players,
            env,
            seed=int(game["gameId"]),
        )
        actual = labels[
            labels.gameId.eq(game["gameId"])
        ].set_index("personId")

        for side in ("home", "away"):
            team_actual = actual[
                actual.teamId.eq(game[side])
            ]
            pids = (
                set(draws[side].columns)
                | set(team_actual.index)
            )
            for pid in pids:
                if (
                    pid not in actual.index
                    or pd.isna(actual.loc[pid, "minutes"])
                ):
                    missing_targets += 1
                    continue

                values = (
                    draws[side][pid].to_numpy()
                    if pid in draws[side]
                    else np.zeros(len(env))
                )
                truth = float(actual.loc[pid, "minutes"])
                lo, hi = np.quantile(values, [0.05, 0.95])

                records.append(dict(
                    gameId=game["gameId"],
                    personId=pid,
                    actual=truth,
                    mean=float(values.mean()),
                    p05=lo,
                    p95=hi,
                    covered90=lo <= truth <= hi,
                ))
                compared += 1

        overtime_scores.append((
            (
                env.n_ot.gt(0).mean()
                - (game["n_ot"] > 0)
            ) ** 2,
            (
                draws["p_blowout"]
                - (abs(game["margin"]) >= BLOWOUT_MARGIN)
            ) ** 2,
        ))

    out = pd.DataFrame(records)
    print(
        "End-to-end evaluation uses prior-game roster proxies; "
        "historical injuries/starters are unknown."
    )
    print(
        f"Compared targets={compared}; "
        f"unknown targets={missing_targets}; "
        f"skipped games={skipped}"
    )

    if len(out):
        mae = np.abs(out.actual - out["mean"]).mean()
        print(
            f"  Minutes MAE {mae:.2f}; "
            f"90% coverage {out.covered90.mean():.1%}"
        )
        played = out[out.actual.gt(0)]
        played_mae = np.abs(
            played.actual - played["mean"]
        ).mean()
        print(
            f"  Actually played: MAE {played_mae:.2f}; "
            f"coverage {played.covered90.mean():.1%}"
        )
        print(
            "  OT / blowout Brier:",
            np.mean(overtime_scores, axis=0).round(4),
        )

    return out
