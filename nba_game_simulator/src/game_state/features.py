"""Pregame team margin and player rolling features."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from game_state.constants import BLOWOUT_MARGIN, CARRY_GAMES


def add_pregame_features(games, shrink=10, home_adv=2.5):
    """Shrunken prior mean point margin, not net rating per 100 possessions."""
    d = games.sort_values(["game_date", "gameId"]).copy()

    total = defaultdict(float)
    count = defaultdict(float)
    played = defaultdict(int)
    seasons = {}
    pace_history, rows = [], {}

    for date, day in d.groupby("game_date", sort=True):
        pace = (
            float(np.mean(pace_history[-300:]))
            if pace_history else 100.0
        )

        for r in day.itertuples():
            for team in (r.home, r.away):
                if seasons.get(team) != r.season:
                    if count[team]:
                        mean = total[team] / count[team]
                        count[team] = min(count[team], CARRY_GAMES)
                        total[team] = mean * count[team]
                    played[team] = 0
                    seasons[team] = r.season

            expected = (
                total[r.home] / (count[r.home] + shrink)
                - total[r.away] / (count[r.away] + shrink)
                + home_adv
            )
            rows[r.gameId] = (
                expected if pd.isna(r.spread) else -r.spread,
                played[r.home],
                played[r.away],
                pace,
            )

        for r in day.itertuples():
            for team, sign in ((r.home, 1), (r.away, -1)):
                total[team] += sign * r.margin
                count[team] += 1
                played[team] += 1
            if pd.notna(r.poss) and r.poss > 0:
                pace_history.append(
                    r.poss * 48 / r.game_minutes
                )

    d[
        ["exp_margin", "home_gp", "away_gp", "prior_pace"]
    ] = pd.DataFrame(
        [rows[gid] for gid in d.gameId],
        index=d.index,
    )
    d["blowout"] = (
        d.margin.abs().ge(BLOWOUT_MARGIN).astype(int)
    )
    return d.reset_index(drop=True)


def build_player_frame(labels, games):
    columns = [
        "gameId", "personId", "teamId", "playerName",
        "minutes", "pf", "starter",
        "reconstructed_minutes", "pbp_pf",
    ]
    d = labels[columns].merge(
        games, on="gameId", validate="many_to_one"
    )

    if (~(
        d.teamId.eq(d.home) | d.teamId.eq(d.away)
    )).any():
        raise ValueError("Player team does not match game teams")

    d = d[
        d.minutes.gt(0)
        & d.minutes.le(d.game_minutes + 0.1)
    ].copy()
    d = d.sort_values(["personId", "game_date", "gameId"])
    d["pf"] = d.pf.fillna(d.pbp_pf)

    for side in ("home", "away"):
        mask = (
            d.teamId.eq(d[side])
            & d.starter.isna()
            & d.lineup_ok
        )
        if mask.any():
            d.loc[mask, "starter"] = [
                float(pid in (starters or ()))
                for pid, starters in zip(
                    d.loc[mask, "personId"],
                    d.loc[mask, side + "_starters"],
                )
            ]

    group = d.groupby("personId", sort=False)
    for source, target, prior in [
        ("minutes", "roll_min", 18),
        ("starter", "roll_start", 0),
        ("pf", "roll_pf", 2),
    ]:
        d[target] = group[source].transform(
            lambda x: x.shift().rolling(
                10, min_periods=1
            ).mean()
        ).fillna(prior)

    d["roll_min_sd"] = group.minutes.transform(
        lambda x: x.shift().rolling(
            10, min_periods=2
        ).std()
    ).fillna(6)

    previous_team = group.teamId.shift()
    d["team_change"] = (
        previous_team.notna()
        & previous_team.ne(d.teamId)
    ).astype(int)
    d["is_home"] = d.teamId.eq(d.home).astype(int)
    d["is_starter"] = d.starter.fillna(
        d.roll_start.ge(0.5).astype(float)
    )

    sign = np.where(d.is_home.eq(1), 1, -1)
    d["team_exp_margin"] = d.exp_margin * sign
    d["team_final_margin"] = d.margin * sign
    d["abs_final_margin"] = d.margin.abs()
    d["early_season"] = np.where(
        d.is_home.eq(1), d.home_gp, d.away_gp
    ) < 10

    lead = d.late_lead_min.copy()
    trail = d.late_trail_min.copy()
    d["late_lead_min"] = np.where(
        d.is_home.eq(1), lead, trail
    )
    d["late_trail_min"] = np.where(
        d.is_home.eq(1), trail, lead
    )
    return d.reset_index(drop=True)
