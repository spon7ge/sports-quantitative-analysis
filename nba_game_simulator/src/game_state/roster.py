"""Live and historical pregame player rows."""
from __future__ import annotations

import pandas as pd


def pregame_players(history, roster, game, team):
    """Roster and starting flags must be pregame inputs.

    game needs game_date, home, exp_margin, home_gp, and away_gp.
    """
    history = history[
        history.game_date < game["game_date"]
    ].sort_values(["game_date", "gameId"])

    rows = []
    for player in roster.itertuples():
        past = history[
            history.personId.eq(player.personId)
            & history.minutes.gt(0)
        ].tail(10)
        same = history[
            history.personId.eq(player.personId)
        ].tail(1)

        roll_start = (
            past.starter.mean()
            if len(past) and past.starter.notna().any()
            else 0.0
        )
        home = int(team == game["home"])

        rows.append(dict(
            personId=player.personId,
            playerName=getattr(
                player, "playerName", str(player.personId)
            ),
            roll_min=(
                float(past.minutes.mean()) if len(past) else 18.0
            ),
            roll_min_sd=(
                float(past.minutes.std()) if len(past) > 1 else 6.0
            ),
            roll_start=float(roll_start),
            roll_pf=(
                float(past.pf.mean())
                if past.pf.notna().any() else 2.0
            ),
            is_starter=float(getattr(
                player, "is_starter", roll_start >= 0.5
            )),
            is_home=home,
            team_exp_margin=(
                game["exp_margin"] * (1 if home else -1)
            ),
            early_season=int(
                game["home_gp" if home else "away_gp"] < 10
            ),
            team_change=int(
                len(same) > 0
                and same.teamId.iloc[0] != team
            ),
            available=bool(getattr(player, "available", True)),
        ))
    return pd.DataFrame(rows)


def historical_roster(history, team):
    """Prior-game roster proxy; no historical injury knowledge is assumed."""
    latest_team = (
        history.sort_values(["game_date", "gameId"])
        .drop_duplicates("personId", keep="last")
    )
    team_history = history[history.teamId.eq(team)]
    recent = (
        team_history.sort_values(["game_date", "gameId"])
        .gameId.drop_duplicates()
        .tail(20)
    )
    ids = set(
        team_history[
            team_history.gameId.isin(recent)
        ].personId
    )
    return latest_team[
        latest_team.teamId.eq(team)
        & latest_team.personId.isin(ids)
    ][["personId", "playerName"]]
