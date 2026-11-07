"""Usable games from play-by-play, plus the starters-on-floor summary."""
from __future__ import annotations

import numpy as np
import pandas as pd

from game_state.constants import LARGE_LEAD, LATE_SECONDS
from game_state.replay import intervals, replay_game


def build_tables(pbp, logs):
    states, reconstructed, games = [], [], []
    official = {
        gid: part for gid, part in logs.groupby("gameId")
    }
    skipped = []

    for gid, part in pbp.groupby("gameId", sort=True):
        if gid not in official:
            skipped.append((gid, "missing official logs"))
            continue
        try:
            s, p, gm = replay_game(part, official[gid])
        except ValueError as error:
            skipped.append((gid, str(error)))
            continue
        states.append(s)
        reconstructed.append(p)
        games.append(gm)

    if not games:
        raise ValueError("No usable games")

    states = pd.concat(states, ignore_index=True)
    reconstructed = pd.concat(reconstructed, ignore_index=True)
    games = pd.DataFrame(games)

    yy = games.gameId.str[3:5].astype(int)
    games["season"] = np.where(
        2000 + yy <= pd.Timestamp.today().year + 1,
        2000 + yy,
        1900 + yy,
    )

    labels = logs[logs.gameId.isin(games.gameId)].copy()
    labels = labels.merge(
        reconstructed,
        on=["gameId", "personId"],
        how="left",
        validate="one_to_one",
    )

    comparison = labels.dropna(
        subset=["minutes", "reconstructed_minutes"]
    )
    error = (
        comparison.minutes - comparison.reconstructed_minutes
    ).abs()
    bad = set(comparison.loc[error.gt(2), "gameId"])
    games.loc[games.gameId.isin(bad), "lineup_ok"] = False

    print(
        f"Replay: {len(games)} usable games; "
        f"{len(skipped)} skipped; "
        f"{games.lineup_ok.mean():.1%} lineups validated"
    )
    print(
        f"Matched reconstructed minutes: MAE {error.mean():.2f}; "
        "official labels used for training"
    )

    q4 = intervals(states)
    q4 = q4[q4.period.eq(4)]
    late_seconds = (
        np.minimum(q4.sec_rem, LATE_SECONDS)
        - np.maximum(q4.sec_rem - q4.dt, 0)
    ).clip(lower=0)

    q4["late_lead_min"] = (
        late_seconds / 60 * q4.score_diff.ge(LARGE_LEAD)
    )
    q4["late_trail_min"] = (
        late_seconds / 60 * q4.score_diff.le(-LARGE_LEAD)
    )
    durations = q4.groupby("gameId")[
        ["late_lead_min", "late_trail_min"]
    ].sum()
    games = games.merge(
        durations, on="gameId", validate="one_to_one"
    )

    if skipped:
        print("First skipped games:", skipped[:5])
    return states, labels, games


def starters_on_floor_table(states, games):
    good = games[games.lineup_ok]
    d = intervals(states)
    d = d[d.period.eq(4)].merge(
        good[["gameId", "home_starters", "away_starters"]],
        on="gameId",
        validate="many_to_one",
    )

    entries = []
    time_bins = [
        (0, 120, "0-2m"),
        (120, 300, "2-5m"),
        (300, 480, "5-8m"),
        (480, 720, "8-12m"),
    ]
    for side, sign in (("home", 1), ("away", -1)):
        for row in d.itertuples():
            lineup = getattr(row, side + "_lineup")
            starters = getattr(row, side + "_starters")
            if (
                lineup is None or starters is None
                or len(lineup) != 5 or len(starters) != 5
            ):
                continue

            n = len(set(lineup) & set(starters))
            for low, high, label in time_bins:
                seconds = max(
                    0,
                    min(row.sec_rem, high)
                    - max(row.sec_rem - row.dt, low),
                )
                if seconds:
                    entries.append((
                        row.gameId,
                        sign * row.score_diff,
                        label,
                        seconds,
                        n * seconds,
                    ))

    out = pd.DataFrame(entries, columns=[
        "gameId", "lead", "time_bin", "seconds", "weighted_n"
    ])
    if out.empty:
        return out

    out["lead_bin"] = pd.cut(
        out.lead,
        [-np.inf, -20, -10, -5, 0, 5, 10, 20, np.inf],
    )
    out = out.groupby(
        ["lead_bin", "time_bin"], observed=True
    ).agg(
        seconds=("seconds", "sum"),
        weighted_n=("weighted_n", "sum"),
        games=("gameId", "nunique"),
    ).reset_index()
    out["mean_starters"] = out.weighted_n / out.seconds
    return out.drop(columns="weighted_n")
