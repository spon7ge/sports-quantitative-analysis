"""Replay one game into on-court lineups and reconstructed minutes."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from game_state.constants import REG_SECONDS


def name_keys(name):
    if pd.isna(name):
        return []

    text = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode()
    )
    words = re.findall(r"[a-z]+", text.lower())
    if words and words[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        words.pop()
    if not words:
        return []

    keys = [" ".join(words), words[-1]]
    if len(words) > 1:
        keys.append(words[0][0] + " " + words[-1])
    return keys


def period_bounds(period):
    if period <= 4:
        return 720 * (period - 1), 720
    return REG_SECONDS + 300 * (period - 5), 300


def replay_game(g, official):
    gid = g.gameId.iloc[0]
    teams = set(official.teamId.unique())
    if len(teams) != 2:
        raise ValueError(
            "Expected official logs for exactly two teams"
        )

    home_ids = set(g.loc[
        g.location.str.lower().eq("h")
        & g.teamId.isin(teams),
        "teamId",
    ])
    if len(home_ids) != 1:
        raise ValueError("Cannot resolve home team")

    home = home_ids.pop()
    away = (teams - {home}).pop()
    periods = sorted(g.period.unique())
    if (
        periods != list(range(1, max(periods) + 1))
        or max(periods) < 4
    ):
        raise ValueError("Incomplete or invalid period sequence")

    aliases = defaultdict(set)
    roster = pd.concat([
        official[["teamId", "personId", "playerName"]],
        g.loc[
            g.personId.gt(0) & g.teamId.isin(teams),
            ["teamId", "personId", "playerName"],
        ],
    ]).drop_duplicates()

    for team, pid, name in roster.itertuples(
        index=False, name=None
    ):
        for key in name_keys(name):
            aliases[(int(team), key)].add(int(pid))

    issues = defaultdict(int)
    seconds = defaultdict(float)
    pf = defaultdict(int)
    timeouts = {home: 0, away: 0}
    score = [0, 0]
    states, starters = [], {}

    def incoming(row, team, current=None):
        match = re.match(
            r"SUB:\s*(.+?)\s+FOR\s+(.+)",
            row.description,
            re.I,
        )
        if not match:
            return None

        for key in name_keys(match[1]):
            candidates = aliases.get((team, key), set())
            if current is not None:
                candidates = candidates - set(current)
            if len(candidates) == 1:
                return next(iter(candidates))
            if len(candidates) > 1:
                return None
        return None

    def infer(rows, team):
        entered, start = set(), set()
        for row in rows:
            if row.teamId != team:
                continue

            kind = row.actionType.lower()
            if kind == "substitution":
                if row.personId > 0 and row.personId not in entered:
                    start.add(row.personId)
                pin = incoming(row, team)
                if pin is None:
                    return None
                entered.add(pin)
            elif kind in {
                "made shot", "missed shot", "rebound",
                "turnover", "free throw", "foul", "",
            }:
                special = (
                    "technical" in row.subType.lower()
                    or "3 second" in row.subType.lower()
                )
                if (
                    row.personId > 0
                    and not special
                    and row.personId not in entered
                ):
                    start.add(row.personId)

        return sorted(start) if len(start) == 5 else None

    for period, part in g.groupby("period", sort=True):
        start, length = period_bounds(period)
        if (
            part.rem.isna().any()
            or not part.rem.between(0, length).all()
            or part.rem.diff().gt(1e-6).any()
        ):
            raise ValueError("Invalid/non-monotonic period clocks")

        rows = list(part.itertuples(index=False))
        lineup, on_since = {}, {}
        team_fouls = {home: 0, away: 0}

        for team in (home, away):
            provided = official.loc[
                official.teamId.eq(team)
                & official.starter.eq(1),
                "personId",
            ].tolist()

            lineup[team] = (
                sorted(provided)
                if period == 1 and len(provided) == 5
                else infer(rows, team)
            )
            if lineup[team] is None:
                issues["bad_starters"] += 1
            else:
                on_since[team] = {
                    pid: start for pid in lineup[team]
                }

            if period == 1:
                starters[team] = (
                    tuple(lineup[team])
                    if lineup[team] is not None else None
                )

        def snapshot(action, elapsed):
            remaining = (
                max(0, REG_SECONDS - elapsed)
                if period <= 4
                else start + length - elapsed
            )
            states.append(dict(
                gameId=gid,
                actionId=action,
                period=period,
                elapsed=elapsed,
                sec_rem=remaining,
                score_diff=score[0] - score[1],
                home_lineup=(
                    tuple(sorted(lineup[home]))
                    if lineup[home] is not None else None
                ),
                away_lineup=(
                    tuple(sorted(lineup[away]))
                    if lineup[away] is not None else None
                ),
                home_pf=(
                    tuple(pf[x] for x in sorted(lineup[home]))
                    if lineup[home] is not None else None
                ),
                away_pf=(
                    tuple(pf[x] for x in sorted(lineup[away]))
                    if lineup[away] is not None else None
                ),
                home_team_fouls=team_fouls[home],
                away_team_fouls=team_fouls[away],
                home_to_used=timeouts[home],
                away_to_used=timeouts[away],
            ))

        snapshot(-1, start)

        for row in rows:
            elapsed = start + length - row.rem
            team = (
                row.teamId if row.teamId in teams
                else row.personId if row.personId in teams
                else None
            )
            kind = row.actionType.lower()
            subtype = row.subType.lower()

            if (
                kind == "substitution"
                and team in teams
                and lineup[team] is not None
            ):
                out = row.personId
                pin = incoming(row, team, lineup[team])

                if out not in lineup[team] or pin is None:
                    issues["bad_substitution"] += 1
                    for pid, since in on_since[team].items():
                        seconds[pid] += elapsed - since
                    lineup[team] = None
                    on_since[team] = {}
                else:
                    seconds[out] += (
                        elapsed - on_since[team].pop(out)
                    )
                    lineup[team].remove(out)
                    lineup[team].append(pin)
                    on_since[team][pin] = elapsed

            elif kind == "foul" and row.personId > 0:
                special = (
                    "technical" in subtype
                    or "3 second" in subtype
                )
                if not special:
                    pf[row.personId] += 1
                    if team in teams and "offensive" not in subtype:
                        team_fouls[team] += 1

            elif kind == "timeout" and team in teams:
                timeouts[team] += 1

            on_court = kind in {
                "made shot", "missed shot", "rebound",
                "turnover", "free throw", "foul", "",
            }
            special = (
                "technical" in subtype or "3 second" in subtype
            )
            if (
                on_court
                and not special
                and team in teams
                and row.personId > 0
                and lineup[team] is not None
            ):
                if row.personId not in lineup[team]:
                    issues["off_court_actor"] += 1
                    for pid, since in on_since[team].items():
                        seconds[pid] += elapsed - since
                    lineup[team], on_since[team] = None, {}

            for i, value in enumerate(
                (row.scoreHome, row.scoreAway)
            ):
                if pd.notna(value):
                    score[i] = int(value)

            snapshot(row.actionId, elapsed)

        for team in (home, away):
            if lineup[team] is not None:
                for pid, since in on_since[team].items():
                    seconds[pid] += start + length - since

        snapshot(int(part.actionId.max()) + 1, start + length)

    reconstructed = pd.DataFrame([
        dict(
            gameId=gid,
            personId=pid,
            reconstructed_minutes=s / 60,
            pbp_pf=pf[pid],
        )
        for pid, s in seconds.items()
    ], columns=[
        "gameId", "personId", "reconstructed_minutes", "pbp_pf"
    ])

    minutes = 48 + 5 * (max(periods) - 4)
    possessions = official.groupby("teamId").team_poss.first()
    valid_poss = possessions.dropna()

    game = dict(
        gameId=gid,
        game_date=official.game_date.iloc[0],
        home=int(home),
        away=int(away),
        home_pts=score[0],
        away_pts=score[1],
        margin=score[0] - score[1],
        game_minutes=minutes,
        n_ot=max(periods) - 4,
        poss=(
            float(valid_poss.mean())
            if len(valid_poss) == 2 else np.nan
        ),
        home_starters=starters.get(home),
        away_starters=starters.get(away),
        lineup_ok=not issues,
        **dict(issues),
    )

    if official.game_date.nunique() != 1:
        raise ValueError("Conflicting game dates")

    home_lines = official[official.teamId.eq(home)]
    away_lines = official[official.teamId.eq(away)]
    hs = home_lines.spread.dropna()
    aws = away_lines.spread.dropna()
    totals = official.total.dropna()

    game["spread"] = (
        float(hs.iloc[0]) if len(hs)
        else -float(aws.iloc[0]) if len(aws)
        else np.nan
    )
    game["total"] = (
        float(totals.iloc[0]) if len(totals) else np.nan
    )
    return pd.DataFrame(states), reconstructed, game


def intervals(states):
    d = states.copy()
    d["dt"] = (
        d.groupby(["gameId", "period"], sort=False)
        .elapsed.shift(-1)
        - d.elapsed
    )
    return d[d.dt.gt(0)].copy()
