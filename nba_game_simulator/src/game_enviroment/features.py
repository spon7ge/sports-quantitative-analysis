from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from game_enviroment.constants import FEATURES, REST_CAP
from lineup import LINEUP_GAME_FEATURES


def _new_state():
    return {
        "num": 0.0,
        "den": 0.0,
        "last_date": None,
        "last_season": None,
    }


def _effective(st, season, carry):
    years = 0 if st["last_season"] is None else max(
        0,
        int(season[:4]) - int(str(st["last_season"])[:4]),
    )
    return st["num"] * carry ** years, st["den"] * carry ** years


def _team_snapshot(st, date, season, carry, prior_w):
    num, den = _effective(st, season, carry)
    rest = REST_CAP if st["last_date"] is None else np.clip(
        (pd.Timestamp(date) - pd.Timestamp(st["last_date"])).days,
        1,
        REST_CAP,
    )
    pace = num / (den + prior_w) if den + prior_w > 0 else 0.0
    return pace, float(rest)


def _finite(value):
    try:
        return (
            value is not None
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(float(value))
        )
    except (TypeError, ValueError):
        return False


def _line_feats(total, spread, league_pts):
    tm, sm = not _finite(total), not _finite(spread)
    return {
        "total_dev": 0.0 if tm else float(total) - league_pts,
        "abs_spread": 0.0 if sm else abs(float(spread)),
        "total_missing": float(tm),
        "spread_missing": float(sm),
    }


def build_features(
    g, halflife, prior_w, carry, window, min_league,
):
    g = g.copy()
    decay = 0.5 ** (1.0 / halflife)
    columns = [
        c for c in FEATURES if c not in LINEUP_GAME_FEATURES
    ] + ["lg_poss", "lg_pts"]
    out = {c: np.full(len(g), np.nan) for c in columns}
    teams = {}
    lp, lt = deque(maxlen=window), deque(maxlen=window)

    for date, indices in g.groupby(
        "game_date", sort=True
    ).indices.items():
        ready = len(lp) >= min_league
        league_poss, league_pts = (
            (float(np.mean(lp)), float(np.mean(lt)))
            if ready else (0.0, 0.0)
        )

        for i in indices:
            row = g.iloc[i]
            h, a = int(row.home_id), int(row.away_id)
            ph, rh = _team_snapshot(
                teams.setdefault(h, _new_state()),
                date, row.season_year, carry, prior_w,
            )
            pa, ra = _team_snapshot(
                teams.setdefault(a, _new_state()),
                date, row.season_year, carry, prior_w,
            )
            if ready:
                values = _line_feats(
                    row.total, row.home_spread, league_pts
                )
                values.update(
                    pace_h=ph,
                    pace_a=pa,
                    b2b_h=float(rh <= 1),
                    b2b_a=float(ra <= 1),
                    rested_h=float(rh >= 3),
                    rested_a=float(ra >= 3),
                    lg_poss=league_poss,
                    lg_pts=league_pts,
                )
                for key, value in values.items():
                    out[key][i] = value

        for i in indices:
            row = g.iloc[i]
            for tid in (int(row.home_id), int(row.away_id)):
                st = teams[tid]
                num, den = _effective(st, row.season_year, carry)
                if ready:
                    num = num * decay + row.poss48 - league_poss
                    den = den * decay + 1.0
                st.update(
                    num=num,
                    den=den,
                    last_date=date,
                    last_season=row.season_year,
                )

        lp.extend(g.iloc[indices].poss48)
        lt.extend(g.iloc[indices].pts_total)

    for column, values in out.items():
        g[column] = values
    g["y_dev"] = g.poss48 - g.lg_poss

    abbreviations = {}
    for side in ("home", "away"):
        abbreviations.update(zip(
            g[f"{side}_id"], g[f"{side}_abbr"]
        ))

    state = {
        "league_poss": float(np.mean(lp)),
        "league_pts": float(np.mean(lt)),
        "teams": {
            str(t): {
                **st,
                "abbr": abbreviations[t],
                "last_date": str(pd.Timestamp(st["last_date"]).date()),
            }
            for t, st in teams.items()
        },
    }
    return g, state
