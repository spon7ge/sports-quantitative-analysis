from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from game_enviroment.constants import OPTIONAL, OT_MIN, REG_MIN, REQUIRED


def load_gamelogs(spec: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    pattern = os.path.join(spec, "*.parquet") if os.path.isdir(spec) else spec
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files found for {spec!r}")

    frames = []
    for path in paths:
        columns = set(pq.read_schema(path).names)
        missing = set(REQUIRED) - columns
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")

        frames.append(pd.read_parquet(
            path,
            columns=[c for c in REQUIRED + OPTIONAL if c in columns],
        ))
        print(f"  loaded {os.path.basename(path)}: {len(frames[-1]):,} rows")

    return pd.concat(
        frames, ignore_index=True
    ).reindex(columns=REQUIRED + OPTIONAL)


def season_of(date) -> str:
    date = pd.Timestamp(date)
    year = date.year if date.month >= 9 else date.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def _season_label(value) -> str:
    year = int(str(value)[:4])
    return f"{year}-{(year + 1) % 100:02d}"


def build_game_table(
    df: pd.DataFrame,
    include_playoffs=False,
) -> pd.DataFrame:
    df = df.copy()

    for column in OPTIONAL:
        if column not in df:
            df[column] = np.nan

    df["game_date"] = pd.to_datetime(
        df["game_date"], errors="coerce"
    ).dt.normalize()
    df = df.dropna(
        subset=["game_id", "game_date", "team_id", "season_year"]
    )
    df["game_id"] = df["game_id"].astype(str)
    df["season_year"] = df["season_year"].map(_season_label)

    allowed = {"Regular Season"}
    if include_playoffs:
        allowed.update({"Playoffs", "PlayIn", "Play-In"})
    df = df[
        df["season_type"].fillna("Regular Season").isin(allowed)
    ]

    numeric = [
        "team_id", "min", "team_poss", "opp_poss", "team_pts",
        "opp_pts", "game_total", "team_spread", "player_team_spread",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["team_id"])

    known = df["player_id"].notna()
    df = pd.concat([
        df[known].drop_duplicates(["game_id", "team_id", "player_id"]),
        df[~known],
    ], ignore_index=True)

    if (~known).any():
        print(
            "  WARNING: missing player IDs; unidentified rows "
            "cannot be deduplicated safely."
        )

    df["spread_any"] = df["team_spread"].fillna(
        df["player_team_spread"]
    )
    groups = df.groupby(["game_id", "team_id"])
    repeated = [
        "game_date", "season_year", "team_poss",
        "opp_poss", "team_pts", "opp_pts",
    ]
    if groups[repeated].nunique().gt(1).any().any():
        raise ValueError(
            "Conflicting team statistics within a game; check the input logs."
        )

    tg = groups.agg(
        season_year=("season_year", "first"),
        game_date=("game_date", "first"),
        abbr=("team_abbreviation", "first"),
        matchup=("matchup", "first"),
        poss=("team_poss", "first"),
        opp_poss=("opp_poss", "first"),
        pts=("team_pts", "first"),
        opp_pts=("opp_pts", "first"),
        total=("game_total", "first"),
        spread=("spread_any", "first"),
        team_min=("min", "sum"),
    ).reset_index()

    home = tg["matchup"].str.contains(
        r"\bvs\.", case=False, na=False
    )
    away = tg["matchup"].str.contains(
        "@", regex=False, na=False
    )
    counts = tg.assign(home=home, away=away).groupby("game_id").agg(
        teams=("team_id", "size"),
        homes=("home", "sum"),
        aways=("away", "sum"),
    )
    valid = counts.index[
        (counts.teams == 2)
        & (counts.homes == 1)
        & (counts.aways == 1)
    ]

    m = tg[home & tg.game_id.isin(valid)].merge(
        tg[away & tg.game_id.isin(valid)],
        on="game_id",
        suffixes=("_h", "_a"),
        validate="one_to_one",
    )

    if (
        (m.game_date_h != m.game_date_a).any()
        or (m.season_year_h != m.season_year_a).any()
    ):
        raise ValueError(
            "The two teams disagree about the game's date or season."
        )

    for own, other in (
        ("poss_h", "opp_poss_a"),
        ("poss_a", "opp_poss_h"),
        ("pts_h", "opp_pts_a"),
        ("pts_a", "opp_pts_h"),
    ):
        known = m[own].notna() & m[other].notna()
        if not np.allclose(
            m.loc[known, own],
            m.loc[known, other],
            atol=0.01,
            rtol=0,
        ):
            raise ValueError(
                "Opposing team logs disagree about possessions or points."
            )

    g = pd.DataFrame({
        "game_id": m.game_id,
        "game_date": m.game_date_h,
        "season_year": m.season_year_h,
        "home_id": m.team_id_h.astype("int64"),
        "away_id": m.team_id_a.astype("int64"),
        "home_abbr": m.abbr_h,
        "away_abbr": m.abbr_a,
        "home_poss": m.poss_h.fillna(m.opp_poss_a),
        "away_poss": m.poss_a.fillna(m.opp_poss_h),
        "home_pts": m.pts_h.fillna(m.opp_pts_a),
        "away_pts": m.pts_a.fillna(m.opp_pts_h),
        "game_minutes": (m.team_min_h + m.team_min_a) / 10.0,
        "total": m.total_h.fillna(m.total_a),
        "home_spread": m.spread_h.fillna(-m.spread_a),
    })

    g["n_ot"] = np.maximum(
        0, np.rint((g.game_minutes - REG_MIN) / OT_MIN)
    ).astype(int)
    g["nominal_min"] = REG_MIN + OT_MIN * g.n_ot
    g["poss"] = (g.home_poss + g.away_poss) / 2.0
    g["poss48"] = g.poss * REG_MIN / g.nominal_min
    g["margin"] = g.home_pts - g.away_pts
    g["pts_total"] = g.home_pts + g.away_pts

    complete = g[
        ["home_poss", "away_poss", "home_pts", "away_pts"]
    ].gt(0).all(axis=1)
    minutes_ok = (
        (m.team_min_h - 5 * g.nominal_min).abs() <= 8
    ) & (
        (m.team_min_a - 5 * g.nominal_min).abs() <= 8
    )
    g = g[complete & minutes_ok & g.poss48.between(70, 130)]

    print(
        f"  kept {len(g):,}/{len(counts):,} games "
        "after pairing and cleaning"
    )
    if g.empty:
        raise ValueError(
            "No usable games; check minutes, matchups, "
            "and possession columns."
        )

    return g.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)
