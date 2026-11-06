"""Starting-five features for the game-environment model.

The five is whoever started that game: ``start_position`` in {G, F, C}.
Pace, offensive rating, and defensive rating are each player's history
from earlier games only, shrunk toward the league level when a player
has not played yet.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

LINEUP_STATS = ("pace", "off_rating", "def_rating")
LINEUP_GAME_FEATURES = [
    "lineup_pace_h",
    "lineup_pace_a",
    "lineup_off_rating_h",
    "lineup_off_rating_a",
    "lineup_def_rating_h",
    "lineup_def_rating_a",
]
STARTER_POSITIONS = frozenset({"G", "F", "C"})
STAT_SOURCES = {
    "pace": ("pace", "team_pace"),
    "off_rating": ("off_rating", "team_off_rating"),
    "def_rating": ("def_rating", "team_def_rating"),
}


def build_lineup_features(
    df: pd.DataFrame,
    halflife: float,
    prior_weight: float,
    season_carry: float,
    window: int,
    include_playoffs: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """One row per team-game, plus the player state used at simulation time.

    ``lineup_pace``, ``lineup_off_rating``, and ``lineup_def_rating`` are the
    mean prior of the players who started that team-game. A player with no
    earlier appearance takes the league average of previous team-games.
    """
    frame = _prepare(df, include_playoffs)
    decay = 0.5 ** (1.0 / halflife)
    players: dict = {}
    league = {stat: deque(maxlen=window) for stat in LINEUP_STATS}
    pieces: list[pd.DataFrame] = []

    for _date, day in frame.groupby("game_date", sort=True):
        day = day.copy()
        center = {
            stat: float(np.mean(league[stat])) if league[stat] else np.nan
            for stat in LINEUP_STATS
        }
        priors = {stat: [] for stat in LINEUP_STATS}
        for pid, season in zip(
            day["player_id"].to_numpy(),
            day["season_year"].to_numpy(),
            strict=True,
        ):
            state = _ensure(players, pid)
            _touch_season(state, season, season_carry)
            for stat in LINEUP_STATS:
                priors[stat].append(
                    _shrunk(
                        state["num"][stat],
                        state["den"][stat],
                        center[stat],
                        prior_weight,
                    )
                )
        for stat in LINEUP_STATS:
            day[f"prior_{stat}"] = priors[stat]

        keys = day[["game_id", "team_id"]].drop_duplicates()
        starters = day.loc[day["started"]]
        if starters.empty:
            agg = keys.copy()
            for stat in LINEUP_STATS:
                agg[f"lineup_{stat}"] = np.nan
        else:
            agg = starters.groupby(["game_id", "team_id"], as_index=False)[
                [f"prior_{stat}" for stat in LINEUP_STATS]
            ].mean()
            agg = agg.rename(
                columns={
                    f"prior_{stat}": f"lineup_{stat}" for stat in LINEUP_STATS
                }
            )
            agg = keys.merge(agg, on=["game_id", "team_id"], how="left")
        for stat in LINEUP_STATS:
            agg[f"lineup_{stat}"] = agg[f"lineup_{stat}"].fillna(center[stat])
        pieces.append(agg)

        played = day.loc[day["played"]].sort_values(["game_id", "player_id"])
        for rec in played.itertuples(index=False):
            state = _ensure(players, rec.player_id)
            _touch_season(state, rec.season_year, season_carry)
            for stat in LINEUP_STATS:
                value = getattr(rec, stat)
                if not np.isfinite(value):
                    continue
                state["num"][stat] = state["num"][stat] * decay + float(value)
                state["den"][stat] = state["den"][stat] * decay + 1.0
        for _key, team_day in played.groupby(["game_id", "team_id"], sort=False):
            for stat in LINEUP_STATS:
                value = float(team_day[stat].mean())
                if np.isfinite(value):
                    league[stat].append(value)

    table = pd.concat(pieces, ignore_index=True)
    table["game_id"] = table["game_id"].astype(str)
    table["team_id"] = pd.to_numeric(table["team_id"]).astype("int64")
    return table, _export_state(players, league, prior_weight, season_carry)


def merge_lineup_features(games: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Attach each side's lineup ratings to a home/away game row."""
    out = games.copy()
    out["game_id"] = out["game_id"].astype(str)
    columns = ["game_id", "team_id", *[f"lineup_{stat}" for stat in LINEUP_STATS]]
    slim = lineups[columns].copy()
    slim["game_id"] = slim["game_id"].astype(str)
    slim["team_id"] = pd.to_numeric(slim["team_id"]).astype("int64")
    for side, key in (("h", "home_id"), ("a", "away_id")):
        part = slim.rename(
            columns={
                "team_id": key,
                **{
                    f"lineup_{stat}": f"lineup_{stat}_{side}"
                    for stat in LINEUP_STATS
                },
            }
        )
        out[key] = pd.to_numeric(out[key]).astype("int64")
        out = out.merge(part, on=["game_id", key], how="left")
    return out


def fill_lineup_gaps(games: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing five with the mean of earlier games only."""
    out = games.copy()
    if "game_date" in out.columns:
        order = np.argsort(pd.to_datetime(out["game_date"]).to_numpy(), kind="mergesort")
    else:
        order = np.arange(len(out))
    for name in LINEUP_GAME_FEATURES:
        if name not in out.columns:
            continue
        values = pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=float)
        filled = values.copy()
        running = 0.0
        seen = 0
        for idx in order:
            if not np.isfinite(filled[idx]):
                if seen:
                    filled[idx] = running / seen
                continue
            running += values[idx]
            seen += 1
        out[name] = filled
    return out


def attach_lineup_fill(state: dict, games: pd.DataFrame) -> dict:
    """Remember the training-average five for a simulation with no lineup."""
    fill = {}
    for name in LINEUP_GAME_FEATURES:
        stat = name.removeprefix("lineup_").rsplit("_", 1)[0]
        fallback = float(state["league"][stat])
        if name not in games.columns:
            fill[name] = fallback
            continue
        center = float(pd.to_numeric(games[name], errors="coerce").mean(skipna=True))
        fill[name] = center if np.isfinite(center) else fallback
    state["fill"] = fill
    return state


def lineup_game_features(
    state: dict,
    season: str,
    home_ids: list | None = None,
    away_ids: list | None = None,
) -> dict[str, float]:
    """Home and away lineup ratings for one game.

    ``home_ids`` and ``away_ids`` are the player ids starting on that date.
    A missing side uses the training-average five stored on ``state``.
    """
    out: dict[str, float] = {}
    fill = state.get("fill") or {}
    for side, player_ids in (("h", home_ids), ("a", away_ids)):
        if player_ids is None:
            for stat in LINEUP_STATS:
                name = f"lineup_{stat}_{side}"
                fallback = float(state["league"][stat])
                value = fill.get(name, fallback)
                out[name] = float(value) if np.isfinite(value) else fallback
            continue
        snap = lineup_snapshot(state, player_ids, season)
        for stat in LINEUP_STATS:
            out[f"lineup_{stat}_{side}"] = snap[stat]
    return out


def lineup_snapshot(state: dict, player_ids, season: str) -> dict[str, float]:
    """Mean prior rating of these players, using state frozen before the game."""
    carry = float(state["season_carry"])
    prior_weight = float(state["prior_weight"])
    players = state["players"]
    season = str(season)
    out = {}
    for stat in LINEUP_STATS:
        center = float(state["league"][stat])
        values = []
        for pid in player_ids:
            key = str(_player_key(pid))
            player = players.get(key)
            if player is None:
                values.append(center)
                continue
            num = float(player["num"][stat])
            den = float(player["den"][stat])
            if player.get("last_season") not in (None, season):
                num *= carry
                den *= carry
            values.append(_shrunk(num, den, center, prior_weight))
        out[stat] = float(np.mean(values))
    return out


def _prepare(df: pd.DataFrame, include_playoffs: bool) -> pd.DataFrame:
    if "start_position" not in df.columns:
        raise ValueError("lineup features need start_position")
    if "player_id" not in df.columns or df["player_id"].isna().all():
        raise ValueError("lineup features need player_id")
    if "min" not in df.columns:
        raise ValueError("lineup features need min")

    chosen = {}
    for stat, options in STAT_SOURCES.items():
        source = next(
            (
                name
                for name in options
                if name in df.columns
                and pd.to_numeric(df[name], errors="coerce").notna().any()
            ),
            None,
        )
        if source is None:
            raise ValueError(
                f"lineup features need {stat} ({' or '.join(options)})"
            )
        chosen[stat] = source

    out = df.copy()
    if "season_type" in out.columns and out["season_type"].notna().any():
        allowed = {"Regular Season"}
        if include_playoffs:
            allowed |= {"Playoffs", "PlayIn", "Play-In"}
        out = out.loc[out["season_type"].fillna("Regular Season").isin(allowed)]
    if out.empty:
        raise ValueError("no games left for lineup features")

    out["game_date"] = pd.to_datetime(out["game_date"]).dt.normalize()
    out["game_id"] = out["game_id"].astype(str)
    out["team_id"] = pd.to_numeric(out["team_id"]).astype("int64")
    out["season_year"] = out["season_year"].astype(str)
    out["player_id"] = out["player_id"].map(_player_key)
    out = out.dropna(subset=["player_id"])
    out = out.drop_duplicates(["game_id", "player_id", "team_id"])
    labels = out["start_position"].astype("string").str.strip().str.upper()
    out["started"] = labels.isin(STARTER_POSITIONS)
    if not out["started"].any():
        raise ValueError(
            "lineup features need start_position values of G, F, or C for starters"
        )
    for stat, source in chosen.items():
        out[stat] = pd.to_numeric(out[source], errors="coerce")
    out["played"] = pd.to_numeric(out["min"], errors="coerce").fillna(0).gt(0)
    return out.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def _new_player() -> dict:
    return {
        "num": {stat: 0.0 for stat in LINEUP_STATS},
        "den": {stat: 0.0 for stat in LINEUP_STATS},
        "last_season": None,
    }


def _ensure(players: dict, pid) -> dict:
    key = _player_key(pid)
    state = players.get(key)
    if state is None:
        state = _new_player()
        players[key] = state
    return state


def _player_key(pid):
    if pd.isna(pid):
        return None
    if isinstance(pid, (int, np.integer)):
        return int(pid)
    if isinstance(pid, (float, np.floating)) and float(pid).is_integer():
        return int(pid)
    return str(pid)


def _touch_season(state: dict, season, carry: float) -> None:
    season = str(season)
    if state["last_season"] == season:
        return
    if state["last_season"] is not None:
        for stat in LINEUP_STATS:
            state["num"][stat] *= carry
            state["den"][stat] *= carry
    state["last_season"] = season


def _shrunk(num: float, den: float, center: float, prior_weight: float) -> float:
    if den == 0:
        return float(center) if np.isfinite(center) else np.nan
    if not np.isfinite(center) or prior_weight == 0:
        return float(num) / float(den)
    return (float(num) + prior_weight * float(center)) / (float(den) + prior_weight)


def _export_state(players: dict, league: dict, prior_weight: float, season_carry: float) -> dict:
    exported = {}
    for pid, state in players.items():
        exported[str(pid)] = {
            "num": {stat: float(state["num"][stat]) for stat in LINEUP_STATS},
            "den": {stat: float(state["den"][stat]) for stat in LINEUP_STATS},
            "last_season": (
                None if state["last_season"] is None else str(state["last_season"])
            ),
        }
    league_level = {}
    for stat in LINEUP_STATS:
        history = league[stat]
        league_level[stat] = float(np.mean(history)) if history else float("nan")
    return {
        "players": exported,
        "league": league_level,
        "prior_weight": float(prior_weight),
        "season_carry": float(season_carry),
        "fill": {},
    }
