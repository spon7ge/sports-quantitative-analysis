"""Pregame role for Layer 2, from the player's side of one game.

A training row is a game already played. Its history is every earlier
appearance of that player, his team, and that game's opponent. The
current box score is the script and the foul total, not an input.

A live row is a game that has not been played. History is the same
windows with no extra shift, so the latest completed game stays in.
The slate names the opponent, the home flag, the spread, and whether
he is starting. The script arrives later from a Layer 1 draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from game_enviroment import REST_CAP, season_of

ROLE_WINDOW = 10
BLOWOUT_MARGIN = 20
EARLY_SEASON_GAMES = 10
REG_MIN = 48.0
OT_MIN = 5.0
STARTER_POSITIONS = frozenset({"G", "F", "C"})

PLAYER_HISTORY = (
    "min_mean_10",
    "min_std_10",
    "start_rate_10",
    "pf_mean_10",
    "season_games_prior",
    "early_season",
)
TEAM_HISTORY = (
    "team_pace_10",
    "team_off_10",
    "team_def_10",
    "team_rest",
    "team_b2b",
    "team_rested",
)
OPP_HISTORY = (
    "opp_pace_10",
    "opp_off_10",
    "opp_def_10",
    "opp_rest",
    "opp_b2b",
    "opp_rested",
)
SLATE_KNOWN = ("is_home", "is_starter", "team_spread")
SCRIPT = (
    "team_final_margin",
    "abs_final_margin",
    "n_ot",
    "went_ot",
    "game_minutes",
    "blowout",
)

_LOG_REQUIRED = ("player_id", "game_id", "game_date", "team_id", "matchup")
_SLATE_REQUIRED = (
    "player_id",
    "game_date",
    "team_id",
    "opp_team_id",
    "is_home",
    "is_starter",
)
_ENV_REQUIRED = ("team_margin", "n_ot", "game_minutes")


def build_role_history(logs: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game. History excludes that game."""
    frame = _prepare_logs(logs)
    teams = _team_games(frame)
    players = _player_history(frame)
    out = players.merge(
        teams[_TEAM_MERGE],
        on=["game_id", "team_id"],
        how="left",
        validate="many_to_one",
    )
    opponent = teams.rename(
        columns={
            "team_id": "opp_team_id",
            "team_pace_10": "opp_pace_10",
            "team_off_10": "opp_off_10",
            "team_def_10": "opp_def_10",
            "team_rest": "opp_rest",
            "team_b2b": "opp_b2b",
            "team_rested": "opp_rested",
        }
    )
    out = out.merge(
        opponent[["game_id", "opp_team_id", *OPP_HISTORY]],
        on=["game_id", "opp_team_id"],
        how="left",
        validate="many_to_one",
    )
    out = out.merge(_script(frame), on="game_id", how="left", validate="many_to_one")
    out["team_final_margin"] = out["team_pts"] - out["opp_pts"]
    out["abs_final_margin"] = out["team_final_margin"].abs()
    out["blowout"] = (out["abs_final_margin"] >= BLOWOUT_MARGIN).astype(float)
    out.loc[out["team_final_margin"].isna(), "blowout"] = np.nan
    return _ordered(
        out,
        (
            "player_id",
            "game_id",
            "game_date",
            "season_year",
            "team_id",
            "opp_team_id",
            *PLAYER_HISTORY,
            *TEAM_HISTORY,
            *OPP_HISTORY,
            *SLATE_KNOWN,
            "pf",
            *SCRIPT,
        ),
    )


def role_for_slate(logs: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """One row per upcoming player. History includes the latest completed game."""
    prepared = _prepare_logs(logs)
    upcoming = _prepare_slate(slate)
    parts = []
    for date, rows in upcoming.groupby("game_date", sort=False):
        prior = prepared.loc[prepared["game_date"] < date].copy()
        parts.append(_live_rows(prior, rows.reset_index(drop=True)))
    return _ordered(
        pd.concat(parts, ignore_index=True),
        (
            "player_id",
            "game_date",
            "season_year",
            "team_id",
            "opp_team_id",
            *PLAYER_HISTORY,
            *TEAM_HISTORY,
            *OPP_HISTORY,
            *SLATE_KNOWN,
        ),
    )


def attach_script(role: pd.DataFrame, environment: pd.DataFrame) -> pd.DataFrame:
    """Copy one Layer 1 draw onto every player. Does not draw a new script."""
    if not isinstance(role, pd.DataFrame) or role.empty:
        raise ValueError("role must be a non-empty DataFrame")
    if not isinstance(environment, pd.DataFrame) or environment.empty:
        raise ValueError("environment must be a non-empty DataFrame")
    missing = [column for column in _ENV_REQUIRED if column not in environment.columns]
    if missing:
        raise ValueError(f"environment needs {missing}")
    env = environment.loc[:, list(_ENV_REQUIRED)].copy()
    env["team_final_margin"] = pd.to_numeric(env["team_margin"], errors="coerce")
    env["n_ot"] = pd.to_numeric(env["n_ot"], errors="coerce")
    env["game_minutes"] = pd.to_numeric(env["game_minutes"], errors="coerce")
    if env[["team_final_margin", "n_ot", "game_minutes"]].isna().any().any():
        raise ValueError("environment script has missing values")
    env["abs_final_margin"] = env["team_final_margin"].abs()
    env["went_ot"] = (env["n_ot"] > 0).astype(float)
    env["blowout"] = (env["abs_final_margin"] >= BLOWOUT_MARGIN).astype(float)
    env["sim_id"] = np.arange(len(env))
    env = env.drop(columns="team_margin")
    base = role.copy()
    overlap = [column for column in SCRIPT if column in base.columns]
    if overlap:
        base = base.drop(columns=overlap)
    base["player_row"] = np.arange(len(base))
    out = base.merge(env, how="cross")
    return out.sort_values(["sim_id", "player_row"]).reset_index(drop=True)


def _live_rows(prior: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    players = _latest_player(prior, slate)
    teams = _latest_team(prior)
    out = slate.merge(players, on="player_id", how="left", validate="one_to_one")
    out = out.merge(teams, on="team_id", how="left", validate="many_to_one")
    opponent = teams.rename(
        columns={
            "team_id": "opp_team_id",
            "team_pace_10": "opp_pace_10",
            "team_off_10": "opp_off_10",
            "team_def_10": "opp_def_10",
            "last_date": "opp_last_date",
        }
    )
    out = out.merge(
        opponent[["opp_team_id", *OPP_HISTORY[:3], "opp_last_date"]],
        on="opp_team_id",
        how="left",
        validate="many_to_one",
    )
    slate_date = out["game_date"]
    out["team_rest"] = _rest(slate_date, out["last_date"])
    out["opp_rest"] = _rest(slate_date, out["opp_last_date"])
    out["team_b2b"] = (out["team_rest"] <= 1).astype(float)
    out["team_rested"] = (out["team_rest"] >= 3).astype(float)
    out["opp_b2b"] = (out["opp_rest"] <= 1).astype(float)
    out["opp_rested"] = (out["opp_rest"] >= 3).astype(float)
    for column in PLAYER_HISTORY:
        if column not in out.columns:
            out[column] = np.nan
    missing_player = out["season_games_prior"].isna()
    out.loc[missing_player, "season_games_prior"] = 0.0
    out.loc[missing_player, "early_season"] = 1.0
    return out


def _latest_player(prior: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["player_id", *PLAYER_HISTORY])
    if prior.empty:
        return empty
    work = prior.sort_values(["player_id", "game_date", "game_id"]).copy()
    work["min_mean_10"] = _through(work, "player_id", "minutes", "mean", 1)
    work["min_std_10"] = _through(work, "player_id", "minutes", "std", 2)
    work["start_rate_10"] = _through(work, "player_id", "started", "mean", 1)
    work["pf_mean_10"] = _through(work, "player_id", "pf", "mean", 1)
    last = work.groupby("player_id", sort=False).tail(1)
    season = slate[["player_id", "season_year"]].drop_duplicates("player_id")
    counted = work.merge(
        season,
        on="player_id",
        how="inner",
        suffixes=("", "_slate"),
    )
    played_season = counted["season_year"].astype(str)
    slate_season = counted["season_year_slate"].astype(str)
    same_season = played_season == slate_season
    counted = counted.loc[same_season]
    games = counted.groupby("player_id").size().rename("season_games_prior")
    last = last.merge(games, on="player_id", how="left")
    last["season_games_prior"] = last["season_games_prior"].fillna(0.0)
    last["early_season"] = _early(last["season_games_prior"])
    return last[["player_id", *PLAYER_HISTORY]]


def _latest_team(prior: pd.DataFrame) -> pd.DataFrame:
    columns = ["team_id", "team_pace_10", "team_off_10", "team_def_10", "last_date"]
    if prior.empty:
        return pd.DataFrame(columns=columns)
    teams = _team_games(prior, entering=False)
    ordered = teams.sort_values(["team_id", "game_date", "game_id"])
    last = ordered.groupby("team_id", sort=False).tail(1)
    return last[columns]


def _prepare_logs(logs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(logs, pd.DataFrame) or logs.empty:
        raise ValueError("logs must be a non-empty DataFrame")
    missing = [column for column in _LOG_REQUIRED if column not in logs.columns]
    if missing:
        raise ValueError(f"logs need {missing}")
    frame = logs.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    if frame["game_date"].isna().any():
        raise ValueError("game_date has unparseable values")
    frame["game_id"] = frame["game_id"].astype(str)
    frame["minutes"] = _minutes(frame)
    _reject_minute_conflicts(frame)
    frame["started"] = _started(frame)
    frame["pf"] = _numeric(frame, "pf")
    frame["team_pts"] = _numeric(frame, "team_pts")
    frame["opp_pts"] = _numeric(frame, "opp_pts")
    home = frame["matchup"].astype("string").str.contains(r"\bvs\.", regex=True)
    frame["is_home"] = home.astype(float)
    frame["team_spread"] = _spread(frame)
    if "season_year" not in frame.columns:
        frame["season_year"] = frame["game_date"].map(season_of)
    frame["season_year"] = frame["season_year"].astype(str)
    frame = frame.drop_duplicates(["game_id", "player_id", "team_id"])
    frame["opp_team_id"] = _opponents(frame)
    return frame.sort_values(["game_date", "game_id", "player_id"]).reset_index(
        drop=True
    )


def _prepare_slate(slate: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(slate, pd.DataFrame) or slate.empty:
        raise ValueError("slate must be a non-empty DataFrame")
    missing = [column for column in _SLATE_REQUIRED if column not in slate.columns]
    if missing:
        raise ValueError(f"slate needs {missing}")
    out = slate.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.normalize()
    if out["game_date"].isna().any():
        raise ValueError("slate game_date has unparseable values")
    if out.duplicated(["player_id", "game_date"]).any():
        raise ValueError("slate has two rows for one player on one date")
    if out["team_id"].eq(out["opp_team_id"]).any():
        raise ValueError("a player cannot be his own opponent")
    out["is_home"] = _flag(out["is_home"], "is_home")
    out["is_starter"] = _flag(out["is_starter"], "is_starter")
    if "team_spread" in out.columns:
        out["team_spread"] = _numeric(out, "team_spread")
    else:
        out["team_spread"] = np.nan
    if "season_year" not in out.columns:
        out["season_year"] = out["game_date"].map(season_of)
    out["season_year"] = out["season_year"].astype(str)
    return out.reset_index(drop=True)


def _player_history(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values(["player_id", "game_date", "game_id"]).copy()
    work["min_mean_10"] = _before(work, "player_id", "minutes", "mean", 1)
    work["min_std_10"] = _before(work, "player_id", "minutes", "std", 2)
    work["start_rate_10"] = _before(work, "player_id", "started", "mean", 1)
    work["pf_mean_10"] = _before(work, "player_id", "pf", "mean", 1)
    counts = (
        work.groupby(["player_id", "season_year", "game_date"], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    played = counts.groupby(["player_id", "season_year"], sort=False)["n"].cumsum()
    counts["season_games_prior"] = played - counts["n"]
    work = work.merge(
        counts[["player_id", "season_year", "game_date", "season_games_prior"]],
        on=["player_id", "season_year", "game_date"],
        how="left",
        validate="many_to_one",
    )
    work["early_season"] = _early(work["season_games_prior"])
    work["is_starter"] = work["started"]
    return work


def _team_games(frame: pd.DataFrame, *, entering: bool = True) -> pd.DataFrame:
    rows = frame.sort_values(["game_date", "game_id", "team_id", "player_id"]).copy()
    rows["_pace"] = _stat(rows, ("team_pace", "pace"))
    rows["_off"] = _stat(rows, ("team_off_rating", "off_rating"))
    rows["_def"] = _stat(rows, ("team_def_rating", "def_rating"))
    teams = rows.groupby(["game_id", "team_id"], as_index=False).agg(
        game_date=("game_date", "first"),
        season_year=("season_year", "first"),
        pace=("_pace", "median"),
        off=("_off", "median"),
        defense=("_def", "median"),
    )
    teams = teams.sort_values(["team_id", "game_date", "game_id"])
    teams = teams.reset_index(drop=True)
    value = _before if entering else _through
    teams["team_pace_10"] = value(teams, "team_id", "pace", "mean", 1)
    teams["team_off_10"] = value(teams, "team_id", "off", "mean", 1)
    teams["team_def_10"] = value(teams, "team_id", "defense", "mean", 1)
    previous = teams.groupby("team_id", sort=False)["game_date"].shift(1)
    teams["last_date"] = previous if entering else teams["game_date"]
    if entering:
        teams["team_rest"] = _rest(teams["game_date"], previous)
    else:
        teams["team_rest"] = np.nan
    teams["team_b2b"] = (teams["team_rest"] <= 1).astype(float)
    teams["team_rested"] = (teams["team_rest"] >= 3).astype(float)
    return teams


_TEAM_MERGE = [
    "game_id",
    "team_id",
    *TEAM_HISTORY,
]


def _script(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["game_id", "team_id"], as_index=False)
    team_minutes = grouped["minutes"].sum()
    totals = team_minutes.groupby("game_id", as_index=False)["minutes"].sum()
    totals["game_minutes"] = totals["minutes"] / 10.0
    totals["n_ot"] = np.clip(
        np.round((totals["game_minutes"] - REG_MIN) / OT_MIN),
        0,
        None,
    ).astype(int)
    totals["went_ot"] = (totals["n_ot"] > 0).astype(float)
    return totals[["game_id", "game_minutes", "n_ot", "went_ot"]]


def _opponents(frame: pd.DataFrame) -> pd.Series:
    sides = frame.loc[:, ["game_id", "team_id"]].drop_duplicates()
    paired = sides.merge(sides, on="game_id", suffixes=("", "_opp"))
    paired = paired.loc[paired["team_id"] != paired["team_id_opp"]]
    if paired.duplicated(["game_id", "team_id"]).any():
        raise ValueError("a game has more than two teams")
    found = frame.merge(
        paired[["game_id", "team_id", "team_id_opp"]],
        on=["game_id", "team_id"],
        how="left",
        validate="many_to_one",
    )
    return found["team_id_opp"]


def _before(
    frame: pd.DataFrame,
    entity: str,
    column: str,
    how: str,
    min_periods: int,
) -> np.ndarray:
    """Window ending on the previous date. Same-day rows share that window."""
    ordered = frame.sort_values([entity, "game_date", "game_id"]).copy()
    ordered["_row"] = np.arange(len(ordered))
    ordered["_through"] = _rolling(ordered, entity, column, how, min_periods)
    right = ordered[[entity, "game_date", "game_id", "_through"]].sort_values(
        ["game_date", "game_id"]
    )
    left = ordered[[entity, "game_date", "game_id", "_row"]].sort_values(
        ["game_date", "game_id"]
    )
    found = pd.merge_asof(
        left,
        right,
        on="game_date",
        by=entity,
        allow_exact_matches=False,
        direction="backward",
    )
    found = found.sort_values("_row")
    return found["_through"].to_numpy()


def _through(
    frame: pd.DataFrame,
    entity: str,
    column: str,
    how: str,
    min_periods: int,
) -> np.ndarray:
    """Window ending on this row. The latest completed game stays in."""
    ordered = frame.sort_values([entity, "game_date", "game_id"]).copy()
    ordered["_row"] = np.arange(len(frame))
    # `_rolling` follows ordered's row order. Put the values back on `frame`.
    ordered["_through"] = _rolling(ordered, entity, column, how, min_periods)
    return ordered.sort_values("_row")["_through"].to_numpy()


def _rolling(
    frame: pd.DataFrame,
    entity: str,
    column: str,
    how: str,
    min_periods: int,
) -> pd.Series:
    window = frame.groupby(entity, sort=False)[column]
    return window.transform(lambda series: _apply_roll(series, how, min_periods))


def _apply_roll(series: pd.Series, how: str, min_periods: int) -> pd.Series:
    rolled = series.rolling(ROLE_WINDOW, min_periods=min_periods)
    return getattr(rolled, how)()


def _early(games: pd.Series) -> pd.Series:
    return (games < EARLY_SEASON_GAMES).astype(float)


def _rest(date: pd.Series, previous: pd.Series) -> pd.Series:
    gap = (pd.to_datetime(date) - pd.to_datetime(previous)).dt.days
    return gap.clip(lower=1, upper=REST_CAP).fillna(REST_CAP).astype(float)


def parse_minutes(values: pd.Series) -> pd.Series:
    """Minutes as a number, or ``MM:SS``. A blank stays missing."""
    text = values.astype("string").str.strip()
    parsed = pd.to_numeric(text, errors="coerce")
    clock = text.str.extract(r"^(\d+):(\d+(?:\.\d+)?)$")
    seconds = pd.to_numeric(clock[1], errors="coerce")
    whole = pd.to_numeric(clock[0], errors="coerce")
    parsed = parsed.where(clock[0].isna(), whole + seconds / 60.0)
    present = text.notna() & text.ne("")
    bad_clock = clock[0].notna() & ~(seconds < 60)
    if (present & (parsed.isna() | bad_clock)).any():
        raise ValueError("minutes must be a number or MM:SS")
    return parsed


def _minutes(frame: pd.DataFrame) -> pd.Series:
    primary = parse_minutes(frame["min"]) if "min" in frame.columns else None
    if "minutes" in frame.columns:
        secondary = parse_minutes(frame["minutes"])
    else:
        secondary = None
    if primary is None and secondary is None:
        raise ValueError("logs need min or minutes")
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    return primary.where(primary.notna(), secondary)


def _reject_minute_conflicts(frame: pd.DataFrame) -> None:
    keys = ["game_id", "player_id", "team_id"]
    counts = frame.groupby(keys, dropna=False)["minutes"].nunique(dropna=False)
    if (counts > 1).any():
        raise ValueError("logs have conflicting minutes for one player-game")


def _started(frame: pd.DataFrame) -> pd.Series:
    if "start_position" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    position = frame["start_position"].astype("string").str.strip().str.upper()
    return position.isin(STARTER_POSITIONS).astype(float)


def _spread(frame: pd.DataFrame) -> pd.Series:
    for column in ("player_team_spread", "team_spread"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _stat(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return values
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _flag(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() | ~numeric.isin([0.0, 1.0])
    if invalid.any():
        raise ValueError(f"{name} must be 0 or 1")
    return numeric.astype(float)


def _ordered(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.loc[:, list(columns)].copy()
    keys = [
        column
        for column in ("game_date", "game_id", "player_id")
        if column in out.columns
    ]
    return out.sort_values(keys).reset_index(drop=True)
