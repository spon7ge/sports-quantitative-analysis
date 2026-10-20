"""Pregame rows for the minutes quantile model."""

from __future__ import annotations

import pandas as pd

from src.features.minutes import add_minutes_features, add_starter_features
from src.features.points.pregame import OUTCOME_COLUMNS
from src.pipeline.silver.columns import parse_minutes
from src.live.nba_lineup import projected_lineups
from src.pipeline.silver.positions import player_name_keys
from src.pipeline.silver.rotowire import normalize_team_abbreviation

# Same columns as MIN_FEATURES in notebooks/nba/minutes/minutes_xgboost.ipynb.
MIN_FEATURES = [
    "starting",
    "starter_rank",
    "start_rate_10",
    "season_start_rate",
    "start_rate_x_abs_spread",
    "starter_min_missing_teammates",
    "total_stars_active",
    "min_avg_lN",
    "min_lag_1",
    "min_ewm_hl_10",
    "player_days_since_appearance",
    "player_appearances_prev_7d",
    "implied_team_total",
    "min_p80_10",
    "min_p80_minus_p20_10",
]

_HISTORY_COLUMNS = (
    "player_id",
    "game_id",
    "game_date",
    "team_id",
    "minutes",
    "season_year",
)
_TONIGHT_COLUMNS = (
    "player_id",
    "player_name",
    "team_abbreviation",
    "team_id",
    "game_id",
    "game_date",
    "season_year",
)


def build_minutes_live(
    history: pd.DataFrame,
    tonight: pd.DataFrame,
    lineups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Feature tonight's rows with the minutes model's columns.

    ``lineups`` is ``{team abbreviation: [five names]}`` from
    ``projected_lineups``. A player in that five, projected or confirmed,
    gets ``starting`` 1. Everyone else gets 0. History keeps its own
    start flags so the trailing start rate is unchanged.
    """
    if lineups is None:
        lineups = projected_lineups()
    _require_columns(history, _HISTORY_COLUMNS, "history")
    _require_columns(tonight, _TONIGHT_COLUMNS, "tonight")

    past = _use_playing_minutes(history)
    past["starting"] = _historical_starting(past)
    slate = _prepare_tonight(past, tonight, lineups)

    panel = pd.concat([past, slate], ignore_index=True)
    panel["game_id"] = panel["game_id"].astype(str)
    panel = add_starter_features(panel)
    panel = add_minutes_features(panel)
    panel["total_stars_active"] = _total_stars_active(panel)

    live_ids = set(slate["game_id"].astype(str))
    live = panel.loc[panel["game_id"].astype(str).isin(live_ids)].copy()
    return live.reset_index(drop=True)


def _prepare_tonight(
    history: pd.DataFrame,
    tonight: pd.DataFrame,
    lineups: dict[str, list[str]],
) -> pd.DataFrame:
    slate = tonight.copy()
    for column in OUTCOME_COLUMNS:
        if column not in slate.columns:
            continue
        if column in {"start_position", "comment"}:
            slate[column] = ""
        else:
            slate[column] = pd.NA
    # Last appearance stands in for "played" so starter-availability
    # features do not treat the whole roster as out. Shifted minutes
    # features never read this value.
    filled = slate["player_id"].map(_last_positive_minutes(history))
    slate["minutes"] = filled
    slate["starting"] = _starting_from_lineups(slate, lineups).to_numpy()
    return _use_playing_minutes(slate)


def _starting_from_lineups(
    frame: pd.DataFrame,
    lineups: dict[str, list[str]],
) -> pd.Series:
    lookup: dict[str, set[str]] = {}
    for abbreviation, names in lineups.items():
        team = normalize_team_abbreviation(abbreviation)
        keys: set[str] = set()
        for name in names:
            keys.update(player_name_keys(name))
        if team:
            lookup.setdefault(team, set()).update(keys)

    teams = frame["team_abbreviation"].map(normalize_team_abbreviation)
    flags = [
        int(bool(set(player_name_keys(name)) & lookup.get(team, set())))
        for name, team in zip(frame["player_name"], teams)
    ]
    return pd.Series(flags, index=frame.index, dtype=int)


def _historical_starting(frame: pd.DataFrame) -> pd.Series:
    if "starting" in frame.columns and frame["starting"].notna().any():
        return pd.to_numeric(frame["starting"], errors="coerce").fillna(0).astype(int)
    if "start_position" not in frame.columns:
        raise ValueError("history needs starting or start_position")
    started = frame["start_position"].astype("string").str.strip()
    return started.isin(["G", "F", "C"]).astype(int)


def _use_playing_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one numeric minutes column.

    ``add_minutes_features`` uses the first of ``min_sec``, ``min``, and
    ``minutes`` that has any value. A live row must not be the only row
    that fills an earlier column, or every past game loses its minutes.
    """
    result = frame.copy()
    minutes = pd.to_numeric(result.get("minutes"), errors="coerce")
    if "min" in result.columns:
        minutes = minutes.fillna(pd.to_numeric(result["min"], errors="coerce"))
    if "min_sec" in result.columns:
        minutes = minutes.fillna(parse_minutes(result["min_sec"]))
    result["minutes"] = minutes
    return result.drop(columns=[name for name in ("min", "min_sec") if name in result.columns])


def _last_positive_minutes(history: pd.DataFrame) -> pd.Series:
    minutes = pd.to_numeric(history["minutes"], errors="coerce")
    played = history.loc[minutes.gt(0)].sort_values(
        ["game_date", "game_id"],
        kind="mergesort",
    )
    if played.empty:
        return pd.Series(dtype="float64")
    last = played.groupby("player_id", sort=False).tail(1)
    return last.set_index("player_id")["minutes"]


def _total_stars_active(frame: pd.DataFrame) -> pd.Series:
    stars = frame["starter_rank"].le(3)
    played = frame["minutes"].fillna(0).gt(0)
    active = (
        frame.loc[stars, ["team_id", "game_id"]]
        .assign(played=played.loc[stars].astype(int))
        .groupby(["team_id", "game_id"], sort=False)["played"]
        .sum()
    )
    indexed = frame.set_index(["team_id", "game_id"])
    return pd.Series(
        indexed.index.map(active),
        index=frame.index,
    ).fillna(0).astype(int)


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
