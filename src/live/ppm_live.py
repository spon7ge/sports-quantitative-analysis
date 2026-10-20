"""Pregame rows for the points-per-minute quantile model."""

from __future__ import annotations

import pandas as pd

from src.features.points import add_points_features
from src.live.minutes_live import (
    _historical_starting,
    _prepare_tonight,
    _use_playing_minutes,
)
from src.live.nba_lineup import projected_lineups
from src.pipeline.silver.rotowire import (
    normalize_team_abbreviation,
    parse_matchup,
)

# Same columns as PTS_FEATURES in notebooks/nba/points/points_xgboost.ipynb.
PTS_FEATURES = [
    "pts_per_min_ewm_hl_10",
    "player_fga_share_10",
    "ppm_trim_mean_20",
    "current_team_pts_per_min",
    "usg_wmean_10",
    "season_pts_per_min",
    "starting",
    "fta_per_min_10",
    "ppm_unstable",
    "opp_def_rating_mean_10",
    "team_spread_canonical",
    "min_mean_10",
]

_HISTORY_COLUMNS = (
    "player_id",
    "game_id",
    "game_date",
    "team_id",
    "team_abbreviation",
    "minutes",
    "season_year",
    "team_def_rating",
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


def build_ppm_live(
    history: pd.DataFrame,
    tonight: pd.DataFrame,
    lineups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Feature tonight's rows with the points-per-minute model's columns.

    ``opp_def_rating_mean_10`` is the opponent's prior defensive rating.
    The opponent comes from ``opp_team_id`` or from the other side of
    ``matchup``. A stub row for that team is added on tonight's game so
    the rating joins to this game instead of the player's own defense.
    """
    if lineups is None:
        lineups = projected_lineups()
    _require_columns(history, _HISTORY_COLUMNS, "history")
    _require_columns(tonight, _TONIGHT_COLUMNS, "tonight")

    past = _use_playing_minutes(history)
    past["starting"] = _historical_starting(past)
    slate = _prepare_tonight(past, _with_opponents(tonight, past), lineups)
    stubs = _opponent_stubs(slate)

    panel = pd.concat([past, slate, stubs], ignore_index=True)
    panel = add_points_features(panel)

    keys = set(zip(slate["player_id"], slate["game_id"]))
    keep = [
        pair in keys
        for pair in zip(panel["player_id"], panel["game_id"])
    ]
    return panel.loc[keep].reset_index(drop=True)


def _with_opponents(tonight: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    slate = tonight.copy()
    known = _team_ids(history)
    if "opp_team_id" not in slate.columns:
        slate["opp_team_id"] = pd.NA

    resolved = []
    for row in slate.itertuples(index=False):
        if pd.notna(row.opp_team_id):
            resolved.append(row.opp_team_id)
            continue
        matchup = getattr(row, "matchup", None)
        if matchup is None or pd.isna(matchup):
            raise ValueError(
                f"{row.player_name} needs opp_team_id or matchup to identify the opponent"
            )
        abbreviation = _opponent_abbreviation(row.team_abbreviation, matchup)
        if abbreviation not in known:
            raise ValueError(f"no team_id for opponent {abbreviation}")
        resolved.append(known[abbreviation])
    slate["opp_team_id"] = resolved
    return slate


def _opponent_abbreviation(team_abbreviation: object, matchup: object) -> str:
    away, home = parse_matchup(matchup)
    team = normalize_team_abbreviation(team_abbreviation)
    if team == home and away:
        return away
    if team == away and home:
        return home
    raise ValueError(
        f"{team_abbreviation} is not in matchup {matchup}"
    )


def _team_ids(history: pd.DataFrame) -> dict[str, object]:
    frame = history.dropna(subset=["team_abbreviation", "team_id"]).copy()
    frame["abbr"] = frame["team_abbreviation"].map(normalize_team_abbreviation)
    frame = frame.sort_values(["game_date", "game_id"], kind="mergesort")
    latest = frame.groupby("abbr", sort=False).tail(1)
    return dict(zip(latest["abbr"], latest["team_id"]))


def _opponent_stubs(slate: pd.DataFrame) -> pd.DataFrame:
    """One row per opponent on tonight's game_id.

    The defensive-rating feature joins the opponent's team row for this
    game. Without that row the prior-10 mean never attaches, even when
    the opponent's older games are in history.
    """
    present = set(zip(slate["game_id"], slate["team_id"]))
    rows = []
    synthetic_id = -1
    for row in slate.itertuples(index=False):
        key = (row.game_id, row.opp_team_id)
        if key in present:
            continue
        present.add(key)
        rows.append(
            {
                "player_id": synthetic_id,
                "team_id": row.opp_team_id,
                "opp_team_id": row.team_id,
                "game_id": row.game_id,
                "game_date": row.game_date,
                "season_year": row.season_year,
                "team_def_rating": pd.NA,
                "team_off_rating": pd.NA,
                "minutes": pd.NA,
                "starting": 0,
            }
        )
        synthetic_id -= 1
    if not rows:
        return slate.iloc[0:0].copy()
    return pd.DataFrame(rows)


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
