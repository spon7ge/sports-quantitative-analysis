"""Live slate features for the empirical-Bayes points model.

Builds one pregame row per board player using:
- rolling form from completed gamelogs before the slate date
- opponent identity / home flag from ``findOpp``
- opponent defensive rating from the opponent's prior 10 completed games
- team spread and game total from Pinnacle team markets
- starting status 1/0 from RotoWire posted fives
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.live.helpers import findOpp
from src.live.nba_lineup import get_starter_status, projected_lineups
from src.live.predict_points import (
    _pinnacle_lines,
    lineups_on,
    pinnacle_games_on,
)
from src.models.points_empirical_bayes import CONTEXT_COLS, WINDOW_COLS
from src.pipeline.silver.positions import player_name_keys
from src.pipeline.silver.rotowire import normalize_team_abbreviation

log = logging.getLogger(__name__)

LIVE_CONTEXT_COLS = CONTEXT_COLS
REQUIRED_HISTORY = (
    "player_id",
    "player_name",
    "team_id",
    "team_abbreviation",
    "game_id",
    "game_date",
    "pts",
    "minutes",
    "usage_pp",
    "starter",
    "fga",
    "fta",
    "team_def_rating",
)


def build_bayesian_live_features(
    history: pd.DataFrame,
    names: Sequence[str],
    date: str,
    *,
    games: list | None = None,
    lineups: dict[str, list[str]] | None = None,
    context_cols: Sequence[str] = LIVE_CONTEXT_COLS,
    min_games: int = 20,
) -> pd.DataFrame:
    """Return one feature row per named board player for ``date``.

    Index is the board player name. Rows that lack enough history or a
    resolvable opponent are skipped with a log message.
    """
    day = str(date)[:10]
    board_names = list(dict.fromkeys(str(name) for name in names))
    if not board_names:
        raise ValueError("names must be non-empty")

    full = _history_before(history, day)
    missing = [column for column in REQUIRED_HISTORY if column not in full.columns]
    if missing:
        raise KeyError(f"player history needs {missing}")

    if games is None:
        games = pinnacle_games_on(day)
        if not games:
            from src.live.predict_points import _load_pinnacle_games

            games = _load_pinnacle_games()
            log.info("no Pinnacle team file for %s; using latest snapshot", day)
    if lineups is None:
        lineups = lineups_on(day)
        if not lineups:
            lineups = projected_lineups()
            log.info("no RotoWire lineup file for %s; using projected page", day)

    rows: list[pd.Series] = []
    for board_name in board_names:
        hist_name = _history_player_name(full, board_name)
        if hist_name is None:
            log.info("no history for board name %r on %s; skipping", board_name, day)
            continue
        try:
            row = _player_live_row(
                full,
                hist_name,
                day,
                games=games,
                lineups=lineups,
                board_name=board_name,
                context_cols=context_cols,
                min_games=min_games,
            )
        except ValueError as exc:
            log.info("%s; skipping %r", exc, board_name)
            continue
        rows.append(row)

    if not rows:
        raise ValueError(f"no live feature rows for {board_names!r} on {day}")
    frame = pd.DataFrame(rows)
    frame.index.name = "board_name"
    return frame


def choose_live_slate_date(
    market_dates: Iterable[object],
    *,
    pinnacle_directory=None,
) -> str:
    """Prefer the newest market date that already has a Pinnacle team file."""
    dates = sorted(
        {
            pd.Timestamp(value).strftime("%Y-%m-%d")
            for value in market_dates
            if pd.notna(value)
        }
    )
    if not dates:
        raise ValueError("no market slate dates")
    for day in reversed(dates):
        if pinnacle_games_on(day, directory=pinnacle_directory):
            return day
    return dates[-1]


def _history_before(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(str(date)[:10]).tz_localize("UTC")
    played = pd.to_datetime(frame["game_date"], utc=True, errors="coerce")
    return (
        frame.loc[played.lt(cutoff)]
        .sort_values(["player_id", "game_date", "game_id"])
        .reset_index(drop=True)
    )


def _history_player_name(history: pd.DataFrame, board_name: str) -> str | None:
    names = history["player_name"].astype("string")
    if names.eq(board_name).any():
        return board_name
    wanted = set(player_name_keys(board_name))
    if not wanted:
        return None
    for hist_name in names.dropna().unique():
        if wanted & set(player_name_keys(str(hist_name))):
            return str(hist_name)
    return None


def _player_live_row(
    history: pd.DataFrame,
    hist_name: str,
    day: str,
    *,
    games: list,
    lineups: dict[str, list[str]],
    board_name: str,
    context_cols: Sequence[str],
    min_games: int,
) -> pd.Series:
    player = history.loc[history["player_name"].eq(hist_name)].copy()
    if len(player) < min_games:
        raise ValueError(f"need {min_games} prior games for {hist_name!r}")

    recent = player.tail(min_games)
    needed = ["pts", "minutes", "usage_pp", "starter", "fga", "fta"]
    if not np.isfinite(
        recent[needed].to_numpy(dtype=float, na_value=np.nan)
    ).all():
        raise ValueError(f"non-finite rolling inputs for {hist_name!r}")

    team_abbr = normalize_team_abbreviation(player["team_abbreviation"].iloc[-1])
    opp_abbr, is_home = findOpp(
        hist_name,
        player,
        day,
        max_days_ahead=3,
        league="nba",
    )
    if opp_abbr is None or is_home is None:
        raise ValueError(f"findOpp found no opponent for {hist_name!r} on {day}")

    opp_def = _opp_def_rating_prior(history, opp_abbr, day)
    if not np.isfinite(opp_def):
        raise ValueError(
            f"opp_def_rating_prior unavailable for {opp_abbr!r} on {day}"
        )

    spread, total = _pinnacle_lines(games, team_abbr)
    if not np.isfinite(spread) or not np.isfinite(total):
        raise ValueError(
            f"Pinnacle spread/total missing for {team_abbr!r} on {day}"
        )

    start_rate = float(recent["starter"].mean())
    starting, starting_source = _starting_flag(
        board_name=board_name,
        hist_name=hist_name,
        team_abbr=team_abbr,
        day=day,
        lineups=lineups,
        start_rate=start_rate,
    )
    implied = (float(total) - float(spread)) / 2.0

    row = {
        "player_id": int(player["player_id"].iloc[-1]),
        "player_name": hist_name,
        "team_abbreviation": team_abbr,
        "opp_abbreviation": normalize_team_abbreviation(opp_abbr),
        "is_home": float(is_home),
        "back2back": _back2back(player, day),
        "opp_def_rating_prior": float(opp_def),
        "team_spread_pregame": float(spread),
        "game_total_pregame": float(total),
        "implied_team_total_pregame": float(implied),
        "start_rate": start_rate,
        "starting": starting,
        "starting_source": starting_source,
        "starter_surprise": starting - start_rate,
        "usage_trend": float(
            recent["usage_pp"].tail(5).mean() - recent["usage_pp"].mean()
        ),
        "minutes_trend": float(
            recent["minutes"].tail(5).mean() - recent["minutes"].mean()
        ),
        "fga_trend": float(recent["fga"].tail(5).mean() - recent["fga"].mean()),
        "fta_trend": float(recent["fta"].tail(5).mean() - recent["fta"].mean()),
    }
    for window in (5, 10, 20):
        row[f"p{window}"] = float(recent["pts"].tail(window).mean())

    required = ["player_id", *WINDOW_COLS, "start_rate", *context_cols, "starting"]
    missing = [column for column in required if column not in row]
    if missing:
        raise KeyError(missing)
    values = {column: row[column] for column in required}
    values.update(
        {
            "player_name": row["player_name"],
            "team_abbreviation": row["team_abbreviation"],
            "opp_abbreviation": row["opp_abbreviation"],
            "game_total_pregame": row["game_total_pregame"],
            "starting_source": row["starting_source"],
        }
    )
    return pd.Series(values, name=board_name)


def _team_has_posted_five(
    team_abbr: str,
    lineups: dict[str, list[str]],
) -> bool:
    wanted = normalize_team_abbreviation(team_abbr)
    for key, names in lineups.items():
        if normalize_team_abbreviation(key) == wanted and names:
            return True
    return False


def _starting_flag(
    *,
    board_name: str,
    hist_name: str,
    team_abbr: str,
    day: str,
    lineups: dict[str, list[str]],
    start_rate: float,
) -> tuple[float, str]:
    """RotoWire 1/0 when the team five is posted; else prior start-rate rule."""
    if _team_has_posted_five(team_abbr, lineups):
        posted = float(
            get_starter_status(board_name, day, lineups=lineups)
            or get_starter_status(hist_name, day, lineups=lineups)
        )
        return posted, "rotowire"
    # Tip-off / missing team pages drop fives; fall back to recent role.
    return (1.0 if start_rate >= 0.5 else 0.0), "start_rate"


def _opp_def_rating_prior(
    history: pd.DataFrame,
    opp_abbr: str,
    day: str,
) -> float:
    wanted = normalize_team_abbreviation(opp_abbr)
    labels = history["team_abbreviation"].map(normalize_team_abbreviation)
    games = (
        history.loc[labels.eq(wanted), ["game_id", "game_date", "team_def_rating"]]
        .drop_duplicates(["game_id"])
        .sort_values(["game_date", "game_id"])
    )
    if len(games) < 10:
        return float("nan")
    return float(
        pd.to_numeric(games["team_def_rating"], errors="coerce").tail(10).mean()
    )


def _back2back(player: pd.DataFrame, day: str) -> float:
    if player.empty:
        return 0.0
    last = pd.to_datetime(player["game_date"].iloc[-1], utc=True, errors="coerce")
    slate = pd.Timestamp(str(day)[:10], tz="UTC")
    if pd.isna(last):
        return 0.0
    return float((slate.normalize() - last.normalize()).days == 1)
