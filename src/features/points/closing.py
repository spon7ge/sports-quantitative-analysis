"""Points features for a PrizePicks closing slate.

The production feature pipeline shifts by one row so the current game
is not an input. A closing line has no played row yet. Putting a blank
row on the slate date makes that shift drop the blank row, and the
window keeps every game the player had already played. Games on the
slate date itself stay out of the panel.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.points.pregame import OUTCOME_COLUMNS
from src.pipeline.silver.positions import player_name_keys

_TEXT_BLANK = frozenset({"start_position", "comment"})
# Previous-game market and opponent ids are not this slate's game.
_SLATE_BLANK = (
    "player_team_spread",
    "game_total",
    "matchup",
)
# No real opponent is attached to the closing file. A shared unused id
# keeps the column numeric and gives the slate row an empty opponent history.
_UNKNOWN_OPPONENT = -1


def features_before(
    history: pd.DataFrame,
    as_of: str | pd.Timestamp,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """One feature row per player using appearances strictly before ``as_of``.

    ``feature_fn`` is the usual shifted builder, such as
    ``build_model_frame``. The blank slate row is the row that shift drops.
    """
    cutoff = pd.Timestamp(as_of).normalize()
    dates = pd.to_datetime(history["game_date"], errors="coerce")
    prior = history.loc[dates.dt.normalize() < cutoff].copy()
    if prior.empty:
        return prior

    prior = prior.sort_values(
        ["player_id", "game_date", "game_id"],
        kind="mergesort",
    )
    through = prior.groupby("player_id", sort=False)["game_date"].max()
    sentinel = _blank_slate(prior.groupby("player_id", sort=False).tail(1))
    sentinel["game_date"] = cutoff
    prior["closing_row"] = 0
    sentinel["closing_row"] = 1
    featured = feature_fn(pd.concat([prior, sentinel], ignore_index=True))
    if "closing_row" not in featured.columns:
        raise KeyError("feature builder dropped closing_row")
    out = featured.loc[featured["closing_row"].eq(1)].copy()
    out["history_through"] = pd.to_datetime(out["player_id"].map(through))
    out["slate_date"] = cutoff
    # These read who played or started in this game. The slate has not
    # been played, and the closing file has no lineup.
    for column in (
        "starting",
        "starter_min_missing_teammates",
        "total_stars_active",
    ):
        out[column] = np.nan
    if not (out["history_through"] < cutoff).all():
        raise AssertionError("a closing row used a game on or after the slate date")
    return out.reset_index(drop=True)


def match_player_ids(
    closing: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``player_id`` using the silver name, suffix, and accent keys."""
    keys_to_id: dict[str, object] = {}
    ambiguous: set[str] = set()
    pairs = history[["player_id", "player_name"]].drop_duplicates()
    for player_id, name in pairs.itertuples(index=False):
        for key in player_name_keys(name):
            previous = keys_to_id.get(key)
            if previous is not None and previous != player_id:
                ambiguous.add(key)
            else:
                keys_to_id[key] = player_id
    for key in ambiguous:
        keys_to_id.pop(key, None)

    def resolve(name: object) -> object:
        for key in player_name_keys(name):
            if key in keys_to_id:
                return keys_to_id[key]
        return np.nan

    matched = closing.copy()
    matched["player_id"] = [resolve(name) for name in matched["player_name"]]
    return matched


def load_closing_points(directory: str | Path) -> pd.DataFrame:
    """Standard PrizePicks points lines from ``prizepicks_nba_closing_*.json``."""
    rows: list[dict] = []
    for path in sorted(Path(directory).glob("prizepicks_nba_closing_*.json")):
        payload = json.loads(path.read_text())
        slate = pd.Timestamp(payload["date"]).normalize()
        for player in payload.get("players", []):
            for line in player.get("lines", []):
                standard_points = (
                    line.get("stat_type") == "Points"
                    and line.get("odds_type") == "standard"
                )
                if not standard_points:
                    continue
                rows.append(
                    {
                        "slate_date": slate,
                        "player_name": player.get("player"),
                        "line": float(line["line_score"]),
                    }
                )
    frame = pd.DataFrame(rows, columns=["slate_date", "player_name", "line"])
    if not frame.empty:
        frame["slate_date"] = pd.to_datetime(frame["slate_date"])
    return frame


def _blank_slate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    highest = pd.to_numeric(result["game_id"], errors="coerce").max()
    start = 0 if pd.isna(highest) else int(highest)
    result["game_id"] = np.arange(len(result)) + start + 1
    for column in (*OUTCOME_COLUMNS, *_SLATE_BLANK):
        if column not in result.columns:
            continue
        if column in _TEXT_BLANK:
            result[column] = ""
        else:
            result[column] = np.nan
    if "opp_team_id" in result.columns:
        result["opp_team_id"] = _UNKNOWN_OPPONENT
    return result
