"""Holdout misses with post-game context.

Score, fouls, and plus/minus explain a miss. They are not model inputs.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

_CONTEXT = [
    "game_id",
    "player_id",
    "team_abbreviation",
    "matchup",
    "wl",
    "pf",
    "pfd",
    "pts",
    "plus_minus",
    "team_pts",
    "opp_pts",
    "abs_spread",
]

_SHOW = [
    "game_date",
    "player_name",
    "matchup",
    "score",
    "wl",
    "minutes",
    "q50",
    "error",
    "band",
    "pf",
    "pfd",
    "starting",
    "pts",
    "plus_minus",
    "margin",
    "min_lag_1",
    "abs_spread",
    "ot",
    "hint",
]

_LABELS = {
    "game_date": "date",
    "player_name": "player",
    "matchup": "game",
    "minutes": "actual_min",
    "q50": "pred_q50",
    "error": "actual − q50",
    "band": "q10–q90",
    "pf": "fouls",
    "pfd": "fouls_drawn",
    "plus_minus": "+/-",
    "min_lag_1": "last_game_min",
    "abs_spread": "pregame_|spread|",
}


def build_miss_frame(
    history: pd.DataFrame,
    holdout: pd.DataFrame,
    preds: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Attach post-game context and a short hint to each holdout row.

    ``pts`` is already on the holdout frame. Merging it again would rename
    both copies, so overlapping context columns other than the keys are
    dropped before the merge.
    """
    context = history[_CONTEXT].drop_duplicates(["game_id", "player_id"])
    team_minutes = (
        history.groupby(["game_id", "team_abbreviation"], as_index=False)["minutes"]
        .sum()
        .rename(columns={"minutes": "team_minutes"})
    )
    miss = holdout.reset_index(drop=True).copy()
    miss["q10"] = np.asarray(preds["q_0.10"], dtype=float)
    miss["q50"] = np.asarray(preds["q_0.50"], dtype=float)
    miss["q90"] = np.asarray(preds["q_0.90"], dtype=float)
    miss["error"] = miss["minutes"] - miss["q50"]
    miss["abs_error"] = miss["error"].abs()
    overlap = context.columns.intersection(miss.columns).difference(
        ["game_id", "player_id"]
    )
    miss = miss.merge(
        context.drop(columns=list(overlap)),
        on=["game_id", "player_id"],
        how="left",
    )
    miss = miss.merge(team_minutes, on=["game_id", "team_abbreviation"], how="left")
    miss["game_date"] = pd.to_datetime(miss["game_date"]).dt.date
    miss["score"] = (
        miss["team_pts"].round().astype("Int64").astype("string")
        + "-"
        + miss["opp_pts"].round().astype("Int64").astype("string")
    )
    miss["margin"] = miss["team_pts"] - miss["opp_pts"]
    miss["ot"] = miss["team_minutes"] >= 260
    miss["band"] = (
        miss["q10"].round().astype("Int64").astype("string")
        + "–"
        + miss["q90"].round().astype("Int64").astype("string")
    )
    miss["hint"] = _hints(miss)
    return miss


def _hints(miss: pd.DataFrame) -> pd.Series:
    sat_early = miss["error"] <= -8
    played_more = miss["error"] >= 8
    blowout = miss["margin"].abs() >= 15
    outside_80 = (miss["minutes"] < miss["q10"]) | (miss["minutes"] > miss["q90"])
    return pd.Series(
        np.select(
            [
                (miss["pf"] >= 6) & sat_early,
                (miss["pf"] >= 5) & sat_early,
                blowout & sat_early & (miss["starting"] == 1),
                blowout & played_more & (miss["starting"] == 0),
                miss["ot"] & played_more,
                (miss["starting"] == 0)
                & (miss["min_lag_1"] >= 20)
                & (miss["error"] <= -10),
                (miss["starting"] == 1)
                & (miss["min_lag_1"].fillna(0) < 15)
                & (miss["error"] >= 10),
                outside_80,
            ],
            [
                "fouled out",
                "foul trouble",
                "blowout rest",
                "garbage time",
                "overtime",
                "unexpected bench",
                "unexpected start",
                "outside 10–90",
            ],
            default="",
        ),
        index=miss.index,
    )


def miss_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[_SHOW].rename(columns=_LABELS).round(1)


def rotation_team_games(miss: pd.DataFrame) -> pd.DataFrame:
    """Team-games ranked by how many q50 ≥ 15 players missed by 10+ minutes."""
    rotation = miss["q50"] >= 15
    off_players = (
        miss.loc[rotation & (miss["abs_error"] >= 10)]
        .sort_values("abs_error", ascending=False)
        .groupby(["game_id", "team_abbreviation"])
        .apply(_player_list, include_groups=False)
        .rename("players_off_by_10")
        .reset_index()
    )
    games = (
        miss.loc[rotation]
        .groupby(
            ["game_id", "game_date", "team_abbreviation", "matchup", "score", "wl"],
            as_index=False,
        )
        .agg(
            n_rotation=("player_id", "size"),
            n_off_10=("abs_error", lambda values: int((values >= 10).sum())),
            mean_abs_error=("abs_error", "mean"),
            margin=("margin", "mean"),
            ot=("ot", "max"),
        )
        .merge(off_players, on=["game_id", "team_abbreviation"], how="left")
        .sort_values(["n_off_10", "mean_abs_error"], ascending=False)
    )
    return games.rename(columns={"game_date": "date", "matchup": "game"}).round(1)


def _player_list(group: pd.DataFrame) -> str:
    return ", ".join(
        f"{row.player_name} ({row.error:+.0f}, {int(row.pf)} PF)"
        for row in group.itertuples()
    )
