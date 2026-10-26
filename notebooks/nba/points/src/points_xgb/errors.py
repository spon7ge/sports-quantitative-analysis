"""Holdout misses. Score, fouls, and plus/minus are post-game context."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from points_xgb.config import TARGET_COL

_CONTEXT = [
    "game_id",
    "player_id",
    "team_abbreviation",
    "matchup",
    "wl",
    "pf",
    "pfd",
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
    TARGET_COL,
    "minutes",
    "pts",
    "q50",
    "error",
    "band",
    "pf",
    "pfd",
    "starting",
    "plus_minus",
    "margin",
    "pts_per_min_lag_1",
    "season_pts_per_min_mean",
    "abs_spread",
    "ot",
    "hint",
]
_LABELS = {
    "game_date": "date",
    "player_name": "player",
    "matchup": "game",
    TARGET_COL: "actual_rate",
    "q50": "pred_q50",
    "error": "actual − q50",
    "band": "q10–q90",
    "pf": "fouls",
    "pfd": "fouls_drawn",
    "plus_minus": "+/-",
    "pts_per_min_lag_1": "last_game_rate",
    "season_pts_per_min_mean": "season_rate",
    "abs_spread": "pregame_|spread|",
}


def build_miss_frame(
    source: pd.DataFrame,
    ppm_holdout: pd.DataFrame,
    preds_ho: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Join holdout predictions to post-game context for display only."""
    context = source[_CONTEXT].drop_duplicates(["game_id", "player_id"])
    team_minutes = (
        source.groupby(["game_id", "team_abbreviation"], as_index=False)["minutes"]
        .sum()
        .rename(columns={"minutes": "team_minutes"})
    )
    miss = ppm_holdout.reset_index(drop=True).copy()
    miss["q10"] = np.asarray(preds_ho["q_0.10"], dtype=float)
    miss["q50"] = np.asarray(preds_ho["q_0.50"], dtype=float)
    miss["q90"] = np.asarray(preds_ho["q_0.90"], dtype=float)
    miss["error"] = miss[TARGET_COL] - miss["q50"]
    miss["abs_error"] = miss["error"].abs()
    miss = miss.merge(context, on=["game_id", "player_id"], how="left")
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


def miss_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns the notebook prints for a miss list."""
    return frame[_SHOW].rename(columns=_LABELS).round(1)


def rotation_team_games(miss: pd.DataFrame) -> pd.DataFrame:
    """Team-games ranked by how many rotation rates missed by 0.25+."""
    rotation = miss["q50"] >= 0.45
    off_players = (
        miss.loc[rotation & (miss["abs_error"] >= 0.25)]
        .sort_values("abs_error", ascending=False)
        .groupby(["game_id", "team_abbreviation"])
        .apply(_player_list, include_groups=False)
        .rename("players_off_by_0.25")
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
            n_off_025=("abs_error", lambda series: int((series >= 0.25).sum())),
            mean_abs_error=("abs_error", "mean"),
            margin=("margin", "mean"),
            ot=("ot", "max"),
        )
        .merge(off_players, on=["game_id", "team_abbreviation"], how="left")
        .sort_values(["n_off_025", "mean_abs_error"], ascending=False)
    )
    return games


def _hints(miss: pd.DataFrame) -> pd.Series:
    sat_early = miss["error"] <= -0.20
    played_more = miss["error"] >= 0.20
    blowout = miss["margin"].abs() >= 15
    outside_80 = (miss[TARGET_COL] < miss["q10"]) | (miss[TARGET_COL] > miss["q90"])
    return pd.Series(
        np.select(
            [
                (miss["pf"] >= 6) & sat_early,
                (miss["pf"] >= 5) & sat_early,
                blowout & sat_early & (miss["starting"] == 1),
                blowout & played_more & (miss["starting"] == 0),
                miss["ot"] & played_more,
                (miss["starting"] == 0)
                & (miss["pts_per_min_lag_1"] >= 0.50)
                & (miss["error"] <= -0.25),
                (miss["starting"] == 1)
                & (miss["pts_per_min_lag_1"].fillna(0) < 0.35)
                & (miss["error"] >= 0.25),
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


def _player_list(group: pd.DataFrame) -> str:
    return ", ".join(
        (
            f"{row.player_name} ({row.minutes:.0f} min, {row.error:+.2f}, "
            f"{int(row.pf)} PF)"
        )
        for row in group.itertuples()
    )
