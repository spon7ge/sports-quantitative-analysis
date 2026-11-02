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
_LABELS = {
    "game_date": "date",
    "player_name": "player",
    "matchup": "game",
    "q50": "pred_q50",
    "error": "actual − q50",
    "band": "q10–q90",
    "pf": "fouls",
    "pfd": "fouls_drawn",
    "plus_minus": "+/-",
    "abs_spread": "pregame_|spread|",
}
_HINT_CUTS = {
    "rate": {
        "sat": -0.20,
        "more": 0.20,
        "bench_lag": 0.50,
        "bench_err": -0.25,
        "start_lag": 0.35,
        "start_err": 0.25,
    },
    "points": {
        "sat": -5.0,
        "more": 5.0,
        "bench_lag": 15.0,
        "bench_err": -8.0,
        "start_lag": 8.0,
        "start_err": 8.0,
    },
}


_RESIDUAL_WINDOWS = (5, 10, 20)


def prior_residual_features(
    frame: pd.DataFrame,
    *,
    actual_col: str = "pts",
    pred_col: str = "q50",
    player_col: str = "player_id",
    date_col: str = "game_date",
    game_col: str = "game_id",
) -> pd.DataFrame:
    """Prior residual median, MAD, and count for each row.

    Residual is actual minus prediction. A window is missing until that
    many earlier games exist. ``residual_count`` is how many earlier
    residuals the player has. The current game is not one of them.
    ``residual_mad_10`` is the median absolute deviation of the last 10
    residuals around their own median.
    """
    required = [player_col, date_col, game_col, actual_col, pred_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"residual history is missing {missing}")
    work = frame.loc[:, required].copy()
    work["_row"] = np.arange(len(frame))
    actual = work[actual_col].to_numpy(dtype=float)
    predicted = work[pred_col].to_numpy(dtype=float)
    residual = actual - predicted
    if residual.size and not np.isfinite(residual).all():
        raise ValueError("residuals must be finite")
    work["_residual"] = residual
    ordered = work.sort_values(
        [player_col, date_col, game_col, "_row"], kind="mergesort"
    )
    medians = {window: np.full(len(frame), np.nan) for window in _RESIDUAL_WINDOWS}
    mad = np.full(len(frame), np.nan)
    count = np.zeros(len(frame), dtype=int)
    for _player, part in ordered.groupby(player_col, sort=False):
        values = part["_residual"].to_numpy(dtype=float)
        rows = part["_row"].to_numpy()
        for index in range(len(values)):
            count[rows[index]] = index
            history = values[:index]
            for window in _RESIDUAL_WINDOWS:
                if index < window:
                    continue
                medians[window][rows[index]] = float(np.median(history[-window:]))
            if index >= 10:
                sample = history[-10:]
                center = float(np.median(sample))
                mad[rows[index]] = float(np.median(np.abs(sample - center)))
    return pd.DataFrame(
        {
            "residual_median_5": medians[5],
            "residual_median_10": medians[10],
            "residual_median_20": medians[20],
            "residual_mad_10": mad,
            "residual_count": count,
        },
        index=frame.index,
    )


def _residual_rows(
    frame: pd.DataFrame,
    actual_col: str,
    pred_col: str = "q50",
) -> pd.DataFrame:
    required = ["player_id", "game_id", "game_date", actual_col, pred_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"residual history is missing {missing}")
    return pd.DataFrame(
        {
            "player_id": frame["player_id"].to_numpy(),
            "game_id": frame["game_id"].to_numpy(),
            "game_date": pd.to_datetime(frame["game_date"]),
            "pts": frame[actual_col].to_numpy(dtype=float),
            "q50": frame[pred_col].to_numpy(dtype=float),
        }
    )


def build_miss_frame(
    source: pd.DataFrame,
    ppm_holdout: pd.DataFrame,
    preds_ho: Mapping[str, np.ndarray],
    *,
    target_col: str = TARGET_COL,
    lag_col: str = "pts_per_min_lag_1",
    unit: str = "rate",
    history: pd.DataFrame | None = None,
    history_pred_col: str = "q50",
) -> pd.DataFrame:
    """Join holdout predictions to post-game context for display only.

    ``history`` holds earlier out-of-sample predictions, with ``target_col``
    and ``history_pred_col``. Those rows extend each player's residual
    trail into the holdout. A history row that is also in the holdout is
    ignored so a game is not its own prior.
    """
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
    miss["error"] = miss[target_col] - miss["q50"]
    miss["abs_error"] = miss["error"].abs()
    residual_rows = _residual_rows(miss, target_col)
    if history is not None:
        earlier = _residual_rows(history, target_col, pred_col=history_pred_col)
        holdout_keys = set(zip(residual_rows["player_id"], residual_rows["game_id"]))
        earlier_keys = list(zip(earlier["player_id"], earlier["game_id"]))
        earlier = earlier.loc[[key not in holdout_keys for key in earlier_keys]]
        residual_rows = pd.concat([earlier, residual_rows], ignore_index=True)
    residual_features = prior_residual_features(residual_rows)
    for column in residual_features.columns:
        miss[column] = residual_features[column].iloc[-len(miss) :].to_numpy()
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
    miss["hint"] = _hints(
        miss,
        target_col=target_col,
        lag_col=lag_col,
        unit=unit,
    )
    return miss


def miss_view(
    frame: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    lag_col: str = "pts_per_min_lag_1",
    season_col: str = "season_pts_per_min_mean",
    unit: str = "rate",
) -> pd.DataFrame:
    """Columns the notebook prints for a miss list."""
    show = [
        "game_date",
        "player_name",
        "matchup",
        "score",
        "wl",
        target_col,
        "minutes",
        "q50",
        "error",
        "residual_median_5",
        "residual_median_10",
        "residual_median_20",
        "residual_mad_10",
        "residual_count",
        "band",
        "pf",
        "pfd",
        "starting",
        "plus_minus",
        "margin",
        lag_col,
        season_col,
        "abs_spread",
        "ot",
        "hint",
    ]
    if target_col != "pts":
        show.insert(show.index("minutes"), "pts")
    labels = {
        **_LABELS,
        target_col: "actual_rate" if unit == "rate" else "actual",
        lag_col: "last_game_rate" if unit == "rate" else "last_game",
        season_col: "season_rate" if unit == "rate" else "season_points",
    }
    shown = frame[show].rename(columns=labels)
    counts = shown["residual_count"]
    shown = shown.round(1)
    shown["residual_count"] = counts.astype(int)
    return shown


def rotation_team_games(
    miss: pd.DataFrame,
    *,
    rotation_min: float = 0.45,
    miss_cutoff: float = 0.25,
) -> pd.DataFrame:
    """Team-games ranked by how many rotation players missed the cutoff."""
    off_column = f"players_off_by_{miss_cutoff:g}"
    count_name = "n_off_025" if miss_cutoff == 0.25 else f"n_off_{miss_cutoff:g}"
    rotation = miss["q50"] >= rotation_min
    off_players = (
        miss.loc[rotation & (miss["abs_error"] >= miss_cutoff)]
        .sort_values("abs_error", ascending=False)
        .groupby(["game_id", "team_abbreviation"])
        .apply(_player_list, include_groups=False)
        .rename(off_column)
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
            **{
                count_name: (
                    "abs_error",
                    lambda series: int((series >= miss_cutoff).sum()),
                )
            },
            mean_abs_error=("abs_error", "mean"),
            margin=("margin", "mean"),
            ot=("ot", "max"),
        )
        .merge(off_players, on=["game_id", "team_abbreviation"], how="left")
        .sort_values([count_name, "mean_abs_error"], ascending=False)
    )
    return games


def _hints(
    miss: pd.DataFrame,
    *,
    target_col: str,
    lag_col: str,
    unit: str,
) -> pd.Series:
    cuts = _HINT_CUTS[unit]
    sat_early = miss["error"] <= cuts["sat"]
    played_more = miss["error"] >= cuts["more"]
    blowout = miss["margin"].abs() >= 15
    outside_80 = (miss[target_col] < miss["q10"]) | (miss[target_col] > miss["q90"])
    last_game = miss[lag_col]
    return pd.Series(
        np.select(
            [
                (miss["pf"] >= 6) & sat_early,
                (miss["pf"] >= 5) & sat_early,
                blowout & sat_early & (miss["starting"] == 1),
                blowout & played_more & (miss["starting"] == 0),
                miss["ot"] & played_more,
                (miss["starting"] == 0)
                & (last_game >= cuts["bench_lag"])
                & (miss["error"] <= cuts["bench_err"]),
                (miss["starting"] == 1)
                & (last_game.fillna(0) < cuts["start_lag"])
                & (miss["error"] >= cuts["start_err"]),
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
