"""Player trailing assists, start rate, rest, and home flag."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def add_player_features(work: pd.DataFrame) -> pd.DataFrame:
    result = work.copy()
    result["game_date"] = pd.to_datetime(
        result["game_date"], errors="coerce"
    )
    result["start_rate_10"] = np.nan
    result["ast_mean_10"] = np.nan
    result["ast_mean_20"] = np.nan
    result["assists_per_min_10"] = np.nan
    result["days_rest"] = np.nan
    result["is_home"] = _is_home(result)

    ordered = result.sort_values(
        ["player_id", "game_date", "game_id", "_builder_row"]
    )
    start_by_row = {}
    ast10_by_row = {}
    ast20_by_row = {}
    rate_by_row = {}
    rest_by_row = {}

    for _, group in ordered.groupby("player_id", sort=False):
        starts: deque[float] = deque(maxlen=10)
        asts10: deque[float] = deque(maxlen=10)
        asts20: deque[float] = deque(maxlen=20)
        pairs: deque[tuple[float, float]] = deque(maxlen=10)
        last_appearance_by_season: dict[object, pd.Timestamp] = {}
        for rec in group.to_dict("records"):
            row_id = rec["_builder_row"]
            minutes = rec["_minutes_obs"]
            ast = rec["_ast_obs"]
            appeared = np.isfinite(minutes) and minutes > 0
            if len(starts) == 0:
                start_by_row[row_id] = np.nan
            else:
                start_by_row[row_id] = float(np.mean(starts))
            if len(asts10) == 0:
                ast10_by_row[row_id] = np.nan
            else:
                ast10_by_row[row_id] = float(np.mean(asts10))
            if len(asts20) == 0:
                ast20_by_row[row_id] = np.nan
            else:
                ast20_by_row[row_id] = float(np.mean(asts20))
            if len(pairs) == 0:
                rate_by_row[row_id] = np.nan
            else:
                ast_sum = sum(item[0] for item in pairs)
                min_sum = sum(item[1] for item in pairs)
                rate_by_row[row_id] = (
                    np.nan if min_sum == 0 else ast_sum / min_sum
                )

            season = rec["season_year"]
            prev = last_appearance_by_season.get(season)
            if prev is None:
                rest_by_row[row_id] = np.nan
            else:
                rest_by_row[row_id] = float(
                    (rec["game_date"] - prev).days
                )

            if appeared:
                starts.append(_started(rec.get("start_position")))
                if np.isfinite(ast):
                    asts10.append(float(ast))
                    asts20.append(float(ast))
                    pairs.append((float(ast), float(minutes)))
                last_appearance_by_season[season] = rec["game_date"]

    result["start_rate_10"] = result["_builder_row"].map(start_by_row)
    result["ast_mean_10"] = result["_builder_row"].map(ast10_by_row)
    result["ast_mean_20"] = result["_builder_row"].map(ast20_by_row)
    result["assists_per_min_10"] = result["_builder_row"].map(rate_by_row)
    result["days_rest"] = result["_builder_row"].map(rest_by_row)
    return result


def _started(value: object) -> float:
    text = (
        ""
        if value is None
        or (isinstance(value, float) and np.isnan(value))
        else str(value).strip()
    )
    if text in {"", "nan", "<NA>", "None"}:
        return 0.0
    return 1.0


def _is_home(frame: pd.DataFrame) -> pd.Series:
    if "matchup" not in frame.columns:
        return pd.Series(
            np.nan, index=frame.index, dtype="float64"
        )
    return (
        frame["matchup"]
        .astype("string")
        .str.contains(r"\bvs\.", regex=True)
        .astype(float)
    )
