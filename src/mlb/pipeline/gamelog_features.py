"""Vectorized starter-history features from game logs (no Statcast)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mlb import FEATURE_SET_VERSION
from src.mlb.config import MlbConfig
from src.mlb.models.shrinkage import shrink_rate
from src.mlb.schemas import FEATURE_ROW_COLUMNS, coerce_frame, empty_frame


def _prior_window_sums(
    starts: pd.DataFrame, days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Strikeouts and BF in ``(event_time - days, event_time)`` per pitcher."""
    k_out = np.zeros(len(starts), dtype=float)
    bf_out = np.zeros(len(starts), dtype=float)
    for _, grp in starts.groupby("pitcher_id", sort=False):
        order = np.argsort(pd.to_datetime(grp["event_time_utc"], utc=True).to_numpy())
        idx = grp.index.to_numpy()[order]
        times = pd.to_datetime(grp.loc[idx, "event_time_utc"], utc=True).to_numpy()
        ks = pd.to_numeric(grp.loc[idx, "strikeouts"], errors="coerce").to_numpy(
            dtype=float
        )
        bfs = pd.to_numeric(grp.loc[idx, "batters_faced"], errors="coerce").to_numpy(
            dtype=float
        )
        ks = np.nan_to_num(ks)
        bfs = np.nan_to_num(bfs)
        left = 0
        c_k = 0.0
        c_bf = 0.0
        horizon = np.timedelta64(int(days), "D")
        for i, pos in enumerate(idx):
            cutoff = times[i]
            lo = cutoff - horizon
            while left < i and times[left] <= lo:
                c_k -= ks[left]
                c_bf -= bfs[left]
                left += 1
            k_out[pos] = c_k
            bf_out[pos] = c_bf
            c_k += ks[i]
            c_bf += bfs[i]
    return k_out, bf_out


def build_gamelog_feature_rows(
    tables: dict[str, pd.DataFrame],
    config: MlbConfig,
) -> pd.DataFrame:
    """Lagged K/BF and workload features from ``pitcher_starts`` only."""
    pregame = tables.get("pregame_snapshots")
    starts = tables.get("pitcher_starts")
    if pregame is None or pregame.empty or starts is None or starts.empty:
        return empty_frame(FEATURE_ROW_COLUMNS)

    frame = starts.copy().reset_index(drop=True)
    frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc"], utc=True)
    frame["scheduled_start_utc"] = pd.to_datetime(
        frame.get("scheduled_start_utc", frame["event_time_utc"]), utc=True
    )
    frame = frame.sort_values(["pitcher_id", "event_time_utc"]).reset_index(drop=True)

    grouped = frame.groupby("pitcher_id", sort=False)
    shifted_bf = grouped["batters_faced"].shift(1)
    shifted_pitches = grouped["pitches"].shift(1)
    prev_start = grouped["scheduled_start_utc"].shift(1)
    rest = (
        frame["scheduled_start_utc"] - pd.to_datetime(prev_start, utc=True)
    ).dt.total_seconds() / 86400.0

    for window in (3, 5, 10):
        frame[f"bf_per_start_{window}"] = grouped["batters_faced"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        frame[f"pitches_per_start_{window}"] = grouped["pitches"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        frame[f"outs_per_start_{window}"] = grouped["outs"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
    frame["pitches_last_start"] = shifted_pitches
    frame["rest_days"] = rest
    frame["expected_bf_oof"] = frame["bf_per_start_5"]
    frame["expected_pitches_oof"] = frame["pitches_per_start_5"]
    frame["expected_outs_oof"] = frame["outs_per_start_5"]
    frame["bf_sd_oof"] = grouped["batters_faced"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std()
    )
    early = (shifted_bf < float(config.early_exit_bf)).astype(float)
    frame["p_early_exit_oof"] = early.groupby(frame["pitcher_id"]).transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )

    k60, bf60 = _prior_window_sums(frame, 60)
    k365, bf365 = _prior_window_sums(frame, 365)
    ordered = frame.sort_values("event_time_utc")
    league_k = ordered["strikeouts"].cumsum().shift(1).fillna(0.0)
    league_bf = ordered["batters_faced"].cumsum().shift(1).fillna(0.0)
    prior_mean = (league_k / league_bf.replace(0, np.nan)).reindex(frame.index)
    prior_mean = prior_mean.fillna(0.22)
    strength = float(config.pitcher_k_prior_strength)
    prior_arr = np.asarray(prior_mean, dtype=float)
    frame["k_bf_shrunk_60"] = shrink_rate(k60, bf60, prior_arr, strength)
    frame["k_bf_shrunk_365"] = shrink_rate(k365, bf365, prior_arr, strength)
    frame["n_eff_k_bf_60"] = bf60
    frame["n_eff_k_bf_365"] = bf365

    season_tot = frame.groupby(["pitcher_id", "season"], as_index=False).agg(
        season_k=("strikeouts", "sum"),
        season_bf=("batters_faced", "sum"),
    )
    prev1 = season_tot.rename(
        columns={
            "season": "join_season",
            "season_k": "k_m1",
            "season_bf": "bf_m1",
        }
    )
    prev2 = season_tot.rename(
        columns={
            "season": "join_season",
            "season_k": "k_m2",
            "season_bf": "bf_m2",
        }
    )
    frame["_s1"] = frame["season"] - 1
    frame["_s2"] = frame["season"] - 2
    frame = frame.merge(
        prev1,
        left_on=["pitcher_id", "_s1"],
        right_on=["pitcher_id", "join_season"],
        how="left",
    ).drop(columns=["join_season"])
    frame = frame.merge(
        prev2,
        left_on=["pitcher_id", "_s2"],
        right_on=["pitcher_id", "join_season"],
        how="left",
    ).drop(columns=["join_season", "_s1", "_s2"])
    prior_k = frame["k_m1"].fillna(0.0) + frame["k_m2"].fillna(0.0)
    prior_bf = frame["bf_m1"].fillna(0.0) + frame["bf_m2"].fillna(0.0)
    frame["k_bf_shrunk_prior2"] = shrink_rate(
        prior_k.to_numpy(dtype=float),
        prior_bf.to_numpy(dtype=float),
        prior_arr,
        strength,
    )
    frame["n_eff_k_bf_prior2"] = prior_bf

    keys = pregame[["pitcher_id", "game_pk", "prediction_cutoff_utc", "pregame_id"]]
    merged = keys.merge(frame, on=["pitcher_id", "game_pk"], how="left")
    merged["feature_set_version"] = getattr(
        config, "feature_set_version", FEATURE_SET_VERSION
    )
    merged["pitcher_throws_L"] = (
        merged.get("pitcher_hand", pd.Series("", index=merged.index))
        .astype(str)
        .str.upper()
        .eq("L")
        .astype(float)
    )
    if "is_home" in merged.columns:
        merged["is_home"] = pd.to_numeric(merged["is_home"], errors="coerce")
    merged["is_opener"] = 0.0
    merged["is_il_return"] = 0.0
    merged["is_restricted"] = 0.0
    merged["missing_pitcher_k"] = (merged["n_eff_k_bf_365"] < 20).astype("int64")
    merged["missing_plate_discipline"] = 1
    merged["missing_workload"] = merged["bf_per_start_3"].isna().astype("int64")
    merged["missing_opponent"] = 1
    merged["missing_lineup"] = 1
    merged["missing_stuff"] = 1
    merged["missing_role"] = 0
    merged["max_input_event_time_utc"] = merged["event_time_utc"]
    merged["max_source_ingestion_time_utc"] = merged["ingested_at_utc"]
    for column, dtype in FEATURE_ROW_COLUMNS.items():
        if column in merged.columns:
            continue
        if dtype == "float64":
            merged[column] = np.nan
        elif dtype == "int64":
            merged[column] = 0
        elif dtype == "string":
            merged[column] = ""
    return coerce_frame(merged, FEATURE_ROW_COLUMNS)
