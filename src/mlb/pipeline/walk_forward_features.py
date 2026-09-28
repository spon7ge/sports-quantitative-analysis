"""Point-in-time strikeout walk-forward features built from earlier official dates only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mlb.models.shrinkage import shrink_rate

WORKLOAD_FEATURES = (
    "home_flag",
    "days_rest_capped",
    "no_prior_regular_start",
    "long_layoff_flag",
    "season_starts_prior",
    "pitcher_prior_regular_starts",
    "pitcher_bf_mean_last3_smoothed",
    "pitcher_bf_mean_last10_smoothed",
    "pitcher_outs_mean_last5_smoothed",
    "pitcher_pitches_mean_last5_smoothed",
)
NB_RATE_FEATURES = (
    "home_flag",
    "season_starts_prior",
    "pitcher_prior_regular_starts",
    "pitcher_prior_bf_last10",
    "pitcher_k_per_bf_last3_smoothed",
    "pitcher_k_per_bf_last10_smoothed",
    "pitcher_k_per_bf_season_to_date_smoothed",
    "opponent_k_rate_vs_starters_last10_smoothed",
    "opponent_k_rate_vs_starters_season_to_date_smoothed",
    "opponent_prior_starts_observed",
)
RAW_SUM_COLUMNS = (
    "pitcher_k_last3",
    "pitcher_bf_last3",
    "pitcher_k_last10",
    "pitcher_bf_last10",
    "pitcher_k_season",
    "pitcher_bf_season",
    "opponent_k_last10",
    "opponent_bf_last10",
    "opponent_k_season",
    "opponent_bf_season",
)
PRIOR_COLUMNS = (
    "league_k_per_bf_prior",
    "population_bf_mean",
    "population_outs_mean",
    "population_pitches_mean",
    "rest_median_capped",
)
REQUIRED_COLUMNS = (
    "pitcher_id",
    "game_pk",
    "game_date",
    "scheduled_start_utc",
    "season",
    "is_home",
    "opponent_team_id",
    "strikeouts",
    "batters_faced",
    "outs",
    "pitches",
)
OUTPUT_COLUMNS = tuple(
    dict.fromkeys(
        ("pitcher_id", "game_pk")
        + WORKLOAD_FEATURES
        + NB_RATE_FEATURES
        + RAW_SUM_COLUMNS
        + PRIOR_COLUMNS
    )
)


def _prepare(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    out = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.normalize()
    out["scheduled_start_utc"] = pd.to_datetime(out["scheduled_start_utc"], utc=True)
    return out.reset_index(drop=True)


def _previous_start_utc(earlier: pd.DataFrame) -> pd.Timestamp | None:
    """Latest scheduled start on the latest earlier ``game_date``."""
    if earlier.empty:
        return None
    last_date = earlier["game_date"].max()
    return earlier.loc[earlier["game_date"] == last_date, "scheduled_start_utc"].max()


def _rest_median(history: pd.DataFrame, rest_cap: float) -> float:
    """Median capped rest gap in ``history``; NaN when no finite gap exists.

    A missing ``scheduled_start_utc`` makes its gaps non-finite, so they are skipped.
    """
    capped: list[float] = []
    for _, starts in history.groupby("pitcher_id", sort=False):
        for date, sched in zip(starts["game_date"], starts["scheduled_start_utc"]):
            previous = _previous_start_utc(starts[starts["game_date"] < date])
            if previous is None or pd.isna(previous) or pd.isna(sched):
                continue
            gap = (sched - previous).total_seconds() / 86400.0
            if not np.isfinite(gap):
                continue
            capped.append(min(gap, rest_cap))
    if not capped:
        return float("nan")
    return float(np.median(capped))


def _smoothed_mean(values: pd.Series, mu: float, kappa: float) -> float:
    return float((values.sum() + kappa * mu) / (len(values) + kappa))


def feature_block(
    history: pd.DataFrame,
    block: pd.DataFrame,
    *,
    m: float,
    kappa: float,
    rest_cap: float,
) -> pd.DataFrame:
    """One feature row per ``block`` start.

    Priors come from ``history`` only. Raw sums use ``history`` plus block starts
    whose ``game_date`` is strictly earlier than the row's ``game_date``.
    """
    history = _prepare(history, "history").sort_values(
        ["game_date", "scheduled_start_utc"], kind="mergesort"
    )
    block = _prepare(block, "block")

    bf_total = float(history["batters_faced"].sum())
    if history.empty or bf_total <= 0:
        raise ValueError("history has no batters faced to form the league strikeout prior")
    league = float(history["strikeouts"].sum()) / bf_total
    bf_mu = float(history["batters_faced"].mean())
    outs_mu = float(history["outs"].mean())
    pitches_mu = float(history["pitches"].mean())
    rest_median = _rest_median(history, rest_cap)

    pool = pd.concat([history, block], ignore_index=True).sort_values(
        ["game_date", "scheduled_start_utc"], kind="mergesort"
    )
    by_pitcher = {key: grp for key, grp in pool.groupby("pitcher_id", sort=False)}
    by_opponent = {key: grp for key, grp in pool.groupby("opponent_team_id", sort=False)}

    records = []
    for _, row in block.iterrows():
        date = row["game_date"]
        pitcher_starts = by_pitcher[row["pitcher_id"]]
        earlier = pitcher_starts[pitcher_starts["game_date"] < date]
        last3 = earlier.tail(3)
        last5 = earlier.tail(5)
        last10 = earlier.tail(10)
        season = earlier[earlier["season"] == row["season"]]

        previous = _previous_start_utc(earlier)
        if previous is None:
            if np.isnan(rest_median):
                raise ValueError(
                    "history has no rest gap to form the first-start rest median"
                )
            no_prior = 1
            layoff = 0
            rest_capped = rest_median
        else:
            no_prior = 0
            rest_raw = (row["scheduled_start_utc"] - previous).total_seconds() / 86400.0
            layoff = int(rest_raw >= rest_cap)
            rest_capped = min(rest_raw, rest_cap)

        k3, bf3 = float(last3["strikeouts"].sum()), float(last3["batters_faced"].sum())
        k10, bf10 = float(last10["strikeouts"].sum()), float(last10["batters_faced"].sum())
        ks, bfs = float(season["strikeouts"].sum()), float(season["batters_faced"].sum())
        k_rate_last10 = shrink_rate(k10, bf10, league, m)
        if len(season) == 0:
            k_rate_season = k_rate_last10
        else:
            k_rate_season = shrink_rate(ks, bfs, league, m)

        opp_starts = by_opponent[row["opponent_team_id"]]
        opp_earlier = opp_starts[opp_starts["game_date"] < date]
        opp_last10 = opp_earlier.tail(10)
        opp_season = opp_earlier[opp_earlier["season"] == row["season"]]
        ok10, obf10 = float(opp_last10["strikeouts"].sum()), float(opp_last10["batters_faced"].sum())
        oks, obfs = float(opp_season["strikeouts"].sum()), float(opp_season["batters_faced"].sum())
        opp_k_rate_last10 = shrink_rate(ok10, obf10, league, m)
        if len(opp_season) == 0:
            opp_k_rate_season = opp_k_rate_last10
        else:
            opp_k_rate_season = shrink_rate(oks, obfs, league, m)

        records.append(
            {
                "pitcher_id": row["pitcher_id"],
                "game_pk": row["game_pk"],
                "home_flag": int(row["is_home"]),
                "days_rest_capped": float(rest_capped),
                "no_prior_regular_start": no_prior,
                "long_layoff_flag": layoff,
                "season_starts_prior": len(season),
                "pitcher_prior_regular_starts": len(earlier),
                "pitcher_bf_mean_last3_smoothed": _smoothed_mean(last3["batters_faced"], bf_mu, kappa),
                "pitcher_bf_mean_last10_smoothed": _smoothed_mean(last10["batters_faced"], bf_mu, kappa),
                "pitcher_outs_mean_last5_smoothed": _smoothed_mean(last5["outs"], outs_mu, kappa),
                "pitcher_pitches_mean_last5_smoothed": _smoothed_mean(last5["pitches"], pitches_mu, kappa),
                "pitcher_prior_bf_last10": bf10,
                "pitcher_k_per_bf_last3_smoothed": shrink_rate(k3, bf3, league, m),
                "pitcher_k_per_bf_last10_smoothed": k_rate_last10,
                "pitcher_k_per_bf_season_to_date_smoothed": k_rate_season,
                "opponent_k_rate_vs_starters_last10_smoothed": opp_k_rate_last10,
                "opponent_k_rate_vs_starters_season_to_date_smoothed": opp_k_rate_season,
                "opponent_prior_starts_observed": len(opp_last10),
                "pitcher_k_last3": k3,
                "pitcher_bf_last3": bf3,
                "pitcher_k_last10": k10,
                "pitcher_bf_last10": bf10,
                "pitcher_k_season": ks,
                "pitcher_bf_season": bfs,
                "opponent_k_last10": ok10,
                "opponent_bf_last10": obf10,
                "opponent_k_season": oks,
                "opponent_bf_season": obfs,
                "league_k_per_bf_prior": league,
                "population_bf_mean": bf_mu,
                "population_outs_mean": outs_mu,
                "population_pitches_mean": pitches_mu,
                "rest_median_capped": rest_median,
            }
        )
    return pd.DataFrame.from_records(records, columns=list(OUTPUT_COLUMNS))
