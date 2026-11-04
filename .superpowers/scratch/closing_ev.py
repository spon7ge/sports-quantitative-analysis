"""Score PrizePicks closing points lines at -119 and summarize settled results.

Uses the saved q50 booster and the walk-forward NB2 alpha already printed
in the points notebook (0.3327). Does not refit either.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from models.shared.points_given_minutes import nb_logpmf
from models.shared.points_nb import over_probability
from src.features.minutes import add_starter_features, add_total_stars_active
from src.features.points import add_points_features
from src.features.points.closing import (
    features_before,
    load_closing_points,
    match_player_ids,
)

SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
)
ALPHA = 0.3327
AMERICAN = -119
WIN_PROFIT = 100 / 119
DECIMAL = 219 / 119
BREAKEVEN = 119 / 219
OUT = ROOT / ".superpowers/scratch/closing_ev"


def build_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "starting" not in prepared.columns:
        start = (
            prepared["start_position"]
            if "start_position" in prepared.columns
            else ""
        )
        prepared["starting"] = (
            start.astype("string").str.strip().isin(["G", "F", "C"]).astype(int)
        )
    return add_total_stars_active(add_points_features(add_starter_features(prepared)))


def side_probabilities(mu, line, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Win, loss, and push probabilities for the over.

    A half-point line cannot push. An integer line pushes when the
    points total equals the line, and that stake is returned.
    """
    mean = np.asarray(mu, dtype=float)
    level = np.asarray(line, dtype=float)
    p_over = over_probability(mean, alpha, level)
    integer = np.isclose(level, np.rint(level))
    p_push = np.zeros(mean.shape, dtype=float)
    if integer.any():
        p_push[integer] = np.exp(nb_logpmf(level[integer], mean[integer], alpha))
    p_under = 1.0 - p_over - p_push
    return p_over, p_under, p_push


def stake_ev(win: np.ndarray, lose: np.ndarray) -> np.ndarray:
    """Expected profit per $1 stake at -119. A push returns the stake."""
    return win * WIN_PROFIT - lose


def main() -> None:
    raw_parts = [
        pd.read_parquet(
            ROOT / "data/silver/nba" / season / "regular_season/player_gamelogs.parquet"
        )
        for season in SEASONS
    ]
    raw = pd.concat(raw_parts, ignore_index=True)
    raw["starting"] = (
        raw["start_position"].astype("string").str.strip().isin(["G", "F", "C"]).astype(int)
    )
    bundle = joblib.load(ROOT / "models/saved_models/pts_nba_model_2026-10-26.joblib")
    model = bundle["quantile_models"]["q_0.50"]
    features = list(bundle["feature_names"])

    lines = match_player_ids(
        load_closing_points(ROOT / "data/props/prizepicks/nba/closing"),
        raw,
    )
    unmatched = int(lines["player_id"].isna().sum())
    raw_dates = pd.to_datetime(raw["game_date"]).dt.normalize()
    parts = []
    for slate, group in lines.dropna(subset=["player_id"]).groupby("slate_date", sort=True):
        featured = features_before(raw, slate, build_model_frame)
        overlap = [
            column
            for column in featured.columns
            if column in group.columns and column != "player_id"
        ]
        played = (
            raw.loc[raw_dates.eq(slate), ["player_id", "pts"]]
            .drop_duplicates("player_id")
            .rename(columns={"pts": "pts_played"})
        )
        joined = group.merge(
            featured.drop(columns=overlap), on="player_id", how="left"
        ).merge(played, on="player_id", how="left")
        parts.append(joined)
        print(
            f"{pd.Timestamp(slate).date()} rows {len(group)} "
            f"history {pd.to_datetime(featured['history_through']).max().date()}",
            flush=True,
        )

    closing = pd.concat(parts, ignore_index=True)
    missing = [column for column in features if column not in closing.columns]
    if missing:
        raise KeyError(f"missing features: {missing}")
    mu = np.clip(model.predict(closing[features]), 0.05, None)
    p_over, p_under, p_push = side_probabilities(mu, closing["line"].to_numpy(), ALPHA)
    ev_over = stake_ev(p_over, p_under)
    ev_under = stake_ev(p_under, p_over)
    take_over = ev_over >= ev_under
    closing["mu"] = mu
    closing["p_over"] = p_over
    closing["p_under"] = p_under
    closing["p_push"] = p_push
    closing["ev_over"] = ev_over
    closing["ev_under"] = ev_under
    closing["side"] = np.where(take_over, "over", "under")
    closing["p_side"] = np.where(take_over, p_over, p_under)
    closing["ev"] = np.maximum(ev_over, ev_under)
    settled = closing["pts_played"].notna()
    played = closing["pts_played"]
    won = np.where(
        take_over,
        played.gt(closing["line"]),
        played.lt(closing["line"]),
    )
    pushed = played.eq(closing["line"])
    closing["result"] = np.where(
        ~settled,
        "pending",
        np.where(pushed, "push", np.where(won, "win", "loss")),
    )
    profit = np.where(
        closing["result"].eq("win"),
        WIN_PROFIT,
        np.where(closing["result"].eq("loss"), -1.0, 0.0),
    )
    closing["profit"] = profit

    OUT.mkdir(parents=True, exist_ok=True)
    keep = [
        "slate_date",
        "player_name",
        "line",
        "mu",
        "p_over",
        "p_side",
        "side",
        "ev",
        "ev_over",
        "ev_under",
        "pts_played",
        "result",
        "profit",
        "history_through",
    ]
    closing[keep].to_parquet(OUT / "lines.parquet", index=False)
    summary = summarize(closing)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["headline"], indent=2), flush=True)


def summarize(closing: pd.DataFrame) -> dict:
    settled = closing.loc[closing["result"].ne("pending")].copy()
    decided = settled.loc[settled["result"].ne("push")].copy()
    plus = closing.loc[closing["ev"] > 0].copy()
    plus_settled = plus.loc[plus["result"].ne("pending")]
    plus_decided = plus_settled.loc[plus_settled["result"].ne("push")]

    def hit_rate(frame: pd.DataFrame) -> float | None:
        decided_rows = frame.loc[frame["result"].isin(["win", "loss"])]
        if decided_rows.empty:
            return None
        return float(decided_rows["result"].eq("win").mean())

    def roi(frame: pd.DataFrame) -> float | None:
        graded = frame.loc[frame["result"].ne("pending")]
        if graded.empty:
            return None
        return float(graded["profit"].sum() / len(graded))

    # Every over, flat $1, is the no-model baseline at the same price.
    over_profit = np.where(
        settled["pts_played"].gt(settled["line"]),
        WIN_PROFIT,
        np.where(settled["pts_played"].lt(settled["line"]), -1.0, 0.0),
    )

    confidence_edges = [0.50, 0.5434, 0.58, 0.62, 0.66, 0.70, 1.01]
    confidence_labels = [
        "50–54%",
        "54–58%",
        "58–62%",
        "62–66%",
        "66–70%",
        "70%+",
    ]
    over_edges = [0.0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.01]
    over_labels = ["under 35%", "35–45%", "45–50%", "50–55%", "55–65%", "65%+"]

    def buckets(frame: pd.DataFrame, column: str, edges: list[float], labels: list[str]) -> list[dict]:
        graded = frame.loc[frame["result"].isin(["win", "loss", "push"])].copy()
        if graded.empty:
            return []
        graded["bucket"] = pd.cut(
            graded[column],
            bins=edges,
            labels=labels,
            right=False,
            include_lowest=True,
        )
        rows = []
        for label, group in graded.groupby("bucket", observed=False):
            decided_rows = group.loc[group["result"].isin(["win", "loss"])]
            rows.append(
                {
                    "bucket": str(label),
                    "n": int(len(group)),
                    "decided": int(len(decided_rows)),
                    "wins": int(decided_rows["result"].eq("win").sum()),
                    "hit_rate": (
                        None
                        if decided_rows.empty
                        else float(decided_rows["result"].eq("win").mean())
                    ),
                    "mean_probability": (
                        None if group.empty else float(group[column].mean())
                    ),
                    "roi": None if group.empty else float(group["profit"].sum() / len(group)),
                }
            )
        return rows

    top = (
        closing.loc[closing["ev"] > 0]
        .sort_values(["ev", "slate_date"], ascending=[False, True])
        .head(20)
    )
    top_rows = []
    for row in top.itertuples(index=False):
        top_rows.append(
            {
                "slate_date": pd.Timestamp(row.slate_date).date().isoformat(),
                "player_name": row.player_name,
                "side": row.side,
                "line": float(row.line),
                "mu": round(float(row.mu), 2),
                "p_side": round(float(row.p_side), 4),
                "ev": round(float(row.ev), 4),
                "pts_played": None if pd.isna(row.pts_played) else float(row.pts_played),
                "result": row.result,
                "profit": round(float(row.profit), 4),
            }
        )

    headline = {
        "lines": int(len(closing)),
        "settled": int(len(settled)),
        "pending": int((closing["result"] == "pending").sum()),
        "pushes": int((settled["result"] == "push").sum()),
        "integer_lines": int(np.isclose(closing["line"], np.rint(closing["line"])).sum()),
        "plus_ev": int(len(plus)),
        "plus_ev_settled": int(len(plus_settled)),
        "favored_hit_rate": hit_rate(decided),
        "plus_ev_hit_rate": hit_rate(plus_decided),
        "plus_ev_roi": roi(plus_settled),
        "all_over_roi": float(np.sum(over_profit) / len(settled)) if len(settled) else None,
        "mean_favored_probability": float(decided["p_side"].mean()) if len(decided) else None,
        "breakeven": BREAKEVEN,
        "alpha": ALPHA,
        "american": AMERICAN,
    }
    return {
        "headline": headline,
        "confidence_buckets": buckets(
            decided, "p_side", confidence_edges, confidence_labels
        ),
        "over_buckets": buckets(settled, "p_over", over_edges, over_labels),
        "plus_ev_buckets": buckets(
            plus_settled, "p_side", confidence_edges, confidence_labels
        ),
        "top": top_rows,
    }


if __name__ == "__main__":
    main()
