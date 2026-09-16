"""One row per player/line/book at a fixed pre-tip snapshot.

Two nights cannot prove profitability. This module checks that pricing,
no-vig conversion, push-aware EV, and DNP/push settlement agree with
the quotes that were live before tip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.points.pregame import (
    appearances_before,
    build_pregame_points_features,
)
from src.models.odds import (
    american_to_decimal,
    american_to_implied_probability,
    remove_vig,
)
from src.models.xgboost_models.example_set import (
    ROOT,
    build_candidate_rows,
    filter_commence_after_as_of,
    load_player_points_quotes,
    load_silver_panel,
    snapshot_as_of,
)
from src.models.xgboost_models.artifact_bundle import load_joint_points_bundle
from src.models.xgboost_models.joint_simulation import (
    JointPointsSimulator,
    JointSimulation,
)
from src.models.xgboost_models.joint_variant_eval import shift_raw_to_postclip_mean
from src.models.xgboost_models.points import expected_role

EDGE_BUCKETS = ("<2%", "2–5%", "5–10%", ">10%")
LOW_HAT_P = 8.0
IMPLAUSIBLE_EDGE = 0.10
ASSUMED_TIP_HOUR_ET = 19
MARKET_KEY = ("BOOKMAKER", "NAME", "LINE", "COMMENCE_TIME")
ROLE_LABEL = {-1: "missing", 0: "bench", 1: "starter"}


def draw_line_probabilities(draws: np.ndarray, line: float) -> dict[str, float]:
    samples = np.asarray(draws, dtype=float)
    p_over = float(np.mean(samples > line))
    p_under = float(np.mean(samples < line))
    p_push = float(np.mean(samples == line))
    return {"p_over": p_over, "p_under": p_under, "p_push": p_push}


def market_over_probability(over_odds: float, under_odds: float) -> float:
    over_raw = american_to_implied_probability(over_odds)
    under_raw = american_to_implied_probability(under_odds)
    return float(over_raw / (over_raw + under_raw))


def expected_return_per_unit(
    *,
    p_win: float,
    p_lose: float,
    p_push: float,
    american_odds: float,
) -> float:
    """Expected profit per unit stake. Pushes return the stake."""
    profit_on_win = american_to_decimal(american_odds) - 1.0
    return float(p_win * profit_on_win + p_push * 0.0 - p_lose)


def settle_sides(
    *,
    pts: float | None,
    minutes: float | None,
    line: float,
) -> dict:
    minutes_num = np.nan if minutes is None else float(minutes)
    pts_num = np.nan if pts is None else float(pts)
    played = bool(np.isfinite(minutes_num) and minutes_num > 0 and np.isfinite(pts_num))
    if not played:
        return {
            "played": False,
            "status": "DNP",
            "over_result": "void",
            "under_result": "void",
            "actual_points": pts_num if np.isfinite(pts_num) else float("nan"),
        }
    if pts_num > line:
        over_result, under_result = "win", "loss"
    elif pts_num < line:
        over_result, under_result = "loss", "win"
    else:
        over_result, under_result = "push", "push"
    return {
        "played": True,
        "status": "played",
        "over_result": over_result,
        "under_result": under_result,
        "actual_points": pts_num,
    }


def edge_bucket(edge: float) -> str:
    value = float(edge)
    if value < 0.02:
        return "<2%"
    if value < 0.05:
        return "2–5%"
    if value < 0.10:
        return "5–10%"
    return ">10%"


def assumed_tip_utc(
    commence: object,
    *,
    hour: int = ASSUMED_TIP_HOUR_ET,
    timezone: str = "America/New_York",
) -> pd.Timestamp:
    local = pd.Timestamp(commence).tz_localize(None).normalize()
    return local.tz_localize(timezone).replace(hour=hour).tz_convert("UTC")


def select_snapshot_quotes(
    quotes: pd.DataFrame,
    *,
    minutes_before_tip: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One over/under pair per player/line/book at a fixed pre-tip cutoff.

    Closing is the last complete pair at or before assumed tip. When the
    file is a single pull, snapshot and closing are the same quotes.
    """
    if quotes.empty:
        empty = quotes.copy()
        empty["LAST_UPDATE"] = pd.to_datetime(empty.get("LAST_UPDATE"), utc=True)
        return empty, empty.copy()

    frame = quotes.copy()
    frame["_last"] = pd.to_datetime(frame["LAST_UPDATE"], utc=True)
    snapshot_parts: list[pd.DataFrame] = []
    closing_parts: list[pd.DataFrame] = []
    grouped = frame.groupby(list(MARKET_KEY), dropna=False)
    for _, group in grouped:
        complete = _complete_quote_times(group)
        if not complete:
            continue
        tip = assumed_tip_utc(group["COMMENCE_TIME"].iloc[0])
        cutoff = tip - pd.Timedelta(minutes=int(minutes_before_tip))
        stamps = pd.DatetimeIndex(complete)
        before_cutoff = stamps[stamps <= cutoff]
        before_tip = stamps[stamps <= tip]
        analysis_ts = (
            before_cutoff.max()
            if len(before_cutoff)
            else (before_tip.max() if len(before_tip) else stamps.max())
        )
        closing_ts = before_tip.max() if len(before_tip) else stamps.max()
        snapshot_parts.append(group.loc[group["_last"].eq(analysis_ts)].copy())
        closing_parts.append(group.loc[group["_last"].eq(closing_ts)].copy())

    snapshot = (
        pd.concat(snapshot_parts, ignore_index=True)
        if snapshot_parts
        else frame.iloc[0:0].copy()
    )
    closing = (
        pd.concat(closing_parts, ignore_index=True)
        if closing_parts
        else frame.iloc[0:0].copy()
    )
    snapshot["LAST_UPDATE"] = snapshot["_last"]
    closing["LAST_UPDATE"] = closing["_last"]
    return (
        snapshot.drop(columns=["_last"]),
        closing.drop(columns=["_last"]),
    )


def _complete_quote_times(group: pd.DataFrame) -> list[pd.Timestamp]:
    times: list[pd.Timestamp] = []
    for ts, sub in group.groupby("_last", sort=True):
        sides = set(
            sub["OVER/UNDER"].astype("string").str.strip().str.lower()
        )
        if sides >= {"over", "under"}:
            times.append(pd.Timestamp(ts))
    return times


def reconstruct_raw_points(sim: JointSimulation) -> np.ndarray:
    shock_center = float(np.mean(sim.minute_draws))
    return (
        sim.hat_p
        + sim.g
        + sim.beta * (sim.minute_draws - shock_center)
        + sim.epsilon_draws
    )


def clip_mean_point_draws(sim: JointSimulation) -> np.ndarray:
    raw = reconstruct_raw_points(sim)
    target = np.array([max(0.0, float(sim.hat_p + sim.g))])
    shifted = shift_raw_to_postclip_mean(raw.reshape(1, -1), target)
    return np.maximum(0.0, shifted[0])


def _empty_integrity() -> dict:
    return {
        "n_rows": 0,
        "odds_after_tip": 0,
        "features_after_or_on_game": 0,
        "dnp_voids": 0,
        "dnp_graded_as_zero": 0,
        "half_point_pushes": 0,
        "implausible_over_edges_gt_10pct": 0,
        "implausible_under_edges_gt_10pct": 0,
    }


def summarize_integrity(rows: pd.DataFrame) -> dict:
    if rows.empty or "played" not in rows.columns:
        return _empty_integrity()
    played = rows["played"].fillna(False).astype(bool)
    dnp = ~played
    over = rows["over_result"].astype("string")
    under = rows["under_result"].astype("string")
    line = pd.to_numeric(rows["line"], errors="coerce")
    half = ~np.isclose(line % 1, 0.0)
    dnp_graded = dnp & (
        over.isin(["win", "loss"]) | under.isin(["win", "loss"])
    )
    half_push = played & half & (
        over.eq("push") | under.eq("push")
    )
    over_edge = (
        pd.to_numeric(rows["current_over_edge"], errors="coerce")
        if "current_over_edge" in rows.columns
        else pd.Series(dtype=float)
    )
    under_edge = (
        pd.to_numeric(rows["current_under_edge"], errors="coerce")
        if "current_under_edge" in rows.columns
        else pd.Series(dtype=float)
    )
    return {
        "n_rows": int(len(rows)),
        "odds_after_tip": int((~rows["odds_before_tip"].fillna(False)).sum()),
        "features_after_or_on_game": int(
            (~rows["features_as_of_before_game"].fillna(False)).sum()
        )
        if "features_as_of_before_game" in rows
        else 0,
        "dnp_voids": int(dnp.sum()),
        "dnp_graded_as_zero": int(dnp_graded.sum()),
        "half_point_pushes": int(half_push.sum()),
        "implausible_over_edges_gt_10pct": int((over_edge > IMPLAUSIBLE_EDGE).sum()),
        "implausible_under_edges_gt_10pct": int((under_edge > IMPLAUSIBLE_EDGE).sum()),
    }


def summarize_current_vs_market(rows: pd.DataFrame, *, n_largest: int = 15) -> dict:
    if rows.empty or "current_over_edge" not in rows.columns:
        return {
            "n": 0,
            "mean_over_edge": float("nan"),
            "mean_abs_over_edge": float("nan"),
            "largest": [],
        }
    over_edge = pd.to_numeric(rows["current_over_edge"], errors="coerce")
    delta = pd.to_numeric(rows["current_p_over"], errors="coerce") - pd.to_numeric(
        rows["p_market_over"], errors="coerce"
    )
    ranked = rows.assign(_abs=delta.abs()).sort_values("_abs", ascending=False)
    largest = []
    for rec in ranked.head(int(n_largest)).to_dict("records"):
        largest.append(
            {
                "player_name": rec.get("player_name"),
                "book": rec.get("book"),
                "line": rec.get("line"),
                "hat_p": rec.get("hat_p"),
                "current_p_over": rec.get("current_p_over"),
                "p_market_over": rec.get("p_market_over"),
                "current_over_edge": rec.get("current_over_edge"),
            }
        )
    return {
        "n": int(over_edge.notna().sum()),
        "mean_over_edge": float(over_edge.mean()) if over_edge.notna().any() else float("nan"),
        "mean_abs_over_edge": float(over_edge.abs().mean())
        if over_edge.notna().any()
        else float("nan"),
        "largest": largest,
    }


def summarize_current_vs_clip(rows: pd.DataFrame) -> dict:
    if rows.empty or "clip_mean_p_over" not in rows.columns:
        return {
            "n": 0,
            "mean_over_shift": float("nan"),
            "mean_abs_over_shift": float("nan"),
            "low_scorer_overs": {"n": 0, "mean_over_shift": float("nan")},
        }
    shift = pd.to_numeric(rows["current_p_over"], errors="coerce") - pd.to_numeric(
        rows["clip_mean_p_over"], errors="coerce"
    )
    hat = pd.to_numeric(rows["hat_p"], errors="coerce")
    low = hat < LOW_HAT_P
    return {
        "n": int(shift.notna().sum()),
        "mean_over_shift": float(shift.mean()) if shift.notna().any() else float("nan"),
        "mean_abs_over_shift": float(shift.abs().mean()) if shift.notna().any() else float("nan"),
        "low_scorer_overs": {
            "n": int((low & shift.notna()).sum()),
            "mean_over_shift": float(shift[low].mean())
            if (low & shift.notna()).any()
            else float("nan"),
        },
    }


def summarize_edge_buckets(rows: pd.DataFrame) -> dict:
    empty_bucket = {
        label: {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "voids": 0}
        for label in EDGE_BUCKETS
    }
    if rows.empty or "current_over_edge" not in rows.columns:
        return {
            "over": empty_bucket,
            "under": {
                label: dict(stats) for label, stats in empty_bucket.items()
            },
        }
    report: dict[str, dict] = {}
    for side in ("over", "under"):
        edge = pd.to_numeric(rows[f"current_{side}_edge"], errors="coerce")
        result = rows[f"{side}_result"].astype("string")
        played = rows["played"].fillna(False).astype(bool)
        buckets: dict[str, dict] = {}
        for label in EDGE_BUCKETS:
            buckets[label] = {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "voids": 0}
        for value, grade, did_play in zip(edge, result, played):
            if not np.isfinite(value) or value < 0:
                continue
            label = edge_bucket(float(value))
            buckets[label]["n"] += 1
            if not did_play or grade == "void":
                buckets[label]["voids"] += 1
            elif grade == "win":
                buckets[label]["wins"] += 1
            elif grade == "loss":
                buckets[label]["losses"] += 1
            elif grade == "push":
                buckets[label]["pushes"] += 1
        report[side] = buckets
    return report


def summarize_side_split(rows: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    if rows.empty or "current_over_edge" not in rows.columns:
        for side in ("over", "under"):
            out[side] = {
                "n": 0,
                "n_positive_edge": 0,
                "mean_edge": float("nan"),
                "mean_positive_edge": float("nan"),
                "positive_edge_wins": 0,
                "positive_edge_losses": 0,
            }
        return out
    for side in ("over", "under"):
        edge = pd.to_numeric(rows[f"current_{side}_edge"], errors="coerce")
        result = rows[f"{side}_result"].astype("string")
        positive = edge > 0
        played_pos = positive & rows["played"].fillna(False).astype(bool)
        wins = int((played_pos & result.eq("win")).sum())
        losses = int((played_pos & result.eq("loss")).sum())
        out[side] = {
            "n": int(edge.notna().sum()),
            "n_positive_edge": int(positive.sum()),
            "mean_edge": float(edge.mean()) if edge.notna().any() else float("nan"),
            "mean_positive_edge": float(edge[positive].mean())
            if positive.any()
            else float("nan"),
            "positive_edge_wins": wins,
            "positive_edge_losses": losses,
        }
    return out


def audit_reports(rows: pd.DataFrame) -> dict:
    return {
        "integrity": summarize_integrity(rows),
        "current_vs_market": summarize_current_vs_market(rows),
        "current_vs_clip_mean": summarize_current_vs_clip(rows),
        "edge_buckets": summarize_edge_buckets(rows),
        "side_split": summarize_side_split(rows),
    }


def _role_label(start_rate_10: object) -> str:
    role_id = int(np.asarray(expected_role(start_rate_10)).reshape(-1)[0])
    return ROLE_LABEL.get(role_id, "missing")


def _bundle_paths(root: Path) -> dict[str, Path]:
    models = root / "artifacts" / "models"
    return {
        "minutes_mean_path": models / "minutes" / "xgboost_minutes.joblib",
        "minutes_dist_path": models / "minutes" / "xgboost_minutes_distribution.joblib",
        "points_path": models / "points" / "xgboost_points.joblib",
        "joint_calibration_path": models / "points" / "joint_calibration.joblib",
    }


def _side_map(frame: pd.DataFrame) -> dict[tuple, dict[str, pd.Series]]:
    keyed: dict[tuple, dict[str, pd.Series]] = {}
    for _, row in frame.iterrows():
        key = (
            str(row["BOOKMAKER"]),
            str(row["NAME"]),
            float(row["LINE"]),
            str(row["COMMENCE_TIME"]),
        )
        side = str(row["OVER/UNDER"]).strip().lower()
        keyed.setdefault(key, {})[side] = row
    return keyed


def pair_snapshot_markets(
    snapshot: pd.DataFrame,
    closing: pd.DataFrame,
) -> pd.DataFrame:
    snap_sides = _side_map(snapshot)
    close_sides = _side_map(closing)
    rows = []
    for key, sides in snap_sides.items():
        if set(sides) < {"over", "under"}:
            continue
        over = sides["over"]
        under = sides["under"]
        close = close_sides.get(key, {})
        close_over = close.get("over")
        close_under = close.get("under")
        odds_ts = pd.Timestamp(over["LAST_UPDATE"])
        close_ts = (
            pd.Timestamp(close_over["LAST_UPDATE"])
            if close_over is not None
            else pd.NaT
        )
        tip = assumed_tip_utc(over["COMMENCE_TIME"])
        rows.append(
            {
                "book": key[0],
                "player_name": key[1],
                "line": key[2],
                "commence_time": key[3],
                "odds_timestamp": odds_ts,
                "over_odds": float(over["ODDS"]),
                "under_odds": float(under["ODDS"]),
                "closing_over_odds": (
                    float(close_over["ODDS"]) if close_over is not None else float("nan")
                ),
                "closing_under_odds": (
                    float(close_under["ODDS"]) if close_under is not None else float("nan")
                ),
                "closing_odds_timestamp": close_ts,
                "closing_distinct": bool(
                    pd.notna(close_ts) and pd.Timestamp(close_ts) != pd.Timestamp(odds_ts)
                ),
                "data_pulled_at": over.get("DATA_PULLED_AT"),
                "game_timestamp": tip,
                "odds_before_tip": bool(odds_ts < tip),
                "snapshot_at_least_60min_before_tip": bool(
                    odds_ts <= tip - pd.Timedelta(minutes=60)
                ),
            }
        )
    return pd.DataFrame(rows)


def _cloud_lookup(
    clouds: dict,
    player_id: int,
    game_date: pd.Timestamp,
) -> JointSimulation | None:
    key = (int(player_id), pd.Timestamp(game_date).date())
    direct = clouds.get(key)
    if direct is not None:
        return direct
    commence = pd.Timestamp(game_date).date()
    for (pid, day), cloud in clouds.items():
        if pid == player_id and abs((day - commence).days) <= 1:
            return cloud
    return None


def _price_draws(draws: np.ndarray, line: float, over_odds: float, under_odds: float) -> dict:
    probs = draw_line_probabilities(draws, line)
    p_raw_over = american_to_implied_probability(over_odds)
    p_raw_under = american_to_implied_probability(under_odds)
    p_market_over, p_market_under = remove_vig(over_odds, under_odds)
    return {
        **probs,
        "p_raw_over": float(p_raw_over),
        "p_raw_under": float(p_raw_under),
        "p_market_over": float(p_market_over),
        "p_market_under": float(p_market_under),
        "over_edge": float(probs["p_over"] - p_market_over),
        "under_edge": float(probs["p_under"] - p_market_under),
        "over_ev": expected_return_per_unit(
            p_win=probs["p_over"],
            p_lose=probs["p_under"],
            p_push=probs["p_push"],
            american_odds=over_odds,
        ),
        "under_ev": expected_return_per_unit(
            p_win=probs["p_under"],
            p_lose=probs["p_over"],
            p_push=probs["p_push"],
            american_odds=under_odds,
        ),
    }


def audit_nba_us_snapshot(
    quotes_path: str | Path,
    *,
    as_of: str | pd.Timestamp | None = None,
    n_draws: int = 4_000,
    minutes_before_tip: int = 60,
    root: Path = ROOT,
) -> tuple[pd.DataFrame, dict]:
    quotes = load_player_points_quotes(quotes_path)
    cutoff = pd.Timestamp(as_of) if as_of is not None else snapshot_as_of(quotes)
    quotes = filter_commence_after_as_of(quotes, cutoff)
    snapshot, closing = select_snapshot_quotes(
        quotes, minutes_before_tip=minutes_before_tip
    )
    paired = pair_snapshot_markets(snapshot, closing)
    if paired.empty:
        empty = paired.copy()
        reports = audit_reports(empty)
        reports["as_of"] = str(pd.Timestamp(cutoff).date())
        reports["quotes_file"] = str(Path(quotes_path).name)
        reports["note"] = "No complete player/line/book pairs after as_of."
        return empty, reports

    panel = load_silver_panel(root=root)
    history = appearances_before(panel, cutoff)
    candidates = build_candidate_rows(
        snapshot,
        history,
        schedule=panel,
        as_of=cutoff,
    )
    if candidates.empty:
        empty = paired.iloc[0:0].copy()
        reports = audit_reports(empty)
        reports["as_of"] = str(pd.Timestamp(cutoff).date())
        reports["quotes_file"] = str(Path(quotes_path).name)
        reports["note"] = "No resolved local games strictly after as_of."
        return empty, reports

    candidates = candidates.copy()
    candidates["player_id"] = pd.to_numeric(candidates["player_id"], errors="coerce")
    candidates["game_date"] = pd.to_datetime(
        candidates["game_date"], errors="coerce"
    ).dt.normalize()
    candidates["commence_date"] = pd.to_datetime(
        candidates["commence_time"], errors="coerce"
    ).dt.normalize()
    resolved = candidates.loc[
        candidates["game_date"].gt(cutoff)
        & ~candidates["game_id"].astype("string").str.startswith("pregame-")
    ].copy()
    identity = resolved.loc[
        :,
        ["player_id", "player_name", "game_date", "game_id", "commence_date"],
    ].drop_duplicates(["player_name", "commence_date"])

    paired = paired.copy()
    paired["commence_date"] = pd.to_datetime(
        paired["commence_time"], errors="coerce"
    ).dt.normalize()
    paired = paired.merge(identity, how="inner", on=["player_name", "commence_date"])
    paired = paired.loc[paired["game_date"].gt(cutoff)].copy()
    if paired.empty:
        reports = audit_reports(paired)
        reports["as_of"] = str(pd.Timestamp(cutoff).date())
        reports["quotes_file"] = str(Path(quotes_path).name)
        reports["note"] = "No local games strictly after as_of in this snapshot."
        return paired, reports

    featured = build_pregame_points_features(
        history,
        resolved,
        as_of=cutoff,
    )
    featured = featured.drop_duplicates(["player_id", "game_date"], keep="first")
    bundle = load_joint_points_bundle(**_bundle_paths(root))
    simulator = JointPointsSimulator(bundle, n_draws=n_draws)
    clouds: dict[tuple, JointSimulation] = {}
    for _, row in featured.iterrows():
        key = (int(row["player_id"]), pd.Timestamp(row["game_date"]).date())
        clouds[key] = simulator.simulate(row)

    actuals = panel.copy()
    actuals["game_date"] = pd.to_datetime(actuals["game_date"], errors="coerce").dt.normalize()
    actuals["player_id"] = pd.to_numeric(actuals["player_id"], errors="coerce")
    paired = paired.merge(
        actuals.loc[:, ["player_id", "game_date", "pts", "minutes"]],
        how="left",
        on=["player_id", "game_date"],
    )
    featured_idx = featured.copy()
    featured_idx["player_id"] = pd.to_numeric(featured_idx["player_id"], errors="coerce")
    featured_idx["game_date"] = pd.to_datetime(
        featured_idx["game_date"], errors="coerce"
    ).dt.normalize()
    rate_cols = ["player_id", "game_date"]
    if "start_rate_10" in featured_idx.columns:
        rate_cols.append("start_rate_10")
    paired = paired.merge(
        featured_idx.loc[:, rate_cols].drop_duplicates(["player_id", "game_date"]),
        how="left",
        on=["player_id", "game_date"],
    )

    records = []
    for rec in paired.to_dict("records"):
        line = float(rec["line"])
        settled = settle_sides(
            pts=rec.get("pts"),
            minutes=rec.get("minutes"),
            line=line,
        )
        local_tip = assumed_tip_utc(rec["game_date"])
        odds_ts = pd.Timestamp(rec["odds_timestamp"])
        if odds_ts.tzinfo is None:
            odds_ts = odds_ts.tz_localize("UTC")
        else:
            odds_ts = odds_ts.tz_convert("UTC")
        cloud = _cloud_lookup(clouds, int(rec["player_id"]), rec["game_date"])
        row = {
            "game_date": rec["game_date"],
            "game_timestamp": local_tip,
            "odds_timestamp": odds_ts,
            "data_pulled_at": rec.get("data_pulled_at"),
            "book": rec["book"],
            "player_name": rec["player_name"],
            "player_id": rec["player_id"],
            "line": line,
            "over_odds": rec["over_odds"],
            "under_odds": rec["under_odds"],
            "closing_over_odds": rec["closing_over_odds"],
            "closing_under_odds": rec["closing_under_odds"],
            "closing_odds_timestamp": rec["closing_odds_timestamp"],
            "closing_distinct": rec["closing_distinct"],
            "actual_points": settled["actual_points"],
            "played": settled["played"],
            "status": settled["status"],
            "over_result": settled["over_result"],
            "under_result": settled["under_result"],
            "odds_before_tip": bool(odds_ts < local_tip),
            "snapshot_at_least_60min_before_tip": bool(
                odds_ts <= local_tip - pd.Timedelta(minutes=60)
            ),
            "features_as_of_before_game": True,
            "feature_as_of": str(pd.Timestamp(cutoff).date()),
        }
        if "start_rate_10" in rec and pd.notna(rec.get("start_rate_10")):
            row["start_rate_10"] = rec["start_rate_10"]
            row["expected_role"] = _role_label(rec["start_rate_10"])
        else:
            row["expected_role"] = "missing"
        if cloud is None:
            row["hat_p"] = float("nan")
            row["hat_m"] = float("nan")
            records.append(row)
            continue
        current = _price_draws(
            cloud.point_draws, line, rec["over_odds"], rec["under_odds"]
        )
        clip_draws = clip_mean_point_draws(cloud)
        challenger = _price_draws(
            clip_draws, line, rec["over_odds"], rec["under_odds"]
        )
        row.update(
            {
                "hat_p": float(cloud.hat_p),
                "hat_m": float(cloud.hat_m),
                "g": float(cloud.g),
                "current_p_over": current["p_over"],
                "current_p_under": current["p_under"],
                "current_p_push": current["p_push"],
                "clip_mean_p_over": challenger["p_over"],
                "clip_mean_p_under": challenger["p_under"],
                "clip_mean_p_push": challenger["p_push"],
                "p_raw_over": current["p_raw_over"],
                "p_raw_under": current["p_raw_under"],
                "p_market_over": current["p_market_over"],
                "p_market_under": current["p_market_under"],
                "current_over_edge": current["over_edge"],
                "current_under_edge": current["under_edge"],
                "current_over_ev": current["over_ev"],
                "current_under_ev": current["under_ev"],
                "clip_mean_over_edge": challenger["over_edge"],
                "clip_mean_under_edge": challenger["under_edge"],
                "clip_mean_over_ev": challenger["over_ev"],
                "clip_mean_under_ev": challenger["under_ev"],
            }
        )
        if np.isfinite(rec["closing_over_odds"]) and np.isfinite(rec["closing_under_odds"]):
            row["closing_p_market_over"] = market_over_probability(
                rec["closing_over_odds"],
                rec["closing_under_odds"],
            )
            row["clv_over"] = (
                row["closing_p_market_over"] - row["p_market_over"]
                if rec["closing_distinct"]
                else float("nan")
            )
        records.append(row)

    table = pd.DataFrame(records)
    reports = audit_reports(table)
    reports["as_of"] = str(pd.Timestamp(cutoff).date())
    reports["quotes_file"] = str(Path(quotes_path).name)
    reports["n_snapshot_pairs"] = int(len(table))
    reports["closing_distinct_n"] = int(table["closing_distinct"].sum()) if len(table) else 0
    if reports["closing_distinct_n"] == 0:
        reports["clv_note"] = (
            "Single-pull files: closing odds equal the analysis snapshot. "
            "CLV is not identified."
        )
    return table, reports
