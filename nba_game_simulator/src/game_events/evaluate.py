"""Score one game's simulated player events against the official box score."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from game_events.checks import _normalize_ids
from game_events.constants import EVALUATED_STATS


def score_event_forecasts(
    stats: Mapping[str, Mapping[str, np.ndarray]],
    player_ids: Mapping[str, Any],
    actual: Mapping[str, pd.DataFrame],
    *,
    game_id: str | None = None,
    interval: float = 0.8,
) -> pd.DataFrame:
    """Score one game's simulated player stats against the official box score.

    ``stats`` and ``player_ids`` are ``allocate_events`` output and
    ``details["player_ids"]``. ``actual[team]`` has ``player_id`` and
    lowercase stat columns. A player in the box score with no forecast is
    scored against zeros and flagged. A forecast player with no box-score
    row, or a missing label, is unknown and skipped.
    """
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must be in (0, 1)")
    if set(stats) != set(player_ids) or not set(actual) <= set(stats):
        raise ValueError("stats, player_ids, and actual must name the same teams")
    rows = []
    for team, box in actual.items():
        ids = tuple(_normalize_ids(list(player_ids[team]), f"{team} player_ids"))
        column_of = {pid: j for j, pid in enumerate(ids)}
        if not isinstance(box, pd.DataFrame) or "player_id" not in box.columns:
            raise ValueError(f"{team} actual needs player_id")
        box = box.copy()
        box["player_id"] = _normalize_ids(
            box["player_id"], f"{team} actual player_id"
        ).to_numpy()
        if box["player_id"].duplicated().any():
            raise ValueError(f"{team} actual has duplicate players")
        if "reb" not in box.columns and {"oreb", "dreb"} <= set(box.columns):
            box["reb"] = box["oreb"] + box["dreb"]
        for stat in EVALUATED_STATS:
            column = stat.lower()
            if column not in box.columns or stat not in stats[team]:
                continue
            simulated = np.asarray(stats[team][stat])
            labels = pd.to_numeric(box[column], errors="coerce")
            for pid, label in zip(box["player_id"], labels):
                if not np.isfinite(label):
                    continue
                j = column_of.get(pid)
                samples = (
                    simulated[:, j] if j is not None else np.zeros(simulated.shape[0])
                )
                rows.append(
                    {
                        "game_id": game_id,
                        "team": team,
                        "player_id": pid,
                        "stat": stat,
                        "forecast_missing": j is None,
                        **_score_samples(samples, float(label), interval),
                    }
                )
    return pd.DataFrame(rows)


def summarize_event_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-stat accuracy and calibration across scored player-games.

    A calibrated forecast has ``pit_mid`` near 0.5 and coverage near the
    nominal interval; discrete counts make inclusive coverage run high.
    """
    if scores.empty:
        return pd.DataFrame()
    return (
        scores.groupby("stat")
        .agg(
            n=("actual", "size"),
            mae_median=("abs_error_median", "mean"),
            rmse_mean=("sq_error_mean", lambda values: float(np.sqrt(values.mean()))),
            bias=("error_mean", "mean"),
            coverage=("covered", "mean"),
            zero_brier=("zero_brier", "mean"),
            crps=("crps", "mean"),
            pit_mid=("pit_mid", "mean"),
            forecast_missing=("forecast_missing", "sum"),
        )
        .reset_index()
    )


def _score_samples(
    samples: np.ndarray, actual: float, interval: float
) -> dict[str, float]:
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    tail = (1.0 - interval) / 2.0
    low, high = np.quantile(x, [tail, 1.0 - tail], method="inverted_cdf")
    mean = float(x.mean())
    median = float(np.quantile(x, 0.5, method="inverted_cdf"))
    p_zero = float(np.mean(x == 0.0))
    # Sample CRPS: E|X - y| - E|X - X'| / 2, with the pair term from order statistics.
    spread = float(np.sum((2.0 * np.arange(1, n + 1) - n - 1.0) * x)) * 2.0 / (n * n)
    return {
        "actual": actual,
        "mean": mean,
        "median": median,
        "low": float(low),
        "high": float(high),
        "covered": bool(low <= actual <= high),
        "p_zero": p_zero,
        "zero_brier": (p_zero - float(actual == 0.0)) ** 2,
        "abs_error_median": abs(median - actual),
        "error_mean": mean - actual,
        "sq_error_mean": (mean - actual) ** 2,
        "crps": float(np.mean(np.abs(x - actual))) - 0.5 * spread,
        "pit_mid": float(np.mean(x < actual) + 0.5 * np.mean(x == actual)),
    }
