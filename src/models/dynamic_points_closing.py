"""Close-line backtest for the dynamic hierarchical points model.

The probability is the plug-in Student-t already used for the walk-forward
probability-integral transform. A closing line is only a grading threshold.
The played point total settles the bet and is not an input.

A row is graded only when both lagged inputs are strictly earlier than the
slate: the player's previous appearance, and the opponent's previous game.
Slates on or before the end of the training window are left out as well.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_POINTS_STATS = frozenset({"points", "pts"})
_PRIZEPICKS_AMERICAN = -119
_SUMMARY_COLUMNS = (
    "book",
    "n",
    "decided",
    "pushes",
    "over_rate",
    "mean_model_p_over",
    "model_brier",
    "market_brier",
    "model_log_loss",
    "favored_hit_rate",
    "favored_roi",
    "plus_ev_n",
    "plus_ev_roi",
)


def load_main_points_markets(
    books: Mapping[str, str | Path],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One two-way main points close per book, player, and slate.

    PrizePicks standard points has no posted price in the closing file, so
    both sides use the posted -119. Other books keep the alternate rung whose
    over and under implied probabilities are closest. One-sided offers, and
    players with two PrizePicks standard points lines, are counted and omitted.
    """
    if not books:
        raise ValueError("at least one closing directory is required")
    sides: list[pd.DataFrame] = []
    for book, directory in books.items():
        sides.append(_load_points_sides(str(book), Path(directory)))
    loaded = pd.concat(sides, ignore_index=True) if sides else pd.DataFrame()
    return select_main_points(loaded)


def pregame_history_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Last player and opponent dates available to the lagged pregame inputs.

    The opponent date is the latest game for that opponent strictly before
    this row's ``(game_date, game_id)`` order, matching the one-game shift
    used to build opponent defense.
    """
    required = {"player_id", "game_id", "game_date", "team_id", "opp_team_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing pregame-date columns: {sorted(missing)}")
    games = frame.loc[:, list(required)].copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="raise")
    if games.duplicated(["player_id", "game_id"]).any():
        raise ValueError("duplicate player/game rows")
    games = games.sort_values(["player_id", "game_date", "game_id"], kind="mergesort")
    games["prior_player_game_date"] = games.groupby("player_id", sort=False)[
        "game_date"
    ].shift(1)
    games["prior_opponent_game_date"] = _prior_opponent_dates(games)
    return games.reset_index(drop=True)


def student_t_side_probabilities(
    mean: np.ndarray,
    scale: np.ndarray,
    df: np.ndarray,
    line: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Over, under, and push probabilities from the plug-in Student-t.

    A half-point line is the survival function at that line. An integer line
    uses a half-point continuity correction, so landing on the number is a push.
    """
    location = np.asarray(mean, dtype=float)
    width = np.asarray(scale, dtype=float)
    degrees = np.asarray(df, dtype=float)
    threshold = np.asarray(line, dtype=float)
    if not (location.shape == width.shape == degrees.shape == threshold.shape):
        raise ValueError("probability inputs must have the same shape")
    if not np.isfinite(location).all() or not np.isfinite(threshold).all():
        raise ValueError("mean and line must be finite")
    if (
        np.any(width <= 0)
        or np.any(~np.isfinite(width))
        or np.any(degrees <= 0)
        or np.any(~np.isfinite(degrees))
    ):
        raise ValueError("scale and degrees of freedom must be positive and finite")
    integer = np.isclose(threshold, np.round(threshold), atol=1e-6)
    upper = np.where(integer, threshold + 0.5, threshold)
    lower = np.where(integer, threshold - 0.5, threshold)
    over = stats.t.sf(upper, degrees, loc=location, scale=width)
    under = stats.t.cdf(lower, degrees, loc=location, scale=width)
    push = np.where(integer, 1.0 - over - under, 0.0)
    finite = np.isfinite(over).all() and np.isfinite(under).all()
    if not finite or np.any(push < -1e-8):
        raise ValueError("student-t probabilities are not finite")
    push = np.clip(push, 0.0, None)
    return over, under, push


def score_closing_backtest(
    predictions: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Grade pregame Student-t predictions against two-way closing lines.

    ``points`` settles the wager. It is not passed to the probability.
    """
    required_predictions = {
        "player_id",
        "game_date",
        "predictive_mean",
        "predictive_var",
        "df",
        "points",
        "prior_player_game_date",
        "prior_opponent_game_date",
    }
    required_markets = {
        "slate_date",
        "book",
        "player_name",
        "player_id",
        "line",
        "over_american",
        "under_american",
    }
    missing_predictions = required_predictions - set(predictions.columns)
    missing_markets = required_markets - set(markets.columns)
    if missing_predictions or missing_markets:
        raise ValueError(
            "missing backtest columns: "
            f"predictions {sorted(missing_predictions)}, "
            f"markets {sorted(missing_markets)}"
        )

    prepared = predictions.copy()
    prepared["player_id"] = _identifier(prepared["player_id"])
    prepared["game_date"] = pd.to_datetime(prepared["game_date"]).dt.normalize()
    for column in ("prior_player_game_date", "prior_opponent_game_date"):
        prepared[column] = pd.to_datetime(
            prepared[column], errors="coerce"
        ).dt.normalize()
    if prepared["player_id"].isna().any():
        raise ValueError("pregame predictions need a player id")
    if prepared.duplicated(["player_id", "game_date"]).any():
        raise ValueError("duplicate pregame predictions for a player and date")

    offers = markets.copy()
    offers["player_id"] = _identifier(offers["player_id"])
    offers["slate_date"] = pd.to_datetime(offers["slate_date"]).dt.normalize()
    cutoff = pd.Timestamp(train_end).normalize()
    unmatched = offers["player_id"].isna()
    named = offers.loc[~unmatched].copy()
    merged = named.merge(
        prepared,
        left_on=["player_id", "slate_date"],
        right_on=["player_id", "game_date"],
        how="left",
        indicator=True,
    )
    status = np.full(len(merged), "scored", dtype=object)
    status[merged["_merge"].eq("left_only").to_numpy()] = "no_pregame_prediction"
    in_train = merged["game_date"].notna() & merged["game_date"].le(cutoff)
    same_player = merged["prior_player_game_date"].notna() & merged[
        "prior_player_game_date"
    ].ge(merged["game_date"])
    same_opponent = merged["prior_opponent_game_date"].notna() & merged[
        "prior_opponent_game_date"
    ].ge(merged["game_date"])
    status[_mask(status, in_train)] = "train_window_excluded"
    status[_mask(status, same_player)] = "same_day_player_excluded"
    status[_mask(status, same_opponent)] = "same_day_opponent_excluded"

    graded = merged.loc[status == "scored"].copy()
    if not graded.empty:
        graded = _attach_probabilities(graded)
        graded = _attach_wagers(graded)
    audit: dict[str, object] = {
        "markets": int(len(offers)),
        "unmatched_names": int(unmatched.sum()),
        "unmatched_player_names": tuple(
            sorted(offers.loc[unmatched, "player_name"].dropna().astype(str).unique())
        ),
        "no_pregame_prediction": int((status == "no_pregame_prediction").sum()),
        "train_window_excluded": int((status == "train_window_excluded").sum()),
        "same_day_player_excluded": int((status == "same_day_player_excluded").sum()),
        "same_day_opponent_excluded": int(
            (status == "same_day_opponent_excluded").sum()
        ),
        "scored": int((status == "scored").sum()),
        "train_end": str(cutoff.date()),
    }
    accounted = (
        audit["unmatched_names"]
        + audit["no_pregame_prediction"]
        + audit["train_window_excluded"]
        + audit["same_day_player_excluded"]
        + audit["same_day_opponent_excluded"]
        + audit["scored"]
    )
    if accounted != audit["markets"]:
        raise AssertionError("closing-line rows were neither scored nor counted")
    return graded.reset_index(drop=True), audit


def closing_book_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-book calibration and flat-stake results on graded closes.

    Brier and log loss compare the model's probability of going over, among
    decided bets, with the close's no-vig over probability. ROI is profit per
    $1 staked, and a push contributes zero.
    """
    if scored.empty:
        return pd.DataFrame(columns=list(_SUMMARY_COLUMNS))
    required = {
        "book",
        "result",
        "went_over",
        "p_over_decided",
        "market_p_over",
        "profit",
        "ev",
    }
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"missing summary columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for book, frame in scored.groupby("book", sort=True):
        decided = frame.loc[frame["result"].isin(["win", "loss"])]
        plus = frame.loc[frame["ev"].gt(0)]
        record: dict[str, object] = {
            "book": book,
            "n": int(len(frame)),
            "decided": int(len(decided)),
            "pushes": int(frame["result"].eq("push").sum()),
            "over_rate": np.nan,
            "mean_model_p_over": np.nan,
            "model_brier": np.nan,
            "market_brier": np.nan,
            "model_log_loss": np.nan,
            "favored_hit_rate": np.nan,
            "favored_roi": float(frame["profit"].sum() / len(frame)),
            "plus_ev_n": int(len(plus)),
            "plus_ev_roi": (
                float(plus["profit"].sum() / len(plus)) if len(plus) else np.nan
            ),
        }
        if not decided.empty:
            outcome = decided["went_over"].to_numpy(dtype=float)
            model_probability = decided["p_over_decided"].to_numpy(dtype=float)
            market_probability = decided["market_p_over"].to_numpy(dtype=float)
            record["over_rate"] = float(outcome.mean())
            record["mean_model_p_over"] = float(model_probability.mean())
            record["model_brier"] = float(np.mean((model_probability - outcome) ** 2))
            record["market_brier"] = float(np.mean((market_probability - outcome) ** 2))
            clipped = np.clip(model_probability, 1e-6, 1 - 1e-6)
            record["model_log_loss"] = float(
                -np.mean(
                    outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped)
                )
            )
            record["favored_hit_rate"] = float(decided["result"].eq("win").mean())
        rows.append(record)
    return pd.DataFrame(rows, columns=list(_SUMMARY_COLUMNS))


_EV_THRESHOLD_COLUMNS = (
    "book",
    "min_ev",
    "n",
    "decided",
    "pushes",
    "hit_rate",
    "mean_ev",
    "roi",
)


def closing_ev_thresholds(
    scored: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Flat-stake ROI for bets whose expected value clears each cutoff.

    A cutoff of 0.05 keeps bets with at least $0.05 of expected profit per
    $1 staked. Higher cutoffs are subsets of the lower ones. The ``all`` rows
    pool every book. ROI includes pushes as zero profit.
    """
    if not thresholds:
        raise ValueError("at least one EV threshold is required")
    cutoffs = tuple(float(value) for value in thresholds)
    if any(not math.isfinite(value) for value in cutoffs):
        raise ValueError("EV thresholds must be finite")
    if list(cutoffs) != sorted(cutoffs):
        raise ValueError("EV thresholds must be in ascending order")
    if scored.empty:
        return pd.DataFrame(columns=list(_EV_THRESHOLD_COLUMNS))
    required = {"book", "ev", "profit", "result"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"missing EV-threshold columns: {sorted(missing)}")
    if not np.isfinite(scored["ev"].to_numpy(dtype=float)).all():
        raise ValueError("expected value must be finite")
    if not np.isfinite(scored["profit"].to_numpy(dtype=float)).all():
        raise ValueError("profit must be finite")

    groups: list[tuple[str, pd.DataFrame]] = [("all", scored)]
    for book, frame in scored.groupby("book", sort=True):
        groups.append((str(book), frame))
    rows = [
        _threshold_record(book, cutoff, frame.loc[frame["ev"].ge(cutoff)])
        for book, frame in groups
        for cutoff in cutoffs
    ]
    return pd.DataFrame(rows, columns=list(_EV_THRESHOLD_COLUMNS))


def _threshold_record(
    book: str,
    cutoff: float,
    kept: pd.DataFrame,
) -> dict[str, object]:
    decided = kept.loc[kept["result"].isin(["win", "loss"])]
    record: dict[str, object] = {
        "book": book,
        "min_ev": cutoff,
        "n": int(len(kept)),
        "decided": int(len(decided)),
        "pushes": int(kept["result"].eq("push").sum()),
        "hit_rate": np.nan,
        "mean_ev": np.nan,
        "roi": np.nan,
    }
    if kept.empty:
        return record
    record["mean_ev"] = float(kept["ev"].mean())
    record["roi"] = float(kept["profit"].sum() / len(kept))
    if not decided.empty:
        record["hit_rate"] = float(decided["result"].eq("win").mean())
    return record


def _load_points_sides(book: str, directory: Path) -> pd.DataFrame:
    if not directory.is_dir():
        raise ValueError(f"closing directory does not exist: {directory}")
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*_nba_closing_*.json")):
        payload = json.loads(path.read_text())
        if "date" not in payload:
            raise ValueError(f"{path.name} has no slate date")
        slate = pd.Timestamp(payload["date"]).normalize()
        for player in payload.get("players", []):
            name = player.get("player")
            for offer in player.get("lines", []):
                stat = str(offer.get("stat_type") or "").strip().lower()
                if stat not in _POINTS_STATS:
                    continue
                if book == "prizepicks" and offer.get("odds_type") != "standard":
                    continue
                line = offer.get("line_score")
                if line is None or not math.isfinite(float(line)):
                    raise ValueError(f"{path.name} has a non-finite points line")
                rows.append(
                    {
                        "slate_date": slate,
                        "book": book,
                        "player_name": name,
                        "line": float(line),
                        "side": offer.get("side"),
                        "american_price": offer.get("american_price"),
                    }
                )
    return pd.DataFrame(rows)


def select_main_points(sides: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = [
        "slate_date",
        "book",
        "player_name",
        "line",
        "over_american",
        "under_american",
    ]
    audit = {
        "side_rows": int(len(sides)),
        "one_sided_left_out": 0,
        "duplicate_standard_left_out": 0,
        "markets": 0,
    }
    if sides.empty:
        return pd.DataFrame(columns=columns), audit

    prizepicks = sides.loc[sides["book"].eq("prizepicks")].copy()
    other = sides.loc[~sides["book"].eq("prizepicks")].copy()
    prize_markets, duplicate_players = _prizepicks_markets(prizepicks)
    other_markets, one_sided = _paired_markets(other)
    markets = pd.concat([prize_markets, other_markets], ignore_index=True)
    audit["one_sided_left_out"] = one_sided
    audit["duplicate_standard_left_out"] = duplicate_players
    audit["markets"] = int(len(markets))
    return markets.loc[:, columns], audit


def _prizepicks_markets(sides: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    columns = [
        "slate_date",
        "book",
        "player_name",
        "line",
        "over_american",
        "under_american",
    ]
    if sides.empty:
        return pd.DataFrame(columns=columns), 0
    unique = sides.drop_duplicates(["slate_date", "player_name", "line"])
    line_count = unique.groupby(["slate_date", "player_name"], sort=False)[
        "line"
    ].transform("size")
    duplicate_players = int(
        unique.loc[line_count.gt(1), ["slate_date", "player_name"]]
        .drop_duplicates()
        .shape[0]
    )
    kept = unique.loc[line_count.eq(1)].copy()
    kept["over_american"] = _PRIZEPICKS_AMERICAN
    kept["under_american"] = _PRIZEPICKS_AMERICAN
    return kept.loc[:, columns], duplicate_players


def _paired_markets(sides: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    columns = [
        "slate_date",
        "book",
        "player_name",
        "line",
        "over_american",
        "under_american",
    ]
    if sides.empty:
        return pd.DataFrame(columns=columns), 0
    priced = sides.dropna(subset=["side", "american_price"]).copy()
    priced["side"] = priced["side"].astype("string").str.lower()
    priced["american_price"] = pd.to_numeric(priced["american_price"], errors="coerce")
    if priced["american_price"].isna().any() or priced["american_price"].eq(0).any():
        raise ValueError("american prices must be finite and nonzero")
    if not np.isfinite(priced["american_price"]).all():
        raise ValueError("american prices must be finite and nonzero")
    duplicate = priced.duplicated(["slate_date", "book", "player_name", "line", "side"])
    if duplicate.any():
        raise ValueError(f"{int(duplicate.sum())} duplicate priced points sides")
    wide = priced.pivot_table(
        index=["slate_date", "book", "player_name", "line"],
        columns="side",
        values="american_price",
        aggfunc="first",
    ).reset_index()
    for column in ("over", "under"):
        if column not in wide.columns:
            wide[column] = np.nan
    paired = wide["over"].notna() & wide["under"].notna()
    one_sided = int((~paired).sum())
    wide = wide.loc[paired].copy()
    if wide.empty:
        return pd.DataFrame(columns=columns), one_sided
    wide["gap"] = (_implied(wide["over"]) - _implied(wide["under"])).abs()
    wide = wide.sort_values(["gap", "line"], kind="mergesort").drop_duplicates(
        ["slate_date", "book", "player_name"]
    )
    wide = wide.rename(columns={"over": "over_american", "under": "under_american"})
    return wide.loc[:, columns], one_sided


def _prior_opponent_dates(games: pd.DataFrame) -> pd.Series:
    teams = games.loc[:, ["team_id", "game_id", "game_date"]].copy()
    conflict = teams.drop_duplicates(["team_id", "game_id", "game_date"]).duplicated(
        ["team_id", "game_id"]
    )
    if conflict.any():
        raise ValueError("a team game has two dates")
    teams = teams.drop_duplicates(["team_id", "game_id"]).sort_values(
        ["team_id", "game_date", "game_id"], kind="mergesort"
    )
    history: dict[object, list[tuple[pd.Timestamp, object]]] = {}
    for team_id, game_id, game_date in teams.itertuples(index=False):
        history.setdefault(team_id, []).append((pd.Timestamp(game_date), game_id))
    priors: list[pd.Timestamp] = []
    for opponent_id, game_date, game_id in zip(
        games["opp_team_id"], games["game_date"], games["game_id"], strict=True
    ):
        played = history.get(opponent_id, [])
        index = bisect_left(played, (pd.Timestamp(game_date), game_id))
        priors.append(pd.NaT if index == 0 else played[index - 1][0])
    return pd.Series(priors, index=games.index, dtype="datetime64[ns]")


def _identifier(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.dropna()
    if not finite.empty and not np.isfinite(finite.to_numpy(dtype=float)).all():
        raise ValueError("player ids must be finite")
    return numeric.round().astype("Int64")


def _mask(status: np.ndarray, condition: pd.Series) -> np.ndarray:
    return (status == "scored") & condition.to_numpy()


def _implied(price: pd.Series) -> pd.Series:
    value = price.astype(float)
    if value.eq(0).any() or not np.isfinite(value).all():
        raise ValueError("american prices must be finite and nonzero")
    negative = value < 0
    probability = pd.Series(np.nan, index=value.index, dtype=float)
    probability.loc[negative] = value.loc[negative].abs() / (
        value.loc[negative].abs() + 100
    )
    probability.loc[~negative] = 100 / (value.loc[~negative] + 100)
    return probability


def _american_profit(price: pd.Series) -> np.ndarray:
    value = price.to_numpy(dtype=float)
    if np.any(value == 0) or not np.isfinite(value).all():
        raise ValueError("american prices must be finite and nonzero")
    return np.where(value < 0, 100.0 / np.abs(value), value / 100.0)


def _attach_probabilities(graded: pd.DataFrame) -> pd.DataFrame:
    variance = graded["predictive_var"].to_numpy(dtype=float)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("predictive variance must be positive and finite")
    points = graded["points"].to_numpy(dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("settlement points must be finite")
    over, under, push = student_t_side_probabilities(
        graded["predictive_mean"].to_numpy(dtype=float),
        np.sqrt(variance),
        graded["df"].to_numpy(dtype=float),
        graded["line"].to_numpy(dtype=float),
    )
    graded["p_over"] = over
    graded["p_under"] = under
    graded["p_push"] = push
    decided_mass = over + under
    if np.any(decided_mass <= 0):
        raise ValueError("over and under probabilities must have positive mass")
    graded["p_over_decided"] = over / decided_mass
    graded["market_p_over"] = (
        _implied(graded["over_american"])
        / (_implied(graded["over_american"]) + _implied(graded["under_american"]))
    ).to_numpy(dtype=float)
    return graded


def _attach_wagers(graded: pd.DataFrame) -> pd.DataFrame:
    over_profit = _american_profit(graded["over_american"])
    under_profit = _american_profit(graded["under_american"])
    over = graded["p_over"].to_numpy(dtype=float)
    under = graded["p_under"].to_numpy(dtype=float)
    graded["ev_over"] = over * over_profit - under
    graded["ev_under"] = under * under_profit - over
    take_over = graded["ev_over"].to_numpy(dtype=float) >= graded["ev_under"].to_numpy(
        dtype=float
    )
    points = graded["points"].to_numpy(dtype=float)
    line = graded["line"].to_numpy(dtype=float)
    push = np.isclose(points, line)
    won = np.where(take_over, points > line, points < line)
    graded["side"] = np.where(take_over, "over", "under")
    graded["price"] = np.where(
        take_over, graded["over_american"], graded["under_american"]
    )
    graded["ev"] = np.maximum(graded["ev_over"], graded["ev_under"])
    graded["result"] = np.where(push, "push", np.where(won, "win", "loss"))
    win_profit = np.where(take_over, over_profit, under_profit)
    graded["profit"] = np.where(
        graded["result"].eq("win"),
        win_profit,
        np.where(graded["result"].eq("loss"), -1.0, 0.0),
    )
    graded["went_over"] = np.where(push, np.nan, points > line)
    return graded
