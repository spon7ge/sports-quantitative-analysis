"""Player-prop odds from ``singleBookies.csv`` and the betting rule used in backtests.

``load_single_bookies`` returns one consensus row per player-game (the line
quoted by the most books, best and median prices across books) and each
book's own quote per player-game. Scoring takes the model's over probability
at each row's line, so any PMF or simulator can be plugged in.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

EV_EDGES = (-np.inf, 0.0, 0.03, 0.05, 0.10, 0.20, np.inf)
EV_LABELS = ("<0%", "0-3%", "3-5%", "5-10%", "10-20%", ">=20%")


def american_to_decimal(price) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    return np.where(price > 0, 1 + price / 100, 1 + 100 / np.abs(price))


def decimal_to_american(decimal) -> np.ndarray:
    decimal = np.asarray(decimal, dtype=float)
    return np.where(decimal >= 2, (decimal - 1) * 100, -100 / (decimal - 1))


def no_vig_over(over_price, under_price) -> np.ndarray:
    """Over probability with the vig removed proportionally. Prices are American."""
    q_over = 1 / american_to_decimal(over_price)
    q_under = 1 / american_to_decimal(under_price)
    return q_over / (q_over + q_under)


def binary_log_loss(p, hit) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    hit = np.asarray(hit, dtype=float)
    return -(hit * np.log(p) + (1 - hit) * np.log(1 - p))


def name_key(name, aliases: Mapping[str, str] | None = None) -> str:
    """Lowercase ASCII name without punctuation or Jr./Sr./II-IV suffixes."""
    text = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    text = re.sub(r"[^a-z ]", "", text.replace("-", " "))
    text = " ".join(re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text).split())
    return (aliases or {}).get(text, text)


def load_single_bookies(
    path: str | Path,
    gamelogs: pd.DataFrame,
    *,
    category: str = "points",
    min_books: int = 1,
    aliases: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consensus line per player-game, and each book's own quote per player-game.

    ``gamelogs`` needs ``player_id``, ``player_name``, ``game_id``, ``game_date``.
    Odds are matched on name and date, then on the previous day, because
    ``GAME_DATE`` is the UTC date. The share of two-sided book lines matched is
    stored in ``consensus.attrs["match_rate"]``.
    """
    raw = pd.read_csv(path)
    raw = raw.loc[raw["CATEGORY"].eq(category)].copy()
    raw["name_key"] = raw["NAME"].map(lambda name: name_key(name, aliases))
    raw["date"] = pd.to_datetime(raw["GAME_DATE"]).dt.normalize()
    raw["decimal"] = american_to_decimal(raw["ODDS"])
    quotes = (
        raw.pivot_table(
            index=["name_key", "date", "BOOKMAKER", "LINE"],
            columns="SIDE",
            values="decimal",
            aggfunc="max",
        )
        .dropna(subset=["over", "under"])
        .reset_index()
    )

    logs = gamelogs[["player_id", "player_name", "game_id", "game_date"]].copy()
    logs["name_key"] = logs["player_name"].map(lambda name: name_key(name, aliases))
    logs["date"] = pd.to_datetime(logs["game_date"]).dt.normalize()
    logs = logs.drop_duplicates(["name_key", "date"], keep=False)
    logs = logs[["name_key", "date", "player_id", "game_id"]]
    same_day = quotes.merge(logs, on=["name_key", "date"], how="left")
    prev_day = quotes.assign(date=quotes["date"] - pd.Timedelta(days=1)).merge(
        logs, on=["name_key", "date"], how="left"
    )
    quotes["player_id"] = same_day["player_id"].fillna(prev_day["player_id"])
    quotes["game_id"] = same_day["game_id"].fillna(prev_day["game_id"])
    match_rate = float(quotes["player_id"].notna().mean()) if len(quotes) else 0.0
    n_quotes = len(quotes)
    quotes = quotes.dropna(subset=["player_id"])

    keys = ["player_id", "game_id"]
    quotes["n_books"] = quotes.groupby([*keys, "LINE"])["BOOKMAKER"].transform("nunique")
    quotes["dist"] = (
        quotes["LINE"] - quotes.groupby(keys)["LINE"].transform("median")
    ).abs()
    main_line = (
        quotes.sort_values(["n_books", "dist"], ascending=[False, True])
        .groupby(keys)["LINE"]
        .first()
        .rename("main_line")
    )
    quotes = quotes.join(main_line, on=keys)

    # A book with several lines on one player-game keeps the one nearest the consensus line.
    by_book = (
        quotes.assign(main_dist=(quotes["LINE"] - quotes["main_line"]).abs())
        .sort_values("main_dist")
        .groupby([*keys, "BOOKMAKER"], as_index=False)
        .first()
    )
    book_quotes = pd.DataFrame({
        "game_id": by_book["game_id"],
        "player_id": by_book["player_id"].astype(int),
        "book": by_book["BOOKMAKER"],
        "line": by_book["LINE"],
        "over_bet": decimal_to_american(by_book["over"]),
        "under_bet": decimal_to_american(by_book["under"]),
    })

    main = quotes.loc[quotes["LINE"].eq(quotes["main_line"]) & quotes["n_books"].ge(min_books)]
    lines = (
        main.groupby(keys)
        .agg(
            line=("LINE", "first"),
            n_books=("n_books", "first"),
            over_best=("over", "max"),
            under_best=("under", "max"),
            over_median=("over", "median"),
            under_median=("under", "median"),
        )
        .reset_index()
    )
    consensus = pd.DataFrame({
        "game_id": lines["game_id"],
        "player_id": lines["player_id"].astype(int),
        "line": lines["line"],
        "n_books": lines["n_books"],
        "over_bet": decimal_to_american(lines["over_best"]),
        "under_bet": decimal_to_american(lines["under_best"]),
        "over_close": decimal_to_american(lines["over_median"]),
        "under_close": decimal_to_american(lines["under_median"]),
    })
    consensus.attrs["match_rate"] = match_rate
    consensus.attrs["n_quotes"] = n_quotes
    return consensus, book_quotes


def over_given_no_push(pmf, line) -> np.ndarray:
    """``P(Y > line | Y != line)`` from integer PMF rows. Half-point lines never push."""
    pmf = np.asarray(pmf, dtype=float)
    line = np.asarray(line, dtype=float).reshape(-1, 1)
    k = np.arange(pmf.shape[1]).reshape(1, -1)
    p_over = np.sum(pmf * (k > line), axis=1)
    p_push = np.sum(pmf * (k == line), axis=1)
    return p_over / np.maximum(1 - p_push, 1e-12)


def game_bootstrap(values, games, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Mean of the finite values with a 95% bootstrap interval over games."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan, np.nan, np.nan
    codes, _ = pd.factorize(np.asarray(games)[finite])
    sums = np.bincount(codes, weights=values[finite])
    counts = np.bincount(codes).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(values[finite].mean()), float(low), float(high)


def choose_sides(p_model, m: pd.DataFrame) -> pd.DataFrame:
    """Higher-EV side of each line at ``m``'s ``over_bet`` / ``under_bet`` prices.

    ``m`` needs ``game_key``, ``pts``, ``line``, ``over_bet``, ``under_bet``.
    ``p_model`` is the over probability given no push.
    """
    p_model = np.asarray(p_model, dtype=float)
    line = m["line"].to_numpy(dtype=float)
    y = m["pts"].to_numpy(dtype=float)
    d_over = american_to_decimal(m["over_bet"])
    d_under = american_to_decimal(m["under_bet"])
    ev_over, ev_under = p_model * d_over - 1, (1 - p_model) * d_under - 1
    take_over = ev_over >= ev_under
    price = np.where(take_over, d_over, d_under)
    settled = y != line
    won = np.where(take_over, y > line, y < line)
    return pd.DataFrame({
        "game_key": m["game_key"].to_numpy(),
        "hit": (y > line).astype(float),
        "p_model": p_model,
        "take_over": take_over,
        "price": price,
        "ev": np.where(take_over, ev_over, ev_under),
        "settled": settled,
        "won": won,
        "profit": np.where(settled, np.where(won, price - 1, -1.0), 0.0),
    })


def score_market(
    p_model,
    m: pd.DataFrame,
    p_market,
    p_consensus,
    *,
    edge_threshold: float,
    n_boot: int,
    seed: int,
) -> tuple[dict, dict | None]:
    """Model vs a de-vigged market price on log loss, then bets above ``edge_threshold``.

    ``p_consensus`` may be NaN where the consensus line differs from ``m``'s
    line; those bets are left out of the edge columns.
    """
    s = choose_sides(p_model, m)
    settled, games, hit = s["settled"].to_numpy(), s["game_key"].to_numpy(), s["hit"].to_numpy()
    diff = np.where(
        settled,
        binary_log_loss(p_market, hit) - binary_log_loss(s["p_model"].to_numpy(), hit),
        np.nan,
    )
    mean, low, high = game_bootstrap(diff, games, n_boot, seed)
    market = {
        "n": int(settled.sum()),
        "market_minus_model_logloss": mean,
        "ci_low": low,
        "ci_high": high,
    }

    bet = s["ev"].to_numpy() > edge_threshold
    if not bet.any():
        return market, None
    take_over, price, won = s["take_over"].to_numpy(), s["price"].to_numpy(), s["won"].to_numpy()
    p_consensus = np.asarray(p_consensus, dtype=float)
    edge = (np.where(take_over, p_consensus, 1 - p_consensus) * price - 1)[bet]
    edge = edge[np.isfinite(edge)]
    roi, roi_low, roi_high = game_bootstrap(
        np.where(bet, s["profit"].to_numpy(), np.nan), games, n_boot, seed
    )
    bets = {
        "n_bets": int(bet.sum()),
        "hit_rate": won[bet & settled].mean(),
        "roi": roi,
        "roi_ci_low": roi_low,
        "roi_ci_high": roi_high,
        "mean_edge_vs_consensus": edge.mean() if edge.size else np.nan,
        "share_beats_consensus": np.mean(edge > 0) if edge.size else np.nan,
    }
    return market, bets


def ev_bucket_records(
    p_model,
    m: pd.DataFrame,
    p_consensus,
    *,
    n_boot: int,
    seed: int,
    edges: Sequence[float] = EV_EDGES,
    labels: Sequence[str] = EV_LABELS,
) -> list[dict]:
    """Hit rate and ROI by the model's EV on its chosen side, every line included."""
    s = choose_sides(p_model, m)
    p_consensus = np.asarray(p_consensus, dtype=float)
    s["p_model_side"] = np.where(s["take_over"], s["p_model"], 1 - s["p_model"])
    s["p_consensus_side"] = np.where(s["take_over"], p_consensus, 1 - p_consensus)
    s["bucket"] = pd.cut(s["ev"], list(edges), labels=list(labels), right=False)
    records = []
    for bucket, g in s.groupby("bucket", observed=True):
        roi, low, high = game_bootstrap(g["profit"], g["game_key"], n_boot, seed)
        records.append({
            "ev_bucket": bucket,
            "n": len(g),
            "hit_rate": g.loc[g["settled"], "won"].mean(),
            "roi": roi,
            "roi_ci_low": low,
            "roi_ci_high": high,
            "mean_ev": g["ev"].mean(),
            "mean_p_model_side": g["p_model_side"].mean(),
            "mean_p_consensus_side": g["p_consensus_side"].mean(),
        })
    return records
