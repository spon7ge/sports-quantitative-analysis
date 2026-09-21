"""Paired model vs de-vigged quote comparison (quoted-price simulation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.markets.odds import (
    american_to_implied,
    expected_profit,
    implied_to_decimal,
    no_vig,
)
from src.mlb.markets.quotes import (
    implied_over_price,
    implied_under_price,
    select_quotes_asof,
)
from src.mlb.schemas import PMF_COLUMNS


def _line_column(line: float, side: str) -> str:
    text = str(line).replace(".", "_")
    return f"p_{side}_{text}"


def _pmf_matrix(frame: pd.DataFrame) -> np.ndarray | None:
    missing = [name for name in PMF_COLUMNS if name not in frame.columns]
    if missing:
        return None
    return frame.loc[:, list(PMF_COLUMNS)].to_numpy(dtype=float)


def _probs_from_pmf(
    pmf: np.ndarray, line: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_max = pmf.shape[1] - 2
    support = np.concatenate(
        [np.arange(k_max + 1, dtype=float), np.array([float(k_max + 1)])]
    )
    p_over = pmf @ (support > line).astype(float)
    p_under = pmf @ (support < line).astype(float)
    p_push = pmf @ (np.isclose(support, line)).astype(float)
    return p_over, p_under, p_push


def _model_side_probs(
    frame: pd.DataFrame, line: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frame)
    p_over = np.full(n, np.nan)
    p_under = np.full(n, np.nan)
    p_push = np.zeros(n)
    pmf = _pmf_matrix(frame)
    unique_lines = pd.unique(pd.Series(line))
    for ln in unique_lines:
        if not np.isfinite(ln):
            continue
        mask = np.isclose(line, float(ln))
        over_col = _line_column(float(ln), "over")
        under_col = _line_column(float(ln), "under")
        if pmf is not None:
            o, u, psh = _probs_from_pmf(pmf[mask], float(ln))
            p_over[mask] = o
            p_under[mask] = u
            p_push[mask] = psh
        elif over_col in frame.columns and under_col in frame.columns:
            p_over[mask] = pd.to_numeric(
                frame.loc[mask, over_col], errors="coerce"
            ).to_numpy()
            p_under[mask] = pd.to_numeric(
                frame.loc[mask, under_col], errors="coerce"
            ).to_numpy()
        else:
            nearest = None
            if hasattr(frame, "attrs"):
                nearest = None
            configured = []
            for col in frame.columns:
                if col.startswith("p_over_"):
                    try:
                        configured.append(
                            float(col.replace("p_over_", "").replace("_", "."))
                        )
                    except ValueError:
                        continue
            if configured:
                pick = min(configured, key=lambda c: abs(c - float(ln)))
                over_col = _line_column(pick, "over")
                under_col = _line_column(pick, "under")
                if over_col in frame.columns:
                    p_over[mask] = pd.to_numeric(
                        frame.loc[mask, over_col], errors="coerce"
                    ).to_numpy()
                if under_col in frame.columns:
                    p_under[mask] = pd.to_numeric(
                        frame.loc[mask, under_col], errors="coerce"
                    ).to_numpy()
            _ = nearest
    return p_over, p_under, p_push


def _paired_log_score(
    p_over: np.ndarray, p_under: np.ndarray, y: np.ndarray, line: np.ndarray
) -> np.ndarray:
    eps = 1e-12
    over = y > line
    under = y < line
    score = np.full(len(y), np.nan, dtype=float)
    score[over] = -np.log(np.clip(p_over[over], eps, 1.0))
    score[under] = -np.log(np.clip(p_under[under], eps, 1.0))
    push = ~(over | under)
    score[push] = 0.0
    return score


def compare_market(
    predictions: pd.DataFrame,
    quotes: pd.DataFrame,
    config: MlbConfig,
) -> pd.DataFrame:
    """Inner-join predictions to the nearest as-of quote and score vs de-vigged prices.

    Market metrics are paired log score and Brier against de-vigged quote
    probabilities. Any simulated edge is labeled ``roi_type='quoted_price_simulation'``
    and is not realized ROI.
    """
    if predictions.empty or quotes.empty:
        empty = predictions.iloc[0:0].copy()
        empty["roi_type"] = pd.Series(dtype="string")
        return empty

    joined = select_quotes_asof(predictions, quotes, config)
    if joined.empty:
        out = joined.copy()
        out["roi_type"] = pd.Series(dtype="string")
        return out

    line_col = "line_quote" if "line_quote" in joined.columns else "line"
    line = pd.to_numeric(joined[line_col], errors="coerce").to_numpy(dtype=float)
    fmt = joined["price_format"] if "price_format" in joined.columns else "american"
    over_raw = np.array(
        [
            implied_over_price(float(price), str(fmt_i))
            for price, fmt_i in zip(
                joined["over_price"].to_numpy(),
                pd.Series(fmt).astype(str),
                strict=False,
            )
        ],
        dtype=float,
    )
    under_raw = np.array(
        [
            implied_under_price(float(price), str(fmt_i))
            for price, fmt_i in zip(
                joined["under_price"].to_numpy(),
                pd.Series(fmt).astype(str),
                strict=False,
            )
        ],
        dtype=float,
    )
    market_p_over = np.empty(len(joined), dtype=float)
    market_p_under = np.empty(len(joined), dtype=float)
    for i, (p_o, p_u) in enumerate(zip(over_raw, under_raw, strict=True)):
        q_o, q_u = no_vig(p_o, p_u)
        market_p_over[i] = q_o
        market_p_under[i] = q_u

    model_p_over, model_p_under, model_p_push = _model_side_probs(joined, line)
    joined = joined.copy()
    joined["market_p_over"] = market_p_over
    joined["market_p_under"] = market_p_under
    joined["model_p_over"] = model_p_over
    joined["model_p_under"] = model_p_under
    joined["market_disagreement"] = model_p_over - market_p_over

    y = None
    for candidate in ("strikeouts", "actual_k", "y"):
        if candidate in joined.columns:
            y = pd.to_numeric(joined[candidate], errors="coerce").to_numpy(dtype=float)
            break
    if y is not None:
        joined["paired_log_score_model"] = _paired_log_score(
            model_p_over, model_p_under, y, line
        )
        joined["paired_log_score_market"] = _paired_log_score(
            market_p_over, market_p_under, y, line
        )
        over_ind = (y > line).astype(float)
        joined["brier_model"] = (model_p_over - over_ind) ** 2
        joined["brier_market"] = (market_p_over - over_ind) ** 2
        decimal_over = np.array(
            [implied_to_decimal(p) for p in market_p_over], dtype=float
        )
        p_push = np.where(np.floor(line) == line, model_p_push, 0.0)
        p_win = np.where(y > line, 1.0, 0.0)  # used only for simulation labeling below
        _ = p_win
        joined["quoted_price_simulation_ev_over"] = [
            expected_profit(
                float(po) * (1.0 - float(pp)),
                float(pu) * (1.0 - float(pp)),
                float(d),
                p_push=float(pp),
            )
            if np.isfinite(po) and np.isfinite(d)
            else np.nan
            for po, pu, d, pp in zip(
                model_p_over, model_p_under, decimal_over, p_push, strict=True
            )
        ]
    else:
        joined["paired_log_score_model"] = np.nan
        joined["paired_log_score_market"] = np.nan
        joined["brier_model"] = np.nan
        joined["brier_market"] = np.nan
        joined["quoted_price_simulation_ev_over"] = np.nan

    joined["roi_type"] = "quoted_price_simulation"
    return joined.reset_index(drop=True)


def quoted_decimal_odds(price: float, price_format: str | None) -> float:
    """Sportsbook decimal odds you would actually be filled at."""
    fmt = str(price_format or "american").lower()
    value = float(price)
    if fmt == "decimal":
        return value
    return float(implied_to_decimal(american_to_implied(value)))


def _settle_side(side: str, y: float, line: float, decimal_odds: float, stake: float) -> tuple[float, str]:
    if not np.isfinite(y):
        return float("nan"), "void"
    if y > line:
        result = "win" if side == "over" else "loss"
    elif y < line:
        result = "win" if side == "under" else "loss"
    else:
        result = "push"
    if result == "win":
        return float(stake) * (float(decimal_odds) - 1.0), result
    if result == "loss":
        return -float(stake), result
    return 0.0, result


def simulate_plus_ev_bets(
    market: pd.DataFrame,
    *,
    min_ev: float = 0.0,
    stake: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Emulate a live bettor: one side per quote, only if EV > ``min_ev``.

    EV is versus the **quoted** over/under price (juice included), not the
    de-vigged fair line. At most one side is taken; if both clear the hurdle,
    the larger EV wins. Flat ``stake`` units. P/L is a quoted-price simulation.
    """
    if market is None or market.empty:
        empty = pd.DataFrame()
        return empty, {
            "n_quotes": 0.0,
            "n_bets": 0.0,
            "n_skipped": 0.0,
            "n_wins": 0.0,
            "n_losses": 0.0,
            "n_pushes": 0.0,
            "units_staked": 0.0,
            "units_pnl": 0.0,
            "roi": float("nan"),
            "hit_rate": float("nan"),
            "mean_ev": float("nan"),
            "min_ev": float(min_ev),
            "stake": float(stake),
        }

    frame = market.copy()
    line_col = "line_quote" if "line_quote" in frame.columns else "line"
    line = pd.to_numeric(frame[line_col], errors="coerce").to_numpy(dtype=float)
    fmt = (
        frame["price_format"].astype(str)
        if "price_format" in frame.columns
        else pd.Series(["decimal"] * len(frame), index=frame.index)
    )
    d_over = np.array(
        [
            quoted_decimal_odds(float(price), str(fmt_i))
            if np.isfinite(price)
            else np.nan
            for price, fmt_i in zip(
                pd.to_numeric(frame["over_price"], errors="coerce").to_numpy(),
                fmt,
                strict=False,
            )
        ],
        dtype=float,
    )
    d_under = np.array(
        [
            quoted_decimal_odds(float(price), str(fmt_i))
            if np.isfinite(price)
            else np.nan
            for price, fmt_i in zip(
                pd.to_numeric(frame["under_price"], errors="coerce").to_numpy(),
                fmt,
                strict=False,
            )
        ],
        dtype=float,
    )
    p_over = pd.to_numeric(frame["model_p_over"], errors="coerce").to_numpy(dtype=float)
    p_under = pd.to_numeric(frame["model_p_under"], errors="coerce").to_numpy(dtype=float)
    p_push = np.clip(1.0 - p_over - p_under, 0.0, 1.0)

    ev_over = np.full(len(frame), np.nan)
    ev_under = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        if np.isfinite(p_over[i]) and np.isfinite(p_under[i]) and np.isfinite(d_over[i]) and d_over[i] > 1.0:
            ev_over[i] = expected_profit(p_over[i], p_under[i], d_over[i], p_push=p_push[i])
        if np.isfinite(p_over[i]) and np.isfinite(p_under[i]) and np.isfinite(d_under[i]) and d_under[i] > 1.0:
            ev_under[i] = expected_profit(p_under[i], p_over[i], d_under[i], p_push=p_push[i])

    y = np.full(len(frame), np.nan)
    for candidate in ("strikeouts", "actual_k", "y"):
        if candidate in frame.columns:
            y = pd.to_numeric(frame[candidate], errors="coerce").to_numpy(dtype=float)
            break

    sides: list[str] = []
    bet_decimal = np.full(len(frame), np.nan)
    bet_ev = np.full(len(frame), np.nan)
    pnl = np.zeros(len(frame), dtype=float)
    results: list[str] = []
    for i in range(len(frame)):
        eo, eu = ev_over[i], ev_under[i]
        take = ""
        if np.isfinite(eo) and eo > float(min_ev) and (not np.isfinite(eu) or eo >= eu):
            take = "over"
        elif np.isfinite(eu) and eu > float(min_ev):
            take = "under"
        sides.append(take)
        if take == "":
            results.append("skip")
            continue
        decimal = d_over[i] if take == "over" else d_under[i]
        bet_decimal[i] = decimal
        bet_ev[i] = eo if take == "over" else eu
        profit, result = _settle_side(take, y[i], line[i], decimal, stake)
        pnl[i] = 0.0 if not np.isfinite(profit) else profit
        results.append(result)

    frame["decimal_over"] = d_over
    frame["decimal_under"] = d_under
    frame["ev_over"] = ev_over
    frame["ev_under"] = ev_under
    frame["bet_side"] = sides
    frame["bet_decimal"] = bet_decimal
    frame["bet_ev"] = bet_ev
    frame["stake"] = np.where(frame["bet_side"].astype(str) != "", float(stake), 0.0)
    frame["result"] = results
    frame["pnl"] = pnl
    frame["roi_type"] = "quoted_price_simulation"

    taken = frame["bet_side"].astype(str) != ""
    settled = frame.loc[taken & frame["result"].isin(["win", "loss", "push"])]
    decided = frame.loc[taken & frame["result"].isin(["win", "loss"])]
    units_staked = float(frame.loc[taken, "stake"].sum())
    units_pnl = float(frame.loc[taken, "pnl"].sum())
    summary = {
        "n_quotes": float(len(frame)),
        "n_bets": float(taken.sum()),
        "n_skipped": float((~taken).sum()),
        "n_wins": float((settled["result"] == "win").sum()),
        "n_losses": float((settled["result"] == "loss").sum()),
        "n_pushes": float((settled["result"] == "push").sum()),
        "units_staked": units_staked,
        "units_pnl": units_pnl,
        "roi": float(units_pnl / units_staked) if units_staked > 0 else float("nan"),
        "hit_rate": float((decided["result"] == "win").mean()) if len(decided) else float("nan"),
        "mean_ev": float(frame.loc[taken, "bet_ev"].mean()) if taken.any() else float("nan"),
        "min_ev": float(min_ev),
        "stake": float(stake),
    }
    return frame.reset_index(drop=True), summary
