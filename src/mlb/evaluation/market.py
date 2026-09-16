"""Paired model vs de-vigged quote comparison (quoted-price simulation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.markets.odds import expected_profit, implied_to_decimal, no_vig
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
