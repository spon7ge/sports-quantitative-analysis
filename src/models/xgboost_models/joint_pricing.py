"""Price a paired full-game PTS market from joint simulation draws."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.odds import (
    american_to_implied_probability,
    expected_value,
    remove_vig,
)
from src.models.xgboost_models.joint_simulation import (
    JointPointsSimulator,
    JointSimulation,
)

PRICED = "PRICED"
NO_BET = "NO_BET"
SKIP_OUT = "SKIP_OUT"
SKIP_NO_FEATURE = "SKIP_NO_FEATURE"
UNSUPPORTED_INTEGER_LINE = "UNSUPPORTED_INTEGER_LINE"
ERROR = "ERROR"

IDENTITY_KEYS = (
    "book",
    "event",
    "player_id",
    "game_id",
    "stat",
    "period",
    "line",
    "quote_ts",
)


@dataclass(frozen=True)
class CanonicalPlayerPointsMarket:
    book: str
    event: str
    player_id: object
    game_id: object
    stat: str
    period: str
    line: float
    over_odds: float
    under_odds: float
    quote_ts: str


def canonicalize_player_points_market(
    over: dict,
    under: dict,
) -> CanonicalPlayerPointsMarket:
    if over.get("side") != "over" or under.get("side") != "under":
        raise ValueError("exact two-sided pairing requires over and under")
    for key in IDENTITY_KEYS:
        if over.get(key) != under.get(key):
            raise ValueError(f"over/under {key} do not match")
    if over.get("stat", "PTS") != "PTS":
        raise ValueError("only PTS is supported")
    if over.get("period", "full_game") != "full_game":
        raise ValueError("only full-game markets are supported")
    return CanonicalPlayerPointsMarket(
        book=str(over["book"]),
        event=str(over["event"]),
        player_id=over["player_id"],
        game_id=over["game_id"],
        stat="PTS",
        period="full_game",
        line=float(over["line"]),
        over_odds=float(over["odds"]),
        under_odds=float(under["odds"]),
        quote_ts=str(over["quote_ts"]),
    )


def price_joint_points_market(
    result: JointSimulation,
    market: CanonicalPlayerPointsMarket,
) -> dict:
    """Map one shared draw cloud onto a half-point PTS market."""
    line = float(market.line)
    if line.is_integer():
        return _terminal(
            UNSUPPORTED_INTEGER_LINE,
            market,
            result=result,
            reason="integer lines are not supported",
        )

    p_over = float(np.mean(result.point_draws > line))
    p_under = 1.0 - p_over
    over_raw = american_to_implied_probability(market.over_odds)
    under_raw = american_to_implied_probability(market.under_odds)
    over_vf, under_vf = remove_vig(market.over_odds, market.under_odds)
    over_ev = expected_value(p_over, p_under, market.over_odds)
    under_ev = expected_value(p_under, p_over, market.under_odds)
    over_edge = p_over - over_vf
    under_edge = p_under - under_vf

    payload = {
        "status": PRICED,
        "selected_side": None,
        "stat": "PTS",
        "period": "full_game",
        "line": line,
        "conditional_on_appearance": True,
        "over": _side(p_over, over_raw, over_vf, over_edge, over_ev),
        "under": _side(p_under, under_raw, under_vf, under_edge, under_ev),
        "hat_m": result.hat_m,
        "expected_minutes": result.expected_minutes,
        "hat_p": result.hat_p,
        "g": result.g,
        "mu": result.mu,
        "beta": result.beta,
        "bundle_hash": result.bundle_hash,
        "n_draws": result.n_draws,
        "seed": result.seed,
    }

    ranked = sorted(
        (("over", over_ev, over_edge), ("under", under_ev, under_edge)),
        key=lambda item: item[1],
        reverse=True,
    )
    best_side, best_ev, best_edge = ranked[0]
    if best_ev > 0:
        payload["selected_side"] = best_side
        payload["status"] = PRICED
    else:
        payload["status"] = NO_BET
        payload["selected_side"] = None
        if best_edge > 0:
            payload["reason"] = (
                "positive vig-free edge with nonpositive EV"
            )
    return payload


def price_one_player_points_line(
    simulator: JointPointsSimulator,
    row: pd.Series | dict,
    market: CanonicalPlayerPointsMarket,
    *,
    availability_probability: float = 1.0,
) -> dict:
    """Price one player line. Never places a wager."""
    if availability_probability <= 0:
        return _terminal(
            SKIP_OUT,
            market,
            reason="availability_probability=0; DNP voids",
        )
    try:
        result = simulator.simulate(row)
    except (KeyError, ValueError, TypeError) as exc:
        return _terminal(
            SKIP_NO_FEATURE,
            market,
            reason=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _terminal(ERROR, market, reason=str(exc))
    return price_joint_points_market(result, market)


def _side(
    model_probability: float,
    raw_implied: float,
    vig_free: float,
    edge: float,
    ev: float,
) -> dict[str, float]:
    return {
        "model_probability": float(model_probability),
        "raw_implied_probability": float(raw_implied),
        "vig_free_probability": float(vig_free),
        "edge": float(edge),
        "ev": float(ev),
    }


def _terminal(
    status: str,
    market: CanonicalPlayerPointsMarket,
    *,
    result: JointSimulation | None = None,
    reason: str | None = None,
) -> dict:
    payload = {
        "status": status,
        "selected_side": None,
        "stat": market.stat,
        "period": market.period,
        "line": market.line,
        "conditional_on_appearance": True,
        "reason": reason,
    }
    if result is not None:
        payload["bundle_hash"] = result.bundle_hash
    return payload
