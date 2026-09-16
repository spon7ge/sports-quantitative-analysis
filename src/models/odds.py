"""Sportsbook probability and expected-value utilities."""

from __future__ import annotations

import numpy as np


def american_to_decimal(
    odds: float,
) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be zero.")

    if odds > 0:
        return 1 + odds / 100

    return 1 + 100 / abs(odds)


def american_to_implied_probability(
    odds: float,
) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be zero.")

    if odds > 0:
        return 100 / (odds + 100)

    return abs(odds) / (abs(odds) + 100)


def remove_vig(
    over_odds: float,
    under_odds: float,
) -> tuple[float, float]:
    over_raw = american_to_implied_probability(over_odds)
    under_raw = american_to_implied_probability(under_odds)

    total = over_raw + under_raw

    return over_raw / total, under_raw / total


def expected_value(
    win_probability: float,
    loss_probability: float,
    american_odds: float,
) -> float:
    """Expected profit per dollar wagered."""
    profit_on_win = (
        american_to_decimal(american_odds) - 1
    )

    return (
        win_probability * profit_on_win
        - loss_probability
    )


def fractional_kelly(
    model_probability: float,
    american_odds: float,
    *,
    fraction: float = 0.25,
    maximum_bet: float = 0.02,
) -> float:
    """Return bankroll fraction for fractional Kelly sizing."""
    decimal_odds = american_to_decimal(
        american_odds
    )
    net_odds = decimal_odds - 1
    loss_probability = 1 - model_probability

    full_kelly = (
        net_odds * model_probability
        - loss_probability
    ) / net_odds

    return float(
        np.clip(
            full_kelly * fraction,
            0,
            maximum_bet,
        )
    )
