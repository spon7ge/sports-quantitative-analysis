"""Sportsbook settlement rules for player props and minutes.

Predictions are conditional on the player playing.

- DNP / zero-minute absences void the wager. They are not graded as
  zero points or zero minutes.
- Overtime counting stats and minutes are included.
- A player missing from an endpoint response is skipped, not scored
  as a zero-point outcome.
"""

from __future__ import annotations

import numpy as np

DNP_POLICY = "void"
OVERTIME_PERIOD_MINUTES = 5
MAXIMUM_OVERTIME_PERIODS = 3


class VoidedBetError(ValueError):
    """Raised when a player prop does not settle because of a DNP."""


def regulation_minutes(league: str) -> int:
    if league == "wnba":
        return 40

    return 48


def maximum_minutes(league: str) -> float:
    """Regulation length plus three overtime periods."""
    return float(
        regulation_minutes(league)
        + OVERTIME_PERIOD_MINUTES * MAXIMUM_OVERTIME_PERIODS
    )


def assert_bet_settles(
    *,
    availability_probability: float = 1.0,
    is_dnp: bool = False,
) -> float:
    """Return P(bet settles), or raise if the player is a DNP."""
    if is_dnp or availability_probability <= 0:
        raise VoidedBetError(
            "Player is a DNP; the wager voids instead of grading as zero."
        )

    if not 0 < availability_probability <= 1:
        raise ValueError(
            "availability_probability must be between 0 and 1"
        )

    return float(availability_probability)


def expected_counts_from_minutes(
    predicted_mean: np.ndarray,
    projected_minutes: np.ndarray,
    minute_samples: np.ndarray,
) -> np.ndarray:
    """Scale a mean prediction by sampled minutes.

    ``predicted_mean`` is the expected count at ``projected_minutes``.
    """
    means = np.asarray(predicted_mean, dtype=float)
    minutes = np.asarray(projected_minutes, dtype=float)
    samples = np.asarray(minute_samples, dtype=float)

    rate = means / np.maximum(minutes, 1e-6)

    return rate[:, None] * samples
