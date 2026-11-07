from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from game_boxscore import portfolio_risk, price_over
from game_boxscore.types import BoxScoreSet, Prop, TeamBoxScore
from src.models.odds import expected_value


def _box() -> BoxScoreSet:
    return BoxScoreSet(
        teams={
            "home": TeamBoxScore(
                player_ids=("A", "B"),
                stats={
                    "PTS": np.array([[10, 10], [20, 30], [30, 30]]),
                    "REB": np.array([[4, 8], [4, 8], [4, 8]]),
                },
            )
        },
        n_sims=3,
        environment=pd.DataFrame({"game_minutes": [48, 48, 48]}),
    )


def test_price_over_counts_a_push_as_neither_win_nor_loss() -> None:
    priced = price_over(_box(), Prop("home", "A", "PTS", 20), -119)

    assert priced.p_over == pytest.approx(1 / 3)
    assert priced.p_under == pytest.approx(1 / 3)
    assert priced.p_push == pytest.approx(1 / 3)
    assert priced.ev == pytest.approx(expected_value(1 / 3, 1 / 3, -119))


def test_portfolio_risk_uses_the_same_rows_for_every_leg() -> None:
    box = _box()
    props = [
        Prop("home", "A", "PTS", 20.5),
        Prop("home", "B", "PTS", 20.5),
    ]
    priced = portfolio_risk(box, props, [-119, -119])
    win_profit = 100 / 119
    profits = np.array([-2.0, win_profit - 1.0, 2 * win_profit])

    assert priced.n == 2
    assert priced.expected_profit == pytest.approx(float(profits.mean()))
    assert priced.risk == pytest.approx(float(profits.std()))
    assert priced.p_loss == pytest.approx(2 / 3)
    assert priced.joint == pytest.approx(1 / 3)
    assert priced.independent_product == pytest.approx((1 / 3) * (2 / 3))


def test_portfolio_rejects_an_empty_book() -> None:
    with pytest.raises(ValueError, match="at least one"):
        portfolio_risk(_box(), [], [])
