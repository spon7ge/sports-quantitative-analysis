"""Train the conditional minutes model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from game_state.constants import (
    PRE,
    RECENCY_DECAY,
    ROLE_EDGES,
    SCRIPT_EDGES,
    STATE,
)


def date_split(games, fraction):
    dates = sorted(games.game_date.unique())
    if not 0 < fraction < 1 or len(dates) < 3:
        raise ValueError(
            "Need fraction in (0, 1) and at least three game dates"
        )
    cutoff = dates[
        min(len(dates) - 1, max(1, int(len(dates) * fraction)))
    ]
    return (
        games[games.game_date < cutoff],
        games[games.game_date >= cutoff],
    )


def role_index(minutes):
    return np.searchsorted(
        ROLE_EDGES, np.asarray(minutes), side="right"
    )


def script_library(games):
    pairs = games[
        ["late_lead_min", "late_trail_min"]
    ].to_numpy(float)
    buckets = np.searchsorted(
        SCRIPT_EDGES,
        games.margin.to_numpy(),
        side="right",
    )
    return [
        pairs[buckets == i]
        if (buckets == i).sum() >= 20
        else pairs[
            np.argsort(
                np.abs(buckets - i), kind="stable"
            )[:min(40, len(pairs))]
        ]
        for i in range(len(SCRIPT_EDGES) + 1)
    ]


def new_model():
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=2,
        verbosity=-1,
    )


def train_models(d, games, test_season=None):
    if test_season is None:
        before, test_games = date_split(games, 0.75)
    else:
        before = games[games.season < test_season]
        test_games = games[games.season == test_season]

    train_games, validation_games = date_split(before, 0.8)
    tr, va, te = [
        d[d.gameId.isin(g.gameId)]
        for g in (train_games, validation_games, test_games)
    ]
    if len(tr) < 100 or min(len(va), len(te)) < 20:
        raise ValueError(
            "Not enough rows for training, calibration, and holdout"
        )

    weights = RECENCY_DECAY ** (
        tr.season.max() - tr.season
    )
    pre = new_model().fit(
        tr[PRE], tr.minutes, sample_weight=weights
    )
    model = new_model().fit(
        tr[PRE + STATE],
        tr.minutes,
        sample_weight=weights,
    )

    residuals = (
        va.minutes.to_numpy()
        - model.predict(va[PRE + STATE])
    )
    roles = role_index(va.roll_min)
    banks = [
        residuals[roles == i]
        if (roles == i).sum() >= 40 else residuals
        for i in range(3)
    ]

    bundle = dict(
        version=2,
        model=model,
        residuals=banks,
        script=script_library(train_games),
        cutoff=str(train_games.game_date.max().date()),
        features=PRE + STATE,
    )

    print(
        f"Minutes rows: train={len(tr)}, "
        f"calibration={len(va)}, holdout={len(te)}"
    )
    for label, prediction in [
        ("rolling average", te.roll_min),
        ("pregame GBM", pre.predict(te[PRE])),
        (
            "conditional GBM (observed state)",
            model.predict(te[PRE + STATE]),
        ),
    ]:
        mae = np.abs(
            np.asarray(prediction) - te.minutes.to_numpy()
        ).mean()
        print(f"  {label}: MAE {mae:.2f}")

    print(
        "Conditional MAE is not an end-to-end pregame forecast metric."
    )
    return bundle, test_games, te
