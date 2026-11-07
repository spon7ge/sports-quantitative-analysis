from __future__ import annotations

import numpy as np
import pandas as pd
from game_enviroment.constants import OT_MIN, REG_MIN, SPREAD_EDGES
from game_enviroment.pace import predict_dev


def _col(df, name, default=0.0):
    if name not in df:
        return np.full(len(df), default)
    return pd.to_numeric(
        df[name], errors="coerce"
    ).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(default).to_numpy(float)

def fit_ot(df):
    overtime = df.n_ot.ge(1).astype(float)
    overall = float(overtime.mean())
    probabilities = []

    for low, high in zip(
        SPREAD_EDGES[:-1], SPREAD_EDGES[1:]
    ):
        bucket = (
            df.home_spread.abs().ge(low)
            & df.home_spread.abs().lt(high)
        )
        probabilities.append(float(
            (overtime[bucket].sum() + 50 * overall)
            / (bucket.sum() + 50)
        ))

    return {
        "p_ot_overall": overall,
        "p_ot_by_spread_bucket": probabilities,
        "p_more_given_ot": float(
            df.n_ot.ge(2).sum() / max(1, overtime.sum())
        ),
    }


def _ot_probability(table, spread):
    spread = np.asarray(spread, dtype=float)
    buckets = np.clip(
        np.searchsorted(
            SPREAD_EDGES, np.abs(spread), side="right"
        ) - 1,
        0,
        3,
    )
    return np.where(
        np.isfinite(spread),
        np.asarray(table["p_ot_by_spread_bucket"])[buckets],
        table["p_ot_overall"],
    )


def fit_split(df):
    counts = (
        (df.home_poss - df.away_poss)
        .round()
        .astype(int)
        .value_counts(normalize=True)
        .sort_index()
    )
    return {
        "values": counts.index.tolist(),
        "probs": counts.tolist(),
    }


def fit_scoring(df):
    poss = np.r_[
        df.home_poss.to_numpy(),
        df.away_poss.to_numpy(),
    ]
    pts = np.r_[
        df.home_pts.to_numpy(),
        df.away_pts.to_numpy(),
    ]
    off = np.r_[
        _col(df, "lineup_off_rating_h"),
        _col(df, "lineup_off_rating_a"),
    ]
    defense = np.r_[
        _col(df, "lineup_def_rating_a"),
        _col(df, "lineup_def_rating_h"),
    ]
    home = np.r_[np.ones(len(df)), np.zeros(len(df))]
    # A constant rating is the training-mean fallback. Leaving it in the
    # least squares problem splits the intercept arbitrarily.
    off_c = off - off.mean()
    def_c = defense - defense.mean()
    if np.nanstd(off_c) < 1e-8:
        off_c = np.zeros_like(off)
    if np.nanstd(def_c) < 1e-8:
        def_c = np.zeros_like(defense)

    x = np.column_stack([
        np.ones(len(poss)),
        off_c,
        def_c,
        home,
    ])
    beta = np.linalg.lstsq(x, pts / poss, rcond=None)[0]
    if np.nanstd(off) < 1e-8:
        beta[1] = 0.0
    if np.nanstd(defense) < 1e-8:
        beta[2] = 0.0
    return {
        "beta": beta.tolist(),
        "off_center": float(off.mean()),
        "def_center": float(defense.mean()),
        "market_weight": 0.5,
    }


def _ppp(scoring, off, defense, home):
    b = scoring["beta"]
    return np.maximum(
        0.05,
        b[0]
        + b[1] * (off - scoring["off_center"])
        + b[2] * (defense - scoring["def_center"])
        + b[3] * home,
    )


def _score_means(scoring, df, home_poss, away_poss):
    h = home_poss * _ppp(
        scoring,
        _col(df, "lineup_off_rating_h"),
        _col(df, "lineup_def_rating_a"),
        1,
    )
    a = away_poss * _ppp(
        scoring,
        _col(df, "lineup_off_rating_a"),
        _col(df, "lineup_def_rating_h"),
        0,
    )
    total, margin = h + a, h - a
    market_total = _col(df, "total", np.nan)
    market_margin = -_col(df, "home_spread", np.nan)
    weight = scoring["market_weight"]

    total = (
        (1 - weight) * total
        + weight * np.where(
            np.isfinite(market_total), market_total, total
        )
    )
    margin = (
        (1 - weight) * margin
        + weight * np.where(
            np.isfinite(market_margin), market_margin, margin
        )
    )
    return (total + margin) / 2.0, (total - margin) / 2.0


def _expected_possessions(model, df):
    pace = np.maximum(
        1.0,
        df.lg_poss.to_numpy() + predict_dev(model["model"], df),
    )
    more = model["ot"]["p_more_given_ot"]
    expected_ot = _ot_probability(
        model["ot"],
        _col(df, "home_spread", np.nan),
    ) * sum(more ** k for k in range(5))

    possessions = pace * (
        REG_MIN + OT_MIN * expected_ot
    ) / REG_MIN
    gap = np.dot(
        model["split"]["values"], model["split"]["probs"]
    )
    return possessions + gap / 2.0, possessions - gap / 2.0


def _paired_t(residual, shape, rng):
    nu = residual["nu"]
    scale = residual["scale"]
    rho = residual["rho"]
    z1, z2 = rng.standard_normal(shape), rng.standard_normal(shape)
    denominator = np.sqrt(rng.chisquare(nu, size=shape) / nu)

    return (
        scale * z1 / denominator,
        scale * (
            rho * z1 + np.sqrt(1 - rho ** 2) * z2
        ) / denominator,
    )


def _draw(model, df, n, rng):
    shape = (n, len(df))
    residual = model["residual_t"]
    pace = np.maximum(
        1.0,
        df.lg_poss.to_numpy()
        + predict_dev(model["model"], df)
        + residual["scale"] * rng.standard_t(
            residual["nu"], size=shape
        ),
    )

    overtime = model["ot"]
    n_ot = (
        rng.random(shape)
        < _ot_probability(
            overtime, _col(df, "home_spread", np.nan)
        )
    ).astype(int)
    for k in range(1, 5):
        n_ot += (
            (n_ot == k)
            & (rng.random(shape) < overtime["p_more_given_ot"])
        ).astype(int)

    minutes = REG_MIN + OT_MIN * n_ot
    possessions = pace * minutes / REG_MIN
    gap = rng.choice(
        model["split"]["values"],
        size=shape,
        p=model["split"]["probs"],
    )
    home_poss = np.maximum(1.0, possessions + gap / 2)
    away_poss = np.maximum(1.0, possessions - gap / 2)

    h, a = _score_means(
        model["scoring"], df, home_poss, away_poss
    )
    eh, ea = _paired_t(
        model["scoring"]["residual"], shape, rng
    )
    h, a = np.maximum(0.0, h + eh), np.maximum(0.0, a + ea)

    return {
        "home_poss": home_poss,
        "away_poss": away_poss,
        "pace": (
            (home_poss + away_poss) * REG_MIN / (2 * minutes)
        ),
        "home_pts": h,
        "away_pts": a,
        "game_total": h + a,
        "margin": h - a,
        "n_ot": n_ot,
        "game_minutes": minutes,
    }
