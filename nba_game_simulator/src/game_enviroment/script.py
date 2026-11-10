from __future__ import annotations

import numpy as np
import pandas as pd
from game_enviroment.constants import OT_MIN, REG_MIN, SPREAD_EDGES
from game_enviroment.pace import predict_dev
from game_enviroment.pbp_script import (
    late_checkpoints,
    margin_bucket,
    ot_path_probability,
)


def _col(df, name, default=0.0):
    if name not in df:
        return np.full(len(df), default)
    return (
        pd.to_numeric(df[name], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .to_numpy(float)
    )


def fit_ot(df):
    overtime = df.n_ot.ge(1).astype(float)
    overall = float(overtime.mean())
    probabilities = []

    for low, high in zip(SPREAD_EDGES[:-1], SPREAD_EDGES[1:]):
        bucket = df.home_spread.abs().ge(low) & df.home_spread.abs().lt(high)
        probabilities.append(
            float((overtime[bucket].sum() + 50 * overall) / (bucket.sum() + 50))
        )

    return {
        "p_ot_overall": overall,
        "p_ot_by_spread_bucket": probabilities,
        "p_more_given_ot": float(df.n_ot.ge(2).sum() / max(1, overtime.sum())),
    }


def _ot_probability(table, spread):
    spread = np.asarray(spread, dtype=float)
    buckets = np.clip(
        np.searchsorted(SPREAD_EDGES, np.abs(spread), side="right") - 1,
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

    x = np.column_stack(
        [
            np.ones(len(poss)),
            off_c,
            def_c,
            home,
        ]
    )
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

    total = (1 - weight) * total + weight * np.where(
        np.isfinite(market_total), market_total, total
    )
    margin = (1 - weight) * margin + weight * np.where(
        np.isfinite(market_margin), market_margin, margin
    )
    return (total + margin) / 2.0, (total - margin) / 2.0


def margin_paths(final, game_minutes, h1=None, regulation=None):
    """One margin per minute, ending on the published final margin.

    With a halftime and a regulation margin, the path runs from a tie to
    halftime, then to the end of regulation, then through overtime. Without
    those splits the path stays at the final margin for the whole game.
    Positive means the home team leads.
    """

    final = np.asarray(final, dtype=float).reshape(-1)
    minutes = np.asarray(game_minutes, dtype=float).reshape(-1)
    if final.shape != minutes.shape:
        raise ValueError("final margin and game length must align")
    flat = h1 is None
    if h1 is None:
        h1 = final
        regulation = final
    else:
        h1 = np.asarray(h1, dtype=float).reshape(-1)
        regulation = np.asarray(regulation, dtype=float).reshape(-1)
        if h1.shape != final.shape or regulation.shape != final.shape:
            raise ValueError("half margins must align with the final margin")
    paths = []
    for start, middle, end, length in zip(h1, regulation, final, minutes):
        ticks = int(np.rint(length))
        if ticks < REG_MIN or (ticks - REG_MIN) % OT_MIN != 0:
            raise ValueError(
                "game length must be 48 minutes plus whole overtime periods"
            )
        if not np.isfinite([start, middle, end]).all():
            raise ValueError("margins must be finite")
        if flat:
            paths.append(np.full(ticks, end, dtype=float))
            continue
        path = np.empty(ticks, dtype=float)
        for tick in range(ticks):
            if tick <= 24:
                path[tick] = (tick / 24.0) * start
            elif tick <= REG_MIN:
                path[tick] = start + ((tick - 24) / 24.0) * (middle - start)
            else:
                span = ticks - REG_MIN
                path[tick] = middle + ((tick - REG_MIN) / span) * (end - middle)
        path[-1] = end
        paths.append(path)
    return paths


def _expected_possessions(model, df):
    pace = np.maximum(
        1.0,
        df.lg_poss.to_numpy() + predict_dev(model["model"], df),
    )
    more = model["ot"]["p_more_given_ot"]
    extra = sum(more**k for k in range(5))
    if "pace_state" in model and "ot_path" in model:
        spread = _col(df, "home_spread", np.nan)
        market_margin = np.where(np.isfinite(spread), -spread, 0.0)
        opening = np.where(np.isfinite(spread), 0.5 * np.abs(spread), 0.0)
        h1_mult, h2_mult = _half_multipliers(
            model["pace_state"], opening, 0.5 * market_margin
        )
        margin_6, margin_2 = late_checkpoints(0.5 * market_margin, market_margin)
        expected_ot = (
            ot_path_probability(model["ot_path"], margin_6, margin_2, market_margin)
            * extra
        )
        possessions = (
            pace * 0.5 * (h1_mult + h2_mult) + pace * (OT_MIN / REG_MIN) * expected_ot
        )
    else:
        expected_ot = (
            _ot_probability(
                model["ot"],
                _col(df, "home_spread", np.nan),
            )
            * extra
        )
        possessions = pace * (REG_MIN + OT_MIN * expected_ot) / REG_MIN
    gap = np.dot(model["split"]["values"], model["split"]["probs"])
    return possessions + gap / 2.0, possessions - gap / 2.0


def _paired_t(residual, shape, rng):
    nu = residual["nu"]
    scale = residual["scale"]
    rho = residual["rho"]
    z1, z2 = rng.standard_normal(shape), rng.standard_normal(shape)
    denominator = np.sqrt(rng.chisquare(nu, size=shape) / nu)

    return (
        scale * z1 / denominator,
        scale * (rho * z1 + np.sqrt(1 - rho**2) * z2) / denominator,
    )


def _draw(model, df, n, rng):
    shape = (n, len(df))
    residual = model["residual_t"]
    pace = np.maximum(
        1.0,
        df.lg_poss.to_numpy()
        + predict_dev(model["model"], df)
        + residual["scale"] * rng.standard_t(residual["nu"], size=shape),
    )
    if "pace_state" in model:
        return _draw_state(model, df, rng, pace)

    overtime = model["ot"]
    n_ot = (
        rng.random(shape) < _ot_probability(overtime, _col(df, "home_spread", np.nan))
    ).astype(int)
    for k in range(1, 5):
        n_ot += (
            (n_ot == k) & (rng.random(shape) < overtime["p_more_given_ot"])
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

    h, a = _score_means(model["scoring"], df, home_poss, away_poss)
    eh, ea = _paired_t(model["scoring"]["residual"], shape, rng)
    h, a = np.maximum(0.0, h + eh), np.maximum(0.0, a + ea)

    return {
        "home_poss": home_poss,
        "away_poss": away_poss,
        "pace": ((home_poss + away_poss) * REG_MIN / (2 * minutes)),
        "home_pts": h,
        "away_pts": a,
        "game_total": h + a,
        "margin": h - a,
        "n_ot": n_ot,
        "game_minutes": minutes,
    }


def _half_multipliers(state, h1_margin, h2_margin):
    multiplier = np.asarray(state["multiplier"], dtype=float)
    edges = state["edges"]
    return (
        multiplier[0, margin_bucket(h1_margin, edges)],
        multiplier[1, margin_bucket(h2_margin, edges)],
    )


def _draw_state(model, df, rng, pace):
    """Split the game at half, then let the lead change the second-half pace.

    Overtime uses the simulated margin at 6:00 and 2:00 and the regulation
    score. The final-score table was counted on completed games; regulation
    is the score known before overtime points exist.
    """

    if "ot_path" not in model:
        raise ValueError("pace_state requires ot_path")

    shape = pace.shape
    spread = _col(df, "home_spread", np.nan)
    opening = np.where(np.isfinite(spread), 0.5 * np.abs(spread), 0.0)
    h1_mult, _ = _half_multipliers(model["pace_state"], opening, opening)
    h1_shared = pace * 0.5 * h1_mult
    gap = rng.choice(model["split"]["values"], size=shape, p=model["split"]["probs"])
    scoring = model["scoring"]
    ppp_home = _ppp(
        scoring,
        _col(df, "lineup_off_rating_h"),
        _col(df, "lineup_def_rating_a"),
        1,
    )
    ppp_away = _ppp(
        scoring,
        _col(df, "lineup_off_rating_a"),
        _col(df, "lineup_def_rating_h"),
        0,
    )

    def _score(poss_home, poss_away):
        shock_home, shock_away = _paired_t(scoring["residual"], shape, rng)
        scale = np.sqrt(
            np.maximum(0.5 * (poss_home + poss_away), 0.0) / np.maximum(pace, 1e-6)
        )
        return (
            np.maximum(0.0, ppp_home * poss_home + shock_home * scale),
            np.maximum(0.0, ppp_away * poss_away + shock_away * scale),
        )

    h1_home, h1_away = _score(h1_shared, h1_shared)
    _, h2_mult = _half_multipliers(model["pace_state"], opening, h1_home - h1_away)
    h2_shared = pace * 0.5 * h2_mult
    home_h2 = np.maximum(0.0, h2_shared + gap / 2.0)
    away_h2 = np.maximum(0.0, h2_shared - gap / 2.0)
    h2_home, h2_away = _score(home_h2, away_h2)
    regulation = (h1_home + h2_home) - (h1_away + h2_away)
    margin_6, margin_2 = late_checkpoints(h1_home - h1_away, regulation)
    # The lookup's third argument is the regulation margin, not a post-OT score.
    probability = ot_path_probability(model["ot_path"], margin_6, margin_2, regulation)
    overtime = model["ot"]
    n_ot = (rng.random(shape) < probability).astype(int)
    for k in range(1, 5):
        n_ot += (
            (n_ot == k) & (rng.random(shape) < overtime["p_more_given_ot"])
        ).astype(int)

    ot_shared = pace * (OT_MIN / REG_MIN) * n_ot
    ot_home, ot_away = _score(ot_shared, ot_shared)
    home_poss = h1_shared + home_h2 + ot_shared
    away_poss = h1_shared + away_h2 + ot_shared
    raw_home = h1_home + h2_home + ot_home
    raw_away = h1_away + h2_away + ot_away
    mean_home = ppp_home * home_poss
    mean_away = ppp_away * away_poss
    weight = float(scoring["market_weight"])
    market_total = _col(df, "total", np.nan)
    market_margin = -spread
    total = (raw_home + raw_away) + weight * np.where(
        np.isfinite(market_total),
        market_total - (mean_home + mean_away),
        0.0,
    )
    margin = (raw_home - raw_away) + weight * np.where(
        np.isfinite(market_margin),
        market_margin - (mean_home - mean_away),
        0.0,
    )
    home = np.maximum(0.0, (total + margin) / 2.0)
    away = np.maximum(0.0, (total - margin) / 2.0)
    minutes = REG_MIN + OT_MIN * n_ot
    published = home - away
    raw_margin = raw_home - raw_away
    scale = np.ones_like(published)
    usable = np.abs(raw_margin) > 1e-6
    scale[usable] = published[usable] / raw_margin[usable]
    h1_margin = (h1_home - h1_away) * scale
    regulation_margin = regulation * scale
    regulation_margin = np.where(n_ot == 0, published, regulation_margin)
    return {
        "home_poss": home_poss,
        "away_poss": away_poss,
        "pace": (home_poss + away_poss) * REG_MIN / (2 * minutes),
        "home_pts": home,
        "away_pts": away,
        "game_total": home + away,
        "margin": published,
        "h1_margin": h1_margin,
        "regulation_margin": regulation_margin,
        "n_ot": n_ot,
        "game_minutes": minutes,
    }
