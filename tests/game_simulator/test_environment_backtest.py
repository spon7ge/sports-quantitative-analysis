"""Chronological backtest of the shared game-environment draw."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from game_enviroment import simulate_game
from game_enviroment.backtest import (
    assign_splits,
    fit_frozen_model,
    implied_market,
    market_comparison,
    summarize_pbp,
    team_view,
)
from game_enviroment.features import _line_feats, build_features
from game_enviroment.script import _draw


def test_split_keeps_a_date_together_and_ignores_test_outcomes():
    games = _games()
    split = assign_splits(games, train_frac=0.6, val_frac=0.2)
    counts = split.groupby("game_date").split.nunique()
    assert counts.max() == 1
    assert set(split.split) == {"train", "val", "test"}

    train = split.loc[split.split.eq("train")].copy()
    val = split.loc[split.split.eq("val")].copy()
    first = fit_frozen_model(train, val, features=_base_features())
    corrupted = split.copy()
    test_rows = corrupted.split.eq("test")
    corrupted.loc[test_rows, "y_dev"] = 80.0
    corrupted.loc[test_rows, "home_pts"] = 200.0
    corrupted.loc[test_rows, "away_pts"] = 70.0
    corrupted.loc[test_rows, "poss48"] = 130.0
    second = fit_frozen_model(
        corrupted.loc[corrupted.split.eq("train")].copy(),
        corrupted.loc[corrupted.split.eq("val")].copy(),
        features=_base_features(),
    )
    assert first["model"]["beta"] == second["model"]["beta"]
    assert first["scoring"]["market_weight"] == second["scoring"]["market_weight"]

    early = train.iloc[[0]]
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    draw_a = _draw(first, early, 20, rng_a)
    draw_b = _draw(second, early, 20, rng_b)
    assert np.allclose(draw_a["home_pts"], draw_b["home_pts"])


def test_future_outcomes_do_not_change_earlier_features():
    games = _feature_games()
    left, _ = build_features(games, 8.0, 4.0, 1.0, 50, 1)
    changed = games.copy()
    changed.loc[changed.game_date.eq(changed.game_date.max()), "poss48"] = 130.0
    changed.loc[changed.game_date.eq(changed.game_date.max()), "pts_total"] = 300.0
    right, _ = build_features(changed, 8.0, 4.0, 1.0, 50, 1)
    early = left.game_date.lt(left.game_date.max())
    assert np.allclose(
        left.loc[early, "pace_h"], right.loc[early, "pace_h"], equal_nan=True
    )
    assert np.allclose(
        left.loc[early, "lg_poss"], right.loc[early, "lg_poss"], equal_nan=True
    )


def test_views_swap_and_reconcile_possessions_with_duration():
    frame = pd.DataFrame(
        {
            "lg_poss": [100.0],
            "home_spread": [0.0],
            "total": [np.nan],
            "lineup_off_rating_h": [110.0],
            "lineup_off_rating_a": [110.0],
            "lineup_def_rating_h": [110.0],
            "lineup_def_rating_a": [110.0],
        }
    )
    model = {
        "residual_t": {"nu": 30.0, "scale": 0.0},
        "model": {"name": "league_prior", "team_ids": [], "beta": [], "lam": 0.0},
        "ot": {
            "p_ot_overall": 1.0,
            "p_ot_by_spread_bucket": [1.0, 1.0, 1.0, 1.0],
            "p_more_given_ot": 0.0,
        },
        "split": {"values": [4.0], "probs": [1.0]},
        "scoring": {
            "beta": [1.0, 0.0, 0.0, 0.0],
            "off_center": 110.0,
            "def_center": 110.0,
            "market_weight": 0.0,
            "residual": {"nu": 30.0, "scale": 0.0, "rho": 0.0},
        },
    }
    samples = _draw(model, frame, 5, np.random.default_rng(1))
    home = team_view(samples, "home")
    away = team_view(samples, "away")

    assert np.allclose(home["team_pts"], away["opp_pts"])
    assert np.allclose(home["opp_pts"], away["team_pts"])
    assert np.allclose(home["team_poss"], away["opp_poss"])
    assert np.allclose(home["team_margin"], -away["team_margin"])
    assert np.allclose(home["game_total"], home["team_pts"] + home["opp_pts"])
    assert np.allclose(home["team_margin"], home["team_pts"] - home["opp_pts"])
    assert np.allclose(home["pace"], away["pace"])
    minutes = samples["game_minutes"]
    pace = (samples["home_poss"] + samples["away_poss"]) * 48.0 / (2.0 * minutes)
    assert np.allclose(samples["pace"], pace)
    assert np.allclose(minutes, 48.0 + 5.0 * samples["n_ot"])
    full = (samples["home_poss"] + samples["away_poss"]) / 2.0
    assert np.allclose(full, samples["pace"] * minutes / 48.0)


def test_market_lines_and_missing_market_fallback():
    prices = implied_market(home_spread=-6.0, total=220.0)
    assert prices["market_home_margin"] == pytest.approx(6.0)
    assert prices["market_home_pts"] == pytest.approx(113.0)
    assert prices["market_away_pts"] == pytest.approx(107.0)

    missing = _line_feats(None, None, 220.0)
    assert missing["total_missing"] == 1.0
    assert missing["spread_missing"] == 1.0
    assert missing["total_dev"] == 0.0
    assert missing["abs_spread"] == 0.0

    scored = market_comparison(
        pd.DataFrame(
            {
                "home_spread": [-6.0, np.nan],
                "total": [220.0, np.nan],
                "pred_home_pts": [113.0, 50.0],
                "pred_away_pts": [107.0, 50.0],
                "pred_total": [220.0, 100.0],
                "pred_margin": [6.0, 0.0],
                "actual_home_pts": [112.0, 90.0],
                "actual_away_pts": [108.0, 80.0],
                "actual_total": [220.0, 170.0],
                "actual_margin": [4.0, 10.0],
            }
        )
    )
    assert scored["n"] == 1
    assert scored["model_margin_mae"] == pytest.approx(2.0)
    assert scored["market_margin_mae"] == pytest.approx(2.0)


def test_one_environment_accepts_a_shared_rotation_scenario_per_draw():
    from lineup import LINEUP_GAME_FEATURES

    fill = {name: 100.0 for name in LINEUP_GAME_FEATURES}
    model = {
        "version": 3,
        "config": {"season_carry": 1.0, "prior_weight": 4.0},
        "lineup_fill": fill,
        "state": {
            "league_poss": 100.0,
            "league_pts": 220.0,
            "teams": {
                "1": {
                    "abbr": "LAL",
                    "num": 0.0,
                    "den": 0.0,
                    "last_date": "2026-04-01",
                    "last_season": "2025-26",
                },
                "2": {
                    "abbr": "BOS",
                    "num": 0.0,
                    "den": 0.0,
                    "last_date": "2026-04-01",
                    "last_season": "2025-26",
                },
            },
            "lineup": {
                "players": {},
                "league": {"pace": 100.0, "off_rating": 110.0, "def_rating": 110.0},
                "prior_weight": 4.0,
                "season_carry": 1.0,
                "fill": fill,
            },
        },
        "residual_t": {"nu": 30.0, "scale": 0.0},
        "model": {
            "name": "league_prior",
            "team_ids": [],
            "beta": [],
            "lam": 0.0,
            "features": [],
        },
        "ot": {
            "p_ot_overall": 0.0,
            "p_ot_by_spread_bucket": [0.0, 0.0, 0.0, 0.0],
            "p_more_given_ot": 0.0,
        },
        "split": {"values": [0.0], "probs": [1.0]},
        "scoring": {
            "beta": [1.1, 0.0, 0.0, 0.0],
            "off_center": 110.0,
            "def_center": 110.0,
            "market_weight": 0.0,
            "residual": {"nu": 30.0, "scale": 0.0, "rho": 0.0},
        },
    }
    plain = simulate_game(
        model, "LAL", "BOS", "2026-11-05", n=3, seed=1, total=None, spread=None
    )
    scenarios = [
        {
            "rotation_pace_h": 100.0,
            "rotation_pace_a": 98.0,
            "rotation_off_rating_h": 110.0,
            "rotation_off_rating_a": 108.0,
            "rotation_def_rating_h": 110.0,
            "rotation_def_rating_a": 109.0,
        },
        {
            "rotation_pace_h": 104.0,
            "rotation_pace_a": 96.0,
            "rotation_off_rating_h": 120.0,
            "rotation_off_rating_a": 100.0,
            "rotation_def_rating_h": 108.0,
            "rotation_def_rating_a": 112.0,
        },
    ]
    drawn = simulate_game(
        model,
        "LAL",
        "BOS",
        "2026-11-05",
        n=2,
        seed=1,
        total=None,
        spread=None,
        rotation_features=scenarios,
        rotation_context={"shared": True},
    )

    assert len(plain) == 3
    assert len(drawn) == 2
    assert np.allclose(drawn["game_total"], drawn["team_pts"] + drawn["opp_pts"])
    assert np.allclose(drawn["team_margin"], drawn["team_pts"] - drawn["opp_pts"])
    assert drawn.attrs["rotation"]["shared"] is True


def test_pbp_keeps_regulation_and_full_game_scores_apart():
    pbp = pd.DataFrame(
        {
            "gameId": ["0022400001", "0022400001", "0022400001", "0022400001"],
            "actionNumber": [1, 2, 3, 4],
            "period": [4, 4, 5, 5],
            "scoreHome": ["90", "", "100", "110"],
            "scoreAway": ["88", "", "100", "108"],
        }
    )
    out = summarize_pbp(pbp)
    row = out.iloc[0]
    assert row.game_id == "0022400001"
    assert row.reg_home == 90
    assert row.reg_away == 88
    assert row.final_home == 110
    assert row.final_away == 108
    assert row.n_ot == 1
    assert row.reg_home + row.reg_away != row.final_home + row.final_away


def _base_features():
    return [
        "pace_h",
        "pace_a",
        "b2b_h",
        "b2b_a",
        "rested_h",
        "rested_a",
        "total_dev",
        "abs_spread",
        "total_missing",
        "spread_missing",
    ]


def _games():
    rng = np.random.default_rng(0)
    rows = []
    dates = pd.date_range("2023-10-01", periods=40, freq="D")
    for date in dates:
        for kick in range(4):
            pace_h = float(rng.normal())
            pace_a = float(rng.normal())
            poss = 100.0 + 2.0 * pace_h
            rows.append(
                {
                    "game_id": f"{date.date()}-{kick}",
                    "game_date": date,
                    "home_id": 1 + (kick % 3),
                    "away_id": 4 + (kick % 3),
                    "pace_h": pace_h,
                    "pace_a": pace_a,
                    "b2b_h": 0.0,
                    "b2b_a": 0.0,
                    "rested_h": 1.0,
                    "rested_a": 0.0,
                    "total_dev": 0.0,
                    "abs_spread": 3.0,
                    "total_missing": 0.0,
                    "spread_missing": 0.0,
                    "lg_poss": 100.0,
                    "poss48": poss,
                    "y_dev": poss - 100.0,
                    "home_poss": poss,
                    "away_poss": 100.0,
                    "home_pts": 1.1 * poss,
                    "away_pts": 110.0,
                    "margin": 1.1 * poss - 110.0,
                    "n_ot": 0,
                    "total": 220.0,
                    "home_spread": -3.0,
                    "pts_total": 1.1 * poss + 110.0,
                    "lineup_off_rating_h": 110.0,
                    "lineup_off_rating_a": 110.0,
                    "lineup_def_rating_h": 110.0,
                    "lineup_def_rating_a": 110.0,
                }
            )
    return pd.DataFrame(rows)


def _feature_games():
    dates = pd.date_range("2024-01-01", periods=6, freq="2D")
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "game_id": f"g{i}",
                "game_date": date,
                "season_year": "2023-24",
                "home_id": 1,
                "away_id": 2,
                "home_abbr": "HOM",
                "away_abbr": "AWY",
                "poss48": 100.0 + i,
                "pts_total": 220.0,
                "total": 220.0,
                "home_spread": -3.0,
            }
        )
    return pd.DataFrame(rows)
