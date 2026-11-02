from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from src.models.dynamic_points_closing import (
    closing_book_summary,
    closing_ev_thresholds,
    load_main_points_markets,
    pregame_history_dates,
    score_closing_backtest,
    student_t_side_probabilities,
)


def _write_close(path: Path, players: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": "2026-10-24", "players": players}))


def _prediction(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "player_id": 1,
        "game_id": "g2",
        "game_date": pd.Timestamp("2026-10-24"),
        "predictive_mean": 20.0,
        "predictive_var": 25.0,
        "df": 1_000_000.0,
        "points": 22.0,
        "prior_player_game_date": pd.Timestamp("2026-10-22"),
        "prior_opponent_game_date": pd.Timestamp("2026-10-21"),
    }
    row.update(overrides)
    return row


def _market(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "slate_date": pd.Timestamp("2026-10-24"),
        "book": "pinnacle",
        "player_name": "Player 1",
        "player_id": 1,
        "line": 19.5,
        "over_american": -110,
        "under_american": -110,
    }
    row.update(overrides)
    return row


def test_main_line_keeps_the_balanced_rung_and_prizepicks_price(tmp_path: Path) -> None:
    _write_close(
        tmp_path / "novig" / "novig_nba_closing_2026-10-24.json",
        [
            {
                "player": "Player 1",
                "lines": [
                    {
                        "stat_type": "points",
                        "line_score": 25.5,
                        "side": "over",
                        "american_price": -300,
                    },
                    {
                        "stat_type": "points",
                        "line_score": 25.5,
                        "side": "under",
                        "american_price": 240,
                    },
                    {
                        "stat_type": "points",
                        "line_score": 20.5,
                        "side": "over",
                        "american_price": -110,
                    },
                    {
                        "stat_type": "points",
                        "line_score": 20.5,
                        "side": "under",
                        "american_price": -110,
                    },
                    {
                        "stat_type": "assists",
                        "line_score": 5.5,
                        "side": "over",
                        "american_price": -120,
                    },
                ],
            }
        ],
    )
    _write_close(
        tmp_path / "prizepicks" / "prizepicks_nba_closing_2026-10-24.json",
        [
            {
                "player": "Player 1",
                "lines": [
                    {
                        "stat_type": "Points",
                        "line_score": 16.5,
                        "odds_type": "standard",
                    },
                    {"stat_type": "Points", "line_score": 20.5, "odds_type": "demon"},
                ],
            }
        ],
    )

    markets, audit = load_main_points_markets(
        {
            "novig": tmp_path / "novig",
            "prizepicks": tmp_path / "prizepicks",
        }
    )

    novig = markets.loc[markets["book"].eq("novig")].iloc[0]
    prizepicks = markets.loc[markets["book"].eq("prizepicks")].iloc[0]
    assert novig["line"] == pytest.approx(20.5)
    assert prizepicks["line"] == pytest.approx(16.5)
    assert prizepicks["over_american"] == -119
    assert prizepicks["under_american"] == -119
    assert audit["markets"] == 2
    assert audit["one_sided_left_out"] == 0


def test_one_sided_and_duplicate_standard_lines_are_left_out(tmp_path: Path) -> None:
    _write_close(
        tmp_path / "novig" / "novig_nba_closing_2026-10-24.json",
        [
            {
                "player": "Player 1",
                "lines": [
                    {
                        "stat_type": "points",
                        "line_score": 10.5,
                        "side": "over",
                        "american_price": -110,
                    },
                ],
            }
        ],
    )
    _write_close(
        tmp_path / "prizepicks" / "prizepicks_nba_closing_2026-10-24.json",
        [
            {
                "player": "Player 2",
                "lines": [
                    {
                        "stat_type": "Points",
                        "line_score": 10.5,
                        "odds_type": "standard",
                    },
                    {
                        "stat_type": "Points",
                        "line_score": 12.5,
                        "odds_type": "standard",
                    },
                ],
            }
        ],
    )

    markets, audit = load_main_points_markets(
        {
            "novig": tmp_path / "novig",
            "prizepicks": tmp_path / "prizepicks",
        }
    )

    assert markets.empty
    assert audit["one_sided_left_out"] == 1
    assert audit["duplicate_standard_left_out"] == 1
    assert audit["markets"] == 0


def test_half_point_line_has_no_push_mass() -> None:
    over, under, push = student_t_side_probabilities(
        np.array([20.0]),
        np.array([5.0]),
        np.array([1_000_000.0]),
        np.array([19.5]),
    )
    expected = float(stats.norm.sf((19.5 - 20.0) / 5.0))
    assert over[0] == pytest.approx(expected, abs=1e-4)
    assert push[0] == pytest.approx(0.0)
    assert over[0] + under[0] == pytest.approx(1.0)


def test_integer_line_uses_a_half_point_continuity_correction() -> None:
    over, under, push = student_t_side_probabilities(
        np.array([10.0]),
        np.array([1.0]),
        np.array([1_000_000.0]),
        np.array([10.0]),
    )
    expected_over = float(stats.norm.sf(0.5))
    expected_under = float(stats.norm.cdf(-0.5))
    assert over[0] == pytest.approx(expected_over, abs=1e-4)
    assert under[0] == pytest.approx(expected_under, abs=1e-4)
    assert push[0] == pytest.approx(1.0 - expected_over - expected_under, abs=1e-4)


def test_pregame_dates_point_at_the_previous_game_only() -> None:
    frame = pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_id": "a",
                "game_date": "2026-10-22",
                "team_id": 10,
                "opp_team_id": 20,
            },
            {
                "player_id": 1,
                "game_id": "b",
                "game_date": "2026-10-24",
                "team_id": 10,
                "opp_team_id": 20,
            },
            {
                "player_id": 1,
                "game_id": "c",
                "game_date": "2026-10-24",
                "team_id": 10,
                "opp_team_id": 20,
            },
            {
                "player_id": 2,
                "game_id": "o1",
                "game_date": "2026-10-21",
                "team_id": 20,
                "opp_team_id": 10,
            },
            {
                "player_id": 3,
                "game_id": "o2",
                "game_date": "2026-10-24",
                "team_id": 20,
                "opp_team_id": 10,
            },
        ]
    )

    dates = pregame_history_dates(frame).set_index(["player_id", "game_id"])
    first = dates.loc[(1, "a")]
    assert pd.isna(first["prior_player_game_date"])
    second = dates.loc[(1, "b")]
    assert second["prior_player_game_date"] == pd.Timestamp("2026-10-22")
    assert second["prior_opponent_game_date"] == pd.Timestamp("2026-10-21")
    later_same_day = dates.loc[(1, "c")]
    assert later_same_day["prior_player_game_date"] == pd.Timestamp("2026-10-24")
    opponent_later = dates.loc[(1, "b"), "prior_opponent_game_date"]
    assert opponent_later < pd.Timestamp("2026-10-24")


def test_score_rejects_same_day_history_and_ignores_the_outcome() -> None:
    markets = pd.DataFrame(
        [
            _market(),
            _market(player_id=2, player_name="Same day"),
            _market(player_id=3, player_name="Opponent same day"),
            _market(
                player_id=4,
                player_name="Train window",
                slate_date=pd.Timestamp("2026-10-20"),
            ),
            _market(player_id=np.nan, player_name="Unknown"),
            _market(player_id=99, player_name="No appearance"),
        ]
    )
    predictions = pd.DataFrame(
        [
            _prediction(),
            _prediction(player_id=2, prior_player_game_date=pd.Timestamp("2026-10-24")),
            _prediction(
                player_id=3, prior_opponent_game_date=pd.Timestamp("2026-10-24")
            ),
            _prediction(player_id=4, game_date=pd.Timestamp("2026-10-20")),
        ]
    )

    scored, audit = score_closing_backtest(
        predictions,
        markets,
        train_end=pd.Timestamp("2026-10-20"),
    )
    rescored, _ = score_closing_backtest(
        predictions.assign(points=0.0),
        markets,
        train_end=pd.Timestamp("2026-10-20"),
    )

    assert scored["player_id"].tolist() == [1]
    assert scored["p_over"].iloc[0] == pytest.approx(rescored["p_over"].iloc[0])
    assert audit["scored"] == 1
    assert audit["same_day_player_excluded"] == 1
    assert audit["same_day_opponent_excluded"] == 1
    assert audit["train_window_excluded"] == 1
    assert audit["unmatched_names"] == 1
    assert audit["no_pregame_prediction"] == 1


def test_book_summary_compares_brier_with_the_no_vig_close() -> None:
    over, under, _ = student_t_side_probabilities(
        np.array([20.0]),
        np.array([5.0]),
        np.array([1_000_000.0]),
        np.array([19.5]),
    )
    scored, _ = score_closing_backtest(
        pd.DataFrame([_prediction(points=22.0)]),
        pd.DataFrame([_market()]),
        train_end=pd.Timestamp("2026-10-20"),
    )
    summary = closing_book_summary(scored)
    row = summary.iloc[0]

    assert row["book"] == "pinnacle"
    assert row["n"] == 1
    assert row["decided"] == 1
    assert row["over_rate"] == pytest.approx(1.0)
    assert row["mean_model_p_over"] == pytest.approx(over[0])
    assert row["model_brier"] == pytest.approx((over[0] - 1.0) ** 2)
    assert row["market_brier"] == pytest.approx(0.25)
    assert row["model_brier"] < row["market_brier"]
    assert row["favored_hit_rate"] == pytest.approx(1.0)
    assert row["favored_roi"] > 0
    assert under[0] < over[0]


def test_ev_thresholds_keep_bets_at_or_above_each_cutoff() -> None:
    scored = pd.DataFrame(
        {
            "book": ["pinnacle", "pinnacle", "novig", "novig"],
            "ev": [-0.02, 0.06, 0.15, 0.04],
            "profit": [-1.0, 1.0, -1.0, 0.0],
            "result": ["loss", "win", "loss", "push"],
        }
    )

    summary = closing_ev_thresholds(scored, thresholds=(0.0, 0.10))
    pooled = summary.loc[summary["book"].eq("all")].set_index("min_ev")
    pinnacle = summary.loc[summary["book"].eq("pinnacle")].set_index("min_ev")

    assert pooled.loc[0.0, "n"] == 3
    assert pooled.loc[0.0, "decided"] == 2
    assert pooled.loc[0.0, "pushes"] == 1
    assert pooled.loc[0.0, "hit_rate"] == pytest.approx(0.5)
    assert pooled.loc[0.0, "roi"] == pytest.approx(0.0)
    assert pooled.loc[0.10, "n"] == 1
    assert pooled.loc[0.10, "roi"] == pytest.approx(-1.0)
    assert pinnacle.loc[0.0, "n"] == 1
    assert pinnacle.loc[0.0, "roi"] == pytest.approx(1.0)
    assert pinnacle.loc[0.10, "n"] == 0
    assert pd.isna(pinnacle.loc[0.10, "roi"])
