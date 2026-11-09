"""Closing-line grading, staking, and leakage checks for the slate backtest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from betting_backtest import (
    BacktestConfig,
    assert_history_before_slate,
    build_wagers,
    load_closing_offers,
    run_strategies,
    unit_ev,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _team_payload() -> dict:
    return {
        "source": "Pinnacle",
        "league": "NBA",
        "date": "2026-11-05",
        "timezone": "America/Los_Angeles",
        "games": [
            {
                "game": "Denver Nuggets vs Indiana Pacers",
                "closing_fetched_at": "2026-11-06T05:00:00Z",
                "next_fetched_at": "2026-11-06T05:15:00Z",
                "opening_markets": [
                    {
                        "market": "moneyline",
                        "side": "Denver Nuggets",
                        "line": None,
                        "american_price": -500,
                    }
                ],
                "markets": [
                    {
                        "market": "moneyline",
                        "side": "Denver Nuggets",
                        "line": None,
                        "american_price": -150,
                    },
                    {
                        "market": "moneyline",
                        "side": "Indiana Pacers",
                        "line": None,
                        "american_price": 130,
                    },
                    {
                        "market": "spread",
                        "side": "Denver Nuggets",
                        "line": -3.0,
                        "american_price": -110,
                    },
                    {
                        "market": "spread",
                        "side": "Indiana Pacers",
                        "line": 3.0,
                        "american_price": -110,
                    },
                    {
                        "market": "total",
                        "side": "over",
                        "line": 220.5,
                        "american_price": -105,
                    },
                    {
                        "market": "total",
                        "side": "under",
                        "line": 220.5,
                        "american_price": -115,
                    },
                ],
            }
        ],
    }


def _props_root(tmp_path: Path) -> Path:
    root = tmp_path / "props"
    _write(
        root
        / "pinnacle"
        / "nba"
        / "team_closing_lines"
        / "pinnacle_nba_team_closing_2026-11-05.json",
        _team_payload(),
    )
    _write(
        root / "pinnacle" / "nba" / "closing" / "pinnacle_nba_closing_2026-11-05.json",
        {
            "source": "Pinnacle",
            "league": "NBA",
            "date": "2026-11-05",
            "timezone": "America/Los_Angeles",
            "players": [
                {
                    "player": "Nikola Jokic",
                    "closing_fetched_at": "2026-11-06T05:00:00Z",
                    "next_fetched_at": "2026-11-06T05:15:00Z",
                    "opening_lines": [
                        {
                            "stat_type": "points",
                            "line_score": 20.5,
                            "side": "over",
                            "american_price": 200,
                        }
                    ],
                    "lines": [
                        {
                            "stat_type": "points",
                            "line_score": 25.5,
                            "side": "over",
                            "american_price": -120,
                        },
                        {
                            "stat_type": "points",
                            "line_score": 25.5,
                            "side": "under",
                            "american_price": 100,
                        },
                    ],
                },
                {
                    "player": "Unpriced Player",
                    "closing_fetched_at": "2026-11-06T05:00:00Z",
                    "lines": [
                        {
                            "stat_type": "points",
                            "line_score": 10.5,
                            "side": "over",
                            "american_price": None,
                        },
                        {
                            "stat_type": "Points",
                            "line_score": 12.5,
                            "side": None,
                            "american_price": None,
                        },
                    ],
                },
            ],
        },
    )
    _write(
        root / "fanduel" / "nba" / "closing" / "fanduel_nba_closing_2026-11-05.json",
        {
            "source": "FanDuel",
            "league": "NBA",
            "date": "2026-11-05",
            "timezone": "America/Los_Angeles",
            "players": [
                {
                    "player": "Nikola Jokic",
                    "closing_fetched_at": "2026-11-06T05:00:00Z",
                    "lines": [
                        {
                            "stat_type": "points",
                            "line_score": 25.5,
                            "side": "over",
                            "american_price": -140,
                        }
                    ],
                }
            ],
        },
    )
    return root


def _sims() -> tuple[pd.DataFrame, pd.DataFrame]:
    home = np.array([110, 108, 100, 115], dtype=float)
    away = np.array([100, 105, 100, 112], dtype=float)
    games = pd.DataFrame(
        [
            {
                "slate_date": "2026-11-05",
                "game_id": "g1",
                "home_abbr": "DEN",
                "away_abbr": "IND",
                "history_through": "2026-11-04",
                "home_samples": home,
                "away_samples": away,
            }
        ]
    )
    points = np.array([30, 20, 26, 22], dtype=float)
    players = pd.DataFrame(
        [
            {
                "slate_date": "2026-11-05",
                "game_id": "g1",
                "player_name": "Nikola Jokic",
                "PTS": points,
                "REB": np.array([10, 11, 9, 12], dtype=float),
                "AST": np.array([8, 8, 7, 9], dtype=float),
            }
        ]
    )
    return games, players


def _actuals() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.DataFrame(
        [
            {
                "slate_date": "2026-11-05",
                "game_id": "g1",
                "home_abbr": "DEN",
                "away_abbr": "IND",
                "home_pts": 110,
                "away_pts": 100,
            }
        ]
    )
    players = pd.DataFrame(
        [
            {
                "slate_date": "2026-11-05",
                "game_id": "g1",
                "player_name": "Nikola Jokic",
                "PTS": 28,
                "REB": 12,
                "AST": 9,
            }
        ]
    )
    return games, players


def test_ev_matches_the_decimal_formula_when_the_line_cannot_push():
    # +100 is decimal 2. A 60% win rate with no push is 0.6 * 2 - 1.
    assert unit_ev(0.6, 0.4, 0.0, 100) == pytest.approx(0.2)


def test_push_probability_returns_the_stake_in_expected_value():
    # Half the mass pushes. EV = 0.25 * 2 - (1 - 0.5) = 0.
    assert unit_ev(0.25, 0.25, 0.5, 100) == pytest.approx(0.0)


def test_prizepicks_standard_lines_are_filled_at_minus_119(tmp_path: Path):
    root = tmp_path / "props"
    _write(
        root / "prizepicks" / "nba" / "closing" / "prizepicks_nba_closing_2026-11-05.json",
        {
            "source": "PrizePicks",
            "league": "NBA",
            "date": "2026-11-05",
            "timezone": "America/Los_Angeles",
            "players": [
                {
                    "player": "Nikola Jokic",
                    "closing_fetched_at": "2026-11-06T05:00:00Z",
                    "next_fetched_at": "2026-11-06T05:15:00Z",
                    "lines": [
                        {
                            "stat_type": "Points",
                            "line_score": 25.5,
                            "odds_type": "standard",
                            "side": None,
                            "american_price": None,
                        },
                        {
                            "stat_type": "Points",
                            "line_score": 22.5,
                            "odds_type": "goblin",
                            "side": None,
                            "american_price": None,
                        },
                        {
                            "stat_type": "Points",
                            "line_score": 28.5,
                            "odds_type": "demon",
                            "side": None,
                            "american_price": None,
                        },
                    ],
                }
            ],
        },
    )
    offers, audit = load_closing_offers(root, ("prizepicks",))
    priced = offers.sort_values("side")
    assert priced["side"].tolist() == ["over", "under"]
    assert priced["american"].tolist() == [-119.0, -119.0]
    assert priced["line"].tolist() == [25.5, 25.5]
    assert audit["unpriced"] == 2


def test_closing_price_is_used_and_the_open_is_ignored(tmp_path: Path):
    offers, audit = load_closing_offers(_props_root(tmp_path), ("pinnacle",))
    jokic = offers.loc[
        offers["market"].eq("points") & offers["selection"].eq("Nikola Jokic over")
    ]
    assert jokic["american"].tolist() == [-120]
    assert audit["unpriced"] == 2
    assert "opening" not in set(offers["price_source"])


def test_a_missing_book_is_not_filled_from_another_book(tmp_path: Path):
    offers, _audit = load_closing_offers(_props_root(tmp_path), ("pinnacle",))
    assert set(offers["book"]) == {"pinnacle"}
    fanduel, _audit = load_closing_offers(_props_root(tmp_path), ("fanduel",))
    assert set(fanduel["book"]) == {"fanduel"}
    assert fanduel.loc[
        fanduel["selection"].eq("Nikola Jokic over"), "american"
    ].tolist() == [-140]


def test_wrong_day_close_is_excluded(tmp_path: Path):
    root = _props_root(tmp_path)
    path = (
        root / "pinnacle" / "nba" / "closing" / "pinnacle_nba_closing_2026-11-05.json"
    )
    payload = json.loads(path.read_text())
    payload["players"][0]["closing_fetched_at"] = "2026-11-07T05:00:00Z"
    path.write_text(json.dumps(payload))
    offers, audit = load_closing_offers(root, ("pinnacle",))
    assert offers.loc[offers["player_name"].eq("Nikola Jokic")].empty
    assert audit["timestamp_mismatch"] >= 1


def test_probabilities_come_from_samples_and_settlement_from_actuals(tmp_path: Path):
    offers, _audit = load_closing_offers(_props_root(tmp_path), ("pinnacle",))
    games, players = _sims()
    actual_games, actual_players = _actuals()
    graded, audit = build_wagers(offers, games, players, actual_games, actual_players)
    assert set(graded["market"]).isdisjoint({"moneyline", "spread", "total"})
    jokic = graded.loc[
        graded["market"].eq("points") & graded["selection"].eq("Nikola Jokic over")
    ].iloc[0]
    # 30, 20, 26, 22 versus 25.5 -> two wins. Actual 28 is a win.
    assert jokic["model_probability"] == pytest.approx(0.5)
    assert jokic["result"] == "win"
    assert audit["graded"] == 2
    assert audit["offers_missing_prediction"] == 0


def test_sim_frame_with_actual_scores_is_rejected(tmp_path: Path):
    offers, _audit = load_closing_offers(_props_root(tmp_path), ("pinnacle",))
    games, players = _sims()
    games["home_pts"] = 110
    actual_games, actual_players = _actuals()
    with pytest.raises(ValueError, match="actual"):
        build_wagers(offers, games, players, actual_games, actual_players)


def test_history_on_the_slate_date_is_rejected():
    frame = pd.DataFrame(
        {"slate_date": ["2026-11-05"], "history_through": ["2026-11-05"]}
    )
    with pytest.raises(ValueError, match="history"):
        assert_history_before_slate(frame)


def test_duplicate_selection_is_counted_once(tmp_path: Path):
    root = _props_root(tmp_path)
    path = (
        root / "pinnacle" / "nba" / "closing" / "pinnacle_nba_closing_2026-11-05.json"
    )
    payload = json.loads(path.read_text())
    payload["players"][0]["lines"].append(
        {
            "stat_type": "points",
            "line_score": 25.5,
            "side": "over",
            "american_price": -160,
        }
    )
    path.write_text(json.dumps(payload))
    offers, audit = load_closing_offers(root, ("pinnacle",))
    jokic = offers.loc[
        offers["market"].eq("points") & offers["selection"].eq("Nikola Jokic over")
    ]
    assert len(jokic) == 1
    assert jokic["american"].iloc[0] == -120
    assert audit["duplicate_offers_dropped"] == 1


def test_flat_staking_caps_the_stake_and_returns_pushes():
    graded = pd.DataFrame(
        [
            _bet("2026-11-05", "pinnacle", "a", 0.50, 100, "loss"),
            _bet("2026-11-05", "pinnacle", "b", 0.20, 100, "loss"),
            _bet("2026-11-05", "pinnacle", "c", 0.01, 100, "win"),
        ]
    )
    config = BacktestConfig(
        start="2026-11-05",
        end="2026-11-05",
        bankroll=15,
        stake=10,
        stake_mode="flat",
        top_n=None,
        books=("pinnacle",),
        markets=("points",),
    )
    recommendations, comparison, _ledgers = run_strategies(graded, config)
    assert recommendations["stake"].tolist() == [10.0, 5.0]
    assert recommendations["profit"].tolist() == [-10.0, -5.0]
    row = comparison.loc[comparison["strategy"].eq("all")].iloc[0]
    assert row["total_bets"] == 2
    assert row["amount_wagered"] == 15
    assert row["profit"] == -15
    assert row["ending_bankroll"] == 0
    assert row["skipped_unfunded"] == 1

    pushed = pd.DataFrame([_bet("2026-11-05", "pinnacle", "a", 0.2, 100, "push")])
    _recommendations, comparison, _ledgers = run_strategies(pushed, config)
    row = comparison.loc[comparison["strategy"].eq("all")].iloc[0]
    assert row["pushes"] == 1
    assert row["profit"] == 0
    assert row["amount_wagered"] == 10
    assert row["ending_bankroll"] == 15


def test_books_and_strategies_keep_separate_bankrolls():
    rows = []
    for book in ("pinnacle", "fanduel"):
        for index, ev in enumerate((0.5, 0.4, 0.3, 0.2, 0.1, -0.2)):
            result = "win" if index == 5 else "loss"
            rows.append(_bet("2026-11-05", book, f"m{index}", ev, 100, result))
    graded = pd.DataFrame(rows)
    config = BacktestConfig(
        start="2026-11-05",
        end="2026-11-05",
        bankroll=100,
        stake=10,
        stake_mode="flat",
        top_n=5,
        books=("all",),
        markets=("points",),
    )
    _recommendations, comparison, _ledgers = run_strategies(graded, config)
    top5 = comparison.loc[comparison["strategy"].eq("top_5")]
    assert set(top5["book"]) == {"pinnacle", "fanduel"}
    assert top5["total_bets"].tolist() == [5, 5]
    assert top5["wins"].tolist() == [0, 0]
    assert top5["profit"].tolist() == [-50, -50]
    everything = comparison.loc[comparison["strategy"].eq("all")]
    assert everything["wins"].tolist() == [1, 1]
    assert everything["profit"].tolist() == [-40, -40]


def test_percent_staking_uses_the_bankroll_after_earlier_bets():
    graded = pd.DataFrame(
        [
            _bet("2026-11-05", "pinnacle", "a", 0.4, 100, "win"),
            _bet("2026-11-06", "pinnacle", "b", 0.3, 100, "win"),
        ]
    )
    config = BacktestConfig(
        start="2026-11-05",
        end="2026-11-06",
        bankroll=100,
        stake=0.10,
        stake_mode="percent",
        top_n=None,
        books=("pinnacle",),
        markets=("points",),
    )
    recommendations, comparison, _ledgers = run_strategies(graded, config)
    assert recommendations["stake"].tolist() == [10, 11]
    assert comparison.iloc[0]["ending_bankroll"] == pytest.approx(121)


def test_main_line_is_kept_and_book_side_labels_are_parsed(tmp_path: Path):
    root = _props_root(tmp_path)
    player_path = (
        root / "pinnacle" / "nba" / "closing" / "pinnacle_nba_closing_2026-11-05.json"
    )
    payload = json.loads(player_path.read_text())
    payload["players"][0]["lines"].extend(
        [
            {
                "stat_type": "points",
                "line_score": 0.5,
                "side": "over",
                "american_price": 470,
            },
            {
                "stat_type": "points_rebounds_assists",
                "line_score": 45.5,
                "side": "over",
                "american_price": -115,
            },
            {
                "stat_type": "points_rebounds_assists",
                "line_score": 45.5,
                "side": "under",
                "american_price": -105,
            },
        ]
    )
    player_path.write_text(json.dumps(payload))

    offers, audit = load_closing_offers(root, ("pinnacle",))
    points = offers.loc[
        offers["player_name"].eq("Nikola Jokic") & offers["market"].eq("points")
    ]
    assert sorted(points["line"].tolist()) == [25.5, 25.5]
    assert audit["alternate_lines_dropped"] == 1
    pra = offers.loc[offers["market"].eq("pra")]
    assert len(pra) == 2
    assert set(offers["market"]).isdisjoint({"moneyline", "spread", "total"})


def test_the_backtest_grades_player_props_only(tmp_path: Path):
    config = BacktestConfig(
        start="2026-11-05",
        end="2026-11-05",
        bankroll=100,
        stake=10,
    )
    assert "points" in config.markets
    assert set(config.markets).isdisjoint({"moneyline", "spread", "total"})
    with pytest.raises(ValueError, match="player prop"):
        BacktestConfig(
            start="2026-11-05",
            end="2026-11-05",
            bankroll=100,
            stake=10,
            markets=("moneyline",),
        )
    offers, _audit = load_closing_offers(_props_root(tmp_path), ("pinnacle",))
    assert set(offers["market"]).isdisjoint({"moneyline", "spread", "total"})


def test_config_rejects_a_nonpositive_bankroll():
    with pytest.raises(ValueError, match="bankroll"):
        BacktestConfig(
            start="2026-11-05",
            end="2026-11-05",
            bankroll=0,
            stake=10,
            books=("pinnacle",),
            markets=("points",),
        )


def _bet(day, book, selection, ev, american, result) -> dict:
    return {
        "slate_date": pd.Timestamp(day),
        "book": book,
        "game_id": "g1",
        "game": "IND @ DEN",
        "market": "points",
        "selection": selection,
        "line": np.nan,
        "model_probability": 0.5,
        "push_probability": 0.0,
        "implied_probability": 0.5,
        "american": american,
        "decimal_odds": 2.0 if american == 100 else 1.909090909,
        "ev": ev,
        "result": result,
        "actual_outcome": "DEN 110, IND 100",
        "unit_profit": {"win": 1.0, "loss": -1.0, "push": 0.0}[result],
    }
