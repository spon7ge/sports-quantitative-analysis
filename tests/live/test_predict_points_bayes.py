"""Tests for Bayesian live slate feature construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.live.predict_points_bayes import (
    build_bayesian_live_features,
    choose_live_slate_date,
)


def _history(n_games: int = 22) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-10-01", tz="UTC")
    for game in range(n_games):
        day = start + pd.Timedelta(days=game)
        rows.append(
            {
                "player_id": 1,
                "player_name": "Scoot Henderson",
                "team_id": 10,
                "team_abbreviation": "POR",
                "game_id": f"g{game}",
                "game_date": day,
                "pts": 15 + game % 5,
                "minutes": 30.0,
                "usage_pp": 24.0,
                "starter": 1.0 if game % 2 == 0 else 0.0,
                "fga": 12.0,
                "fta": 3.0,
                "team_def_rating": 110.0 + (game % 3),
            }
        )
        rows.append(
            {
                "player_id": 2,
                "player_name": "Ja Morant",
                "team_id": 20,
                "team_abbreviation": "MEM",
                "game_id": f"g{game}",
                "game_date": day,
                "pts": 22,
                "minutes": 32.0,
                "usage_pp": 30.0,
                "starter": 1.0,
                "fga": 18.0,
                "fta": 6.0,
                "team_def_rating": 112.0 + (game % 2),
            }
        )
    return pd.DataFrame(rows)


_GAMES = [
    {
        "participants": ["Memphis Grizzlies", "Portland Trail Blazers"],
        "team_markets": {
            "spread": [
                {
                    "period": 0,
                    "lines": [
                        {
                            "team": "Portland Trail Blazers",
                            "points": -4.5,
                            "side": "home",
                        },
                        {
                            "team": "Memphis Grizzlies",
                            "points": 4.5,
                            "side": "away",
                        },
                    ],
                }
            ],
            "total": [
                {
                    "period": 0,
                    "lines": [
                        {"side": "over", "points": 228.5},
                        {"side": "under", "points": 228.5},
                    ],
                }
            ],
        },
    }
]


def test_build_bayesian_live_features_uses_findopp_pinnacle_and_lineup(
    monkeypatch,
) -> None:
    history = _history()
    calls = {"findOpp": 0}

    def fake_find_opp(name, players_df, game_date, **kwargs):
        calls["findOpp"] += 1
        assert name == "Scoot Henderson"
        assert game_date == "2026-11-01"
        return "MEM", 1

    monkeypatch.setattr(
        "src.live.predict_points_bayes.findOpp",
        fake_find_opp,
    )

    live = build_bayesian_live_features(
        history,
        ["Scoot Henderson"],
        "2026-11-01",
        games=_GAMES,
        lineups={"POR": ["Scoot Henderson", "A", "B", "C", "D"]},
    )

    assert calls["findOpp"] == 1
    assert list(live.index) == ["Scoot Henderson"]
    row = live.iloc[0]
    assert row["starting"] == 1.0
    assert row["starting_source"] == "rotowire"
    assert row["is_home"] == 1.0
    assert row["opp_abbreviation"] == "MEM"
    assert row["team_spread_pregame"] == pytest.approx(-4.5)
    assert row["game_total_pregame"] == pytest.approx(228.5)
    assert row["implied_team_total_pregame"] == pytest.approx(
        (228.5 - (-4.5)) / 2
    )
    assert np.isfinite(row["opp_def_rating_prior"])
    assert row["starter_surprise"] == pytest.approx(
        1.0 - row["start_rate"]
    )


def test_bench_player_gets_starting_zero(monkeypatch) -> None:
    history = _history()
    monkeypatch.setattr(
        "src.live.predict_points_bayes.findOpp",
        lambda *args, **kwargs: ("MEM", 1),
    )
    live = build_bayesian_live_features(
        history,
        ["Scoot Henderson"],
        "2026-11-01",
        games=_GAMES,
        lineups={"POR": ["Someone Else", "A", "B", "C", "D"]},
    )
    assert live.iloc[0]["starting"] == 0.0
    assert live.iloc[0]["starting_source"] == "rotowire"


def test_missing_team_lineup_falls_back_to_start_rate(monkeypatch) -> None:
    history = _history()
    monkeypatch.setattr(
        "src.live.predict_points_bayes.findOpp",
        lambda *args, **kwargs: ("MEM", 1),
    )
    live = build_bayesian_live_features(
        history,
        ["Scoot Henderson"],
        "2026-11-01",
        games=_GAMES,
        lineups={"BOS": ["Someone Else", "A", "B", "C", "D"]},
    )
    # History alternates starter 1/0, so start_rate is ~0.5; threshold uses >= 0.5.
    assert live.iloc[0]["starting_source"] == "start_rate"
    assert live.iloc[0]["starting"] in (0.0, 1.0)


def test_choose_live_slate_date_prefers_pinnacle_coverage(tmp_path: Path) -> None:
    folder = tmp_path / "pinnacle"
    folder.mkdir()
    (folder / "pinnacle_nba_2026-11-01_120000_team.json").write_text(
        '{"games":[{"participants":["A","B"]}]}'
    )
    chosen = choose_live_slate_date(
        ["2026-11-01", "2026-11-02"],
        pinnacle_directory=folder,
    )
    assert chosen == "2026-11-01"
