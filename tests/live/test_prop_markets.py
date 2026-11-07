from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from src.live.prop_markets import attach_roster, book_names, latest_prop_markets


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prizepicks_keeps_one_standard_line_per_stat_at_minus_119(tmp_path: Path) -> None:
    props = tmp_path / "props"
    _write(
        props / "prizepicks/nba/prizepicks_nba_2026-11-07_000331.json",
        {
            "fetched_at": "2026-11-07T08:03:31Z",
            "projections": [
                {
                    "player": "Nikola Jokić",
                    "stat_type": "Points",
                    "line_score": 24.5,
                    "odds_type": "standard",
                },
                {
                    "player": "Nikola Jokić",
                    "stat_type": "Points",
                    "line_score": 32.5,
                    "odds_type": "demon",
                },
                {
                    "player": "Nikola Jokić",
                    "stat_type": "Points (Combo)",
                    "line_score": 40.5,
                    "odds_type": "standard",
                },
                {
                    "player": "Nikola Jokić",
                    "stat_type": "Rebounds",
                    "line_score": 12.5,
                    "odds_type": "standard",
                },
                {
                    "player": "Nikola Jokić",
                    "stat_type": "Assists",
                    "line_score": 9.5,
                    "odds_type": "standard",
                },
                {
                    "player": "Nikola Jokić",
                    "stat_type": "Steals",
                    "line_score": 1.5,
                    "odds_type": "standard",
                },
                {
                    "player": "Nikola Jokić",
                    "stat_type": "Blocked Shots",
                    "line_score": 0.5,
                    "odds_type": "standard",
                },
                {
                    "player": "Nikola Jokić",
                    "stat_type": "Blks+Stls",
                    "line_score": 2.5,
                    "odds_type": "standard",
                },
                {
                    "player": "Jamal Murray",
                    "stat_type": "Points",
                    "line_score": 18.5,
                    "odds_type": "standard",
                },
                {
                    "player": "Jamal Murray",
                    "stat_type": "Points",
                    "line_score": 22.5,
                    "odds_type": "standard",
                },
            ],
        },
    )

    markets, audit = latest_prop_markets(props, "PrizePicks")

    jokic = markets.loc[markets["player_name"].eq("Nikola Jokić")].set_index("stat")
    assert set(jokic.index) == {"PTS", "REB", "AST", "STL", "BLK"}
    assert jokic.loc["PTS", "line"] == pytest.approx(24.5)
    assert jokic.loc["REB", "line"] == pytest.approx(12.5)
    assert jokic.loc["AST", "line"] == pytest.approx(9.5)
    assert jokic.loc["STL", "line"] == pytest.approx(1.5)
    assert jokic.loc["BLK", "line"] == pytest.approx(0.5)
    assert set(jokic["over_american"]) == {-119}
    assert set(jokic["under_american"]) == {-119}
    assert "Jamal Murray" not in set(markets["player_name"])
    assert audit["book"] == "prizepicks"
    assert audit["source_file"].endswith("2026-11-07_000331.json")


def test_other_books_keep_the_closest_two_way_line_and_ignore_combos(
    tmp_path: Path,
) -> None:
    props = tmp_path / "props"
    _write(
        props / "fanduel/nba/fanduel_nba_2026-11-07_000332.json",
        {
            "fetched_at": "2026-11-07T08:03:32Z",
            "picks": [
                {
                    "full_name": "Jamal Murray",
                    "stat_name": "points",
                    "stat_value": 24.5,
                    "choice": "over",
                    "american_price": -110,
                    "start_time": "2026-11-08T01:10:00Z",
                },
                {
                    "full_name": "Jamal Murray",
                    "stat_name": "points",
                    "stat_value": 24.5,
                    "choice": "under",
                    "american_price": -110,
                    "start_time": "2026-11-08T01:10:00Z",
                },
                {
                    "full_name": "Jamal Murray",
                    "stat_name": "points",
                    "stat_value": 30.5,
                    "choice": "over",
                    "american_price": -200,
                    "start_time": "2026-11-08T01:10:00Z",
                },
                {
                    "full_name": "Jamal Murray",
                    "stat_name": "points",
                    "stat_value": 30.5,
                    "choice": "under",
                    "american_price": 160,
                    "start_time": "2026-11-08T01:10:00Z",
                },
                {
                    "full_name": "Jamal Murray",
                    "stat_name": "points_rebounds_assists",
                    "stat_value": 40.5,
                    "choice": "over",
                    "american_price": -110,
                    "start_time": "2026-11-08T01:10:00Z",
                },
                {
                    "full_name": "Jamal Murray",
                    "stat_name": "points_rebounds_assists",
                    "stat_value": 40.5,
                    "choice": "under",
                    "american_price": -110,
                    "start_time": "2026-11-08T01:10:00Z",
                },
                {
                    "full_name": "Jamal Murray",
                    "stat_name": "assists",
                    "stat_value": 6.5,
                    "choice": "over",
                    "american_price": -115,
                    "start_time": "2026-11-08T01:10:00Z",
                },
                {
                    "full_name": "Jamal Murray",
                    "stat_name": "assists",
                    "stat_value": 6.5,
                    "choice": "under",
                    "american_price": -105,
                    "start_time": "2026-11-08T01:10:00Z",
                },
            ],
        },
    )

    markets, _ = latest_prop_markets(props, "fanduel")
    points = markets.loc[markets["stat"].eq("PTS")].iloc[0]
    assists = markets.loc[markets["stat"].eq("AST")].iloc[0]

    assert len(markets) == 2
    assert points["line"] == pytest.approx(24.5)
    assert points["over_american"] == -110
    assert assists["line"] == pytest.approx(6.5)
    assert assists["over_american"] == -115
    assert markets["slate_date"].iloc[0] == pd.Timestamp("2026-11-07")


def test_unknown_book_names_the_saved_books(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prizepicks") as caught:
        latest_prop_markets(tmp_path, "betmgm")
    message = str(caught.value)
    for book in book_names():
        assert book in message


def test_attach_roster_matches_accented_names_on_this_game() -> None:
    markets = pd.DataFrame(
        {
            "slate_date": [pd.Timestamp("2026-11-07")] * 2,
            "book": ["prizepicks", "prizepicks"],
            "player_name": ["Nikola Jokic", "Jayson Tatum"],
            "stat": ["PTS", "PTS"],
            "line": [24.5, 27.5],
            "over_american": [-119, -119],
            "under_american": [-119, -119],
        }
    )
    roster = pd.DataFrame(
        {
            "team": ["DEN"],
            "side": ["home"],
            "player_id": [203999],
            "player_name": ["Nikola Jokić"],
        }
    )

    matched = attach_roster(markets, roster)

    assert len(matched) == 1
    assert matched.iloc[0]["player_name"] == "Nikola Jokić"
    assert matched.iloc[0]["player_id"] == "203999"
    assert matched.iloc[0]["team"] == "DEN"
    assert matched.iloc[0]["side"] == "home"
    assert matched.iloc[0]["line"] == pytest.approx(24.5)
