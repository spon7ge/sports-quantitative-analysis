from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from src.live.latest_points import event_context, latest_points_markets


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_latest_snapshot_sets_the_pacific_slate_and_ignores_older_files(
    tmp_path: Path,
) -> None:
    props = tmp_path / "props"
    _write(
        props / "prizepicks/nba/prizepicks_nba_2026-10-28_120000.json",
        {
            "fetched_at": "2026-10-28T19:00:00Z",
            "projections": [
                {
                    "player": "Donovan Mitchell",
                    "stat_type": "Points",
                    "line_score": 99.5,
                    "odds_type": "standard",
                }
            ],
        },
    )
    _write(
        props / "prizepicks/nba/prizepicks_nba_2026-10-29_144500.json",
        {
            "fetched_at": "2026-10-29T21:45:00Z",
            "projections": [
                {
                    "player": "Donovan Mitchell",
                    "stat_type": "Points",
                    "line_score": 24.5,
                    "odds_type": "standard",
                },
                {
                    "player": "Donovan Mitchell",
                    "stat_type": "Points",
                    "line_score": 30.5,
                    "odds_type": "demon",
                },
            ],
        },
    )
    _write(
        props / "pinnacle/nba/pinnacle_nba_2026-10-29_144500_props.json",
        {
            "fetched_at": "2026-10-29T21:45:00Z",
            "games": [
                {
                    "label": "Cleveland Cavaliers @ Atlanta Hawks",
                    "start_time": "2026-10-30T00:10:00Z",
                    "props": [
                        {
                            "player": "Donovan Mitchell",
                            "stat": "points",
                            "line": 24.5,
                            "american_over": -104,
                            "american_under": -127,
                        }
                    ],
                }
            ],
        },
    )
    _write(
        props / "pinnacle/nba/pinnacle_nba_2026-10-29_144500_team.json",
        {"games": []},
    )

    markets, events, audit = latest_points_markets(props)
    prizepicks = markets.loc[markets["book"].eq("prizepicks")].iloc[0]
    pinnacle = markets.loc[markets["book"].eq("pinnacle")].iloc[0]

    assert prizepicks["line"] == pytest.approx(24.5)
    assert prizepicks["slate_date"] == pd.Timestamp("2026-10-29")
    assert pinnacle["over_american"] == -104
    assert pinnacle["slate_date"] == pd.Timestamp("2026-10-29")
    assert audit["source_files"]["prizepicks"].endswith("2026-10-29_144500.json")
    assert "team" not in audit["source_files"]["pinnacle"]
    assert events.iloc[0]["event_name"] == "Cleveland Cavaliers @ Atlanta Hawks"


def test_latest_snapshot_reads_utf8_names(tmp_path: Path) -> None:
    props = tmp_path / "props"
    _write(
        props / "prizepicks/nba/prizepicks_nba_2026-10-29_144500.json",
        {
            "fetched_at": "2026-10-29T21:45:00Z",
            "projections": [
                {
                    "player": "Luka Dončić",
                    "stat_type": "Points",
                    "line_score": 28.5,
                    "odds_type": "standard",
                }
            ],
        },
    )
    markets, _, _ = latest_points_markets(props)
    assert markets.iloc[0]["player_name"] == "Luka Dončić"


def test_event_context_reads_away_at_home() -> None:
    opponent, is_home = event_context("Cleveland Cavaliers @ Atlanta Hawks", "CLE")
    assert opponent == "ATL"
    assert is_home == pytest.approx(0.0)
    opponent, is_home = event_context("Cleveland Cavaliers @ Atlanta Hawks", "ATL")
    assert opponent == "CLE"
    assert is_home == pytest.approx(1.0)
    opponent, is_home = event_context("Cleveland Cavaliers vs Atlanta Hawks", "CLE")
    assert opponent is None
    assert pd.isna(is_home)
