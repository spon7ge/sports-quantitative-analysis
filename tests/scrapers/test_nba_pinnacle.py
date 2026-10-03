"""NBA Pinnacle scraper defaults, prop parsing, and league routing."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.odds.load_snapshots import _pinnacle_props_table, _pinnacle_team_table
from src.odds.quote_specs import QUOTE_SPECS
from src.odds.snapshot_rows import selenium_pinnacle_props_to_rows
from src.scrapers.nba.nba_pinnacle import (
    LEAGUE_ARCADIA_IDS,
    LEAGUE_MATCHUPS_URL,
    PLAYER_PROP_UNITS,
    PinnacleScraper,
    _DEFAULT_PINNACLE_BASE_DIR,
    collect_game_urls,
    game_urls_from_league_matchups,
)
from src.scrapers.nba.paths import repo_root


def test_nba_pinnacle_targets_nba_board_and_dir() -> None:
    root = repo_root()
    assert LEAGUE_ARCADIA_IDS == {"nba": 487}
    assert (
        LEAGUE_MATCHUPS_URL["nba"]
        == "https://www.pinnacle.com/en/basketball/nba/matchups/#all"
    )
    assert _DEFAULT_PINNACLE_BASE_DIR == str(root / "data" / "props" / "pinnacle")
    assert PLAYER_PROP_UNITS["Points"] == "points"
    assert PLAYER_PROP_UNITS["Pts+Rebs+Asts"] == "points_rebounds_assists"
    assert "Hits" not in PLAYER_PROP_UNITS


def test_nba_pinnacle_tables_and_quote_specs() -> None:
    assert _pinnacle_props_table("nba") == "nba_pinnacle"
    assert _pinnacle_team_table("nba") == "nba_pinnacle_team"
    assert _pinnacle_props_table("mlb") == "mlb_pinnacle"
    assert _pinnacle_props_table("wnba") == "wnba_pinnacle"
    assert _pinnacle_team_table("wnba") == "wnba_pinnacle_team"
    assert QUOTE_SPECS["nba_pinnacle"] is QUOTE_SPECS["mlb_pinnacle"]
    assert QUOTE_SPECS["nba_pinnacle_team"] is QUOTE_SPECS["mlb_pinnacle_team"]


def test_collect_game_urls_keeps_nba_and_drops_mlb() -> None:
    hrefs = [
        "https://www.pinnacle.com/en/basketball/nba/lakers-vs-celtics/12345/#all",
        "https://www.pinnacle.com/en/baseball/mlb/dodgers-vs-giants/99/",
    ]
    assert collect_game_urls(hrefs, "", "nba") == [
        "https://www.pinnacle.com/en/basketball/nba/lakers-vs-celtics/12345/#all"
    ]


def test_game_urls_from_league_matchups() -> None:
    rows = [
        {
            "type": "matchup",
            "id": 555,
            "participants": [
                {"alignment": "away", "name": "Los Angeles Lakers"},
                {"alignment": "home", "name": "Boston Celtics"},
            ],
        }
    ]
    assert game_urls_from_league_matchups(rows, "nba") == [
        "https://www.pinnacle.com/en/basketball/nba/los-angeles-lakers-vs-boston-celtics/555/#all"
    ]


def test_props_from_arcadia_maps_points(monkeypatch) -> None:
    monkeypatch.delenv("PINNACLE_OUTPUT", raising=False)
    monkeypatch.delenv("PINNACLE_OUTPUT_DIR", raising=False)
    scraper = PinnacleScraper("nba")
    assert "/data/props/pinnacle/nba/pinnacle_nba_" in scraper.output_path.replace("\\", "/")
    assert scraper.output_path.endswith("_props.json")

    props = scraper.props_from_arcadia_arrays(
        [
            {
                "matchupId": 99,
                "type": "total",
                "period": 0,
                "prices": [
                    {"price": -115, "points": 27.5, "participantId": 1},
                    {"price": -105, "points": 27.5, "participantId": 2},
                ],
            }
        ],
        [
            {
                "type": "special",
                "units": "Points",
                "id": 99,
                "special": {"description": "Nikola Jokic Total Points"},
            },
            {
                "type": "special",
                "units": "Hits",
                "id": 100,
                "special": {"description": "Shohei Ohtani Total Hits"},
            },
        ],
    )
    assert len(props) == 1
    assert props[0]["player"] == "Nikola Jokic"
    assert props[0]["stat"] == "points"
    assert props[0]["line"] == 27.5

    rows = selenium_pinnacle_props_to_rows(
        [{"props": props}],
        league="nba",
        scraped_at=datetime.now(timezone.utc),
    )
    assert {row["market_type"] for row in rows} == {"player_points"}
    assert {row["league"] for row in rows} == {"nba"}


def test_nba_pinnacle_imports_as_loose_script() -> None:
    """`python nba_pinnacle.py` from src/scrapers/nba must resolve paths.py."""
    nba_dir = Path(__file__).resolve().parents[2] / "src" / "scrapers" / "nba"
    probe = (
        "import runpy; "
        "ns = runpy.run_path('nba_pinnacle.py', run_name='not_main'); "
        "assert ns.get('_ROOT'); "
        "assert ns['LEAGUE_ARCADIA_IDS'] == {'nba': 487}"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(nba_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
