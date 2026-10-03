"""NBA Underdog scraper defaults and league routing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.odds.load_snapshots import _underdog_table
from src.odds.quote_specs import QUOTE_SPECS
from src.scrapers.mlb.mlb_underdog import _DEFAULT_OUTPUT_DIR as MLB_OUTPUT_DIR
from src.scrapers.nba.nba_underdog import (
    _DEFAULT_CONFIG,
    _DEFAULT_OUTPUT_DIR as NBA_OUTPUT_DIR,
    extract_picks,
    sport_to_league,
)
from src.scrapers.nba.paths import repo_root


def test_underdog_json_dirs_are_split_by_league() -> None:
    root = repo_root()
    assert MLB_OUTPUT_DIR == str(root / "data" / "props" / "underdogs" / "mlb")
    assert NBA_OUTPUT_DIR == str(root / "data" / "props" / "underdogs" / "nba")


def test_underdog_default_is_nba_v1() -> None:
    assert _DEFAULT_CONFIG["sport_allowlist"] == ["NBA"]
    url = str(_DEFAULT_CONFIG["ud_pickem_url"])
    assert url == "https://api.underdogfantasy.com/v1/over_under_lines"


def test_sport_to_league_maps_nba_only() -> None:
    assert sport_to_league("NBA") == "nba"
    assert sport_to_league("nba") == "nba"
    assert sport_to_league("MLB") is None


def test_nba_underdog_table_and_quote_spec() -> None:
    assert _underdog_table("nba") == "nba_underdogs"
    assert _underdog_table("mlb") == "mlb_underdogs"
    assert _underdog_table("wnba") == "wnba_underdogs"
    assert QUOTE_SPECS["nba_underdogs"] is QUOTE_SPECS["mlb_underdogs"]


def test_extract_picks_keeps_nba_and_drops_other_sports() -> None:
    payload = {
        "players": [
            {
                "id": "p1",
                "position_id": "pos",
                "team_id": "t1",
                "first_name": "Nikola",
                "last_name": "Jokic",
            },
            {
                "id": "p2",
                "position_id": "pos",
                "team_id": "t2",
                "first_name": "Shohei",
                "last_name": "Ohtani",
            },
        ],
        "games": [
            {"id": "g1", "sport_id": "NBA"},
            {"id": "g2", "sport_id": "MLB"},
        ],
        "appearances": [
            {"id": "a1", "match_id": "g1", "player_id": "p1", "position_id": "pos", "team_id": "t1"},
            {"id": "a2", "match_id": "g2", "player_id": "p2", "position_id": "pos", "team_id": "t2"},
        ],
        "over_under_lines": [
            {
                "status": "active",
                "stat_value": "27.5",
                "updated_at": "2026-10-03T00:00:00Z",
                "over_under": {"appearance_stat": {"stat": "points", "appearance_id": "a1"}},
                "options": [
                    {
                        "appearance_id": "a1",
                        "choice": "higher",
                        "american_price": -115,
                        "payout_multiplier": 1.0,
                    }
                ],
            },
            {
                "status": "active",
                "stat_value": "1.5",
                "over_under": {"appearance_stat": {"stat": "hits", "appearance_id": "a2"}},
                "options": [{"appearance_id": "a2", "choice": "lower"}],
            },
        ],
    }

    picks = extract_picks(payload, frozenset({"NBA"}))
    assert len(picks) == 1
    assert picks[0].full_name == "Nikola Jokic"
    assert picks[0].stat_name == "points"
    assert picks[0].choice == "over"
    assert picks[0].sport_id == "NBA"


def test_nba_underdog_imports_as_loose_script() -> None:
    """`python nba_underdog.py` from src/scrapers/nba must resolve paths.py."""
    nba_dir = Path(__file__).resolve().parents[2] / "src" / "scrapers" / "nba"
    probe = (
        "import runpy; "
        "ns = runpy.run_path('nba_underdog.py', run_name='not_main'); "
        "assert ns.get('_ROOT')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(nba_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
