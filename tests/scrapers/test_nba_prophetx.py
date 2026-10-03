"""NBA ProphetX scraper defaults, prop parsing, and league routing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.odds.load_snapshots import _prophetx_props_table, _prophetx_team_table
from src.odds.quote_specs import QUOTE_SPECS
from src.scrapers.nba.nba_prophetx import (
    NBA_TOURNAMENT_ID,
    _DEFAULT_OUTPUT_DIR,
    extract_props,
    player_name_from_market,
)
from src.scrapers.nba.paths import repo_root


def test_nba_prophetx_targets_nba_tournament_and_dir() -> None:
    root = repo_root()
    assert NBA_TOURNAMENT_ID == 132
    assert _DEFAULT_OUTPUT_DIR == str(root / "data" / "props" / "prophetx" / "nba")


def test_nba_prophetx_tables() -> None:
    assert _prophetx_props_table("nba") == "nba_prophetx"
    assert _prophetx_team_table("nba") == "nba_prophetx_team"
    assert _prophetx_props_table("mlb") == "mlb_prophetx"
    assert QUOTE_SPECS["nba_prophetx"] is QUOTE_SPECS["mlb_prophetx"]
    assert QUOTE_SPECS["nba_prophetx_team"] is QUOTE_SPECS["mlb_prophetx_team"]


def test_player_name_keeps_combo_suffix_intact() -> None:
    assert (
        player_name_from_market(
            {"name": "Breanna Stewart Total Points, Rebounds & Assists"}
        )
        == "Breanna Stewart"
    )
    assert player_name_from_market({"name": "Nikola Jokic Total Points"}) == "Nikola Jokic"


def test_extract_props_maps_points_over_under() -> None:
    markets = [
        {
            "id": 11,
            "name": "Nikola Jokic Total Points",
            "subType": "player_total_points",
            "marketLines": [
                {
                    "favourite": True,
                    "selections": [
                        [{"name": "over 27.5", "odds": -115, "stake": 50, "line": 27.5}],
                        [{"name": "under 27.5", "odds": -105, "stake": 40, "line": 27.5}],
                    ],
                }
            ],
        }
    ]
    rows = extract_props(markets)
    assert len(rows) == 1
    assert rows[0]["player"] == "Nikola Jokic"
    assert rows[0]["stat"] == "points"
    assert rows[0]["line"] == 27.5
    assert rows[0]["is_main"] is True
    assert rows[0]["over"]["american"] == -115
    assert rows[0]["under"]["american"] == -105


def test_nba_prophetx_imports_as_loose_script() -> None:
    nba_dir = Path(__file__).resolve().parents[2] / "src" / "scrapers" / "nba"
    probe = (
        "import runpy; "
        "ns = runpy.run_path('nba_prophetx.py', run_name='not_main'); "
        "assert ns.get('NBA_TOURNAMENT_ID') == 132"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(nba_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
