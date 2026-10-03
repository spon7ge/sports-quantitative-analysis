"""DraftKings NBA discovery (Nash controldata) and league routing."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.odds.load_snapshots import _draftkings_table
from src.odds.quote_specs import QUOTE_SPECS
from src.odds.snapshot_rows import draftkings_picks_to_rows
from src.scrapers.nba.nba_draftking import (
    _DEFAULT_CONFIG,
    _DEFAULT_OUTPUT_DIR,
    Subcategory,
    extract_offer_picks,
    fetch_event_group,
    is_game_market,
    match_stat,
    payload_to_offers,
    resolve_output_path,
    sport_to_league,
    subcategories_from_nav,
)
from src.scrapers.nba.paths import repo_root

_NAV_HTML = """
<html><script>
window.__LAYOUT__ = {"id":"NavRoot","title":"Nav Root","children":[
  {"id":"games","seoId":"games","title":"Games","parameters":{"leagueId":"42648"},"children":[
    {"id":"points","seoId":"player-points","title":"Player Points","parameters":{"sportId":"2","leagueId":"42648"},"children":[
      {"id":"pts","seoId":"points","title":"Points","tags":["PlayerProps"],
       "parameters":{"sportId":"2","leagueId":"42648","categoryId":"1215","subcategoryId":"12488"}}
    ]},
    {"id":"combos","seoId":"player-combos","title":"Player Combos","parameters":{"sportId":"2","leagueId":"42648"},"children":[
      {"id":"pra","seoId":"pts-reb-ast","title":"Pts + Reb + Ast","tags":["PlayerProps"],
       "parameters":{"sportId":"2","leagueId":"42648","categoryId":"1216","subcategoryId":"12492"}}
    ]},
    {"id":"gl","seoId":"game-lines","title":"Game Lines","tags":["PrimaryMarket"],
     "parameters":{"sportId":"2","leagueId":"42648","categoryId":"487","subcategoryId":"4511"}}
  ]}
]};
</script></html>
"""


def _points_payload() -> dict:
    return {
        "events": [
            {
                "id": "34316771",
                "name": "DEN Nuggets @ OKC Thunder",
                "startEventDate": "2026-10-03T23:30:00.0000000Z",
                "leagueId": "42648",
            }
        ],
        "markets": [
            {
                "id": "m1",
                "eventId": "34316771",
                "name": "Nikola Jokic Points",
                "subcategoryId": "12488",
                "tags": ["PlayerProps"],
                "marketType": {"name": "Points Milestones"},
            }
        ],
        "selections": [
            {
                "id": "sel-1",
                "marketId": "m1",
                "label": "28+",
                "milestoneValue": 28,
                "displayOdds": {"american": "−115", "decimal": "1.87"},
                "trueOdds": 1.87,
                "participants": [
                    {
                        "name": "Nikola Jokic (DEN)",
                        "type": "Player",
                        "seoIdentifier": "Nikola Jokic",
                    }
                ],
            }
        ],
    }


def _counters() -> dict[str, int]:
    return {
        "added": 0,
        "suspended_offers": 0,
        "game_markets": 0,
        "exact_markets": 0,
        "no_stat": 0,
        "no_player": 0,
        "no_price": 0,
        "skipped_sport": 0,
    }


def test_nba_draftking_defaults() -> None:
    root = repo_root()
    assert _DEFAULT_CONFIG["sport_allowlist"] == ["NBA"]
    assert _DEFAULT_CONFIG["dk_event_group"] == "42648"
    assert _DEFAULT_CONFIG["dk_league_page"].endswith("/leagues/basketball/nba")
    assert "pitcher" not in _DEFAULT_CONFIG["dk_category_allowlist"]
    assert _DEFAULT_OUTPUT_DIR == str(root / "data" / "props" / "draftkings")
    assert sport_to_league("NBA") == "nba"
    assert sport_to_league("WNBA") is None
    assert sport_to_league("MLB") is None


def test_nba_draftkings_table_and_quote_spec() -> None:
    assert _draftkings_table("nba") == "nba_draftkings"
    assert _draftkings_table("mlb") == "mlb_draftkings"
    assert QUOTE_SPECS["nba_draftkings"] is QUOTE_SPECS["mlb_draftkings"]


def test_subcategories_from_nav_keep_player_props_not_game_lines() -> None:
    subs = subcategories_from_nav(_NAV_HTML, _DEFAULT_CONFIG["dk_category_allowlist"])
    assert {(s.category_name, s.name, s.subcategory_id) for s in subs} == {
        ("Player Points", "Points", "12488"),
        ("Player Combos", "Pts + Reb + Ast", "12492"),
    }


def test_fetch_event_group_reads_nav_from_league_page() -> None:
    cfg = {
        "dk_league_page": _DEFAULT_CONFIG["dk_league_page"],
        "dk_category_allowlist": _DEFAULT_CONFIG["dk_category_allowlist"],
        "dk_timeout": 30,
        "dk_max_retries": 1,
    }
    with patch("src.scrapers.nba.nba_draftking.fetch_text", return_value=_NAV_HTML):
        events, subs = fetch_event_group(MagicMock(), cfg)

    assert events == {}
    assert [s.subcategory_id for s in subs] == ["12488", "12492"]


def test_extract_offer_picks_keeps_points_milestone() -> None:
    offers, events = payload_to_offers(_points_payload(), competition="NBA")
    sub = Subcategory(
        category_id="1215",
        subcategory_id="12488",
        category_name="Player Points",
        name="Points",
    )
    picks = extract_offer_picks(
        offers, sub, events, "2026-10-03T20:00:00Z", _counters(), set(), frozenset({"NBA"})
    )
    assert {(p.full_name, p.stat_name, p.choice, p.stat_value, p.american_price) for p in picks} == {
        ("Nikola Jokic", "points", "over", 27.5, -115),
    }
    rows = draftkings_picks_to_rows(
        [p.to_dict() for p in picks],
        league="nba",
        scraped_at=datetime(2026, 10, 3, tzinfo=ZoneInfo("UTC")),
    )
    assert rows[0]["market_type"] == "player_points"
    assert rows[0]["league"] == "nba"


def test_match_stat_maps_nba_props_not_baseball() -> None:
    assert match_stat("Points") == "points"
    assert match_stat("Pts + Reb + Ast") == "points_rebounds_assists"
    assert match_stat("Threes") == "three_pointers_made"
    assert match_stat("Strikeouts") is None
    assert is_game_market("Total Points")
    assert not is_game_market("Points")


def test_resolve_output_path_uses_nba_slug(monkeypatch) -> None:
    monkeypatch.delenv("DK_OUTPUT", raising=False)
    path = resolve_output_path("NBA")
    assert path.startswith(_DEFAULT_OUTPUT_DIR)
    assert "draftkings_nba_" in path


def test_nba_draftking_imports_as_loose_script() -> None:
    """`python nba_draftking.py` from src/scrapers/nba must resolve paths.py."""
    nba_dir = Path(__file__).resolve().parents[2] / "src" / "scrapers" / "nba"
    probe = (
        "import runpy; "
        "ns = runpy.run_path('nba_draftking.py', run_name='not_main'); "
        "assert ns.get('_ROOT'); "
        "assert ns['_DEFAULT_CONFIG']['dk_event_group'] == '42648'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(nba_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
