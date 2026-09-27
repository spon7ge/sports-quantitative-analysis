"""DraftKings MLB discovery (Nash controldata) and repo-root setup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.scrapers.mlb.mlb_draftking import (
    _DEFAULT_OUTPUT_DIR,
    _ROOT,
    Event,
    Subcategory,
    extract_offer_picks,
    fetch_event_group,
    payload_to_offers,
    subcategories_from_nav,
)
from src.scrapers.mlb.paths import repo_root

_NAV_HTML = """
<html><script>
window.__LAYOUT__ = {"id":"NavRoot","title":"Nav Root","children":[
  {"id":"games","seoId":"games","title":"Games","parameters":{"leagueId":"84240"},"children":[
    {"id":"batter","seoId":"batter","title":"Batter","parameters":{"sportId":"7","leagueId":"84240"},"children":[
      {"id":"hr","seoId":"home-runs","title":"Home Runs","tags":["PlayerProps"],
       "parameters":{"sportId":"7","leagueId":"84240","categoryId":"743","subcategoryId":"17319"}}
    ]},
    {"id":"pitcher","seoId":"pitcher","title":"Pitcher","parameters":{"sportId":"7","leagueId":"84240","categoryId":"1031"},"children":[
      {"id":"k","seoId":"strikeouts","title":"Strikeouts","tags":["PlayerProps"],
       "parameters":{"sportId":"7","leagueId":"84240","categoryId":"1031","subcategoryId":"17323"}}
    ]},
    {"id":"gl","seoId":"game-lines","title":"Game Lines","tags":["PrimaryMarket"],
     "parameters":{"sportId":"7","leagueId":"84240","categoryId":"493","subcategoryId":"4519"}}
  ]}
]};
</script></html>
"""


def _markets_payload() -> dict:
    """Shape of the live leagueSubcategory/v1/markets payload."""
    return {
        "events": [
            {
                "id": "34704704",
                "name": "TB Rays @ NY Yankees",
                "startEventDate": "2026-09-22T17:05:00.0000000Z",
                "leagueId": "84240",
            }
        ],
        "markets": [
            {
                "id": "372211077",
                "eventId": "34704704",
                "name": "Carlos Rodon Strikeouts",
                "subcategoryId": "17323",
                "tags": ["PlayerProps"],
                "marketType": {"name": "Strikeouts Thrown Milestones"},
            }
        ],
        "selections": [
            {
                "id": "sel-1",
                "marketId": "372211077",
                "label": "5+",
                "milestoneValue": 5,
                "displayOdds": {"american": "−154", "decimal": "1.6494"},
                "trueOdds": 1.6494,
                "participants": [
                    {
                        "name": "Carlos Rodon (NYY)",
                        "type": "Player",
                        "seoIdentifier": "Carlos Rodon",
                    }
                ],
            },
            {
                "id": "sel-2",
                "marketId": "372211077",
                "label": "6+",
                "milestoneValue": 6,
                "displayOdds": {"american": "+120", "decimal": "2.2"},
                "trueOdds": 2.2,
                "participants": [
                    {
                        "name": "Carlos Rodon (NYY)",
                        "type": "Player",
                        "seoIdentifier": "Carlos Rodon",
                    }
                ],
            },
        ],
    }


def _outs_ou_payload() -> dict:
    return {
        "events": [
            {
                "id": "34704704",
                "name": "TB Rays @ NY Yankees",
                "startEventDate": "2026-09-22T17:05:00.0000000Z",
                "leagueId": "84240",
            }
        ],
        "markets": [
            {
                "id": "372216627",
                "eventId": "34704704",
                "name": "Zack Wheeler Outs O/U",
                "subcategoryId": "17413",
                "tags": ["PlayerProps"],
                "marketType": {"name": "Outs O/U"},
            }
        ],
        "selections": [
            {
                "id": "sel-over",
                "marketId": "372216627",
                "label": "Over",
                "points": 15.5,
                "outcomeType": "Over",
                "displayOdds": {"american": "-148", "decimal": "1.67"},
                "trueOdds": 1.67,
                "participants": [
                    {"name": "Zack Wheeler", "type": "Player", "seoIdentifier": "Zack Wheeler"}
                ],
            },
            {
                "id": "sel-under",
                "marketId": "372216627",
                "label": "Under",
                "points": 15.5,
                "outcomeType": "Under",
                "displayOdds": {"american": "+114", "decimal": "2.14"},
                "trueOdds": 2.14,
                "participants": [
                    {"name": "Zack Wheeler", "type": "Player", "seoIdentifier": "Zack Wheeler"}
                ],
            },
        ],
    }


def test_draftkings_root_is_repo_root_not_src() -> None:
    root = repo_root()
    assert _ROOT == str(root)
    assert _DEFAULT_OUTPUT_DIR == str(root / "data" / "props" / "draftkings")


def test_subcategories_from_nav_keep_player_props_not_game_lines() -> None:
    subs = subcategories_from_nav(
        _NAV_HTML, allow=["batter", "pitcher", "prop", "home run", "strikeout"]
    )
    assert {(s.category_name, s.name, s.subcategory_id) for s in subs} == {
        ("Batter", "Home Runs", "17319"),
        ("Pitcher", "Strikeouts", "17323"),
    }


def test_fetch_event_group_reads_nav_from_league_page() -> None:
    cfg = {
        "dk_league_page": "https://sportsbook.draftkings.com/leagues/baseball/mlb",
        "dk_category_allowlist": ["batter", "pitcher", "prop", "home run", "strikeout"],
        "dk_timeout": 30,
        "dk_max_retries": 1,
    }
    with patch("src.scrapers.mlb.mlb_draftking.fetch_text", return_value=_NAV_HTML):
        events, subs = fetch_event_group(MagicMock(), cfg)

    assert events == {}
    assert [s.subcategory_id for s in subs] == ["17319", "17323"]


def test_payload_to_offers_uses_player_seo_name_and_milestone_label() -> None:
    offers, events = payload_to_offers(_markets_payload(), competition="MLB")
    assert list(events) == ["34704704"]
    assert events["34704704"].name == "TB Rays @ NY Yankees"
    assert len(offers) == 1
    offer = offers[0]
    assert offer["label"] == "Carlos Rodon Strikeouts"
    assert offer["eventId"] == "34704704"
    outcomes = list(offer["outcomes"])
    assert outcomes[0]["participant"] == "Carlos Rodon"
    assert outcomes[0]["label"] == "5+"
    assert outcomes[0]["oddsAmerican"] == "−154"


def test_extract_offer_picks_keeps_player_name_and_milestone_side() -> None:
    offers, events = payload_to_offers(_markets_payload(), competition="MLB")
    sub = Subcategory(
        category_id="1031",
        subcategory_id="17323",
        category_name="Pitcher",
        name="Strikeouts",
    )
    counters = {
        "added": 0,
        "suspended_offers": 0,
        "game_markets": 0,
        "exact_markets": 0,
        "no_stat": 0,
        "no_player": 0,
        "no_price": 0,
        "skipped_sport": 0,
    }
    picks = extract_offer_picks(
        offers, sub, events, "2026-09-22T04:00:00Z", counters, set(), frozenset({"MLB"})
    )
    assert {(p.full_name, p.choice, p.stat_value, p.american_price) for p in picks} == {
        ("Carlos Rodon", "over", 4.5, -154),
        ("Carlos Rodon", "over", 5.5, 120),
    }


def test_extract_offer_picks_reads_over_under_points() -> None:
    offers, events = payload_to_offers(_outs_ou_payload(), competition="MLB")
    sub = Subcategory(
        category_id="1031",
        subcategory_id="17413",
        category_name="Pitcher",
        name="Outs",
    )
    counters = {
        "added": 0,
        "suspended_offers": 0,
        "game_markets": 0,
        "exact_markets": 0,
        "no_stat": 0,
        "no_player": 0,
        "no_price": 0,
        "skipped_sport": 0,
    }
    picks = extract_offer_picks(
        offers, sub, events, "2026-09-22T04:00:00Z", counters, set(), frozenset({"MLB"})
    )
    assert {(p.full_name, p.choice, p.stat_value, p.american_price) for p in picks} == {
        ("Zack Wheeler", "over", 15.5, -148),
        ("Zack Wheeler", "under", 15.5, 114),
    }


def test_parse_outcome_selection_reads_milestone_plus() -> None:
    from src.scrapers.mlb.mlb_draftking import parse_outcome_selection

    assert parse_outcome_selection("5+", 5) == ("over", 4.5)
    assert parse_outcome_selection("1+", 1) == ("over", 0.5)


def test_match_stat_maps_pitcher_outs() -> None:
    from src.scrapers.mlb.mlb_draftking import match_stat

    assert match_stat("Outs") == "pitching_outs"
    assert match_stat("Strikeouts") == "strikeouts"
    assert match_stat("Extra Base Hits") == "extra_base_hits"
    assert match_stat("Hits") == "hits"
