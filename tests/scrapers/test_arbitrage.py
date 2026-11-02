"""Cross-book arbitrage on the four odds snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.scrapers.arbitrage import (
    canonical_team,
    find_arbs,
    format_report,
    quotes_from_payload,
    report_arbitrage,
)


def test_canonical_team_tolerates_missing_label() -> None:
    assert canonical_team(None, "nba") is None
    assert canonical_team("", "nba") is None
    assert canonical_team("Portland Trail Blazers", "nba") == "por"


def _novig_game() -> dict:
    return {
        "name": "Minnesota Timberwolves @ Cleveland Cavaliers",
        "scheduled": "2026-10-26T23:05:00.000Z",
        "competitors": [
            {"name": "Cleveland Cavaliers"},
            {"name": "Minnesota Timberwolves"},
        ],
        "props": [
            {
                "player": "LaMelo Ball",
                "stat": "points",
                "line": 22.5,
                "over": {"american": 130},
                "under": {"american": -140},
            }
        ],
        "team_markets": {
            "moneyline": [
                {"name": "CLE", "american": 277, "line": None},
                {"name": "MIN", "american": -285, "line": None},
            ],
            "spread": [
                {"name": "CLE +7.5", "american": -113, "line": 7.5},
                {"name": "MIN -7.5", "american": -108, "line": -7.5},
            ],
            "total": [
                {"name": "Over 230.5", "american": -120, "line": 230.5},
                {"name": "Under 230.5", "american": -102, "line": 230.5},
            ],
            "first_half_total": [
                {"name": "Over 115.5", "american": 500, "line": 115.5},
                {"name": "Under 115.5", "american": 500, "line": 115.5},
            ],
        },
    }


def test_same_line_prop_with_both_prices_plus_money_is_an_arb() -> None:
    game = _novig_game()
    game["name"] = "Charlotte Hornets @ Miami Heat"
    game["competitors"] = [
        {"name": "Charlotte Hornets"},
        {"name": "Miami Heat"},
    ]
    game["props"] = [
        {
            "player": "LaMelo Ball",
            "stat": "points",
            "line": 22.5,
            "over": {"american": 110},
            "under": None,
        }
    ]
    game["team_markets"] = {}
    novig = {"games": [game]}
    fanduel = {
        "picks": [
            {
                "full_name": "LaMelo Ball",
                "stat_name": "points",
                "stat_value": 22.5,
                "choice": "under",
                "american_price": 110,
                "event_name": "Charlotte Hornets @ Miami Heat",
                "start_time": "2026-10-26T23:05:00.000Z",
            }
        ]
    }
    quotes = quotes_from_payload("novig", "nba", novig)
    quotes += quotes_from_payload("fanduel", "nba", fanduel)
    arbs = find_arbs(quotes)
    assert len(arbs) == 1
    arb = arbs[0]
    assert arb.market == "prop"
    assert arb.edge > 0
    books = {arb.left.book, arb.right.book}
    assert books == {"novig", "fanduel"}
    assert {arb.left.side_label, arb.right.side_label} == {"over", "under"}


def test_standard_vig_is_not_an_arb() -> None:
    payload = {
        "games": [
            {
                "name": "Boston Celtics @ New York Knicks",
                "scheduled": "2026-10-26T23:30:00Z",
                "props": [
                    {
                        "player": "Jalen Brunson",
                        "stat": "points",
                        "line": 28.5,
                        "over": {"american": -110},
                        "under": {"american": -110},
                    }
                ],
                "team_markets": {},
            }
        ]
    }
    assert find_arbs(quotes_from_payload("novig", "nba", payload)) == []


def test_even_money_both_sides_is_not_an_arb() -> None:
    payload = {
        "picks": [
            {
                "full_name": "Jalen Brunson",
                "stat_name": "points",
                "stat_value": 28.5,
                "choice": "over",
                "american_price": 100,
                "event_name": "Boston Celtics @ New York Knicks",
                "start_time": "2026-10-26T23:30:00Z",
            },
            {
                "full_name": "Jalen Brunson",
                "stat_name": "points",
                "stat_value": 28.5,
                "choice": "under",
                "american_price": 100,
                "event_name": "Boston Celtics @ New York Knicks",
                "start_time": "2026-10-26T23:30:00Z",
            },
        ]
    }
    assert find_arbs(quotes_from_payload("draftkings", "nba", payload)) == []


def test_different_lines_do_not_match() -> None:
    quotes = quotes_from_payload(
        "novig",
        "nba",
        {
            "games": [
                {
                    "name": "Boston Celtics @ New York Knicks",
                    "scheduled": "2026-10-26T23:30:00Z",
                    "props": [
                        {
                            "player": "Jalen Brunson",
                            "stat": "points",
                            "line": 22.5,
                            "over": {"american": 200},
                            "under": None,
                        }
                    ],
                    "team_markets": {},
                }
            ]
        },
    )
    quotes += quotes_from_payload(
        "fanduel",
        "nba",
        {
            "picks": [
                {
                    "full_name": "Jalen Brunson",
                    "stat_name": "points",
                    "stat_value": 23.5,
                    "choice": "under",
                    "american_price": 200,
                    "event_name": "Boston Celtics @ New York Knicks",
                    "start_time": "2026-10-26T23:30:00Z",
                }
            ]
        },
    )
    assert find_arbs(quotes) == []


def test_player_name_punctuation_and_best_price() -> None:
    quotes = quotes_from_payload(
        "novig",
        "nba",
        {
            "games": [
                {
                    "name": "Dallas Mavericks @ Miami Heat",
                    "scheduled": "2026-10-26T23:40:00Z",
                    "props": [
                        {
                            "player": "P.J. Washington",
                            "stat": "rebounds",
                            "line": 6.5,
                            "over": {"american": 100},
                            "under": None,
                        }
                    ],
                    "team_markets": {},
                }
            ]
        },
    )
    quotes += quotes_from_payload(
        "fanduel",
        "nba",
        {
            "picks": [
                {
                    "full_name": "PJ Washington",
                    "stat_name": "rebounds",
                    "stat_value": 6.5,
                    "choice": "over",
                    "american_price": 150,
                    "event_name": "Dallas Mavericks @ Miami Heat",
                    "start_time": "2026-10-26T23:40:00Z",
                }
            ]
        },
    )
    quotes += quotes_from_payload(
        "draftkings",
        "nba",
        {
            "picks": [
                {
                    "full_name": "P. J. Washington",
                    "stat_name": "rebounds",
                    "stat_value": 6.5,
                    "choice": "under",
                    "american_price": 110,
                    "event_name": "DAL Mavericks @ MIA Heat",
                    "start_time": "2026-10-26T23:40:00Z",
                }
            ]
        },
    )
    arbs = find_arbs(quotes)
    assert len(arbs) == 1
    over = arb_side(arbs[0], "over")
    under = arb_side(arbs[0], "under")
    assert over.book == "fanduel"
    assert over.american == 150
    assert under.book == "draftkings"
    assert under.american == 110


def test_moneyline_abbreviations_match_full_names() -> None:
    quotes = quotes_from_payload("novig", "nba", {"games": [_novig_game()]})
    quotes += quotes_from_payload(
        "prophetx",
        "nba",
        {
            "games": [
                {
                    "name": "Minnesota Timberwolves at Cleveland Cavaliers",
                    "scheduled": "2026-10-26T23:05:00Z",
                    "competitors": [
                        {"name": "Cleveland Cavaliers", "abbreviation": "CLE"},
                        {"name": "Minnesota Timberwolves", "abbreviation": "MIN"},
                    ],
                    "props": [],
                    "team_markets": {
                        "moneyline": [
                            {"name": "Cleveland Cavaliers", "american": 300, "line": None},
                            {"name": "Minnesota Timberwolves", "american": -400, "line": None},
                        ]
                    },
                }
            ]
        },
    )
    arbs = [arb for arb in find_arbs(quotes) if arb.market == "moneyline"]
    assert len(arbs) == 1
    assert arbs[0].edge > 0
    assert {arbs[0].left.book, arbs[0].right.book} == {"novig", "prophetx"}


def test_spread_requires_the_same_number() -> None:
    quotes = quotes_from_payload("novig", "nba", {"games": [_novig_game()]})
    other = _novig_game()
    other["team_markets"] = {
        "spread": [
            {"name": "CLE +4.5", "american": 200, "line": 4.5},
            {"name": "MIN -4.5", "american": 200, "line": -4.5},
        ]
    }
    quotes += quotes_from_payload("prophetx", "nba", {"games": [other]})
    spreads = [arb for arb in find_arbs(quotes) if arb.market == "spread"]
    assert len(spreads) == 1
    labels = f"{spreads[0].left.side_label} {spreads[0].right.side_label}"
    assert "4.5" in labels
    assert "7.5" not in labels


def test_matching_spread_is_an_arb_and_first_half_is_ignored() -> None:
    quotes = quotes_from_payload("novig", "nba", {"games": [_novig_game()]})
    other = _novig_game()
    other["team_markets"] = {
        "spread": [
            {"name": "Cleveland Cavaliers +7.5", "american": 130, "line": 7.5},
            {"name": "Minnesota Timberwolves -7.5", "american": -130, "line": -7.5},
        ],
        "total": [
            {"name": "over 230.5", "american": 115, "line": 230.5},
            {"name": "under 230.5", "american": -140, "line": 230.5},
        ],
    }
    quotes += quotes_from_payload("prophetx", "nba", {"games": [other]})
    arbs = find_arbs(quotes)
    markets = {arb.market for arb in arbs}
    assert "spread" in markets
    assert all("115.5" not in arb.description for arb in arbs)
    total = next(arb for arb in arbs if arb.market == "total")
    assert total.edge > 0


def test_short_event_names_match_full_names() -> None:
    quotes = quotes_from_payload(
        "draftkings",
        "nba",
        {
            "picks": [],
            "game_lines": [
                {
                    "market": "total",
                    "event_name": "PHO Suns @ OKC Thunder",
                    "start_time": "2026-10-27T02:10:00.0000000Z",
                    "side": "over",
                    "line": 220.5,
                    "american": 120,
                },
                {
                    "market": "total",
                    "event_name": "PHO Suns @ OKC Thunder",
                    "start_time": "2026-10-27T02:10:00.0000000Z",
                    "side": "under",
                    "line": 220.5,
                    "american": -150,
                },
            ],
        },
    )
    quotes += quotes_from_payload(
        "fanduel",
        "nba",
        {
            "picks": [],
            "game_lines": [
                {
                    "market": "total",
                    "event_name": "Phoenix Suns @ Oklahoma City Thunder",
                    "start_time": "2026-10-27T02:10:00Z",
                    "side": "under",
                    "line": 220.5,
                    "american": 120,
                }
            ],
        },
    )
    arbs = find_arbs(quotes)
    assert len(arbs) == 1
    assert arbs[0].market == "total"
    under = arb_side(arbs[0], "under")
    assert under.book == "fanduel"
    assert under.american == 120


def test_mlb_run_line_matches_abbreviation_to_full_name() -> None:
    quotes = quotes_from_payload(
        "novig",
        "mlb",
        {
            "games": [
                {
                    "name": "Tampa Bay Rays @ Cleveland Guardians",
                    "scheduled": "2026-10-15T17:00:00.000Z",
                    "competitors": [
                        {"name": "Cleveland Guardians"},
                        {"name": "Tampa Bay Rays"},
                    ],
                    "props": [],
                    "team_markets": {
                        "run_line": [
                            {"name": "TB +1.5", "american": 140, "line": 1.5},
                            {"name": "CLE -1.5", "american": -160, "line": -1.5},
                        ]
                    },
                }
            ]
        },
    )
    quotes += quotes_from_payload(
        "prophetx",
        "mlb",
        {
            "games": [
                {
                    "name": "Tampa Bay Rays at Cleveland Guardians",
                    "scheduled": "2026-10-15T17:00:00Z",
                    "props": [],
                    "team_markets": {
                        "run_line": [
                            {"name": "Tampa Bay Rays +1.5", "american": -120, "line": 1.5},
                            {"name": "Cleveland Guardians -1.5", "american": 130, "line": -1.5},
                        ]
                    },
                }
            ]
        },
    )
    arbs = find_arbs(quotes)
    assert len(arbs) == 1
    assert arbs[0].market == "run line"
    assert arbs[0].edge > 0


def test_format_report_lists_edge_and_none() -> None:
    quotes = quotes_from_payload(
        "novig",
        "nba",
        {
            "games": [
                {
                    "name": "Boston Celtics @ New York Knicks",
                    "scheduled": "2026-10-26T23:30:00Z",
                    "props": [
                        {
                            "player": "Jalen Brunson",
                            "stat": "points",
                            "line": 28.5,
                            "over": {"american": 110},
                            "under": None,
                        }
                    ],
                    "team_markets": {},
                }
            ]
        },
    )
    quotes += quotes_from_payload(
        "fanduel",
        "nba",
        {
            "picks": [
                {
                    "full_name": "Jalen Brunson",
                    "stat_name": "points",
                    "stat_value": 28.5,
                    "choice": "under",
                    "american_price": 110,
                    "event_name": "Boston Celtics @ New York Knicks",
                    "start_time": "2026-10-26T23:30:00Z",
                }
            ]
        },
    )
    text = "\n".join(
        format_report(
            find_arbs(quotes),
            books_seen=["fanduel", "novig"],
            books_missing=["draftkings", "prophetx"],
        )
    )
    assert "=== Arbitrage ===" in text
    assert "books: fanduel, novig" in text
    assert "missing: draftkings, prophetx" in text
    assert "nba prop: Jalen Brunson points 28.5" in text
    assert "over novig +110" in text
    assert "under fanduel +110" in text
    assert "edge " in text
    empty = "\n".join(format_report([], books_seen=["novig"], books_missing=[]))
    assert empty.endswith("none")


def test_report_reads_the_newest_snapshot_only(tmp_path: Path, capsys) -> None:
    league_dir = tmp_path / "novig" / "nba"
    league_dir.mkdir(parents=True)
    other = tmp_path / "fanduel" / "nba"
    other.mkdir(parents=True)
    old = league_dir / "novig_nba_old_props.json"
    new = league_dir / "novig_nba_new_props.json"
    old.write_text(
        json.dumps(
            {
                "league": "nba",
                "games": [
                    {
                        "name": "Boston Celtics @ New York Knicks",
                        "scheduled": "2026-10-26T23:30:00Z",
                        "props": [
                            {
                                "player": "Jalen Brunson",
                                "stat": "points",
                                "line": 10.5,
                                "over": {"american": 300},
                                "under": {"american": 300},
                            }
                        ],
                        "team_markets": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    new.write_text(
        json.dumps(
            {
                "league": "nba",
                "games": [
                    {
                        "name": "Boston Celtics @ New York Knicks",
                        "scheduled": "2026-10-26T23:30:00Z",
                        "props": [
                            {
                                "player": "Jalen Brunson",
                                "stat": "points",
                                "line": 10.5,
                                "over": {"american": -110},
                                "under": {"american": -110},
                            }
                        ],
                        "team_markets": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    os.utime(old, (1_700_000_000, 1_700_000_000))
    os.utime(new, (1_800_000_000, 1_800_000_000))
    (other / "fanduel_nba.json").write_text(
        json.dumps({"picks": []}),
        encoding="utf-8",
    )
    count = report_arbitrage(league="nba", not_before=None, root=tmp_path)
    assert count == 0
    assert "none" in capsys.readouterr().out


def test_report_skips_snapshots_from_before_this_scrape(tmp_path: Path, capsys) -> None:
    league_dir = tmp_path / "novig" / "nba"
    league_dir.mkdir(parents=True)
    path = league_dir / "novig_nba_props.json"
    path.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "name": "Boston Celtics @ New York Knicks",
                        "scheduled": "2026-10-26T23:30:00Z",
                        "props": [
                            {
                                "player": "Jalen Brunson",
                                "stat": "points",
                                "line": 10.5,
                                "over": {"american": 300},
                                "under": {"american": 300},
                            }
                        ],
                        "team_markets": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    os.utime(path, (1_700_000_000, 1_700_000_000))
    count = report_arbitrage(league="nba", not_before=1_700_000_100, root=tmp_path)
    out = capsys.readouterr().out
    assert count == 0
    missing = next(line for line in out.splitlines() if "missing:" in line)
    assert "novig" in missing
    assert "Jalen Brunson" not in out


def arb_side(arb, label: str):
    if arb.left.side_label == label or arb.left.side_label.startswith(label):
        return arb.left
    return arb.right
