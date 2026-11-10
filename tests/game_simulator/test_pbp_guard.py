"""Play-by-play features keep only games that match the box and predate the slate."""

from __future__ import annotations

import pandas as pd
import pytest
from game_events.links import EventLinks
from pbp_guard import (
    HEAVE_ALLOWANCE,
    links_for_slate,
    possessions_for_slate,
    reconcile_links,
    reconcile_possessions,
)


def _possession(game_id, shots, *, fts=(), tov=0, date="2024-01-15"):
    return {
        "game_id": game_id,
        "game_date": date,
        "shot_player_ids": tuple(player for player, _, _ in shots),
        "shot_zones": tuple(zone for _, zone, _ in shots),
        "shot_made": tuple(made for _, _, made in shots),
        "ft_fta": tuple(attempts for _, attempts, _ in fts),
        "ft_ftm": tuple(makes for _, _, makes in fts),
        "tov": tov,
    }


def _logs(*rows):
    frame = []
    for row in rows:
        frame.append(
            {
                "game_id": row["game_id"],
                "game_date": row.get("game_date", "2024-01-15"),
                "fga": row.get("fga", 0),
                "fgm": row.get("fgm", 0),
                "fta": row.get("fta", 0),
                "ftm": row.get("ftm", 0),
                "tov": row.get("tov", 0),
                "ast": row.get("ast", 0),
                "oreb": row.get("oreb", 0),
                "dreb": row.get("dreb", 0),
                "stl": row.get("stl", 0),
                "blk": row.get("blk", 0),
            }
        )
    return pd.DataFrame(frame)


def test_a_game_that_matches_the_box_is_kept():
    table = pd.DataFrame(
        [
            _possession(
                "g1", ((1, "rim", True), (1, "mid", False)), fts=((1, 2, 1),), tov=1
            ),
        ]
    )
    logs = _logs({"game_id": "g1", "fga": 2, "fgm": 1, "fta": 2, "ftm": 1, "tov": 1})

    kept, flags = reconcile_possessions(table, logs)

    assert list(kept["game_id"]) == ["g1"]
    assert flags.empty


def test_a_short_shot_count_beyond_heaves_is_dropped():
    table = pd.DataFrame([_possession("g1", ((1, "rim", True),))])
    logs = _logs({"game_id": "g1", "fga": 1 + HEAVE_ALLOWANCE + 1, "fgm": 1})

    kept, flags = reconcile_possessions(table, logs)

    assert kept.empty
    assert flags.iloc[0]["game_id"] == "g1"
    assert flags.iloc[0]["stat"] == "fga"
    assert flags.iloc[0]["reason"] == "mismatch"


def test_more_play_by_play_shots_than_the_box_is_dropped():
    table = pd.DataFrame([_possession("g1", ((1, "rim", True), (2, "mid", False)))])
    logs = _logs({"game_id": "g1", "fga": 1, "fgm": 1})

    kept, flags = reconcile_possessions(table, logs)

    assert kept.empty
    assert set(flags["stat"]) >= {"fga"}


def test_a_free_throw_or_turnover_mismatch_drops_the_game():
    table = pd.DataFrame(
        [_possession("g1", ((1, "rim", True),), fts=((1, 2, 2),), tov=0)]
    )
    logs = _logs({"game_id": "g1", "fga": 1, "fgm": 1, "fta": 2, "ftm": 1, "tov": 0})

    kept, flags = reconcile_possessions(table, logs)

    assert kept.empty
    assert flags.iloc[0]["stat"] == "ftm"


def test_possessions_on_the_slate_date_are_excluded_even_when_they_match():
    table = pd.DataFrame(
        [
            _possession("g1", ((1, "rim", True),), date="2024-01-15"),
            _possession("g2", ((1, "rim", False),), date="2024-02-01"),
        ]
    )
    logs = _logs(
        {"game_id": "g1", "game_date": "2024-01-15", "fga": 1, "fgm": 1},
        {"game_id": "g2", "game_date": "2024-02-01", "fga": 1, "fgm": 0},
    )

    kept = possessions_for_slate(table, "2024-02-01", logs)

    assert list(kept["game_id"]) == ["g1"]


def test_a_game_with_no_box_score_is_flagged_and_dropped():
    table = pd.DataFrame([_possession("g9", ((1, "rim", True),))])
    logs = _logs({"game_id": "g1", "fga": 1, "fgm": 1})

    kept, flags = reconcile_possessions(table, logs)

    assert kept.empty
    assert flags.iloc[0]["reason"] == "missing_box"


def _links(game_id, *, assisted=1, oreb=0, dreb=0, stl=0, blk=0):
    shots = []
    for index in range(assisted):
        shots.append(
            {
                "game_id": game_id,
                "assisted": True,
                "passer_id": 2,
                "blocked": index < blk,
                "made": True,
            }
        )
    if blk > assisted:
        for _ in range(blk - assisted):
            shots.append(
                {
                    "game_id": game_id,
                    "assisted": False,
                    "passer_id": pd.NA,
                    "blocked": True,
                    "made": False,
                }
            )
    rebounds = [{"game_id": game_id, "outcome": "oreb"} for _ in range(oreb)]
    rebounds.extend({"game_id": game_id, "outcome": "dreb"} for _ in range(dreb))
    turnovers = [{"game_id": game_id, "stealer_id": 4} for _ in range(stl)]
    return EventLinks(
        pd.DataFrame(shots),
        pd.DataFrame(rebounds),
        pd.DataFrame(turnovers),
    )


def test_missing_assist_tags_drop_the_game_from_credit():
    links = _links("g1", assisted=1)
    logs = _logs({"game_id": "g1", "ast": 4, "fga": 0})

    kept, flags = reconcile_links(links, logs)

    assert kept.shots.empty
    assert flags.iloc[0]["stat"] == "ast"
    assert flags.iloc[0]["reason"] == "mismatch"


def test_misattributed_rebounds_drop_the_game_from_credit():
    links = _links("g1", assisted=0, oreb=1, dreb=0)
    logs = _logs({"game_id": "g1", "ast": 0, "oreb": 3, "dreb": 0})

    kept, flags = reconcile_links(links, logs)

    assert kept.rebounds.empty
    assert flags.iloc[0]["stat"] == "oreb"


def test_credit_links_stop_before_the_slate_date():
    early = _links("g1", assisted=2, oreb=1)
    late = _links("g2", assisted=5, oreb=4)
    links = EventLinks(
        pd.concat([early.shots, late.shots], ignore_index=True),
        pd.concat([early.rebounds, late.rebounds], ignore_index=True),
        pd.DataFrame(columns=["game_id", "stealer_id"]),
    )
    logs = _logs(
        {
            "game_id": "g1",
            "game_date": "2024-01-15",
            "ast": 2,
            "oreb": 1,
        },
        {
            "game_id": "g2",
            "game_date": "2024-02-01",
            "ast": 5,
            "oreb": 4,
        },
    )

    kept = links_for_slate(links, "2024-02-01", logs)

    assert set(kept.shots["game_id"]) == {"g1"}
    assert set(kept.rebounds["game_id"]) == {"g1"}


def test_rows_without_a_date_cannot_be_used_before_a_slate():
    table = pd.DataFrame([_possession("g1", ((1, "rim", True),))])
    table = table.drop(columns="game_date")
    logs = _logs({"game_id": "g9", "fga": 1, "fgm": 1})

    with pytest.raises(ValueError, match="game date"):
        possessions_for_slate(table, "2024-02-01", logs)
