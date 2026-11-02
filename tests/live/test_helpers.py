"""Opponent lookup uses the season schedule, not the gamelog."""

import pandas as pd

from src.live.helpers import findOpp


def test_find_opp_uses_the_schedule_window(monkeypatch):
    schedule = pd.DataFrame(
        [
            {
                "gameDate": "2026-10-28",
                "homeTeam_teamTricode": "POR",
                "awayTeam_teamTricode": "MEM",
            },
            {
                "gameDate": "2026-11-02",
                "homeTeam_teamTricode": "POR",
                "awayTeam_teamTricode": "BOS",
            },
        ]
    )
    monkeypatch.setattr(
        "src.live.helpers.getUpcomingGamesCached",
        lambda *args, **kwargs: schedule,
    )
    players = pd.DataFrame(
        {
            "player_name": ["Scoot Henderson", "Ja Morant"],
            "team_abbreviation": ["POR", "MEM"],
        }
    )

    assert findOpp("Scoot Henderson", players, "2026-10-28") == ("MEM", 1)
    assert findOpp("Ja Morant", players, "2026-10-28") == ("POR", 0)
