"""Live minutes features use the Rotowire five as the starting flag."""

import pandas as pd

from src.live.minutes_live import MIN_FEATURES, build_minutes_live


def _game(player_id, name, team, minutes, game_id, date, *, start="G"):
    return {
        "player_id": player_id,
        "player_name": name,
        "team_id": 1 if team in {"BOS", "CHO"} else 2,
        "team_abbreviation": team,
        "opp_team_id": 9,
        "game_id": game_id,
        "game_date": date,
        "season_year": "2024-25",
        "season_type": "Regular Season",
        "matchup": "BOS vs. NYK",
        "minutes": minutes,
        "start_position": start,
        "usg_pct": 20.0,
        "tchs": 40,
        "pass": 20,
        "fga": 10,
        "assists": 3,
        "pf": 2,
        "reb": 4,
        "stl": 1,
        "blk": 0,
        "team_pace": 100.0,
        "team_net_rating": 1.0,
        "opp_pace": 99.0,
        "opp_net_rating": -1.0,
        "player_team_spread": -4.0,
        "game_total": 220.0,
        "available_flag": 1,
        "comment": "",
        "wl": "W",
    }


def _history():
    rows = []
    for index, date in enumerate(
        ["2025-01-01", "2025-01-03", "2025-01-05", "2025-01-07", "2025-01-09", "2025-01-11"],
        start=1,
    ):
        rows.append(_game(1, "Jayson Tatum", "BOS", 30 + index, index, date, start="G"))
        rows.append(_game(2, "Payton Pritchard", "BOS", 12, 100 + index, date, start=""))
    return pd.DataFrame(rows)


def _tonight():
    decoy = _game(1, "Jayson Tatum", "CHO", 99, "live-1", "2025-01-13", start="G")
    bench = _game(2, "Payton Pritchard", "CHO", 99, "live-1", "2025-01-13", start="G")
    return pd.DataFrame([decoy, bench])


LINEUPS = {
    "CHA": [
        "Jayson Tatum",
        "Jaylen Brown",
        "Derrick White",
        "Jrue Holiday",
        "Al Horford",
    ]
}


def test_projected_or_confirmed_names_start_and_everyone_else_sits() -> None:
    live = build_minutes_live(_history(), _tonight(), lineups=LINEUPS)

    flags = live.set_index("player_id")["starting"]
    assert int(flags.loc[1]) == 1
    assert int(flags.loc[2]) == 0
    assert live.loc[live["player_id"].eq(1), "min_lag_1"].iloc[0] == 36
    assert set(MIN_FEATURES).issubset(live.columns)


def test_missing_lineup_argument_uses_projected_lineups(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.live.minutes_live.projected_lineups",
        lambda: LINEUPS,
    )

    live = build_minutes_live(_history(), _tonight())

    assert int(live.set_index("player_id").loc[1, "starting"]) == 1
