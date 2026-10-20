"""Live points-per-minute features use the opponent's defensive rating."""

import pandas as pd

from src.live.ppm_live import build_ppm_live


def _game(player_id, name, team, team_id, opponent_id, game_id, date, defense, matchup):
    return {
        "player_id": player_id,
        "player_name": name,
        "team_abbreviation": team,
        "team_id": team_id,
        "opp_team_id": opponent_id,
        "game_id": game_id,
        "game_date": date,
        "season_year": "2024-25",
        "season_type": "Regular Season",
        "matchup": matchup,
        "minutes": 30,
        "start_position": "G",
        "pts": 20,
        "fgm": 8,
        "fga": 16,
        "fg3_m": 2,
        "fg3_a": 6,
        "ftm": 2,
        "fta": 2,
        "usg_pct": 25,
        "tchs": 50,
        "pass": 30,
        "assists": 5,
        "reb": 4,
        "stl": 1,
        "blk": 0,
        "team_pace": 100,
        "team_net_rating": 1,
        "team_off_rating": 112,
        "team_def_rating": defense,
        "team_fga": 90,
        "opp_pace": 100,
        "opp_net_rating": -1,
        "opp_def_rating": 108,
        "player_team_spread": -3.0,
        "game_total": 220.0,
        "available_flag": 1,
        "comment": "",
        "wl": "W",
    }


def test_defense_rating_comes_from_the_opponent_in_the_matchup() -> None:
    history = pd.DataFrame(
        [
            _game(1, "Jayson Tatum", "BOS", 100, 200, 1, "2025-01-01", 90, "BOS vs. NYK"),
            _game(2, "Jalen Brunson", "NYK", 200, 100, 1, "2025-01-01", 110, "NYK @ BOS"),
            _game(1, "Jayson Tatum", "BOS", 100, 200, 2, "2025-01-03", 94, "BOS @ NYK"),
            _game(2, "Jalen Brunson", "NYK", 200, 100, 2, "2025-01-03", 130, "NYK vs. BOS"),
        ]
    )
    tonight = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Jayson Tatum",
                "team_abbreviation": "BOS",
                "team_id": 100,
                "game_id": 3,
                "game_date": "2025-01-05",
                "season_year": "2024-25",
                "matchup": "BOS @ NYK",
                "player_team_spread": -2.0,
                "game_total": 224.0,
                "minutes": 99,
                "pts": 50,
            }
        ]
    )

    live = build_ppm_live(history, tonight, lineups={"BOS": ["Jayson Tatum"]})
    row = live.loc[live["player_id"].eq(1)].iloc[0]

    assert int(row["opp_team_id"]) == 200
    assert row["opp_def_rating_mean_10"] == 120
    assert int(row["starting"]) == 1
