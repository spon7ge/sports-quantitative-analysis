"""Layer 2 role history is the player, his team, and the opponent."""

from __future__ import annotations

import pandas as pd
import pytest
from role_context import attach_script, build_role_history, role_for_slate


def _logs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _appearance(**overrides) -> dict:
    row = {
        "season_year": "2023-24",
        "game_id": "g1",
        "game_date": "2024-01-01",
        "team_id": 10,
        "team_abbreviation": "HOM",
        "opp_team_id": 20,
        "matchup": "HOM vs. AWY",
        "player_id": 1,
        "start_position": "G",
        "min": 10.0,
        "pf": 2.0,
        "pace": 100.0,
        "off_rating": 110.0,
        "def_rating": 108.0,
        "team_pts": 110.0,
        "opp_pts": 100.0,
    }
    row.update(overrides)
    return row


def test_training_minutes_use_earlier_games_only():
    logs = _logs(
        [
            _appearance(game_id="g1", game_date="2024-01-01", min=10.0, start_position="G"),
            _appearance(game_id="g2", game_date="2024-01-03", min=30.0, start_position=""),
            _appearance(
                game_id="g1",
                game_date="2024-01-01",
                team_id=20,
                team_abbreviation="AWY",
                opp_team_id=10,
                matchup="AWY @ HOM",
                player_id=2,
                min=240.0,
            ),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                team_id=20,
                team_abbreviation="AWY",
                opp_team_id=10,
                matchup="AWY @ HOM",
                player_id=2,
                min=240.0,
            ),
        ]
    )

    history = build_role_history(logs)
    second = history.loc[history["game_id"].eq("g2") & history["player_id"].eq(1)].iloc[0]

    assert second["min_mean_10"] == 10.0
    assert pd.isna(second["min_std_10"])
    assert second["start_rate_10"] == 1.0
    assert second["is_starter"] == 0.0


def test_same_day_appearances_share_the_prior_window():
    logs = _logs(
        [
            _appearance(game_id="g1", game_date="2024-01-01", min=10.0),
            _appearance(game_id="g1b", game_date="2024-01-01", min=30.0),
            _appearance(game_id="g2", game_date="2024-01-03", min=50.0),
        ]
    )

    history = build_role_history(logs)
    same_day = history.loc[history["game_date"].eq("2024-01-01") & history["player_id"].eq(1)]
    later = history.loc[history["game_id"].eq("g2") & history["player_id"].eq(1)].iloc[0]

    assert same_day["min_mean_10"].isna().all()
    assert later["min_mean_10"] == 20.0


def test_live_minutes_keep_the_latest_game_and_the_posted_start():
    logs = _logs(
        [
            _appearance(game_id="g1", game_date="2024-01-01", min=10.0, start_position="G"),
            _appearance(game_id="g2", game_date="2024-01-03", min=30.0, start_position=""),
        ]
    )
    slate = pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": "2024-01-05",
                "team_id": 10,
                "opp_team_id": 30,
                "is_home": 1,
                "is_starter": 1,
                "team_spread": -4.5,
            }
        ]
    )

    live = role_for_slate(logs, slate).iloc[0]

    assert live["min_mean_10"] == 20.0
    assert live["start_rate_10"] == 0.5
    assert live["is_starter"] == 1.0
    assert live["team_spread"] == -4.5
    assert "team_final_margin" not in live.index


def test_opponent_history_is_that_opponent_and_excludes_the_game_itself():
    logs = _logs(
        [
            _appearance(game_id="g1", game_date="2024-01-01", pace=100.0),
            _appearance(
                game_id="g1",
                game_date="2024-01-01",
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=2,
                pace=80.0,
            ),
            _appearance(game_id="g2", game_date="2024-01-03", pace=999.0),
            _appearance(
                game_id="g2",
                game_date="2024-01-03",
                team_id=20,
                team_abbreviation="AWY",
                matchup="AWY @ HOM",
                player_id=2,
                pace=140.0,
            ),
            _appearance(
                game_id="g3",
                game_date="2024-01-04",
                team_id=30,
                team_abbreviation="NEW",
                matchup="NEW vs. OLD",
                player_id=3,
                pace=70.0,
            ),
        ]
    )
    history = build_role_history(logs)
    second = history.loc[history["game_id"].eq("g2") & history["player_id"].eq(1)].iloc[0]
    assert second["team_pace_10"] == 100.0
    assert second["opp_pace_10"] == 80.0

    slate = pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": "2024-01-06",
                "team_id": 10,
                "opp_team_id": 30,
                "is_home": 0,
                "is_starter": 1,
            }
        ]
    )
    live = role_for_slate(logs, slate).iloc[0]
    assert live["team_pace_10"] == pytest.approx((100.0 + 999.0) / 2)
    assert live["opp_pace_10"] == 70.0
    assert live["opp_team_id"] == 30


def test_training_script_is_this_game_from_the_players_side():
    home = _appearance(
        game_id="g1",
        min=240.0,
        team_pts=120.0,
        opp_pts=100.0,
        player_team_spread=-6.0,
    )
    away = _appearance(
        game_id="g1",
        team_id=20,
        team_abbreviation="AWY",
        matchup="AWY @ HOM",
        player_id=2,
        min=240.0,
        team_pts=100.0,
        opp_pts=120.0,
    )
    overtime_home = _appearance(
        game_id="g2",
        game_date="2024-01-03",
        min=265.0,
        team_pts=110.0,
        opp_pts=108.0,
    )
    overtime_away = _appearance(
        game_id="g2",
        game_date="2024-01-03",
        team_id=20,
        team_abbreviation="AWY",
        matchup="AWY @ HOM",
        player_id=2,
        min=265.0,
        team_pts=108.0,
        opp_pts=110.0,
    )

    history = build_role_history(_logs([home, away, overtime_home, overtime_away]))
    blowout = history.loc[history["game_id"].eq("g1")].set_index("player_id")
    close = history.loc[history["game_id"].eq("g2") & history["player_id"].eq(1)].iloc[0]

    assert blowout.loc[1, "team_final_margin"] == 20.0
    assert blowout.loc[2, "team_final_margin"] == -20.0
    assert blowout["blowout"].tolist() == [1.0, 1.0]
    assert blowout.loc[1, "game_minutes"] == 48.0
    assert blowout.loc[1, "went_ot"] == 0.0
    assert blowout.loc[1, "team_spread"] == -6.0
    assert close["n_ot"] == 1
    assert close["went_ot"] == 1.0
    assert close["game_minutes"] == 53.0
    assert close["blowout"] == 0.0
    assert close["team_final_margin"] == 2.0


def test_rest_uses_the_previous_game_in_training_and_the_slate_date_live():
    logs = _logs(
        [
            _appearance(game_id="g1", game_date="2024-01-01"),
            _appearance(game_id="g2", game_date="2024-01-02"),
        ]
    )
    history = build_role_history(logs)
    first = history.loc[history["game_id"].eq("g1") & history["player_id"].eq(1)].iloc[0]
    second = history.loc[history["game_id"].eq("g2") & history["player_id"].eq(1)].iloc[0]
    assert first["team_rest"] == 7.0
    assert first["team_b2b"] == 0.0
    assert second["team_rest"] == 1.0
    assert second["team_b2b"] == 1.0
    assert second["team_rested"] == 0.0

    slate = pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": "2024-01-06",
                "team_id": 10,
                "opp_team_id": 20,
                "is_home": 1,
                "is_starter": 0,
            }
        ]
    )
    live = role_for_slate(logs, slate).iloc[0]
    assert live["team_rest"] == 4.0
    assert live["team_b2b"] == 0.0
    assert live["team_rested"] == 1.0


def test_attach_script_reuses_the_layer1_draw():
    role = pd.DataFrame(
        [
            {"player_id": 1, "min_mean_10": 32.0},
            {"player_id": 2, "min_mean_10": 18.0},
        ]
    )
    environment = pd.DataFrame(
        {
            "team_margin": [5.0, 30.0],
            "n_ot": [0, 1],
            "game_minutes": [48.0, 53.0],
        }
    )

    attached = attach_script(role, environment)

    assert list(attached["sim_id"]) == [0, 0, 1, 1]
    assert attached.loc[attached["sim_id"].eq(0), "team_final_margin"].tolist() == [5.0, 5.0]
    assert attached.loc[attached["sim_id"].eq(1), "blowout"].tolist() == [1.0, 1.0]
    assert attached.loc[attached["sim_id"].eq(1), "went_ot"].tolist() == [1.0, 1.0]
    assert attached.loc[attached["sim_id"].eq(1), "game_minutes"].tolist() == [53.0, 53.0]


def test_slate_rejects_a_start_flag_that_is_not_zero_or_one():
    logs = _logs([_appearance()])
    slate = pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": "2024-01-05",
                "team_id": 10,
                "opp_team_id": 20,
                "is_home": 1,
                "is_starter": 2,
            }
        ]
    )

    with pytest.raises(ValueError, match="is_starter"):
        role_for_slate(logs, slate)


def test_live_ignores_games_on_or_after_the_slate_date():
    logs = _logs(
        [
            _appearance(game_id="g1", game_date="2024-01-01", min=10.0),
            _appearance(game_id="g2", game_date="2024-01-05", min=90.0),
        ]
    )
    slate = pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": "2024-01-05",
                "team_id": 10,
                "opp_team_id": 20,
                "is_home": 1,
                "is_starter": 1,
            }
        ]
    )

    live = role_for_slate(logs, slate).iloc[0]

    assert live["min_mean_10"] == 10.0
