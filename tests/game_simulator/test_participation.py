"""Low-minute players are drawn in, then the budget is split."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from game_rotations import (
    RotationSimulator,
    allocate_rotation,
    play_prior,
    play_rates,
    shrink_play_rate,
)
from game_rotations.participation import draw_availability


def test_no_team_games_uses_the_minute_tier_prior():
    rate = shrink_play_rate(0, 0, 2.0)

    assert float(rate) == pytest.approx(float(play_prior(2.0)))


def test_a_short_streak_shrinks_toward_the_prior():
    prior = float(play_prior(2.0))
    played_both = float(shrink_play_rate(2, 2, 2.0))
    sat_both = float(shrink_play_rate(0, 2, 2.0))

    assert sat_both > 0.0
    assert played_both < 1.0
    assert sat_both < prior < played_both


def test_more_appearances_than_team_games_is_rejected():
    with pytest.raises(ValueError, match="appearances"):
        shrink_play_rate(3, 2, 2.0)


def test_play_rates_ignore_the_game_being_scored():
    logs = pd.DataFrame(
        [
            _log("g1", "2024-01-01", 1, 2.0),
            _log("g1", "2024-01-01", 9, 30.0),
            _log("g2", "2024-01-03", 9, 30.0),
            _log("g3", "2024-01-05", 9, 30.0),
            _log("g4", "2024-01-07", 1, 40.0),
            _log("g4", "2024-01-07", 9, 30.0),
        ]
    )
    roster = pd.DataFrame(
        [
            {
                "player_id": 1,
                "team_id": 10,
                "game_date": "2024-01-07",
                "min_mean_10": 2.0,
            }
        ]
    )

    rates = play_rates(logs, roster)

    assert int(rates["appearances"].iloc[0]) == 1
    assert int(rates["games"].iloc[0]) == 3
    assert float(rates["play_rate"].iloc[0]) == pytest.approx(
        float(shrink_play_rate(1, 3, 2.0))
    )


def test_rotation_players_stay_in_when_their_rate_is_zero():
    roster = _six()
    roster["min_mean_10"] = [30.0] * 5 + [2.0]
    roster["play_rate"] = [0.0] * 6
    active, factors = draw_availability(
        roster,
        np.array([[50.0] * 5 + [2.0]]),
        np.ones((1, 6), dtype=bool),
        np.array([240.0]),
        np.random.default_rng(0),
    )

    assert active[0, :5].all()
    assert not active[0, 5]
    assert factors[0] < 1.0


def test_a_posted_starter_and_a_probable_player_stay_in():
    roster = _six()
    roster["min_mean_10"] = 2.0
    roster["play_rate"] = 0.0
    roster["is_starter"] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    roster["probable"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    weights = np.array([[10.0, 10.0, 80.0, 80.0, 80.0, 80.0]])
    active, _ = draw_availability(
        roster,
        weights,
        np.ones((1, 6), dtype=bool),
        np.array([240.0]),
        np.random.default_rng(0),
    )

    assert active[0, 0] and active[0, 1]


def test_a_ruled_out_player_stays_out():
    roster = _six()
    roster["min_mean_10"] = 30.0
    roster["play_rate"] = 1.0
    base = np.ones((1, 6), dtype=bool)
    base[0, 5] = False
    active, _ = draw_availability(
        roster,
        np.full((1, 6), 40.0),
        base,
        np.array([240.0]),
        np.random.default_rng(0),
    )

    assert not active[0, 5]


def test_a_short_draw_adds_the_most_likely_player_back():
    roster = _six()
    roster["min_mean_10"] = [30.0] * 5 + [2.0]
    roster["is_starter"] = [1.0] * 5 + [0.0]
    roster["play_rate"] = [1.0] * 5 + [0.0]
    active, factors = draw_availability(
        roster,
        np.array([[40.0] * 5 + [80.0]]),
        np.ones((1, 6), dtype=bool),
        np.array([240.0]),
        np.random.default_rng(0),
    )

    assert active[0, 5]
    assert factors[0] <= 1.0


def test_a_probable_badge_is_listed_and_an_out_badge_is_not(tmp_path):
    from src.live.predict_points import probable_on, unavailable_on

    lineups = tmp_path / "lineups"
    lineups.mkdir()
    (lineups / "rotowire_nba_2026-11-05_180000.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "home": {
                            "team": "LAL",
                            "may_not_play": [
                                {"name": "Anthony Davis", "badge": "Out"},
                                {"name": "Rui Hachimura", "badge": "Probable"},
                            ],
                        },
                        "away": {"team": "BOS", "may_not_play": []},
                    }
                ]
            }
        )
    )

    assert probable_on("2026-11-05", directory=lineups)["LAL"] == ["Rui Hachimura"]
    assert unavailable_on("2026-11-05", directory=lineups)["LAL"] == ["Anthony Davis"]


def test_stint_exposure_splits_a_run_at_the_blowout_margin():
    from game_rotations.pbp import script_exposure

    crossed = _exposure_game("g1", "2024-01-01", score="21-0", sub=True)
    inside = _exposure_game("g2", "2024-01-03", score="20-0", sub=False)
    table = script_exposure(
        pd.DataFrame(crossed + inside),
        pd.DataFrame(_exposure_logs("g1", "2024-01-01") + _exposure_logs("g2", "2024-01-03")),
    )
    first = table.loc[table["game_id"].eq("g1")].set_index("player_id")
    second = table.loc[table["game_id"].eq("g2")].set_index("player_id")

    assert bool(first["reached_20"].iloc[0])
    assert first.loc["1", "close_min"] == pytest.approx(6.0)
    assert first.loc["1", "garbage_app"]
    assert first.loc["5", "close_min"] == pytest.approx(6.0)
    assert first.loc["5", "garbage_app"]
    assert first.loc["6", "close_min"] == pytest.approx(0.0)
    assert first.loc["6", "garbage_app"]
    assert not bool(second["reached_20"].iloc[0])
    assert second.loc["1", "close_min"] == pytest.approx(12.0)
    assert not bool(second.loc["1", "garbage_app"])
    assert "6" not in second.index


def test_script_rates_use_recent_team_games_and_only_games_that_reached_20():
    from game_rotations.participation import script_rates

    rows = []
    for game in range(12):
        rows.append(
            {
                "player_id": 1,
                "team_id": 10,
                "game_id": f"g{game}",
                "game_date": f"2024-01-{game + 1:02d}",
                "close_min": 10.0 if game < 3 or game >= 9 else 0.0,
                "garbage_app": game >= 10,
                "reached_20": game >= 8,
            }
        )
    # A teammate keeps the games the player sat in the team window.
    for game in (6, 7):
        rows.append(
            {
                "player_id": 2,
                "team_id": 10,
                "game_id": f"g{game}",
                "game_date": f"2024-01-{game + 1:02d}",
                "close_min": 30.0,
                "garbage_app": False,
                "reached_20": False,
            }
        )
    roster = pd.DataFrame(
        [
            {
                "player_id": 1,
                "team_id": 10,
                "game_date": "2024-02-01",
            }
        ]
    )

    rates = script_rates(pd.DataFrame(rows), roster)

    # Last 10 team games are g2..g11. Close hits are g2 and g9..g11.
    # Blowouts are g8..g11, and he appeared in garbage in g10 and g11.
    assert int(rates["games"].iloc[0]) == 10
    assert int(rates["close_hits"].iloc[0]) == 4
    assert int(rates["blowout_games"].iloc[0]) == 4
    assert int(rates["garbage_hits"].iloc[0]) == 2
    assert float(rates["close_rate"].iloc[0]) == pytest.approx(5 / 12)
    assert float(rates["blowout_rate"].iloc[0]) == pytest.approx(3 / 6)


def test_script_rates_start_at_the_first_game_with_the_team():
    from game_rotations.participation import script_rates

    rows = [
        _exposure_row(2, "g0", "2024-01-01", 20.0),
        _exposure_row(1, "g1", "2024-01-02", 20.0),
        _exposure_row(2, "g2", "2024-01-03", 30.0),
        _exposure_row(1, "g3", "2024-01-04", 4.0),
    ]
    roster = pd.DataFrame(
        [{"player_id": 1, "team_id": 10, "game_date": "2024-02-01"}]
    )

    rates = script_rates(pd.DataFrame(rows), roster)

    assert int(rates["games"].iloc[0]) == 3
    assert int(rates["close_hits"].iloc[0]) == 2
    assert float(rates["close_rate"].iloc[0]) == pytest.approx(3 / 5)
    assert float(rates["blowout_rate"].iloc[0]) == pytest.approx(1 / 2)


def test_script_probability_follows_the_margin():
    from game_rotations.participation import script_probability

    close = script_probability(0.25, 0.5, 19.9)
    tied = script_probability(0.0, 1.0, 0.0)
    blowout = script_probability(0.25, 0.5, -20.0)

    assert float(close) == pytest.approx(0.25)
    assert float(tied) == pytest.approx(0.0)
    assert float(blowout) == pytest.approx(1 - 0.75 * 0.5)


def test_posted_starters_play_and_ruled_out_players_do_not():
    from game_rotations.participation import draw_script_availability

    roster = _eight()
    roster["close_rate"] = 0.0
    roster["blowout_rate"] = 1.0
    roster["is_starter"] = [1.0] * 5 + [0.0] * 3
    base = np.ones((2, 8), dtype=bool)
    base[:, 4] = False
    active = draw_script_availability(
        roster,
        np.array([0.0, 25.0]),
        base,
        np.random.default_rng(0),
    )

    assert list(active[0, :4]) == [True, True, True, True]
    assert not active[0, 4]
    assert active[1, 5:].all()
    assert not active[1, 4]


def test_a_short_active_set_adds_the_highest_probabilities_until_seven(monkeypatch):
    from game_rotations.participation import draw_script_availability

    monkeypatch.setattr(
        "game_rotations.participation._uniforms",
        lambda rng, size: np.full(size, 0.95),
    )
    roster = _eight()
    roster["is_starter"] = [1.0] * 5 + [0.0] * 3
    roster["close_rate"] = [1.0] * 5 + [0.1, 0.9, 0.4]
    roster["blowout_rate"] = 0.0
    active = draw_script_availability(
        roster,
        np.array([0.0]),
        np.ones((1, 8), dtype=bool),
        np.random.default_rng(0),
    )

    assert active[0, :5].all()
    assert not active[0, 5]
    assert active[0, 6] and active[0, 7]
    assert int(active.sum()) == 7


def test_inactive_players_get_zero_and_the_budget_is_split_over_the_rest():
    roster = _eight()
    roster["is_starter"] = [1.0] * 5 + [0.0] * 3
    roster["min_mean_10"] = [40.0] * 5 + [30.0, 24.0, 18.0]
    roster["play_rate"] = 1.0
    environment = pd.DataFrame(
        {
            "team_margin": np.zeros(300),
            "n_ot": np.zeros(300),
            "game_minutes": np.full(300, 48.0),
        }
    )
    before_model = RotationSimulator(fallback_sd=1.0)
    before = allocate_rotation(
        before_model,
        roster,
        environment,
        np.random.default_rng(2),
    )
    roster["close_rate"] = [1.0] * 7 + [0.0]
    roster["blowout_rate"] = 0.0
    after_model = RotationSimulator(fallback_sd=1.0)
    after = allocate_rotation(
        after_model,
        roster,
        environment,
        np.random.default_rng(2),
    )

    assert np.all(after[:, 7] == 0.0)
    assert abs(after.sum(axis=1) - 240.0).max() < 1e-6
    assert after[:, :5].mean() > before[:, :5].mean() + 1.0
    assert after_model.participation_scale_.mean() > (
        before_model.participation_scale_.mean() + 0.02
    )


def _exposure_row(player_id: int, game_id: str, game_date: str, close_min: float) -> dict:
    return {
        "player_id": player_id,
        "team_id": 10,
        "game_id": game_id,
        "game_date": game_date,
        "close_min": close_min,
        "garbage_app": False,
        "reached_20": False,
    }


def _eight() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": np.arange(8),
            "team_id": np.ones(8, dtype=int),
            "is_starter": np.zeros(8),
        }
    )


def _exposure_logs(game_id: str, game_date: str) -> list[dict]:
    names = {
        "1": "Alpha One",
        "2": "Bravo Two",
        "3": "Charlie Three",
        "4": "Delta Four",
        "5": "Echo Five",
        "6": "Foxtrot Six",
    }
    return [
        {
            "game_id": game_id,
            "player_id": player_id,
            "team_id": "10",
            "game_date": game_date,
            "minutes": 12.0,
            "is_starter": player_id != "6",
            "player_name": name,
        }
        for player_id, name in names.items()
    ]


def _exposure_game(game_id: str, game_date: str, *, score: str, sub: bool) -> list[dict]:
    home, away = score.split("-")
    rows = [
        _exposure_event(game_id, 1, "12:00", None, None, "period", "start", "", "0", "0"),
        _exposure_event(
            game_id, 2, "6:00", "10", "1", "made shot", "", "Alpha makes", home, away
        ),
    ]
    action = 3
    if sub:
        rows.append(
            _exposure_event(
                game_id,
                action,
                "3:00",
                "10",
                None,
                "substitution",
                "",
                "SUB: Foxtrot Six FOR Echo Five",
                home,
                away,
            )
        )
        action += 1
    rows.append(
        _exposure_event(
            game_id, action, "0:00", None, None, "period", "end", "", home, away
        )
    )
    del game_date
    return rows


def _exposure_event(
    game_id, action, clock, team_id, player_id, action_type, sub_type, description, home, away
):
    return {
        "game_id": game_id,
        "action_number": action,
        "period": 1,
        "clock": clock,
        "team_id": team_id,
        "player_id": player_id,
        "action_type": action_type,
        "sub_type": sub_type,
        "score_home": home,
        "score_away": away,
        "description": description,
    }


def test_low_minute_player_misses_scripts_without_breaking_the_budget():
    roster = _six()
    roster["min_mean_10"] = [48.0] * 5 + [2.0]
    roster["is_starter"] = [1.0] * 5 + [0.0]
    roster["play_rate"] = [1.0] * 5 + [0.5]
    environment = pd.DataFrame(
        {
            "team_margin": np.zeros(400),
            "n_ot": np.zeros(400),
            "game_minutes": np.full(400, 48.0),
        }
    )
    minutes = allocate_rotation(
        RotationSimulator(fallback_mu=20.0, fallback_sd=1.0),
        roster,
        environment,
        np.random.default_rng(1),
    )

    absent = minutes[:, 5] == 0.0
    assert absent.mean() == pytest.approx(0.5, abs=0.08)
    assert minutes[~absent, 5].min() > 0.0
    assert np.abs(minutes.sum(axis=1) - 240.0).max() < 1e-6


def _six() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": np.arange(6),
            "team_id": np.ones(6, dtype=int),
            "is_starter": np.zeros(6),
        }
    )


def _log(game_id: str, game_date: str, player_id: int, minutes: float) -> dict:
    return {
        "player_id": player_id,
        "team_id": 10,
        "game_id": game_id,
        "game_date": game_date,
        "min": minutes,
    }
