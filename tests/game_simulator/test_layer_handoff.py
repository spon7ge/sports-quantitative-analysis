"""Layer 4 hands zone counts to Layer 5, which draws the shared events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from game_events import allocate_events
from game_events.credit import CHANCE_PRIOR, credit_rates
from game_events.links import EventLinks
from game_events.model import EventModel
from game_opportunities import (
    OpportunityModel,
    allocate_game_opportunities,
    allocate_opportunities,
)
from possession_table import ZONES

RIM = ZONES.index("rim")
ATB = ZONES.index("atb3")


def _opportunity_model() -> OpportunityModel:
    return OpportunityModel(
        fga_per_min=0.5,
        p_3pa=0.0,
        p_3p=0.0,
        p_2p=0.5,
        fta_per_fga=0.0,
        ft_pct=0.0,
        tov_per_min=0.1,
        fga_per_poss=1.0,
        tov_per_poss=0.5,
        fta_per_min=0.0,
        fta_per_poss=0.0,
        alpha=0.0,
        tov_rho=0.0,
    )


def _opportunity_profile(player_ids) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": list(player_ids),
            "p_2p": 0.5,
            "p_3pa": 0.0,
            "p_live_tov": 1.0,
            "rim_share": 0.5,
            "mid_share": 0.5,
            "corner3_share": 0.0,
            "atb3_share": 0.0,
            "rim_fg_pct": 0.5,
            "mid_fg_pct": 0.5,
            "corner3_fg_pct": 0.0,
            "atb3_fg_pct": 0.0,
        }
    )


def _quarters(players) -> list[dict]:
    return [
        {
            "period": period,
            "start_min": 0.0,
            "end_min": 12.0,
            "players": list(players),
        }
        for period in range(1, 5)
    ]


def _assert_layer4_contract(drawn: dict) -> None:
    zone_fga = drawn["ZONE_FGA"]
    zone_fgm = drawn["ZONE_FGM"]
    assert zone_fga.shape[-1] == len(ZONES)
    assert np.array_equal(zone_fga.sum(axis=2), drawn["FGA"])
    assert np.array_equal(zone_fgm.sum(axis=2), drawn["FGM"])
    assert np.array_equal(zone_fga[:, :, 0] + zone_fga[:, :, 1], drawn["2PA"])
    assert np.array_equal(zone_fga[:, :, 2] + zone_fga[:, :, 3], drawn["3PA"])
    assert np.array_equal(zone_fgm[:, :, 0] + zone_fgm[:, :, 1], drawn["2PM"])
    assert np.array_equal(zone_fgm[:, :, 2] + zone_fgm[:, :, 3], drawn["3PM"])
    assert np.all(zone_fgm <= zone_fga)
    points = (
        2 * (zone_fgm[:, :, 0] + zone_fgm[:, :, 1])
        + 3 * (zone_fgm[:, :, 2] + zone_fgm[:, :, 3])
        + drawn["FTM"]
    )
    assert np.array_equal(points, drawn["PTS"])
    assert np.array_equal(drawn["PTS"].sum(axis=1), np.full(drawn["PTS"].shape[0], 2))
    assert np.array_equal(
        drawn["LIVE_TOV"] + drawn["DEAD_TOV"],
        drawn["TOV"],
    )
    assert np.array_equal(drawn["LIVE_TOV"], drawn["TOV"])
    assert np.array_equal(drawn["DEAD_TOV"], np.zeros_like(drawn["TOV"]))
    misses = (zone_fga - zone_fgm).sum(axis=1)
    assert np.array_equal(drawn["MISS_BY_ZONE"], misses)
    assert drawn["FGA"].sum() > drawn["FGM"].sum()


def test_layer4_outputs_follow_the_conditioned_score():
    players = [1, 2, 3, 4, 5]
    minutes = np.full((4, 5), 48.0)
    drawn = allocate_opportunities(
        _opportunity_model(),
        _opportunity_profile(players),
        minutes,
        np.full(4, 40.0),
        np.random.default_rng(1),
        team_points=np.full(4, 2.0),
        game_minutes=np.full(4, 48.0),
    )
    _assert_layer4_contract(drawn)

    segmented = allocate_opportunities(
        _opportunity_model(),
        _opportunity_profile(players),
        minutes,
        np.full(4, 40.0),
        np.random.default_rng(2),
        team_points=np.full(4, 2.0),
        game_minutes=np.full(4, 48.0),
        segments=[_quarters(players)] * 4,
        return_intervals=True,
    )
    _assert_layer4_contract(segmented)
    for interval in segmented["intervals"]:
        for values in interval["stats"].values():
            assert values.ndim == 2


def _event_model() -> EventModel:
    return EventModel(
        version=2,
        cutoff="2024-01-01",
        ast_per_min=0.0,
        oreb_per_min=0.0,
        dreb_per_min=0.0,
        stl_per_min=0.0,
        blk_per_min=0.0,
        assisted_rate=0.0,
        p_rebound=0.0,
        p_oreb=0.0,
        p_oreb_allowed=0.0,
        p_steal=0.0,
        p_steal_against=0.0,
        p_block=0.0,
        p_block_against=0.0,
    )


def _event_profile(rows: list[dict], **team) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column, value in team.items():
        frame[column] = value
    return frame


def test_layer5_draws_events_from_the_zone_arrays():
    n_sims = 5
    zone_fgm = np.zeros((n_sims, 2, 4), dtype=np.int64)
    zone_fgm[:, 0, RIM] = 4
    misses = np.zeros((n_sims, 4), dtype=np.int64)
    misses[:, RIM] = 2
    misses[:, ATB] = 4
    shots = {
        "FGA": np.tile(np.array([10, 0]), (n_sims, 1)),
        "FGM": np.tile(np.array([4, 0]), (n_sims, 1)),
        "TOV": np.tile(np.array([5, 0]), (n_sims, 1)),
        "ZONE_FGM": zone_fgm,
        "LIVE_TOV": np.tile(np.array([1, 0]), (n_sims, 1)),
        "MISS_BY_ZONE": misses,
        # These would be reused if Layer 5 did not consume the arrays.
        "AST": np.zeros((n_sims, 2), dtype=np.int64),
        "OREB": np.full((n_sims, 2), 99, dtype=np.int64),
        "RIM_FGA": np.tile(np.array([8, 0]), (n_sims, 1)),
    }
    quiet = {
        "FGA": np.zeros((n_sims, 2), dtype=np.int64),
        "FGM": np.zeros((n_sims, 2), dtype=np.int64),
        "TOV": np.zeros((n_sims, 2), dtype=np.int64),
        "ZONE_FGM": np.zeros((n_sims, 2, 4), dtype=np.int64),
        "LIVE_TOV": np.zeros((n_sims, 2), dtype=np.int64),
        "MISS_BY_ZONE": np.zeros((n_sims, 4), dtype=np.int64),
    }
    offense = _event_profile(
        [
            {"player_id": 1, "ast_per_min": 0.0, "oreb_per_min": 1.0},
            {"player_id": 2, "ast_per_min": 1.0, "oreb_per_min": 1.0},
        ],
        assisted_rate=1.0,
        p_rebound=0.0,
        p_oreb=0.0,
        rim_p_oreb=1.0,
        mid_p_oreb=0.0,
        corner3_p_oreb=0.0,
        atb3_p_oreb=0.0,
        p_steal=0.0,
        p_block=0.0,
    )
    defense = _event_profile(
        [
            {
                "player_id": 3,
                "stl_per_min": 1.0,
                "blk_per_min": 1.0,
                "dreb_per_min": 1.0,
            },
            {
                "player_id": 4,
                "stl_per_min": 1.0,
                "blk_per_min": 1.0,
                "dreb_per_min": 1.0,
            },
        ],
        assisted_rate=0.0,
        p_rebound=0.0,
        p_oreb=0.0,
        p_steal=1.0,
        p_block=1.0,
    )
    drawn = allocate_events(
        _event_model(),
        {"OFF": offense, "DEF": defense},
        {"OFF": shots, "DEF": quiet},
        {
            "OFF": np.full((n_sims, 2), 20.0),
            "DEF": np.tile(np.array([20.0, 0.0]), (n_sims, 1)),
        },
        np.random.default_rng(3),
        dispersion=False,
    )
    # Four rim makes, all assisted, and the shooter cannot assist himself.
    assert np.array_equal(drawn["OFF"]["AST"][:, 1], np.full(n_sims, 4))
    assert np.all(drawn["OFF"]["AST"][:, 0] == 0)
    # Rim misses are offensive rebounds. Above-the-break misses are not.
    assert np.array_equal(drawn["OFF"]["OREB"].sum(1), np.full(n_sims, 2))
    assert np.array_equal(drawn["DEF"]["DREB"].sum(1), np.full(n_sims, 4))
    # One live-ball turnover and two rim misses, credited to the defender
    # who is on the floor. The bench defender and the dead-ball turnovers
    # do not add steals, and made rim attempts do not add blocks.
    assert np.array_equal(drawn["DEF"]["STL"].sum(1), np.full(n_sims, 1))
    assert np.all(drawn["DEF"]["STL"][:, 1] == 0)
    assert np.array_equal(drawn["DEF"]["BLK"].sum(1), np.full(n_sims, 2))
    assert np.all(drawn["DEF"]["BLK"][:, 1] == 0)


def test_offensive_rebound_rate_depends_on_the_miss_zone():
    lineup = {f"off_{index}": index for index in range(1, 6)}
    lineup.update({f"def_{index}": index + 5 for index in range(1, 6)})
    rebounds = []
    for _ in range(2):
        rebounds.append(
            {
                "offense_team_id": 10,
                "defense_team_id": 20,
                "zone": "rim",
                "outcome": "oreb",
                "rebounder_id": 1,
                **lineup,
            }
        )
    for _ in range(8):
        rebounds.append(
            {
                "offense_team_id": 10,
                "defense_team_id": 20,
                "zone": "atb3",
                "outcome": "dreb",
                "rebounder_id": 6,
                **lineup,
            }
        )
    shots = pd.DataFrame(columns=["zone", "defense_team_id", "blocked"])
    turnovers = pd.DataFrame(columns=["live_ball", "defense_team_id"])
    rates = credit_rates(EventLinks(shots, pd.DataFrame(rebounds), turnovers))
    home = rates.teams.loc[rates.teams["team_id"].eq(10)].iloc[0]
    league = 2 / 10
    assert home["rim_p_oreb"] == pytest.approx(
        (2 + CHANCE_PRIOR * league) / (2 + CHANCE_PRIOR)
    )
    assert home["atb3_p_oreb"] == pytest.approx(
        (CHANCE_PRIOR * league) / (8 + CHANCE_PRIOR)
    )
    assert home["rim_p_oreb"] > home["atb3_p_oreb"]


def test_layer6_forwards_zone_arrays_and_drops_upstream_events():
    from game_boxscore.simulate import _shots_for_events

    source = {
        "FGA": np.ones((1, 1), dtype=np.int64),
        "FGM": np.ones((1, 1), dtype=np.int64),
        "2PA": np.ones((1, 1), dtype=np.int64),
        "2PM": np.ones((1, 1), dtype=np.int64),
        "3PA": np.zeros((1, 1), dtype=np.int64),
        "3PM": np.zeros((1, 1), dtype=np.int64),
        "FTA": np.zeros((1, 1), dtype=np.int64),
        "FTM": np.zeros((1, 1), dtype=np.int64),
        "PTS": np.full((1, 1), 2, dtype=np.int64),
        "TOV": np.ones((1, 1), dtype=np.int64),
        "AST": np.ones((1, 1), dtype=np.int64),
        "OREB": np.ones((1, 1), dtype=np.int64),
        "ZONE_FGM": np.ones((1, 1, 4), dtype=np.int64),
        "LIVE_TOV": np.ones((1, 1), dtype=np.int64),
        "DEAD_TOV": np.zeros((1, 1), dtype=np.int64),
        "MISS_BY_ZONE": np.zeros((1, 4), dtype=np.int64),
    }
    forwarded = _shots_for_events(source, "Layer 4 home")
    assert "ZONE_FGM" in forwarded
    assert "LIVE_TOV" in forwarded
    assert "MISS_BY_ZONE" in forwarded
    assert "AST" not in forwarded
    assert "OREB" not in forwarded
    assert "FGA" in forwarded


def test_full_layer4_dict_is_what_layer5_draws_from():
    """The notebooks pass each side's whole Layer 4 dict into Layer 5."""

    players = [1, 2, 3, 4, 5]
    opponents = [11, 12, 13, 14, 15]
    environment = pd.DataFrame(
        {
            "is_home": [1],
            "team_poss": [30.0],
            "opp_poss": [30.0],
            "game_minutes": [48.0],
            "n_ot": [0],
            "team_pts": [20.0],
            "opp_pts": [18.0],
        }
    )
    drawn = allocate_game_opportunities(
        _opportunity_model(),
        _opportunity_profile(players),
        _opportunity_profile(opponents),
        pd.DataFrame(np.full((1, 5), 48.0), columns=players),
        pd.DataFrame(np.full((1, 5), 48.0), columns=opponents),
        environment,
        np.random.default_rng(4),
        home_segments=[_quarters(players)],
        away_segments=[_quarters(opponents)],
    )
    home, away = drawn["home"], drawn["away"]

    def event_profile(ids):
        frame = pd.DataFrame({"player_id": ids})
        for column, value in {
            "assisted_rate": 1.0,
            "p_rebound": 1.0,
            "p_oreb": 0.25,
            "p_oreb_allowed": 0.25,
            "p_steal": 1.0,
            "p_steal_against": 1.0,
            "p_block": 1.0,
            "p_block_against": 1.0,
            "ast_per_min": 0.2,
            "oreb_per_min": 0.1,
            "dreb_per_min": 0.3,
            "stl_per_min": 0.05,
            "blk_per_min": 0.05,
        }.items():
            frame[column] = value
        return frame

    stats, details = allocate_events(
        _event_model(),
        {"H": event_profile(players), "A": event_profile(opponents)},
        {"H": home, "A": away},
        {"H": np.full((1, 5), 48.0), "A": np.full((1, 5), 48.0)},
        np.random.default_rng(5),
        segments={"H": [_quarters(players)], "A": [_quarters(opponents)]},
        return_details=True,
        dispersion=False,
    )
    assert details["sources"]["H"]["AST"] == "layer5"
    assert details["sources"]["H"]["OREB"] == "layer5"
    assert not np.array_equal(stats["H"]["AST"], home["AST"])
    misses = home["FGA"].sum(axis=1) - home["FGM"].sum(axis=1)
    rebounds = stats["H"]["OREB"].sum(axis=1) + stats["A"]["DREB"].sum(axis=1)
    assert np.array_equal(rebounds, misses)
    assert np.all(stats["A"]["STL"].sum(axis=1) <= home["LIVE_TOV"].sum(axis=1))
    assert np.all(stats["A"]["BLK"].sum(axis=1) <= home["MISS_BY_ZONE"][:, RIM])
