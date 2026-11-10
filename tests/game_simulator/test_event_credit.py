"""Event-level credit: shooter-zone assists, on-floor rebound chances,
rim attempts, and live-ball turnovers.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from game_events.credit import (
    ASSIST_PRIOR_MAKES,
    CHANCE_PRIOR,
    CreditRates,
    allocate_zone_assists,
    assisted_matrix,
    credit_rates,
    passer_tensor,
    profile_with_credit,
)
from game_events.links import event_links
from possession_table import ZONES
from shared_events import allocate_events, simulate_shared_events
from test_event_profiles import _model, _players, _shots
from test_shared_events import _team

GAME = "0022500001"
HOME, AWAY = 100, 200
NAMES = {
    1: "Alpha",
    2: "Bravo",
    3: "Charlie",
    4: "Delta",
    5: "Echo",
    6: "Foxtrot",
    7: "Golf",
    8: "Hector",
    9: "India",
    10: "Juliet",
    11: "Kilo",
}


def _starters():
    return pd.DataFrame(
        [
            {
                "game_id": GAME,
                "team_id": HOME if player < 6 else AWAY,
                "player_id": player,
                "player_name": NAMES[player],
            }
            for player in range(1, 11)
        ]
    )


def _ev(**kwargs):
    team = kwargs.get("team", 0)
    number = kwargs["n"]
    return {
        "gameId": GAME,
        "actionNumber": number,
        "actionId": number,
        "clock": kwargs.get("clock", "PT12M00.00S"),
        "period": kwargs.get("period", 1),
        "teamId": team,
        "personId": kwargs.get("player", 0),
        "playerName": kwargs.get("name", ""),
        "teamTricode": "HOM" if team == HOME else "AWY" if team == AWAY else "",
        "xLegacy": kwargs.get("x", 0),
        "yLegacy": kwargs.get("y", 0),
        "shotDistance": kwargs.get("dist", 0),
        "scoreHome": kwargs.get("home", ""),
        "scoreAway": kwargs.get("away", ""),
        "location": "h" if team == HOME else "v" if team == AWAY else "",
        "description": kwargs.get("description", ""),
        "actionType": kwargs.get("action", ""),
        "subType": kwargs.get("sub", ""),
        "shotValue": kwargs.get("value", 0),
    }


def _opening():
    return [
        _ev(n=1, action="period", sub="start", description="Start of 1st Period"),
        _ev(
            n=2,
            action="Jump Ball",
            team=HOME,
            player=1,
            name="Alpha",
            description="Jump Ball Alpha vs. Foxtrot: Tip to Alpha",
        ),
    ]


def _finish(number):
    return _ev(n=number, action="period", sub="end", clock="PT00M00.00S")


def _links(rows):
    return event_links(pd.DataFrame(rows), _starters())


def _sample():
    """One assisted rim make, one unassisted three, two misses, three turnovers."""
    rows = [
        *_opening(),
        _ev(
            n=3,
            action="Made Shot",
            sub="Layup Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M40.00S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Alpha 2' Layup (2 PTS) (Bravo 1 AST)",
        ),
        _ev(
            n=4,
            action="Made Shot",
            sub="Jump Shot",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT11M20.00S",
            value=3,
            dist=26,
            x=30,
            y=250,
            home="2",
            away="3",
            description="Foxtrot 26' 3PT Jump Shot (3 PTS)",
        ),
        _ev(
            n=5,
            action="Missed Shot",
            sub="Layup Shot",
            team=HOME,
            player=2,
            name="Bravo",
            clock="PT11M00.00S",
            value=2,
            dist=2,
            y=10,
            description="MISS Bravo 2' Layup",
        ),
        _ev(
            n=6,
            action="",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT11M00.00S",
            description="Foxtrot BLOCK (1 BLK)",
        ),
        _ev(
            n=7,
            action="Rebound",
            sub="Unknown",
            team=HOME,
            player=3,
            name="Charlie",
            clock="PT10M58.00S",
            description="Charlie REBOUND (Off:1 Def:0)",
        ),
        _ev(
            n=8,
            action="Missed Shot",
            sub="Jump Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M40.00S",
            value=2,
            dist=15,
            y=150,
            description="MISS Alpha 15' Jump Shot",
        ),
        _ev(
            n=9,
            action="Rebound",
            sub="Unknown",
            team=0,
            player=AWAY,
            name="",
            clock="PT10M38.00S",
            description="AWY Rebound",
        ),
        _ev(
            n=10,
            action="Turnover",
            sub="Bad Pass",
            team=AWAY,
            player=7,
            name="Golf",
            clock="PT10M20.00S",
            description="Golf Bad Pass Turnover (P1.T1)",
        ),
        _ev(
            n=11,
            action="",
            team=HOME,
            player=4,
            name="Delta",
            clock="PT10M20.00S",
            description="Delta STEAL (1 STL)",
        ),
        _ev(
            n=12,
            action="Turnover",
            sub="Lost Ball",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M00.00S",
            description="Alpha Lost Ball Turnover (P1.T2)",
        ),
        _ev(
            n=13,
            action="",
            team=AWAY,
            player=8,
            name="Hector",
            clock="PT10M00.00S",
            description="Hector STEAL (1 STL)",
        ),
        _ev(
            n=14,
            action="Turnover",
            sub="Traveling",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT09M40.00S",
            description="Foxtrot Traveling Turnover (P2.T3)",
        ),
        _ev(
            n=15,
            action="Turnover",
            sub="Out of Bounds - Bad Pass Turnover",
            team=HOME,
            player=2,
            name="Bravo",
            clock="PT09M20.00S",
            description="Bravo Out of Bounds - Bad Pass Turnover (P2.T4)",
        ),
        _finish(16),
    ]
    return _links(rows)


def test_assist_links_record_the_shooter_the_passer_and_the_zone():
    links = _sample()
    made = links.shots.loc[links.shots["made"]].reset_index(drop=True)
    assisted = made.loc[made["shooter_id"].eq(1)].iloc[0]
    alone = made.loc[made["shooter_id"].eq(6)].iloc[0]

    assert assisted["zone"] == "rim"
    assert bool(assisted["assisted"])
    assert assisted["passer_id"] == 2
    assert list(assisted[["off_1", "off_2", "off_3", "off_4", "off_5"]]) == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert not bool(alone["assisted"])
    assert pd.isna(alone["passer_id"])
    assert alone["zone"] == "atb3"


def test_rebound_chances_are_misses_and_team_rebounds_are_not_players():
    links = _sample()
    chances = links.rebounds
    assert list(chances["outcome"]) == ["oreb", "team"]
    assert list(chances["zone"]) == ["rim", "mid"]
    assert chances.iloc[0]["rebounder_id"] == 3
    assert pd.isna(chances.iloc[1]["rebounder_id"])

    rates = credit_rates(links)
    charlie = rates.rebounds.loc[rates.rebounds["player_id"].eq(3)].iloc[0]
    # Charlie was on the floor for both home misses and grabbed one.
    assert charlie["off_chances"] == 2
    assert charlie["oreb"] == 1
    assert charlie["oreb_per_chance"] == pytest.approx(0.5)
    foxtrot = rates.rebounds.loc[rates.rebounds["player_id"].eq(6)].iloc[0]
    # Foxtrot defended both misses and the team rebound is not his.
    assert foxtrot["def_chances"] == 2
    assert foxtrot["dreb"] == 0

    home = rates.teams.loc[rates.teams["team_id"].eq(HOME)].iloc[0]
    assert home["misses"] == 2
    assert home["team_rebounds"] == 1
    assert home["player_rebounds"] == 1
    assert home["player_rebounds"] + home["team_rebounds"] == home["misses"]


def test_a_player_off_the_floor_does_not_get_the_rebound_chance():
    rows = [
        *_opening(),
        _ev(
            n=3,
            action="substitution",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M30.00S",
            description="SUB: Kilo FOR Alpha",
        ),
        _ev(
            n=4,
            action="Missed Shot",
            sub="Jump Shot",
            team=HOME,
            player=2,
            name="Bravo",
            clock="PT11M00.00S",
            value=2,
            dist=15,
            y=150,
            description="MISS Bravo 15' Jump Shot",
        ),
        _ev(
            n=5,
            action="Rebound",
            sub="Unknown",
            team=HOME,
            player=11,
            name="Kilo",
            clock="PT10M58.00S",
            description="Kilo REBOUND (Off:1 Def:0)",
        ),
        _finish(6),
    ]
    rates = credit_rates(_links(rows))
    chances = rates.rebounds.set_index("player_id")
    assert 1 not in chances.index or chances.loc[1, "off_chances"] == 0
    assert chances.loc[11, "off_chances"] == 1
    assert chances.loc[11, "oreb"] == 1
    assert chances.loc[2, "off_chances"] == 1
    assert chances.loc[2, "oreb"] == 0


def test_blocks_are_rim_attempts_faced_by_the_defender_on_the_floor():
    rates = credit_rates(_sample())
    foxtrot = rates.blocks.loc[rates.blocks["player_id"].eq(6)].iloc[0]
    # The made layup and the blocked layup are rim attempts. The jumper is not.
    assert foxtrot["rim_faced"] == 2
    assert foxtrot["blk"] == 1
    assert foxtrot["blk_per_rim"] == pytest.approx(0.5)
    assert rates.blocks["rim_faced"].sum() == 10
    away = rates.teams.loc[rates.teams["team_id"].eq(AWAY)].iloc[0]
    assert away["rim_attempts"] == 2
    assert away["blocks"] == 1


def test_steals_attach_only_to_live_ball_turnover_labels():
    links = _sample()
    live = set(links.turnovers.loc[links.turnovers["live_ball"], "subtype"])
    dead = set(links.turnovers.loc[~links.turnovers["live_ball"], "subtype"])
    assert live == {"bad pass", "lost ball"}
    assert "traveling" in dead
    assert "out of bounds - bad pass turnover" in dead
    assert links.turnovers.loc[~links.turnovers["live_ball"], "stealer_id"].isna().all()

    rates = credit_rates(links)
    delta = rates.steals.loc[rates.steals["player_id"].eq(4)].iloc[0]
    assert delta["live_faced"] == 1
    assert delta["stl"] == 1
    home = rates.teams.loc[rates.teams["team_id"].eq(HOME)].iloc[0]
    # Home defended one live-ball turnover and stole it. Traveling is not a chance.
    assert home["live_turnovers"] == 1
    assert home["steals"] == 1


def test_assisted_probability_depends_on_the_shooter_and_the_zone():
    links = _sample()
    # A second rim make by Alpha, this one unassisted, and a mid make that is assisted.
    extra = links.shots.copy()
    assisted_mid = extra.iloc[0].copy()
    assisted_mid["zone"] = "mid"
    unassisted_rim = extra.iloc[0].copy()
    unassisted_rim["assisted"] = False
    unassisted_rim["passer_id"] = pd.NA
    links = replace(
        links,
        shots=pd.concat(
            [extra, pd.DataFrame([assisted_mid, unassisted_rim])], ignore_index=True
        ),
    )
    assisted = credit_rates(links).assisted.set_index(["player_id", "zone"])
    assert assisted.loc[(1, "rim"), "p_assisted"] == pytest.approx(0.5)
    assert assisted.loc[(1, "mid"), "p_assisted"] == pytest.approx(1.0)
    assert assisted.loc[(6, "atb3"), "p_assisted"] == pytest.approx(0.0)
    assert list(assisted.index.unique(level="zone").difference(ZONES)) == []


def test_passer_tensor_keeps_the_passers_of_each_shooter():
    counts = np.array(
        [
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 4.0],
            [9.0, 0.0, 0.0],
        ]
    )
    on_floor = np.array([[True, True, True]])
    weights = passer_tensor(
        creation=np.ones((1, 3)),
        counts=counts,
        strength=0.0,
        on_floor=on_floor,
    )
    assert weights.shape == (1, 3, 3)
    assert np.array_equal(weights[0, 0], [0.0, 5.0, 0.0])
    assert np.array_equal(weights[0, 1], [0.0, 0.0, 4.0])
    # A teammate who left the floor cannot assist, even with a history.
    sat = passer_tensor(
        creation=np.ones((1, 3)),
        counts=counts,
        strength=0.0,
        on_floor=np.array([[True, False, True]]),
    )
    assert np.array_equal(sat[0, 0], [0.0, 0.0, 0.0])
    assert sat[0, 2, 0] == pytest.approx(9.0)


def test_zone_assists_follow_who_shot_and_not_a_team_rate():
    rng = np.random.default_rng(1)
    n_sims = 20
    zone_fgm = np.zeros((n_sims, 3, 4), dtype=np.int64)
    zone_fgm[:, 0, 0] = 4  # player 0, rim
    zone_fgm[:, 2, 1] = 6  # player 2, mid
    p_assisted = np.zeros((3, 4))
    p_assisted[0, 0] = 1.0
    p_assisted[2, 1] = 1.0
    counts = np.zeros((3, 3))
    counts[0, 1] = 8  # player 1 assists player 0
    counts[2, 0] = 8  # player 0 assists player 2
    weights = passer_tensor(
        creation=np.ones((n_sims, 3)),
        counts=counts,
        strength=0.0,
        on_floor=np.ones((n_sims, 3), dtype=bool),
    )
    assists, pairs = allocate_zone_assists(zone_fgm, p_assisted, weights, rng)
    assert np.array_equal(assists[:, 1], np.full(n_sims, 4))
    assert np.array_equal(assists[:, 0], np.full(n_sims, 6))
    assert np.all(assists[:, 2] == 0)
    assert np.all(pairs[:, 1, 1] == 0)
    assert np.all(pairs[:, 0, 0] == 0)


def test_shared_draw_blocks_rim_attempts_and_steals_live_turnovers():
    n_sims = 12
    offense = _team(n_sims, 2, fga=10, fgm=0, tov=4)
    offense["rim_fga"] = np.array([[2, 0]] * n_sims, dtype=np.int64)
    offense["live_tov"] = np.array([[1, 0]] * n_sims, dtype=np.int64)
    defense = _team(n_sims, 1, fga=0, fgm=0, tov=0)
    result = simulate_shared_events(
        {"OFF": offense, "DEF": defense},
        n_sims=n_sims,
        rng=np.random.default_rng(2),
        p_rebound=0.0,
        p_steal=1.0,
        p_block=1.0,
    )
    # Ten misses, but only two are rim attempts. Four turnovers, one is live.
    assert np.array_equal(result.teams["DEF"].stats["BLK"].sum(1), np.full(n_sims, 2))
    assert np.array_equal(result.teams["DEF"].stats["STL"].sum(1), np.full(n_sims, 1))


def test_chance_rates_replace_per_minute_rebound_weights():
    n_sims = 30
    offense = _players(
        {
            "p_rebound": 1.0,
            "p_oreb": 1.0,
            "assisted_rate": 0.0,
            "p_steal": 0.0,
            "p_block": 0.0,
        },
        [
            {"player_id": 1, "oreb_per_min": 5.0, "oreb_per_chance": 0.0},
            {"player_id": 2, "oreb_per_min": 0.0, "oreb_per_chance": 1.0},
        ],
    )
    defense = _players(
        {
            "p_rebound": 0.0,
            "p_oreb": 0.0,
            "assisted_rate": 0.0,
            "p_steal": 0.0,
            "p_block": 0.0,
        },
        [{"player_id": 3, "dreb_per_min": 1.0}],
    )
    drawn = allocate_events(
        _model(),
        {"OFF": offense, "DEF": defense},
        {
            "OFF": _shots(n_sims, [5, 5], [0, 0], [0, 0]),
            "DEF": _shots(n_sims, [0], [0], [0]),
        },
        {
            "OFF": np.tile(np.array([[20.0, 20.0]]), (n_sims, 1)),
            "DEF": np.ones((n_sims, 1)) * 20.0,
        },
        np.random.default_rng(3),
    )
    assert np.array_equal(drawn["OFF"]["OREB"][:, 1], np.full(n_sims, 10))
    assert np.all(drawn["OFF"]["OREB"][:, 0] == 0)


def _deterministic():
    """Assisted mid make, one blocked rim miss that the team rebounds, one steal."""
    rows = [
        *_opening(),
        _ev(
            n=3,
            action="Made Shot",
            sub="Jump Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M40.00S",
            value=2,
            dist=15,
            y=150,
            home="2",
            away="0",
            description="Alpha 15' Jump Shot (2 PTS) (Bravo 1 AST)",
        ),
        _ev(
            n=4,
            action="Missed Shot",
            sub="Layup Shot",
            team=HOME,
            player=2,
            name="Bravo",
            clock="PT11M00.00S",
            value=2,
            dist=2,
            y=10,
            description="MISS Bravo 2' Layup",
        ),
        _ev(
            n=5,
            action="",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT11M00.00S",
            description="Foxtrot BLOCK (1 BLK)",
        ),
        _ev(
            n=6,
            action="Rebound",
            sub="Unknown",
            team=0,
            player=AWAY,
            name="",
            clock="PT10M58.00S",
            description="AWY Rebound",
        ),
        _ev(
            n=7,
            action="Turnover",
            sub="Bad Pass",
            team=AWAY,
            player=7,
            name="Golf",
            clock="PT10M20.00S",
            description="Golf Bad Pass Turnover (P1.T1)",
        ),
        _ev(
            n=8,
            action="",
            team=HOME,
            player=4,
            name="Delta",
            clock="PT10M20.00S",
            description="Delta STEAL (1 STL)",
        ),
        _finish(9),
    ]
    return _links(rows)


def test_credit_profiles_drive_the_allocation():
    rates = credit_rates(_deterministic())
    away = rates.teams.loc[rates.teams["team_id"].eq(AWAY)].iloc[0]
    home = rates.teams.loc[rates.teams["team_id"].eq(HOME)].iloc[0]
    assert away["p_block"] == pytest.approx(1.0)
    assert home["p_steal"] == pytest.approx(1.0)
    assert home["p_rebound"] == pytest.approx(0.0)

    n_sims = 80
    home_ids = [1, 2, 3, 4, 5]
    away_ids = [6, 7, 8, 9, 10]
    zone = np.zeros((n_sims, 5, 4), dtype=np.int64)
    zone[:, 0, 1] = 4  # Alpha midrange makes
    home_shots = _shots(n_sims, [6, 0, 0, 0, 0], [4, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    home_shots["ZONE_FGM"] = zone
    home_shots["RIM_FGA"] = np.zeros((n_sims, 5), dtype=np.int64)
    home_shots["RIM_FGA"][:, 0] = 2
    home_shots["LIVE_TOV"] = np.zeros((n_sims, 5), dtype=np.int64)
    away_shots = _shots(n_sims, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [3, 0, 0, 0, 0])
    away_shots["ZONE_FGM"] = np.zeros((n_sims, 5, 4), dtype=np.int64)
    away_shots["RIM_FGA"] = np.zeros((n_sims, 5), dtype=np.int64)
    away_shots["LIVE_TOV"] = np.zeros((n_sims, 5), dtype=np.int64)
    away_shots["LIVE_TOV"][:, 0] = 1
    drawn = allocate_events(
        _model(p_block=0.0, p_steal=0.0, p_rebound=1.0, p_oreb=1.0),
        {
            "HOM": pd.DataFrame({"player_id": home_ids, "team_id": HOME}),
            "AWY": pd.DataFrame({"player_id": away_ids, "team_id": AWAY}),
        },
        {"HOM": home_shots, "AWY": away_shots},
        {"HOM": np.full((n_sims, 5), 10.0), "AWY": np.full((n_sims, 5), 10.0)},
        np.random.default_rng(4),
        credit=rates,
        resimulate=("AST",),
    )
    # The links say Alpha's midrange makes are assisted. One observed pass to
    # Bravo is shrunk across the other teammates on the floor, and Alpha
    # cannot assist himself.
    assists = drawn["HOM"]["AST"].sum(axis=0)
    assert assists.sum() == n_sims * 4
    assert assists[0] == 0
    assert assists[1] > assists[2]
    assert assists[1] > assists[3]
    assert assists[1] > assists[4]
    assert assists[1] < assists.sum()
    assert np.array_equal(drawn["AWY"]["BLK"].sum(1), np.full(n_sims, 2))
    assert np.array_equal(drawn["HOM"]["STL"].sum(1), np.full(n_sims, 1))
    # The only miss in the links was a team rebound, so players get none of
    # the two simulated misses.
    assert np.all(drawn["HOM"]["OREB"].sum(1) == 0)
    assert np.all(drawn["AWY"]["DREB"].sum(1) == 0)


def test_shooter_zone_assist_rate_shrinks_toward_the_zone():
    empty = pd.DataFrame(columns=["player_id"])
    rates = CreditRates(
        assisted=pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "zone": "rim",
                    "makes": 2,
                    "assisted": 1,
                    "p_assisted": 0.5,
                },
                {
                    "player_id": 9,
                    "zone": "rim",
                    "makes": 20,
                    "assisted": 2,
                    "p_assisted": 0.1,
                },
            ]
        ),
        passers=empty,
        rebounds=empty,
        blocks=empty,
        steals=empty,
        teams=empty,
    )
    matrix = assisted_matrix(rates, (1, 9))
    league = 3 / 22
    assert matrix[0, 0] == pytest.approx(
        (1 + ASSIST_PRIOR_MAKES * league) / (2 + ASSIST_PRIOR_MAKES)
    )
    assert matrix[0, 0] != pytest.approx(0.5)


def test_chance_rates_on_the_profile_shrink_toward_the_table():
    rebounds = pd.DataFrame(
        [
            {
                "player_id": 1,
                "off_chances": 2,
                "oreb": 2,
                "oreb_per_chance": 1.0,
                "def_chances": 0,
                "dreb": 0,
                "dreb_per_chance": np.nan,
            },
            {
                "player_id": 2,
                "off_chances": 20,
                "oreb": 0,
                "oreb_per_chance": 0.0,
                "def_chances": 0,
                "dreb": 0,
                "dreb_per_chance": np.nan,
            },
        ]
    )
    empty = pd.DataFrame(columns=["player_id", "team_id"])
    rates = CreditRates(
        assisted=empty,
        passers=empty,
        rebounds=rebounds,
        blocks=empty,
        steals=empty,
        teams=empty,
    )
    profile = profile_with_credit(pd.DataFrame({"player_id": [1, 2]}), rates)
    league = 2 / 22
    assert profile.iloc[0]["oreb_per_chance"] == pytest.approx(
        (2 + CHANCE_PRIOR * league) / (2 + CHANCE_PRIOR)
    )
    assert profile.iloc[0]["oreb_per_chance"] < 0.5
