"""One row per possession: lineups, margin, and the shot or foul outcome."""

from __future__ import annotations

import pandas as pd
import pytest
from possession_table import garbage_time, possession_table, shot_zone

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


def starters():
    rows = [
        {
            "game_id": GAME,
            "team_id": HOME if player < 6 else AWAY,
            "player_id": player,
            "player_name": NAMES[player],
        }
        for player in range(1, 11)
    ]
    return pd.DataFrame(rows)


def ev(**kwargs):
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
        "playerNameI": kwargs.get("short", ""),
        "actionType": kwargs.get("action", ""),
        "subType": kwargs.get("sub", ""),
        "shotValue": kwargs.get("value", 0),
    }


def opening():
    return [
        ev(n=1, action="period", sub="start", description="Start of 1st Period"),
        ev(
            n=2,
            action="Jump Ball",
            team=HOME,
            player=1,
            name="Alpha",
            description="Jump Ball Alpha vs. Foxtrot: Tip to Alpha",
        ),
    ]


def finish(number):
    return ev(n=number, action="period", sub="end", clock="PT00M00.00S")


def table(rows):
    return possession_table(pd.DataFrame(rows), starters())


def test_shot_zone_splits_rim_mid_corner_and_above_the_break():
    assert shot_zone(2, 4, 0, 0) == "rim"
    assert shot_zone(2, 4.1, 0, 0) == "mid"
    assert shot_zone(2, 0, 0, 30) == "rim"
    assert shot_zone(2, 0, 0, 150) == "mid"
    assert shot_zone(3, 23, -220, 92) == "corner3"
    assert shot_zone(3, 23, 40, 93) == "atb3"


def test_shot_zone_rejects_a_three_without_the_corner_coordinate():
    with pytest.raises(ValueError, match="yLegacy"):
        shot_zone(3, 25, 0, None)


def test_garbage_time_follows_the_lead_and_the_clock():
    # A 20-point lead at the opening tip is still a live game.
    assert garbage_time(1, 720, 20) is False
    # Two minutes left in the fourth, the same lead is not competitive.
    assert garbage_time(4, 120, 20) is True
    assert garbage_time(4, 120, -20) is True
    assert garbage_time(4, 120, 11) is False
    assert garbage_time(4, 120, 12) is True
    assert garbage_time(5, 120, 0) is False


def test_made_field_goal_records_both_lineups_and_the_next_margin():
    rows = [
        *opening(),
        ev(
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
            description="Alpha 2' Layup (2 PTS)",
        ),
        ev(
            n=4,
            action="Turnover",
            sub="Bad Pass",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT11M20.00S",
            description="Foxtrot Bad Pass Turnover (P1.T1)",
        ),
        finish(5),
    ]

    result = table(rows)

    assert list(result.terminal) == ["make", "turnover"]
    first, second = result.iloc[0], result.iloc[1]
    assert list(first[["off_1", "off_2", "off_3", "off_4", "off_5"]]) == [1, 2, 3, 4, 5]
    assert list(first[["def_1", "def_2", "def_3", "def_4", "def_5"]]) == [6, 7, 8, 9, 10]
    assert first.margin == 0
    assert not first.garbage
    assert first.seconds_remaining == pytest.approx(720)
    assert first.fga == 1 and first.fgm == 1 and first.rim_fga == 1
    assert first.shot_player_ids == (1,)
    assert first.shot_zones == ("rim",)
    assert first.shot_made == (True,)
    assert second.offense_team_id == AWAY
    assert second.margin == -2
    assert second.seconds_remaining == pytest.approx(700)
    assert second.tov == 1 and second.tov_player_id == 6
    assert second.fga == 0


def test_offensive_rebound_keeps_the_same_possession():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Missed Shot",
            sub="Jump Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M00.00S",
            value=2,
            dist=15,
            y=150,
            description="MISS Alpha 15' Jump Shot",
        ),
        ev(
            n=4,
            action="Rebound",
            sub="Unknown",
            team=HOME,
            player=2,
            name="Bravo",
            clock="PT10M58.00S",
            description="Bravo REBOUND (Off:1 Def:0)",
        ),
        ev(
            n=5,
            action="Made Shot",
            sub="Layup Shot",
            team=HOME,
            player=2,
            name="Bravo",
            clock="PT10M50.00S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Bravo 2' Layup (2 PTS)",
        ),
        finish(6),
    ]

    result = table(rows)

    assert len(result) == 1
    row = result.iloc[0]
    assert row.oreb == 1
    assert row.fga == 2 and row.fgm == 1
    assert row.mid_fga == 1 and row.mid_fgm == 0
    assert row.rim_fga == 1 and row.rim_fgm == 1
    assert row.shot_player_ids == (1, 2)
    assert row.shot_zones == ("mid", "rim")
    assert row.shot_made == (False, True)
    assert row.terminal == "make"
    assert row.possession_id == 1


def test_defensive_rebound_and_team_rebound_change_possession():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Missed Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M00.00S",
            value=2,
            dist=15,
            y=150,
            description="MISS Alpha 15' Jump Shot",
        ),
        ev(
            n=4,
            action="Rebound",
            sub="Unknown",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT10M58.00S",
            description="Foxtrot REBOUND (Off:0 Def:1)",
        ),
        ev(
            n=5,
            action="Missed Shot",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT10M40.00S",
            value=3,
            dist=24,
            x=-220,
            y=20,
            description="MISS Foxtrot 24' 3PT",
        ),
        ev(
            n=6,
            action="Rebound",
            sub="Unknown",
            team=0,
            player=HOME,
            name="",
            clock="PT10M38.00S",
            description="HOM Rebound",
        ),
        ev(
            n=7,
            action="Turnover",
            sub="Traveling",
            team=HOME,
            player=3,
            name="Charlie",
            clock="PT10M20.00S",
            description="Charlie Traveling Turnover",
        ),
        finish(8),
    ]

    result = table(rows)

    assert list(result.terminal) == ["miss", "miss", "turnover"]
    assert result.iloc[0].fga == 1 and result.iloc[0].fgm == 0
    assert result.iloc[0].oreb == 0
    assert result.iloc[1].offense_team_id == AWAY
    assert result.iloc[1].corner3_fga == 1 and result.iloc[1].corner3_fgm == 0
    assert result.iloc[2].offense_team_id == HOME
    assert result.iloc[2].tov_player_id == 3


def test_heave_is_not_a_field_goal_attempt():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Turnover",
            sub="Bad Pass",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M00.00S",
            description="Alpha Bad Pass Turnover",
        ),
        ev(
            n=4,
            action="Heave",
            sub="Team Field Goal Attempt",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT00M00.30S",
            value=0,
            description="Foxtrot heave",
        ),
        finish(5),
    ]

    result = table(rows)

    assert len(result) == 1
    assert result.iloc[0].terminal == "turnover"
    assert result.iloc[0].fga == 0


def test_shooting_foul_trip_is_not_a_field_goal():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Foul",
            sub="Shooting",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT10M00.00S",
            description="Foxtrot S.FOUL (P1.T1)",
        ),
        ev(
            n=4,
            action="Free Throw",
            sub="Free Throw 1 of 2",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M00.00S",
            description="MISS Alpha Free Throw 1 of 2",
        ),
        ev(
            n=5,
            action="Rebound",
            sub="Normal Rebound",
            team=0,
            player=HOME,
            name="",
            clock="PT10M00.00S",
            description="HOM Rebound",
        ),
        ev(
            n=6,
            action="Free Throw",
            sub="Free Throw 2 of 2",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M00.00S",
            home="1",
            away="0",
            description="Alpha Free Throw 2 of 2 (1 PTS)",
        ),
        ev(
            n=7,
            action="Turnover",
            sub="Bad Pass",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT09M40.00S",
            description="Foxtrot Bad Pass Turnover",
        ),
        finish(8),
    ]

    result = table(rows)

    assert list(result.terminal) == ["shooting_ft", "turnover"]
    trip = result.iloc[0]
    assert trip.fga == 0
    assert trip.shooting2_trips == 1 and trip.shooting3_trips == 0
    assert trip.and1_trips == 0 and trip.bonus_trips == 0
    assert trip.fta_shooting == 2 and trip.ftm_shooting == 1
    assert trip.ft_kinds == ("shooting2",)
    assert trip.ft_player_ids == (1,)
    assert trip.ft_fta == (2,) and trip.ft_ftm == (1,)
    assert trip.oreb == 0
    assert result.iloc[1].offense_team_id == AWAY
    assert result.iloc[1].margin == -1


def test_three_shot_foul_is_its_own_trip():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Foul",
            sub="Shooting",
            team=AWAY,
            player=6,
            name="Foxtrot",
            description="Foxtrot S.FOUL",
        ),
        ev(
            n=4,
            action="Free Throw",
            sub="Free Throw 1 of 3",
            team=HOME,
            player=1,
            name="Alpha",
            description="MISS Alpha Free Throw 1 of 3",
        ),
        ev(
            n=5,
            action="Free Throw",
            sub="Free Throw 2 of 3",
            team=HOME,
            player=1,
            name="Alpha",
            home="1",
            away="0",
            description="Alpha Free Throw 2 of 3 (1 PTS)",
        ),
        ev(
            n=6,
            action="Free Throw",
            sub="Free Throw 3 of 3",
            team=HOME,
            player=1,
            name="Alpha",
            home="2",
            away="0",
            description="Alpha Free Throw 3 of 3 (2 PTS)",
        ),
        finish(7),
    ]

    result = table(rows)

    assert len(result) == 1
    row = result.iloc[0]
    assert row.shooting3_trips == 1 and row.shooting2_trips == 0
    assert row.fta_shooting == 3 and row.ftm_shooting == 2
    assert row.ft_kinds == ("shooting3",)
    assert row.terminal == "shooting_ft"
    assert row.fga == 0


def test_and_one_links_the_free_throw_to_the_made_shot_zone():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Made Shot",
            sub="Layup Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT09M00.00S",
            value=2,
            dist=1,
            y=8,
            home="2",
            away="0",
            description="Alpha 1' Layup (2 PTS)",
        ),
        ev(
            n=4,
            action="Foul",
            sub="Shooting",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT09M00.00S",
            description="Foxtrot S.FOUL (P1.T1)",
        ),
        ev(
            n=5,
            action="Free Throw",
            sub="Free Throw 1 of 1",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT09M00.00S",
            home="3",
            away="0",
            description="Alpha Free Throw 1 of 1 (3 PTS)",
        ),
        finish(6),
    ]

    result = table(rows)

    assert len(result) == 1
    row = result.iloc[0]
    assert row.terminal == "make"
    assert row.fga == 1 and row.rim_fgm == 1
    assert row.and1_trips == 1 and row.and1_zone == "rim"
    assert row.shooting2_trips == 0
    assert row.fta_and1 == 1 and row.ftm_and1 == 1
    assert row.ft_kinds == ("and1",)
    assert row.ft_zones == ("rim",)


def test_bonus_free_throws_do_not_end_the_possession():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Foul",
            sub="Personal",
            team=AWAY,
            player=7,
            name="Golf",
            clock="PT08M00.00S",
            description="Golf P.FOUL (P1.T1)",
        ),
        ev(
            n=4,
            action="Free Throw",
            sub="Free Throw 1 of 2",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT08M00.00S",
            home="1",
            away="0",
            description="Alpha Free Throw 1 of 2 (1 PTS)",
        ),
        ev(
            n=5,
            action="Free Throw",
            sub="Free Throw 2 of 2",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT08M00.00S",
            home="2",
            away="0",
            description="Alpha Free Throw 2 of 2 (2 PTS)",
        ),
        ev(
            n=6,
            action="Made Shot",
            sub="Jump Shot",
            team=HOME,
            player=2,
            name="Bravo",
            clock="PT07M30.00S",
            value=2,
            dist=16,
            y=160,
            home="4",
            away="0",
            description="Bravo 16' Jump Shot (2 PTS)",
        ),
        finish(7),
    ]

    result = table(rows)

    assert len(result) == 1
    row = result.iloc[0]
    assert row.bonus_trips == 1
    assert row.fta_bonus == 2 and row.ftm_bonus == 2
    assert row.fga == 1 and row.mid_fgm == 1
    assert row.terminal == "make"
    assert row.shooting2_trips == 0


def test_substitution_before_the_shot_updates_the_lineup():
    rows = [
        *opening(),
        ev(
            n=3,
            action="substitution",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M30.00S",
            description="SUB: Kilo FOR Alpha",
        ),
        ev(
            n=4,
            action="Made Shot",
            sub="Jump Shot",
            team=HOME,
            player=11,
            name="Kilo",
            clock="PT11M00.00S",
            value=3,
            dist=26,
            x=30,
            y=250,
            home="3",
            away="0",
            description="Kilo 26' 3PT Jump Shot (3 PTS)",
        ),
        finish(5),
    ]

    result = table(rows)

    assert len(result) == 1
    assert list(result.iloc[0][["off_1", "off_2", "off_3", "off_4", "off_5"]]) == [
        2,
        3,
        4,
        5,
        11,
    ]
    assert result.iloc[0].atb3_fga == 1
    assert not result.iloc[0].lineup_changed


def test_garbage_flag_uses_the_possession_margin_and_clock():
    rows = [
        ev(
            n=1,
            action="period",
            sub="start",
            period=4,
            clock="PT02M00.00S",
            home="80",
            away="60",
        ),
        ev(
            n=2,
            action="Jump Ball",
            team=HOME,
            player=1,
            name="Alpha",
            period=4,
            clock="PT02M00.00S",
            home="80",
            away="60",
            description="Jump Ball Alpha vs. Foxtrot: Tip to Alpha",
        ),
        ev(
            n=3,
            action="Turnover",
            sub="Lost Ball",
            team=HOME,
            player=1,
            name="Alpha",
            period=4,
            clock="PT01M50.00S",
            description="Alpha Lost Ball Turnover",
        ),
        finish(4),
    ]

    result = table(rows)

    assert len(result) == 1
    assert result.iloc[0].period == 4
    assert result.iloc[0].margin == 20
    assert result.iloc[0].seconds_remaining == pytest.approx(120)
    assert bool(result.iloc[0].garbage) is True


def test_opening_lineup_is_inferred_when_starters_are_omitted():
    rows = [ev(n=1, action="period", sub="start")]
    number = 2
    for player in range(1, 11):
        rows.append(
            ev(
                n=number,
                action="Foul",
                sub="Personal",
                team=HOME if player < 6 else AWAY,
                player=player,
                name=NAMES[player],
                description=f"{NAMES[player]} P.FOUL",
            )
        )
        number += 1
    rows.extend(
        [
            ev(
                n=number,
                action="Made Shot",
                team=HOME,
                player=1,
                name="Alpha",
                value=2,
                dist=2,
                y=10,
                home="2",
                away="0",
                description="Alpha 2' Layup (2 PTS)",
            ),
            finish(number + 1),
        ]
    )

    result = possession_table(pd.DataFrame(rows))

    assert len(result) == 1
    assert list(result.iloc[0][["off_1", "off_2", "off_3", "off_4", "off_5"]]) == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert list(result.iloc[0][["def_1", "def_2", "def_3", "def_4", "def_5"]]) == [
        6,
        7,
        8,
        9,
        10,
    ]


def test_unlogged_change_of_possession_starts_a_new_row():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Foul",
            sub="Personal",
            team=AWAY,
            player=7,
            name="Golf",
            description="Golf P.FOUL (P1.PN)",
        ),
        ev(
            n=4,
            action="Free Throw",
            sub="Free Throw 1 of 2",
            team=HOME,
            player=1,
            name="Alpha",
            home="1",
            away="0",
            description="Alpha Free Throw 1 of 2 (1 PTS)",
        ),
        ev(
            n=5,
            action="Free Throw",
            sub="Free Throw 2 of 2",
            team=HOME,
            player=1,
            name="Alpha",
            home="2",
            away="0",
            description="Alpha Free Throw 2 of 2 (2 PTS)",
        ),
        ev(
            n=6,
            action="Made Shot",
            sub="Jump Shot",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT07M00.00S",
            value=2,
            dist=16,
            y=160,
            home="2",
            away="2",
            description="Foxtrot 16' Jump Shot (2 PTS)",
        ),
        finish(7),
    ]

    result = table(rows)

    assert list(result.terminal) == ["change", "make"]
    assert result.iloc[0].offense_team_id == HOME
    assert result.iloc[0].bonus_trips == 1
    assert result.iloc[0].fga == 0
    assert result.iloc[1].offense_team_id == AWAY
    assert result.iloc[1].fga == 1


def test_initialed_name_picks_the_right_teammate():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Timeout",
            team=HOME,
            player=11,
            name="Johnson",
            short="T. Johnson",
        ),
        ev(
            n=4,
            action="Timeout",
            team=HOME,
            player=12,
            name="Johnson",
            short="A. Johnson",
        ),
        ev(
            n=5,
            action="substitution",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M30.00S",
            description="SUB: T. Johnson FOR Alpha",
        ),
        ev(
            n=6,
            action="Made Shot",
            team=HOME,
            player=11,
            name="Johnson",
            short="T. Johnson",
            clock="PT11M00.00S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="T. Johnson 2' Layup (2 PTS)",
        ),
        finish(7),
    ]

    result = table(rows)

    assert list(result.iloc[0][["off_1", "off_2", "off_3", "off_4", "off_5"]]) == [
        2,
        3,
        4,
        5,
        11,
    ]


def test_a_shot_without_a_logged_rebound_changes_possession():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Missed Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M00.00S",
            value=2,
            dist=15,
            y=150,
            description="MISS Alpha 15' Jump Shot",
        ),
        ev(
            n=4,
            action="Made Shot",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT10M40.00S",
            value=2,
            dist=2,
            y=10,
            away="2",
            home="0",
            description="Foxtrot 2' Layup (2 PTS)",
        ),
        finish(5),
    ]

    result = table(rows)

    assert list(result.terminal) == ["change", "make"]
    assert result.iloc[0].offense_team_id == HOME and result.iloc[0].fga == 1
    assert result.iloc[1].offense_team_id == AWAY and result.iloc[1].rim_fgm == 1


def test_unknown_substitution_name_does_not_drop_the_game():
    rows = [
        *opening(),
        ev(
            n=3,
            action="substitution",
            team=HOME,
            player=1,
            name="Alpha",
            description="SUB: Nobody FOR Alpha",
        ),
        ev(
            n=4,
            action="Made Shot",
            team=HOME,
            player=2,
            name="Bravo",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Bravo 2' Layup (2 PTS)",
        ),
        finish(5),
    ]

    result = table(rows)

    assert len(result) == 1
    assert result.iloc[0].shot_player_ids == (2,)
    assert 1 in list(result.iloc[0][["off_1", "off_2", "off_3", "off_4", "off_5"]])


def test_a_late_action_number_does_not_reorder_the_substitution():
    # The feed numbers some substitutions after the free throws they interrupt.
    # The row order is the play order. Kilo checks in, then shoots.
    rows = [
        *opening(),
        ev(
            n=50,
            action="Substitution",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M00.00S",
            description="SUB: Kilo FOR Alpha",
        ),
        ev(
            n=10,
            action="Made Shot",
            team=HOME,
            player=11,
            name="Kilo",
            clock="PT09M00.00S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Kilo 2' Layup (2 PTS)",
        ),
        finish(60),
    ]

    result = table(rows)

    lineup = list(result.iloc[0][["off_1", "off_2", "off_3", "off_4", "off_5"]])
    assert result.iloc[0].shot_player_ids == (11,)
    assert 11 in lineup
    assert 1 not in lineup


def test_a_repeated_rebound_total_is_not_a_second_rebound():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Missed Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M00.00S",
            value=2,
            dist=15,
            y=150,
            description="MISS Alpha 15' Jump Shot",
        ),
        ev(
            n=4,
            action="Rebound",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M58.00S",
            sub="Unknown",
            description="Alpha REBOUND (Off:1 Def:0)",
        ),
        ev(
            n=5,
            action="Rebound",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M58.00S",
            sub="Unknown",
            description="Alpha REBOUND (Off:1 Def:0)",
        ),
        ev(
            n=6,
            action="Made Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M40.00S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Alpha 2' Layup (2 PTS)",
        ),
        finish(7),
    ]

    result = table(rows)

    assert len(result) == 1
    assert result.iloc[0].oreb == 1
    assert result.iloc[0].fga == 2
    assert result.iloc[0].terminal == "make"


def test_shot_clock_turnover_is_credited_to_the_team():
    rows = [
        ev(n=1, action="period", sub="start", description="Start of 1st Period"),
        ev(
            n=2,
            action="Timeout",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M30.00S",
            description="Home Timeout",
        ),
        ev(
            n=3,
            action="Turnover",
            team=0,
            player=HOME,
            clock="PT11M00.00S",
            sub="Shot Clock Turnover",
            description="HOM Turnover: Shot Clock (T#1)",
        ),
        ev(
            n=4,
            action="Made Shot",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT10M40.00S",
            value=2,
            dist=2,
            y=10,
            home="0",
            away="2",
            description="Foxtrot 2' Layup (2 PTS)",
        ),
        finish(5),
    ]

    result = table(rows)

    assert list(result.terminal) == ["turnover", "make"]
    assert result.iloc[0].offense_team_id == HOME
    assert result.iloc[0].tov == 1
    assert result.iloc[1].offense_team_id == AWAY


def test_only_the_first_five_actors_start_the_period():
    rows = [ev(n=1, action="period", sub="start")]
    number = 2
    for player in range(1, 11):
        rows.append(
            ev(
                n=number,
                action="Foul",
                sub="Personal",
                team=HOME if player < 6 else AWAY,
                player=player,
                name=NAMES[player],
                description=f"{NAMES[player]} P.FOUL",
            )
        )
        number += 1
    rows.append(
        ev(
            n=number,
            action="Foul",
            sub="Personal",
            team=HOME,
            player=11,
            name="Kilo",
            description="Kilo P.FOUL",
        )
    )
    number += 1
    rows.extend(
        [
            ev(
                n=number,
                action="Made Shot",
                team=HOME,
                player=1,
                name="Alpha",
                value=2,
                dist=2,
                y=10,
                home="2",
                away="0",
                description="Alpha 2' Layup (2 PTS)",
            ),
            finish(number + 1),
        ]
    )

    result = possession_table(pd.DataFrame(rows))

    assert list(result.iloc[0][["off_1", "off_2", "off_3", "off_4", "off_5"]]) == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_an_unlogged_check_in_replaces_the_teammate_who_leaves():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Made Shot",
            team=HOME,
            player=11,
            name="Kilo",
            clock="PT11M00.00S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Kilo 2' Layup (2 PTS)",
        ),
    ]
    # Bravo through Echo each touch the ball later, so they are still on the floor.
    # Alpha does not. Kilo took his place.
    for number, player in enumerate(range(2, 6), start=4):
        rows.append(
            ev(
                n=number,
                action="Foul",
                sub="Personal",
                team=HOME,
                player=player,
                name=NAMES[player],
                clock="PT08M00.00S",
                description=f"{NAMES[player]} P.FOUL",
            )
        )
    rows.append(finish(8))

    result = table(rows)
    lineup = list(result.iloc[0][["off_1", "off_2", "off_3", "off_4", "off_5"]])

    assert 11 in lineup
    assert 1 not in lineup
    assert result.iloc[0].shot_player_ids == (11,)


def test_a_lane_violation_keeps_the_free_throw_that_was_shot():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Foul",
            sub="Shooting",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT11M00.00S",
            description="Foxtrot S.FOUL (P1.T1)",
        ),
        ev(
            n=4,
            action="Free Throw",
            sub="Free Throw 1 of 2",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M00.00S",
            description="MISS Alpha Free Throw 1 of 2",
        ),
        ev(
            n=5,
            action="Turnover",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M00.00S",
            sub="Lane Violation",
            description="Alpha Lane Violation Turnover (P1.T1)",
        ),
        ev(
            n=6,
            action="Made Shot",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT10M40.00S",
            value=2,
            dist=2,
            y=10,
            home="0",
            away="2",
            description="Foxtrot 2' Layup (2 PTS)",
        ),
        finish(7),
    ]

    result = table(rows)

    assert list(result.terminal) == ["turnover", "make"]
    assert result.iloc[0].fta_shooting == 1
    assert result.iloc[0].ftm_shooting == 0
    assert result.iloc[0].shooting2_trips == 1


def test_a_corrected_rebound_total_is_not_a_new_rebound():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Missed Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M00.00S",
            value=2,
            dist=15,
            y=150,
            description="MISS Alpha 15' Jump Shot",
        ),
        ev(
            n=4,
            action="Rebound",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M58.00S",
            sub="Unknown",
            description="Alpha REBOUND (Off:1 Def:0)",
        ),
        ev(
            n=5,
            action="Rebound",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M58.00S",
            sub="Unknown",
            description="Alpha REBOUND (Off:0 Def:0)",
        ),
        ev(
            n=6,
            action="Made Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M40.00S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Alpha 2' Layup (2 PTS)",
        ),
        finish(7),
    ]

    result = table(rows)

    assert len(result) == 1
    assert result.iloc[0].oreb == 1
    assert result.iloc[0].terminal == "make"


def test_a_team_rebound_after_the_period_is_ignored():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Made Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT00M00.20S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Alpha 2' Layup (2 PTS)",
        ),
        finish(4),
        ev(
            n=5,
            action="Rebound",
            team=0,
            player=HOME,
            clock="PT00M00.00S",
            sub="Unknown",
            description="HOM Rebound",
        ),
    ]

    result = table(rows)

    assert list(result.terminal) == ["make"]
    assert result.iloc[0].fga == 1


def test_an_ambiguous_check_in_is_kept_and_flagged():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Made Shot",
            team=HOME,
            player=11,
            name="Kilo",
            clock="PT11M00.00S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Kilo 2' Layup (2 PTS)",
        ),
    ]
    # Alpha and Bravo never appear again, so either could have left.
    for number, player in enumerate(range(3, 6), start=4):
        rows.append(
            ev(
                n=number,
                action="Foul",
                sub="Personal",
                team=HOME,
                player=player,
                name=NAMES[player],
                clock="PT08M00.00S",
                description=f"{NAMES[player]} P.FOUL",
            )
        )
    rows.append(finish(7))

    result = table(rows)
    lineup = list(result.iloc[0][["off_1", "off_2", "off_3", "off_4", "off_5"]])

    assert 11 in lineup
    assert 1 not in lineup
    assert result.iloc[0].lineup_changed


def test_a_bench_technical_does_not_need_a_player_team():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Foul",
            team=0,
            player=HOME,
            clock="PT11M00.00S",
            sub="Bench",
            description="HOM Non-Unsportsmanlike Tech Foul - Bench",
        ),
        ev(
            n=4,
            action="Made Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT10M40.00S",
            value=2,
            dist=2,
            y=10,
            home="2",
            away="0",
            description="Alpha 2' Layup (2 PTS)",
        ),
        finish(5),
    ]

    result = table(rows)

    assert list(result.terminal) == ["make"]
    assert result.iloc[0].shot_player_ids == (1,)


def test_a_rebound_whose_totals_both_move_follows_the_rebounder_team():
    rows = [
        *opening(),
        ev(
            n=3,
            action="Missed Shot",
            team=HOME,
            player=1,
            name="Alpha",
            clock="PT11M00.00S",
            value=2,
            dist=15,
            y=150,
            description="MISS Alpha 15' Jump Shot",
        ),
        ev(
            n=4,
            action="Rebound",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT10M58.00S",
            sub="Unknown",
            description="Foxtrot REBOUND (Off:1 Def:1)",
        ),
        ev(
            n=5,
            action="Made Shot",
            team=AWAY,
            player=6,
            name="Foxtrot",
            clock="PT10M40.00S",
            value=2,
            dist=2,
            y=10,
            home="0",
            away="2",
            description="Foxtrot 2' Layup (2 PTS)",
        ),
        finish(6),
    ]

    result = table(rows)

    assert list(result.terminal) == ["miss", "make"]
    assert result.iloc[0].offense_team_id == HOME
    assert result.iloc[1].offense_team_id == AWAY


def test_missing_play_by_play_column_is_rejected():
    rows = pd.DataFrame(opening())
    broken = rows.drop(columns=["shotValue"])

    with pytest.raises(ValueError, match="shotValue"):
        possession_table(broken, starters())


def test_starters_must_name_five_players_per_team():
    with pytest.raises(ValueError, match="five"):
        possession_table(pd.DataFrame(opening()), starters().iloc[:-1])
