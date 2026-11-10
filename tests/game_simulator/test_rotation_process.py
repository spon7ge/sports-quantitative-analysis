"""Minutes come from within-game hazards, not a copied rotation template.

Foul trouble, blowout exits, stint length, and rare exits share one clock
with scoring. Closing fives read the late margin path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from game_rotations import (
    expand_stints,
    fit_rotation_process,
    simulate_rotations,
)
from game_rotations.process import (
    BlowoutRule,
    ClosingRule,
    ExitMixture,
    FoulProcess,
    RotationProcess,
    StintHazard,
    ahead_of_foul_pace,
    blowout_threshold,
    late_closeness,
)
from game_rotations.segments import invalid_segment_count

_STARTERS = ["1", "2", "3", "4", "5"]
_BENCH = ["6", "7", "8", "9", "10"]


def _quiet() -> RotationProcess:
    return RotationProcess(
        stint=StintHazard(intercept=-20.0, stint_minutes=0.0, fatigue=0.0, timeout=0.0),
        foul=FoulProcess(intercept=-20.0, ahead=0.0),
        blowout=BlowoutRule(
            cushion=80.0,
            per_minute=0.0,
            exit_steepness=1.0,
            return_probability=0.0,
        ),
        exits=ExitMixture(0.0, 0.0, 0.2, 0.5),
        closing=ClosingRule(late_minutes=0.0, close_margin=8.0, closeness_share=0.5),
    )


def _roster(
    foul_rate=0.0,
    points_per_minute=0.0,
    *,
    injury_rate=None,
    ejection_rate=None,
) -> pd.DataFrame:
    ids = _STARTERS + _BENCH
    frame = pd.DataFrame(
        {
            "player_id": ids,
            "is_starter": [player in _STARTERS for player in ids],
            "foul_rate": foul_rate,
            "points_per_minute": points_per_minute,
        }
    )
    if injury_rate is not None:
        frame["injury_rate"] = injury_rate
    if ejection_rate is not None:
        frame["ejection_rate"] = ejection_rate
    return frame


def _path(margin, n_ticks=48) -> np.ndarray:
    return np.full(n_ticks, margin, dtype=float)


def test_foul_pace_and_blowout_threshold_are_explicit():
    assert ahead_of_foul_pace(3, 8) == pytest.approx(3 - 6 * 8 / 48)
    rule = BlowoutRule(10.0, 0.5, 2.0, 0.4)
    assert blowout_threshold(20.0, rule) == pytest.approx(20.0)


def test_late_closeness_uses_the_window_not_the_final_margin():
    close = np.full(48, 4.0)
    # Same ending, but the previous five late minutes were a blowout.
    collapsed = np.full(48, 4.0)
    collapsed[-6:-1] = 22.0
    rule = ClosingRule(6.0, 8.0, 0.5)

    assert late_closeness(close, 1.0, rule) == pytest.approx(1.0)
    assert late_closeness(collapsed, 1.0, rule) == pytest.approx(1 / 6)
    assert close[-1] == collapsed[-1]


def test_a_quiet_rotation_leaves_the_starters_in_and_covers_the_game():
    draw = simulate_rotations(
        _quiet(),
        _roster(points_per_minute=1.0),
        _path(2.0),
        np.random.default_rng(0),
    )
    role = _roster()

    assert draw.minutes.shape == (1, 10)
    assert draw.minutes[0, :5] == pytest.approx(48.0)
    assert draw.minutes[0, 5:] == pytest.approx(0.0)
    assert draw.points[0, 5:] == pytest.approx(0.0)
    assert draw.minutes.sum() == pytest.approx(5 * 48)
    assert {segment["period"] for segment in draw.segments[0]} == {1, 2, 3, 4}
    assert set(draw.segments[0][-1]["players"]) == set(_STARTERS)
    assert invalid_segment_count(
        draw.segments[0],
        role,
        game_minutes=48.0,
        n_ot=0.0,
    ) == 0


def test_overtime_extends_the_same_five_man_clock():
    draw = simulate_rotations(
        _quiet(),
        _roster(),
        _path(1.0, 53),
        np.random.default_rng(1),
    )

    assert draw.minutes.sum() == pytest.approx(5 * 53)
    assert draw.segments[0][-1]["period"] == 5
    assert invalid_segment_count(
        draw.segments[0],
        _roster(),
        game_minutes=53.0,
        n_ot=1.0,
    ) == 0


def test_foul_trouble_cuts_minutes_and_points_together():
    process = _quiet()
    process = RotationProcess(
        stint=process.stint,
        foul=FoulProcess(intercept=-1.5, ahead=2.5),
        blowout=process.blowout,
        exits=process.exits,
        closing=process.closing,
    )
    rates = np.zeros(10)
    rates[0] = 0.5
    roster = _roster(foul_rate=rates, points_per_minute=4.0)
    draw = simulate_rotations(
        process,
        roster,
        _path(0.0),
        np.random.default_rng(2),
        n=400,
    )
    troubled = draw.minutes[:, 0]
    calm = draw.minutes[:, 1]
    correlation = np.corrcoef(troubled, draw.points[:, 0])[0, 1]

    assert troubled.mean() < 30
    assert calm.mean() > 45
    assert draw.points[:, 0].mean() < draw.points[:, 1].mean()
    assert correlation > 0.75
    assert np.all(draw.fouls[:, 0] >= draw.fouls[:, 1])


def test_a_short_bench_still_fields_five_in_a_blowout():
    base = _quiet()
    process = RotationProcess(
        base.stint,
        base.foul,
        BlowoutRule(cushion=0.0, per_minute=0.0, exit_steepness=20.0, return_probability=0.0),
        base.exits,
        base.closing,
    )
    roster = _roster().iloc[:7].reset_index(drop=True)

    draw = simulate_rotations(process, roster, _path(40.0), np.random.default_rng(13))

    assert draw.minutes.shape == (1, 7)
    assert draw.minutes.sum() == pytest.approx(5 * 48)
    assert draw.minutes[0, 5:].sum() > 0
    assert np.any(draw.minutes[0, :5] < 48)
    assert invalid_segment_count(
        draw.segments[0],
        roster,
        game_minutes=48.0,
        n_ot=0.0,
    ) == 0


def test_blowout_benches_starters_and_a_return_brings_them_back():
    stayed = _path(30.0)
    came_back = np.full(48, 30.0)
    came_back[30:] = 0.0
    stay_rule = BlowoutRule(10.0, 0.5, 12.0, 0.0)
    return_rule = BlowoutRule(10.0, 0.5, 12.0, 0.85)
    base = _quiet()
    roster = _roster()

    blowout = simulate_rotations(
        RotationProcess(base.stint, base.foul, stay_rule, base.exits, base.closing),
        roster,
        stayed,
        np.random.default_rng(3),
        n=40,
    )
    returned = simulate_rotations(
        RotationProcess(base.stint, base.foul, return_rule, base.exits, base.closing),
        roster,
        came_back,
        np.random.default_rng(4),
        n=40,
    )
    back_in = [
        set(_STARTERS) <= set(schedule[-1]["players"]) for schedule in returned.segments
    ]

    assert blowout.minutes[:, :5].mean() < 20
    assert returned.minutes[:, :5].mean() > blowout.minutes[:, :5].mean() + 10
    assert np.mean(back_in) > 0.8


def test_closing_lineup_follows_late_closeness_at_the_same_final_margin():
    close = _path(6.0)
    garbage = np.full(48, 28.0)
    garbage[-1] = 6.0
    rule = ClosingRule(late_minutes=6.0, close_margin=8.0, closeness_share=0.5)
    base = _quiet()
    process = RotationProcess(
        base.stint,
        base.foul,
        BlowoutRule(10.0, 0.5, 20.0, 0.0),
        base.exits,
        rule,
    )
    roster = _roster()
    reopened = np.full(48, 28.0)
    reopened[-6:] = 4.0
    tight = simulate_rotations(process, roster, close, np.random.default_rng(5))
    blown = simulate_rotations(process, roster, garbage, np.random.default_rng(6))
    brought_back = simulate_rotations(
        process,
        roster,
        reopened,
        np.random.default_rng(13),
    )

    assert close[-1] == garbage[-1]
    assert tight.late_closeness > rule.closeness_share
    assert blown.late_closeness < rule.closeness_share
    assert set(tight.segments[0][-1]["players"]) == set(_STARTERS)
    assert set(blown.segments[0][-1]["players"]) == set(_BENCH)
    # Return probability is zero. Closers come back only because the last
    # six minutes are close, which the final margin of the garbage path is not.
    assert set(brought_back.segments[0][-1]["players"]) == set(_STARTERS)


def test_timeouts_and_long_stints_bring_the_bench_in():
    base = _quiet()
    resting = RotationProcess(
        StintHazard(-8.0, 0.0, 0.0, 0.0),
        base.foul,
        base.blowout,
        base.exits,
        base.closing,
    )
    timeouts = RotationProcess(
        StintHazard(-8.0, 0.0, 0.0, 12.0),
        base.foul,
        base.blowout,
        base.exits,
        base.closing,
    )
    roster = _roster()
    path = _path(1.0)
    rng = np.random.default_rng(7)
    quiet = simulate_rotations(resting, roster, path, rng, n=30)
    active = simulate_rotations(
        timeouts,
        roster,
        path,
        np.random.default_rng(8),
        n=30,
        timeouts=np.arange(6.0, 48.0, 6.0),
    )

    assert quiet.minutes[:, 5:].sum(axis=1).mean() < 2
    assert active.minutes[:, 5:].sum(axis=1).mean() > 15


def test_injury_and_ejection_remove_a_player_for_good():
    base = _quiet()
    exits = ExitMixture(0.0, 0.0, 10 / 48, 10 / 48)
    process = RotationProcess(base.stint, base.foul, base.blowout, exits, base.closing)
    injured = np.zeros(10)
    injured[0] = 1.0
    ejected = np.zeros(10)
    ejected[2] = 1.0
    roster = _roster(injury_rate=injured, ejection_rate=ejected)
    draw = simulate_rotations(process, roster, _path(0.0), np.random.default_rng(9))
    final = set(draw.segments[0][-1]["players"])

    assert draw.minutes[0, 0] == pytest.approx(10.0)
    assert draw.minutes[0, 2] == pytest.approx(10.0)
    assert draw.exit_reason[0][0] == "injury"
    assert draw.exit_reason[0][2] == "ejection"
    assert "1" not in final
    assert "3" not in final
    assert draw.minutes[0, 1] == pytest.approx(48.0)


def test_exit_mixture_adds_a_short_minute_component():
    base = _quiet()
    mixed = RotationProcess(
        base.stint,
        base.foul,
        base.blowout,
        ExitMixture(0.3, 0.0, 0.1, 0.25),
        base.closing,
    )
    # Twenty names so a 30% exit rate still leaves five players who can stand.
    deep = pd.DataFrame(
        {
            "player_id": [str(i) for i in range(1, 21)],
            "is_starter": [i < 5 for i in range(20)],
        }
    )
    clean = simulate_rotations(
        base,
        deep,
        _path(0.0),
        np.random.default_rng(10),
        n=500,
    )
    drawn = simulate_rotations(
        mixed,
        deep,
        _path(0.0),
        np.random.default_rng(11),
        n=500,
    )
    starter_minutes = drawn.minutes[:, 0]

    assert np.quantile(clean.minutes[:, 0], 0.1) == pytest.approx(48.0)
    assert np.quantile(starter_minutes, 0.1) < 20
    assert np.mean(starter_minutes < 20) > 0.15
    assert starter_minutes.mean() < clean.minutes[:, 0].mean() - 5


def test_stint_ticks_are_a_larger_sample_than_team_games():
    rows = []
    for game in ("g1", "g2"):
        for index, start in enumerate(range(0, 48, 6)):
            rows.append(
                {
                    "game_id": game,
                    "team_id": "10",
                    "player_id": str((index % 5) + 1),
                    "is_starter": 1,
                    "start_elapsed": float(start),
                    "end_elapsed": float(start + 6),
                    "game_minutes": 48.0,
                    "margin": 4.0,
                    "fouls": 1.0,
                    "timeout": int(index % 2 == 0),
                    "reason": "rotation" if index % 2 == 0 else "censored",
                }
            )
    ticks = expand_stints(pd.DataFrame(rows))
    _, diagnostics = fit_rotation_process(ticks)

    assert len(ticks) == 2 * 8 * 6
    assert diagnostics["n_player_ticks"] == len(ticks)
    assert diagnostics["n_team_games"] == 2
    assert diagnostics["n_player_ticks"] > diagnostics["n_team_games"]


def test_hazards_recover_on_player_ticks():
    rng = np.random.default_rng(12)
    n = 6000
    stint = rng.uniform(0, 16, n)
    fatigue = rng.uniform(0, 30, n)
    timeout = rng.random(n) < 0.15
    stint_beta = np.array([-3.0, 0.18, 0.04, 2.2])
    stint_eta = stint_beta @ np.vstack([np.ones(n), stint, fatigue, timeout])
    rotation_sub = rng.random(n) < 1 / (1 + np.exp(-stint_eta))
    ahead = rng.uniform(-1, 4, n)
    elapsed = rng.uniform(0, 40, n)
    fouls = ahead + 6 * elapsed / 48
    foul_beta = np.array([-2.0, 1.6])
    foul_eta = foul_beta @ np.vstack([np.ones(n), ahead])
    foul_exit = rng.random(n) < 1 / (1 + np.exp(-foul_eta))
    margin = rng.uniform(-30, 30, n)
    minutes_left = rng.uniform(0, 48, n)
    cushion, per_minute, steepness = 12.0, 0.7, 2.0
    gap = np.abs(margin) - cushion - per_minute * minutes_left
    blowout_exit = rng.random(n) < 1 / (1 + np.exp(-steepness * gap))
    # Thirty ticks per player-game. Injuries and ejections are different games,
    # so the mixture is a per-game latent rather than a run of exit ticks.
    game_index = np.arange(n) // 30
    injured = game_index < 8
    ejected = (game_index >= 8) & (game_index < 11)
    benched = np.zeros(n, dtype=bool)
    benched[100:300] = True
    returned = benched & (rng.random(n) < 0.25)
    ticks = pd.DataFrame(
        {
            "game_id": [f"g{index}" for index in game_index],
            "team_id": "10",
            "player_id": "1",
            "stint_minutes": stint,
            "fatigue_minutes": fatigue,
            "timeout": timeout.astype(float),
            "rotation_sub": rotation_sub.astype(float),
            "fouls": fouls,
            "elapsed_minutes": elapsed,
            "foul_trouble_exit": foul_exit.astype(float),
            "margin": margin,
            "minutes_left": minutes_left,
            "is_starter": 1.0,
            "blowout_exit": blowout_exit.astype(float),
            "injury_exit": injured.astype(float),
            "ejection_exit": ejected.astype(float),
            "blowout_benched": benched.astype(float),
            "inside_threshold": benched.astype(float),
            "returned": returned.astype(float),
        }
    )
    process, _ = fit_rotation_process(ticks)

    assert process.stint.intercept == pytest.approx(stint_beta[0], abs=0.35)
    assert process.stint.stint_minutes == pytest.approx(stint_beta[1], abs=0.08)
    assert process.stint.fatigue == pytest.approx(stint_beta[2], abs=0.05)
    assert process.stint.timeout == pytest.approx(stint_beta[3], abs=0.45)
    assert process.stint.stint_minutes > 0
    assert process.stint.timeout > 0
    assert process.foul.intercept == pytest.approx(foul_beta[0], abs=0.35)
    assert process.foul.ahead == pytest.approx(foul_beta[1], abs=0.25)
    assert process.blowout.cushion == pytest.approx(cushion, abs=2.0)
    assert process.blowout.per_minute == pytest.approx(per_minute, abs=0.2)
    assert process.exits.injury_rate < 0.05
    assert process.exits.ejection_rate < process.exits.injury_rate
    assert process.blowout.return_probability == pytest.approx(0.25, abs=0.08)


def test_bad_rotation_inputs_are_rejected():
    process = _quiet()
    roster = _roster()
    path = _path(0.0)

    with pytest.raises(ValueError, match="rng"):
        simulate_rotations(process, roster, path, rng=object())

    with pytest.raises(ValueError, match="margin path"):
        simulate_rotations(process, roster, path[:-1], np.random.default_rng(0))

    with pytest.raises(ValueError, match="five"):
        simulate_rotations(process, roster.iloc[:4], path, np.random.default_rng(0))

    with pytest.raises(ValueError, match="player"):
        duplicate = roster.copy()
        duplicate.loc[1, "player_id"] = "1"
        simulate_rotations(process, duplicate, path, np.random.default_rng(0))

    with pytest.raises(ValueError, match="column"):
        fit_rotation_process(pd.DataFrame({"game_id": ["g"]}))
