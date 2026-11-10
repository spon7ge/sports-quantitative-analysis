"""Usage, zone, and free-throw rates read from the possession table."""

from __future__ import annotations

import pandas as pd
import pytest
from possession_rates import (
    FTA_PRIOR_POSSESSIONS,
    USAGE_PRIOR_POSSESSIONS,
    ZONE_PRIOR_ATTEMPTS,
    ft_structure,
    possession_player_rates,
    shrunk_usage,
    usage_rate,
    zone_make_rates,
    zone_shot_mix,
)


def _row(**kwargs):
    players = kwargs.get("offense", (1, 2, 3, 4, 5))
    shots = kwargs.get("shots", ())
    trips = kwargs.get("trips", ())
    row = {
        "garbage": kwargs.get("garbage", False),
        "lineup_changed": kwargs.get("lineup_changed", False),
        "shot_player_ids": tuple(player for player, _, _ in shots),
        "shot_zones": tuple(zone for _, zone, _ in shots),
        "shot_made": tuple(made for _, _, made in shots),
        "ft_kinds": tuple(kind for kind, _, _, _, _ in trips),
        "ft_player_ids": tuple(player for _, player, _, _, _ in trips),
        "ft_fta": tuple(attempts for _, _, attempts, _, _ in trips),
        "ft_ftm": tuple(makes for _, _, _, makes, _ in trips),
        "ft_zones": tuple(zone for _, _, _, _, zone in trips),
    }
    for index, player in enumerate(players, start=1):
        row[f"off_{index}"] = player
    return row


def _table(rows):
    return pd.DataFrame(rows)


def test_usage_is_field_goals_per_possession_on_the_floor():
    frame = _table(
        [
            _row(shots=((1, "rim", True),)),
            _row(shots=((2, "mid", False),)),
            _row(offense=(2, 3, 4, 5, 6), shots=((6, "rim", True),)),
        ]
    )

    assert usage_rate(frame, 1) == pytest.approx(0.5)
    assert usage_rate(frame, 2) == pytest.approx(1 / 3)


def test_garbage_and_changed_lineups_drop_out_of_the_rate():
    frame = _table(
        [
            _row(shots=((1, "rim", True),)),
            _row(garbage=True, shots=((1, "rim", True),) * 5),
            _row(lineup_changed=True, shots=((1, "mid", False),) * 4),
        ]
    )

    assert usage_rate(frame, 1) == pytest.approx(1.0)
    assert usage_rate(frame, 1, garbage_weight=1) == pytest.approx(6 / 2)


def test_with_teammate_usage_shrinks_toward_the_unconditional_rate():
    rows = [_row(shots=((1, "rim", True),)) for _ in range(10)]
    rows.extend(_row(offense=(1, 3, 4, 5, 6)) for _ in range(10))
    frame = _table(rows)

    shrunk = shrunk_usage(frame, 1, 2, teammate_on=True, prior_possessions=400)

    assert usage_rate(frame, 1) == pytest.approx(0.5)
    assert shrunk == pytest.approx((10 + 400 * 0.5) / 410)
    assert shrunk < 0.6


def test_with_teammate_usage_ignores_the_slate_date():
    past = [_row(shots=((1, "rim", True),)) for _ in range(10)]
    for row in past:
        row["game_date"] = "2024-01-15"
    slate = _row(offense=(1, 3, 4, 5, 6))
    slate["game_date"] = "2024-02-01"
    frame = _table([*past, slate])

    kept = shrunk_usage(frame, 1, 2, teammate_on=True, before="2024-02-01")
    past_only = shrunk_usage(_table(past), 1, 2, teammate_on=True)

    assert kept == pytest.approx(past_only)
    assert kept == pytest.approx((10 + 400 * 1.0) / 410)
    with pytest.raises(ValueError, match="game_date"):
        shrunk_usage(
            _table([_row(shots=((1, "rim", True),))]),
            1,
            2,
            teammate_on=True,
            before="2024-02-01",
        )


def test_shrinkage_priors_are_strong():
    assert USAGE_PRIOR_POSSESSIONS >= 200
    assert ZONE_PRIOR_ATTEMPTS >= 50


def test_zone_make_rates_shrink_each_zone_on_its_own():
    frame = _table(
        [
            _row(shots=((1, "rim", True),) * 4 + ((1, "mid", False),) * 4),
            _row(
                offense=(7, 8, 9, 10, 11),
                shots=((7, "rim", False),) * 10 + ((7, "mid", True),) * 10,
            ),
        ]
    )

    rates = zone_make_rates(frame, 1, prior_attempts=80)

    league_rim = 4 / 14
    league_mid = 10 / 14
    assert rates["rim"] == pytest.approx((4 + 80 * league_rim) / 84)
    assert rates["mid"] == pytest.approx((0 + 80 * league_mid) / 84)
    assert rates["rim"] < 0.5 < rates["mid"]
    assert rates["corner3"] == pytest.approx(0.0)
    assert set(rates) == {"rim", "mid", "corner3", "atb3"}


def test_zone_mix_shrinks_toward_the_league_share():
    frame = _table(
        [
            _row(shots=((1, "rim", True),) * 4),
            _row(
                offense=(7, 8, 9, 10, 11),
                shots=((7, "mid", True),) * 10,
            ),
        ]
    )

    mix = zone_shot_mix(frame, 1, prior_attempts=80)

    assert mix["rim"] == pytest.approx((4 + 80 * (4 / 14)) / 84)
    assert mix["mid"] == pytest.approx((0 + 80 * (10 / 14)) / 84)
    assert mix["corner3"] == pytest.approx(0.0)
    assert mix["atb3"] == pytest.approx(0.0)
    assert sum(mix.values()) == pytest.approx(1.0)


def test_free_throw_structure_separates_trip_types():
    frame = _table(
        [
            _row(trips=(("shooting2", 1, 2, 1, None),)),
            _row(trips=(("shooting3", 1, 3, 2, None),)),
            _row(trips=(("and1", 1, 1, 1, "rim"),)),
            _row(trips=(("bonus", 1, 2, 2, None),)),
            _row(offense=(2, 3, 4, 5, 6), trips=(("and1", 2, 1, 0, "mid"),)),
        ]
    )

    structure = ft_structure(frame, 1)
    prior = FTA_PRIOR_POSSESSIONS
    zone_prior = ZONE_PRIOR_ATTEMPTS

    assert structure["shooting2_per_poss"] == pytest.approx(
        (1 + prior * 0.2) / (4 + prior)
    )
    assert structure["shooting3_per_poss"] == pytest.approx(
        (1 + prior * 0.2) / (4 + prior)
    )
    assert structure["and1_per_poss"] == pytest.approx((1 + prior * 0.4) / (4 + prior))
    assert structure["bonus_per_poss"] == pytest.approx((1 + prior * 0.2) / (4 + prior))
    assert structure["and1_zone_share"]["rim"] == pytest.approx(
        (1 + zone_prior * 0.5) / (1 + zone_prior)
    )
    assert structure["and1_zone_share"]["mid"] == pytest.approx(
        (zone_prior * 0.5) / (1 + zone_prior)
    )


def test_player_with_no_possessions_is_rejected():
    frame = _table([_row()])

    with pytest.raises(ValueError, match="offensive possessions"):
        usage_rate(frame, 99)


def test_possession_player_rates_replace_minutes_with_usage_and_zones():
    frame = _table([_row(shots=((1, "rim", True),))])

    rates = possession_player_rates(frame, [1])
    row = rates.iloc[0]

    assert row["fga_per_min"] == pytest.approx(
        (1 + USAGE_PRIOR_POSSESSIONS * 0.2) / (1 + USAGE_PRIOR_POSSESSIONS)
    )
    assert row["p_3pa"] == pytest.approx(0.0)
    assert row["p_2p"] == pytest.approx(1.0)
    assert row["rim_share"] == pytest.approx(1.0)
    assert row["atb3_share"] == pytest.approx(0.0)
