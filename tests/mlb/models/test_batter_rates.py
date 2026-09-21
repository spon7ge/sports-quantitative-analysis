import math
from dataclasses import replace

import pandas as pd
import pytest
from src.mlb.config import load_config
from src.mlb.models.batter_rates import (
    LeagueKPa,
    is_strikeout,
    league_platoon_odds_ratio,
    rate_version,
    shrink_batter_k_pa,
)
from src.mlb.models.shrinkage import shrink_rate


def _pas(*, right_events: list[str], left_events: list[str]) -> pd.DataFrame:
    cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
    rows = []
    for index, (hand, event_type) in enumerate(
        [("R", event) for event in right_events]
        + [("L", event) for event in left_events]
    ):
        rows.append(
            {
                "batter_id": 1,
                "pitcher_hand": hand,
                "event_type": event_type,
                "event_time_utc": cutoff - pd.Timedelta(days=index + 1),
                "event_time_imputed": 0,
            }
        )
    return pd.DataFrame(rows)


def _league() -> LeagueKPa:
    return LeagueKPa(
        overall=0.22,
        by_bats={"R": 0.22},
        by_bats_hand={("R", "R"): 0.26, ("R", "L"): 0.18},
    )


def test_is_strikeout_derived_from_event_set() -> None:
    assert is_strikeout("strikeout") == 1
    assert is_strikeout("walk") == 0
    assert is_strikeout("strikeout_double_play") == 1
    assert is_strikeout("strikeout", events=frozenset({"walk"})) == 0


def test_rate_version_changes_with_priors_not_l2() -> None:
    config = load_config()
    base = rate_version(config)
    assert base.startswith("kpa_")
    assert len(base) == 16
    bumped = replace(config, batter_hand_prior_strength=401.0)
    assert rate_version(bumped) != base
    l2 = replace(config, strikeout_l2=99.0)
    assert rate_version(l2) == base


def test_missing_bats_ratio_is_one() -> None:
    assert league_platoon_odds_ratio(bats="", league_k_pa_cell=0.28, league_k_pa_bats=0.22) == 1.0
    assert league_platoon_odds_ratio(bats="S", league_k_pa_cell=0.24, league_k_pa_bats=0.22) != 1.0


def test_switch_uses_s_cells_not_rhb_offset() -> None:
    rhb = league_platoon_odds_ratio(bats="R", league_k_pa_cell=0.26, league_k_pa_bats=0.22)
    switch = league_platoon_odds_ratio(bats="S", league_k_pa_cell=0.23, league_k_pa_bats=0.22)
    assert switch != rhb
    assert switch > 1.0


def test_zero_vs_l_returns_prior_not_nan() -> None:
    config = load_config()
    league = _league()
    out = shrink_batter_k_pa(
        _pas(
            right_events=["strikeout", "strikeout", "out", "out", "out", "out"],
            left_events=[],
        ),
        batter_id=1,
        opposing_pitcher_hand="L",
        bats="R",
        cutoff=pd.Timestamp("2026-07-01T00:00:00Z"),
        league=league,
        config=config,
    )

    overall = shrink_rate(2, 6, league.overall, config.batter_k_prior_strength)
    ratio = league_platoon_odds_ratio(
        bats="R",
        league_k_pa_cell=league.by_bats_hand[("R", "L")],
        league_k_pa_bats=league.by_bats["R"],
    )
    prior_odds = overall / (1.0 - overall) * ratio
    prior_mean = prior_odds / (1.0 + prior_odds)

    assert out["365"]["pa_vs_hand"] == 0.0
    assert math.isfinite(out["365"]["k_pa_vs_hand_shrunk"])
    assert out["365"]["k_pa_vs_hand_shrunk"] == pytest.approx(prior_mean)


def test_overall_excludes_the_split() -> None:
    config = replace(
        load_config(),
        batter_k_prior_strength=1.0,
        batter_hand_prior_strength=1.0,
    )
    league = _league()
    out = shrink_batter_k_pa(
        _pas(
            right_events=["strikeout", "strikeout", "out", "out"],
            left_events=["out", "out", "out", "out"],
        ),
        batter_id=1,
        opposing_pitcher_hand="R",
        bats="R",
        cutoff=pd.Timestamp("2026-07-01T00:00:00Z"),
        league=league,
        config=config,
    )

    rest_only = shrink_rate(0, 4, league.overall, 1.0)
    all_pa = shrink_rate(2, 8, league.overall, 1.0)
    assert out["365"]["k_pa_overall_shrunk"] == pytest.approx(rest_only)
    assert out["365"]["k_pa_overall_shrunk"] != pytest.approx(all_pa)
    assert out["365"]["pa_vs_hand"] == 4.0
    assert out["365"]["pa_all"] == 8.0


def test_unknown_hand_keeps_overall_and_sets_vs_hand_nan() -> None:
    out = shrink_batter_k_pa(
        _pas(
            right_events=["strikeout", "out"],
            left_events=["out", "out"],
        ),
        batter_id=1,
        opposing_pitcher_hand="",
        bats="R",
        cutoff=pd.Timestamp("2026-07-01T00:00:00Z"),
        league=_league(),
        config=load_config(),
    )

    assert set(out) == {"60", "365", "prior2"}
    for window in out.values():
        assert math.isfinite(window["k_pa_overall_shrunk"])
        assert math.isnan(window["k_pa_vs_hand_shrunk"])


def test_filters_decoys_and_uses_calendar_years_for_prior2() -> None:
    cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
    pas = pd.DataFrame(
        [
            # Eligible in both rolling windows, but not prior2.
            (1, "2026-06-20T00:00:00Z", 0, "strikeout"),
            # Eligible in prior2, but both are outside the rolling 365 days.
            (1, "2025-01-15T00:00:00Z", 0, "strikeout"),
            (1, "2024-06-15T00:00:00Z", 0, "out"),
            # Decoys exercise every binding eligibility filter.
            (2, "2025-02-01T00:00:00Z", 0, "strikeout"),
            (1, "2024-05-01T00:00:00Z", 1, "strikeout"),
            (1, "2026-07-01T00:00:00Z", 0, "strikeout"),
            (1, "2026-07-02T00:00:00Z", 0, "strikeout"),
        ],
        columns=[
            "batter_id",
            "event_time_utc",
            "event_time_imputed",
            "event_type",
        ],
    )
    pas["pitcher_hand"] = "R"
    config = replace(load_config(), batter_k_prior_strength=1.0)

    out = shrink_batter_k_pa(
        pas,
        batter_id=1,
        opposing_pitcher_hand="",
        bats="R",
        cutoff=cutoff,
        league=_league(),
        config=config,
    )

    recent_expected = shrink_rate(1, 1, 0.22, 1.0)
    prior2_expected = shrink_rate(1, 2, 0.22, 1.0)
    assert out["60"]["pa_all"] == 1.0
    assert out["365"]["pa_all"] == 1.0
    assert out["prior2"]["pa_all"] == 2.0
    assert out["60"]["k_pa_overall_shrunk"] == pytest.approx(recent_expected)
    assert out["365"]["k_pa_overall_shrunk"] == pytest.approx(recent_expected)
    assert out["prior2"]["k_pa_overall_shrunk"] == pytest.approx(prior2_expected)
