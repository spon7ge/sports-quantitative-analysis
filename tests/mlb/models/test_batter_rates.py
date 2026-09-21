from dataclasses import replace

from src.mlb.config import load_config
from src.mlb.models.batter_rates import is_strikeout, league_platoon_odds_ratio, rate_version


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
