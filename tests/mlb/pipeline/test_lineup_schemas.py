from src.mlb.config import load_config
from src.mlb.schemas import BATTER_PA_COLUMNS, LINEUP_SLOT_COLUMNS, TABLE_SCHEMAS


def test_batter_pas_has_no_is_strikeout_column() -> None:
    assert "is_strikeout" not in BATTER_PA_COLUMNS
    assert "event_type" in BATTER_PA_COLUMNS
    assert "event_time_imputed" in BATTER_PA_COLUMNS
    assert "is_pitcher_in_game" in BATTER_PA_COLUMNS


def test_lineup_slots_stores_three_windows_and_overall() -> None:
    for window in ("60", "365", "prior2"):
        assert f"k_pa_vs_hand_shrunk_{window}" in LINEUP_SLOT_COLUMNS
        assert f"k_pa_overall_shrunk_{window}" in LINEUP_SLOT_COLUMNS
        assert f"pa_vs_hand_{window}" in LINEUP_SLOT_COLUMNS
        assert f"pa_all_{window}" in LINEUP_SLOT_COLUMNS
    assert "n_eff_hand" not in LINEUP_SLOT_COLUMNS
    assert "observed_before_cutoff" in LINEUP_SLOT_COLUMNS
    assert "opposing_pitcher_hand" in LINEUP_SLOT_COLUMNS
    assert TABLE_SCHEMAS["batter_pas"] is BATTER_PA_COLUMNS
    assert TABLE_SCHEMAS["lineup_slots"] is LINEUP_SLOT_COLUMNS


def test_config_loads_hand_prior() -> None:
    assert load_config().batter_hand_prior_strength == 400.0
