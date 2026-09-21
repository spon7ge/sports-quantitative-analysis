from src.mlb.pipeline.hf_tables import pregame_from_starts
from src.mlb.schemas import STRIKEOUT_FEATURE_COLUMNS


def test_gamelog_pregame_still_team_fallback(mlb_config, fixture_tables) -> None:
    starts = fixture_tables["pitcher_starts"]
    pre = pregame_from_starts(starts, mlb_config)
    assert set(pre["lineup_state"].unique()) == {"team_fallback"}
    assert pre["lineup_batter_ids_json"].eq("[]").all()


def test_strikeout_feature_columns_unchanged() -> None:
    assert "k_bf_shrunk_365" in STRIKEOUT_FEATURE_COLUMNS
    assert "k_pa_vs_hand_shrunk_365" not in STRIKEOUT_FEATURE_COLUMNS
