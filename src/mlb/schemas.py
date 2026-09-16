"""Append-only table schemas and prediction/feature contracts."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

UTC_DTYPE = "datetime64[us, UTC]"

RAW_SNAPSHOT_COLUMNS: dict[str, str] = {
    "snapshot_id": "string",
    "source": "string",
    "request_params_json": "string",
    "fetched_at_utc": UTC_DTYPE,
    "payload_hash": "string",
    "parser_version": "string",
    "payload_path": "string",
}

GAME_VERSION_COLUMNS: dict[str, str] = {
    "game_pk": "int64",
    "scheduled_start_utc": UTC_DTYPE,
    "status": "string",
    "home_team_id": "int64",
    "away_team_id": "int64",
    "venue_id": "int64",
    "doubleheader": "int64",
    "probable_home_pitcher_id": "Int64",
    "probable_away_pitcher_id": "Int64",
    "valid_from_utc": UTC_DTYPE,
    "valid_to_utc": UTC_DTYPE,
    "snapshot_id": "string",
}

PITCH_EVENT_COLUMNS: dict[str, str] = {
    "game_pk": "int64",
    "pitch_id": "string",
    "at_bat_number": "int64",
    "pitch_number": "int64",
    "pitcher_id": "int64",
    "batter_id": "int64",
    "event_time_utc": UTC_DTYPE,
    "pitch_result": "string",
    "pa_result": "string",
    "pitcher_hand": "string",
    "batter_stand": "string",
    "release_speed": "float64",
    "pfx_x": "float64",
    "pfx_z": "float64",
    "plate_x": "float64",
    "plate_z": "float64",
    "pitch_type": "string",
    "is_swing": "int64",
    "is_whiff": "int64",
    "is_called_strike": "int64",
    "is_in_zone": "int64",
    "is_chase": "int64",
    "source_vintage": "string",
    "ingested_at_utc": UTC_DTYPE,
    "snapshot_id": "string",
}

PLATE_APPEARANCE_COLUMNS: dict[str, str] = {
    "game_pk": "int64",
    "pa_id": "string",
    "at_bat_number": "int64",
    "pitcher_id": "int64",
    "batter_id": "int64",
    "result": "string",
    "is_strikeout": "int64",
    "batting_slot": "int64",
    "pitcher_hand": "string",
    "batter_stand": "string",
    "tto_number": "int64",
    "pitches_in_pa": "int64",
    "event_time_utc": UTC_DTYPE,
    "ingested_at_utc": UTC_DTYPE,
    "snapshot_id": "string",
}

PITCHER_START_COLUMNS: dict[str, str] = {
    "pitcher_id": "int64",
    "game_pk": "int64",
    "game_date": "string",
    "season": "int64",
    "strikeouts": "int64",
    "batters_faced": "int64",
    "pitches": "int64",
    "outs": "int64",
    "role": "string",
    "is_home": "int64",
    "opponent_team_id": "int64",
    "team_id": "int64",
    "venue_id": "int64",
    "pitcher_hand": "string",
    "scheduled_start_utc": UTC_DTYPE,
    "event_time_utc": UTC_DTYPE,
    "ingested_at_utc": UTC_DTYPE,
    "doubleheader": "int64",
}

PREGAME_SNAPSHOT_COLUMNS: dict[str, str] = {
    "pregame_id": "string",
    "game_pk": "int64",
    "pitcher_id": "int64",
    "prediction_cutoff_utc": UTC_DTYPE,
    "forecast_horizon_hours": "float64",
    "scheduled_start_utc": UTC_DTYPE,
    "game_date": "string",
    "season": "int64",
    "starter_state": "string",
    "lineup_state": "string",
    "roster_state": "string",
    "is_opener": "int64",
    "is_il_return": "int64",
    "is_restricted": "int64",
    "is_home": "int64",
    "team_id": "int64",
    "opponent_team_id": "int64",
    "venue_id": "int64",
    "pitcher_hand": "string",
    "expected_rhb_share": "float64",
    "lineup_batter_ids_json": "string",
    "source_snapshot_ids_json": "string",
}

MARKET_QUOTE_COLUMNS: dict[str, str] = {
    "quote_id": "string",
    "game_pk": "int64",
    "pitcher_id": "int64",
    "sportsbook": "string",
    "jurisdiction": "string",
    "line": "float64",
    "over_price": "float64",
    "under_price": "float64",
    "price_format": "string",
    "fetched_at_utc": UTC_DTYPE,
    "quote_status": "string",
    "rule_version": "string",
    "fill_flag": "Int64",
    "rejected_flag": "Int64",
    "limit": "float64",
    "void_flag": "Int64",
    "settled_flag": "Int64",
}

ID_MAP_COLUMNS: dict[str, str] = {
    "mlb_id": "int64",
    "key_mlbam": "int64",
    "key_fangraphs": "Int64",
    "key_bbref": "string",
    "key_retro": "string",
    "name": "string",
    "bats": "string",
    "throws": "string",
}

PLATE_DISCIPLINE_METRICS = (
    "csw",
    "whiff",
    "chase",
    "zone",
    "swing",
    "called_strike",
)
PLATE_WINDOWS = (300, 750, 365)
WORKLOAD_WINDOWS = (3, 5, 10)

FEATURE_VALUE_COLUMNS: tuple[str, ...] = (
    "k_bf_shrunk_60",
    "k_bf_shrunk_365",
    "k_bf_shrunk_prior2",
    "n_eff_k_bf_60",
    "n_eff_k_bf_365",
    "n_eff_k_bf_prior2",
    *[
        f"{metric}_{window}"
        for window in PLATE_WINDOWS
        for metric in PLATE_DISCIPLINE_METRICS
    ],
    *[
        f"{stat}_per_start_{window}"
        for window in WORKLOAD_WINDOWS
        for stat in ("bf", "pitches", "outs")
    ],
    "pitches_last_start",
    "rest_days",
    "expected_bf_oof",
    "bf_sd_oof",
    "expected_pitches_oof",
    "expected_outs_oof",
    "p_early_exit_oof",
    "opp_k_rate_vs_hand_shrunk",
    "n_eff_opp_k",
    "lineup_k_rate_shrunk",
    "lineup_state_code",
    "pitcher_throws_L",
    "expected_rhb_share",
    "fb_velo_300",
    "fb_velo_365",
    "fb_velo_delta",
    "ff_share_300",
    "bb_share_300",
    "os_share_300",
    "ff_share_delta",
    "bb_share_delta",
    "os_share_delta",
    "is_opener",
    "is_il_return",
    "is_restricted",
    "is_home",
    "venue_id",
    "season",
    "rules_era",
    "starter_state_code",
)

MISSINGNESS_FLAGS: tuple[str, ...] = (
    "missing_pitcher_k",
    "missing_plate_discipline",
    "missing_workload",
    "missing_opponent",
    "missing_lineup",
    "missing_stuff",
    "missing_role",
)

FEATURE_KEY_COLUMNS: dict[str, str] = {
    "pitcher_id": "int64",
    "game_pk": "int64",
    "prediction_cutoff_utc": UTC_DTYPE,
    "feature_set_version": "string",
    "pregame_id": "string",
}

FEATURE_ROW_COLUMNS: dict[str, str] = {
    **FEATURE_KEY_COLUMNS,
    **{name: "float64" for name in FEATURE_VALUE_COLUMNS if name not in {
        "lineup_state_code",
        "rules_era",
        "starter_state_code",
        "venue_id",
        "season",
    }},
    "lineup_state_code": "int64",
    "rules_era": "string",
    "starter_state_code": "int64",
    "venue_id": "int64",
    "season": "int64",
    **{name: "int64" for name in MISSINGNESS_FLAGS},
    "max_input_event_time_utc": UTC_DTYPE,
    "max_source_ingestion_time_utc": UTC_DTYPE,
}

PMF_COLUMNS: tuple[str, ...] = tuple(f"pmf_{k:02d}" for k in range(16)) + (
    "pmf_tail",
)

PREDICTION_COLUMNS: dict[str, str] = {
    "pitcher_id": "int64",
    "game_pk": "int64",
    "prediction_cutoff_utc": UTC_DTYPE,
    "forecast_horizon_hours": "float64",
    "expected_bf": "float64",
    "expected_pitches": "float64",
    "expected_outs": "float64",
    "expected_innings": "float64",
    "expected_k": "float64",
    "variance_k": "float64",
    "pi_lower": "float64",
    "pi_upper": "float64",
    **{name: "float64" for name in PMF_COLUMNS},
    "p_over_4_5": "float64",
    "p_under_4_5": "float64",
    "p_over_5_5": "float64",
    "p_under_5_5": "float64",
    "p_over_6_5": "float64",
    "p_under_6_5": "float64",
    "model_version": "string",
    "feature_version": "string",
    "workload_model_version": "string",
    "source_snapshot_ids_json": "string",
    "missing_data_flags_json": "string",
    "restriction_flags_json": "string",
    "market_p_over": "float64",
    "market_p_under": "float64",
    "market_disagreement": "float64",
}

TABLE_SCHEMAS: dict[str, dict[str, str]] = {
    "raw_snapshots": RAW_SNAPSHOT_COLUMNS,
    "game_versions": GAME_VERSION_COLUMNS,
    "pitch_events": PITCH_EVENT_COLUMNS,
    "plate_appearances": PLATE_APPEARANCE_COLUMNS,
    "pitcher_starts": PITCHER_START_COLUMNS,
    "pregame_snapshots": PREGAME_SNAPSHOT_COLUMNS,
    "feature_rows": FEATURE_ROW_COLUMNS,
    "market_quotes": MARKET_QUOTE_COLUMNS,
    "id_map": ID_MAP_COLUMNS,
    "predictions": PREDICTION_COLUMNS,
}

STARTER_STATE_CODES = {"unknown": 0, "probable": 1, "announced": 2, "scratched": 3}
LINEUP_STATE_CODES = {"team_fallback": 0, "expected": 1, "announced": 2}

FASTBALL_TYPES = frozenset({"FF", "SI", "FC", "FA"})
BREAKING_TYPES = frozenset({"SL", "ST", "SV", "KN", "CS"})
OFFSPEED_TYPES = frozenset({"CH", "FS", "CU", "KC", "EP"})
PITCH_FAMILY = {
    **{code: "ff" for code in FASTBALL_TYPES},
    **{code: "bb" for code in BREAKING_TYPES},
    **{code: "os" for code in OFFSPEED_TYPES},
}

WORKLOAD_FEATURE_COLUMNS: tuple[str, ...] = (
    "rest_days",
    "bf_per_start_5",
    "pitches_per_start_5",
    "outs_per_start_5",
    "is_opener",
    "is_restricted",
    "is_il_return",
    "is_home",
)

STRIKEOUT_FEATURE_COLUMNS: tuple[str, ...] = (
    "expected_bf_oof",
    "bf_sd_oof",
    "p_early_exit_oof",
    "k_bf_shrunk_365",
    "k_bf_shrunk_60",
    "opp_k_rate_vs_hand_shrunk",
    "lineup_k_rate_shrunk",
    "csw_750",
    "whiff_750",
    "fb_velo_delta",
    "ff_share_delta",
    "rest_days",
    "is_home",
    "is_opener",
    "is_restricted",
    "pitcher_throws_L",
)


def empty_frame(schema: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype=dtype) for column, dtype in schema.items()}
    )


def coerce_frame(frame: pd.DataFrame, schema: Mapping[str, str]) -> pd.DataFrame:
    ordered = []
    for column, dtype in schema.items():
        if column not in frame:
            series = pd.Series(pd.NA, index=frame.index, dtype=dtype)
        elif dtype == UTC_DTYPE:
            series = pd.to_datetime(frame[column], utc=True).astype(UTC_DTYPE)
        else:
            series = frame[column].astype(dtype)
        ordered.append(series.rename(column))
    return pd.concat(ordered, axis=1)


def validate_frame(
    frame: pd.DataFrame,
    schema: Mapping[str, str],
    *,
    name: str,
) -> None:
    missing = [column for column in schema if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def rules_era(season: int) -> str:
    if season <= 2019:
        return "pre2020"
    if season == 2020:
        return "2020"
    if season <= 2022:
        return "2021_22"
    return "pitch_clock"
