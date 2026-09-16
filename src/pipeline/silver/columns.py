"""Column normalization and cleanup for silver datasets."""

from __future__ import annotations

import re

import pandas as pd

TRACKING_RENAME = {
    "position": "start_position",
    "speed": "spd",
    "distance": "dist",
    "rebound_chances_offensive": "orbc",
    "rebound_chances_defensive": "drbc",
    "rebound_chances_total": "rbc",
    "touches": "tchs",
    "secondary_assists": "sast",
    "free_throw_assists": "ftast",
    "passes": "pass",
    "contested_field_goals_made": "cfgm",
    "contested_field_goals_attempted": "cfga",
    "contested_field_goal_percentage": "cfg_pct",
    "uncontested_field_goals_made": "ufgm",
    "uncontested_field_goals_attempted": "ufga",
    "uncontested_field_goal_percentage": "ufg_pct",
    "defended_at_rim_field_goals_made": "dfgm",
    "defended_at_rim_field_goals_attempted": "dfga",
    "defended_at_rim_field_goal_percentage": "dfg_pct",
}

TRACKING_ID_ALIASES = {
    "person_id": "player_id",
    "personid": "player_id",
}

def snake_case(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return value.strip("_").lower()

def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()

    result = frame.copy()
    result.columns = [snake_case(str(column)) for column in result.columns]

    if "fetched_at" in result:
        result = result.drop(columns="fetched_at")

    return result

def prepare_tracking(frame: pd.DataFrame) -> pd.DataFrame:
    result = normalize_columns(frame)
    result = result.rename(
        columns={
            source: target
            for source, target in TRACKING_ID_ALIASES.items()
            if source in result and target not in result
        }
    )

    rename = {
        source: target
        for source, target in TRACKING_RENAME.items()
        if source in result
    }

    return result.rename(columns=rename)

MINUTE_SOURCE_COLUMNS = ("min", "minutes")


def parse_minutes(series: pd.Series) -> pd.Series:
    """Convert numeric or MM:SS values to floating-point minutes."""
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string")
    has_clock_format = text.str.contains(":", regex=False, na=False)
    if not has_clock_format.any():
        return numeric

    clock = text.str.extract(r"^(\d+):(\d{1,2})$")
    parsed_clock = (
        pd.to_numeric(clock[0], errors="coerce")
        + pd.to_numeric(clock[1], errors="coerce") / 60
    )
    return numeric.where(~has_clock_format, parsed_clock)


def assign_playing_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Set canonical ``minutes`` from the first usable silver source.

    Boxscore logs store playing time as ``min``. Tracking merges can add a
    string ``minutes`` column that is often empty. Prefer ``min`` when it
    contains values.
    """
    result = frame.copy()
    for column in MINUTE_SOURCE_COLUMNS:
        if column not in result:
            continue
        values = parse_minutes(result[column])
        if values.notna().any():
            result["minutes"] = values
            return result

    raise ValueError(
        "no usable minutes column found; expected min or minutes"
    )


def drop_silver_junk(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    removable = [
        column
        for column in result.columns
        if "_rank" in column
        or column.endswith("_adv")
        or _is_redundant_team_column(column)
    ]

    result = result.drop(columns=removable, errors="ignore")
    return result.loc[:, ~result.columns.duplicated()]

def _is_redundant_team_column(column: str) -> bool:
    prefixes = ("team_", "opp_")
    suffixes = (
        "season_year",
        "season_year_base",
        "season_year_adv",
        "team_abbreviation",
        "team_abbreviation_base",
        "team_abbreviation_adv",
        "team_name",
        "team_name_base",
        "team_name_adv",
        "game_date",
        "game_date_base",
        "game_date_adv",
        "matchup",
        "matchup_base",
        "matchup_adv",
        "wl",
        "wl_base",
        "wl_adv",
        "min",
        "min_base",
        "min_adv",
        "available_flag",
        "available_flag_base",
        "available_flag_adv",
    )

    return any(
        column == f"{prefix}{suffix}"
        for prefix in prefixes
        for suffix in suffixes
    )