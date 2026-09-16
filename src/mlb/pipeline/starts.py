"""Build pitcher-start rows from plate appearances, pitches, and game versions."""

from __future__ import annotations

import pandas as pd

from src.mlb.schemas import PITCHER_START_COLUMNS, coerce_frame, empty_frame


def build_pitcher_starts(
    plate_appearances: pd.DataFrame,
    pitch_events: pd.DataFrame,
    game_versions: pd.DataFrame,
) -> pd.DataFrame:
    if plate_appearances is None or plate_appearances.empty:
        return empty_frame(PITCHER_START_COLUMNS)

    pas = plate_appearances.copy()
    grouped = pas.groupby(["game_pk", "pitcher_id"], as_index=False).agg(
        strikeouts=("is_strikeout", "sum"),
        batters_faced=("pa_id", "size"),
        pitcher_hand=("pitcher_hand", "first"),
        event_time_utc=("event_time_utc", "min"),
        ingested_at_utc=("ingested_at_utc", "max"),
    )
    if "result" in pas.columns:
        outs = (
            pas.assign(
                _out=pas["result"].fillna("").astype(str).str.lower().ne("onbase")
            )
            .groupby(["game_pk", "pitcher_id"], as_index=False)["_out"]
            .sum()
            .rename(columns={"_out": "outs"})
        )
        grouped = grouped.merge(outs, on=["game_pk", "pitcher_id"], how="left")
    else:
        grouped["outs"] = grouped["strikeouts"]
    grouped["outs"] = grouped["outs"].fillna(0).clip(upper=27).astype("int64")

    if pitch_events is not None and not pitch_events.empty:
        pitch_counts = (
            pitch_events.groupby(["game_pk", "pitcher_id"], as_index=False)
            .size()
            .rename(columns={"size": "pitches"})
        )
        grouped = grouped.merge(pitch_counts, on=["game_pk", "pitcher_id"], how="left")
    else:
        grouped["pitches"] = 0
    grouped["pitches"] = grouped["pitches"].fillna(0).astype("int64")

    versions = _latest_game_versions(game_versions)
    if versions.empty:
        grouped["scheduled_start_utc"] = grouped["event_time_utc"]
        grouped["game_date"] = pd.to_datetime(
            grouped["scheduled_start_utc"], utc=True
        ).dt.strftime("%Y-%m-%d")
        grouped["season"] = pd.to_datetime(
            grouped["scheduled_start_utc"], utc=True
        ).dt.year
        grouped["role"] = "starter"
        grouped["is_home"] = 0
        grouped["opponent_team_id"] = 0
        grouped["team_id"] = 0
        grouped["venue_id"] = 0
        grouped["doubleheader"] = 0
        return coerce_frame(grouped, PITCHER_START_COLUMNS)

    merged = grouped.merge(versions, on="game_pk", how="left")
    home_id = merged["probable_home_pitcher_id"]
    away_id = merged["probable_away_pitcher_id"]
    pitcher = merged["pitcher_id"]
    is_home = home_id.notna() & (home_id.astype("Int64") == pitcher.astype("Int64"))
    is_away = away_id.notna() & (away_id.astype("Int64") == pitcher.astype("Int64"))
    merged["is_home"] = is_home.astype("int64")
    merged["team_id"] = merged["home_team_id"].where(is_home, merged["away_team_id"])
    merged["opponent_team_id"] = merged["away_team_id"].where(
        is_home, merged["home_team_id"]
    )
    merged["role"] = "starter"
    merged.loc[~(is_home | is_away), "role"] = "reliever"
    scheduled = pd.to_datetime(merged["scheduled_start_utc"], utc=True)
    merged["scheduled_start_utc"] = scheduled.fillna(
        pd.to_datetime(merged["event_time_utc"], utc=True)
    )
    merged["game_date"] = merged["scheduled_start_utc"].dt.strftime("%Y-%m-%d")
    merged["season"] = merged["scheduled_start_utc"].dt.year.astype("int64")
    merged["venue_id"] = merged["venue_id"].fillna(0)
    merged["doubleheader"] = merged["doubleheader"].fillna(0)
    merged["event_time_utc"] = merged["scheduled_start_utc"]
    return coerce_frame(merged, PITCHER_START_COLUMNS)


def _latest_game_versions(game_versions: pd.DataFrame) -> pd.DataFrame:
    if game_versions is None or game_versions.empty:
        return pd.DataFrame()
    versions = game_versions.copy()
    versions["valid_from_utc"] = pd.to_datetime(versions["valid_from_utc"], utc=True)
    versions = versions.sort_values(["game_pk", "valid_from_utc"])
    keep = [
        "game_pk",
        "scheduled_start_utc",
        "home_team_id",
        "away_team_id",
        "venue_id",
        "doubleheader",
        "probable_home_pitcher_id",
        "probable_away_pitcher_id",
    ]
    keep = [col for col in keep if col in versions.columns]
    return versions.groupby("game_pk", as_index=False).tail(1)[keep]
