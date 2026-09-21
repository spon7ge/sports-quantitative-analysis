"""Parse starting lineup slots from Stats API boxscores."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.models.batter_rates import (
    LeagueKPa,
    is_strikeout,
    league_eligible_pas,
    rate_version,
    shrink_batter_k_pa,
)
from src.mlb.pipeline.features import select_game_version
from src.mlb.pipeline.http import HttpFn
from src.mlb.schemas import LINEUP_SLOT_COLUMNS, coerce_frame
from src.mlb.storage import MlbStore

BOXSCORE_00_LEAD = pd.Timedelta(hours=24)
LINEUP_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


def _now_utc() -> datetime:
    return datetime.now(UTC)

STARTING_NINE_COLUMNS = (
    "game_pk",
    "team_id",
    "side",
    "slot",
    "batter_id",
    "slot_is_pitcher",
    "season",
)


def _as_text(raw_payload: str | bytes) -> str:
    if isinstance(raw_payload, bytes):
        return raw_payload.decode("utf-8-sig")
    return raw_payload


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_starting_nine(
    raw_payload: str | bytes,
    game_pk: int | None = None,
) -> pd.DataFrame:
    """Return each side's original 1-9 from player ``battingOrder`` 00 codes."""
    data = json.loads(_as_text(raw_payload))
    if not isinstance(data, dict):
        return pd.DataFrame(columns=list(STARTING_NINE_COLUMNS))

    game = (data.get("gameData") or {}).get("game") or {}
    resolved_pk = (
        game_pk if game_pk is not None else data.get("gamePk") or game.get("pk")
    )
    season = _as_int(game.get("season"))

    live = data.get("liveData") or {}
    box = live.get("boxscore") or data.get("boxscore") or data
    teams = box.get("teams") if isinstance(box, dict) else None
    if not isinstance(teams, dict):
        return pd.DataFrame(columns=list(STARTING_NINE_COLUMNS))

    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        team = teams.get(side) or {}
        team_id = (team.get("team") or {}).get("id")
        selected: dict[int, dict[str, Any]] = {}
        players = team.get("players") or {}
        if not isinstance(players, dict):
            continue

        for player_key, player in players.items():
            if not isinstance(player, dict):
                continue
            batting_order = player.get("battingOrder")
            if batting_order is None or batting_order == "":
                continue
            order_text = str(batting_order)
            if not order_text.endswith("0"):
                continue
            if len(order_text) == 3:
                slot = _as_int(order_text[0])
            elif len(order_text) == 1:
                slot = _as_int(order_text)
            else:
                continue
            if slot not in range(1, 10):
                continue

            position = ((player.get("position") or {}).get("abbreviation") or "")
            is_pitcher = int(str(position).upper() == "P")
            person = player.get("person") or {}
            batter_id = person.get("id")
            if batter_id is None:
                batter_id = str(player_key).removeprefix("ID")
            candidate = {
                "game_pk": resolved_pk,
                "team_id": team_id,
                "side": side,
                "slot": slot,
                "batter_id": _as_int(batter_id),
                "slot_is_pitcher": is_pitcher,
                "season": season,
            }
            current = selected.get(slot)
            if current is None or (
                current["slot_is_pitcher"] == 1 and candidate["slot_is_pitcher"] == 0
            ):
                selected[slot] = candidate

        rows.extend(selected[slot] for slot in sorted(selected))

    return pd.DataFrame(rows, columns=list(STARTING_NINE_COLUMNS))


def _league_k_pa(pas: pd.DataFrame) -> LeagueKPa:
    if pas.empty:
        return LeagueKPa(overall=0.22, by_bats={}, by_bats_hand={})
    strikeouts = pas["event_type"].map(is_strikeout)
    overall = float(strikeouts.mean())
    if pd.isna(overall):
        raise ValueError("cannot freeze lineup rates without eligible league PAs")

    rates = pas.assign(_is_strikeout=strikeouts)
    bats_rates = (
        rates.groupby("batter_bats", dropna=False)["_is_strikeout"]
        .mean()
        .rename("rate")
        .reset_index()
    )
    cell_rates = (
        rates.groupby(["batter_bats", "pitcher_hand"], dropna=False)[
            "_is_strikeout"
        ]
        .mean()
        .rename("rate")
        .reset_index()
    )
    return LeagueKPa(
        overall=overall,
        by_bats={
            str(row.batter_bats): float(row.rate)
            for row in bats_rates.itertuples(index=False)
        },
        by_bats_hand={
            (str(row.batter_bats), str(row.pitcher_hand)): float(row.rate)
            for row in cell_rates.itertuples(index=False)
        },
    )


def freeze_lineup_slot_rates(
    slots: pd.DataFrame,
    batter_pas: pd.DataFrame,
    people: pd.DataFrame,
    config: MlbConfig,
    *,
    league_nines: pd.DataFrame | None = None,
    cutoff: pd.Timestamp,
    opposing_pitcher_hand: str,
    vs_pitcher_id: int | None,
) -> pd.DataFrame:
    """Freeze cutoff-safe three-window batter K/PA rates onto lineup slots."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    eligibility_nines = slots if league_nines is None else league_nines
    league_pas = league_eligible_pas(batter_pas, eligibility_nines)
    event_times = pd.to_datetime(league_pas["event_time_utc"], utc=True)
    league_pas = league_pas.loc[
        (event_times < cutoff)
        & (event_times >= cutoff - pd.Timedelta(days=365))
        & (league_pas["event_time_imputed"] == 0)
    ].copy()
    league = _league_k_pa(league_pas)

    people_id_column = "mlb_id" if "mlb_id" in people else "batter_id"
    bats_by_id = (
        people.drop_duplicates(people_id_column)
        .set_index(people_id_column)["bats"]
        .to_dict()
    )
    hand = str(opposing_pitcher_hand or "")
    frozen = slots.copy()
    frozen["opposing_pitcher_hand"] = hand
    frozen["vs_pitcher_id"] = vs_pitcher_id
    frozen["rate_version"] = rate_version(config)

    for index, slot in frozen.iterrows():
        batter_id = int(slot["batter_id"])
        bats_value = bats_by_id.get(batter_id, "")
        bats = "" if pd.isna(bats_value) else str(bats_value)
        rates = shrink_batter_k_pa(
            batter_pas,
            batter_id=batter_id,
            opposing_pitcher_hand=hand,
            bats=bats,
            cutoff=cutoff,
            league=league,
            config=config,
        )
        for window, values in rates.items():
            for metric, value in values.items():
                frozen.at[index, f"{metric}_{window}"] = value

    return coerce_frame(frozen, LINEUP_SLOT_COLUMNS)


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def select_lineup_slots(
    slots: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    rate_version: str,
) -> pd.DataFrame:
    """Select the latest cutoff-safe card for each versioned lineup slot."""
    if slots is None or slots.empty:
        return slots.copy()
    selected = slots.loc[slots["rate_version"] == rate_version].copy()
    if selected.empty:
        return selected
    ingested_at = pd.to_datetime(selected["ingested_at_utc"], utc=True)
    selected = selected.loc[ingested_at < _utc_timestamp(cutoff)].copy()
    if selected.empty:
        return selected
    selected["_ingested_at_utc"] = pd.to_datetime(
        selected["ingested_at_utc"], utc=True
    )
    selected = selected.sort_values("_ingested_at_utc").drop_duplicates(
        ["game_pk", "team_id", "slot", "rate_version"], keep="last"
    )
    return selected.drop(columns="_ingested_at_utc").reset_index(drop=True)


def select_probable_pitcher(
    game_versions: pd.DataFrame,
    game_pk: int,
    cutoff: pd.Timestamp,
    *,
    batting_is_home: bool,
) -> tuple[int | None, str]:
    """Return the opposing probable pitcher from the latest valid version."""
    version = select_game_version(game_versions, game_pk, _utc_timestamp(cutoff))
    if version is None:
        return None, ""
    side = "away" if batting_is_home else "home"
    pitcher_id = version.get(f"probable_{side}_pitcher_id")
    hand = version.get(f"probable_{side}_pitcher_hand", "")
    resolved_id = None if pd.isna(pitcher_id) else int(pitcher_id)
    resolved_hand = "" if pd.isna(hand) else str(hand)
    return resolved_id, resolved_hand


def earliest_scheduled_start(
    game_versions: pd.DataFrame, game_pk: int
) -> pd.Timestamp:
    """Return the earliest known scheduled start for a game."""
    if game_versions is None or game_versions.empty:
        return pd.NaT
    starts = pd.to_datetime(
        game_versions.loc[
            game_versions["game_pk"] == game_pk, "scheduled_start_utc"
        ],
        utc=True,
    )
    return starts.min()


def lineup_identity_mismatch_rate(
    live: pd.DataFrame, official: pd.DataFrame
) -> float:
    """Return batter identity mismatch after joining each team's slots."""
    keys = ["game_pk", "team_id", "slot"]
    joined = live[keys + ["batter_id"]].merge(
        official[keys + ["batter_id"]],
        on=keys,
        how="inner",
        suffixes=("_live", "_official"),
    )
    if joined.empty:
        return float("nan")
    return float((joined["batter_id_live"] != joined["batter_id_official"]).mean())


def copy_live_ingest_clock(
    existing: pd.DataFrame, new_rows: pd.DataFrame
) -> pd.DataFrame:
    """Copy each original live card clock onto newly frozen slot rows."""
    copied = new_rows.copy()
    if existing is None or existing.empty or copied.empty:
        return copied
    keys = ["game_pk", "team_id", "slot"]
    live = existing.loc[existing["provenance"] == "live_feed"].copy()
    if live.empty:
        return copied
    live["_original_ingested_at_utc"] = pd.to_datetime(
        live["ingested_at_utc"], utc=True
    )
    clocks = (
        live.groupby(keys, as_index=False)["_original_ingested_at_utc"].min()
    )
    copied = copied.merge(clocks, on=keys, how="left")
    replace_clock = (copied["provenance"] == "live_feed") & copied[
        "_original_ingested_at_utc"
    ].notna()
    copied.loc[replace_clock, "ingested_at_utc"] = copied.loc[
        replace_clock, "_original_ingested_at_utc"
    ]
    return copied.drop(columns="_original_ingested_at_utc")


def is_dummy_dh2_start(start: Any, game1_start: Any = None) -> bool:
    """Return whether a DH2 scheduled start is unusable as an as-of timestamp."""
    start = _utc_timestamp(start)
    if pd.isna(start):
        return True
    if start.hour == 0 and start.minute == 0 and start.second == 0:
        return True
    game1 = _utc_timestamp(game1_start)
    return bool(
        not pd.isna(game1)
        and start.date() == game1.date()
        and start == game1
    )


def write_lineup_coverage(
    slots: pd.DataFrame,
    skips: pd.DataFrame,
    *,
    season: int,
) -> dict[str, int]:
    """Summarize written nines and explicit skip reasons for one season."""
    season_slots = slots.loc[slots.get("season", pd.Series(dtype="int64")) == season]
    season_skips = skips.loc[skips.get("season", pd.Series(dtype="int64")) == season]
    complete = 0
    if not season_slots.empty:
        complete = int(
            season_slots.groupby(["game_pk", "team_id"])["slot"].nunique().eq(9).sum()
        )
    skipped_games = (
        set(season_skips["game_pk"].astype(int)) if not season_skips.empty else set()
    )
    written_games = (
        set(season_slots["game_pk"].astype(int)) if not season_slots.empty else set()
    )
    reasons = season_skips.get("reason", pd.Series(dtype="string"))
    incomplete_games = set(
        season_skips.loc[reasons == "incomplete_nine", "game_pk"].astype(int)
    )
    return {
        "season": int(season),
        "n_game_pks": len(written_games | skipped_games),
        "n_boxscore_ok": len(written_games | incomplete_games),
        "n_sides_complete_nine": complete,
        "n_slots_written": int(len(season_slots)),
        "n_skip_dh2_dummy_start": int((reasons == "dh2_dummy_start").sum()),
        "n_skip_no_boxscore": int((reasons == "no_boxscore").sum()),
        "n_skip_incomplete_nine": int((reasons == "incomplete_nine").sum()),
    }


def assert_lineup_coverage(coverage: dict[str, Any], *, season: int) -> None:
    """Raise when a season has no slots or fewer than 95% complete sides."""
    if int(coverage.get("n_slots_written", 0)) == 0:
        raise ValueError(f"lineup coverage for {season} wrote zero slots")
    n_boxscore_ok = int(coverage.get("n_boxscore_ok", 0))
    denominator = 2 * n_boxscore_ok
    complete_rate = (
        float(coverage.get("n_sides_complete_nine", 0)) / denominator
        if denominator
        else 0.0
    )
    if complete_rate < 0.95:
        raise ValueError(
            f"lineup complete-nine coverage for {season} is {complete_rate:.3f}"
        )


def _local_lineup_payload(config: MlbConfig, game_pk: int) -> bytes:
    candidates = (
        config.raw_dir / "mlb_lineup_slots" / f"{game_pk}.json",
        Path(config.fixture_dir) / "raw" / f"lineups_{game_pk}.json",
        Path(config.fixture_dir) / "raw" / "lineups.json",
    )
    for path in candidates:
        if path.exists():
            return path.read_bytes()
    raise FileNotFoundError(f"No local lineup payload for game {game_pk}")


def _original_game_version(
    game_versions: pd.DataFrame, game_pk: int
) -> pd.Series | None:
    versions = game_versions.loc[game_versions["game_pk"] == game_pk]
    if versions.empty:
        return None
    valid_from = pd.to_datetime(versions["valid_from_utc"], utc=True)
    if valid_from.notna().any():
        return versions.loc[valid_from.idxmin()]
    return versions.iloc[0]


def _pitcher_hand(people: pd.DataFrame, pitcher_id: int | None) -> str:
    if pitcher_id is None or pd.isna(pitcher_id) or people.empty:
        return ""
    id_column = "mlb_id" if "mlb_id" in people else "pitcher_id"
    match = people.loc[people[id_column] == int(pitcher_id)]
    if match.empty:
        return ""
    value = match.iloc[-1].get("throws", "")
    return "" if pd.isna(value) else str(value)


def ingest_lineup_slots(
    config: MlbConfig,
    *,
    game_pks: list[int],
    provenance: str,
    http: HttpFn | None = None,
) -> pd.DataFrame:
    """Fetch complete starting nines, freeze cutoff-safe rates, and persist them."""
    from src.mlb.pipeline.ingest import snapshot_raw

    if provenance not in {"live_feed", "boxscore_00"}:
        raise ValueError(f"unsupported lineup provenance: {provenance}")

    store = MlbStore(config)
    game_versions = store.read_table("game_versions")
    people = store.read_table("id_map")
    batter_pas = store.read_table("batter_pas")
    existing = store.read_table("lineup_slots")
    known_nines = existing[
        ["game_pk", "batter_id", "slot_is_pitcher"]
    ].drop_duplicates()
    frames: list[pd.DataFrame] = []
    skip_rows: list[dict[str, Any]] = []
    seasons: set[int] = set()

    for game_pk in game_pks:
        params = {"game_pk": int(game_pk)}
        payload = (
            _local_lineup_payload(config, int(game_pk))
            if http is None
            else http(LINEUP_URL_TEMPLATE.format(game_pk=int(game_pk)), params)
        )
        parsed = parse_starting_nine(payload, game_pk=int(game_pk))
        original = _original_game_version(game_versions, int(game_pk))
        original_start = (
            _utc_timestamp(original["scheduled_start_utc"])
            if original is not None
            else pd.NaT
        )
        start = earliest_scheduled_start(game_versions, int(game_pk))
        season = int(parsed["season"].iloc[0]) if not parsed.empty else 0
        game_rows = game_versions.loc[game_versions["game_pk"] == int(game_pk)]
        if season == 0:
            if "season" in game_rows and game_rows["season"].notna().any():
                season = int(
                    game_rows.loc[game_rows["season"].notna(), "season"].iloc[0]
                )
            elif not pd.isna(start):
                season = int(start.year)
        if not parsed.empty:
            parsed["season"] = season
            known_nines = pd.concat(
                [
                    known_nines,
                    parsed[["game_pk", "batter_id", "slot_is_pitcher"]],
                ],
                ignore_index=True,
            ).drop_duplicates()
        seasons.add(season)
        doubleheader = (
            int(original.get("doubleheader", 0)) if original is not None else 0
        )
        game1_start = None
        if doubleheader == 2 and original is not None and not game_versions.empty:
            peers = game_versions.loc[
                (game_versions["home_team_id"] == original["home_team_id"])
                & (game_versions["away_team_id"] == original["away_team_id"])
                & (game_versions["doubleheader"] == 1)
            ]
            if not peers.empty:
                game1_start = peers.iloc[0]["scheduled_start_utc"]
        if (
            provenance == "boxscore_00"
            and doubleheader == 2
            and is_dummy_dh2_start(original_start, game1_start)
        ):
            skip_rows.append(
                {"game_pk": game_pk, "season": season, "reason": "dh2_dummy_start"}
            )
            continue
        if parsed.empty:
            skip_rows.append(
                {"game_pk": game_pk, "season": season, "reason": "no_boxscore"}
            )
            continue
        side_sizes = parsed.groupby(["team_id", "side"])["slot"].nunique()
        complete_keys = set(side_sizes.loc[side_sizes == 9].index)
        if len(complete_keys) != 2:
            skip_rows.append(
                {"game_pk": game_pk, "season": season, "reason": "incomplete_nine"}
            )
        if not complete_keys:
            continue

        effective_start = start
        if (
            provenance == "live_feed"
            and doubleheader == 2
            and is_dummy_dh2_start(original_start, game1_start)
        ):
            usable = game_rows.loc[
                [
                    not is_dummy_dh2_start(value, game1_start)
                    for value in game_rows["scheduled_start_utc"]
                ]
            ]
            if not usable.empty:
                valid_from = pd.to_datetime(usable["valid_from_utc"], utc=True)
                effective_start = _utc_timestamp(
                    usable.loc[valid_from.idxmax(), "scheduled_start_utc"]
                )
        if pd.isna(effective_start):
            skip_rows.append(
                {"game_pk": game_pk, "season": season, "reason": "dh2_dummy_start"}
            )
            continue
        cutoff = effective_start - pd.Timedelta(hours=config.forecast_horizon_hours)
        ingested_at = (
            effective_start - BOXSCORE_00_LEAD
            if provenance == "boxscore_00"
            else pd.Timestamp(_now_utc())
        )
        if provenance == "live_feed" and ingested_at >= cutoff:
            continue
        snapshot_id = snapshot_raw(
            config, "mlb_lineup_slots", params, payload, ingested_at
        )
        version = select_game_version(game_versions, int(game_pk), cutoff)

        for (team_id, side), side_slots in parsed.groupby(
            ["team_id", "side"], sort=False
        ):
            if (team_id, side) not in complete_keys:
                continue
            pitcher_field = (
                "probable_away_pitcher_id"
                if side == "home"
                else "probable_home_pitcher_id"
            )
            pitcher_id = version.get(pitcher_field) if version is not None else None
            frozen = freeze_lineup_slot_rates(
                side_slots,
                batter_pas,
                people,
                config,
                league_nines=known_nines,
                cutoff=cutoff,
                opposing_pitcher_hand=_pitcher_hand(people, pitcher_id),
                vs_pitcher_id=None if pd.isna(pitcher_id) else int(pitcher_id),
            )
            frozen["lineup_state"] = "announced"
            frozen["observed_before_cutoff"] = int(provenance == "live_feed")
            frozen["provenance"] = provenance
            frozen["ingested_at_utc"] = ingested_at
            frozen["snapshot_id"] = snapshot_id
            frames.append(coerce_frame(frozen, LINEUP_SLOT_COLUMNS))

    written = (
        pd.concat(frames, ignore_index=True)
        if frames
        else coerce_frame(pd.DataFrame(), LINEUP_SLOT_COLUMNS)
    )
    if not written.empty:
        if provenance == "live_feed":
            written = copy_live_ingest_clock(existing, written)
        keys = ["game_pk", "team_id", "slot", "rate_version", "provenance"]
        old_keys = set(existing[keys].itertuples(index=False, name=None))
        written = written.loc[
            [
                tuple(row) not in old_keys
                for row in written[keys].itertuples(index=False, name=None)
            ]
        ].reset_index(drop=True)
        if not written.empty:
            store.write_table(
                "lineup_slots",
                coerce_frame(
                    pd.concat([existing, written], ignore_index=True),
                    LINEUP_SLOT_COLUMNS,
                ),
            )

    skips = pd.DataFrame(skip_rows, columns=["game_pk", "season", "reason"])
    if provenance == "boxscore_00":
        coverage_slots = pd.concat([existing, written], ignore_index=True)
        if not coverage_slots.empty:
            coverage_slots = coverage_slots.loc[
                coverage_slots["game_pk"].isin(game_pks)
                & (coverage_slots["provenance"] == provenance)
                & (coverage_slots["rate_version"] == rate_version(config))
            ]
        for season in sorted(seasons):
            coverage = write_lineup_coverage(coverage_slots, skips, season=season)
            store.write_json(
                config.artifact_dir / f"lineup_coverage_{season}.json", coverage
            )
            assert_lineup_coverage(coverage, season=season)
    return written
