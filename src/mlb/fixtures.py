"""Deterministic synthetic MLB tables for offline tests and backtests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.mlb.config import MlbConfig, load_config
from src.mlb.schemas import (
    FEATURE_ROW_COLUMNS,
    GAME_VERSION_COLUMNS,
    ID_MAP_COLUMNS,
    MARKET_QUOTE_COLUMNS,
    PITCH_EVENT_COLUMNS,
    PITCHER_START_COLUMNS,
    PLATE_APPEARANCE_COLUMNS,
    PREGAME_SNAPSHOT_COLUMNS,
    RAW_SNAPSHOT_COLUMNS,
    coerce_frame,
    empty_frame,
    rules_era,
)

PITCHERS = (
    (111001, "Ace Lefty", "L", 0.30, 95.2, 0.22),
    (111002, "Stable Righty", "R", 0.24, 93.1, 0.18),
    (111003, "Soft Lefty", "L", 0.19, 90.4, 0.15),
    (111004, "Wild Righty", "R", 0.27, 96.8, 0.28),
    (111005, "Opener Righty", "R", 0.26, 94.0, 0.20),
    (111006, "Veteran Righty", "R", 0.22, 91.6, 0.16),
)

TEAMS = (
    (133, "OAK"),
    (134, "SEA"),
    (135, "TEX"),
    (136, "HOU"),
    (137, "LAA"),
    (138, "NYY"),
)

BATTERS_PER_TEAM = 9


def _utc(year: int, month: int, day: int, hour: int = 23, minute: int = 10) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _switch_stand(throws: str, bats: str) -> str:
    if bats != "S":
        return bats
    return "R" if throws == "L" else "L"


def build_synthetic_tables(config: MlbConfig | None = None) -> dict[str, pd.DataFrame]:
    cfg = config or load_config()
    rng = np.random.default_rng(cfg.seed)
    ingested = _utc(2016, 12, 1, 12, 0)

    id_rows = []
    batter_by_team: dict[int, list[int]] = {}
    batter_k: dict[int, float] = {}
    batter_bats: dict[int, str] = {}
    next_batter = 222000
    for team_id, _abbr in TEAMS:
        batters = []
        for slot in range(BATTERS_PER_TEAM):
            next_batter += 1
            bats = "S" if slot == 2 else ("L" if slot % 3 == 0 else "R")
            k_rate = float(np.clip(0.18 + 0.04 * rng.normal(), 0.08, 0.38))
            batters.append(next_batter)
            batter_k[next_batter] = k_rate
            batter_bats[next_batter] = bats
            id_rows.append(
                {
                    "mlb_id": next_batter,
                    "key_mlbam": next_batter,
                    "key_fangraphs": next_batter - 100000,
                    "key_bbref": f"bat{next_batter}",
                    "key_retro": f"B{next_batter}",
                    "name": f"Batter {next_batter}",
                    "bats": bats,
                    "throws": "R",
                }
            )
        batter_by_team[team_id] = batters

    for pitcher_id, name, throws, _k, _v, _w in PITCHERS:
        id_rows.append(
            {
                "mlb_id": pitcher_id,
                "key_mlbam": pitcher_id,
                "key_fangraphs": pitcher_id - 50000,
                "key_bbref": f"pit{pitcher_id}",
                "key_retro": f"P{pitcher_id}",
                "name": name,
                "bats": "R",
                "throws": throws,
            }
        )
    id_map = coerce_frame(pd.DataFrame(id_rows), ID_MAP_COLUMNS)

    schedule: list[dict[str, object]] = []
    game_pk = 500000
    season_starts = {2017: 8, 2018: 6, 2019: 4}
    for season, n_starts in season_starts.items():
        start = datetime(season, 4, 7, tzinfo=UTC)
        for start_idx in range(n_starts):
            day = start + timedelta(days=7 * start_idx)
            for pair in range(3):
                home = TEAMS[pair][0]
                away = TEAMS[pair + 3][0]
                home_pitcher = PITCHERS[pair][0]
                away_pitcher = PITCHERS[pair + 3][0]
                game_pk += 1
                scheduled = _utc(day.year, day.month, day.day, 23, 10)
                schedule.append(
                    {
                        "game_pk": game_pk,
                        "season": season,
                        "game_date": day.strftime("%Y-%m-%d"),
                        "scheduled_start_utc": scheduled,
                        "home_team_id": home,
                        "away_team_id": away,
                        "home_pitcher_id": home_pitcher,
                        "away_pitcher_id": away_pitcher,
                        "venue_id": 1000 + pair,
                        "doubleheader": 0,
                    }
                )

    # Doubleheader on the last 2018 Friday.
    last_2018 = [row for row in schedule if row["season"] == 2018][-1]
    game_pk += 1
    dh = dict(last_2018)
    dh.update(
        {
            "game_pk": game_pk,
            "scheduled_start_utc": last_2018["scheduled_start_utc"]
            + timedelta(hours=4),
            "doubleheader": 2,
        }
    )
    last_2018["doubleheader"] = 1
    schedule.append(dh)

    # Postponement: clone an early 2019 game to the next day.
    first_2019 = [row for row in schedule if row["season"] == 2019][0]
    postponed_pk = int(first_2019["game_pk"])

    # Changed starter: replace probable away pitcher after first snapshot.
    change_game = [row for row in schedule if row["season"] == 2019][1]
    original_away = int(change_game["away_pitcher_id"])
    replacement_away = 111006
    if replacement_away == original_away:
        replacement_away = 111003
    change_game["away_pitcher_id"] = replacement_away

    game_versions_rows: list[dict[str, object]] = []
    pregame_rows: list[dict[str, object]] = []
    start_rows: list[dict[str, object]] = []
    pa_rows: list[dict[str, object]] = []
    pitch_rows: list[dict[str, object]] = []
    quote_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []

    snapshot_counter = 0

    def new_snapshot(source: str, params: dict[str, object]) -> str:
        nonlocal snapshot_counter
        snapshot_counter += 1
        snapshot_id = f"snap-{snapshot_counter:04d}"
        raw_rows.append(
            {
                "snapshot_id": snapshot_id,
                "source": source,
                "request_params_json": json.dumps(params, sort_keys=True),
                "fetched_at_utc": ingested,
                "payload_hash": f"hash-{snapshot_counter:04d}",
                "parser_version": cfg.parser_version,
                "payload_path": f"fixtures/raw/{snapshot_id}.json",
            }
        )
        return snapshot_id

    pitcher_last_start: dict[int, datetime] = {}
    pitcher_traits = {
        pid: {"k": k, "velo": velo, "whiff": whiff, "throws": throws, "name": name}
        for pid, name, throws, k, velo, whiff in PITCHERS
    }

    for game in schedule:
        snap = new_snapshot("mlb_stats_api", {"game_pk": game["game_pk"]})
        scheduled = game["scheduled_start_utc"]
        assert isinstance(scheduled, datetime)
        valid_from = scheduled - timedelta(days=2)
        status = "scheduled"
        if int(game["game_pk"]) == postponed_pk:
            game_versions_rows.append(
                {
                    "game_pk": game["game_pk"],
                    "scheduled_start_utc": scheduled,
                    "status": "postponed",
                    "home_team_id": game["home_team_id"],
                    "away_team_id": game["away_team_id"],
                    "venue_id": game["venue_id"],
                    "doubleheader": game["doubleheader"],
                    "probable_home_pitcher_id": game["home_pitcher_id"],
                    "probable_away_pitcher_id": original_away
                    if game is change_game
                    else game["away_pitcher_id"],
                    "valid_from_utc": valid_from,
                    "valid_to_utc": scheduled - timedelta(hours=20),
                    "snapshot_id": snap,
                }
            )
            scheduled = scheduled + timedelta(days=1)
            game["scheduled_start_utc"] = scheduled
            game["game_date"] = scheduled.strftime("%Y-%m-%d")
            status = "rescheduled"

        probable_away = (
            original_away if game is change_game else game["away_pitcher_id"]
        )
        game_versions_rows.append(
            {
                "game_pk": game["game_pk"],
                "scheduled_start_utc": scheduled,
                "status": status,
                "home_team_id": game["home_team_id"],
                "away_team_id": game["away_team_id"],
                "venue_id": game["venue_id"],
                "doubleheader": game["doubleheader"],
                "probable_home_pitcher_id": game["home_pitcher_id"],
                "probable_away_pitcher_id": probable_away,
                "valid_from_utc": valid_from
                if int(game["game_pk"]) != postponed_pk
                else scheduled - timedelta(hours=20),
                "valid_to_utc": scheduled + timedelta(hours=6),
                "snapshot_id": snap,
            }
        )
        if game is change_game:
            later = new_snapshot(
                "mlb_stats_api",
                {"game_pk": game["game_pk"], "revision": "starter_change"},
            )
            game_versions_rows.append(
                {
                    "game_pk": game["game_pk"],
                    "scheduled_start_utc": scheduled,
                    "status": "scheduled",
                    "home_team_id": game["home_team_id"],
                    "away_team_id": game["away_team_id"],
                    "venue_id": game["venue_id"],
                    "doubleheader": game["doubleheader"],
                    "probable_home_pitcher_id": game["home_pitcher_id"],
                    "probable_away_pitcher_id": replacement_away,
                    "valid_from_utc": scheduled - timedelta(hours=1, minutes=30),
                    "valid_to_utc": scheduled + timedelta(hours=6),
                    "snapshot_id": later,
                }
            )

        for side, pitcher_id, team_id, opp_id, is_home in (
            (
                "home",
                int(game["home_pitcher_id"]),
                int(game["home_team_id"]),
                int(game["away_team_id"]),
                1,
            ),
            (
                "away",
                int(game["away_pitcher_id"]),
                int(game["away_team_id"]),
                int(game["home_team_id"]),
                0,
            ),
        ):
            traits = pitcher_traits[pitcher_id]
            throws = str(traits["throws"])
            cutoff = scheduled - timedelta(hours=cfg.forecast_horizon_hours)
            starter_state = "probable"
            if game is change_game and side == "away":
                starter_state = "announced"
            if pitcher_id == 111005 and int(game["season"]) == 2019:
                role = "opener"
            else:
                role = "starter"
            is_opener = int(role == "opener")
            is_restricted = int(
                pitcher_id == 111004 and int(game["season"]) == 2017 and is_home
            )
            is_il_return = int(
                pitcher_id == 111006 and game["game_date"] == "2018-04-07"
            )
            if int(game["season"]) >= 2019:
                lineup_state = "announced"
            else:
                lineup_state = "team_fallback"
            batters = batter_by_team[opp_id]
            lineup_ids = batters if lineup_state == "announced" else []
            rhb_share = float(
                np.mean(
                    [
                        1.0 if _switch_stand(throws, batter_bats[b]) == "R" else 0.0
                        for b in batters
                    ]
                )
            )
            stamp = cutoff.strftime("%Y%m%dT%H%M")
            pregame_id = f"{game['game_pk']}-{pitcher_id}-{stamp}"
            pregame_rows.append(
                {
                    "pregame_id": pregame_id,
                    "game_pk": game["game_pk"],
                    "pitcher_id": pitcher_id,
                    "prediction_cutoff_utc": cutoff,
                    "forecast_horizon_hours": cfg.forecast_horizon_hours,
                    "scheduled_start_utc": scheduled,
                    "game_date": game["game_date"],
                    "season": game["season"],
                    "starter_state": starter_state,
                    "lineup_state": lineup_state,
                    "roster_state": "active",
                    "is_opener": is_opener,
                    "is_il_return": is_il_return,
                    "is_restricted": is_restricted,
                    "is_home": is_home,
                    "team_id": team_id,
                    "opponent_team_id": opp_id,
                    "venue_id": game["venue_id"],
                    "pitcher_hand": throws,
                    "expected_rhb_share": rhb_share,
                    "lineup_batter_ids_json": json.dumps(lineup_ids),
                    "source_snapshot_ids_json": json.dumps([snap]),
                }
            )

            mean_bf = 18 if is_opener else (16 if is_restricted else 23)
            n_pa = int(np.clip(rng.normal(mean_bf, 3.0), 9, 32))
            event_time = scheduled
            strikeouts = 0
            pitches = 0
            outs = 0
            for pa_idx in range(n_pa):
                batter_id = batters[pa_idx % 9]
                tto = pa_idx // 9 + 1
                stand = _switch_stand(throws, batter_bats[batter_id])
                p_k = float(
                    np.clip(
                        traits["k"]
                        + 0.4 * (batter_k[batter_id] - 0.22)
                        + rng.normal(0, 0.02),
                        0.05,
                        0.55,
                    )
                )
                is_k = int(rng.random() < p_k)
                strikeouts += is_k
                n_pitches = int(np.clip(rng.integers(3, 8), 3, 8))
                pitches += n_pitches
                if is_k:
                    result = "strikeout"
                elif rng.random() < 0.7:
                    result = "out"
                else:
                    result = "onbase"
                if result != "onbase":
                    outs += 1
                pa_time = scheduled + timedelta(minutes=3 * pa_idx)
                pa_id = f"{game['game_pk']}-{pitcher_id}-{pa_idx + 1}"
                pa_rows.append(
                    {
                        "game_pk": game["game_pk"],
                        "pa_id": pa_id,
                        "at_bat_number": pa_idx + 1,
                        "pitcher_id": pitcher_id,
                        "batter_id": batter_id,
                        "result": result,
                        "is_strikeout": is_k,
                        "batting_slot": (pa_idx % 9) + 1,
                        "pitcher_hand": throws,
                        "batter_stand": stand,
                        "tto_number": tto,
                        "pitches_in_pa": n_pitches,
                        "event_time_utc": pa_time,
                        "ingested_at_utc": pa_time + timedelta(hours=4),
                        "snapshot_id": snap,
                    }
                )
                for pitch_n in range(n_pitches):
                    in_zone = int(rng.random() < 0.48)
                    swing = int(rng.random() < (0.68 if in_zone else 0.32))
                    whiff = int(swing and rng.random() < traits["whiff"])
                    called = int((not swing) and in_zone and rng.random() < 0.55)
                    chase = int(swing and not in_zone)
                    family_draw = rng.random()
                    if family_draw < 0.55:
                        ptype = "FF"
                    elif family_draw < 0.80:
                        ptype = "SL"
                    else:
                        ptype = "CH"
                    velo = float(
                        traits["velo"]
                        + rng.normal(0, 1.1)
                        - (0.4 if ptype != "FF" else 0)
                    )
                    pitch_time = pa_time + timedelta(seconds=20 * pitch_n)
                    pitch_rows.append(
                        {
                            "game_pk": game["game_pk"],
                            "pitch_id": f"{pa_id}-{pitch_n + 1}",
                            "at_bat_number": pa_idx + 1,
                            "pitch_number": pitch_n + 1,
                            "pitcher_id": pitcher_id,
                            "batter_id": batter_id,
                            "event_time_utc": pitch_time,
                            "pitch_result": (
                                "whiff"
                                if whiff
                                else ("called_strike" if called else "other")
                            ),
                            "pa_result": result,
                            "pitcher_hand": throws,
                            "batter_stand": stand,
                            "release_speed": velo,
                            "pfx_x": float(rng.normal(0, 0.6)),
                            "pfx_z": float(rng.normal(0.8, 0.4)),
                            "plate_x": float(rng.normal(0, 0.5)),
                            "plate_z": float(rng.normal(2.5, 0.4)),
                            "pitch_type": ptype,
                            "is_swing": swing,
                            "is_whiff": whiff,
                            "is_called_strike": called,
                            "is_in_zone": in_zone,
                            "is_chase": chase,
                            "source_vintage": "synthetic_v1",
                            "ingested_at_utc": pitch_time + timedelta(hours=4),
                            "snapshot_id": snap,
                        }
                    )
                event_time = pa_time

            start_rows.append(
                {
                    "pitcher_id": pitcher_id,
                    "game_pk": game["game_pk"],
                    "game_date": game["game_date"],
                    "season": game["season"],
                    "strikeouts": strikeouts,
                    "batters_faced": n_pa,
                    "pitches": pitches,
                    "outs": min(outs, 27),
                    "role": role,
                    "is_home": is_home,
                    "opponent_team_id": opp_id,
                    "team_id": team_id,
                    "venue_id": game["venue_id"],
                    "pitcher_hand": throws,
                    "scheduled_start_utc": scheduled,
                    "event_time_utc": scheduled,
                    "ingested_at_utc": event_time + timedelta(hours=4),
                    "doubleheader": game["doubleheader"],
                }
            )
            pitcher_last_start[pitcher_id] = scheduled

            if int(game["season"]) == 2019 and rng.random() < 0.7:
                line = 5.5
                quote_time = cutoff - timedelta(minutes=15)
                quote_rows.append(
                    {
                        "quote_id": f"q-{game['game_pk']}-{pitcher_id}",
                        "game_pk": game["game_pk"],
                        "pitcher_id": pitcher_id,
                        "sportsbook": "FixtureBook",
                        "jurisdiction": "US-NJ",
                        "line": line,
                        "over_price": -115.0,
                        "under_price": -105.0,
                        "price_format": "american",
                        "fetched_at_utc": quote_time,
                        "quote_status": "open",
                        "rule_version": "k_standard_v1",
                        "fill_flag": pd.NA,
                        "rejected_flag": pd.NA,
                        "limit": np.nan,
                        "void_flag": pd.NA,
                        "settled_flag": pd.NA,
                    }
                )

    # Leakage probe: a quote fetched after one 2019 cutoff must not enter that forecast.
    probe = pregame_rows[-1]
    quote_rows.append(
        {
            "quote_id": "q-late-probe",
            "game_pk": probe["game_pk"],
            "pitcher_id": probe["pitcher_id"],
            "sportsbook": "FixtureBook",
            "jurisdiction": "US-NJ",
            "line": 5.5,
            "over_price": -200.0,
            "under_price": 170.0,
            "price_format": "american",
            "fetched_at_utc": probe["prediction_cutoff_utc"] + timedelta(minutes=5),
            "quote_status": "current",
            "rule_version": "k_standard_v1",
            "fill_flag": pd.NA,
            "rejected_flag": pd.NA,
            "limit": np.nan,
            "void_flag": pd.NA,
            "settled_flag": pd.NA,
        }
    )

    tables = {
        "raw_snapshots": coerce_frame(
            pd.DataFrame(raw_rows), RAW_SNAPSHOT_COLUMNS
        ),
        "game_versions": coerce_frame(
            pd.DataFrame(game_versions_rows), GAME_VERSION_COLUMNS
        ),
        "pitch_events": coerce_frame(
            pd.DataFrame(pitch_rows), PITCH_EVENT_COLUMNS
        ),
        "plate_appearances": coerce_frame(
            pd.DataFrame(pa_rows), PLATE_APPEARANCE_COLUMNS
        ),
        "pitcher_starts": coerce_frame(
            pd.DataFrame(start_rows), PITCHER_START_COLUMNS
        ),
        "pregame_snapshots": coerce_frame(
            pd.DataFrame(pregame_rows), PREGAME_SNAPSHOT_COLUMNS
        ),
        "market_quotes": coerce_frame(
            pd.DataFrame(quote_rows), MARKET_QUOTE_COLUMNS
        ),
        "id_map": id_map,
        "feature_rows": empty_frame(FEATURE_ROW_COLUMNS),
    }
    _ = rules_era
    return tables


def write_fixture_tables(
    root: Path | None = None,
    config: MlbConfig | None = None,
) -> Path:
    cfg = config or load_config()
    tables = build_synthetic_tables(cfg)
    out = Path(root) if root is not None else cfg.fixture_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    raw_dir = out / "raw"
    raw_dir.mkdir(exist_ok=True)
    schedule_payload = {
        "source": "mlb_stats_api",
        "games": tables["game_versions"].head(4).astype(str).to_dict(orient="records"),
    }
    (raw_dir / "schedule.json").write_text(json.dumps(schedule_payload, indent=2))
    savant_payload = tables["pitch_events"].head(20).to_csv(index=False)
    (raw_dir / "statcast_sample.csv").write_text(savant_payload)
    announced = tables["pregame_snapshots"]
    announced = announced.loc[announced["lineup_state"] == "announced"]
    lineup_rows: list[dict[str, object]] = []
    if not announced.empty:
        rec = announced.iloc[0]
        try:
            batter_ids = json.loads(str(rec["lineup_batter_ids_json"] or "[]"))
        except json.JSONDecodeError:
            batter_ids = []
        side = "away" if int(rec["is_home"]) == 1 else "home"
        for slot, batter_id in enumerate(batter_ids, start=1):
            lineup_rows.append(
                {
                    "game_pk": int(rec["game_pk"]),
                    "team_id": int(rec["opponent_team_id"]),
                    "batter_id": int(batter_id),
                    "batting_slot": slot,
                    "side": side,
                }
            )
    (raw_dir / "lineups.json").write_text(json.dumps(lineup_rows, indent=2))
    quotes = tables["market_quotes"]
    quotes.to_csv(out / "quotes.csv", index=False)
    return out


def load_fixture_tables(root: Path | None = None) -> dict[str, pd.DataFrame]:
    cfg = load_config()
    out = Path(root) if root is not None else cfg.fixture_dir
    names = [
        "raw_snapshots",
        "game_versions",
        "pitch_events",
        "plate_appearances",
        "pitcher_starts",
        "pregame_snapshots",
        "market_quotes",
        "id_map",
    ]
    missing = [name for name in names if not (out / f"{name}.parquet").exists()]
    if missing:
        write_fixture_tables(out, cfg)
    from src.mlb.schemas import TABLE_SCHEMAS

    loaded = {}
    for name in names:
        loaded[name] = coerce_frame(
            pd.read_parquet(out / f"{name}.parquet"),
            TABLE_SCHEMAS[name],
        )
    return loaded
