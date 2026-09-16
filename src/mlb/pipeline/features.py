"""Cutoff-strict feature rows for the starter strikeout MVP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.mlb import FEATURE_SET_VERSION
from src.mlb.config import MlbConfig
from src.mlb.models.shrinkage import shrink_rate
from src.mlb.schemas import (
    FASTBALL_TYPES,
    FEATURE_ROW_COLUMNS,
    LINEUP_STATE_CODES,
    PITCH_FAMILY,
    STARTER_STATE_CODES,
    coerce_frame,
    empty_frame,
    rules_era,
)

_PITCH_FAMILY = PITCH_FAMILY


def load_fixture_tables(root: Path | None = None) -> dict[str, pd.DataFrame]:
    from src.mlb.fixtures import load_fixture_tables as _load

    return _load(root)


def resolved_batter_stand(
    bats: str | None,
    pitcher_hand: str,
    pitch_stand: str | None = None,
) -> str:
    if pitch_stand is not None and str(pitch_stand) not in {"", "nan", "None", "<NA>"}:
        return str(pitch_stand)
    if bats == "S":
        return "R" if pitcher_hand == "L" else "L"
    return str(bats) if bats is not None else ""


def select_game_version(
    game_versions: pd.DataFrame,
    game_pk: int,
    cutoff: pd.Timestamp,
) -> pd.Series | None:
    if game_versions is None or game_versions.empty:
        return None
    gv = game_versions.loc[game_versions["game_pk"] == game_pk].copy()
    if gv.empty:
        return None
    valid_from = pd.to_datetime(gv["valid_from_utc"], utc=True)
    valid_to = pd.to_datetime(gv["valid_to_utc"], utc=True)
    open_ended = valid_to.isna()
    to_ok = open_ended | (cutoff <= valid_to)
    known = gv.loc[(valid_from < cutoff) & to_ok]
    if known.empty:
        return None
    idx = valid_from.loc[known.index].idxmax()
    return known.loc[idx]


def _utc(values: Any) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(values, utc=True)


def _cutoff_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _strict_before(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    time_col: str = "event_time_utc",
    ingest_col: str = "ingested_at_utc",
) -> pd.DataFrame:
    if frame is None or frame.empty or time_col not in frame.columns:
        return frame.iloc[0:0].copy() if frame is not None else pd.DataFrame()
    mask = _utc(frame[time_col]) < cutoff
    if ingest_col in frame.columns:
        mask = mask & (_utc(frame[ingest_col]) < cutoff)
    return frame.loc[mask].copy()


def _shrink_rate(
    successes: float,
    trials: float,
    prior_mean: float,
    strength: float,
) -> float:
    successes = float(successes)
    trials = float(trials)
    if pd.isna(prior_mean):
        if trials > 0:
            return successes / trials
        return float("nan")
    return float(shrink_rate(successes, trials, prior_mean, strength))


def _nanmean(values: pd.Series | np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype="float64")))


def _lineup_ids(raw: Any) -> list[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text or text in {"[]", "nan", "None"}:
        return []
    parsed = json.loads(text)
    return [int(x) for x in parsed]


def _note_times(
    event_times: list[pd.Timestamp],
    ingest_times: list[pd.Timestamp],
    frame: pd.DataFrame | None,
) -> None:
    if frame is None or frame.empty:
        return
    if "event_time_utc" in frame.columns:
        event_times.append(pd.Timestamp(_utc(frame["event_time_utc"]).max()))
    if "ingested_at_utc" in frame.columns:
        ingest_times.append(pd.Timestamp(_utc(frame["ingested_at_utc"]).max()))


def _plate_metrics(pitches: pd.DataFrame) -> dict[str, float]:
    if pitches is None or pitches.empty:
        return {
            "csw": float("nan"),
            "whiff": float("nan"),
            "chase": float("nan"),
            "zone": float("nan"),
            "swing": float("nan"),
            "called_strike": float("nan"),
            "n": 0,
        }
    n = int(len(pitches))
    called = float(pitches["is_called_strike"].sum())
    whiff = float(pitches["is_whiff"].sum())
    swing = float(pitches["is_swing"].sum())
    in_zone = float(pitches["is_in_zone"].sum())
    chase = float(pitches["is_chase"].sum())
    out_of_zone = n - in_zone
    return {
        "csw": (called + whiff) / n if n else float("nan"),
        "whiff": (whiff / swing) if swing else float("nan"),
        "chase": (chase / out_of_zone) if out_of_zone else float("nan"),
        "zone": in_zone / n if n else float("nan"),
        "swing": swing / n if n else float("nan"),
        "called_strike": called / n if n else float("nan"),
        "n": n,
    }


def _mix_and_velo(pitches: pd.DataFrame) -> dict[str, float]:
    empty = {
        "ff_share": float("nan"),
        "bb_share": float("nan"),
        "os_share": float("nan"),
        "fb_velo": float("nan"),
        "n_fb": 0,
    }
    if pitches is None or pitches.empty:
        return empty
    family = pitches["pitch_type"].astype(str).map(_PITCH_FAMILY)
    classified = family.notna()
    n_class = int(classified.sum())
    if n_class == 0:
        ff_share = bb_share = os_share = float("nan")
    else:
        ff_share = float((family == "ff").sum()) / n_class
        bb_share = float((family == "bb").sum()) / n_class
        os_share = float((family == "os").sum()) / n_class
    fb = pitches.loc[pitches["pitch_type"].astype(str).isin(FASTBALL_TYPES)]
    fb_velo = float(fb["release_speed"].mean()) if not fb.empty else float("nan")
    return {
        "ff_share": ff_share,
        "bb_share": bb_share,
        "os_share": os_share,
        "fb_velo": fb_velo,
        "n_fb": int(len(fb)),
    }


def _fill_pa_stand(pas: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    if pas is None or pas.empty:
        return pas
    out = pas.copy()
    if id_map is None or id_map.empty or "batter_id" not in out.columns:
        return out
    bats_map = None
    if "mlb_id" in id_map.columns:
        bats_map = id_map.set_index("mlb_id")["bats"]
    if bats_map is None:
        return out
    missing = out["batter_stand"].isna() | (out["batter_stand"].astype(str) == "")
    if not missing.any():
        return out

    def _fill(row: pd.Series) -> str:
        if pd.notna(row["batter_id"]):
            bats = bats_map.get(int(row["batter_id"]), "")
        else:
            bats = ""
        stand = row["batter_stand"] if pd.notna(row.get("batter_stand")) else None
        if stand is not None and str(stand) not in {"", "nan"}:
            return str(stand)
        bats_val = str(bats) if pd.notna(bats) else None
        return resolved_batter_stand(bats_val, str(row["pitcher_hand"]))

    out.loc[missing, "batter_stand"] = out.loc[missing].apply(_fill, axis=1)
    return out


def build_feature_rows(
    tables: dict[str, pd.DataFrame],
    config: MlbConfig,
) -> pd.DataFrame:
    pregame = tables.get("pregame_snapshots")
    if pregame is None or pregame.empty:
        return empty_frame(FEATURE_ROW_COLUMNS)

    pitches_all = tables.get("pitch_events", pd.DataFrame())
    pas_all = tables.get("plate_appearances", pd.DataFrame())
    starts_all = tables.get("pitcher_starts", pd.DataFrame())
    game_versions = tables.get("game_versions", pd.DataFrame())
    id_map = tables.get("id_map", pd.DataFrame())
    if pas_all is not None and not pas_all.empty:
        pas_all = _fill_pa_stand(pas_all, id_map)

    k_strength = float(config.pitcher_k_prior_strength)
    batter_strength = float(config.batter_k_prior_strength)
    feature_version = getattr(config, "feature_set_version", FEATURE_SET_VERSION)

    league_kbf_cache: dict[pd.Timestamp, float] = {}
    league_kpa_cache: dict[pd.Timestamp, float] = {}

    def league_k_bf(cutoff: pd.Timestamp) -> float:
        if cutoff not in league_kbf_cache:
            prior = _strict_before(starts_all, cutoff)
            bf = float(prior["batters_faced"].sum()) if not prior.empty else 0.0
            ks = float(prior["strikeouts"].sum()) if not prior.empty else 0.0
            league_kbf_cache[cutoff] = (ks / bf) if bf else float("nan")
        return league_kbf_cache[cutoff]

    def league_k_pa(cutoff: pd.Timestamp) -> float:
        if cutoff not in league_kpa_cache:
            prior = _strict_before(pas_all, cutoff)
            n = float(len(prior)) if prior is not None and not prior.empty else 0.0
            ks = float(prior["is_strikeout"].sum()) if n else 0.0
            league_kpa_cache[cutoff] = (ks / n) if n else float("nan")
        return league_kpa_cache[cutoff]

    rows: list[dict[str, Any]] = []
    for raw in pregame.to_dict(orient="records"):
        cutoff = _cutoff_ts(raw["prediction_cutoff_utc"])
        pitcher_id = int(raw["pitcher_id"])
        game_pk = int(raw["game_pk"])
        season = int(raw["season"])
        pitcher_hand = str(raw["pitcher_hand"])
        opponent_team_id = int(raw["opponent_team_id"])
        lineup_state = str(raw.get("lineup_state") or "team_fallback")
        starter_state = str(raw.get("starter_state") or "unknown")
        lineup_ids = _lineup_ids(raw.get("lineup_batter_ids_json"))
        version = select_game_version(game_versions, game_pk, cutoff)
        if version is not None and pd.notna(version.get("scheduled_start_utc")):
            this_scheduled = _cutoff_ts(version["scheduled_start_utc"])
        else:
            this_scheduled = _cutoff_ts(raw["scheduled_start_utc"])

        event_times: list[pd.Timestamp] = []
        ingest_times: list[pd.Timestamp] = []

        starts = _strict_before(starts_all, cutoff)
        if starts.empty or "pitcher_id" not in starts.columns:
            pitcher_starts = starts
        else:
            pitcher_starts = starts.loc[starts["pitcher_id"] == pitcher_id].copy()
        pitches = _strict_before(pitches_all, cutoff)
        if pitches.empty or "pitcher_id" not in pitches.columns:
            pitcher_pitches = pitches
        else:
            pitcher_pitches = pitches.loc[pitches["pitcher_id"] == pitcher_id].copy()
        pas = _strict_before(pas_all, cutoff)

        prior_mean = league_k_bf(cutoff)
        k_windows = {}
        for days, name in ((60, "60"), (365, "365")):
            lo = cutoff - pd.Timedelta(days=days)
            window = pitcher_starts.loc[_utc(pitcher_starts["event_time_utc"]) > lo]
            bf = float(window["batters_faced"].sum()) if not window.empty else 0.0
            ks = float(window["strikeouts"].sum()) if not window.empty else 0.0
            k_windows[name] = {
                "shrunk": _shrink_rate(ks, bf, prior_mean, k_strength),
                "n_eff": bf,
            }
            _note_times(event_times, ingest_times, window)

        prior2 = pitcher_starts.loc[
            pitcher_starts["season"].isin({season - 1, season - 2})
        ]
        bf2 = float(prior2["batters_faced"].sum()) if not prior2.empty else 0.0
        ks2 = float(prior2["strikeouts"].sum()) if not prior2.empty else 0.0
        k_prior2 = _shrink_rate(ks2, bf2, prior_mean, k_strength)
        _note_times(event_times, ingest_times, prior2)

        pitcher_pitches = pitcher_pitches.sort_values("event_time_utc")
        last_300 = pitcher_pitches.tail(300)
        last_750 = pitcher_pitches.tail(750)
        last_365 = pitcher_pitches.loc[
            _utc(pitcher_pitches["event_time_utc"]) > cutoff - pd.Timedelta(days=365)
        ]
        plate = {
            300: _plate_metrics(last_300),
            750: _plate_metrics(last_750),
            365: _plate_metrics(last_365),
        }
        mix_300 = _mix_and_velo(last_300)
        mix_365 = _mix_and_velo(last_365)
        _note_times(event_times, ingest_times, last_300)
        _note_times(event_times, ingest_times, last_750)
        _note_times(event_times, ingest_times, last_365)

        workload_pool = pitcher_starts
        if "scheduled_start_utc" in workload_pool.columns:
            workload_pool = workload_pool.loc[
                _utc(workload_pool["scheduled_start_utc"]) < cutoff
            ]
        workload_pool = workload_pool.sort_values("scheduled_start_utc")
        workload_vals: dict[str, float] = {}
        for window in (3, 5, 10):
            chunk = workload_pool.tail(window)
            if chunk.empty:
                bf_mean = pitches_mean = outs_mean = float("nan")
            else:
                bf_mean = _nanmean(chunk["batters_faced"])
                pitches_mean = _nanmean(chunk["pitches"])
                outs_mean = _nanmean(chunk["outs"])
            workload_vals[f"bf_per_start_{window}"] = bf_mean
            workload_vals[f"pitches_per_start_{window}"] = pitches_mean
            workload_vals[f"outs_per_start_{window}"] = outs_mean
        if workload_pool.empty:
            pitches_last_start = float("nan")
            rest_days = float("nan")
        else:
            last_start = workload_pool.iloc[-1]
            pitches_last_start = float(last_start["pitches"])
            prev_start = _cutoff_ts(last_start["scheduled_start_utc"])
            this_day = this_scheduled.tz_convert("UTC").normalize()
            prev_day = prev_start.tz_convert("UTC").normalize()
            rest_days = float((this_day - prev_day).days)
        _note_times(event_times, ingest_times, workload_pool.tail(10))

        batter_prior = league_k_pa(cutoff)
        opp_pas = _opponent_pas(
            pas,
            starts,
            pitcher_hand=pitcher_hand,
            opponent_team_id=opponent_team_id,
            lineup_ids=(
                lineup_ids if lineup_state == "announced" and lineup_ids else None
            ),
            cutoff=cutoff,
            season=season,
        )
        if opp_pas is not None and not opp_pas.empty:
            n_eff_opp = float(len(opp_pas))
        else:
            n_eff_opp = 0.0
        opp_k = float(opp_pas["is_strikeout"].sum()) if n_eff_opp else 0.0
        opp_k_rate = _shrink_rate(opp_k, n_eff_opp, batter_prior, batter_strength)
        _note_times(event_times, ingest_times, opp_pas)

        team_pas = _opponent_pas(
            pas,
            starts,
            pitcher_hand=pitcher_hand,
            opponent_team_id=opponent_team_id,
            lineup_ids=None,
            cutoff=cutoff,
            season=season,
        )
        if lineup_state == "announced" and lineup_ids:
            lineup_k_rate = _lineup_shrunk_k(
                pas,
                lineup_ids=lineup_ids,
                pitcher_hand=pitcher_hand,
                cutoff=cutoff,
                season=season,
                prior_mean=batter_prior,
                strength=batter_strength,
            )
            lineup_event_pas = _windowed_pas(
                pas.loc[
                    (pas["batter_id"].isin(lineup_ids))
                    & (pas["pitcher_hand"].astype(str) == pitcher_hand)
                ],
                cutoff,
                season,
            )
            _note_times(event_times, ingest_times, lineup_event_pas)
            missing_lineup = 0
        else:
            if team_pas is not None and not team_pas.empty:
                team_n = float(len(team_pas))
            else:
                team_n = 0.0
            team_k = float(team_pas["is_strikeout"].sum()) if team_n else 0.0
            lineup_k_rate = _shrink_rate(team_k, team_n, batter_prior, batter_strength)
            missing_lineup = int(not lineup_ids and team_n == 0)
            _note_times(event_times, ingest_times, team_pas)

        fallback = cutoff - pd.Timedelta(seconds=1)
        if event_times:
            max_event = max(event_times)
        else:
            max_event = fallback
        if ingest_times:
            max_ingest = max(ingest_times)
        else:
            max_ingest = fallback
        if pd.isna(max_event):
            max_event = fallback
        if pd.isna(max_ingest):
            max_ingest = fallback

        n_eff_365 = k_windows["365"]["n_eff"]
        missing_pitcher_k = int(n_eff_365 < 20)
        missing_plate = int(plate[300]["n"] < 50)
        missing_workload = int(len(workload_pool) < 3)
        missing_opponent = int(n_eff_opp < 30)
        n_fb = int(mix_365["n_fb"])
        missing_stuff = int(n_fb < 30)
        missing_role = 0

        row = {
            "pitcher_id": pitcher_id,
            "game_pk": game_pk,
            "prediction_cutoff_utc": cutoff,
            "feature_set_version": feature_version,
            "pregame_id": raw["pregame_id"],
            "k_bf_shrunk_60": k_windows["60"]["shrunk"],
            "k_bf_shrunk_365": k_windows["365"]["shrunk"],
            "k_bf_shrunk_prior2": k_prior2,
            "n_eff_k_bf_60": k_windows["60"]["n_eff"],
            "n_eff_k_bf_365": n_eff_365,
            "n_eff_k_bf_prior2": bf2,
            "csw_300": plate[300]["csw"],
            "whiff_300": plate[300]["whiff"],
            "chase_300": plate[300]["chase"],
            "zone_300": plate[300]["zone"],
            "swing_300": plate[300]["swing"],
            "called_strike_300": plate[300]["called_strike"],
            "csw_750": plate[750]["csw"],
            "whiff_750": plate[750]["whiff"],
            "chase_750": plate[750]["chase"],
            "zone_750": plate[750]["zone"],
            "swing_750": plate[750]["swing"],
            "called_strike_750": plate[750]["called_strike"],
            "csw_365": plate[365]["csw"],
            "whiff_365": plate[365]["whiff"],
            "chase_365": plate[365]["chase"],
            "zone_365": plate[365]["zone"],
            "swing_365": plate[365]["swing"],
            "called_strike_365": plate[365]["called_strike"],
            "bf_per_start_3": workload_vals["bf_per_start_3"],
            "pitches_per_start_3": workload_vals["pitches_per_start_3"],
            "outs_per_start_3": workload_vals["outs_per_start_3"],
            "bf_per_start_5": workload_vals["bf_per_start_5"],
            "pitches_per_start_5": workload_vals["pitches_per_start_5"],
            "outs_per_start_5": workload_vals["outs_per_start_5"],
            "bf_per_start_10": workload_vals["bf_per_start_10"],
            "pitches_per_start_10": workload_vals["pitches_per_start_10"],
            "outs_per_start_10": workload_vals["outs_per_start_10"],
            "pitches_last_start": pitches_last_start,
            "rest_days": rest_days,
            "expected_bf_oof": float("nan"),
            "bf_sd_oof": float("nan"),
            "expected_pitches_oof": float("nan"),
            "expected_outs_oof": float("nan"),
            "p_early_exit_oof": float("nan"),
            "opp_k_rate_vs_hand_shrunk": opp_k_rate,
            "n_eff_opp_k": n_eff_opp,
            "lineup_k_rate_shrunk": lineup_k_rate,
            "lineup_state_code": int(LINEUP_STATE_CODES.get(lineup_state, 0)),
            "pitcher_throws_L": 1.0 if pitcher_hand == "L" else 0.0,
            "expected_rhb_share": float(raw["expected_rhb_share"]),
            "fb_velo_300": mix_300["fb_velo"],
            "fb_velo_365": mix_365["fb_velo"],
            "fb_velo_delta": mix_300["fb_velo"] - mix_365["fb_velo"],
            "ff_share_300": mix_300["ff_share"],
            "bb_share_300": mix_300["bb_share"],
            "os_share_300": mix_300["os_share"],
            "ff_share_delta": mix_300["ff_share"] - mix_365["ff_share"],
            "bb_share_delta": mix_300["bb_share"] - mix_365["bb_share"],
            "os_share_delta": mix_300["os_share"] - mix_365["os_share"],
            "is_opener": float(raw["is_opener"]),
            "is_il_return": float(raw["is_il_return"]),
            "is_restricted": float(raw["is_restricted"]),
            "is_home": float(raw["is_home"]),
            "venue_id": int(raw["venue_id"]),
            "season": season,
            "rules_era": rules_era(season),
            "starter_state_code": int(STARTER_STATE_CODES.get(starter_state, 0)),
            "missing_pitcher_k": missing_pitcher_k,
            "missing_plate_discipline": missing_plate,
            "missing_workload": missing_workload,
            "missing_opponent": missing_opponent,
            "missing_lineup": missing_lineup,
            "missing_stuff": missing_stuff,
            "missing_role": missing_role,
            "max_input_event_time_utc": max_event,
            "max_source_ingestion_time_utc": max_ingest,
        }
        rows.append(row)

    return coerce_frame(pd.DataFrame(rows), FEATURE_ROW_COLUMNS)


def _windowed_pas(
    pas: pd.DataFrame,
    cutoff: pd.Timestamp,
    season: int,
) -> pd.DataFrame:
    if pas is None or pas.empty:
        return pas.iloc[0:0].copy() if pas is not None else pd.DataFrame()
    event = _utc(pas["event_time_utc"])
    if "season" in pas.columns:
        pa_season = pas["season"]
    else:
        pa_season = event.dt.year
    mask = (event > cutoff - pd.Timedelta(days=365)) | pa_season.isin(
        {season - 1, season - 2}
    )
    return pas.loc[mask].copy()


def _opponent_pas(
    pas: pd.DataFrame,
    starts: pd.DataFrame,
    *,
    pitcher_hand: str,
    opponent_team_id: int,
    lineup_ids: list[int] | None,
    cutoff: pd.Timestamp,
    season: int,
) -> pd.DataFrame:
    if pas is None or pas.empty:
        return pd.DataFrame()
    vs_hand = pas.loc[pas["pitcher_hand"].astype(str) == pitcher_hand].copy()
    if vs_hand.empty:
        return vs_hand
    if lineup_ids:
        vs_hand = vs_hand.loc[vs_hand["batter_id"].isin(lineup_ids)]
    elif starts is not None and not starts.empty:
        cols = ["game_pk", "pitcher_id", "opponent_team_id", "season"]
        start_key = starts[cols].drop_duplicates(subset=["game_pk", "pitcher_id"])
        vs_hand = vs_hand.merge(
            start_key,
            on=["game_pk", "pitcher_id"],
            how="inner",
            suffixes=("", "_start"),
        )
        vs_hand = vs_hand.loc[vs_hand["opponent_team_id"] == opponent_team_id]
    return _windowed_pas(vs_hand, cutoff, season)


def _lineup_shrunk_k(
    pas: pd.DataFrame,
    *,
    lineup_ids: list[int],
    pitcher_hand: str,
    cutoff: pd.Timestamp,
    season: int,
    prior_mean: float,
    strength: float,
) -> float:
    vs_hand = pas.loc[
        (pas["pitcher_hand"].astype(str) == pitcher_hand)
        & (pas["batter_id"].isin(lineup_ids))
    ]
    vs_hand = _windowed_pas(vs_hand, cutoff, season)
    rates: list[float] = []
    for batter_id in lineup_ids:
        if vs_hand.empty:
            sub = vs_hand
        else:
            sub = vs_hand.loc[vs_hand["batter_id"] == batter_id]
        n = float(len(sub)) if sub is not None and not sub.empty else 0.0
        ks = float(sub["is_strikeout"].sum()) if n else 0.0
        rates.append(_shrink_rate(ks, n, prior_mean, strength))
    if not rates:
        return float("nan")
    return float(np.nanmean(np.asarray(rates, dtype="float64")))
