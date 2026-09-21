from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.models.shrinkage import shrink_rate

STRIKEOUT_EVENT_TYPES = frozenset({
    "strikeout",
    "strikeout_double_play",
    "strikeout_triple_play",
})
RATE_VERSION_PREFIX = "kpa_"


@dataclass(frozen=True)
class LeagueKPa:
    overall: float
    by_bats: dict[str, float]
    by_bats_hand: dict[tuple[str, str], float]


def is_strikeout(
    event_type: str,
    *,
    events: frozenset[str] = STRIKEOUT_EVENT_TYPES,
) -> int:
    return int(str(event_type) in events)


def league_eligible_pas(pas: pd.DataFrame, nines: pd.DataFrame) -> pd.DataFrame:
    two_way = nines.loc[
        nines["slot_is_pitcher"] == 0, ["game_pk", "batter_id"]
    ].drop_duplicates()
    tagged = pas.merge(
        two_way.assign(_two_way=1),
        on=["game_pk", "batter_id"],
        how="left",
    )
    keep = (tagged["is_pitcher_in_game"] == 0) | (tagged["_two_way"] == 1)
    return pas.loc[keep.to_numpy()].copy()


def rate_version(config: MlbConfig) -> str:
    payload = {
        "exclude_pitcher_batters": "pbp_pitcher_ids_minus_two_way",
        "hand_prior": float(config.batter_hand_prior_strength),
        "overall": "leave_one_split_out",
        "overall_prior": float(config.batter_k_prior_strength),
        "platoon": "odds_ratio",
        "prior_seasons": 2,
        "strikeout_events": sorted(STRIKEOUT_EVENT_TYPES),
        "windows_days": [60, 365],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return RATE_VERSION_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _odds(p: float) -> float:
    clipped = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return clipped / (1.0 - clipped)


def league_platoon_odds_ratio(
    *,
    bats: str,
    league_k_pa_cell: float,
    league_k_pa_bats: float,
) -> float:
    if bats is None or str(bats).strip() in {"", "nan", "<NA>", "None"}:
        return 1.0
    return _odds(league_k_pa_cell) / _odds(league_k_pa_bats)


def shrink_batter_k_pa(
    pas: pd.DataFrame,
    *,
    batter_id: int,
    opposing_pitcher_hand: str,
    bats: str,
    cutoff: pd.Timestamp,
    league: LeagueKPa,
    config: MlbConfig,
) -> dict[str, dict[str, float]]:
    cutoff = pd.Timestamp(cutoff)
    event_times = pd.to_datetime(pas["event_time_utc"], utc=True)
    eligible = pas.loc[
        (pas["batter_id"] == batter_id)
        & (event_times < cutoff)
        & (pas["event_time_imputed"] == 0)
    ].copy()
    eligible["_event_time_utc"] = event_times.loc[eligible.index]
    eligible["_is_strikeout"] = (
        eligible["event_type"].map(is_strikeout).astype("int64")
    )

    windows = {
        "60": eligible["_event_time_utc"] >= cutoff - pd.Timedelta(days=60),
        "365": eligible["_event_time_utc"] >= cutoff - pd.Timedelta(days=365),
        "prior2": eligible["_event_time_utc"].dt.year.isin(
            {cutoff.year - 1, cutoff.year - 2}
        ),
    }

    results: dict[str, dict[str, float]] = {}
    for name, mask in windows.items():
        history = eligible.loc[mask]
        pa_all = float(len(history))
        k_all = float(history["_is_strikeout"].sum())

        if opposing_pitcher_hand:
            hand_history = history.loc[
                history["pitcher_hand"] == opposing_pitcher_hand
            ]
            pa_hand = float(len(hand_history))
            k_hand = float(hand_history["_is_strikeout"].sum())
        else:
            pa_hand = 0.0
            k_hand = 0.0

        overall = shrink_rate(
            k_all - k_hand,
            pa_all - pa_hand,
            league.overall,
            config.batter_k_prior_strength,
        )

        if opposing_pitcher_hand:
            ratio = league_platoon_odds_ratio(
                bats=bats,
                league_k_pa_cell=league.by_bats_hand.get(
                    (bats, opposing_pitcher_hand), league.overall
                ),
                league_k_pa_bats=league.by_bats.get(bats, league.overall),
            )
            prior_odds = _odds(overall) * ratio
            prior_mean = prior_odds / (1.0 + prior_odds)
            vs_hand = shrink_rate(
                k_hand,
                pa_hand,
                prior_mean,
                config.batter_hand_prior_strength,
            )
        else:
            vs_hand = float("nan")

        results[name] = {
            "k_pa_vs_hand_shrunk": float(vs_hand),
            "k_pa_overall_shrunk": float(overall),
            "pa_vs_hand": pa_hand,
            "pa_all": pa_all,
        }

    return results
