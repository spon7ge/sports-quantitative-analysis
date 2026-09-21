from __future__ import annotations

import hashlib
import json

from src.mlb.config import MlbConfig

STRIKEOUT_EVENT_TYPES = frozenset({
    "strikeout",
    "strikeout_double_play",
    "strikeout_triple_play",
})
RATE_VERSION_PREFIX = "kpa_"


def is_strikeout(
    event_type: str,
    *,
    events: frozenset[str] = STRIKEOUT_EVENT_TYPES,
) -> int:
    return int(str(event_type) in events)


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
