from __future__ import annotations

import re
import unicodedata

import pandas as pd

_EVENT_SEP = re.compile(r"\s+(?:@|vs\.?|at)\s+", re.IGNORECASE)
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?", re.IGNORECASE)
_TEAM_TOKEN_ALIASES = {
    "dal mavericks": "dallas mavericks",
    "mil bucks": "milwaukee bucks",
    "phi 76ers": "philadelphia 76ers",
    "sa spurs": "san antonio spurs",
}

_PROP_ALIASES: dict[str, str] = {
    "points": "points",
    "player_points": "points",
    "pts": "points",
    "rebounds": "rebounds",
    "rebs": "rebounds",
    "assists": "assists",
    "asts": "assists",
    "threes": "threes",
    "3_pointers": "threes",
    "3 pointers": "threes",
    "3-pointers": "threes",
    "made_threes": "threes",
    "made threes": "threes",
    "pra": "pra",
    "pts_rebs_asts": "pra",
    "pts rebs asts": "pra",
    "pts + rebs + asts": "pra",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
}


def normalize_player(name: str | None) -> str | None:
    if name is None:
        return None
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace(".", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _SUFFIX.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _prop_lookup_key(stat: str) -> str:
    key = stat.strip().lower()
    key = key.replace("+", " + ")
    key = re.sub(r"\s+", " ", key).strip()
    return key


def normalize_prop(stat: str | None) -> str | None:
    if stat is None:
        return None
    raw = stat.strip()
    if not raw:
        return None
    key = _prop_lookup_key(raw)
    if key in _PROP_ALIASES:
        return _PROP_ALIASES[key]
    snake_key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    if snake_key in _PROP_ALIASES:
        return _PROP_ALIASES[snake_key]
    return snake_key or None


def normalize_side(side: str | None) -> str | None:
    if side is None:
        return None
    token = side.strip().lower()
    if token in ("over", "o"):
        return "over"
    if token in ("under", "u"):
        return "under"
    return None


def normalize_period(period: str | None) -> str:
    if period is None or not str(period).strip():
        return "game"
    return str(period).strip().lower()


def _normalize_team_token(team: str) -> str:
    text = team.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    normalized = re.sub(r"\s+", " ", text).strip()
    return _TEAM_TOKEN_ALIASES.get(normalized, normalized)


def _collapse_event_name(event_name: str) -> str:
    text = event_name.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def normalize_event_key(event_name: str | None) -> str | None:
    if event_name is None:
        return None
    raw = event_name.strip()
    if not raw:
        return None
    parts = _EVENT_SEP.split(raw, maxsplit=1)
    if len(parts) == 2:
        teams = [_normalize_team_token(parts[0]), _normalize_team_token(parts[1])]
        if all(teams):
            return "__".join(sorted(teams))
    collapsed = _collapse_event_name(raw)
    return collapsed or None


def _normalize_nullable(value: object, normalizer):
    if pd.isna(value):
        return normalizer(None)
    return normalizer(str(value))


def normalize_quotes(raw: pd.DataFrame) -> pd.DataFrame:
    """Add canonical market identifiers without mutating the source frame."""
    required = {"player", "stat", "side", "event_name"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"quotes missing required columns: {', '.join(missing)}")

    out = raw.copy()
    out["player_norm"] = out["player"].map(
        lambda value: _normalize_nullable(value, normalize_player)
    )
    out["prop"] = out["stat"].map(
        lambda value: _normalize_nullable(value, normalize_prop)
    )
    period = (
        out["period"]
        if "period" in out.columns
        else pd.Series(None, index=out.index, dtype="object")
    )
    out["period"] = period.map(
        lambda value: _normalize_nullable(value, normalize_period)
    )
    out["side"] = out["side"].map(
        lambda value: _normalize_nullable(value, normalize_side)
    )
    out["event_key"] = out["event_name"].map(
        lambda value: _normalize_nullable(value, normalize_event_key)
    )
    return out
