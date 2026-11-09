"""Prior-only player and team event rates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from game_events.checks import _positive_float, _positive_int
from game_events.constants import (
    _EVENT_IDS,
    _EVENT_TEAM_COLUMNS,
    _PLAYER_SOURCES,
    _TEAM_PRIOR_TRIALS,
    EVENT_RATES,
    LEAGUE_SEASON_DECAY,
    PLAYER_EVENT_RATES,
    PLAYER_PRIOR_MINUTES,
    PLAYER_WINDOW,
    TEAM_EVENT_RATES,
    TEAM_WINDOW,
)
from game_events.logs import _prepare_event_logs, _prepare_event_slate
from game_events.model import EventModel
from game_events.strength import prior_minutes_map


def training_events(
    logs: pd.DataFrame, model: EventModel | None = None
) -> pd.DataFrame:
    """One player-game. Event rates stop before that game's date.

    The model supplies the selected windows and prior strengths; without one
    the defaults are used.
    """
    frame = _prepare_event_logs(logs)
    targets = frame.loc[:, [*_EVENT_IDS, *_role_columns(frame)]].reset_index(drop=True)
    return _rate_profile(frame, targets, _settings(model))


def live_events(
    logs: pd.DataFrame,
    slate: pd.DataFrame,
    model: EventModel | None = None,
) -> pd.DataFrame:
    """One upcoming player. The latest completed game stays in the rates."""
    frame = _prepare_event_logs(logs)
    upcoming = _prepare_event_slate(slate)
    if model is not None and (upcoming["game_date"] <= model.cutoff_date).any():
        raise ValueError("slate dates must follow the event model cutoff")
    targets = upcoming.loc[
        :, ["player_id", "game_date", "team_id", *_role_columns(upcoming)]
    ]
    return _rate_profile(frame, targets, _settings(model))


def _settings(model: EventModel | None) -> dict[str, Any]:
    if model is None:
        return {
            "player_window": PLAYER_WINDOW,
            "player_prior_minutes": PLAYER_PRIOR_MINUTES,
            "team_window": TEAM_WINDOW,
            "team_prior_scale": 1.0,
        }
    if not isinstance(model, EventModel):
        raise ValueError("model must be an EventModel")
    return {
        "player_window": model.player_window,
        "player_prior_minutes": model.player_prior_minutes,
        "team_window": model.team_window,
        "team_prior_scale": model.team_prior_scale,
    }


def _targets(rows: pd.DataFrame, entity: str) -> pd.DataFrame:
    targets = rows[[entity, "game_date", *_role_columns(rows)]].reset_index(drop=True)
    targets = targets.copy()
    targets["_row"] = np.arange(len(targets))
    return targets


def _rate_profile(
    frame: pd.DataFrame, targets: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    targets = targets.reset_index(drop=True).copy()
    targets["_row"] = np.arange(len(targets))
    played = _player_counts(frame)
    teams = _team_counts(frame)
    players = _player_rates(
        played, targets, settings["player_window"], settings["player_prior_minutes"]
    )
    team_rates = _team_rates(
        teams, targets, settings["team_window"], settings["team_prior_scale"]
    )
    profile = pd.concat([targets.drop(columns="_row"), players, team_rates], axis=1)
    # Keep the public column order stable for callers that slice EVENT_RATES.
    ordered = [
        column for column in (*_EVENT_IDS, *EVENT_RATES) if column in profile.columns
    ]
    extras = [column for column in profile.columns if column not in ordered]
    return profile.loc[:, [*ordered, *extras]]


def _player_rates(
    played: pd.DataFrame,
    targets: pd.DataFrame,
    window: int | None,
    prior_minutes: float | dict[str, float],
) -> pd.DataFrame:
    """Shrunk per-minute rates. ``window`` None keeps every earlier game."""
    parts = _player_rate_components(played, targets, window)
    strength = prior_minutes_map(prior_minutes)
    out = pd.DataFrame(index=range(len(targets)))
    for rate in PLAYER_EVENT_RATES:
        out[rate] = _shrunk(
            parts[f"{rate}_n"],
            parts[f"{rate}_d"],
            parts[f"{rate}_t"],
            strength[rate],
        )
    return out


def _player_rate_components(
    played: pd.DataFrame, targets: pd.DataFrame, window: int | None
) -> pd.DataFrame:
    """Prior-window events, exposure, and target rate, before the target date.

    ``{rate}_n`` and ``{rate}_d`` are the window. ``{rate}_t`` is the role
    rate when that role has already been seen, otherwise the league rate.
    """
    window = _player_window(window)
    columns = [f"{rate}_{part}" for rate in PLAYER_EVENT_RATES for part in "nd"]
    prior = _prior_window_sums(played, targets, "player_id", columns, window)
    league = _prior_league_sums(played, targets, columns)
    roles = _target_roles(played, targets)
    role_sums = _prior_role_sums(played, targets, columns, roles)
    role_known = roles.ne("").to_numpy()
    out = pd.DataFrame(index=range(len(targets)))
    for rate in PLAYER_EVENT_RATES:
        league_rate = _ratio(league[f"{rate}_n"], league[f"{rate}_d"])
        role_rate = _ratio(role_sums[f"{rate}_n"], role_sums[f"{rate}_d"])
        role_denom = role_sums[f"{rate}_d"].to_numpy(dtype=float)
        use_role = role_known & np.isfinite(role_denom) & (role_denom > 0.0)
        out[f"{rate}_n"] = prior[f"{rate}_n"].to_numpy(dtype=float)
        out[f"{rate}_d"] = prior[f"{rate}_d"].to_numpy(dtype=float)
        out[f"{rate}_t"] = np.where(use_role, role_rate, league_rate)
    return out


def _player_window(window: int | None) -> int | None:
    if window is None:
        return None
    return _positive_int(window, "player_window")


def _team_rates(
    teams: pd.DataFrame, targets: pd.DataFrame, window: int, prior_scale: float
) -> pd.DataFrame:
    window = _positive_int(window, "team_window")
    prior_scale = _positive_float(prior_scale, "team_prior_scale")
    columns = [f"{rate}_{part}" for rate in TEAM_EVENT_RATES for part in "nd"]
    prior = _prior_window_sums(teams, targets, "team_id", columns, window)
    league = _prior_league_sums(teams, targets, columns)
    out = pd.DataFrame(index=range(len(targets)))
    for rate in TEAM_EVENT_RATES:
        out[rate] = np.clip(
            _shrunk(
                prior[f"{rate}_n"],
                prior[f"{rate}_d"],
                np.minimum(_ratio(league[f"{rate}_n"], league[f"{rate}_d"]), 1.0),
                prior_scale * _TEAM_PRIOR_TRIALS[rate],
            ),
            0.0,
            1.0,
        )
    return out


def _player_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Positive-minute appearances with per-stat event and minute exposure."""
    played = frame.loc[frame["minutes"] > 0].copy()
    for rate, stat in _PLAYER_SOURCES.items():
        ok = played[stat].notna()
        played[f"{rate}_n"] = played[stat].where(ok, 0.0).astype(float)
        played[f"{rate}_d"] = played["minutes"].where(ok, 0.0).astype(float)
    return played.sort_values(["player_id", "game_date", "game_id"]).reset_index(
        drop=True
    )


def _team_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with the opponent's row attached."""
    teams = frame.groupby(["game_id", "team_id"], as_index=False, sort=False).agg(
        game_date=("game_date", "first"),
        **{column: (column, "first") for column in _EVENT_TEAM_COLUMNS},
    )
    opponent = teams.drop(columns="game_date").rename(
        columns={column: f"{column}_o" for column in ("team_id", *_EVENT_TEAM_COLUMNS)}
    )
    pairs = teams[["game_id", "team_id"]].merge(opponent, on="game_id")
    pairs = pairs.loc[pairs["team_id"] != pairs["team_id_o"]]
    teams = teams.merge(
        pairs, on=["game_id", "team_id"], how="left", validate="one_to_one"
    )

    both = teams["opp_fga"].notna() & teams["team_fga_o"].notna()
    if (teams.loc[both, "opp_fga"] != teams.loc[both, "team_fga_o"]).any():
        raise ValueError("opp_fga does not match the opponent's team_fga")

    own_misses = teams["team_fga"] - teams["team_fgm"]
    opp_misses = teams["team_fga_o"] - teams["team_fgm_o"]
    for blocks, misses in (("team_blk", opp_misses), ("team_blk_o", own_misses)):
        both = teams[blocks].notna() & misses.notna()
        if (teams.loc[both, blocks] > misses[both]).any():
            raise ValueError("blocks cannot exceed the opponent's missed field goals")

    definitions = {
        "assisted_rate": (teams["team_ast"], teams["team_fgm"]),
        "p_rebound": (teams["team_oreb"] + teams["opp_dreb"], own_misses),
        "p_oreb": (teams["team_oreb"], teams["team_oreb"] + teams["opp_dreb"]),
        "p_oreb_allowed": (
            teams["team_oreb_o"],
            teams["team_oreb_o"] + teams["opp_dreb_o"],
        ),
        "p_steal": (teams["team_stl"], teams["opp_tov"]),
        "p_steal_against": (teams["team_stl_o"], teams["opp_tov_o"]),
        "p_block": (teams["team_blk"], opp_misses),
        "p_block_against": (teams["team_blk_o"], own_misses),
    }
    for rate, (numer, denom) in definitions.items():
        ok = numer.notna() & denom.notna()
        teams[f"{rate}_n"] = numer.where(ok, 0.0).astype(float)
        teams[f"{rate}_d"] = denom.where(ok, 0.0).astype(float)
    return teams.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)


def _prior_window_sums(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    entity: str,
    columns: list[str],
    window: int | None,
) -> pd.DataFrame:
    """Sums over the last ``window`` rows strictly before each target date.

    ``window`` None keeps every earlier row of that player or team.
    """
    if history.empty:
        return pd.DataFrame(0.0, index=range(len(targets)), columns=columns)
    grouped = history.groupby(entity, sort=False)[columns]
    through = grouped.cumsum()
    if window is None:
        summed = through
    else:
        lagged = through.groupby(history[entity], sort=False).shift(window).fillna(0.0)
        summed = through - lagged
    table = history[[entity, "game_date"]].copy()
    table[columns] = summed.to_numpy()
    merged = pd.merge_asof(
        targets[[entity, "game_date", "_row"]].sort_values("game_date"),
        table.sort_values("game_date"),
        on="game_date",
        by=entity,
        allow_exact_matches=False,
        direction="backward",
    )
    merged = merged.set_index("_row").reindex(targets["_row"])
    return merged[columns].fillna(0.0).reset_index(drop=True)


# Listed position when the log has one. The starting slot is only a fallback:
# a bench center has a blank slot, and power forwards do not block like centers.
_POSITION_ROLES = {
    "PG": "PG",
    "SG": "SG",
    "SF": "SF",
    "PF": "PF",
    "C": "C",
    "G": "G",
    "F": "F",
    "GUARD": "G",
    "FORWARD": "F",
    "CENTER": "C",
    "G-F": "SF",
    "F-G": "SF",
    "F-C": "PF",
    "C-F": "PF",
}
# A starting slot of G or F does not say which of the two listed spots it was.
_COARSE_POOLS = {
    "G": frozenset({"PG", "SG", "G"}),
    "F": frozenset({"SF", "PF", "F"}),
    "C": frozenset({"C"}),
}


def _role_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("pos", "start_position") if column in frame.columns]


def _season_start_year(dates: pd.Series) -> pd.Series:
    """NBA season starts in September. November 2026 is the 2026-27 season."""
    stamps = pd.to_datetime(dates)
    year = stamps.dt.year
    return year.where(stamps.dt.month >= 9, year - 1).astype("int64")


def _map_position(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.upper()
    return text.map(_POSITION_ROLES).fillna("").astype(object)


def _position_roles(frame: pd.DataFrame) -> pd.Series:
    """Listed ``pos`` when present, otherwise that game's starting slot."""
    roles = pd.Series("", index=frame.index, dtype=object)
    if "start_position" in frame.columns:
        roles = _map_position(frame["start_position"])
    if "pos" in frame.columns:
        listed = _map_position(frame["pos"])
        roles = listed.where(listed.ne(""), roles)
    return roles


def _target_roles(played: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """Role known before the target game. A blank slate keeps the prior role."""
    direct = _position_roles(targets).reset_index(drop=True)
    history_roles = _position_roles(played)
    known_history = bool(history_roles.ne("").any())
    if played.empty or not known_history or bool(direct.ne("").all()):
        return direct
    known = played.loc[history_roles.ne(""), ["player_id", "game_date"]].copy()
    known["role"] = history_roles.loc[known.index].to_numpy()
    lookup = targets.loc[:, ["player_id", "game_date", "_row"]].copy()
    merged = pd.merge_asof(
        lookup.sort_values("game_date", kind="mergesort"),
        known.sort_values("game_date", kind="mergesort"),
        on="game_date",
        by="player_id",
        allow_exact_matches=False,
        direction="backward",
    )
    merged = merged.set_index("_row").reindex(targets["_row"])
    inherited = merged["role"].fillna("").reset_index(drop=True)
    return direct.where(direct.ne(""), inherited)


def _prior_role_sums(
    played: pd.DataFrame,
    targets: pd.DataFrame,
    columns: list[str],
    roles: pd.Series,
) -> pd.DataFrame:
    """Season-weighted totals for each target's role. Missing roles stay NaN."""
    history_roles = _position_roles(played).reset_index(drop=True)
    out = pd.DataFrame(np.nan, index=range(len(targets)), columns=columns)
    labels = [label for label in history_roles.unique() if label]
    if not labels:
        return out
    played = played.reset_index(drop=True)
    buckets = {
        label: _prior_league_sums(
            played.loc[history_roles.eq(label)], targets, columns
        )
        for label in labels
    }
    for label in [role for role in roles.unique() if role]:
        members = _COARSE_POOLS.get(label, frozenset({label}))
        parts = [buckets[name] for name in members if name in buckets]
        if not parts:
            continue
        summed = parts[0]
        for part in parts[1:]:
            summed = summed.add(part, fill_value=0.0)
        mask = roles.eq(label).to_numpy()
        out.loc[mask, columns] = summed.loc[mask, columns].to_numpy()
    return out


def _older_season_mass(
    totals: np.ndarray,
    seasons: np.ndarray,
    target_seasons: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Earlier seasons only. The target season is added from its date prefix."""
    out = np.zeros((len(target_seasons), totals.shape[1]), dtype=float)
    if len(seasons) == 0:
        return out
    seasons = np.asarray(seasons, dtype=np.int64)
    target_seasons = np.asarray(target_seasons, dtype=np.int64)
    for year in np.unique(target_seasons):
        lag = int(year) - seasons
        weights = np.zeros(len(seasons), dtype=float)
        earlier = lag > 0
        weights[earlier] = np.power(decay, lag[earlier])
        out[target_seasons == year] = weights @ totals
    return out


def _prior_league_sums(
    history: pd.DataFrame, targets: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Totals before each target date, with the target season at full weight.

    A row from k seasons earlier weighs ``LEAGUE_SEASON_DECAY ** k``. Dates
    on the target date, and every later date, stay out. For a 2026-27 slate
    that season is the heaviest term once it has about 60k minutes.
    """
    if history.empty:
        return pd.DataFrame(np.nan, index=range(len(targets)), columns=columns)
    work = history.loc[:, ["game_date", *columns]].copy()
    work["_season"] = _season_start_year(work["game_date"]).to_numpy()
    daily = (
        work.groupby(["_season", "game_date"], sort=True)[columns]
        .sum()
        .reset_index()
        .sort_values(["_season", "game_date"], kind="mergesort")
    )
    daily[columns] = daily.groupby("_season", sort=False)[columns].cumsum()
    lookup = targets.loc[:, ["game_date", "_row"]].copy()
    lookup["_season"] = _season_start_year(lookup["game_date"]).to_numpy()
    current = pd.merge_asof(
        lookup.sort_values(["_season", "game_date"], kind="mergesort"),
        daily,
        on="game_date",
        by="_season",
        allow_exact_matches=False,
        direction="backward",
    )
    current = current.set_index("_row").reindex(targets["_row"])
    totals = work.groupby("_season", sort=True)[columns].sum()
    older = _older_season_mass(
        totals.to_numpy(dtype=float),
        totals.index.to_numpy(dtype=np.int64),
        current["_season"].to_numpy(dtype=np.int64),
        LEAGUE_SEASON_DECAY,
    )
    combined = current[columns].fillna(0.0).to_numpy(dtype=float) + older
    empty = np.sum(np.abs(combined), axis=1) == 0.0
    combined[empty] = np.nan
    return pd.DataFrame(combined, columns=columns)


def _ratio(numer: Any, denom: Any) -> np.ndarray:
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.full(numer.shape, np.nan)
    ok = np.isfinite(numer) & np.isfinite(denom) & (denom > 0)
    out[ok] = numer[ok] / denom[ok]
    return out


def _shrunk(numer: Any, denom: Any, target: Any, strength: float) -> np.ndarray:
    """``(events + k * target) / (exposure + k)``; raw ratio without a target."""
    numer = np.nan_to_num(np.asarray(numer, dtype=float))
    denom = np.nan_to_num(np.asarray(denom, dtype=float))
    target = np.asarray(target, dtype=float)
    out = np.full(numer.shape, np.nan)
    has_target = np.isfinite(target)
    out[has_target] = (numer[has_target] + strength * target[has_target]) / (
        denom[has_target] + strength
    )
    raw = ~has_target & (denom > 0)
    out[raw] = numer[raw] / denom[raw]
    return out
