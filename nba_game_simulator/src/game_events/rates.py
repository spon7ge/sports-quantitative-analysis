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
    PLAYER_EVENT_RATES,
    PLAYER_PRIOR_MINUTES,
    PLAYER_WINDOW,
    TEAM_EVENT_RATES,
    TEAM_WINDOW,
)
from game_events.logs import _prepare_event_logs, _prepare_event_slate
from game_events.model import EventModel


def training_events(
    logs: pd.DataFrame, model: EventModel | None = None
) -> pd.DataFrame:
    """One player-game. Event rates stop before that game's date.

    The model supplies the selected windows and prior strengths; without one
    the defaults are used.
    """
    frame = _prepare_event_logs(logs)
    targets = frame.loc[:, list(_EVENT_IDS)].reset_index(drop=True)
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
    targets = upcoming.loc[:, ["player_id", "game_date", "team_id"]]
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
    targets = rows[[entity, "game_date"]].reset_index(drop=True).copy()
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
    played: pd.DataFrame, targets: pd.DataFrame, window: int, prior_minutes: float
) -> pd.DataFrame:
    window = _positive_int(window, "player_window")
    prior_minutes = _positive_float(prior_minutes, "player_prior_minutes")
    columns = [f"{rate}_{part}" for rate in PLAYER_EVENT_RATES for part in "nd"]
    prior = _prior_window_sums(played, targets, "player_id", columns, window)
    league = _prior_league_sums(played, targets, columns)
    out = pd.DataFrame(index=range(len(targets)))
    for rate in PLAYER_EVENT_RATES:
        out[rate] = _shrunk(
            prior[f"{rate}_n"],
            prior[f"{rate}_d"],
            _ratio(league[f"{rate}_n"], league[f"{rate}_d"]),
            prior_minutes,
        )
    return out


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
    window: int,
) -> pd.DataFrame:
    """Sums over the last ``window`` rows strictly before each target date."""
    if history.empty:
        return pd.DataFrame(0.0, index=range(len(targets)), columns=columns)
    grouped = history.groupby(entity, sort=False)[columns]
    through = grouped.cumsum()
    lagged = through.groupby(history[entity], sort=False).shift(window).fillna(0.0)
    table = history[[entity, "game_date"]].copy()
    table[columns] = (through - lagged).to_numpy()
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


def _prior_league_sums(
    history: pd.DataFrame, targets: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """League totals over every date strictly before each target date."""
    if history.empty:
        return pd.DataFrame(np.nan, index=range(len(targets)), columns=columns)
    daily = (
        history.groupby("game_date")[columns].sum().sort_index().cumsum().reset_index()
    )
    merged = pd.merge_asof(
        targets[["game_date", "_row"]].sort_values("game_date"),
        daily,
        on="game_date",
        allow_exact_matches=False,
        direction="backward",
    )
    merged = merged.set_index("_row").reindex(targets["_row"])
    return merged[columns].reset_index(drop=True)


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
