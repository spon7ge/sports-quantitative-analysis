"""Prior-only rotation minutes and the ratings those minutes weight.

Historical lineups come from start_position: G, F, or C started, and a
blank came off the bench. Minutes and ratings still use earlier games only.
There is no injury snapshot, so a player missing from the log is not marked
out. GTD participation is an assumption unless a caller passes a rate that
was estimated from timestamped snapshots and later participation.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

REGULATION_BUDGET = 240.0
PLAYER_CAP = 48.0
OT_TEAM_MINUTES = 25.0
OT_PLAYER_CAP = 5.0
DEFAULT_GTD_PROBABILITY = 0.5
GTD_PROBABILITY_SOURCE = "assumption"
STARTER_PRIOR_MINUTES = 32.0
BENCH_PRIOR_MINUTES = 14.0
AVAILABILITY_SOURCE = "prior_only"

_OUT = {"out", "o", "inactive", "ofs", "suspended"}
_GTD = {
    "gtd",
    "questionable",
    "doubtful",
    "ques",
    "q",
    "toss up",
    "toss up to play",
}
_AVAILABLE = {
    "available",
    "confirmed",
    "active",
    "probable",
    "very likely to play",
    "likely to play",
}
_STARTER_POSITIONS = {"G", "F", "C"}
_RATE_STATS = ("pace", "off_rating", "def_rating")
_LINEUP_ALIAS = {
    "rotation_pace_h": "lineup_pace_h",
    "rotation_pace_a": "lineup_pace_a",
    "rotation_off_rating_h": "lineup_off_rating_h",
    "rotation_off_rating_a": "lineup_off_rating_a",
    "rotation_def_rating_h": "lineup_def_rating_h",
    "rotation_def_rating_a": "lineup_def_rating_a",
}


def classify_status(value) -> str:
    """Map a scraper status onto out, available, gtd, or unknown."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip().casefold()
    if text in {"", "nan", "none", "unknown", "<na>"}:
        return "unknown"
    if text in _OUT:
        return "out"
    if text in _GTD:
        return "gtd"
    if text in _AVAILABLE:
        return "available"
    return "unknown"


def allocate_regulation_minutes(preferred, eligible, locked=None):
    """Scale preferred minutes onto 240, capped at 48.

    Fewer than five eligible players is an invalid rotation: the returned
    minutes stay at zero instead of a silent four-man split.
    Confirmed starters in ``locked`` keep their preferred minutes when the
    remaining players can still absorb the rest of the budget.
    """
    preferred = np.asarray(preferred, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if locked is None:
        locked = np.zeros(len(preferred), dtype=bool)
    else:
        locked = np.asarray(locked, dtype=bool) & eligible
    minutes = np.zeros(len(preferred), dtype=float)
    info = {"valid": int(eligible.sum()) >= 5, "starters_held": False}
    if not info["valid"]:
        return minutes, info

    base = np.where(
        eligible & np.isfinite(preferred),
        np.clip(preferred, 0.0, PLAYER_CAP),
        0.0,
    )
    free = eligible & ~locked
    if locked.any():
        held = base[locked]
        budget_left = REGULATION_BUDGET - float(held.sum())
        room = PLAYER_CAP * int(free.sum())
        if budget_left >= -1e-8 and budget_left <= room + 1e-8:
            minutes[locked] = held
            if free.any() and budget_left > 1e-8:
                minutes[free] = _distribute(base[free], budget_left, PLAYER_CAP)
            info["starters_held"] = True
            return minutes, info

    minutes[eligible] = _distribute(base[eligible], REGULATION_BUDGET, PLAYER_CAP)
    return minutes, info


def choose_starters(players: pd.DataFrame) -> list[int]:
    """Keep available confirmed starters, then replace an out starter.

    The replacement is the eligible bench player at the same position with
    the most preferred minutes. A higher-minute player at another position
    does not jump that match.
    """
    frame = players.copy()
    frame["kind"] = frame["status"].map(classify_status)
    available = frame["kind"].ne("out")
    chosen: list[int] = []
    confirmed = frame.loc[
        available
        & frame["starter_confirmed"].astype(bool)
        & frame["is_starter"].astype(bool)
    ]
    for pid in confirmed["player_id"]:
        chosen.append(int(pid))
        if len(chosen) == 5:
            return chosen

    projected = frame.loc[
        available
        & frame["is_starter"].astype(bool)
        & ~frame["player_id"].isin(chosen)
    ]
    for pid in projected["player_id"]:
        chosen.append(int(pid))
        if len(chosen) == 5:
            return chosen

    missing = frame.loc[
        frame["is_starter"].astype(bool) & frame["kind"].eq("out"), "position"
    ]
    bench = frame.loc[available & ~frame["player_id"].isin(chosen)].copy()
    for position in missing:
        if len(chosen) >= 5 or bench.empty:
            break
        group = _position_group(position)
        match = bench.loc[bench["position"].map(_position_group).eq(group)]
        pool = match if not match.empty else bench
        pool = pool.sort_values(
            ["preferred_minutes", "player_id"],
            ascending=[False, True],
        )
        pid = int(pool["player_id"].iloc[0])
        chosen.append(pid)
        bench = bench.loc[~bench["player_id"].eq(pid)]
    return chosen[:5]


def overtime_player_minutes(reg_minutes, n_ot) -> np.ndarray:
    """Add 25 team minutes per overtime period on top of regulation minutes.

    Regulation minutes are unchanged. Each player can take at most five
    minutes of a single overtime period.
    """
    reg = np.asarray(reg_minutes, dtype=float)
    periods = int(n_ot)
    if periods <= 0:
        return reg.copy()
    weights = np.where(np.isfinite(reg) & (reg > 0), reg, 0.0)
    extra = _distribute(
        weights,
        OT_TEAM_MINUTES * periods,
        OT_PLAYER_CAP * periods,
    )
    return reg + extra


def sample_rotation(players, n, rng, gtd_probability=DEFAULT_GTD_PROBABILITY):
    """One availability draw per simulation, shared by both teams.

    ``gtd_probability`` is not estimated here. Callers that did not fit it
    on timestamped snapshots should leave the default and label it as an
    assumption.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(gtd_probability) or not 0.0 <= float(gtd_probability) <= 1.0:
        raise ValueError("gtd_probability must be between 0 and 1")

    frame = players.copy().reset_index(drop=True)
    if "side" not in frame.columns or set(frame["side"]) != {"home", "away"}:
        raise ValueError("rotation players need a home side and an away side")
    frame["kind"] = frame["status"].map(classify_status)
    gtd_order = np.flatnonzero(frame["kind"].eq("gtd").to_numpy())
    gtd_order = gtd_order[np.argsort(frame.loc[gtd_order, "player_id"].to_numpy())]
    player_rows = []
    feature_rows = []
    for sim in range(int(n)):
        resolved = frame["kind"].tolist()
        if len(gtd_order):
            draws = np.asarray(rng.random(len(gtd_order)), dtype=float)
            for index, draw in zip(gtd_order, draws, strict=True):
                resolved[index] = (
                    "available" if draw < float(gtd_probability) else "out"
                )
        pieces = []
        features = {
            "gtd_probability": float(gtd_probability),
            "gtd_probability_source": GTD_PROBABILITY_SOURCE,
        }
        for side in ("home", "away"):
            side_index = np.flatnonzero(frame["side"].eq(side).to_numpy())
            block = frame.loc[side_index].copy()
            block["resolved_status"] = [resolved[i] for i in side_index]
            block["sim"] = sim
            eligible = block["resolved_status"].isin(["available", "unknown"]).to_numpy()
            locked = (
                block["starter_confirmed"].astype(bool)
                & block["is_starter"].astype(bool)
                & eligible
            )
            minutes, info = allocate_regulation_minutes(
                block["preferred_minutes"].to_numpy(float),
                eligible,
                locked,
            )
            block["projected_reg_minutes"] = minutes
            block["rotation_valid"] = bool(info["valid"])
            chosen = set(choose_starters(_resolved_frame(block)))
            block["is_expected_starter"] = block["player_id"].map(
                lambda pid: int(pid) in chosen
            )
            pieces.append(block)
            ratings = _side_features(block, info["valid"])
            for stat, value in ratings.items():
                features[f"rotation_{stat}_{side[0]}"] = value
        player_rows.append(pd.concat(pieces, ignore_index=True))
        feature_rows.append(features)
    return pd.concat(player_rows, ignore_index=True), feature_rows


def lineup_overrides(features: dict) -> dict:
    """Copy rotation ratings into the lineup columns the scorer already reads."""
    out = dict(features)
    for source, target in _LINEUP_ALIAS.items():
        if source in features and features[source] is not None:
            out[target] = features[source]
    return out


def build_rotation_history(
    logs: pd.DataFrame,
    halflife: float = 8.0,
    prior_weight: float = 4.0,
    roster_window: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project each logged lineup from earlier games only.

    ``start_position`` of G, F, or C is a starter. A blank is the bench.
    That label is the historical lineup. Minutes and ratings still come
    from earlier games; this game's minutes are stored only after the
    projection. ``roster_window`` remains accepted so older callers do not
    break; the lineup itself is the logged ``start_position``.
    """
    if halflife <= 0 or prior_weight < 0 or roster_window < 1:
        raise ValueError("halflife, prior_weight, and roster_window are out of range")
    prepared = _prepare_logs(logs)
    decay = 0.5 ** (1.0 / float(halflife))
    players: dict = {}
    league = {stat: deque(maxlen=300) for stat in _RATE_STATS}
    team_hist = defaultdict(lambda: {stat: deque(maxlen=300) for stat in _RATE_STATS})
    player_rows: list[dict] = []
    team_rows: list[dict] = []

    for date, day in prepared.groupby("game_date", sort=True):
        league_now = {
            stat: float(np.mean(league[stat])) if league[stat] else np.nan
            for stat in _RATE_STATS
        }
        for (game_id, team_id), team_day in day.groupby(
            ["game_id", "team_id"], sort=False
        ):
            team_id = int(team_id)
            center = {
                stat: (
                    float(np.mean(team_hist[team_id][stat]))
                    if team_hist[team_id][stat]
                    else league_now[stat]
                )
                for stat in _RATE_STATS
            }
            lineup = [
                {
                    "player_id": int(rec.player_id),
                    "started": _is_listed_starter(rec.start_position),
                    "position": _position_group(rec.start_position),
                }
                for rec in team_day.itertuples(index=False)
            ]
            projection = _project_team(
                lineup,
                players,
                team_id,
                center,
                prior_weight,
            )
            for row in projection["players"]:
                row["game_id"] = str(game_id)
                row["game_date"] = date
            projection["team"]["game_id"] = str(game_id)
            projection["team"]["game_date"] = date
            player_rows.extend(projection["players"])
            team_rows.append(projection["team"])

        for (_game_id, team_id), team_day in day.groupby(
            ["game_id", "team_id"], sort=False
        ):
            team_id = int(team_id)
            rate_values = {stat: [] for stat in _RATE_STATS}
            for rec in team_day.itertuples(index=False):
                pid = int(rec.player_id)
                state = players.setdefault((team_id, pid), _new_player())
                minutes = float(rec.min) if np.isfinite(rec.min) else np.nan
                if np.isfinite(minutes):
                    state["min_num"] = state["min_num"] * decay + minutes
                    state["min_den"] = state["min_den"] * decay + 1.0
                    state["role"] = (
                        "starter" if _is_listed_starter(rec.start_position) else "bench"
                    )
                    state["position"] = _position_group(rec.start_position)
                for stat in _RATE_STATS:
                    value = float(getattr(rec, stat))
                    if not np.isfinite(value):
                        continue
                    state["rate_num"][stat] = state["rate_num"][stat] * decay + value
                    state["rate_den"][stat] = state["rate_den"][stat] * decay + 1.0
                    rate_values[stat].append(value)
            for stat in _RATE_STATS:
                if rate_values[stat]:
                    value = float(np.mean(rate_values[stat]))
                    team_hist[team_id][stat].append(value)
                    league[stat].append(value)

    player_frame = pd.DataFrame(player_rows)
    team_frame = pd.DataFrame(team_rows)
    if player_frame.empty:
        player_frame = pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "team_id",
                "player_id",
                "projected_reg_minutes",
                "status",
                "is_expected_starter",
            ]
        )
    return player_frame, team_frame


def _project_team(lineup, players, team_id, center, prior_weight):
    ids = []
    preferred = []
    positions = []
    roles = []
    started_flags = []
    ratings = {stat: [] for stat in _RATE_STATS}
    for item in lineup:
        pid = int(item["player_id"])
        started = bool(item["started"])
        state = players.get((team_id, pid))
        role = "starter" if started else "bench"
        prior = STARTER_PRIOR_MINUTES if started else BENCH_PRIOR_MINUTES
        if state is None:
            num = den = 0.0
            rate_num = {stat: 0.0 for stat in _RATE_STATS}
            rate_den = {stat: 0.0 for stat in _RATE_STATS}
            position = item["position"]
        else:
            num = state["min_num"]
            den = state["min_den"]
            rate_num = state["rate_num"]
            rate_den = state["rate_den"]
            position = item["position"] or state["position"]
        ids.append(pid)
        started_flags.append(started)
        preferred.append(_shrunk(num, den, prior, prior_weight))
        positions.append(position)
        roles.append(role)
        for stat in _RATE_STATS:
            ratings[stat].append(
                _shrunk(rate_num[stat], rate_den[stat], center[stat], prior_weight)
            )

    preferred_arr = np.asarray(preferred, dtype=float)
    for stat in _RATE_STATS:
        column = np.asarray(ratings[stat], dtype=float)
        if np.isfinite(center[stat]):
            column = np.where(np.isfinite(column), column, float(center[stat]))
        ratings[stat] = column
    eligible_mask = np.ones(len(ids), dtype=bool)
    locked = np.asarray(started_flags, dtype=bool)
    minutes, info = allocate_regulation_minutes(
        preferred_arr, eligible_mask, locked
    )
    starter_ids = {pid for pid, started in zip(ids, started_flags, strict=True) if started}

    player_rows = []
    for i, pid in enumerate(ids):
        player_rows.append(
            {
                "team_id": team_id,
                "player_id": pid,
                "projected_reg_minutes": float(minutes[i]),
                "preferred_minutes": float(preferred_arr[i]),
                "position": positions[i],
                "role": roles[i],
                "status": "unknown",
                "availability_source": AVAILABILITY_SOURCE,
                "is_expected_starter": pid in starter_ids,
                "pace": ratings["pace"][i],
                "off_rating": ratings["off_rating"][i],
                "def_rating": ratings["def_rating"][i],
            }
        )
    if info["valid"] and minutes.sum() > 0:
        pace = _weighted(minutes, ratings["pace"])
        offense = _weighted(minutes, ratings["off_rating"])
        defense = _weighted(minutes, ratings["def_rating"])
    else:
        pace = offense = defense = np.nan
    team_row = {
        "team_id": team_id,
        "rotation_pace": pace,
        "rotation_off_rating": offense,
        "rotation_def_rating": defense,
        "n_eligible": int(len(ids)),
        "rotation_valid": bool(info["valid"]),
        "availability_source": AVAILABILITY_SOURCE,
        "expected_starters": ",".join(str(pid) for pid in ids if pid in starter_ids),
    }
    # game_id and game_date are filled by the caller after this returns.
    for row in player_rows:
        row["game_id"] = None
        row["game_date"] = None
    return {"players": player_rows, "team": team_row}


def _side_features(block: pd.DataFrame, valid: bool) -> dict[str, float]:
    if not valid:
        return {stat: np.nan for stat in _RATE_STATS}
    minutes = block["projected_reg_minutes"].to_numpy(float)
    return {
        stat: _weighted(minutes, block[stat].to_numpy(float))
        for stat in _RATE_STATS
    }


def _resolved_frame(block: pd.DataFrame) -> pd.DataFrame:
    """Starter replacement sees the sampled status, not the pre-draw label."""
    out = block.copy()
    out["status"] = out["resolved_status"]
    out["is_starter"] = out["is_starter"].astype(bool) & out["resolved_status"].ne("out")
    # An out confirmed starter is no longer a starter; the bench row is not.
    # choose_starters also needs the original starter flag to know who to replace.
    out["is_starter"] = block["is_starter"].astype(bool)
    out["status"] = block["resolved_status"]
    return out


def _distribute(weights, budget, cap) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    out = np.zeros(len(weights), dtype=float)
    if len(weights) == 0 or budget <= 1e-10:
        return out
    if weights.sum() <= 0:
        weights = np.ones(len(weights))
    out = weights / weights.sum() * float(budget)
    room_left = np.full(len(weights), cap, dtype=float)
    for _ in range(40):
        out = np.minimum(out, room_left)
        short = float(budget) - float(out.sum())
        if short <= 1e-8:
            break
        room = room_left - out
        free = room > 1e-8
        if not np.any(free):
            break
        share = np.where(free, weights, 0.0)
        if share.sum() <= 0:
            share = free.astype(float)
        add = np.minimum(short * share / share.sum(), room)
        out = out + add
    return out


def _weighted(minutes, values) -> float:
    minutes = np.asarray(minutes, dtype=float)
    values = np.asarray(values, dtype=float)
    # Missing ratings fall out of the average. Weights are still the
    # regulation shares, so a full rotation divides by 240.
    ok = np.isfinite(minutes) & np.isfinite(values) & (minutes > 0)
    if not ok.any():
        return np.nan
    return float(np.sum(values[ok] * minutes[ok] / REGULATION_BUDGET))


def _shrunk(num, den, center, prior_weight) -> float:
    if den <= 0:
        return float(center) if np.isfinite(center) else np.nan
    if not np.isfinite(center) or prior_weight == 0:
        return float(num) / float(den)
    return (float(num) + prior_weight * float(center)) / (float(den) + prior_weight)


def _new_player() -> dict:
    return {
        "min_num": 0.0,
        "min_den": 0.0,
        "rate_num": {stat: 0.0 for stat in _RATE_STATS},
        "rate_den": {stat: 0.0 for stat in _RATE_STATS},
        "role": "bench",
        "position": "",
    }


def _is_listed_starter(value) -> bool:
    """G, F, or C in start_position is the historical starting lineup."""
    return str(value or "").strip().upper() in _STARTER_POSITIONS


def _position_group(value) -> str:
    text = str(value or "").strip().upper()
    if text in {"C", "CENTER"}:
        return "C"
    if text in {"G", "PG", "SG", "GUARD"}:
        return "G"
    if text in {"F", "SF", "PF", "FORWARD"}:
        return "F"
    return ""


def _prepare_logs(logs: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "game_date",
        "team_id",
        "player_id",
        "min",
        "start_position",
        *_RATE_STATS,
    }
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"rotation history needs {sorted(missing)}")
    out = logs.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    out["game_id"] = out["game_id"].astype(str)
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out["min"] = pd.to_numeric(out["min"], errors="coerce")
    for stat in _RATE_STATS:
        out[stat] = pd.to_numeric(out[stat], errors="coerce")
    out = out.dropna(subset=["game_date", "game_id", "team_id", "player_id"])
    out["team_id"] = out["team_id"].astype(int)
    out["player_id"] = out["player_id"].astype(int)
    out["start_position"] = out["start_position"].astype("string").fillna("")
    return (
        out.drop_duplicates(["game_id", "team_id", "player_id"])
        .sort_values(["game_date", "game_id", "team_id", "player_id"])
        .reset_index(drop=True)
    )
