"""Pace and overtime targets counted from the play-by-play.

The player logs store a box estimate, FGA + 0.44·FTA − OREB + TOV. That
estimate is not a trip down the floor. Possession rows are. A lead changes
the length of a trip, and the two halves do not run at the same rate, so the
draw looks up a pace multiplier by half and margin. Overtime is the chance
the regulation margin reaches zero. The chance is counted from the margin
with six and two minutes left and from the final score.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from possession_table import REQUIRED_COLUMNS, _game_id, possession_table

BOX_FTA_WEIGHT = 0.44
PACE_EDGES = (0.0, 1.0, 4.0, 8.0, 13.0, 21.0, 1.0e9)
# A thin bucket borrows duration from the tied trips in that half.
PACE_PRIOR = 200.0
OT_PRIOR = 40.0
OT_MIN_GAMES = 20
PACE_MIN_TIED = 20
# Same screen build_game_table uses. A parsed game outside it is not a pace.
POSS_PACE_MIN = 70.0
POSS_PACE_MAX = 130.0
# Six minutes left is halfway through the second half. Two minutes left is
# ten of those twelve minutes.
LATE_CLOCKS = (("margin_6", 360.0), ("margin_2", 120.0))
_CLOCK = re.compile(r"PT(?:(\d+(?:\.\d+)?)M)?(\d+(?:\.\d+)?)S$", re.IGNORECASE)
_COUNT_COLUMNS = ("game_id", "team_id", "poss")
_EXPOSURE_COLUMNS = ("game_id", "half", "offense_margin", "seconds")
_PATH_COLUMNS = (
    "game_id",
    "margin_6",
    "margin_2",
    "reg_margin",
    "final_margin",
    "went_ot",
)


def box_possession_estimate(fga, fta, oreb, tov):
    """Box estimate the play-by-play count replaces."""

    return (
        np.asarray(fga, dtype=float)
        + BOX_FTA_WEIGHT * np.asarray(fta, dtype=float)
        - np.asarray(oreb, dtype=float)
        + np.asarray(tov, dtype=float)
    )


def margin_bucket(margin, edges):
    """Absolute-margin bucket. Zero is tied; larger indexes are bigger leads."""

    bounds = np.asarray(edges, dtype=float)
    index = np.searchsorted(
        bounds, np.abs(np.asarray(margin, dtype=float)), side="right"
    ) - 1
    return np.clip(index, 0, len(bounds) - 2)


def late_checkpoints(h1_margin, regulation_margin):
    """Margin at 6:00 and 2:00 of the fourth, on a straight second half."""

    opening = np.asarray(h1_margin, dtype=float)
    change = np.asarray(regulation_margin, dtype=float) - opening
    return opening + 0.5 * change, opening + (10.0 / 12.0) * change


def script_tables(pbp, starters=None):
    """Possession counts, trip lengths, and late margins for one feed."""

    frame = _require_pbp(pbp)
    counts, exposure, failed = [], [], []
    grouped = list(frame.groupby("gameId", sort=False))
    for index, (game_id, game) in enumerate(grouped, start=1):
        if index % 500 == 0:
            print(f"  possessions {index:,}/{len(grouped):,}")
        label = _game_id(game_id)
        try:
            table = possession_table(game, starters)
        except (ValueError, KeyError, TypeError) as exc:
            failed.append({"game_id": label, "error": str(exc)})
            continue
        if table.empty:
            failed.append({"game_id": label, "error": "no possessions"})
            continue
        counts.append(_count_game(table))
        exposure.append(_exposure_game(table))

    if not counts:
        raise ValueError("No games produced possession counts")
    if failed:
        print(f"  WARNING: skipped {len(failed)} play-by-play games with no possession count")

    return {
        "counts": pd.concat(counts, ignore_index=True),
        "exposure": pd.concat(exposure, ignore_index=True),
        "paths": late_margin_paths(frame),
        "failed": failed,
    }


def apply_possession_counts(games, counts):
    """Replace team possessions where the play-by-play count is a real pace.

    A game with one team missing, or a per-48 rate outside the usual screen,
    keeps the logged box estimate.
    """

    required = {
        "game_id", "home_id", "away_id", "home_poss", "away_poss", "nominal_min",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games are missing {sorted(missing)}")
    if (pd.to_numeric(games["nominal_min"], errors="coerce") <= 0).any():
        raise ValueError("nominal_min must be positive")
    counted = _checked_counts(counts)

    out = games.copy()
    out["_gid"] = out["game_id"].map(_game_id)
    for side in ("home_id", "away_id"):
        out[side] = pd.to_numeric(out[side], errors="coerce")
    if out[["home_id", "away_id"]].isna().any().any():
        raise ValueError("home_id and away_id must be numeric")
    for side in ("home_id", "away_id"):
        out[side] = out[side].astype("int64")
    home = counted.rename(columns={"team_id": "home_id", "poss": "_home_poss"})
    away = counted.rename(columns={"team_id": "away_id", "poss": "_away_poss"})
    out = out.merge(
        home,
        left_on=["_gid", "home_id"],
        right_on=["game_id", "home_id"],
        how="left",
        suffixes=("", "_home"),
    )
    out = out.merge(
        away,
        left_on=["_gid", "away_id"],
        right_on=["game_id", "away_id"],
        how="left",
        suffixes=("", "_away"),
    )
    out = out.drop(columns=[c for c in out.columns if c.startswith("game_id_")])

    both = out["_home_poss"].gt(0) & out["_away_poss"].gt(0)
    pace = (out["_home_poss"] + out["_away_poss"]) / 2.0
    pace = pace * 48.0 / pd.to_numeric(out["nominal_min"], errors="coerce")
    keep = both & pace.between(POSS_PACE_MIN, POSS_PACE_MAX)
    out["poss_source"] = np.where(keep, "pbp", "box")
    out.loc[keep, "home_poss"] = out.loc[keep, "_home_poss"]
    out.loc[keep, "away_poss"] = out.loc[keep, "_away_poss"]
    out.loc[keep, "poss"] = (
        out.loc[keep, "home_poss"] + out.loc[keep, "away_poss"]
    ) / 2.0
    if "poss48" in out.columns or keep.any():
        nominal = pd.to_numeric(out["nominal_min"], errors="coerce")
        out.loc[keep, "poss48"] = out.loc[keep, "poss"] * 48.0 / nominal.loc[keep]

    print(
        f"  pace target: {int(keep.sum()):,} games from play-by-play, "
        f"{int((~keep).sum()):,} kept the box estimate"
    )
    return out.drop(columns=["_gid", "_home_poss", "_away_poss"]).reset_index(drop=True)


def fit_pace_state(exposure, game_ids=None):
    """Multipliers by half and absolute margin, plus who holds the ball.

    The tied first-half rate and the tied second-half rate average to 1, so a
    tied game keeps the pregame pace. A lead bucket is slower when the
    leader's extra seconds outweigh the trailer's shorter trips.
    """

    frame = _filter_games(_exposure_frame(exposure), game_ids)
    edges = list(PACE_EDGES)
    buckets = margin_bucket(frame["offense_margin"], edges)
    n_buckets = len(edges) - 1
    seconds = np.zeros((2, n_buckets))
    counts = np.zeros((2, n_buckets))
    lead_n = np.zeros(n_buckets)
    trail_n = np.zeros(n_buckets)
    lead_sum = np.zeros(n_buckets)
    trail_sum = np.zeros(n_buckets)

    for half in (1, 2):
        part = frame["half"].eq(half).to_numpy()
        duration = frame.loc[part, "seconds"].to_numpy(float)
        bucket = np.asarray(buckets)[part]
        sign = np.sign(frame.loc[part, "offense_margin"].to_numpy(float))
        for index in range(n_buckets):
            chosen = bucket == index
            counts[half - 1, index] = int(chosen.sum())
            if chosen.any():
                seconds[half - 1, index] = float(duration[chosen].mean())
        if half == 2:
            for index in range(1, n_buckets):
                lead = (bucket == index) & (sign > 0)
                trail = (bucket == index) & (sign < 0)
                lead_n[index] = int(lead.sum())
                trail_n[index] = int(trail.sum())
                if lead.any():
                    lead_sum[index] = float(duration[lead].sum())
                if trail.any():
                    trail_sum[index] = float(duration[trail].sum())

    for half in (1, 2):
        tied_n = counts[half - 1, 0]
        tied = seconds[half - 1, 0]
        if tied_n < PACE_MIN_TIED or not np.isfinite(tied) or tied <= 0:
            raise ValueError(
                f"Need at least {PACE_MIN_TIED} tied possessions in half {half}"
            )
        for index in range(n_buckets):
            observed = counts[half - 1, index]
            mean = seconds[half - 1, index] if observed else tied
            seconds[half - 1, index] = (
                observed * mean + PACE_PRIOR * tied
            ) / (observed + PACE_PRIOR)

    rate = 60.0 / seconds
    neutral = 0.5 * (rate[0, 0] + rate[1, 0])
    if not np.isfinite(neutral) or neutral <= 0:
        raise ValueError("Tied pace is not a positive rate")
    tied_h2 = seconds[1, 0]
    lead_seconds = []
    trail_seconds = []
    for index in range(n_buckets):
        lead_seconds.append(
            None if lead_n[index] == 0
            else float(
                (lead_sum[index] + PACE_PRIOR * tied_h2)
                / (lead_n[index] + PACE_PRIOR)
            )
        )
        trail_seconds.append(
            None if trail_n[index] == 0
            else float(
                (trail_sum[index] + PACE_PRIOR * tied_h2)
                / (trail_n[index] + PACE_PRIOR)
            )
        )

    return {
        "edges": edges,
        "multiplier": (rate / neutral).tolist(),
        "seconds": seconds.tolist(),
        "counts": counts.tolist(),
        "lead_seconds": lead_seconds,
        "trail_seconds": trail_seconds,
        "neutral": float(neutral),
        "prior": PACE_PRIOR,
    }


def fit_ot_path(paths, game_ids=None, prior=OT_PRIOR):
    """P(OT) by the late-margin cell and by the final-score bucket."""

    frame = _filter_games(_path_frame(paths), game_ids)
    if len(frame) < OT_MIN_GAMES:
        raise ValueError(
            f"Need at least {OT_MIN_GAMES} games with late margins to calibrate overtime"
        )
    if prior < 0:
        raise ValueError("overtime prior must be nonnegative")

    edges = list(PACE_EDGES)
    outcome = frame["went_ot"].to_numpy(float)
    overall = float(outcome.mean())
    late_6 = margin_bucket(frame["margin_6"], edges)
    late_2 = margin_bucket(frame["margin_2"], edges)
    final_bucket = margin_bucket(frame["final_margin"], edges)
    n_buckets = len(edges) - 1
    trajectory = np.empty((n_buckets, n_buckets))
    trajectory_n = np.zeros((n_buckets, n_buckets))
    final = np.empty(n_buckets)
    final_n = np.zeros(n_buckets)

    for left in range(n_buckets):
        for right in range(n_buckets):
            chosen = (late_6 == left) & (late_2 == right)
            trajectory_n[left, right] = int(chosen.sum())
            trajectory[left, right] = _shrunk(outcome[chosen], overall, prior)
        chosen = final_bucket == left
        final_n[left] = int(chosen.sum())
        final[left] = _shrunk(outcome[chosen], overall, prior)

    return {
        "edges": edges,
        "overall": overall,
        "trajectory": trajectory.tolist(),
        "trajectory_n": trajectory_n.tolist(),
        "final": final.tolist(),
        "final_n": final_n.tolist(),
        "blend": 0.5,
        "prior": float(prior),
    }


def choose_ot_blend(model, paths, game_ids):
    """Weight on the late path. The rest is the final-score table."""

    frame = _filter_games(_path_frame(paths), game_ids)
    if len(frame) < OT_MIN_GAMES:
        raise ValueError(
            f"Need at least {OT_MIN_GAMES} validation games to weight overtime"
        )
    outcome = frame["went_ot"].to_numpy(float)
    path = ot_path_probability(
        {**model, "blend": 1.0},
        frame["margin_6"],
        frame["margin_2"],
        frame["final_margin"],
    )
    final = ot_path_probability(
        {**model, "blend": 0.0},
        frame["margin_6"],
        frame["margin_2"],
        frame["final_margin"],
    )
    best_weight, best = 0.0, np.inf
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        brier = float(np.mean((weight * path + (1.0 - weight) * final - outcome) ** 2))
        if brier < best:
            best, best_weight = brier, weight
    return float(best_weight)


def ot_path_probability(model, margin_6, margin_2, final_margin):
    """Blend the late-margin cell with the final-score bucket."""

    weight = float(model["blend"])
    if not 0.0 <= weight <= 1.0:
        raise ValueError("overtime blend must be between 0 and 1")
    edges = model["edges"]
    trajectory = np.asarray(model["trajectory"], dtype=float)
    final = np.asarray(model["final"], dtype=float)
    return (
        weight * trajectory[margin_bucket(margin_6, edges), margin_bucket(margin_2, edges)]
        + (1.0 - weight) * final[margin_bucket(final_margin, edges)]
    )


def late_margin_paths(pbp):
    """Home margin at 6:00 and 2:00 of the fourth, plus the final score."""

    frame = _require_pbp(pbp, extra=("clock",))
    frame = frame.copy()
    frame["game_id"] = frame["gameId"].map(_game_id)
    frame["action"] = pd.to_numeric(frame["actionNumber"], errors="coerce")
    frame["period"] = pd.to_numeric(frame["period"], errors="coerce")
    frame["home"] = pd.to_numeric(frame["scoreHome"], errors="coerce")
    frame["away"] = pd.to_numeric(frame["scoreAway"], errors="coerce")
    rows = []
    usable = frame.dropna(subset=["game_id", "period"])
    for game_id, game in usable.groupby("game_id", sort=False):
        ordered = game.sort_values("action")
        record = {
            "game_id": game_id,
            "margin_6": np.nan,
            "margin_2": np.nan,
            "reg_margin": np.nan,
            "final_margin": np.nan,
            "went_ot": 0,
        }
        margin = 0.0
        scored = False
        for row in ordered.itertuples(index=False):
            period = int(row.period)
            clock = _clock_seconds(row.clock)
            if period == 4 and clock is not None:
                for name, mark in LATE_CLOCKS:
                    if clock <= mark and np.isnan(record[name]):
                        record[name] = margin
            if np.isfinite(row.home) and np.isfinite(row.away):
                margin = float(row.home) - float(row.away)
                scored = True
            if scored and period <= 4:
                record["reg_margin"] = margin
            if scored:
                record["final_margin"] = margin
            if period >= 5:
                record["went_ot"] = 1
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=list(_PATH_COLUMNS))
    return pd.DataFrame(rows, columns=list(_PATH_COLUMNS))


def load_play_by_play(folder):
    """Read every season file and count it. Missing folders return None."""

    if not folder:
        return None
    root = Path(folder)
    if not root.exists():
        return None
    paths = sorted(root.glob("s*_pbp.parquet"))
    if not paths:
        return None

    import pyarrow.parquet as pq

    frames = []
    for path in paths:
        present = set(pq.read_schema(path).names)
        missing = [name for name in REQUIRED_COLUMNS if name not in present]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        columns = list(REQUIRED_COLUMNS)
        if "playerNameI" in present:
            columns.append("playerNameI")
        frames.append(pd.read_parquet(path, columns=columns))
        print(f"  play-by-play {path.name}: {len(frames[-1]):,} rows")
    print("Counting possessions")
    return script_tables(pd.concat(frames, ignore_index=True))


def _shrunk(outcome, overall, prior):
    total = float(np.asarray(outcome).sum()) if len(outcome) else 0.0
    return (total + prior * overall) / (len(outcome) + prior)


def _checked_counts(counts):
    missing = [name for name in _COUNT_COLUMNS if name not in counts.columns]
    if missing:
        raise ValueError(f"possession counts are missing {missing}")
    counted = counts.loc[:, list(_COUNT_COLUMNS)].copy()
    counted["game_id"] = counted["game_id"].map(_game_id)
    counted["team_id"] = pd.to_numeric(counted["team_id"], errors="coerce")
    counted["poss"] = pd.to_numeric(counted["poss"], errors="coerce")
    if counted[["team_id", "poss"]].isna().any().any():
        raise ValueError("possession counts must be numeric")
    counted["team_id"] = counted["team_id"].astype("int64")
    if counted.duplicated(["game_id", "team_id"]).any():
        raise ValueError("possession counts repeat a team in a game")
    return counted


def _exposure_frame(exposure):
    missing = [name for name in _EXPOSURE_COLUMNS if name not in exposure.columns]
    if missing:
        raise ValueError(f"possession exposure is missing {missing}")
    frame = exposure.loc[:, list(_EXPOSURE_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].map(_game_id)
    frame["half"] = pd.to_numeric(frame["half"], errors="coerce")
    frame["offense_margin"] = pd.to_numeric(frame["offense_margin"], errors="coerce")
    frame["seconds"] = pd.to_numeric(frame["seconds"], errors="coerce")
    frame = frame.dropna()
    frame = frame[frame["seconds"].gt(0) & frame["seconds"].le(12 * 60)]
    frame = frame[frame["half"].isin([1, 2])]
    if frame.empty:
        raise ValueError("No possession exposure to fit pace by state")
    return frame.reset_index(drop=True)


def _path_frame(paths):
    needed = ("game_id", "margin_6", "margin_2", "final_margin", "went_ot")
    missing = [name for name in needed if name not in paths.columns]
    if missing:
        raise ValueError(f"overtime paths are missing {missing}")
    frame = paths.loc[:, list(needed)].copy()
    frame["game_id"] = frame["game_id"].map(_game_id)
    for name in needed[1:]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("No late-margin paths")
    return frame.reset_index(drop=True)


def _filter_games(frame, game_ids):
    if game_ids is None:
        return frame
    chosen = {_game_id(game_id) for game_id in game_ids}
    return frame.loc[frame["game_id"].isin(chosen)].reset_index(drop=True)


def _count_game(table):
    counted = (
        table.groupby("offense_team_id", sort=False)
        .size()
        .rename("poss")
        .reset_index()
        .rename(columns={"offense_team_id": "team_id"})
    )
    counted.insert(0, "game_id", table["game_id"].iloc[0])
    return counted


def _exposure_game(table):
    ordered = table.sort_values("possession_id")
    same_period = ordered["period"].shift(-1).eq(ordered["period"])
    seconds = np.where(
        same_period.to_numpy(),
        ordered["seconds_remaining"].to_numpy(float)
        - ordered["seconds_remaining"].shift(-1).to_numpy(float),
        ordered["seconds_remaining"].to_numpy(float),
    )
    period = ordered["period"].to_numpy()
    keep = (period <= 4) & np.isfinite(seconds) & (seconds > 0) & (seconds <= 12 * 60)
    return pd.DataFrame(
        {
            "game_id": ordered["game_id"].to_numpy()[keep],
            "half": np.where(period[keep] <= 2, 1, 2),
            "offense_margin": ordered["margin"].to_numpy()[keep],
            "seconds": seconds[keep],
        }
    )


def _require_pbp(pbp, extra=()):
    if not isinstance(pbp, pd.DataFrame) or pbp.empty:
        raise ValueError("play-by-play must be a non-empty DataFrame")
    needed = ["gameId", "actionNumber", "period", "scoreHome", "scoreAway", *extra]
    missing = [name for name in needed if name not in pbp.columns]
    if missing:
        raise ValueError(f"play-by-play needs {missing}")
    return pbp


def _clock_seconds(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    match = _CLOCK.search(str(value).strip())
    if match is None:
        return None
    return float(match.group(1) or 0.0) * 60.0 + float(match.group(2))
