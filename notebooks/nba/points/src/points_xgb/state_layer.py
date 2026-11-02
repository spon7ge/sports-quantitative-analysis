"""Player state on normal scores from the calibrated points distribution.

The quantile model is not refit here. Calibration offsets, tail tables,
group means, and Kalman variances are fit on earlier walk-forward
windows only. 2022-23 (window 1) warms the state and is not a scored
season. Window -1 is the 2026-27 holdout and never enters a fit.

The observation is a normal score. Jitter the recorded points by less
than half a point, take the PIT under that game's calibrated quantile
function, and set z = Phi^{-1}(PIT). Outcomes outside q05–q95 use the
empirical tail tables, so a large miss is not stuck at the end knot.
Positive z means the player scored more than the quantile model expected.

The state is that normal score. A player's first game starts at the
group mean. At a season boundary the mean shrinks toward the group and
the offseason variance is added once. Observation variance is chosen
with the drift variances by marginal likelihood on earlier windows.

A game with no saved quantile has no honest score. It is counted and
left out of the filter.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

from points_xgb.config import QUANTILES
from points_xgb.paths import ensure_repo_on_path

ensure_repo_on_path()
from models.shared.market import (  # noqa: E402
    binary_log_loss,
    name_key,
    no_vig_over,
)
from models.shared.metrics import paired_date_bootstrap  # noqa: E402
from models.shared.minutes_sampler import row_crps  # noqa: E402
from models.shared.minutes_sampler import _cdf_from_knots, _row_breakpoints  # noqa: E402
from models.shared.quantile_offsets import (  # noqa: E402
    apply_offsets,
    calibration_table,
    fit_offsets,
    line_band_gap,
    row_mean_pinball,
)

NOISE_WIDTH = 2.56
NOISE_FLOOR = 0.5
OOF_START = pd.Timestamp("2022-10-01")
HOLDOUT_WINDOW = -1
WARMUP_WINDOW = 1
SCORED_WINDOWS = (2, 3, 4)
SIGNAL_ACF = 0.02
SIGNAL_ICC = 0.02
MIN_GROUP_N = 30
MIN_MARKET_N = 200
CLIP_LOW = 0.0
CLIP_HIGH = 100.0
POINTS_STATS = frozenset({"points", "pts"})
PACIFIC = ZoneInfo("America/Los_Angeles")
# One local-level variance per game, in points squared. The grid is the
# marginal-likelihood search, not a value tuned on the holdout.
PROCESS_Q = (0.25, 1.0, 4.0)
PROCESS_Q_OFF = (1.0, 4.0, 16.0)
PROCESS_P0 = (1.0, 4.0, 16.0)
OBS_VAR = (0.5, 1.0, 2.0)
# Fitted drift, then successive halves, then zero. Zero is the hierarchical model.
DRIFT_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0)
PIT_EPS = 1e-6
BOOK_PREFERENCE = ("pinnacle", "novig", "prophetx", "underdog")

_QUANTILE_COLUMNS = [f"q_{level:.2f}" for level in QUANTILES]


@dataclass(frozen=True)
class FilterPath:
    pred_mean: np.ndarray
    pred_var: np.ndarray
    post_mean: np.ndarray
    post_var: np.ndarray
    gain: np.ndarray
    games_before: np.ndarray
    loglik: float


@dataclass
class _Cell:
    weight: float = 0.0
    weighted: float = 0.0
    count: int = 0


@dataclass(frozen=True)
class PointTails:
    """Empirical excesses outside q05 and q95, by integer group."""

    arrays: dict[tuple[str, int], np.ndarray]


@dataclass
class GroupPrior:
    """Leave-one-player group means and a player variance per group."""

    fine: dict[str, _Cell] = field(default_factory=dict)
    coarse: dict[str, _Cell] = field(default_factory=dict)
    fine_players: dict[str, dict[str, _Cell]] = field(default_factory=dict)
    coarse_players: dict[str, dict[str, _Cell]] = field(default_factory=dict)
    group_tau2: dict[str, float] = field(default_factory=dict)
    tau2: float = 0.0
    min_n: int = MIN_GROUP_N

    def tau2_for(self, fine: object) -> float:
        key = str(fine)
        value = float(self.group_tau2.get(key, self.tau2))
        if _wide_group(key):
            return max(value, float(self.tau2))
        return value

    def mean_for(self, player: object, fine: object, coarse: object) -> float:
        player_id = str(player)
        fine_mean = _loo_mean(
            self.fine.get(str(fine)),
            self.fine_players.get(str(fine), {}).get(player_id),
            self.min_n,
        )
        if fine_mean is not None:
            return fine_mean
        coarse_mean = _loo_mean(
            self.coarse.get(str(coarse)),
            self.coarse_players.get(str(coarse), {}).get(player_id),
            self.min_n,
        )
        if coarse_mean is not None:
            return coarse_mean
        return 0.0


@dataclass(frozen=True)
class ProcessNoise:
    per_game: float
    offseason: float
    initial: float
    loglik: float
    group_per_game: dict[str, float]
    obs_var: float = 1.0

    def q_for(self, group: object) -> float:
        return float(self.group_per_game.get(str(group), self.per_game))

    def scaled(self, factor: float) -> ProcessNoise:
        scale = float(factor)
        return ProcessNoise(
            per_game=self.per_game * scale,
            offseason=self.offseason * scale,
            initial=self.initial,
            loglik=self.loglik,
            group_per_game={
                key: value * scale for key, value in self.group_per_game.items()
            },
            obs_var=self.obs_var,
        )


def noise_scale(q10, q90) -> tuple[np.ndarray, int]:
    """Gaussian scale of the 10–90 band, floored so a zero width cannot divide."""
    low = np.asarray(q10, dtype=float)
    high = np.asarray(q90, dtype=float)
    if low.shape != high.shape:
        raise ValueError("q10 and q90 shapes differ")
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("quantiles must be finite")
    if np.any(high < low):
        raise ValueError("q90 is below q10")
    scale = (high - low) / NOISE_WIDTH
    floored = scale < NOISE_FLOOR
    return np.maximum(scale, NOISE_FLOOR), int(floored.sum())


def audit_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game, holdout flagged, and no OOF row before 2022-23."""
    required = {"game_id", "player_id", "game_date", "window_id", "is_holdout"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"predictions are missing {sorted(missing)}")
    if frame.empty:
        raise ValueError("predictions are empty")
    keys = pd.DataFrame(
        {
            "game_id": frame["game_id"].astype(str),
            "player_id": frame["player_id"].astype(str),
        }
    )
    if keys.duplicated().any():
        raise ValueError("predictions have duplicated player-games")
    dates = pd.to_datetime(frame["game_date"])
    if dates.isna().any():
        raise ValueError("predictions have a missing game date")
    holdout = frame["is_holdout"].astype(bool)
    windows = frame["window_id"].astype(int)
    if not holdout.eq(windows.eq(HOLDOUT_WINDOW)).all():
        raise ValueError("is_holdout does not match window_id -1")
    oof = ~holdout.to_numpy()
    if oof.any() and dates.loc[oof].min() < OOF_START:
        raise ValueError("OOF coverage must start in 2022-23")
    for column in _QUANTILE_COLUMNS:
        if (
            column in frame.columns
            and not np.isfinite(frame[column].to_numpy(float)).all()
        ):
            raise ValueError(f"{column} is not finite")
    summary = (
        pd.DataFrame(
            {"window_id": windows, "game_date": dates, "game_id": keys["game_id"]}
        )
        .groupby("window_id", sort=True)
        .agg(
            rows=("game_id", "size"),
            first=("game_date", "min"),
            last=("game_date", "max"),
        )
    )
    _require_ordered_windows(summary)
    return summary


def walk_forward_quantiles(y, grid, taus, windows, seed: int) -> np.ndarray:
    """Width-scaled offsets fit on earlier windows only.

    Window 1 has no earlier out-of-fold season, so those knots stay raw.
    The holdout offset is fit on every positive window and not on itself.
    """
    outcome = np.asarray(y, dtype=float).reshape(-1)
    knots = np.asarray(grid, dtype=float)
    levels = np.asarray(taus, dtype=float).reshape(-1)
    fold = np.asarray(windows).astype(int)
    if knots.shape != (outcome.shape[0], levels.shape[0]):
        raise ValueError("grid shape does not match y and taus")
    out = knots.copy()
    positive = sorted(int(value) for value in np.unique(fold) if int(value) > 0)
    for window in positive[1:]:
        past = (fold > 0) & (fold < window)
        current = fold == window
        fitted = fit_offsets(
            outcome[past],
            knots[past],
            levels,
            np.random.default_rng(int(seed) + window),
        )
        out[current] = apply_offsets(knots[current], fitted, levels)
    holdout = fold < 0
    past = fold > 0
    if bool(holdout.any()) and bool(past.any()):
        fitted = fit_offsets(
            outcome[past],
            knots[past],
            levels,
            np.random.default_rng(int(seed) + 10_000),
        )
        out[holdout] = apply_offsets(knots[holdout], fitted, levels)
    np.clip(out, CLIP_LOW, CLIP_HIGH, out=out)
    return out


def unscored_played_games(
    predictions: pd.DataFrame, played: pd.DataFrame
) -> dict[str, int]:
    """Count positive-minute games in ``played`` that have no quantile row.

    ``played`` is the candidate set. Callers restrict it to the seasons the
    state layer is allowed to see. This does not invent residuals for them.
    """
    minutes = pd.to_numeric(played["minutes"], errors="coerce")
    eligible = played.loc[minutes.gt(0)]
    pred_keys = set(
        zip(
            predictions["game_id"].astype(str),
            predictions["player_id"].astype(str),
            strict=True,
        )
    )
    play_keys = list(
        zip(
            eligible["game_id"].astype(str),
            eligible["player_id"].astype(str),
            strict=True,
        )
    )
    scored = sum(key in pred_keys for key in play_keys)
    return {
        "played": len(play_keys),
        "scored": scored,
        "unscored": len(play_keys) - scored,
    }


def attach_context(predictions: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Join season, team, position, and name. An unmatched prediction is an error."""
    needed = ["game_id", "player_id", "season_year", "team_id", "pos", "player_name"]
    missing = [column for column in needed if column not in context.columns]
    if missing:
        raise ValueError(f"context is missing {missing}")
    left = predictions.copy()
    left["_game"] = left["game_id"].astype(str)
    left["_player"] = left["player_id"].astype(str)
    right = context.loc[:, needed].copy()
    right["_game"] = right["game_id"].astype(str)
    right["_player"] = right["player_id"].astype(str)
    right = right.drop(columns=["game_id", "player_id"]).drop_duplicates(
        ["_game", "_player"]
    )
    merged = left.merge(
        right, on=["_game", "_player"], how="left", validate="many_to_one"
    )
    unmatched = int(merged["season_year"].isna().sum())
    if unmatched:
        raise ValueError(f"{unmatched} prediction rows are unmatched in the gamelogs")
    return merged.drop(columns=["_game", "_player"])


def assign_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Position, with rookie and new-team on the finer key.

    Age and experience bands are appended when those columns are already
    present. They are not inferred here.
    """
    out = frame.copy()
    pos = out["pos"].astype("string").fillna("UNK").astype(str)
    rookie = _flag(out, "rookie")
    new_team = _flag(out, "new_team")
    out["coarse_group"] = pos
    out["fine_group"] = (
        pos
        + "|rookie="
        + rookie.astype(int).astype(str)
        + "|new="
        + new_team.astype(int).astype(str)
    )
    for column in ("age_band", "experience_band"):
        if column in out.columns:
            out["fine_group"] = out["fine_group"] + "|" + out[column].astype(str)
    if "starting" in out.columns:
        out["role"] = np.where(
            out["starting"].to_numpy(dtype=int) == 1, "starter", "bench"
        )
    return out


def add_history_flags(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Debut, new-team, returner, and first-10-team-game flags.

    Debut means the player's first season in ``history``, which starts in
    2019-20, so it is a panel debut rather than an official draft class.
    A returner entered this season on the team they finished last season with.
    """
    hist = _history_keys(history)
    out = frame.copy()
    out["player_id"] = out["player_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    out["game_id"] = out["game_id"].astype(str)
    out["season_year"] = out["season_year"].astype(str)
    out["game_date"] = pd.to_datetime(out["game_date"])
    first_season = hist.groupby("player_id", sort=False)["season_year"].min()
    out["panel_debut"] = (
        out["season_year"].eq(out["player_id"].map(first_season)).fillna(False)
    )

    games = hist.sort_values(
        ["season_year", "team_id", "game_date", "game_id"], kind="mergesort"
    )
    games = games.drop_duplicates(["season_year", "team_id", "game_id"])
    games["team_game"] = (
        games.groupby(["season_year", "team_id"], sort=False).cumcount() + 1
    )
    out = out.merge(
        games[["season_year", "team_id", "game_id", "team_game"]],
        on=["season_year", "team_id", "game_id"],
        how="left",
    )
    out["early"] = out["team_game"].le(10).fillna(False)
    out["tier"] = points_tier(
        out["b_q_0.50"] if "b_q_0.50" in out.columns else out["q_0.50"]
    )

    ordered = out.sort_values(["player_id", "game_date", "game_id"], kind="mergesort")
    previous_team = ordered.groupby("player_id", sort=False)["team_id"].shift()
    previous_season = ordered.groupby("player_id", sort=False)["season_year"].shift()
    ordered["new_team"] = previous_team.notna() & previous_team.ne(ordered["team_id"])
    opener = ordered.groupby(["player_id", "season_year"], sort=False).head(1)
    return_season: set[tuple[str, str]] = set()
    opener_team: dict[tuple[str, str], str] = {}
    for row in opener.itertuples(index=False):
        key = (str(row.player_id), str(row.season_year))
        opener_team[key] = str(row.team_id)
    for index, row in opener.iterrows():
        prev_season = previous_season.loc[index]
        prev_team = previous_team.loc[index]
        same_club = (
            pd.notna(prev_season)
            and prev_season != row["season_year"]
            and prev_team == row["team_id"]
        )
        if same_club:
            return_season.add((str(row["player_id"]), str(row["season_year"])))
    keys = list(
        zip(
            ordered["player_id"].astype(str),
            ordered["season_year"].astype(str),
            strict=True,
        )
    )
    ordered["returner"] = [
        key in return_season and ordered["team_id"].iloc[i] == opener_team[key]
        for i, key in enumerate(keys)
    ]
    ordered["rookie"] = ordered["panel_debut"].astype(bool)
    return ordered.sort_index()


def points_tier(median) -> pd.Series:
    values = pd.Series(np.asarray(median, dtype=float))
    return pd.cut(
        values,
        bins=[0.0, 10.0, 20.0, 30.0, np.inf],
        labels=["0-10", "10-20", "20-30", "30+"],
        right=False,
    )


def prepare_residuals(
    frame: pd.DataFrame, seed: int = 0
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Normal scores under the walk-forward calibrated distribution.

    Window 1 has no earlier season, so its tail table is fit on window 1.
    Every later window, including the holdout, uses tails from earlier
    positive windows only. The score itself is always the game's own
    jittered PIT. ``obs_sd`` stays 1 until the likelihood chooses the
    observation variance.
    """
    out = frame.sort_values(
        ["player_id", "game_date", "game_id"], kind="mergesort"
    ).copy()
    knots, _levels = quantile_grid(out, "b_q_")
    groups = (
        out["starting"].to_numpy(dtype=int)
        if "starting" in out.columns
        else np.zeros(len(out), dtype=int)
    )
    pit, scores, tails = _window_tails_and_scores(
        out["pts"].to_numpy(dtype=float),
        knots,
        out["window_id"].to_numpy(),
        groups,
        int(seed),
    )
    if not np.isfinite(scores).all():
        raise ValueError("normal scores are not finite")
    out["pit"] = pit
    out["z"] = scores
    out["residual"] = scores
    out["obs_sd"] = 1.0
    out["rest_days"] = (
        out.groupby("player_id", sort=False)["game_date"]
        .diff()
        .dt.days.to_numpy(dtype=float)
    )
    previous = out.groupby("player_id", sort=False)["season_year"].shift()
    out["season_boundary"] = previous.notna() & previous.astype(str).ne(
        out["season_year"].astype(str)
    )
    deviation = float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan")
    return out, {
        "rows": int(len(out)),
        "z_mean": float(np.mean(scores)),
        "z_sd": deviation,
        "tails": tails,
    }


def residual_dependence(frame: pd.DataFrame) -> dict[str, object]:
    """Pooled within-player autocorrelation and an ICC, plus the requested slices."""
    work = frame.reset_index(drop=True)
    acf = np.array(
        [_lag_corr(work["z"], work["player_id"], lag) for lag in range(1, 11)]
    )
    report: dict[str, object] = {
        "acf": acf,
        "acf_1": float(acf[0]),
        "icc": _icc(work["z"].to_numpy(dtype=float), work["player_id"]),
        "slices": _dependence_slices(work),
    }
    return report


def signal_remains(
    acf_1: float,
    icc: float,
    *,
    acf_max: float = SIGNAL_ACF,
    icc_max: float = SIGNAL_ICC,
) -> bool:
    """True when either summary clears the predeclared near-zero gate."""
    if not math.isfinite(acf_1) or not math.isfinite(icc):
        return False
    return abs(float(acf_1)) >= acf_max or float(icc) >= icc_max


def leave_one_player_mean(
    residuals,
    weights,
    players,
    groups,
    coarse,
    player,
    fine_key,
    coarse_key,
    min_n: int = MIN_GROUP_N,
) -> float:
    """Group mean after removing ``player``. Falls back from team to position."""
    weight = np.asarray(weights, dtype=float)
    if np.any(weight <= 0) or not np.isfinite(weight).all():
        raise ValueError("weights must be positive and finite")
    frame = pd.DataFrame(
        {
            "residual": np.asarray(residuals, dtype=float),
            "obs_sd": 1.0 / np.sqrt(weight),
            "player_id": np.asarray(players).astype(str),
            "fine_group": np.asarray(groups).astype(str),
            "coarse_group": np.asarray(coarse).astype(str),
        }
    )
    prior = fit_group_prior(frame, min_n=min_n)
    return prior.mean_for(player, fine_key, coarse_key)


def fit_group_prior(frame: pd.DataFrame, min_n: int = MIN_GROUP_N) -> GroupPrior:
    """Normal-normal group locations from this frame only."""
    _require_columns(
        frame,
        ["residual", "obs_sd", "player_id", "fine_group", "coarse_group"],
        "group prior",
    )
    prior = GroupPrior(min_n=int(min_n))
    residual = frame["residual"].to_numpy(dtype=float)
    sd = frame["obs_sd"].to_numpy(dtype=float)
    if np.any(sd <= 0) or not np.isfinite(sd).all() or not np.isfinite(residual).all():
        raise ValueError("group prior residuals and scales must be finite and positive")
    weight = 1.0 / np.square(sd)
    players = frame["player_id"].astype(str).to_numpy()
    fine = frame["fine_group"].astype(str).to_numpy()
    coarse = frame["coarse_group"].astype(str).to_numpy()
    for index in range(len(frame)):
        _add_cell(
            prior.fine,
            prior.fine_players,
            fine[index],
            players[index],
            weight[index],
            residual[index],
        )
        _add_cell(
            prior.coarse,
            prior.coarse_players,
            coarse[index],
            players[index],
            weight[index],
            residual[index],
        )
    prior.tau2 = _tau2(residual, weight, players, coarse, prior.coarse)
    for group in pd.unique(fine):
        chosen = fine == group
        if int(chosen.sum()) < int(min_n):
            continue
        prior.group_tau2[str(group)] = _tau2(
            residual[chosen],
            weight[chosen],
            players[chosen],
            fine[chosen],
            prior.fine,
        )
    return prior


def fit_process_variances(
    frame: pd.DataFrame,
    prior: GroupPrior,
    *,
    q_grid: Sequence[float] = PROCESS_Q,
    q_off_grid: Sequence[float] = PROCESS_Q_OFF,
    p0_grid: Sequence[float] = PROCESS_P0,
    obs_grid: Sequence[float] = OBS_VAR,
    group_q_grid: Sequence[float] | None = PROCESS_Q,
    min_n: int = MIN_GROUP_N,
) -> ProcessNoise:
    """Grid marginal likelihood. Rows outside ``frame`` are not read."""
    _require_columns(
        frame,
        [
            "player_id",
            "residual",
            "obs_sd",
            "season_year",
            "team_id",
            "rest_days",
            "coarse_group",
            "fine_group",
        ],
        "process fit",
    )
    best_ll = -np.inf
    best = (
        float(q_grid[0]),
        float(q_off_grid[0]),
        float(p0_grid[0]),
        float(obs_grid[0]),
    )
    for per_game in q_grid:
        for offseason in q_off_grid:
            for initial in p0_grid:
                for obs_var in obs_grid:
                    score = _total_loglik(
                        frame,
                        prior,
                        float(per_game),
                        float(offseason),
                        float(initial),
                        obs_var=float(obs_var),
                    )
                    if score > best_ll:
                        best_ll = score
                        best = (
                            float(per_game),
                            float(offseason),
                            float(initial),
                            float(obs_var),
                        )
    grouped: dict[str, float] = {}
    if group_q_grid is not None:
        for group, part in frame.groupby(frame["coarse_group"].astype(str), sort=False):
            if len(part) < min_n:
                continue
            local_ll = -np.inf
            local_q = best[0]
            for per_game in group_q_grid:
                score = _total_loglik(
                    part,
                    prior,
                    float(per_game),
                    best[1],
                    best[2],
                    obs_var=best[3],
                )
                if score > local_ll:
                    local_ll = score
                    local_q = float(per_game)
            grouped[str(group)] = shrink_variance(
                local_q, len(part), best[0], prior_n=min_n
            )
    return ProcessNoise(
        per_game=best[0],
        offseason=best[1],
        initial=best[2],
        loglik=float(best_ll),
        group_per_game=grouped,
        obs_var=best[3],
    )


def shrink_variance(
    group_q: float, group_n: int, global_q: float, prior_n: int = MIN_GROUP_N
) -> float:
    """Pull a group's drift toward the global drift. ``prior_n`` is the prior count."""
    if group_n < 0 or prior_n <= 0:
        raise ValueError("shrinkage counts must be non-negative, with a positive prior")
    weight = group_n / (group_n + prior_n)
    return float(weight * group_q + (1.0 - weight) * global_q)


def filter_player(
    residual,
    obs_sd,
    season,
    team,
    rest_days,
    per_game,
    *,
    offseason: float,
    initial: float,
    initial_mean: float,
    reset_on_new_team: bool,
    group_mean=None,
    group_tau2=None,
) -> FilterPath:
    """Predict, then update. Game ``i`` does not enter ``pred_*[i]``.

    Within a season, rest of about two days is one process step. Extra
    rest adds steps. A season change shrinks the mean toward the group
    prior when ``group_mean`` is passed, then adds ``offseason`` once.
    A new team lifts variance back to ``initial`` when
    ``reset_on_new_team`` is set. The first appearance uses the group
    prior and does not add process noise first.
    """
    observed = np.asarray(residual, dtype=float).reshape(-1)
    scale = np.asarray(obs_sd, dtype=float).reshape(-1)
    seasons = np.asarray(season).astype(str)
    teams = np.asarray(team).astype(str)
    rest = np.asarray(rest_days, dtype=float).reshape(-1)
    drift = np.asarray(per_game, dtype=float).reshape(-1)
    n = observed.shape[0]
    if not all(array.shape[0] == n for array in (scale, seasons, teams, rest, drift)):
        raise ValueError("filter inputs must have one row each")
    if n == 0:
        raise ValueError("filter player has no games")
    if (
        np.any(scale <= 0)
        or not np.isfinite(scale).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("residuals and observation scales must be finite and positive")
    if min(initial, offseason) < 0 or not math.isfinite(initial_mean):
        raise ValueError("initial variance and offseason variance must be valid")
    if np.any(drift < 0) or not np.isfinite(drift).all():
        raise ValueError("per-game variance must be non-negative and finite")

    pred_mean = np.empty(n)
    pred_var = np.empty(n)
    post_mean = np.empty(n)
    post_var = np.empty(n)
    gain = np.empty(n)
    games_before = np.empty(n, dtype=int)
    mean = float(initial_mean)
    variance = float(initial)
    loglik = 0.0
    prev_season: str | None = None
    prev_team: str | None = None
    for index in range(n):
        if prev_season is None:
            pass
        elif seasons[index] != prev_season:
            mean = _shrink_toward_group(
                mean, variance, group_mean, group_tau2, index
            )
            variance = variance + float(offseason)
            if reset_on_new_team and teams[index] != prev_team:
                variance = max(variance, float(initial))
        else:
            variance = variance + float(drift[index]) * _missed_steps(rest[index])
            if reset_on_new_team and teams[index] != prev_team:
                mean = _shrink_toward_group(
                    mean, variance, group_mean, group_tau2, index
                )
                variance = max(variance, float(initial))
        obs_var = float(scale[index] ** 2)
        predictive = variance + obs_var
        innovation = float(observed[index] - mean)
        step_gain = 0.0 if variance <= 0 else variance / predictive
        pred_mean[index] = mean
        pred_var[index] = variance
        gain[index] = step_gain
        games_before[index] = index
        loglik += -0.5 * (
            math.log(2.0 * math.pi)
            + math.log(predictive)
            + innovation * innovation / predictive
        )
        mean = mean + step_gain * innovation
        variance = (1.0 - step_gain) * variance
        post_mean[index] = mean
        post_var[index] = variance
        prev_season = seasons[index]
        prev_team = teams[index]
    if not math.isfinite(loglik):
        raise ValueError("filter log likelihood is not finite")
    return FilterPath(
        pred_mean, pred_var, post_mean, post_var, gain, games_before, loglik
    )


def apply_layer(
    frame: pd.DataFrame,
    prior: GroupPrior,
    process: ProcessNoise,
    *,
    dynamic: bool,
) -> pd.DataFrame:
    """Predictive state on ``frame`` only. Sort is by player, then date."""
    ordered = frame.sort_values(["player_id", "game_date", "game_id"], kind="mergesort")
    pieces = []
    for player, part in ordered.groupby(ordered["player_id"].astype(str), sort=False):
        fine = part["fine_group"].astype(str)
        coarse = part["coarse_group"].astype(str)
        mean0 = prior.mean_for(player, fine.iloc[0], coarse.iloc[0])
        gmean = np.array(
            [prior.mean_for(player, item, other) for item, other in zip(fine, coarse, strict=True)]
        )
        gtau = np.array([prior.tau2_for(item) for item in fine])
        # Zero per-game and offseason variance is the static hierarchical
        # model: same prior, same observation variance, no time step.
        if dynamic and not _no_drift(process):
            per_game = np.array(
                [process.q_for(group) for group in coarse],
                dtype=float,
            )
            offseason = process.offseason
            initial = _opening_variance(prior, process, part, dynamic=True)
            reset = True
            scale = np.full(len(part), math.sqrt(process.obs_var))
            boundary_mean = gmean
            boundary_tau = gtau
        else:
            per_game = np.zeros(len(part))
            offseason = 0.0
            initial = _opening_variance(prior, process, part, dynamic=True)
            reset = False
            if process.obs_var <= 0:
                raise ValueError("observation variance must be positive")
            scale = np.full(len(part), math.sqrt(process.obs_var))
            boundary_mean = None
            boundary_tau = None
        path = filter_player(
            part["residual"].to_numpy(dtype=float),
            scale,
            part["season_year"].astype(str).to_numpy(),
            part["team_id"].astype(str).to_numpy(),
            part["rest_days"].to_numpy(dtype=float),
            per_game,
            offseason=offseason,
            initial=initial,
            initial_mean=mean0,
            reset_on_new_team=reset,
            group_mean=boundary_mean,
            group_tau2=boundary_tau,
        )
        piece = pd.DataFrame(
            {
                "state_mean": path.pred_mean,
                "state_var": path.pred_var,
                "post_mean": path.post_mean,
                "post_var": path.post_var,
                "gain": path.gain,
                "games_before": path.games_before,
                "obs_sd": scale,
            },
            index=part.index,
        )
        if "season_boundary" in part.columns:
            piece["season_boundary"] = part["season_boundary"].to_numpy(dtype=bool)
        pieces.append(piece)
    if not pieces:
        raise ValueError("state layer received no rows")
    return pd.concat(pieces).sort_index()


def fit_window(
    frame: pd.DataFrame,
    window: int,
    *,
    dynamic: bool,
    **grid,
) -> tuple[GroupPrior, ProcessNoise]:
    """Group mean and process variances from rows strictly before ``window``."""
    past = pd.Series(
        history_mask(frame["window_id"].to_numpy(), int(window)),
        index=frame.index,
    )
    if not bool(past.any()):
        raise ValueError(f"window {int(window)} has no earlier rows")
    prior = fit_group_prior(frame.loc[past])
    if dynamic:
        process = fit_process_variances(frame.loc[past], prior, **grid)
    else:
        process = ProcessNoise(0.0, 0.0, float(prior.tau2), float("nan"), {})
    return prior, process


def apply_window(
    frame: pd.DataFrame,
    window: int,
    prior: GroupPrior,
    process: ProcessNoise,
    *,
    dynamic: bool,
) -> pd.DataFrame:
    """Filter history plus ``window``. Return only that window's predictive rows."""
    target_window = int(window)
    past = pd.Series(
        history_mask(frame["window_id"].to_numpy(), target_window),
        index=frame.index,
    )
    target = pd.Series(
        frame["window_id"].to_numpy(dtype=int) == target_window,
        index=frame.index,
    )
    states = apply_layer(frame.loc[past | target], prior, process, dynamic=dynamic)
    keep = frame.loc[states.index, "window_id"].to_numpy(dtype=int) == target_window
    piece = states.loc[keep].copy()
    piece["window_id"] = target_window
    return piece


def window_states(
    frame: pd.DataFrame,
    windows: Sequence[int],
    *,
    dynamic: bool,
    **grid,
) -> pd.DataFrame:
    """Predictive states for ``windows``, with fits that stop at the window before.

    A later window's residuals are not in the fit and not in the filter path.
    """
    pieces = [
        apply_window(
            frame,
            window,
            *fit_window(frame, window, dynamic=dynamic, **grid),
            dynamic=dynamic,
        )
        for window in windows
    ]
    if not pieces:
        raise ValueError("no windows were scored")
    return pd.concat(pieces)


def history_mask(window_id, target_window: int) -> np.ndarray:
    windows = np.asarray(window_id).astype(int)
    if int(target_window) < 0:
        return windows > 0
    return (windows > 0) & (windows < int(target_window))


def adjust_quantiles(
    knots,
    taus,
    state_mean,
    state_var,
    obs_sd,
    *,
    tails: Mapping[int, PointTails] | None = None,
    windows=None,
    groups=None,
) -> np.ndarray:
    """Map each tau through the state's normal-score predictive distribution.

    A zero state with unit observation noise returns the calibrated knots.
    ``s**2`` is the observation variance plus the pregame state variance.
    """
    grid = np.asarray(knots, dtype=float)
    levels = np.asarray(taus, dtype=float).reshape(-1)
    mean = np.asarray(state_mean, dtype=float).reshape(-1)
    variance = np.asarray(state_var, dtype=float).reshape(-1)
    scale = np.asarray(obs_sd, dtype=float).reshape(-1)
    if grid.shape != (mean.shape[0], levels.shape[0]):
        raise ValueError("knots do not match the state rows")
    if np.any(scale <= 0) or np.any(variance < 0):
        raise ValueError("observation scale and state variance must be valid")
    spread = np.sqrt(np.square(scale) + variance)
    uniforms = norm.cdf(mean[:, None] + spread[:, None] * norm.ppf(levels)[None, :])
    window_ids = None if windows is None else np.asarray(windows).astype(int)
    group_ids = (
        np.zeros(mean.shape[0], dtype=int)
        if groups is None
        else np.asarray(groups, dtype=int).reshape(-1)
    )
    out = np.empty_like(grid)
    for index in range(mean.shape[0]):
        tail = None
        if tails is not None and window_ids is not None:
            tail = tails[int(window_ids[index])]
        out[index] = _quantiles_at(
            uniforms[index], grid[index], levels, tail, int(group_ids[index])
        )
    out.sort(axis=1)
    np.clip(out, CLIP_LOW, CLIP_HIGH, out=out)
    return out


def p_over_state(
    line,
    knots,
    taus,
    state_mean,
    state_var,
    *,
    obs_var: float,
    tails: PointTails,
    groups,
) -> np.ndarray:
    """P(points > line) after shifting the calibrated PIT by the pregame state.

    ``s**2 = obs_var + state_var``. Once the observation noise has been
    standardized, ``obs_var`` is about 1. A half-point line is evaluated
    directly. An integer line is the half-point above it, the same
    continuous threshold the jittered PIT uses.
    """
    raw = np.asarray(line, dtype=float).reshape(-1)
    integer = np.isfinite(raw) & (np.abs(raw - np.round(raw)) < 1e-8)
    threshold = np.where(integer, raw + 0.5, raw)
    grid = np.asarray(knots, dtype=float)
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)
    if grid.shape[0] == 1 and raw.shape[0] > 1:
        grid = np.repeat(grid, raw.shape[0], axis=0)
    group_ids = np.asarray(groups, dtype=int).reshape(-1)
    if group_ids.shape[0] == 1 and raw.shape[0] > 1:
        group_ids = np.repeat(group_ids, raw.shape[0])
    mean = np.asarray(state_mean, dtype=float).reshape(-1)
    variance = np.asarray(state_var, dtype=float).reshape(-1)
    if float(obs_var) <= 0 or np.any(variance < 0):
        raise ValueError("observation variance and state variance must be valid")
    pit = np.array(
        [
            _row_cdf(float(threshold[index]), grid[index], tails, int(group_ids[index]))
            for index in range(raw.shape[0])
        ]
    )
    pit = np.clip(pit, PIT_EPS, 1.0 - PIT_EPS)
    spread = np.sqrt(float(obs_var) + variance)
    return 1.0 - norm.cdf((norm.ppf(pit) - mean) / spread)


def cdf_values(thresholds, knots, taus) -> np.ndarray:
    """Linear CDF from a monotone quantile grid.

    Outside the grid, the CDF stays at the end taus.
    """
    from models.shared.points_calibration import quantile_cdf  # noqa: PLC0415

    return quantile_cdf(thresholds, knots, taus)


def p_over(line, knots, taus) -> np.ndarray:
    """P(points > line). An integer line is the half-point above it.

    Recorded points are integers. ``pts > 20`` is the same event as a
    jittered total clearing 20.5, which is the contract an x.5 line prices.
    """
    raw = np.asarray(line, dtype=float).reshape(-1)
    integer = np.isfinite(raw) & (np.abs(raw - np.round(raw)) < 1e-8)
    threshold = np.where(integer, raw + 0.5, raw)
    return 1.0 - cdf_values(threshold, knots, taus)


def quantile_grid(
    frame: pd.DataFrame, prefix: str = "q_"
) -> tuple[np.ndarray, np.ndarray]:
    columns = [f"{prefix}{level:.2f}" for level in QUANTILES]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing quantile columns {missing}")
    return frame.loc[:, columns].to_numpy(dtype=float), np.asarray(
        QUANTILES, dtype=float
    )


def score_distribution(y, knots, taus, rng) -> dict[str, object]:
    actual = np.asarray(y, dtype=float).reshape(-1)
    pinball = row_mean_pinball(actual, knots, taus)
    crps = row_crps(actual, knots, taus)
    table = calibration_table(actual, knots, taus, rng)
    return {
        "pinball": pinball,
        "crps": crps,
        "mean_pinball": float(pinball.mean()),
        "mean_crps": float(crps.mean()),
        "band_gap": line_band_gap(table),
        "table": table,
    }


def paired_improvement(new_score, base_score, dates, *, n_boot: int, seed: int) -> dict:
    """Mean of new minus base. A negative pinball or CRPS difference is a gain."""
    return paired_date_bootstrap(base_score, new_score, dates, n_boot=n_boot, seed=seed)


def slice_pinball(
    y,
    knots_base,
    knots_new,
    taus,
    dates,
    masks: Mapping[str, np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    actual = np.asarray(y, dtype=float)
    when = pd.to_datetime(np.asarray(dates))
    rows = []
    for name, mask in masks.items():
        chosen = np.asarray(mask, dtype=bool)
        if int(chosen.sum()) < 2:
            continue
        base = row_mean_pinball(actual[chosen], knots_base[chosen], taus)
        new = row_mean_pinball(actual[chosen], knots_new[chosen], taus)
        summary = paired_improvement(new, base, when[chosen], n_boot=n_boot, seed=seed)
        rows.append(
            {
                "slice": name,
                "n": int(chosen.sum()),
                "pinball_base": float(base.mean()),
                "pinball_new": float(new.mean()),
                "delta": summary["diff"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def permute_within_player(values, player_id, rng) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    players = np.asarray(player_id).astype(str)
    generator = np.random.default_rng(rng) if not hasattr(rng, "permutation") else rng
    for player in pd.unique(players):
        index = np.flatnonzero(players == player)
        out[index] = generator.permutation(out[index])
    return out


def experience_gain_table(states: pd.DataFrame):
    """Kalman gain by games already in the filter. Reuses the dynamic-points bins."""
    from src.models.dynamic_points_diagnostics import gain_summary

    trace = pd.DataFrame(
        {
            "games_in_model_before": states["games_before"].to_numpy(dtype=int),
            "kalman_gain": states["gain"].to_numpy(dtype=float),
            "season_boundary": states["season_boundary"].to_numpy(dtype=bool),
        }
    )
    return gain_summary(trace)


def latest_player_state(states: pd.DataFrame) -> pd.DataFrame:
    ordered = states.sort_values(
        ["player_id", "game_date", "game_id"], kind="mergesort"
    )
    return (
        ordered.groupby(ordered["player_id"].astype(str), sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


def prediction_hash(frame: pd.DataFrame) -> str:
    """Hash of the saved medians. A refit of XGBoost changes the residual history."""
    keys = pd.DataFrame(
        {
            "window_id": frame["window_id"].astype(int),
            "game_id": frame["game_id"].astype(str),
            "player_id": frame["player_id"].astype(str),
            "q_0.50": np.round(frame["q_0.50"].to_numpy(dtype=float), 6),
        }
    ).sort_values(["window_id", "game_id", "player_id"], kind="mergesort")
    return hashlib.sha256(keys.to_csv(index=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_state_artifacts(
    directory: Path,
    payload: Mapping,
    states: pd.DataFrame,
) -> dict[str, Path]:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    hyperparameters = folder / "hyperparameters.json"
    hyperparameters.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n")
    state_path = folder / "player_state.parquet"
    states.to_parquet(state_path, index=False)
    return {"hyperparameters": hyperparameters, "player_state": state_path}


def layer_decision(
    *,
    pinball_low: float,
    pinball_high: float,
    market_n: int,
    market_low: float,
    market_high: float,
    shuffle_delta: float,
    real_delta: float,
    narrow: bool,
    min_market_n: int = MIN_MARKET_N,
) -> tuple[str, str]:
    """Keep only when pinball and closing-line log loss both improve.

    Differences are dynamic minus calibrated. Negative is better.
    A gain that a within-player shuffle can match is not a temporal effect.
    """
    if not math.isfinite(pinball_low) or not math.isfinite(pinball_high):
        return "drop", "pinball interval is missing"
    if pinball_low > 0:
        return "drop", "dynamic pinball is worse than calibrated XGB"
    if not pinball_high < 0:
        return "drop", "pinball gain is not distinguishable from zero"
    if (
        math.isfinite(shuffle_delta)
        and math.isfinite(real_delta)
        and shuffle_delta <= real_delta
    ):
        return "drop", "shuffle test: the gain survives a within-player permutation"
    if narrow:
        return "drop", "the gain flips sign when the drift is halved or doubled"
    if int(market_n) < int(min_market_n):
        return (
            "drop",
            "priced closing lines do not cover enough settled bets to show edge",
        )
    if (
        not math.isfinite(market_low)
        or not math.isfinite(market_high)
        or market_low > 0
    ):
        return "drop", "closing-line log loss is worse for the dynamic layer"
    if market_high < 0:
        return "keep", "pinball and closing-line log loss both improve"
    return "drop", "closing-line log loss does not improve"


def choose_sweep(sweep: pd.DataFrame) -> tuple[str, str]:
    """Choose hierarchical, a smaller drift, or the calibrated quantiles.

    Deltas are candidate minus baseline pinball. Negative is a gain. The
    gain is outside the noise when the upper end of the date-bootstrap
    interval is below zero. Scale 0 is the hierarchical model. A positive
    scale below 1 is a small drift. The best row is the lowest pinball
    delta, and the smaller scale breaks a tie.
    """
    required = {"scale", "delta_b", "ci_high_b", "delta_c", "ci_high_c"}
    missing = required - set(sweep.columns)
    if missing:
        raise ValueError(f"drift sweep is missing {sorted(missing)}")
    if sweep.empty:
        raise ValueError("drift sweep is empty")
    work = sweep.sort_values(["delta_b", "scale"], kind="mergesort")
    if not bool(work["scale"].eq(0).any()):
        raise ValueError("drift sweep is missing the zero-drift row")
    best = work.iloc[0]
    zero = work.loc[work["scale"].eq(0)].iloc[0]
    best_scale = float(best["scale"])
    small = 0.0 < best_scale < 1.0
    if small and _beats(best["ci_high_b"]) and _beats(best["ci_high_c"]):
        return (
            "dynamic",
            (
                f"drift scale {best_scale:g} beats the calibrated quantiles "
                "and the zero-drift hierarchical model; both intervals sit below zero"
            ),
        )
    if _beats(zero["ci_high_b"]):
        return (
            "hierarchical",
            "zero-drift hierarchical model beats the calibrated quantiles; the interval sits below zero",
        )
    return (
        "drop",
        "no point on the drift sweep beats the calibrated quantiles, so the layer is dropped and the calibrated model is the one to ship",
    )


def _beats(upper) -> bool:
    return math.isfinite(float(upper)) and float(upper) < 0.0


def _no_drift(process: ProcessNoise) -> bool:
    if process.per_game != 0.0 or process.offseason != 0.0:
        return False
    return all(value == 0.0 for value in process.group_per_game.values())


def load_closing_directory(directory: Path, book: str) -> pd.DataFrame:
    """Long quotes from ``*_nba_closing_*.json`` written by the closing rule."""
    rows: list[dict] = []
    for path in sorted(Path(directory).glob("*_nba_closing_*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or "date" not in payload:
            raise ValueError(f"{path.name} is not a closing file")
        slate = pd.Timestamp(payload["date"]).normalize()
        for player in payload.get("players") or []:
            for line in player.get("lines") or []:
                rows.append(
                    {
                        "book": book,
                        "slate_date": slate,
                        "player_name": player.get("player"),
                        "line": line.get("line_score"),
                        "side": line.get("side"),
                        "american": line.get("american_price"),
                        "stat_type": line.get("stat_type"),
                    }
                )
    columns = [
        "book",
        "slate_date",
        "player_name",
        "line",
        "side",
        "american",
        "stat_type",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame["slate_date"] = pd.to_datetime(frame["slate_date"])
    return frame


def pair_points_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """One two-sided points price per book, player, and date.

    Several lines collapse to the pair whose no-vig over is closest to 1/2.
    Assists and other stats are ignored. A missing American price is dropped,
    because ROI at the vig needs both sides.
    """
    required = {
        "book",
        "slate_date",
        "player_name",
        "line",
        "side",
        "american",
        "stat_type",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"quotes are missing {sorted(missing)}")
    points = frame.loc[
        frame["stat_type"].astype(str).str.lower().isin(POINTS_STATS)
    ].copy()
    if points.empty:
        return _empty_pairs()
    points["side"] = points["side"].astype(str).str.lower()
    points = points.loc[points["side"].isin(["over", "under"])]
    points["line"] = pd.to_numeric(points["line"], errors="coerce")
    points["american"] = pd.to_numeric(points["american"], errors="coerce")
    points = points.dropna(subset=["line", "american", "player_name"])
    if points.empty:
        return _empty_pairs()
    points["slate_date"] = pd.to_datetime(points["slate_date"]).dt.normalize()
    wide = points.pivot_table(
        index=["book", "slate_date", "player_name", "line"],
        columns="side",
        values="american",
        aggfunc="last",
    ).reset_index()
    if "over" not in wide.columns or "under" not in wide.columns:
        return _empty_pairs()
    wide = wide.dropna(subset=["over", "under"])
    wide["novig_over"] = no_vig_over(wide["over"], wide["under"])
    wide["balance"] = (wide["novig_over"] - 0.5).abs()
    wide = wide.sort_values("balance", kind="mergesort")
    chosen = wide.groupby(["book", "slate_date", "player_name"], sort=False).head(1)
    return (
        chosen.rename(columns={"over": "over_american", "under": "under_american"})
        .drop(columns=["balance"])
        .reset_index(drop=True)
    )


def opening_quotes(snapshots: Iterable, book: str) -> pd.DataFrame:
    """Earliest snapshot on a Pacific date that still lists the player."""
    earliest: dict[tuple, tuple] = {}
    for snapshot in snapshots:
        fetched = snapshot.fetched_at
        if fetched.tzinfo is None:
            raise ValueError("snapshot fetched_at must be timezone-aware")
        day = fetched.astimezone(PACIFIC).date()
        by_player: dict[str, list] = defaultdict(list)
        for line in snapshot.lines:
            if str(getattr(line, "stat_type", "")).lower() not in POINTS_STATS:
                continue
            by_player[str(line.player)].append(line)
        for player, lines in by_player.items():
            key = (day, player)
            current = earliest.get(key)
            if current is None or fetched < current[0]:
                earliest[key] = (fetched, lines)
    rows = []
    for (day, player), (_fetched, lines) in earliest.items():
        for line in lines:
            rows.append(
                {
                    "book": book,
                    "slate_date": pd.Timestamp(day),
                    "player_name": player,
                    "line": line.line_score,
                    "side": line.side,
                    "american": line.american_price,
                    "stat_type": line.stat_type,
                }
            )
    if not rows:
        return _empty_pairs()
    return pair_points_quotes(pd.DataFrame(rows))


def probability_clv(open_over, close_over, take_over, line_unchanged) -> np.ndarray:
    """Closing no-vig probability minus the open, on the side that was bet.

    A moved points line is a different contract, so its CLV is missing.
    """
    opened = np.asarray(open_over, dtype=float)
    closed = np.asarray(close_over, dtype=float)
    side = np.asarray(take_over, dtype=bool)
    same = np.asarray(line_unchanged, dtype=bool)
    open_side = np.where(side, opened, 1.0 - opened)
    close_side = np.where(side, closed, 1.0 - closed)
    clv = close_side - open_side
    return np.where(same, clv, np.nan)


def align_market(
    quotes: pd.DataFrame, games: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Match a quote to a player-game on the slate date, then the previous day.

    ``GAME_DATE`` in the gamelogs is the UTC date. The closing file is a
    Pacific slate, which can fall on the day before.
    """
    board = quotes.copy()
    board["name_key"] = [name_key(name) for name in board["player_name"]]
    board["slate_date"] = pd.to_datetime(board["slate_date"]).dt.normalize()
    logs = games.loc[
        :, ["player_id", "player_name", "game_id", "game_date", "pts"]
    ].copy()
    logs["name_key"] = [name_key(name) for name in logs["player_name"]]
    logs["game_date"] = pd.to_datetime(logs["game_date"]).dt.normalize()
    logs = logs.drop(columns=["player_name"]).drop_duplicates(["name_key", "game_date"])
    same = board.merge(
        logs,
        left_on=["name_key", "slate_date"],
        right_on=["name_key", "game_date"],
        how="left",
    )
    previous = board.assign(
        slate_date=board["slate_date"] - pd.Timedelta(days=1)
    ).merge(
        logs,
        left_on=["name_key", "slate_date"],
        right_on=["name_key", "game_date"],
        how="left",
        suffixes=("", "_prev"),
    )
    for column in ("player_id", "game_id", "game_date", "pts"):
        same[column] = same[column].where(same[column].notna(), previous[column])
    matched = same.dropna(subset=["player_id", "pts", "line"]).copy()
    counts = {
        "quotes": int(len(board)),
        "matched": int(len(matched)),
        "unmatched": int(len(board) - len(matched)),
    }
    return matched.reset_index(drop=True), counts


def preferred_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """One quote per player-date. Pinnacle wins a tie, then the smaller overround."""
    if quotes.empty:
        return quotes.copy()
    board = quotes.copy()
    rank = {name: index for index, name in enumerate(BOOK_PREFERENCE)}
    board["preference"] = board["book"].map(rank).fillna(len(rank))
    over = 1.0 / _decimal(board["over_american"])
    under = 1.0 / _decimal(board["under_american"])
    board["overround"] = over + under - 1.0
    board = board.sort_values(["preference", "overround"], kind="mergesort")
    keys = (
        ["name_key", "slate_date"]
        if "name_key" in board.columns
        else ["player_name", "slate_date"]
    )
    return board.groupby(keys, sort=False).head(1).reset_index(drop=True)


def market_scores(
    aligned: pd.DataFrame,
    probability,
    *,
    n_boot: int,
    seed: int,
    thresholds: Sequence[float] = (0.0, 0.03, 0.05, 0.10),
) -> dict[str, object]:
    """Brier, log loss, reliability, and ROI at each edge, on settled bets only.

    A push (points equal to the line) does not settle. DNP rows should
    already be absent: this frame is joined to played predictions.
    """
    from models.shared.market import score_market

    prob = np.asarray(probability, dtype=float).reshape(-1)
    if len(prob) != len(aligned):
        raise ValueError("probabilities do not match market rows")
    hit = (
        aligned["pts"].to_numpy(dtype=float) > aligned["line"].to_numpy(dtype=float)
    ).astype(float)
    settled = aligned["pts"].to_numpy(dtype=float) != aligned["line"].to_numpy(
        dtype=float
    )
    novig = no_vig_over(aligned["over_american"], aligned["under_american"])
    brier = (
        float(np.mean((prob[settled] - hit[settled]) ** 2))
        if settled.any()
        else float("nan")
    )
    log_loss = (
        float(np.mean(binary_log_loss(prob[settled], hit[settled])))
        if settled.any()
        else float("nan")
    )
    market_loss = (
        float(np.mean(binary_log_loss(novig[settled], hit[settled])))
        if settled.any()
        else float("nan")
    )
    board = aligned.copy()
    board["game_key"] = board["game_id"].astype(str)
    board["over_bet"] = board["over_american"]
    board["under_bet"] = board["under_american"]
    thresholds_out = []
    for threshold in thresholds:
        _market, bets = score_market(
            prob,
            board,
            novig,
            novig,
            edge_threshold=float(threshold),
            n_boot=n_boot,
            seed=seed,
        )
        thresholds_out.append(
            {"edge": float(threshold), "market": _market, "bets": bets}
        )
    return {
        "n_settled": int(settled.sum()),
        "brier": brier,
        "log_loss": log_loss,
        "market_log_loss": market_loss,
        "reliability": reliability_table(prob[settled], hit[settled]),
        "thresholds": thresholds_out,
        "settled_log_loss": np.where(settled, binary_log_loss(prob, hit), np.nan),
    }


def reliability_table(probability, hit, n_bins: int = 10) -> pd.DataFrame:
    prob = np.asarray(probability, dtype=float)
    outcome = np.asarray(hit, dtype=float)
    if prob.shape != outcome.shape:
        raise ValueError("reliability inputs differ in length")
    if prob.size == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_p", "hit_rate"])
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(int(n_bins))]
    bucket = pd.cut(prob, bins=edges, labels=labels, include_lowest=True)
    rows = []
    for label, index in pd.Series(bucket).groupby(bucket, observed=True).groups.items():
        chosen = np.asarray(list(index))
        rows.append(
            {
                "bin": str(label),
                "n": int(len(chosen)),
                "mean_p": float(prob[chosen].mean()),
                "hit_rate": float(outcome[chosen].mean()),
            }
        )
    return pd.DataFrame(rows)


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "book",
            "slate_date",
            "player_name",
            "line",
            "over_american",
            "under_american",
            "novig_over",
        ]
    )


def _decimal(american) -> np.ndarray:
    price = np.asarray(american, dtype=float)
    return np.where(price > 0, 1.0 + price / 100.0, 1.0 + 100.0 / np.abs(price))


def _history_keys(history: pd.DataFrame) -> pd.DataFrame:
    needed = ["player_id", "team_id", "game_id", "game_date", "season_year"]
    missing = [column for column in needed if column not in history.columns]
    if missing:
        raise ValueError(f"history is missing {missing}")
    hist = history.loc[:, needed].copy()
    hist["player_id"] = hist["player_id"].astype(str)
    hist["team_id"] = hist["team_id"].astype(str)
    hist["game_id"] = hist["game_id"].astype(str)
    hist["season_year"] = hist["season_year"].astype(str)
    hist["game_date"] = pd.to_datetime(hist["game_date"])
    return hist


def _loo_mean(total: _Cell | None, own: _Cell | None, min_n: int) -> float | None:
    if total is None:
        return None
    own_weight = 0.0 if own is None else own.weight
    own_sum = 0.0 if own is None else own.weighted
    own_count = 0 if own is None else own.count
    remain_n = total.count - own_count
    remain_w = total.weight - own_weight
    if remain_n < min_n or remain_w <= 0:
        return None
    return (total.weighted - own_sum) / remain_w


def _add_cell(
    cells, players, group: str, player: str, weight: float, residual: float
) -> None:
    cell = cells.get(group)
    if cell is None:
        cell = _Cell()
        cells[group] = cell
    cell.weight += weight
    cell.weighted += weight * residual
    cell.count += 1
    bucket = players.setdefault(group, {})
    owned = bucket.get(player)
    if owned is None:
        owned = _Cell()
        bucket[player] = owned
    owned.weight += weight
    owned.weighted += weight * residual
    owned.count += 1


def _tau2(residual, weight, players, coarse, coarse_cells: dict[str, _Cell]) -> float:
    demeaned = residual.copy()
    for index, group in enumerate(coarse):
        cell = coarse_cells[group]
        demeaned[index] = residual[index] - cell.weighted / cell.weight
    means = []
    sampling = []
    for player in pd.unique(players):
        chosen = players == player
        if int(chosen.sum()) < 2:
            continue
        precision = float(weight[chosen].sum())
        means.append(float(np.sum(weight[chosen] * demeaned[chosen]) / precision))
        sampling.append(1.0 / precision)
    if len(means) < 2:
        return 0.0
    between = float(np.var(np.asarray(means), ddof=1))
    return max(0.0, between - float(np.mean(sampling)))


def _total_loglik(
    frame,
    prior: GroupPrior,
    per_game: float,
    offseason: float,
    initial: float,
    *,
    obs_var: float = 1.0,
) -> float:
    total = 0.0
    ordered = frame.sort_values(["player_id", "game_date"], kind="mergesort")
    for player, part in ordered.groupby(ordered["player_id"].astype(str), sort=False):
        fine = part["fine_group"].astype(str)
        coarse = part["coarse_group"].astype(str)
        mean0 = prior.mean_for(player, fine.iloc[0], coarse.iloc[0])
        gmean = np.array(
            [
                prior.mean_for(player, item, other)
                for item, other in zip(fine, coarse, strict=True)
            ]
        )
        gtau = np.array([prior.tau2_for(item) for item in fine])
        path = filter_player(
            part["residual"].to_numpy(dtype=float),
            np.full(len(part), math.sqrt(obs_var)),
            part["season_year"].astype(str).to_numpy(),
            part["team_id"].astype(str).to_numpy(),
            part["rest_days"].to_numpy(dtype=float),
            np.full(len(part), per_game),
            offseason=offseason,
            initial=initial,
            initial_mean=mean0,
            reset_on_new_team=True,
            group_mean=gmean,
            group_tau2=gtau,
        )
        total += path.loglik
    return total


def fit_point_tails(y, knots, groups) -> PointTails:
    """Lower ratios ``y / q05`` and upper excesses ``y - q95``, by group."""
    outcome = np.asarray(y, dtype=float).reshape(-1)
    grid = np.asarray(knots, dtype=float)
    group_ids = np.asarray(groups, dtype=int).reshape(-1)
    if grid.shape != (outcome.shape[0], len(QUANTILES)):
        raise ValueError("tail knots must match the quantile grid")
    if not np.isfinite(grid).all() or not np.isfinite(outcome).all():
        raise ValueError("tail outcomes and knots must be finite")
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for index in range(outcome.shape[0]):
        group = int(group_ids[index])
        q05 = float(grid[index, 0])
        q95 = float(grid[index, -1])
        value = float(outcome[index])
        if value < q05:
            buckets[("lower", group)].append(0.0 if q05 <= 0 else value / q05)
        if value > q95:
            buckets[("upper", group)].append(value - q95)
    seen = sorted({int(group) for group in group_ids}) or [0]
    arrays: dict[tuple[str, int], np.ndarray] = {}
    for group in seen:
        arrays[("lower", group)] = np.sort(
            np.asarray([*buckets.get(("lower", group), []), 1.0], dtype=float)
        )
        arrays[("upper", group)] = np.sort(
            np.asarray([*buckets.get(("upper", group), []), 0.0], dtype=float)
        )
    return PointTails(arrays=arrays)


def normal_scores(y, knots, jitter, tails: PointTails, groups) -> tuple[np.ndarray, np.ndarray]:
    """Jittered PIT, then the standard normal quantile. Positive means over."""
    outcome = np.asarray(y, dtype=float).reshape(-1)
    grid = np.asarray(knots, dtype=float)
    noise = np.asarray(jitter, dtype=float).reshape(-1)
    group_ids = np.asarray(groups, dtype=int).reshape(-1)
    if grid.shape != (outcome.shape[0], len(QUANTILES)):
        raise ValueError("normal-score knots must match the quantile grid")
    if noise.shape[0] != outcome.shape[0] or group_ids.shape[0] != outcome.shape[0]:
        raise ValueError("normal scores need one jitter and group per game")
    if not np.isfinite(noise).all() or np.any(np.abs(noise) > 0.5):
        raise ValueError("jitter must be finite and within half a point")
    pit = np.array(
        [
            _row_cdf(
                float(outcome[index] + noise[index]),
                grid[index],
                tails,
                int(group_ids[index]),
            )
            for index in range(outcome.shape[0])
        ]
    )
    pit = np.clip(pit, PIT_EPS, 1.0 - PIT_EPS)
    return pit, norm.ppf(pit)


def _window_tails_and_scores(y, knots, windows, groups, seed: int):
    fold = np.asarray(windows).astype(int)
    outcome = np.asarray(y, dtype=float)
    grid = np.asarray(knots, dtype=float)
    group_ids = np.asarray(groups, dtype=int)
    pit = np.empty(outcome.shape[0])
    scores = np.empty(outcome.shape[0])
    tails_by_window: dict[int, PointTails] = {}
    positive = sorted(int(value) for value in np.unique(fold) if int(value) > 0)
    if not positive:
        raise ValueError("normal scores need a positive walk-forward window")
    targets = positive + ([HOLDOUT_WINDOW] if bool(np.any(fold < 0)) else [])
    for window in targets:
        current = fold == window if window > 0 else fold < 0
        if not bool(current.any()):
            continue
        if window == positive[0]:
            past = fold == window
        elif window < 0:
            past = fold > 0
        else:
            past = (fold > 0) & (fold < window)
        tails = fit_point_tails(outcome[past], grid[past], group_ids[past])
        tails_by_window[int(window)] = tails
        jitter = np.random.default_rng(int(seed) + 1009 * (int(window) + 11)).uniform(
            -0.5, 0.5, int(current.sum())
        )
        pit[current], scores[current] = normal_scores(
            outcome[current], grid[current], jitter, tails, group_ids[current]
        )
    return pit, scores, tails_by_window


def _tail_group(tails: PointTails, group: int) -> int:
    if ("lower", int(group)) in tails.arrays:
        return int(group)
    return int(next(iter(tails.arrays))[1])


def _row_cdf(value: float, knots: np.ndarray, tails: PointTails, group: int) -> float:
    chosen = _tail_group(tails, group)
    probability, quantile = _row_breakpoints(knots, chosen, chosen, tails)
    return float(_cdf_from_knots(probability, quantile, np.array([value]))[0])


def _quantiles_at(uniforms, knots, levels, tails: PointTails | None, group: int) -> np.ndarray:
    if tails is None:
        return np.interp(np.asarray(uniforms, dtype=float), levels, knots)
    chosen = _tail_group(tails, group)
    probability, quantile = _row_breakpoints(knots, chosen, chosen, tails)
    order = np.argsort(probability, kind="mergesort")
    return np.interp(np.asarray(uniforms, dtype=float), probability[order], quantile[order])


def _shrink_toward_group(mean, variance, group_mean, group_tau2, index: int) -> float:
    if group_mean is None or group_tau2 is None:
        return float(mean)
    tau2 = float(np.asarray(group_tau2, dtype=float).reshape(-1)[index])
    target = float(np.asarray(group_mean, dtype=float).reshape(-1)[index])
    if tau2 <= 0 or variance <= 0 or not math.isfinite(target):
        return float(mean)
    precision = 1.0 / float(variance) + 1.0 / tau2
    return (float(mean) / float(variance) + target / tau2) / precision


def _opening_variance(prior: GroupPrior, process: ProcessNoise, part: pd.DataFrame, *, dynamic: bool) -> float:
    fine = str(part.iloc[0]["fine_group"])
    group_var = prior.tau2_for(fine)
    if not dynamic:
        return group_var if group_var > 0 else float(prior.tau2)
    if _wide_group(fine):
        return max(float(process.initial), group_var)
    return float(process.initial)


def _wide_group(key: str) -> bool:
    return "rookie=1" in key or "new=1" in key


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def _missed_steps(rest_days: float) -> int:
    if not math.isfinite(rest_days):
        return 1
    return max(1, int(round(float(rest_days) / 2.0)))


def _require_columns(frame: pd.DataFrame, columns: list[str], caller: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{caller} is missing {missing}")


def _require_ordered_windows(summary: pd.DataFrame) -> None:
    positive = summary.loc[summary.index > 0].sort_index()
    if positive.empty:
        return
    last = None
    for _window, row in positive.iterrows():
        if last is not None and row["first"] <= last:
            raise ValueError("walk-forward windows overlap or run backwards")
        last = row["last"]


def _lag_corr(values, player_id, lag: int) -> float:
    frame = pd.DataFrame({"z": np.asarray(values, dtype=float), "player_id": player_id})
    previous = frame.groupby("player_id", sort=False)["z"].shift(lag)
    pair = pd.DataFrame({"now": frame["z"], "prev": previous}).dropna()
    if len(pair) < 3:
        return float("nan")
    if float(pair["now"].std(ddof=1)) == 0 or float(pair["prev"].std(ddof=1)) == 0:
        return float("nan")
    return float(pair["now"].corr(pair["prev"]))


def _icc(values: np.ndarray, groups) -> float:
    frame = pd.DataFrame({"y": values, "g": np.asarray(groups).astype(str)})
    counts = frame.groupby("g", sort=False).size()
    keep = counts[counts >= 2].index
    frame = frame.loc[frame["g"].isin(keep)]
    if frame["g"].nunique() < 2:
        return float("nan")
    means = frame.groupby("g", sort=False)["y"].mean()
    sizes = frame.groupby("g", sort=False).size()
    grand = float(frame["y"].mean())
    n_obs = len(frame)
    n_groups = int(len(sizes))
    between = float((((means - grand) ** 2) * sizes).sum())
    within = float(((frame["y"] - frame["g"].map(means)) ** 2).sum())
    msb = between / (n_groups - 1)
    msw = within / (n_obs - n_groups)
    n0 = (n_obs - float((sizes.to_numpy(dtype=float) ** 2).sum()) / n_obs) / (
        n_groups - 1
    )
    if msw == 0:
        return 1.0
    return float((msb - msw) / (msb + (n0 - 1.0) * msw))


def _dependence_slices(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "season_boundary" in frame.columns:
        rows.append(
            _slice_row(frame, "season_boundary", frame["season_boundary"].astype(bool))
        )
        rows.append(
            _slice_row(frame, "within_season", ~frame["season_boundary"].astype(bool))
        )
    if "early" in frame.columns:
        rows.append(_slice_row(frame, "early", frame["early"].astype(bool)))
    if "tier" in frame.columns:
        for tier, part in frame.groupby("tier", observed=True, sort=True):
            rows.append(
                {
                    "slice": f"tier {tier}",
                    "n": int(len(part)),
                    "acf_1": _lag_corr(part["z"], part["player_id"], 1),
                    "icc": _icc(part["z"].to_numpy(dtype=float), part["player_id"]),
                }
            )
    return pd.DataFrame(rows)


def _slice_row(frame: pd.DataFrame, name: str, mask) -> dict[str, object]:
    part = frame.loc[np.asarray(mask, dtype=bool)]
    return {
        "slice": name,
        "n": int(len(part)),
        "acf_1": _lag_corr(part["z"], part["player_id"], 1)
        if len(part)
        else float("nan"),
        "icc": _icc(part["z"].to_numpy(dtype=float), part["player_id"])
        if len(part)
        else float("nan"),
    }
