"""Hierarchical Bayesian player-points model.

points = 2*FGM2 + 3*FGM3 + FTM

minutes ~ TruncatedNormal(role + game context + a player walk)
The minutes noise and the walk step are each a player-specific scale.
attempts ~ NegativeBinomial(minutes * rate)
makes ~ Binomial(attempts, p)

Attempt rates pool player effects by position and add team, opponent, and a
player-level Gaussian random walk over calendar blocks. Efficiency uses a
tighter position-pooled player effect plus an opponent effect. A posterior
predictive draw samples minutes, then attempts given those minutes, then makes
given those attempts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

MINUTES_UPPER = 72.0
# Opening night has no in-season gap. Seven days is a stated assumption so the
# rest slope is not fit to the offseason.
SEASON_OPENER_REST_DAYS = 7.0
TRAILING_WINDOW = 10
PACE_WINDOW = 10
# Inside each prior window, a game one half-life older counts half as much.
# Four games back weighs 1/2; the oldest game in a 10-game window weighs about 1/5.
PRIOR_HALF_LIFE_GAMES = 4.0

BASE_MINUTES_COLUMNS = (
    "spread",
    "total",
    "rest_days",
    "b2b",
    "minutes_trail",
    "starter",
)
BASE_RATE_COLUMNS = ("is_home", "pace_prior", "rest_days")
MINUTES_COLUMNS = BASE_MINUTES_COLUMNS
RATE_COLUMNS = BASE_RATE_COLUMNS
CONTEXT_COVARIATES = (
    "opp_def_prior",
    "opp_pace_prior",
    "stars_out",
    "usg_stars_in",
    "usg_stars_out",
    "points_trend",
)
AVAILABLE_COVARIATES = tuple(
    dict.fromkeys((*BASE_MINUTES_COLUMNS, *BASE_RATE_COLUMNS, *CONTEXT_COVARIATES))
)
STAR_SLOTS = 2
STAR_MIN_MINUTES = 24.0
TREND_SHORT = 3
TREND_LONG = 10
USG_WINDOW = 10
# Player-level sd of a varying slope, on that slope's own scale.
# Tight on purpose: a slope stays shared unless the data pulls this off zero.
VARYING_SLOPE_SD = 0.05
VARYING_SLOPE_NAMES = (
    "spread",
    "stars_out",
    "usg_stars_in",
    "usg_stars_out",
)
# Season-average minutes cutoffs for the optional role tier. Bench is below
# the first cutoff, starter is at or above the second.
ROLE_ROTATION_MINUTES = 15.0
ROLE_STARTER_MINUTES = 24.0

_ATTEMPTS = (
    ("fga2", float(np.log(0.22))),
    ("fga3", float(np.log(0.10))),
    ("fta", float(np.log(0.08))),
)
_MAKES = (
    ("fgm2", "fga2", 0.0),
    ("fgm3", "fga3", -0.6),
    ("ftm", "fta", 1.2),
)
_REQUIRED_COLUMNS = (
    "player_id",
    "team_id",
    "opp_team_id",
    "game_id",
    "game_date",
    "season_year",
    "pos",
    "matchup",
    "minutes",
    "fga",
    "fgm",
    "fg3_a",
    "fg3_m",
    "fta",
    "ftm",
    "pts",
    "start_position",
    "game_total",
    "player_team_spread",
    "team_pace",
)
_COUNT_COLUMNS = ("fga", "fgm", "fg3_a", "fg3_m", "fta", "ftm")
_DATA_KEYS = (
    "player",
    "team",
    "opp",
    "block",
    "pos_of_player",
    "X_min",
    "X_rate",
    "minutes",
    "fga2",
    "fga3",
    "fta",
    "fgm2",
    "fgm3",
    "ftm",
)


@dataclass(frozen=True)
class CovariateSpec:
    """Which prepared columns enter each equation, in coefficient order.

    Delete a name to drop that slope. The same name may sit in both
    equations. ``build_design`` standardizes whatever this lists.
    ``opp_def_prior`` needs ``team_def_rating``. The usage regimes need
    ``usg_pct``. Stars are the top ``star_slots`` teammates by prior minutes,
    counting only those whose prior mean is at least ``star_min_minutes``.
    ``points_trend`` is the prior short points mean minus the prior long mean.
    """

    minutes: tuple[str, ...] = BASE_MINUTES_COLUMNS
    rate: tuple[str, ...] = BASE_RATE_COLUMNS
    star_slots: int = STAR_SLOTS
    star_min_minutes: float = STAR_MIN_MINUTES
    trend_short: int = TREND_SHORT
    trend_long: int = TREND_LONG
    usg_window: int = USG_WINDOW
    prior_half_life: float = PRIOR_HALF_LIFE_GAMES


def expanded_covariates() -> CovariateSpec:
    """Base columns plus opponent context, stars, usage regimes, and form."""
    return CovariateSpec(
        minutes=BASE_MINUTES_COLUMNS
        + ("opp_def_prior", "opp_pace_prior", "stars_out", "points_trend"),
        rate=BASE_RATE_COLUMNS
        + (
            "opp_def_prior",
            "opp_pace_prior",
            "stars_out",
            "usg_stars_in",
            "usg_stars_out",
            "points_trend",
        ),
    )


@dataclass(frozen=True)
class HierarchicalDesign:
    """Training design: indices, standardized covariates, and the row audit."""

    data: dict[str, np.ndarray]
    n_players: int
    n_teams: int
    n_pos: int
    n_blocks: int
    player_ids: np.ndarray
    team_ids: np.ndarray
    positions: np.ndarray
    player_index: dict[Any, int]
    team_index: dict[Any, int]
    block_origin: pd.Timestamp
    block_days: int
    minutes_columns: tuple[str, ...]
    rate_columns: tuple[str, ...]
    star_slots: int
    star_min_minutes: float
    trend_short: int
    trend_long: int
    usg_window: int
    prior_half_life: float
    minutes_center: np.ndarray
    minutes_scale: np.ndarray
    rate_center: np.ndarray
    rate_scale: np.ndarray
    points: np.ndarray
    row_player_ids: np.ndarray
    row_game_ids: np.ndarray
    audit: dict[str, int]


def crps_ensemble(draws: np.ndarray, y: float) -> float:
    """Empirical CRPS for one player-game. Lower is better.

    Uses the sorted-draw identity for the pairwise term, the same one as
    ``src.models.evaluation.crps``.
    """
    samples = np.asarray(draws, dtype=float).reshape(-1)
    outcome = float(y)
    if samples.size == 0 or not np.isfinite(outcome) or not np.isfinite(samples).all():
        raise ValueError("CRPS inputs must be finite and non-empty")
    ordered = np.sort(samples)
    absolute_error = float(np.mean(np.abs(ordered - outcome)))
    n_draws = ordered.size
    weights = 2 * np.arange(1, n_draws + 1) - n_draws - 1
    pairwise = float(np.sum(weights * ordered) / (n_draws * n_draws))
    return absolute_error - pairwise


def prob_over(
    point_draws: np.ndarray,
    line: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """P(points > line) and P(points < line). Pushes count in neither.

    A half-point line has no push, so the two probabilities sum to one.
    """
    draws = np.asarray(point_draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError("point_draws must have shape (n_rows, n_draws)")
    if not np.isfinite(draws).all():
        raise ValueError("point_draws must be finite")
    threshold = np.asarray(line, dtype=float)
    if threshold.ndim == 0:
        threshold = np.broadcast_to(threshold, (draws.shape[0],))
    elif threshold.shape != (draws.shape[0],):
        raise ValueError("line must be a scalar or one threshold per row")
    if not np.isfinite(threshold).all():
        raise ValueError("line must be finite")
    over = np.mean(draws > threshold[:, None], axis=1)
    under = np.mean(draws < threshold[:, None], axis=1)
    return over, under


def build_design(
    frame: pd.DataFrame,
    *,
    block_days: int = 14,
    covariates: CovariateSpec | None = None,
) -> HierarchicalDesign:
    """Build the training design from player-game rows.

    Trailing minutes and the pace proxy are means of up to ten prior games.
    The current game does not enter either feature. Covariates are standardized
    with this frame's mean and standard deviation; score later games with
    ``transform_design`` so the scaler is not refit. Pass ``covariates`` to
    choose which prepared columns get a slope. Rows missing a selected column
    are removed and counted in the audit.
    """
    if block_days < 1:
        raise ValueError("block_days must be a positive integer")
    spec = covariates or CovariateSpec()
    _validate_spec(spec)
    prepared, audit = _prepare(frame, spec)
    prepared = _drop_incomplete_covariates(prepared, spec, audit)
    if prepared.empty:
        raise ValueError("no rows left after the design filters")
    return _compose_design(prepared, audit, block_days=block_days, spec=spec)


def transform_design(
    frame: pd.DataFrame,
    design: HierarchicalDesign,
    keep: pd.Series,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Covariates for ``keep`` rows, using the training scaler and index maps.

    ``frame`` has to include the earlier games those rows shift from. Block
    indices past the last training block use that last block: the fitted walk
    does not grow extra steps at prediction time.
    """
    if not isinstance(keep, pd.Series):
        raise TypeError("keep must be a boolean Series aligned to frame.index")
    if not frame.index.is_unique or not keep.index.equals(frame.index):
        raise ValueError("keep must use the same unique index as frame")
    if keep.dtype != bool:
        raise ValueError("keep must be boolean")
    spec = CovariateSpec(
        minutes=design.minutes_columns,
        rate=design.rate_columns,
        star_slots=design.star_slots,
        star_min_minutes=design.star_min_minutes,
        trend_short=design.trend_short,
        trend_long=design.trend_long,
        usg_window=design.usg_window,
        prior_half_life=design.prior_half_life,
    )
    prepared, _audit = _prepare(frame, spec)
    keep_labels = set(frame.index[keep.to_numpy()])
    chosen = prepared.loc[prepared["_source_index"].isin(keep_labels)].copy()
    removed = int(keep.sum()) - len(chosen)
    if removed:
        raise ValueError(f"{removed} scoring rows were removed while building features")
    if chosen.empty:
        raise ValueError("keep selected no rows")
    _require_finite_columns(chosen, design.minutes_columns + design.rate_columns)
    raw_minutes, raw_rate = _covariate_matrix(
        chosen, design.minutes_columns, design.rate_columns
    )
    data = _indexed_arrays(
        chosen,
        player_index=design.player_index,
        team_index=design.team_index,
        pos_of_player=np.asarray(design.data["pos_of_player"], dtype=int),
        minutes_x=_apply_scale(
            raw_minutes, design.minutes_center, design.minutes_scale
        ),
        rate_x=_apply_scale(raw_rate, design.rate_center, design.rate_scale),
        block=_blocks_for_scoring(chosen["game_date"], design),
    )
    raw_block = _raw_blocks(chosen["game_date"], design.block_origin, design.block_days)
    if np.any(raw_block < 0):
        raise ValueError("scoring rows precede the training block origin")
    return data, {
        "scoring_rows": int(len(chosen)),
        "block_indices_clamped": int(np.sum(raw_block > design.n_blocks - 1)),
    }


@dataclass(frozen=True)
class SlopeConfig:
    """Which of the four eligible slopes vary by player.

    A missing key is shared. ``opp_def_prior``, ``opp_pace_prior``, and
    ``points_trend`` are not in this map and always stay one league slope.
    The same column can sit in the minutes mean and in each attempt equation.
    Those are already separate coefficients, so a toggle pools each of them
    across players and does not force the attempt types to share one number.
    Minutes names are ``{column}_mu``, ``{column}_sd``, ``{column}_z``,
    ``{column}``. Attempt names are prefixed with the stat, for example
    ``fga2_usg_stars_out_mu``. With ``use_role_tier``, the player sd is
    ``{stem}_player_sd`` and the role term is ``{stem}_tier_sd`` and
    ``{stem}_tier_z``.
    """

    varying: Mapping[str, bool]
    use_role_tier: bool = False
    varying_sd: float = VARYING_SLOPE_SD

    def enabled(self, column: str) -> bool:
        return bool(self.varying.get(column, False))


def default_slope_config() -> SlopeConfig:
    """Roll out ``usg_stars_out`` first. The other three stay shared."""
    return SlopeConfig(
        varying={
            "usg_stars_out": True,
            "usg_stars_in": False,
            "stars_out": False,
            "spread": False,
        }
    )


def shared_slope_config() -> SlopeConfig:
    """Every eligible slope stays the original shared coefficient."""
    return SlopeConfig(varying={name: False for name in VARYING_SLOPE_NAMES})


def _player_scale(stem: str, log_center: float) -> pt.TensorVariable:
    """Positive per-player scale, pooled on the log.

    ``stem_log_mu`` is the population log mean, ``stem_log_sd`` is how far
    players spread around it, and ``stem_z`` is the unit deviation.
    """
    log_mu = pm.Normal(f"{stem}_log_mu", log_center, 0.5)
    log_sd = pm.HalfNormal(f"{stem}_log_sd", 0.5)
    z_player = pm.Normal(f"{stem}_z", 0, 1, dims="player")
    return pm.Deterministic(stem, pt.exp(log_mu + log_sd * z_player), dims="player")


def _any_varying(
    config: SlopeConfig,
    minutes_columns: tuple[str, ...] | None,
    rate_columns: tuple[str, ...] | None,
) -> bool:
    names = (*(minutes_columns or ()), *(rate_columns or ()))
    return any(config.enabled(name) for name in names)


def _validate_slope_config(
    config: SlopeConfig,
    minutes_columns: tuple[str, ...] | None,
    rate_columns: tuple[str, ...] | None,
) -> None:
    unknown = [name for name in config.varying if name not in VARYING_SLOPE_NAMES]
    if unknown:
        raise ValueError(
            f"unknown varying slopes {unknown}; "
            f"eligible columns are {VARYING_SLOPE_NAMES}"
        )
    if not np.isfinite(config.varying_sd) or config.varying_sd <= 0:
        raise ValueError("varying_sd must be a positive finite number")
    enabled = [name for name in VARYING_SLOPE_NAMES if config.enabled(name)]
    if not enabled:
        return
    if minutes_columns is None or rate_columns is None:
        raise ValueError("varying slopes need minutes_columns and rate_columns")
    present = set(minutes_columns) | set(rate_columns)
    missing = [name for name in enabled if name not in present]
    if missing:
        raise ValueError(f"varying slopes {missing} are not in the design columns")


def _role_tier(d: dict[str, np.ndarray], n_players: int) -> np.ndarray:
    """0 bench, 1 rotation, 2 starter, from each player's mean training minutes."""
    player = np.asarray(d["player"], dtype=int)
    minutes = np.asarray(d["minutes"], dtype=float)
    totals = np.bincount(player, weights=minutes, minlength=n_players)
    counts = np.bincount(player, minlength=n_players).astype(float)
    means = np.divide(totals, counts, out=np.zeros(n_players), where=counts > 0)
    tier = np.zeros(n_players, dtype=np.int64)
    tier[means >= ROLE_ROTATION_MINUTES] = 1
    tier[means >= ROLE_STARTER_MINUTES] = 2
    return tier


def _varying_slope(
    stem: str,
    prior_sd: float,
    config: SlopeConfig,
    tier: np.ndarray | None,
) -> pt.TensorVariable:
    """Non-centered player slope. The shared prior stays on ``{stem}_mu``."""
    mu = pm.Normal(f"{stem}_mu", 0.0, prior_sd)
    z_player = pm.Normal(f"{stem}_z", 0.0, 1.0, dims="player")
    if tier is None:
        sd = pm.HalfNormal(f"{stem}_sd", config.varying_sd)
        return pm.Deterministic(stem, mu + sd * z_player, dims="player")
    tier_sd = pm.HalfNormal(f"{stem}_tier_sd", config.varying_sd)
    player_sd = pm.HalfNormal(f"{stem}_player_sd", config.varying_sd)
    z_tier = pm.Normal(f"{stem}_tier_z", 0.0, 1.0, dims="role")
    role = pt.as_tensor(tier)
    return pm.Deterministic(
        stem,
        mu + tier_sd * z_tier[role] + player_sd * z_player,
        dims="player",
    )


def _linear_predictor(
    name: str,
    covariates: pt.TensorVariable,
    n_columns: int,
    columns: tuple[str, ...] | None,
    player_idx: pt.TensorVariable,
    *,
    prior_sd: float,
    config: SlopeConfig,
    tier: np.ndarray | None,
) -> pt.TensorVariable:
    """Shared dot product, with toggled columns replaced by player slopes.

    When no column in this equation varies, the shared coefficient keeps the
    name ``name`` and its original length. Toggled columns are dropped from
    that vector, in their original order, and enter as ``beta[player] * x``.
    """
    varying_at = []
    if columns is not None:
        varying_at = [
            index for index, column in enumerate(columns) if config.enabled(column)
        ]
    if columns is not None and len(columns) != n_columns:
        raise ValueError(f"{name} has {n_columns} columns but {len(columns)} names")
    if not varying_at:
        shared = pm.Normal(name, 0.0, prior_sd, shape=n_columns)
        return pt.dot(covariates, shared)
    shared_at = [index for index in range(n_columns) if index not in set(varying_at)]
    parts: list[pt.TensorVariable] = []
    if shared_at:
        shared = pm.Normal(name, 0.0, prior_sd, shape=len(shared_at))
        shared_x = pt.stack([covariates[:, index] for index in shared_at], axis=1)
        parts.append(pt.dot(shared_x, shared))
    for index in varying_at:
        column = columns[index] if columns is not None else str(index)
        stem = column if name == "b_min" else f"{name.removeprefix('b_')}_{column}"
        beta = _varying_slope(stem, prior_sd, config, tier)
        parts.append(beta[player_idx] * covariates[:, index])
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


def build_model(
    d: dict[str, np.ndarray],
    n_players: int,
    n_teams: int,
    n_pos: int,
    n_blocks: int,
    *,
    minutes_columns: tuple[str, ...] | None = None,
    rate_columns: tuple[str, ...] | None = None,
    slopes: SlopeConfig | None = None,
) -> pm.Model:
    """Assemble the minutes, attempt, and make model.

    ``pos_of_player`` is one position index per player id, not per game.
    Observed attempts are the binomial trial counts in training. In the
    posterior predictive those trial counts are the sampled attempt draws,
    because the binomial ``n`` is the attempt random variable.

    ``slopes`` pools ``spread``, ``stars_out``, ``usg_stars_in``, and
    ``usg_stars_out`` across players when their flags are true. Omit it, or
    pass ``shared_slope_config()``, and every coefficient stays the original
    shared vector ``b_min`` or ``b_{stat}``. Column names are required when
    any flag is true.
    """
    _validate_design(d, n_players, n_teams, n_pos, n_blocks)
    slope_config = slopes or shared_slope_config()
    _validate_slope_config(slope_config, minutes_columns, rate_columns)
    pos_of_player = np.asarray(d["pos_of_player"], dtype=int)
    tier = _role_tier(d, n_players) if slope_config.use_role_tier else None
    coords = {
        "player": np.arange(n_players),
        "team": np.arange(n_teams),
        "pos": np.arange(n_pos),
        "block": np.arange(n_blocks),
    }
    if tier is not None and _any_varying(slope_config, minutes_columns, rate_columns):
        coords["role"] = np.array(["bench", "rotation", "starter"])
    with pm.Model(coords=coords) as model:
        player_idx = pm.Data("player_idx", np.asarray(d["player"], dtype=int))
        team_idx = pm.Data("team_idx", np.asarray(d["team"], dtype=int))
        opp_idx = pm.Data("opp_idx", np.asarray(d["opp"], dtype=int))
        block_idx = pm.Data("block_idx", np.asarray(d["block"], dtype=int))
        x_min = pm.Data("X_min", np.asarray(d["X_min"], dtype=float))
        x_rate = pm.Data("X_rate", np.asarray(d["X_rate"], dtype=float))
        minutes_obs = pm.Data("minutes_obs", np.asarray(d["minutes"], dtype=float))

        mu_min0 = pm.Normal("min_mu0", 26, 8)
        player_minutes = pm.Normal("min_player", 0, 5, dims="player")
        # A typical player steps about a minute per block. The log spread
        # lets some players wander more than others.
        sd_min_drift = _player_scale("min_sd_drift", log_center=0.0)
        min_drift = pm.GaussianRandomWalk(
            "min_drift",
            sigma=sd_min_drift,
            init_dist=pm.Normal.dist(0, 2),
            dims=("player", "block"),
        )
        mu_minutes = (
            mu_min0
            + player_minutes[player_idx]
            + min_drift[player_idx, block_idx]
            + _linear_predictor(
                "b_min",
                x_min,
                int(np.asarray(d["X_min"]).shape[1]),
                minutes_columns,
                player_idx,
                prior_sd=3.0,
                config=slope_config,
                tier=tier,
            )
        )
        # A typical player is about six minutes of noise around that mean.
        sd_min = _player_scale("min_sd", log_center=float(np.log(6.0)))
        minutes = pm.TruncatedNormal(
            "minutes",
            mu=mu_minutes,
            sigma=sd_min[player_idx],
            lower=0,
            upper=MINUTES_UPPER,
            observed=minutes_obs,
        )

        attempts = {}
        for stat, base in _ATTEMPTS:
            player, team, opp, drift = _latent_effects(stat, pos_of_player)
            log_rate = (
                base
                + player[player_idx]
                + team[team_idx]
                + opp[opp_idx]
                + drift[player_idx, block_idx]
                + _linear_predictor(
                    f"b_{stat}",
                    x_rate,
                    int(np.asarray(d["X_rate"]).shape[1]),
                    rate_columns,
                    player_idx,
                    prior_sd=0.2,
                    config=slope_config,
                    tier=tier,
                )
            )
            dispersion = pm.Exponential(f"{stat}_disp", 0.5)
            observed = pm.Data(f"{stat}_obs", np.asarray(d[stat], dtype=int))
            attempts[stat] = pm.NegativeBinomial(
                stat,
                mu=minutes * pt.exp(log_rate),
                alpha=dispersion,
                observed=observed,
            )

        for stat, attempt_name, base in _MAKES:
            position = pm.Normal(f"{stat}_pos", 0, 0.15, dims="pos")
            sd_player = pm.HalfNormal(f"{stat}_sd_player", 0.15)
            z_player = pm.Normal(f"{stat}_z", 0, 1, dims="player")
            opponent = pm.Normal(f"{stat}_opp", 0, 0.08, dims="team")
            logit_p = (
                base
                + position[pos_of_player][player_idx]
                + sd_player * z_player[player_idx]
                + opponent[opp_idx]
            )
            observed = pm.Data(f"{stat}_obs", np.asarray(d[stat], dtype=int))
            pm.Binomial(
                stat,
                n=attempts[attempt_name],
                p=pm.math.sigmoid(logit_p),
                observed=observed,
            )
    return model


def fit(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    *,
    random_seed: int = 0,
    nuts_sampler: str | None = None,
    progressbar: bool = True,
):
    """Sample the posterior.

    The default sampler is PyMC NUTS. Pass ``nuts_sampler`` only when that
    backend is installed. The printed summary is the convergence check for
    scales and slopes; non-centered ``_z`` variables and the walk itself are
    omitted because there is one of each per player.
    """
    import arviz as az

    sample_kwargs: dict[str, Any] = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": 0.9,
        "random_seed": random_seed,
        "progressbar": progressbar,
    }
    if nuts_sampler is not None:
        sample_kwargs["nuts_sampler"] = nuts_sampler
    with model:
        idata = pm.sample(**sample_kwargs)
    divergences = int(idata.sample_stats["diverging"].sum())
    print(f"Divergences: {divergences}")
    summary = az.summary(
        idata,
        var_names=["~_z", "~_drift"],
        filter_vars="like",
    )
    print(summary.head(30))
    return idata


def simulate_points(
    model: pm.Model,
    idata,
    new_rows: dict[str, np.ndarray],
    *,
    random_seed: int = 0,
) -> np.ndarray:
    """Posterior predictive points with shape ``(n_rows, n_draws)``.

    One draw samples minutes, attempts given those minutes, and makes given
    those attempts. ``new_rows`` uses the training index maps and the training
    covariate scaling. Outcome fields on ``new_rows`` are ignored: the
    predictive replaces them.
    """
    player = np.asarray(new_rows["player"], dtype=int)
    n_rows = int(player.shape[0])
    payload = {
        "player_idx": player,
        "team_idx": np.asarray(new_rows["team"], dtype=int),
        "opp_idx": np.asarray(new_rows["opp"], dtype=int),
        "block_idx": np.asarray(new_rows["block"], dtype=int),
        "X_min": np.asarray(new_rows["X_min"], dtype=float),
        "X_rate": np.asarray(new_rows["X_rate"], dtype=float),
        "minutes_obs": np.full(n_rows, 20.0),
        "fga2_obs": np.zeros(n_rows, dtype=np.int64),
        "fga3_obs": np.zeros(n_rows, dtype=np.int64),
        "fta_obs": np.zeros(n_rows, dtype=np.int64),
        "fgm2_obs": np.zeros(n_rows, dtype=np.int64),
        "fgm3_obs": np.zeros(n_rows, dtype=np.int64),
        "ftm_obs": np.zeros(n_rows, dtype=np.int64),
    }
    observed_names = ("minutes", "fga2", "fga3", "fta", "fgm2", "fgm3", "ftm")
    with model:
        pm.set_data(payload)
        predictive = pm.sample_posterior_predictive(
            idata,
            var_names=list(observed_names),
            predictions=True,
            random_seed=random_seed,
        )
    samples = predictive.predictions
    for made, attempted in (("fgm2", "fga2"), ("fgm3", "fga3"), ("ftm", "fta")):
        if np.any(
            np.asarray(samples[made].values) > np.asarray(samples[attempted].values)
        ):
            raise AssertionError(f"sampled {made} exceeded sampled {attempted}")
    twos = _as_draw_matrix(samples["fgm2"])
    threes = _as_draw_matrix(samples["fgm3"])
    free_throws = _as_draw_matrix(samples["ftm"])
    return 2.0 * twos + 3.0 * threes + free_throws


def _latent_effects(name: str, pos_of_player: np.ndarray, prior_sd: float = 0.3):
    """Player effect pooled by position, plus team, opponent, and a slow walk."""
    position = pm.Normal(f"{name}_pos", 0, prior_sd, dims="pos")
    sd_player = pm.HalfNormal(f"{name}_sd_player", prior_sd)
    z_player = pm.Normal(f"{name}_z", 0, 1, dims="player")
    player = position[pos_of_player] + sd_player * z_player
    team = pm.Normal(f"{name}_team", 0, 0.15, dims="team")
    opponent = pm.Normal(f"{name}_opp", 0, 0.15, dims="team")
    # Random walk rather than a mean-reverting process. Switch to an AR(1)
    # only if a held-out season prefers it.
    sd_drift = pm.HalfNormal(f"{name}_sd_drift", 0.05)
    drift = pm.GaussianRandomWalk(
        f"{name}_drift",
        sigma=sd_drift,
        init_dist=pm.Normal.dist(0, 0.05),
        dims=("player", "block"),
    )
    return player, team, opponent, drift


def _validate_design(
    d: dict[str, np.ndarray],
    n_players: int,
    n_teams: int,
    n_pos: int,
    n_blocks: int,
) -> None:
    missing = [key for key in _DATA_KEYS if key not in d]
    if missing:
        raise ValueError(f"design is missing {missing}")
    counts = {
        name: int(value)
        for name, value in (
            ("n_players", n_players),
            ("n_teams", n_teams),
            ("n_pos", n_pos),
            ("n_blocks", n_blocks),
        )
    }
    if any(value < 1 for value in counts.values()):
        raise ValueError(f"index sizes must be positive: {counts}")
    n_rows = int(np.asarray(d["player"]).shape[0])
    if n_rows < 1:
        raise ValueError("design has no rows")
    for key in (
        "team",
        "opp",
        "block",
        "minutes",
        "fga2",
        "fga3",
        "fta",
        "fgm2",
        "fgm3",
        "ftm",
    ):
        if int(np.asarray(d[key]).shape[0]) != n_rows:
            raise ValueError(f"{key} must have one entry per row")
    pos_of_player = np.asarray(d["pos_of_player"])
    if pos_of_player.shape != (n_players,):
        raise ValueError(
            "pos_of_player must contain one position index per player; "
            f"got shape {pos_of_player.shape} for {n_players} players"
        )
    _require_index("player", d["player"], n_players)
    _require_index("team", d["team"], n_teams)
    _require_index("opp", d["opp"], n_teams)
    _require_index("block", d["block"], n_blocks)
    _require_index("pos_of_player", pos_of_player, n_pos)
    x_min = np.asarray(d["X_min"], dtype=float)
    x_rate = np.asarray(d["X_rate"], dtype=float)
    if x_min.ndim != 2 or x_min.shape[0] != n_rows or x_min.shape[1] < 1:
        raise ValueError("X_min must have shape (n_rows, n_covariates)")
    if x_rate.ndim != 2 or x_rate.shape[0] != n_rows or x_rate.shape[1] < 1:
        raise ValueError("X_rate must have shape (n_rows, n_covariates)")
    if not np.isfinite(x_min).all() or not np.isfinite(x_rate).all():
        raise ValueError("covariates must be finite")
    minutes = np.asarray(d["minutes"], dtype=float)
    if np.any((minutes <= 0) | (minutes > MINUTES_UPPER) | ~np.isfinite(minutes)):
        raise ValueError(f"minutes must be finite and inside (0, {MINUTES_UPPER}]")
    for made, attempted, _base in _MAKES:
        makes = np.asarray(d[made])
        trials = np.asarray(d[attempted])
        if np.any(makes < 0) or np.any(trials < 0) or np.any(makes > trials):
            raise ValueError(f"{made} must be between 0 and {attempted}")


def _require_index(name: str, values: np.ndarray, size: int) -> None:
    index = np.asarray(values)
    if index.size and (np.min(index) < 0 or np.max(index) >= size):
        raise ValueError(f"{name} contains an index outside 0..{size - 1}")


def _prepare(
    frame: pd.DataFrame, spec: CovariateSpec
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.duplicated(["player_id", "game_id"]).any():
        raise ValueError("duplicate player/game rows")
    optional = [
        column for column in ("team_def_rating", "usg_pct") if column in frame.columns
    ]
    result = frame.loc[:, list(_REQUIRED_COLUMNS) + optional].copy()
    result["_source_index"] = frame.index
    result["game_date"] = pd.to_datetime(result["game_date"], errors="raise")
    result["minutes"] = _numeric(result["minutes"], "minutes")
    for column in ("game_total", "player_team_spread", "team_pace", *optional):
        result[column] = _numeric(result[column], column)
    for column in _COUNT_COLUMNS:
        result[column] = _numeric(result[column], column)

    audit: dict[str, int] = {"input_rows": int(len(result))}
    played = result["minutes"].gt(0)
    audit["nonpositive_minute_rows_removed"] = int((~played).sum())
    result = result.loc[played].copy()

    shot_missing = result.loc[:, list(_COUNT_COLUMNS)].isna().any(axis=1)
    audit["missing_shot_rows_removed"] = int(shot_missing.sum())
    result = result.loc[~shot_missing].copy()
    if result["minutes"].isna().any():
        raise ValueError("minutes contains null values after the played-row filter")

    for column in _COUNT_COLUMNS:
        result[column] = _whole_counts(result[column], column)
    _reject_impossible_shots(result)
    above_cap = result["minutes"].gt(MINUTES_UPPER)
    if above_cap.any():
        raise ValueError(
            f"{int(above_cap.sum())} rows have minutes above {MINUTES_UPPER}; "
            "the truncated minutes likelihood cannot include them"
        )
    points = _numeric(result["pts"], "pts")
    reconstructed = (
        2 * (result["fgm"] - result["fg3_m"]) + 3 * result["fg3_m"] + result["ftm"]
    )
    if points.isna().any() or np.any(
        np.abs(points.to_numpy() - reconstructed.to_numpy()) > 1e-6
    ):
        raise ValueError("pts does not equal 2*FGM2 + 3*FGM3 + FTM")
    result["pts"] = reconstructed

    result = result.sort_values(
        ["player_id", "game_date", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    result["minutes_trail"] = _grouped_prior_mean(
        result,
        "minutes",
        "player_id",
        TRAILING_WINDOW,
        spec.prior_half_life,
    )
    result = _attach_pace_prior(result, spec.prior_half_life)
    result["rest_days"], result["b2b"], openers = _rest_features(result)
    audit["season_opener_rest_filled"] = int(openers.sum())
    result["starter"] = (
        result["start_position"]
        .astype("string")
        .str.strip()
        .isin(["G", "F", "C"])
        .astype(float)
    )
    result["is_home"] = (
        result["matchup"]
        .astype("string")
        .str.contains("vs", case=False, regex=False)
        .astype(float)
    )
    result["spread"] = result["player_team_spread"]
    result["total"] = result["game_total"]
    result = _attach_context(result, spec, audit)
    modes, mixed, missing_positions = _modal_positions(result)
    audit["missing_position_filled"] = missing_positions
    audit["players_with_mixed_positions"] = mixed
    result["position"] = result["player_id"].map(modes)

    market_missing = result["spread"].isna() | result["total"].isna()
    audit["missing_market_rows_removed"] = int(market_missing.sum())
    result = result.loc[~market_missing].copy()
    missing_trail = result["minutes_trail"].isna()
    audit["no_prior_minutes_rows_removed"] = int(missing_trail.sum())
    result = result.loc[~missing_trail].copy()
    missing_pace = result["pace_prior"].isna()
    audit["no_prior_pace_rows_removed"] = int(missing_pace.sum())
    result = result.loc[~missing_pace].copy()
    audit["output_rows"] = int(len(result))
    return result.reset_index(drop=True), audit


def _compose_design(
    prepared: pd.DataFrame,
    audit: dict[str, int],
    *,
    block_days: int,
    spec: CovariateSpec,
) -> HierarchicalDesign:
    player_ids = np.asarray(sorted(prepared["player_id"].unique().tolist()))
    team_ids = np.asarray(
        sorted(set(prepared["team_id"]).union(prepared["opp_team_id"]), key=str)
    )
    positions = np.asarray(sorted(prepared["position"].unique().tolist()))
    player_index = {player_id: index for index, player_id in enumerate(player_ids)}
    team_index = {team_id: index for index, team_id in enumerate(team_ids)}
    position_index = {position: index for index, position in enumerate(positions)}
    pos_of_player = np.asarray(
        [
            position_index[
                prepared.loc[prepared["player_id"].eq(player_id), "position"].iloc[0]
            ]
            for player_id in player_ids
        ],
        dtype=int,
    )
    origin = pd.Timestamp(prepared["game_date"].min()).normalize()
    block = _raw_blocks(prepared["game_date"], origin, block_days)
    raw_minutes, raw_rate = _covariate_matrix(prepared, spec.minutes, spec.rate)
    minutes_center, minutes_scale = _fit_scale(raw_minutes)
    rate_center, rate_scale = _fit_scale(raw_rate)
    data = _indexed_arrays(
        prepared,
        player_index=player_index,
        team_index=team_index,
        pos_of_player=pos_of_player,
        minutes_x=_apply_scale(raw_minutes, minutes_center, minutes_scale),
        rate_x=_apply_scale(raw_rate, rate_center, rate_scale),
        block=block,
    )
    return HierarchicalDesign(
        data=data,
        n_players=int(len(player_ids)),
        n_teams=int(len(team_ids)),
        n_pos=int(len(positions)),
        n_blocks=int(block.max()) + 1,
        player_ids=player_ids,
        team_ids=team_ids,
        positions=positions,
        player_index=player_index,
        team_index=team_index,
        block_origin=origin,
        block_days=block_days,
        minutes_columns=spec.minutes,
        rate_columns=spec.rate,
        star_slots=spec.star_slots,
        star_min_minutes=spec.star_min_minutes,
        trend_short=spec.trend_short,
        trend_long=spec.trend_long,
        usg_window=spec.usg_window,
        prior_half_life=spec.prior_half_life,
        minutes_center=minutes_center,
        minutes_scale=minutes_scale,
        rate_center=rate_center,
        rate_scale=rate_scale,
        points=prepared["pts"].to_numpy(dtype=float),
        row_player_ids=prepared["player_id"].to_numpy(),
        row_game_ids=prepared["game_id"].to_numpy(),
        audit=audit,
    )


def _indexed_arrays(
    prepared: pd.DataFrame,
    *,
    player_index: dict[Any, int],
    team_index: dict[Any, int],
    pos_of_player: np.ndarray,
    minutes_x: np.ndarray,
    rate_x: np.ndarray,
    block: np.ndarray,
) -> dict[str, np.ndarray]:
    unseen_players = sorted(set(prepared["player_id"]).difference(player_index))
    if unseen_players:
        raise ValueError(
            f"{len(unseen_players)} scoring player ids are outside the training design"
        )
    unseen_teams = sorted(
        set(prepared["team_id"]).union(prepared["opp_team_id"]).difference(team_index)
    )
    if unseen_teams:
        raise ValueError(
            f"{len(unseen_teams)} scoring team ids are outside the training design"
        )
    return {
        "player": prepared["player_id"].map(player_index).to_numpy(dtype=int),
        "team": prepared["team_id"].map(team_index).to_numpy(dtype=int),
        "opp": prepared["opp_team_id"].map(team_index).to_numpy(dtype=int),
        "block": np.asarray(block, dtype=int),
        "pos_of_player": np.asarray(pos_of_player, dtype=int),
        "X_min": np.asarray(minutes_x, dtype=float),
        "X_rate": np.asarray(rate_x, dtype=float),
        "minutes": prepared["minutes"].to_numpy(dtype=float),
        "fga2": (prepared["fga"] - prepared["fg3_a"]).to_numpy(dtype=int),
        "fga3": prepared["fg3_a"].to_numpy(dtype=int),
        "fta": prepared["fta"].to_numpy(dtype=int),
        "fgm2": (prepared["fgm"] - prepared["fg3_m"]).to_numpy(dtype=int),
        "fgm3": prepared["fg3_m"].to_numpy(dtype=int),
        "ftm": prepared["ftm"].to_numpy(dtype=int),
    }


def _blocks_for_scoring(dates: pd.Series, design: HierarchicalDesign) -> np.ndarray:
    raw = _raw_blocks(dates, design.block_origin, design.block_days)
    if np.any(raw < 0):
        raise ValueError("scoring rows precede the training block origin")
    return np.minimum(raw, design.n_blocks - 1)


def _raw_blocks(dates: pd.Series, origin: pd.Timestamp, block_days: int) -> np.ndarray:
    day_number = (
        pd.to_datetime(dates).dt.normalize() - pd.Timestamp(origin).normalize()
    ).dt.days
    return (day_number.to_numpy(dtype=int) // block_days).astype(int)


def _validate_spec(spec: CovariateSpec) -> None:
    if spec.star_slots < 1:
        raise ValueError("star_slots must be positive")
    if not np.isfinite(spec.star_min_minutes) or spec.star_min_minutes <= 0:
        raise ValueError("star_min_minutes must be a positive finite number")
    if spec.trend_short < 1 or spec.trend_long <= spec.trend_short:
        raise ValueError("trend_long must be longer than trend_short")
    if spec.usg_window < 1:
        raise ValueError("usg_window must be positive")
    if not np.isfinite(spec.prior_half_life) or spec.prior_half_life <= 0:
        raise ValueError("prior_half_life must be a positive finite number")
    for label, names in (("minutes", spec.minutes), ("rate", spec.rate)):
        if len(names) < 1:
            raise ValueError(f"{label} covariates need at least one column")
        if len(set(names)) != len(names):
            raise ValueError(f"{label} covariates contain a duplicate name")
        unknown = [name for name in names if name not in AVAILABLE_COVARIATES]
        if unknown:
            available = ", ".join(AVAILABLE_COVARIATES)
            raise ValueError(
                f"unknown {label} covariates {unknown}; "
                f"available columns are {available}"
            )


def _drop_incomplete_covariates(
    frame: pd.DataFrame,
    spec: CovariateSpec,
    audit: dict[str, int],
) -> pd.DataFrame:
    requested = tuple(dict.fromkeys((*spec.minutes, *spec.rate)))
    for name in requested:
        if name in frame.columns:
            continue
        if name == "opp_def_prior":
            raise ValueError("opp_def_prior needs team_def_rating")
        if name in ("usg_stars_in", "usg_stars_out"):
            raise ValueError(f"{name} needs usg_pct")
        raise ValueError(f"covariate {name} was not prepared")
    keep = np.ones(len(frame), dtype=bool)
    for name in requested:
        bad = ~np.isfinite(frame[name].to_numpy(dtype=float))
        if bad.any():
            audit[f"missing_{name}_rows"] = int(bad.sum())
        keep &= ~bad
    audit["missing_covariate_rows_removed"] = int((~keep).sum())
    chosen = frame.loc[keep].reset_index(drop=True)
    audit["output_rows"] = int(len(chosen))
    return chosen


def _require_finite_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [
        name
        for name in dict.fromkeys(columns)
        if name not in frame.columns
        or not np.isfinite(frame[name].to_numpy(dtype=float)).all()
    ]
    if missing:
        raise ValueError(f"scoring covariates are not finite: {missing}")


def _attach_context(
    frame: pd.DataFrame,
    spec: CovariateSpec,
    audit: dict[str, int],
) -> pd.DataFrame:
    result = frame.copy()
    result["opp_pace_prior"] = _opponent_prior(
        result, "team_pace", PACE_WINDOW, spec.prior_half_life
    )
    if "team_def_rating" in result.columns:
        result["opp_def_prior"] = _opponent_prior(
            result, "team_def_rating", PACE_WINDOW, spec.prior_half_life
        )
    result["stars_out"] = _stars_out(
        result, spec.star_slots, spec.star_min_minutes, spec.prior_half_life
    )
    result["points_trend"] = _points_trend(
        result, spec.trend_short, spec.trend_long, spec.prior_half_life
    )
    if "usg_pct" in result.columns:
        usage_in, usage_out, filled_in, filled_out = _usage_regimes(
            result, spec.usg_window, spec.prior_half_life
        )
        result["usg_stars_in"] = usage_in
        result["usg_stars_out"] = usage_out
        audit["usg_stars_in_filled"] = filled_in
        audit["usg_stars_out_filled"] = filled_out
    else:
        audit["usg_stars_in_filled"] = 0
        audit["usg_stars_out_filled"] = 0
    return result


def _recency_prior(
    values: np.ndarray,
    window: int,
    half_life: float,
    *,
    include_current: bool = False,
) -> np.ndarray:
    """Weighted mean of up to ``window`` earlier rows. The newest weighs 1.

    A non-finite row stays in the window but contributes no weight. Pass
    ``include_current`` when the mean describes the row itself, as the star
    trail does for games already played.
    """
    decay = float(0.5 ** (1.0 / half_life))
    history = np.asarray(values, dtype=float)
    out = np.full(history.shape[0], np.nan)
    for index in range(history.shape[0]):
        end = index + 1 if include_current else index
        start = max(0, end - window)
        sample = history[start:end]
        if sample.size == 0:
            continue
        finite = np.isfinite(sample)
        if not finite.any():
            continue
        ages = np.arange(sample.size - 1, -1, -1, dtype=float)
        out[index] = np.average(sample[finite], weights=(decay**ages)[finite])
    return out


def _series_prior(
    values: pd.Series, window: int, half_life: float, *, include_current: bool = False
) -> pd.Series:
    return pd.Series(
        _recency_prior(
            values.to_numpy(dtype=float),
            window,
            half_life,
            include_current=include_current,
        ),
        index=values.index,
    )


def _opponent_prior(
    frame: pd.DataFrame, column: str, window: int, half_life: float
) -> np.ndarray:
    games = (
        frame.groupby(["team_id", "game_id"], as_index=False)
        .agg(game_date=("game_date", "min"), value=(column, "first"))
        .sort_values(["team_id", "game_date", "game_id"], kind="mergesort")
    )
    games["prior"] = games.groupby("team_id", sort=False)["value"].transform(
        lambda values: _series_prior(values, window, half_life)
    )
    opponent = games.rename(columns={"team_id": "opp_team_id"})
    merged = frame.merge(
        opponent.loc[:, ["opp_team_id", "game_id", "prior"]],
        on=["opp_team_id", "game_id"],
        how="left",
        validate="many_to_one",
    )
    return merged["prior"].to_numpy(dtype=float)


def _stars_out(
    frame: pd.DataFrame, slots: int, min_minutes: float, half_life: float
) -> np.ndarray:
    """How many of this team's stars have no positive-minute row in the game.

    A star is one of the top ``slots`` teammates by minutes entering the game,
    and only if that prior mean is at least ``min_minutes``. The mean uses
    earlier appearances for this team. A player who has already appeared for
    another team is no longer that team's star.
    """
    played = frame.loc[
        frame["minutes"].gt(0),
        ["season_year", "team_id", "player_id", "game_id", "game_date", "minutes"],
    ].copy()
    played["season_year"] = played["season_year"].astype(str)
    if played.empty:
        return np.zeros(len(frame), dtype=float)
    team_games = (
        played.groupby(["season_year", "team_id", "game_id"], as_index=False)
        .agg(game_date=("game_date", "min"))
        .sort_values(
            ["season_year", "team_id", "game_date", "game_id"], kind="mergesort"
        )
    )
    team_games["team_game_order"] = team_games.groupby(
        ["season_year", "team_id"], sort=False
    ).cumcount()
    appearances = played.merge(
        team_games.loc[:, ["season_year", "team_id", "game_id", "team_game_order"]],
        on=["season_year", "team_id", "game_id"],
        how="left",
        validate="many_to_one",
    )
    appearances = appearances.sort_values(
        ["player_id", "game_date", "game_id"], kind="mergesort"
    )
    appearances["next_team"] = appearances.groupby("player_id", sort=False)[
        "team_id"
    ].shift(-1)
    appearances["next_date"] = appearances.groupby("player_id", sort=False)[
        "game_date"
    ].shift(-1)
    appearances = appearances.sort_values(
        ["season_year", "team_id", "player_id", "team_game_order"],
        kind="mergesort",
    )
    appearances["trail_after"] = appearances.groupby(
        ["season_year", "team_id", "player_id"], sort=False
    )["minutes"].transform(
        lambda values: _series_prior(
            values, TRAILING_WINDOW, half_life, include_current=True
        )
    )
    history = appearances.loc[
        :,
        [
            "season_year",
            "team_id",
            "player_id",
            "team_game_order",
            "trail_after",
            "next_team",
            "next_date",
        ],
    ].rename(columns={"team_game_order": "appearance_order"})
    candidates = team_games.merge(
        history,
        on=["season_year", "team_id"],
        how="inner",
    )
    candidates = candidates.loc[
        candidates["team_game_order"].gt(candidates["appearance_order"])
    ]
    candidates = candidates.sort_values(
        ["season_year", "team_id", "game_id", "player_id", "appearance_order"],
        kind="mergesort",
    )
    latest = candidates.groupby(
        ["season_year", "team_id", "game_id", "player_id"], sort=False
    ).tail(1)
    still_here = (
        latest["next_team"].isna()
        | latest["next_team"].eq(latest["team_id"])
        | latest["next_date"].ge(latest["game_date"])
    )
    latest = latest.loc[still_here].sort_values(
        ["season_year", "team_id", "game_id", "trail_after", "player_id"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    latest["rank"] = (
        latest.groupby(["season_year", "team_id", "game_id"], sort=False).cumcount() + 1
    )
    stars = latest.loc[latest["rank"].le(slots) & latest["trail_after"].ge(min_minutes)]
    played_keys = ["season_year", "team_id", "game_id", "player_id"]
    played_flag = played.loc[:, played_keys].assign(played=1)
    stars = stars.merge(
        played_flag,
        on=["season_year", "team_id", "game_id", "player_id"],
        how="left",
    )
    absent = stars.loc[stars["played"].isna()]
    counts = absent.groupby(["season_year", "team_id", "game_id"], sort=False).size()
    keys = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "season_year": frame["season_year"].astype(str),
                "team_id": frame["team_id"],
                "game_id": frame["game_id"],
            }
        )
    )
    return keys.map(counts).fillna(0).to_numpy(dtype=float)


def _points_trend(
    frame: pd.DataFrame, short: int, long: int, half_life: float
) -> np.ndarray:
    ordered = frame.sort_values(["player_id", "game_date", "game_id"], kind="mergesort")
    grouped = ordered.groupby("player_id", sort=False)["pts"]
    short_mean = grouped.transform(
        lambda values: _series_prior(values, short, half_life)
    )
    long_mean = grouped.transform(lambda values: _series_prior(values, long, half_life))
    trend = (short_mean - long_mean).reindex(frame.index)
    return trend.to_numpy(dtype=float)


def _usage_regimes(
    frame: pd.DataFrame, window: int, half_life: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ordered = frame.sort_values(["player_id", "game_date", "game_id"], kind="mergesort")
    usage = ordered["usg_pct"]
    player = ordered["player_id"]

    def _prior(mask: pd.Series) -> pd.Series:
        masked = usage.where(mask)
        return masked.groupby(player, sort=False).transform(
            lambda values: _series_prior(values, window, half_life)
        )

    overall = usage.groupby(player, sort=False).transform(
        lambda values: _series_prior(values, window, half_life)
    )
    with_stars = _prior(ordered["stars_out"].eq(0))
    without_stars = _prior(ordered["stars_out"].gt(0))
    filled_in = with_stars.isna() & overall.notna()
    filled_out = without_stars.isna() & overall.notna()
    with_stars = with_stars.fillna(overall).reindex(frame.index)
    without_stars = without_stars.fillna(overall).reindex(frame.index)
    return (
        with_stars.to_numpy(dtype=float),
        without_stars.to_numpy(dtype=float),
        int(filled_in.sum()),
        int(filled_out.sum()),
    )


def _covariate_matrix(
    prepared: pd.DataFrame,
    minutes_columns: tuple[str, ...],
    rate_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    minutes = np.column_stack(
        [prepared[column].to_numpy(dtype=float) for column in minutes_columns]
    )
    rate = np.column_stack(
        [prepared[column].to_numpy(dtype=float) for column in rate_columns]
    )
    if not np.isfinite(minutes).all() or not np.isfinite(rate).all():
        raise ValueError("covariates must be finite after the design filters")
    return minutes, rate


def _fit_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    # A constant training column has no slope information. Keep the column,
    # centered at that constant, so a later change still has a finite value.
    scale = np.where(scale == 0, 1.0, scale)
    return center.astype(float), scale.astype(float)


def _apply_scale(
    values: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (values - center) / scale


def _grouped_prior_mean(
    ordered: pd.DataFrame,
    column: str,
    group: str,
    window: int,
    half_life: float,
) -> np.ndarray:
    pieces: list[pd.Series] = []
    for _key, part in ordered.groupby(group, sort=False):
        pieces.append(_series_prior(part[column], window, half_life))
    return pd.concat(pieces).to_numpy(dtype=float)


def _attach_pace_prior(frame: pd.DataFrame, half_life: float) -> pd.DataFrame:
    team_games = (
        frame.groupby(["team_id", "game_id"], as_index=False)
        .agg(game_date=("game_date", "min"), team_pace=("team_pace", "first"))
        .sort_values(["team_id", "game_date", "game_id"], kind="mergesort")
    )
    team_games["pace_prior"] = _grouped_prior_mean(
        team_games,
        "team_pace",
        "team_id",
        PACE_WINDOW,
        half_life,
    )
    merged = frame.merge(
        team_games.loc[:, ["team_id", "game_id", "pace_prior"]],
        on=["team_id", "game_id"],
        how="left",
        validate="many_to_one",
    )
    return merged


def _rest_features(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    ordered = frame.sort_values(
        ["player_id", "season_year", "game_date", "game_id"],
        kind="mergesort",
    )
    gaps = (
        ordered.groupby(["player_id", "season_year"], sort=False)["game_date"]
        .diff()
        .dt.days
    )
    opener = gaps.isna()
    rest = gaps.fillna(SEASON_OPENER_REST_DAYS)
    back_to_back = (rest.le(1) & ~opener).astype(float)
    aligned = pd.DataFrame(
        {"rest_days": rest, "b2b": back_to_back, "opener": opener.to_numpy()},
        index=ordered.index,
    ).reindex(frame.index)
    return aligned["rest_days"], aligned["b2b"], aligned["opener"]


def _modal_positions(frame: pd.DataFrame) -> tuple[dict[Any, str], int, int]:
    labels = frame["pos"].astype("string").str.strip()
    missing = (
        labels.isna() | labels.eq("") | labels.str.lower().isin(["nan", "none", "<na>"])
    )
    labels = labels.mask(missing, "UNK")
    modes: dict[Any, str] = {}
    mixed = 0
    for player_id, player_labels in labels.groupby(frame["player_id"], sort=False):
        counts = player_labels.value_counts()
        top = counts.max()
        winners = sorted(str(label) for label in counts.loc[counts.eq(top)].index)
        modes[player_id] = winners[0]
        if player_labels.nunique() > 1:
            mixed += 1
    return modes, mixed, int(missing.sum())


def _reject_impossible_shots(frame: pd.DataFrame) -> None:
    twos_made = frame["fgm"] - frame["fg3_m"]
    twos_attempted = frame["fga"] - frame["fg3_a"]
    checks = (
        ("fgm2", twos_made, twos_attempted),
        ("fgm3", frame["fg3_m"], frame["fg3_a"]),
        ("ftm", frame["ftm"], frame["fta"]),
    )
    for name, made, attempted in checks:
        if np.any((attempted < 0) | (made < 0) | (made > attempted)):
            raise ValueError(f"{name} must be between 0 and its attempts")


def _numeric(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        raise ValueError(f"{name} contains {int(invalid.sum())} non-numeric values")
    return numeric


def _whole_counts(series: pd.Series, name: str) -> pd.Series:
    values = series.to_numpy(dtype=float)
    rounded = np.rint(values)
    if np.any(~np.isfinite(values)) or np.any(np.abs(values - rounded) > 1e-6):
        raise ValueError(f"{name} contains values that are not whole-number counts")
    return pd.Series(rounded.astype(np.int64), index=series.index)


def _as_draw_matrix(data_array) -> np.ndarray:
    values = np.asarray(data_array.values)
    if values.ndim != 3:
        raise ValueError(
            f"predictive array must have shape (chain, draw, row), got {values.shape}"
        )
    n_chain, n_draw, n_rows = values.shape
    return values.transpose(2, 0, 1).reshape(n_rows, n_chain * n_draw)
