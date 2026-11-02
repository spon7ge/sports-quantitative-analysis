"""Empirical-Bayes NBA points model with optional baseline-linked noise.

Mean model
----------
Global simplex weights on prior 5/10/20 game point averages form baseline B.
Shared slopes adjust contextual features (starter surprise, schedule/market
context, and scoring-volume trends). Player intercepts are partially pooled
Gaussians with empirical-Bayes variance.

Observation variance
--------------------
Shared mode uses one noise variance. Baseline mode uses

    log(sigma_i^2) = gamma0 + gamma1 * (log(1 + B_i) - c)

with training-set center c. Variance hyperparameters and baseline weights are
plug-in empirical-Bayes estimates; their uncertainty is not integrated out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

WINDOW_COLS = ("p5", "p10", "p20")

# Default contextual specification: spread + implied team total (not game total).
# Including spread, game total, and implied team total together is redundant.
CONTEXT_COLS = (
    "starter_surprise",
    "is_home",
    "back2back",
    "opp_def_rating_prior",
    "team_spread_pregame",
    "implied_team_total_pregame",
    "usage_trend",
    "minutes_trend",
    "fga_trend",
    "fta_trend",
)

# Prior SDs for shared coefficients in ORIGINAL feature units.
# Starting regularization settings, not empirically established NBA effects.
PRIOR_SD_RAW: dict[str, float] = {
    "level": 3.0,
    "starter_surprise": 5.0,
    "is_home": 2.0,
    "back2back": 2.0,
    "opp_def_rating_prior": 0.30,
    "team_spread_pregame": 0.15,
    "implied_team_total_pregame": 0.20,
    "game_total_pregame": 0.10,
    "usage_trend": 0.50,
    "minutes_trend": 0.75,
    "fga_trend": 1.0,
    "fta_trend": 0.75,
}

VarianceMode = Literal["shared", "baseline"]


def fit_window_weights(
    features: np.ndarray,
    target: np.ndarray,
    step: float = 0.01,
) -> np.ndarray:
    ticks = int(round(1.0 / step))
    if ticks < 1 or not np.isclose(ticks * step, 1.0):
        raise ValueError("GRID_STEP must divide 1 exactly.")

    grid = np.array(
        [
            (i / ticks, j / ticks, (ticks - i - j) / ticks)
            for i in range(ticks + 1)
            for j in range(ticks + 1 - i)
        ]
    )
    gram = features.T @ features
    rhs = features.T @ target
    sse = np.einsum("gi,ij,gj->g", grid, gram, grid) - 2 * (grid @ rhs)
    return grid[np.argmin(sse)].copy()


def observation_variance(
    baseline: np.ndarray,
    gamma0: float,
    gamma1: float,
    log_b_center: float,
) -> np.ndarray:
    """Return per-row observation variances from the log-linear baseline link."""
    baseline = np.asarray(baseline, dtype=float)
    return np.exp(gamma0 + gamma1 * (np.log1p(baseline) - log_b_center))


def shared_noise_posterior_moments(
    *,
    design: np.ndarray,
    target: np.ndarray,
    codes: np.ndarray,
    sigma2: np.ndarray,
    player_var: float,
    prior_sd: np.ndarray | None = None,
    prior_precision: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Gaussian posterior moments for shared slopes and player intercepts.

    Observation noise may be heteroscedastic. Player intercepts are i.i.d.
    N(0, player_var). Shared coefficients have independent Gaussian priors
    encoded by ``prior_sd`` or an explicit diagonal ``prior_precision``.
    Returns the posterior after integrating the Gaussian likelihood.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    codes = np.asarray(codes, dtype=int)
    sigma2 = np.asarray(sigma2, dtype=float)

    if prior_precision is None:
        if prior_sd is None:
            raise ValueError("Provide prior_sd or prior_precision.")
        prior_sd = np.asarray(prior_sd, dtype=float)
        if not np.isfinite(prior_sd).all() or (prior_sd <= 0).any():
            raise ValueError("prior_sd must be finite and positive.")
        prior_precision = np.diag(1.0 / prior_sd**2)
    else:
        prior_precision = np.asarray(prior_precision, dtype=float)
        if prior_sd is None:
            diag = np.diag(prior_precision)
            if (diag <= 0).any():
                raise ValueError("prior_precision diagonal must be positive.")
            prior_sd = 1.0 / np.sqrt(diag)
        else:
            prior_sd = np.asarray(prior_sd, dtype=float)

    if not (
        np.isfinite(design).all()
        and np.isfinite(target).all()
        and np.isfinite(sigma2).all()
        and np.isfinite(player_var)
        and player_var > 0
        and (sigma2 > 0).all()
        and np.isfinite(prior_precision).all()
    ):
        raise ValueError(
            "Design, target, sigma2, prior, and player_var must be "
            "finite with positive variances."
        )

    n_players = int(codes.max()) + 1 if len(codes) else 0
    if n_players < 1:
        raise ValueError("Need at least one player code.")

    weight = 1.0 / sigma2
    player_precision = np.bincount(codes, weights=weight, minlength=n_players)
    between = player_var / (1.0 + player_var * player_precision)

    weighted_x = design * weight[:, None]
    xx = design.T @ weighted_x
    xy = weighted_x.T @ target
    yy = float(np.sum(weight * target**2))

    sum_wx = np.column_stack(
        [
            np.bincount(codes, weights=weight * design[:, j], minlength=n_players)
            for j in range(design.shape[1])
        ]
    )
    sum_wy = np.bincount(codes, weights=weight * target, minlength=n_players)

    precision = prior_precision + xx - sum_wx.T @ (between[:, None] * sum_wx)
    rhs = xy - sum_wx.T @ (between * sum_wy)
    chol = cho_factor(precision, lower=True, check_finite=False)
    beta = cho_solve(chol, rhs, check_finite=False)
    beta_cov = cho_solve(chol, np.eye(design.shape[1]), check_finite=False)

    conditional_var = 1.0 / (1.0 / player_var + player_precision)
    intercept = conditional_var * (sum_wy - sum_wx @ beta)
    coupling = conditional_var[:, None] * sum_wx
    intercept_marginal_var = conditional_var + np.einsum(
        "ni,ij,nj->n",
        coupling,
        beta_cov,
        coupling,
    )

    quadratic = yy - float(np.sum(between * sum_wy**2)) - float(rhs @ beta)
    logdet = (
        float(np.sum(np.log(sigma2)))
        + float(np.log1p(player_var * player_precision).sum())
        + float(np.log(prior_sd**2).sum())
        + 2.0 * float(np.log(np.diag(chol[0])).sum())
    )
    nll = 0.5 * (len(target) * np.log(2.0 * np.pi) + logdet + quadratic)

    return {
        "beta": beta,
        "beta_cov": beta_cov,
        "intercept": intercept,
        "coupling": coupling,
        "conditional_var": conditional_var,
        "intercept_marginal_var": intercept_marginal_var,
        "nll": float(nll),
        "chol": chol,
        "player_precision": player_precision,
        "sum_wx": sum_wx,
        "sum_wy": sum_wy,
        "prior_sd": prior_sd,
        "prior_precision": prior_precision,
    }


def _resolve_context_cols(
    context_cols: Sequence[str] | None,
) -> tuple[str, ...]:
    cols = tuple(CONTEXT_COLS if context_cols is None else context_cols)
    if not cols:
        raise ValueError("context_cols must be non-empty.")
    missing_prior = [column for column in cols if column not in PRIOR_SD_RAW]
    if missing_prior:
        raise ValueError(f"Missing PRIOR_SD_RAW entries: {missing_prior}")
    return cols


def _resolve_prior_sd_raw(
    prior_sd_raw: Mapping[str, float] | None,
    prior_sd_scale: float,
) -> dict[str, float]:
    if prior_sd_scale <= 0 or not np.isfinite(prior_sd_scale):
        raise ValueError("prior_sd_scale must be finite and positive.")
    source = PRIOR_SD_RAW if prior_sd_raw is None else dict(prior_sd_raw)
    return {key: float(value) * prior_sd_scale for key, value in source.items()}


def _coefficient_prior_sd(
    *,
    context_cols: Sequence[str],
    scale: np.ndarray,
    prior_sd_raw: Mapping[str, float],
    standardize: bool,
) -> np.ndarray:
    """Build prior SDs for [level, *context] under the active design scale."""
    level = float(prior_sd_raw["level"])
    if standardize:
        context = [
            float(prior_sd_raw[column]) * float(scale[index])
            for index, column in enumerate(context_cols)
        ]
    else:
        context = [float(prior_sd_raw[column]) for column in context_cols]
    return np.asarray([level, *context], dtype=float)


def _prepare_train(
    train: pd.DataFrame,
    grid_step: float,
    context_cols: Sequence[str],
    prior_sd_raw: Mapping[str, float],
    standardize: bool,
) -> dict[str, Any]:
    windows = train.loc[:, list(WINDOW_COLS)].to_numpy(dtype=float)
    actual = train["pts"].to_numpy(dtype=float)
    raw = train.loc[:, list(context_cols)].to_numpy(dtype=float)

    if not all(np.isfinite(a).all() for a in (windows, actual, raw)):
        raise ValueError("Training targets and features must be finite.")

    weights = fit_window_weights(windows, actual, grid_step)
    baseline = windows @ weights
    target = actual - baseline

    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    if standardize:
        design = np.column_stack([np.ones(len(train)), (raw - center) / scale])
        design_scale = scale.copy()
    else:
        design = np.column_stack([np.ones(len(train)), raw - center])
        design_scale = np.ones(len(context_cols), dtype=float)

    codes, ids = pd.factorize(train["player_id"], sort=True)
    n = np.bincount(codes).astype(float)
    if len(ids) < 2 or n.max() < 2:
        raise ValueError(
            "Variance estimation needs multiple players and repeated games."
        )

    log_b_center = float(np.mean(np.log1p(baseline)))
    prior_sd = _coefficient_prior_sd(
        context_cols=context_cols,
        scale=scale,
        prior_sd_raw=prior_sd_raw,
        standardize=standardize,
    )
    prior_precision = np.diag(1.0 / prior_sd**2)
    return {
        "weights": weights,
        "baseline": baseline,
        "target": target,
        "center": center,
        "scale": scale,
        "design_scale": design_scale,
        "design": design,
        "codes": codes,
        "ids": np.asarray(ids, dtype=int),
        "n": n,
        "log_b_center": log_b_center,
        "prior_sd": prior_sd,
        "prior_precision": prior_precision,
        "noise_start": max(float(np.var(target, ddof=1)), 0.25),
        "context_cols": tuple(context_cols),
        "standardize": bool(standardize),
        "prior_sd_raw": dict(prior_sd_raw),
    }


def _bounds_for_mode(
    variance_mode: VarianceMode,
    fixed_gamma1: float | None,
) -> list[tuple[float, float]]:
    gamma0_bounds = (np.log(1e-6), np.log(1e4))
    gamma1_bounds = (-5.0, 5.0)
    # Allow player-intercept variance to approach zero when supported.
    player_bounds = (np.log(1e-8), np.log(1e4))
    if variance_mode == "shared" or fixed_gamma1 is not None:
        return [gamma0_bounds, player_bounds]
    return [gamma0_bounds, gamma1_bounds, player_bounds]


def _pack_params(
    gamma0: float,
    gamma1: float,
    player_var: float,
    variance_mode: VarianceMode,
    fixed_gamma1: float | None,
) -> np.ndarray:
    if variance_mode == "shared" or fixed_gamma1 is not None:
        return np.array([gamma0, np.log(player_var)], dtype=float)
    return np.array([gamma0, gamma1, np.log(player_var)], dtype=float)


def _unpack_params(
    values: np.ndarray,
    variance_mode: VarianceMode,
    fixed_gamma1: float | None,
) -> tuple[float, float, float]:
    if variance_mode == "shared" or fixed_gamma1 is not None:
        gamma1 = 0.0 if variance_mode == "shared" else float(fixed_gamma1)
        return float(values[0]), gamma1, float(np.exp(values[1]))
    return float(values[0]), float(values[1]), float(np.exp(values[2]))


def _at_bound(value: float, low: float, high: float, tol: float = 1e-6) -> bool:
    return abs(value - low) <= tol or abs(value - high) <= tol


def fit_points_model(
    train: pd.DataFrame,
    grid_step: float = 0.01,
    variance_mode: VarianceMode = "baseline",
    fixed_gamma1: float | None = None,
    init_gamma0: float | None = None,
    init_player_var: float | None = None,
    context_cols: Sequence[str] | None = None,
    prior_sd_raw: Mapping[str, float] | None = None,
    prior_sd_scale: float = 1.0,
    standardize: bool = True,
) -> dict[str, Any]:
    """Fit baseline weights, shared slopes, and empirical-Bayes variances.

    Parameters
    ----------
    variance_mode:
        ``shared`` fixes gamma1=0. ``baseline`` estimates gamma0 and gamma1
        unless ``fixed_gamma1`` is provided.
    context_cols:
        Shared contextual predictors. Defaults to ``CONTEXT_COLS``.
    prior_sd_raw:
        Prior SDs in original feature units. Defaults to ``PRIOR_SD_RAW``.
    prior_sd_scale:
        Multiplies every prior SD (for sensitivity checks).
    standardize:
        If True, design uses ``(x - center) / scale`` and
        ``prior_sd_standardized[j] = prior_sd_raw[column] * scale[j]``.
        If False, design uses ``x - center`` with original-unit priors.
    """
    if variance_mode not in ("shared", "baseline"):
        raise ValueError("variance_mode must be 'shared' or 'baseline'.")
    if variance_mode == "shared" and fixed_gamma1 not in (None, 0.0):
        raise ValueError("shared mode requires gamma1=0.")

    cols = _resolve_context_cols(context_cols)
    priors = _resolve_prior_sd_raw(prior_sd_raw, prior_sd_scale)
    prepared = _prepare_train(
        train,
        grid_step,
        cols,
        priors,
        standardize,
    )
    design = prepared["design"]
    target = prepared["target"]
    codes = prepared["codes"]
    baseline = prepared["baseline"]
    prior_sd = prepared["prior_sd"]
    prior_precision = prepared["prior_precision"]
    log_b_center = prepared["log_b_center"]
    bounds = _bounds_for_mode(variance_mode, fixed_gamma1)

    def objective(values: np.ndarray) -> float:
        gamma0, gamma1, player_var = _unpack_params(
            values, variance_mode, fixed_gamma1
        )
        sigma2 = observation_variance(
            baseline, gamma0, gamma1, log_b_center
        )
        moments = shared_noise_posterior_moments(
            design=design,
            target=target,
            codes=codes,
            sigma2=sigma2,
            player_var=player_var,
            prior_sd=prior_sd,
            prior_precision=prior_precision,
        )
        return float(moments["nll"])

    gamma0_start = (
        float(init_gamma0)
        if init_gamma0 is not None
        else float(np.log(prepared["noise_start"]))
    )
    player_starts = (
        [float(init_player_var)]
        if init_player_var is not None
        else [0.1, 2.0]
    )
    gamma1_starts = (
        [0.0]
        if variance_mode == "shared" or fixed_gamma1 is not None
        else [0.0, 0.5]
    )

    attempts = []
    for player_var0 in player_starts:
        for gamma1_0 in gamma1_starts:
            x0 = _pack_params(
                gamma0_start,
                float(fixed_gamma1) if fixed_gamma1 is not None else gamma1_0,
                player_var0,
                variance_mode,
                fixed_gamma1,
            )
            attempts.append(
                minimize(
                    objective,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 400, "ftol": 1e-10},
                )
            )

    converged = [
        result
        for result in attempts
        if result.success and np.isfinite(result.fun)
    ]
    if not converged:
        raise RuntimeError(
            "Variance optimization did not converge: "
            + str([result.message for result in attempts])
        )

    result = min(converged, key=lambda value: value.fun)
    gamma0, gamma1, player_var = _unpack_params(
        result.x, variance_mode, fixed_gamma1
    )
    sigma2 = observation_variance(baseline, gamma0, gamma1, log_b_center)
    moments = shared_noise_posterior_moments(
        design=design,
        target=target,
        codes=codes,
        sigma2=sigma2,
        player_var=player_var,
        prior_sd=prior_sd,
        prior_precision=prior_precision,
    )

    boundary_flags = []
    for packed, (low, high), label in zip(
        result.x,
        bounds,
        (
            ["gamma0", "player_var"]
            if len(result.x) == 2
            else ["gamma0", "gamma1", "player_var"]
        ),
        strict=True,
    ):
        if _at_bound(float(packed), low, high):
            boundary_flags.append(label)

    # For reporting, shared noise is exp(gamma0). Under the baseline link the
    # same quantity is the observation variance at the centered baseline.
    noise_var = float(np.exp(gamma0))
    ids = prepared["ids"]
    conversion = np.r_[1.0, prepared["scale"]]
    beta = np.asarray(moments["beta"], dtype=float)
    beta_cov = np.asarray(moments["beta_cov"], dtype=float)
    if standardize:
        beta_original = beta / conversion
        beta_cov_original = beta_cov / np.outer(conversion, conversion)
    else:
        beta_original = beta.copy()
        beta_cov_original = beta_cov.copy()

    return {
        "weights": prepared["weights"],
        "center": prepared["center"],
        "scale": prepared["scale"],
        "design_scale": prepared["design_scale"],
        "beta": beta,
        "beta_cov": beta_cov,
        "beta_original": beta_original,
        "beta_cov_original": beta_cov_original,
        "noise_var": noise_var,
        "gamma0": float(gamma0),
        "gamma1": float(gamma1),
        "log_b_center": log_b_center,
        "player_var": float(player_var),
        "ids": ids,
        "n": prepared["n"],
        "id_to_slot": {int(pid): i for i, pid in enumerate(ids)},
        "intercept": moments["intercept"],
        "coupling": moments["coupling"],
        "conditional_var": moments["conditional_var"],
        "nll": float(moments["nll"]),
        "variance_mode": variance_mode,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "at_bound": boundary_flags,
        "context_cols": prepared["context_cols"],
        "standardize": prepared["standardize"],
        "prior_sd": prior_sd,
        "prior_precision": prior_precision,
        "prior_sd_raw": prepared["prior_sd_raw"],
        "prior_sd_scale": float(prior_sd_scale),
        # Plug-in EB: baseline weights and variance hyperparameters are fixed
        # after maximization; only coefficient and intercept uncertainty enters
        # predictive variance.
        "uncertainty_approximation": (
            "plug-in empirical Bayes for baseline weights and variance "
            "hyperparameters; posterior uncertainty for shared coefficients "
            "and player intercepts only"
        ),
    }


def predict_points(
    frame: pd.DataFrame,
    fitted: dict[str, Any],
    starter_value: float | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    context_cols = tuple(fitted.get("context_cols", CONTEXT_COLS))
    raw = frame.loc[:, list(context_cols)].to_numpy(dtype=float).copy()
    if starter_value is not None:
        if "starter_surprise" not in context_cols:
            raise ValueError(
                "starter_value requires starter_surprise in context_cols."
            )
        raw[:, context_cols.index("starter_surprise")] = (
            np.asarray(starter_value) - frame["start_rate"].to_numpy(dtype=float)
        )

    centered = raw - fitted["center"]
    if fitted.get("standardize", True):
        design = np.column_stack(
            [np.ones(len(frame)), centered / fitted["scale"]]
        )
    else:
        design = np.column_stack([np.ones(len(frame)), centered])
    baseline = frame.loc[:, list(WINDOW_COLS)].to_numpy(dtype=float) @ fitted["weights"]
    if not np.isfinite(design).all() or not np.isfinite(baseline).all():
        raise ValueError("Prediction features must be finite.")

    slots = np.array(
        [fitted["id_to_slot"].get(int(pid), -1) for pid in frame["player_id"]]
    )
    known = slots >= 0
    intercept = np.zeros(len(frame))
    conditional_var = np.full(len(frame), fitted["player_var"])
    coupling = np.zeros_like(design)
    intercept[known] = fitted["intercept"][slots[known]]
    conditional_var[known] = fitted["conditional_var"][slots[known]]
    coupling[known] = fitted["coupling"][slots[known]]

    adjusted_design = design - coupling
    coefficient_var = conditional_var + np.einsum(
        "ni,ij,nj->n",
        adjusted_design,
        fitted["beta_cov"],
        adjusted_design,
    )
    coefficient_var = np.maximum(coefficient_var, 0.0)

    if "gamma0" in fitted:
        obs_var = observation_variance(
            baseline,
            fitted["gamma0"],
            fitted["gamma1"],
            fitted["log_b_center"],
        )
    else:
        obs_var = np.full(len(frame), float(fitted["noise_var"]))

    return {
        "mean": baseline + design @ fitted["beta"] + intercept,
        "variance": obs_var + coefficient_var,
        "observation_variance": obs_var,
        "coefficient_variance": coefficient_var,
        "baseline": baseline,
    }
