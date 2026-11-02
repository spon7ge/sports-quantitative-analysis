"""Time-split comparison of shared slopes against player-level slopes.

The fit uses every game before each player's last ``n`` appearances. Those
last appearances are the holdout. Covariate histories may include earlier
holdout games, because those games would already have been played. The
likelihood does not. The score is the posterior mean of the joint log
likelihood of minutes, attempts, and makes, summed across holdout rows.
A slope stays varying only when that score beats the all-shared model and
the 5% quantile of its player-level sd is above ``NEAR_ZERO``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.hierarchical_points import (
    VARYING_SLOPE_NAMES,
    SlopeConfig,
    shared_slope_config,
)

NEAR_ZERO = 0.01
OBSERVED = ("minutes", "fga2", "fga3", "fta", "fgm2", "fgm3", "ftm")


def one_at_a_time_configs(
    *,
    use_role_tier: bool = False,
) -> list[tuple[str, SlopeConfig]]:
    """The all-shared model, then each eligible slope turned on alone."""
    configs = [("shared", shared_slope_config())]
    for name in VARYING_SLOPE_NAMES:
        flags = {column: column == name for column in VARYING_SLOPE_NAMES}
        configs.append((name, SlopeConfig(varying=flags, use_role_tier=use_role_tier)))
    return configs


def last_games_mask(
    player_id: np.ndarray,
    game_date: pd.Series | np.ndarray,
    game_id: np.ndarray,
    n_games: int,
) -> np.ndarray:
    """True on the last ``n_games`` rows of each player, by date then game id.

    A player with ``n_games`` or fewer rows stays in the training set. Holding
    out every appearance would leave that player unknown at scoring time.
    """
    if n_games < 1:
        raise ValueError("n_games must be positive")
    frame = pd.DataFrame(
        {
            "player_id": np.asarray(player_id),
            "game_date": pd.to_datetime(game_date),
            "game_id": np.asarray(game_id),
            "row": np.arange(len(player_id)),
        }
    )
    ordered = frame.sort_values(
        ["player_id", "game_date", "game_id", "row"], kind="mergesort"
    )
    counts = ordered.groupby("player_id", sort=False).size()
    eligible = counts[counts > n_games].index
    chosen = (
        ordered.loc[ordered["player_id"].isin(eligible)]
        .groupby("player_id", sort=False)
        .tail(n_games)
    )
    mask = np.zeros(len(frame), dtype=bool)
    mask[chosen["row"].to_numpy()] = True
    return mask


def modeled_holdout_mask(
    frame: pd.DataFrame,
    n_games: int,
    covariates=None,
    *,
    season: str | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Last ``n_games`` per player among rows the design can score.

    A trailing game with no total or spread is not a holdout row. It is
    counted in ``unscored_trailing_rows`` and left out of both the fit and
    the score. The training scaler is fit later, on the complement of this
    mask, so this pass only decides which rows are eligible. ``season``
    limits the holdout to that ``season_year``. Earlier seasons stay in
    the training complement. Eligibility still uses the whole frame, so a
    current-season trail can see last season.
    """
    from src.models.hierarchical_points import (
        CovariateSpec,
        _drop_incomplete_covariates,
        _prepare,
    )

    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    spec = covariates if covariates is not None else CovariateSpec()
    prepared, audit = _prepare(frame, spec)
    prepared = _drop_incomplete_covariates(prepared, spec, audit)
    survivor = np.zeros(len(frame), dtype=bool)
    positions = frame.index.get_indexer(prepared["_source_index"])
    if np.any(positions < 0):
        raise ValueError("prepared rows are not in the frame index")
    survivor[positions] = True
    in_season = np.ones(len(frame), dtype=bool)
    if season is not None:
        if "season_year" not in frame.columns:
            raise ValueError("a season holdout needs season_year")
        in_season = frame["season_year"].astype(str).eq(str(season)).to_numpy()
    raw = np.zeros(len(frame), dtype=bool)
    raw_rows = np.flatnonzero(in_season)
    raw[raw_rows] = last_games_mask(
        frame["player_id"].to_numpy()[raw_rows],
        frame["game_date"].to_numpy()[raw_rows],
        frame["game_id"].to_numpy()[raw_rows],
        n_games,
    )
    eligible = np.flatnonzero(survivor & in_season)
    chosen = last_games_mask(
        frame["player_id"].to_numpy()[eligible],
        frame["game_date"].to_numpy()[eligible],
        frame["game_id"].to_numpy()[eligible],
        n_games,
    )
    mask = np.zeros(len(frame), dtype=bool)
    mask[eligible[chosen]] = True
    missing_total = frame["game_total"].isna().to_numpy()
    missing_spread = frame["player_team_spread"].isna().to_numpy()
    market = missing_total | missing_spread
    skipped = raw & ~survivor
    return mask, {
        "raw_trailing_rows": int(raw.sum()),
        "unscored_trailing_rows": int(skipped.sum()),
        "missing_market_trailing_rows": int((skipped & market).sum()),
        "holdout_rows": int(mask.sum()),
    }


def pointwise_elpd(log_likelihood: np.ndarray) -> float:
    """Sum of the draw-averaged pointwise log likelihood."""
    values = np.asarray(log_likelihood, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("log likelihood must be finite and non-empty")
    if values.ndim == 1:
        return float(values.sum())
    pointwise = values.reshape(-1, values.shape[-1]).mean(axis=0)
    return float(pointwise.sum())


def sd_away_from_zero(samples: np.ndarray, near_zero: float = NEAR_ZERO) -> bool:
    """True when the lower 5% of the sd posterior clears ``near_zero``."""
    flat = np.asarray(samples, dtype=float).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("sd samples must be finite and non-empty")
    return float(np.quantile(flat, 0.05)) > near_zero


def sd_summary(samples: np.ndarray, near_zero: float = NEAR_ZERO) -> dict[str, float]:
    flat = np.asarray(samples, dtype=float).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("sd samples must be finite and non-empty")
    return {
        "median": float(np.median(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q95": float(np.quantile(flat, 0.95)),
        "fraction_below_near_zero": float(np.mean(flat < near_zero)),
    }


def recommend_slope(
    elpd_shared: float,
    elpd_varying: float,
    sd_samples: np.ndarray,
    near_zero: float = NEAR_ZERO,
) -> str:
    """``vary`` only when the holdout score improves and the sd is away from zero."""
    improves = float(elpd_varying) > float(elpd_shared)
    if improves and sd_away_from_zero(sd_samples, near_zero):
        return "vary"
    return "share"


def sampling_diagnostics(idata, var_names: list[str]) -> pd.DataFrame:
    """R-hat, bulk ESS, and the run's divergence count for ``var_names``."""
    import arviz as az

    summary = az.summary(idata, var_names=var_names, round_to=None)
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    rows = []
    for name in summary.index.astype(str):
        rows.append(
            {
                "parameter": name,
                "r_hat": float(summary.loc[name, "r_hat"]),
                "ess_bulk": float(summary.loc[name, "ess_bulk"]),
                "divergences": divergences,
            }
        )
    return pd.DataFrame(rows)


def plot_shrinkage(
    player_draws: np.ndarray,
    mu_draws: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    """Player posterior means and 90% intervals against the shared mean."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    draws = np.asarray(player_draws, dtype=float)
    shared = np.asarray(mu_draws, dtype=float).reshape(-1)
    if draws.ndim != 2 or draws.shape[0] < 1:
        raise ValueError("player_draws must have shape (draw, player)")
    if not np.isfinite(draws).all() or not np.isfinite(shared).all():
        raise ValueError("shrinkage draws must be finite")
    center = draws.mean(axis=0)
    low = np.quantile(draws, 0.05, axis=0)
    high = np.quantile(draws, 0.95, axis=0)
    order = np.argsort(center)
    position = np.arange(center.size)
    shared_mean = float(shared.mean())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvline(shared_mean, color="black", linewidth=1.0, label="shared mean")
    ax.hlines(position, low[order], high[order], color="C0", linewidth=1.0)
    ax.plot(center[order], position, "o", color="C0", markersize=3)
    ax.set(
        xlabel="Slope",
        ylabel="Player, ordered by posterior mean",
        title=title,
    )
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def slope_stems(
    config: SlopeConfig,
    minutes_columns: tuple[str, ...],
    rate_columns: tuple[str, ...],
) -> list[str]:
    """Player-slope names created for the toggles that are on.

    A minutes column keeps its own name. Each attempt equation gets its own
    copy, prefixed with the stat, because those coefficients were already
    separate shared slopes.
    """
    from src.models.hierarchical_points import _ATTEMPTS

    stems = [column for column in minutes_columns if config.enabled(column)]
    for stat, _base in _ATTEMPTS:
        stems.extend(
            f"{stat}_{column}" for column in rate_columns if config.enabled(column)
        )
    return stems


def posterior_draws(idata, name: str) -> np.ndarray:
    """Flatten chain and draw. A vector parameter stays ``(draw, player)``."""
    values = np.asarray(idata.posterior[name].values, dtype=float)
    if values.ndim < 2:
        raise ValueError(f"{name} has no draw dimension")
    if values.ndim == 2:
        return values.reshape(-1)
    return values.reshape(-1, values.shape[-1])


def holdout_elpd(model, idata, rows: dict[str, np.ndarray]) -> float:
    """Joint holdout elpd after pointing the fitted model at ``rows``.

    The score adds the log likelihood of minutes, attempts, and makes. It is
    the posterior mean of that sum, not a PSIS-LOO estimate.
    """
    import pymc as pm

    payload = {
        "player_idx": np.asarray(rows["player"], dtype=int),
        "team_idx": np.asarray(rows["team"], dtype=int),
        "opp_idx": np.asarray(rows["opp"], dtype=int),
        "block_idx": np.asarray(rows["block"], dtype=int),
        "X_min": np.asarray(rows["X_min"], dtype=float),
        "X_rate": np.asarray(rows["X_rate"], dtype=float),
        "minutes_obs": np.asarray(rows["minutes"], dtype=float),
        "fga2_obs": np.asarray(rows["fga2"], dtype=np.int64),
        "fga3_obs": np.asarray(rows["fga3"], dtype=np.int64),
        "fta_obs": np.asarray(rows["fta"], dtype=np.int64),
        "fgm2_obs": np.asarray(rows["fgm2"], dtype=np.int64),
        "fgm3_obs": np.asarray(rows["fgm3"], dtype=np.int64),
        "ftm_obs": np.asarray(rows["ftm"], dtype=np.int64),
    }
    with model:
        pm.set_data(payload)
        scored = pm.compute_log_likelihood(
            idata,
            var_names=list(OBSERVED),
            extend_inferencedata=False,
            progressbar=False,
        )
    variables = getattr(scored, "data_vars", None)
    if variables is not None and "minutes" in variables:
        group = scored
    else:
        group = scored.log_likelihood
    total = None
    for name in OBSERVED:
        values = np.asarray(group[name].values, dtype=float)
        total = values if total is None else total + values
    return pointwise_elpd(total)
