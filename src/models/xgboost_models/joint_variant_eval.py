"""Leakage-safe preholdout scoring for joint coupling variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluation import (
    interval_coverage,
    interval_width,
    negative_log_likelihood,
    pit_histogram,
    pit_shape_penalty,
    probability_integral_transform,
    role_coverage_gap,
)
from src.models.settlement import maximum_minutes
from src.models.xgboost_models.joint_calibration import (
    CURRENT_OVERLAY_FEATURES,
    DEFAULT_MINUTES_BINS,
    EPSILON_BINS,
    FIRST_HAT_FOLD,
    HOLDOUT_SEASON,
    JOINT_VARIANTS,
    MIN_POOL_SIZE,
    PREGAME_OVERLAY_FEATURES,
    ROLE_BENCH,
    ROLE_STARTER,
    EpsilonPools,
    _VARIANT_CONFIG,
    apply_beta,
    apply_overlay,
    assert_preholdout,
    build_epsilon_pools,
    classify_universes,
    fit_beta,
    fit_joint_variant,
    fit_overlay,
    train_joint_calibration,
)

N_DRAWS = 2_000
EVAL_SEED = 42
PROMOTION_CANDIDATE = "g_off_shrunk_role"
PROMOTION_WIDTH_ALLOWANCE = 1.02
COVERAGE_BAND = (0.78, 0.82)
PIT_BAND = (0.48, 0.52)
UNDERCOVERED = 0.78
PREHOLDOUT_SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
)
OOF_PANEL_COLUMNS = (
    "season_year",
    "game_id",
    "player_id",
    "game_date",
    "minutes",
    "pts",
    "minutes_hat",
    "points_hat",
    "base_fold",
    "start_rate_10",
    "min_mean_10",
    "pts_per_min_10",
    "usg_wmean_10",
)
OOF_PANEL_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "models"
    / "points"
    / "preholdout_oof_panel.parquet"
)
VARIANT_METRICS_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "models"
    / "points"
    / "joint_variant_preholdout.json"
)


@dataclass
class FoldFit:
    variant: str
    fold: int
    prior_max_date: pd.Timestamp
    valid_min_date: pd.Timestamp
    overlay: object | None
    beta: object | None
    epsilon: object
    minutes_pools: dict[int, np.ndarray]
    minutes_bins: np.ndarray


@dataclass
class PromotionDecision:
    selected: str
    candidate: str = PROMOTION_CANDIDATE
    passed: bool = False
    reasons: list[str] = field(default_factory=list)
    pooled: dict = field(default_factory=dict)


def load_preholdout_appearances(
    root: Path,
) -> pd.DataFrame:
    """Load silver 2019-20–2024-25 only. Never reads 2025-26."""
    from src.features.points import add_points_features

    frames = []
    for season in PREHOLDOUT_SEASONS:
        path = (
            root
            / "data"
            / "silver"
            / "nba"
            / season
            / "regular_season"
            / "player_gamelogs.parquet"
        )
        frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    if HOLDOUT_SEASON in set(
        frame["season_year"].astype("string")
    ):
        raise ValueError(
            f"{HOLDOUT_SEASON} rows are closed for joint calibration"
        )
    assert_preholdout(frame)
    featured = add_points_features(frame)
    appearances = featured.loc[
        featured["minutes"].gt(0)
    ].copy()
    appearances["game_date"] = pd.to_datetime(
        appearances["game_date"]
    )
    assert_preholdout(appearances)
    return appearances


def cache_oof_panel(panel: pd.DataFrame, path: Path) -> Path:
    assert_preholdout(panel)
    keep = [col for col in OOF_PANEL_COLUMNS if col in panel]
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.loc[:, keep].to_parquet(dest, index=False)
    return dest


def load_oof_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel["game_date"] = pd.to_datetime(panel["game_date"])
    assert_preholdout(panel)
    return panel


def prepare_oof_panel(
    appearances: pd.DataFrame,
    *,
    minutes_mean_path: Path,
    minutes_dist_path: Path,
    points_path: Path,
    cache_path: Path | None = OOF_PANEL_ARTIFACT,
    reuse_cache: bool = True,
) -> pd.DataFrame:
    """Build or reuse chronological OOF hats. Holdout stays sealed."""
    assert_preholdout(appearances)
    if reuse_cache and cache_path is not None and cache_path.exists():
        print(f"Reusing cached OOF panel {cache_path}", flush=True)
        return load_oof_panel(cache_path)
    print("Generating chronological OOF minutes/points hats...", flush=True)
    artifact, panel = train_joint_calibration(
        appearances,
        minutes_mean_path=minutes_mean_path,
        minutes_dist_path=minutes_dist_path,
        points_path=points_path,
        return_panel=True,
        variant="current",
    )
    del artifact
    if cache_path is not None:
        cache_oof_panel(panel, cache_path)
    return panel


def scored_folds(panel: pd.DataFrame) -> list[int]:
    folds = sorted(
        pd.to_numeric(panel["base_fold"], errors="coerce")
        .dropna()
        .unique()
        .astype(int)
        .tolist()
    )
    return [fold for fold in folds if fold > FIRST_HAT_FOLD]


def _eligible_prior(panel: pd.DataFrame, fold: int) -> pd.DataFrame:
    hats = (
        np.isfinite(panel["minutes_hat"])
        & np.isfinite(panel["points_hat"])
    )
    prior = panel.loc[
        panel["base_fold"].lt(fold) & hats
    ].copy()
    return prior


def _eligible_valid(panel: pd.DataFrame, fold: int) -> pd.DataFrame:
    hats = (
        np.isfinite(panel["minutes_hat"])
        & np.isfinite(panel["points_hat"])
    )
    return panel.loc[
        panel["base_fold"].eq(fold) & hats
    ].copy()


def _assert_earlier_fold_only(
    prior: pd.DataFrame,
    valid: pd.DataFrame,
    date_column: str = "game_date",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    prior_max = pd.Timestamp(prior[date_column].max())
    valid_min = pd.Timestamp(valid[date_column].min())
    if prior_max >= valid_min:
        raise ValueError(
            "coupling estimates leaked into the scored fold: "
            f"prior_max={prior_max} valid_min={valid_min}"
        )
    return prior_max, valid_min


def _minutes_pools(
    prior: pd.DataFrame,
    bins: np.ndarray,
    min_pool_size: int = MIN_POOL_SIZE,
) -> dict[int, np.ndarray]:
    residual = (
        pd.to_numeric(prior["minutes"], errors="coerce")
        - pd.to_numeric(prior["minutes_hat"], errors="coerce")
    ).to_numpy(dtype=float)
    centers = pd.to_numeric(
        prior["minutes_hat"], errors="coerce"
    ).to_numpy(dtype=float)
    finite = np.isfinite(residual) & np.isfinite(centers)
    residual = residual[finite]
    centers = centers[finite]
    edges = np.asarray(bins, dtype=float)
    bin_ids = np.clip(
        np.digitize(centers, edges) - 1,
        0,
        len(edges) - 2,
    )
    fallback = residual
    pools: dict[int, np.ndarray] = {}
    for bin_id in range(len(edges) - 1):
        pool = residual[bin_ids == bin_id]
        if len(pool) < min_pool_size:
            pool = fallback
        pools[bin_id] = pool
    return pools


def fit_fold_variant(
    panel: pd.DataFrame,
    *,
    fold: int,
    variant: str,
    minutes_bins: np.ndarray | None = None,
) -> FoldFit:
    if variant not in _VARIANT_CONFIG:
        raise ValueError(f"Unknown joint variant: {variant}")
    cfg = _VARIANT_CONFIG[variant]
    bins = (
        np.asarray(minutes_bins, dtype=float)
        if minutes_bins is not None
        else DEFAULT_MINUTES_BINS
    )
    prior = _eligible_prior(panel, fold)
    valid = _eligible_valid(panel, fold)
    if prior.empty or valid.empty:
        raise ValueError(f"Fold {fold} is missing prior or valid rows")
    prior_max, valid_min = _assert_earlier_fold_only(prior, valid)

    residual = (
        pd.to_numeric(prior["pts"], errors="coerce")
        - pd.to_numeric(prior["points_hat"], errors="coerce")
    ).to_numpy(dtype=float)
    overlay = None
    leftover = residual
    if cfg["g_enabled"]:
        overlay = fit_overlay(
            prior,
            residual,
            feature_names=cfg["overlay_features"]
            or CURRENT_OVERLAY_FEATURES,
        )
        leftover = residual - apply_overlay(overlay, prior)
        prior = prior.copy()
        prior["g_hat"] = apply_overlay(overlay, prior)
    else:
        prior = prior.copy()
        prior["g_hat"] = 0.0

    beta = None
    if cfg["fit_beta"]:
        u = (
            pd.to_numeric(prior["minutes"], errors="coerce")
            - pd.to_numeric(prior["minutes_hat"], errors="coerce")
        ).to_numpy(dtype=float)
        beta = fit_beta(
            u,
            leftover,
            prior["minutes_hat"].to_numpy(dtype=float),
            bins,
            mode=cfg["beta_mode"] or "shrunk",
        )
        prior["beta_hat"] = apply_beta(
            beta,
            prior["minutes_hat"].to_numpy(dtype=float),
        )
    else:
        prior["beta_hat"] = 0.0

    labeled = classify_universes(prior)
    epsilon = build_epsilon_pools(
        labeled,
        overlay_enabled=cfg["overlay_enabled"],
        role_aware=cfg["role_aware"],
        global_epsilon=cfg["global_epsilon"],
    )
    return FoldFit(
        variant=variant,
        fold=int(fold),
        prior_max_date=prior_max,
        valid_min_date=valid_min,
        overlay=overlay,
        beta=beta,
        epsilon=epsilon,
        minutes_pools=_minutes_pools(prior, bins),
        minutes_bins=bins,
    )


def _bin_ids(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bounds = np.asarray(edges, dtype=float)
    return np.clip(
        np.digitize(values, bounds) - 1,
        0,
        len(bounds) - 2,
    ).astype(int)


def _role_ids(start_rate: np.ndarray) -> np.ndarray:
    rates = np.asarray(start_rate, dtype=float)
    roles = np.full(len(rates), -1, dtype=int)
    known = np.isfinite(rates)
    roles[known & (rates >= 0.5)] = ROLE_STARTER
    roles[known & (rates < 0.5)] = ROLE_BENCH
    return roles


def _draw_from_pools(
    bin_ids: np.ndarray,
    pools: dict[int, np.ndarray],
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_rows = len(bin_ids)
    draws = np.empty((n_rows, n_draws), dtype=float)
    fallback = np.concatenate(
        [np.asarray(pool, dtype=float) for pool in pools.values()]
    )
    for bin_id in np.unique(bin_ids):
        mask = bin_ids == bin_id
        pool = np.asarray(
            pools.get(int(bin_id), fallback),
            dtype=float,
        )
        if len(pool) == 0:
            pool = fallback
        draws[mask] = rng.choice(
            pool,
            size=(int(mask.sum()), n_draws),
            replace=True,
        )
    return draws


def simulate_fold(
    valid: pd.DataFrame,
    fit: FoldFit,
    *,
    n_draws: int = N_DRAWS,
    seed: int = EVAL_SEED,
    minutes_upper: float | None = None,
    return_raw: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    cap = (
        maximum_minutes("nba")
        if minutes_upper is None
        else float(minutes_upper)
    )
    hat_m = pd.to_numeric(
        valid["minutes_hat"], errors="coerce"
    ).to_numpy(dtype=float)
    hat_p = pd.to_numeric(
        valid["points_hat"], errors="coerce"
    ).to_numpy(dtype=float)
    if fit.overlay is None:
        g = np.zeros(len(valid), dtype=float)
    else:
        g = apply_overlay(fit.overlay, valid)
    if fit.beta is None:
        beta = np.zeros(len(valid), dtype=float)
    else:
        beta = apply_beta(fit.beta, hat_m)

    rng = np.random.default_rng(
        seed + 1_000 * fit.fold + 17 * (
            list(JOINT_VARIANTS).index(fit.variant)
        )
    )
    minute_bins = _bin_ids(hat_m, fit.minutes_bins)
    minute_resid = _draw_from_pools(
        minute_bins,
        fit.minutes_pools,
        n_draws=n_draws,
        rng=rng,
    )
    minute_draws = np.clip(hat_m[:, None] + minute_resid, 0.0, cap)
    shock_center = minute_draws.mean(axis=1, keepdims=True)

    mu = np.maximum(0.0, hat_p + g)
    epsilon = fit.epsilon
    eps_bins = _bin_ids(mu, epsilon.bins)
    n_eps = len(np.asarray(epsilon.bins, dtype=float)) - 1
    eps_draws = _draw_from_pools(
        eps_bins,
        epsilon.pools,
        n_draws=n_draws,
        rng=rng,
    )
    role_pools = getattr(epsilon, "role_pools", None)
    if role_pools and "start_rate_10" in valid.columns:
        roles = _role_ids(
            pd.to_numeric(
                valid["start_rate_10"], errors="coerce"
            ).to_numpy(dtype=float)
        )
        for role_id in (ROLE_BENCH, ROLE_STARTER):
            for bin_id in range(n_eps):
                key = int(role_id) * n_eps + int(bin_id)
                pool = role_pools.get(key)
                if pool is None or len(pool) == 0:
                    continue
                mask = (roles == role_id) & (eps_bins == bin_id)
                if not mask.any():
                    continue
                eps_draws[mask] = rng.choice(
                    np.asarray(pool, dtype=float),
                    size=(int(mask.sum()), n_draws),
                    replace=True,
                )
    raw = (
        hat_p[:, None]
        + g[:, None]
        + beta[:, None] * (minute_draws - shock_center)
        + eps_draws
    )
    clipped = np.maximum(0.0, raw)
    if return_raw:
        return clipped, raw
    return clipped


def overlay_shift(valid: pd.DataFrame, fit: FoldFit) -> np.ndarray:
    """g applied to ``valid``, matching ``simulate_fold``."""
    if fit.overlay is None:
        return np.zeros(len(valid), dtype=float)
    return np.asarray(apply_overlay(fit.overlay, valid), dtype=float)


def fold_fit_from_bundle(bundle, *, variant: str = "current") -> FoldFit:
    """Wrap a production bundle as a ``FoldFit`` for vectorized diagnostics."""
    minutes = bundle.minutes_distribution.regressor
    calibration = bundle.joint_calibration
    if calibration.epsilon is None:
        raise ValueError("joint calibration is missing epsilon pools")
    pools = minutes.residual_pools
    bins = minutes.residual_pool_bins
    if pools is None or bins is None:
        raise ValueError("minutes distribution is missing residual pools")
    return FoldFit(
        variant=variant,
        fold=0,
        prior_max_date=pd.Timestamp("1970-01-01"),
        valid_min_date=pd.Timestamp("2100-01-01"),
        overlay=calibration.overlay,
        beta=calibration.beta,
        epsilon=calibration.epsilon,
        minutes_pools=dict(pools),
        minutes_bins=np.asarray(bins, dtype=float),
    )


def fit_preholdout_variant(
    panel: pd.DataFrame,
    variant: str,
    minutes_bins: np.ndarray | None = None,
) -> EpsilonPools:
    """Fit ε pools on the sealed-out preholdout OOF panel."""
    assert_preholdout(panel)
    _, _, epsilon, _ = fit_joint_variant(
        panel,
        variant,
        minutes_bins=minutes_bins,
    )
    return epsilon


def simulate_residual_around_p(
    hat_p: np.ndarray,
    epsilon: EpsilonPools,
    *,
    n_draws: int = N_DRAWS,
    seed: int = EVAL_SEED,
) -> np.ndarray:
    """Holdout draws for ``max(0, hat_p + ε)`` with no minutes coupling."""
    loc = np.asarray(hat_p, dtype=float)
    pool = np.asarray(next(iter(epsilon.pools.values())), dtype=float)
    rng = np.random.default_rng(seed)
    eps = rng.choice(pool, size=(len(loc), n_draws), replace=True)
    return np.maximum(0.0, loc[:, None] + eps).astype(np.float32)


def decompose_location(
    actual: np.ndarray,
    hat_p: np.ndarray,
    g: np.ndarray,
    mean_pre_clip: np.ndarray,
    mean_post_clip: np.ndarray,
) -> dict[str, float]:
    """Split signed error into mean, overlay, residual, and clipping."""
    y = np.asarray(actual, dtype=float)
    loc = np.asarray(hat_p, dtype=float)
    overlay = np.asarray(g, dtype=float)
    pre = np.asarray(mean_pre_clip, dtype=float)
    post = np.asarray(mean_post_clip, dtype=float)
    finite = (
        np.isfinite(y)
        & np.isfinite(loc)
        & np.isfinite(overlay)
        & np.isfinite(pre)
        & np.isfinite(post)
    )
    y = y[finite]
    loc = loc[finite]
    overlay = overlay[finite]
    pre = pre[finite]
    post = post[finite]
    hat_p_plus_g = loc + overlay
    return {
        "n": int(len(y)),
        "mean_actual": float(np.mean(y)),
        "mean_hat_p": float(np.mean(loc)),
        "mean_hat_p_plus_g": float(np.mean(hat_p_plus_g)),
        "mean_pre_clip": float(np.mean(pre)),
        "mean_post_clip": float(np.mean(post)),
        "error_vs_hat_p": float(np.mean(y - loc)),
        "error_vs_hat_p_plus_g": float(np.mean(y - hat_p_plus_g)),
        "error_vs_pre_clip": float(np.mean(y - pre)),
        "error_vs_post_clip": float(np.mean(y - post)),
        "overlay_shift": float(np.mean(overlay)),
        "residual_location": float(np.mean(pre - hat_p_plus_g)),
        "clipping_uplift": float(np.mean(post - pre)),
    }


def clipping_uplift_by_band(
    mean_pre_clip: np.ndarray,
    mean_post_clip: np.ndarray,
    centers: np.ndarray,
    edges: np.ndarray,
) -> pd.DataFrame:
    """E[max(0, Z)] − E[Z] in ``centers`` bins."""
    pre = np.asarray(mean_pre_clip, dtype=float)
    post = np.asarray(mean_post_clip, dtype=float)
    loc = np.asarray(centers, dtype=float)
    bounds = np.asarray(edges, dtype=float)
    uplift = post - pre
    bin_ids = np.clip(
        np.digitize(loc, bounds) - 1,
        0,
        len(bounds) - 2,
    )
    rows = []
    for bin_id in range(len(bounds) - 1):
        mask = np.isfinite(uplift) & np.isfinite(loc) & (bin_ids == bin_id)
        lo = bounds[bin_id]
        hi = bounds[bin_id + 1]
        hi_label = "∞" if not np.isfinite(hi) else f"{hi:g}"
        rows.append(
            {
                "band": f"[{lo:g}, {hi_label})",
                "n": int(mask.sum()),
                "mean_pre_clip": (
                    float(np.mean(pre[mask])) if mask.any() else float("nan")
                ),
                "mean_post_clip": (
                    float(np.mean(post[mask])) if mask.any() else float("nan")
                ),
                "clipping_uplift": (
                    float(np.mean(uplift[mask])) if mask.any() else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def shift_raw_to_postclip_mean(
    raw: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Add a per-row constant so E[max(0, Z+δ)] equals ``target``.

    Used to undo zero-clipping uplift without changing β or ε shape.
    """
    latent = np.asarray(raw, dtype=float)
    if latent.ndim == 1:
        latent = latent[None, :]
    goals = np.asarray(target, dtype=float).reshape(-1)
    if len(goals) != len(latent):
        raise ValueError("target must have one value per row")
    n_draws = latent.shape[1]
    ordered = np.sort(latent, axis=1)
    suffix = np.cumsum(ordered[:, ::-1], axis=1)[:, ::-1]
    delta = np.zeros(len(goals), dtype=float)
    for index, goal in enumerate(goals):
        if not np.isfinite(goal) or goal <= 0.0:
            delta[index] = -ordered[index, -1] if n_draws else 0.0
            continue
        chosen = None
        for clipped in range(n_draws):
            kept = n_draws - clipped
            total = float(suffix[index, clipped])
            shift = (n_draws * float(goal) - total) / kept
            lo_ok = True
            hi_ok = True
            if clipped > 0:
                lo_ok = ordered[index, clipped - 1] + shift < 0.0
            hi_ok = ordered[index, clipped] + shift >= 0.0
            if lo_ok and hi_ok:
                chosen = shift
                break
        if chosen is None:
            chosen = float(goal) - float(ordered[index].mean())
        delta[index] = chosen
    return latent + delta[:, None]


def signed_error_by_band(
    actual: np.ndarray,
    predicted: np.ndarray,
    centers: np.ndarray,
    edges: np.ndarray,
) -> pd.DataFrame:
    """Mean signed error (actual − predicted) in ``centers`` bins."""
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    loc = np.asarray(centers, dtype=float)
    bounds = np.asarray(edges, dtype=float)
    error = y - yhat
    bin_ids = np.clip(
        np.digitize(loc, bounds) - 1,
        0,
        len(bounds) - 2,
    )
    rows = []
    for bin_id in range(len(bounds) - 1):
        mask = np.isfinite(error) & np.isfinite(loc) & (bin_ids == bin_id)
        lo = bounds[bin_id]
        hi = bounds[bin_id + 1]
        hi_label = "∞" if not np.isfinite(hi) else f"{hi:g}"
        band_error = error[mask]
        rows.append(
            {
                "band": f"[{lo:g}, {hi_label})",
                "n": int(mask.sum()),
                "mean_actual": (
                    float(np.mean(y[mask])) if mask.any() else float("nan")
                ),
                "mean_predicted": (
                    float(np.mean(yhat[mask])) if mask.any() else float("nan")
                ),
                "mean_signed_error": (
                    float(np.mean(band_error)) if mask.any() else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def pit_decile_table(pit_values: np.ndarray, *, bins: int = 10) -> pd.DataFrame:
    pit = np.asarray(pit_values, dtype=float)
    pit = pit[np.isfinite(pit)]
    counts = pit_histogram(pit, bins=bins)
    total = int(counts.sum())
    edges = np.linspace(0.0, 1.0, bins + 1)
    share = counts.astype(float) / total if total else np.zeros(bins)
    expected = 1.0 / bins
    return pd.DataFrame(
        {
            "decile": [
                f"[{edges[i]:.1f}, {edges[i + 1]:.1f})"
                for i in range(bins)
            ],
            "count": counts,
            "share": share,
            "expected_share": expected,
            "share_minus_uniform": share - expected,
        }
    )


def _role_widths(
    samples: np.ndarray,
    start_rate_10: np.ndarray,
) -> dict[str, float]:
    draws = np.asarray(samples, dtype=float)
    rate = np.asarray(start_rate_10, dtype=float)
    widths = (
        np.quantile(draws, 0.90, axis=1)
        - np.quantile(draws, 0.10, axis=1)
    )
    finite = np.isfinite(rate)
    starter = finite & (rate >= 0.5)
    bench = finite & (rate < 0.5)
    return {
        "starter_width_80": (
            float(np.mean(widths[starter])) if starter.any() else float("nan")
        ),
        "bench_width_80": (
            float(np.mean(widths[bench])) if bench.any() else float("nan")
        ),
    }


def score_samples(
    samples: np.ndarray,
    actual: np.ndarray,
    start_rate_10: np.ndarray | None = None,
) -> dict[str, float]:
    pit = probability_integral_transform(samples, actual)
    metrics = {
        "n": int(len(actual)),
        "nll": negative_log_likelihood(samples, actual),
        "coverage_80": interval_coverage(samples, actual),
        "width_80": interval_width(samples),
        "pit_mean": float(np.mean(pit)),
        "pit_shape": pit_shape_penalty(pit),
        "pit_hist": pit_histogram(pit).tolist(),
    }
    if start_rate_10 is not None:
        role = role_coverage_gap(samples, actual, start_rate_10)
        metrics["starter_coverage_80"] = role["starter_coverage"]
        metrics["bench_coverage_80"] = role["bench_coverage"]
        metrics["role_gap"] = role["gap"]
        metrics.update(_role_widths(samples, start_rate_10))
    return metrics


def score_variant_folds(
    panel: pd.DataFrame,
    *,
    variants: tuple[str, ...] = JOINT_VARIANTS,
    n_draws: int = N_DRAWS,
    seed: int = EVAL_SEED,
    minutes_bins: np.ndarray | None = None,
    verbose: bool = False,
) -> dict[str, list[dict]]:
    assert_preholdout(panel)
    results: dict[str, list[dict]] = {name: [] for name in variants}
    folds = scored_folds(panel)
    for fold in folds:
        valid = _eligible_valid(panel, fold)
        actual = pd.to_numeric(valid["pts"], errors="coerce").to_numpy(
            dtype=float
        )
        start_rate = (
            pd.to_numeric(valid["start_rate_10"], errors="coerce").to_numpy(
                dtype=float
            )
            if "start_rate_10" in valid.columns
            else None
        )
        for variant in variants:
            if verbose:
                print(
                    f"scoring fold={fold} variant={variant} "
                    f"n={len(valid):,} draws={n_draws}",
                    flush=True,
                )
            fit = fit_fold_variant(
                panel,
                fold=fold,
                variant=variant,
                minutes_bins=minutes_bins,
            )
            samples = simulate_fold(
                valid,
                fit,
                n_draws=n_draws,
                seed=seed,
            )
            metrics = score_samples(samples, actual, start_rate)
            metrics.update(
                {
                    "fold": int(fold),
                    "variant": variant,
                    "prior_max_date": str(fit.prior_max_date),
                    "valid_min_date": str(fit.valid_min_date),
                    "n_draws": int(n_draws),
                    "seed": int(seed),
                }
            )
            results[variant].append(metrics)
            if verbose:
                print(
                    f"  nll={metrics['nll']:.4f} "
                    f"cov={metrics['coverage_80']:.3f} "
                    f"pit={metrics['pit_mean']:.3f} "
                    f"width={metrics['width_80']:.3f}",
                    flush=True,
                )
    return results


def pool_metrics(folds: list[dict]) -> dict[str, float]:
    if not folds:
        return {}
    weights = np.array([float(row["n"]) for row in folds], dtype=float)
    total = float(weights.sum())

    def weighted(key: str) -> float:
        values = np.array(
            [float(row[key]) for row in folds],
            dtype=float,
        )
        return float(np.sum(weights * values) / total)

    pooled = {
        "n": int(total),
        "n_folds": len(folds),
        "nll": weighted("nll"),
        "coverage_80": weighted("coverage_80"),
        "width_80": weighted("width_80"),
        "pit_mean": weighted("pit_mean"),
        "pit_shape": weighted("pit_shape"),
    }
    optional = (
        "starter_coverage_80",
        "bench_coverage_80",
        "role_gap",
        "starter_width_80",
        "bench_width_80",
    )
    for key in optional:
        if all(key in row for row in folds):
            pooled[key] = weighted(key)
    hist = np.sum(
        [np.asarray(row["pit_hist"], dtype=float) for row in folds],
        axis=0,
    )
    pooled["pit_hist"] = hist.tolist()
    return pooled


def _nll_beats(candidate: dict, baseline: dict) -> bool:
    return float(candidate["nll"]) < float(baseline["nll"])


def _most_folds_nll(
    candidate_folds: list[dict],
    baseline_folds: list[dict],
) -> bool:
    wins = 0
    for cand, base in zip(candidate_folds, baseline_folds):
        if float(cand["nll"]) < float(base["nll"]):
            wins += 1
    return wins * 2 > len(candidate_folds)


def _width_ok(
    candidate: dict,
    baseline: dict,
    *,
    allowance: float = PROMOTION_WIDTH_ALLOWANCE,
) -> tuple[bool, str]:
    cand_w = float(candidate["width_80"])
    base_w = float(baseline["width_80"])
    if cand_w <= base_w * allowance:
        return True, "overall width within allowance"
    starter_cov = float(baseline.get("starter_coverage_80", 1.0))
    bench_cov = float(baseline.get("bench_coverage_80", 1.0))
    cand_sw = candidate.get("starter_width_80")
    cand_bw = candidate.get("bench_width_80")
    base_sw = baseline.get("starter_width_80")
    base_bw = baseline.get("bench_width_80")
    if None in (cand_sw, cand_bw, base_sw, base_bw):
        return False, "overall width rose more than 2%"
    starter_under = starter_cov < UNDERCOVERED
    bench_under = bench_cov < UNDERCOVERED
    starter_ok = float(cand_sw) <= float(base_sw) * allowance
    bench_ok = float(cand_bw) <= float(base_bw) * allowance
    if starter_under and (not bench_under) and bench_ok:
        return True, "width increase isolated to under-covered starters"
    if bench_under and (not starter_under) and starter_ok:
        return True, "width increase isolated to under-covered bench"
    return False, "overall width rose more than 2%"


def decide_promotion(
    fold_metrics: dict[str, list[dict]],
    *,
    candidate: str = PROMOTION_CANDIDATE,
) -> PromotionDecision:
    """Promote candidate only against current and residual_around_p."""
    reasons: list[str] = []
    if candidate not in fold_metrics or "current" not in fold_metrics:
        return PromotionDecision(
            selected="current",
            candidate=candidate,
            passed=False,
            reasons=["missing candidate or current fold metrics"],
        )
    if "residual_around_p" not in fold_metrics:
        return PromotionDecision(
            selected="current",
            candidate=candidate,
            passed=False,
            reasons=["missing residual_around_p baseline"],
        )

    cand_folds = fold_metrics[candidate]
    current_folds = fold_metrics["current"]
    residual_folds = fold_metrics["residual_around_p"]
    cand = pool_metrics(cand_folds)
    current = pool_metrics(current_folds)
    residual = pool_metrics(residual_folds)
    pooled = {
        candidate: cand,
        "current": current,
        "residual_around_p": residual,
    }

    passed = True
    if not (
        _nll_beats(cand, current) and _nll_beats(cand, residual)
    ):
        passed = False
        reasons.append(
            "NLL does not beat both current and residual_around_p"
        )
    else:
        reasons.append("NLL beats current and residual_around_p")

    if not (
        _most_folds_nll(cand_folds, current_folds)
        and _most_folds_nll(cand_folds, residual_folds)
    ):
        passed = False
        reasons.append("NLL does not improve on most folds")
    else:
        reasons.append("NLL improves on most folds")

    coverage = float(cand["coverage_80"])
    if not (COVERAGE_BAND[0] <= coverage <= COVERAGE_BAND[1]):
        passed = False
        reasons.append(
            f"coverage {coverage:.3f} outside {COVERAGE_BAND[0]}–{COVERAGE_BAND[1]}"
        )
    else:
        reasons.append(
            f"coverage {coverage:.3f} inside {COVERAGE_BAND[0]}–{COVERAGE_BAND[1]}"
        )

    if "role_gap" in cand and "role_gap" in current:
        if float(cand["role_gap"]) >= float(current["role_gap"]):
            passed = False
            reasons.append("starter/bench coverage gap did not shrink")
        else:
            reasons.append("starter/bench coverage gap shrank")

    pit = float(cand["pit_mean"])
    if not (PIT_BAND[0] <= pit <= PIT_BAND[1]):
        passed = False
        reasons.append(
            f"PIT mean {pit:.3f} outside {PIT_BAND[0]}–{PIT_BAND[1]}"
        )
    else:
        reasons.append(
            f"PIT mean {pit:.3f} inside {PIT_BAND[0]}–{PIT_BAND[1]}"
        )

    cand_shape = float(cand["pit_shape"])
    current_shape = float(current["pit_shape"])
    if cand_shape > current_shape:
        passed = False
        reasons.append("PIT shape is worse than current")
    else:
        reasons.append("PIT shape is no worse than current")

    width_ok, width_reason = _width_ok(cand, current)
    reasons.append(width_reason)
    if not width_ok:
        passed = False

    selected = candidate if passed else "current"
    return PromotionDecision(
        selected=selected,
        candidate=candidate,
        passed=passed,
        reasons=reasons,
        pooled=pooled,
    )


def write_promoted_artifact(
    panel: pd.DataFrame,
    *,
    variant: str,
    source_path: Path,
    dest_path: Path,
) -> Path:
    """Refit coupling on a cached OOF panel; do not retrain means."""
    import joblib

    from src.models.xgboost_models.joint_calibration import (
        JointCalibration,
        fit_joint_variant,
        save_joint_calibration,
    )

    assert_preholdout(panel)
    source = joblib.load(source_path)
    overlay, beta, pools, meta = fit_joint_variant(panel, variant)
    labeled = meta["panel"]
    n_pre = len(panel)
    n_base = int(labeled["base_oof_eligible"].sum())
    n_coupling = int(labeled["coupling_oof_eligible"].sum())
    artifact = JointCalibration(
        fingerprints=getattr(source, "fingerprints", {}),
        random_seed=int(getattr(source, "random_seed", EVAL_SEED)),
        base_oof_max_training_date=getattr(
            source, "base_oof_max_training_date", None
        ),
        final_calibration_max_date=getattr(
            source, "final_calibration_max_date", None
        ),
        fold_records=list(getattr(source, "fold_records", []) or []),
        minutes_bins=getattr(source, "minutes_bins", DEFAULT_MINUTES_BINS),
        overlay=overlay,
        beta=beta,
        epsilon=pools,
        n_preholdout_appearances=n_pre,
        n_base_oof_eligible=n_base,
        n_coupling_oof_eligible=n_coupling,
        oof_coverage=(n_coupling / n_pre if n_pre else 0.0),
        exclusions=getattr(source, "exclusions", {}) or {},
        variant=variant,
    )
    return save_joint_calibration(artifact, dest_path)
