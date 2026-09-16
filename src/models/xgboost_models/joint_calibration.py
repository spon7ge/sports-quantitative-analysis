"""Offline joint minutes → points calibration (coupling only)."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import joblib

HOLDOUT_SEASON = "2025-26"
JOINT_CALIBRATION_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "models"
    / "points"
    / "joint_calibration.joblib"
)
APPEARANCE_KEY_COLUMNS = (
    "season_year",
    "game_id",
    "player_id",
)
FIRST_HAT_FOLD = 1
PREGAME_OVERLAY_FEATURES = (
    "pts_per_min_10",
    "usg_wmean_10",
    "start_rate_10",
)
CURRENT_OVERLAY_FEATURES = (
    "minutes_shock",
    "pts_per_min_10",
    "usg_wmean_10",
    "start_rate_10",
)
OVERLAY_FEATURE_NAMES = CURRENT_OVERLAY_FEATURES
RIDGE_ALPHA = 1.0
BETA_SHRINKAGE = 20.0
BETA_CLIP = (0.0, 2.5)
LOCATION_SHRINKAGE = 20.0
ROLE_BENCH = 0
ROLE_STARTER = 1
STANDALONE_EPSILON_SOURCE = "standalone_points"
COUPLING_EPSILON_SOURCE = "coupling_residuals"
JOINT_VARIANTS = (
    "residual_around_p",
    "current",
    "g_off_global",
    "g_off_shrunk_role",
    "g_on_shrunk_role",
)
# Preholdout 2026-09-15: no promotion. Keep current in production.
# g_off_shrunk_role DEFER (better role gap, loses NLL/majority/PIT).
# g_on_shrunk_role DEFER (no distribution benefit from g).
# g_off_global REJECT (role gap 0.155). residual_around_p is baseline only.
PRODUCTION_JOINT_VARIANT = "current"
NLL_ATOL = 1e-9
NLL_FOLD_MARGIN = 0.02
COVERAGE_TIGHT = (0.78, 0.82)
COVERAGE_WIDE = (0.75, 0.85)
PIT_TIGHT = (0.48, 0.52)
PIT_WORSE_MARGIN = 0.01
WIDTH_ALLOWANCE = 1.05
TARGET_COVERAGE = 0.80
TARGET_PIT = 0.50
DEFAULT_MINUTES_BINS = np.array(
    [0.0, 12.0, 18.0, 24.0, 30.0, 36.0, 64.0]
)
EPSILON_BINS = np.array([0.0, 8.0, 14.0, 20.0, 28.0, np.inf])
MIN_POOL_SIZE = 40
WINSOR_QUANTILES = (0.005, 0.995)


@dataclass(frozen=True)
class OverlayFit:
    coef: np.ndarray
    intercept: float
    feature_names: tuple[str, ...]
    impute_median: np.ndarray
    scale_mean: np.ndarray
    scale_std: np.ndarray
    alpha: float


@dataclass(frozen=True)
class BetaMap:
    beta_global: float
    beta_by_bin: dict[int, float]
    bins: np.ndarray
    shrinkage: float


@dataclass(frozen=True)
class EpsilonPools:
    pools: dict[int, np.ndarray]
    bins: np.ndarray
    removed_means: dict[int, float]
    raw_stds: dict[int, float]
    winsor: tuple[float, float]
    min_pool_size: int
    centered: bool = True
    overlay_enabled: bool = True
    role_aware: bool = False
    role_pools: dict | None = None
    epsilon_source: str = COUPLING_EPSILON_SOURCE


def assert_preholdout(frame: pd.DataFrame) -> None:
    if "season_year" not in frame.columns:
        raise ValueError("season_year is required")
    seasons = frame["season_year"].astype("string")
    if seasons.eq(HOLDOUT_SEASON).any():
        raise ValueError(
            f"{HOLDOUT_SEASON} rows are closed for joint calibration"
        )


def content_hash(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return sha256(data).hexdigest()


def fingerprint_artifact(
    path: str | Path,
    extra: dict | None = None,
) -> dict:
    artifact = Path(path)
    payload = {
        "path": str(artifact),
        "content_hash": content_hash(artifact),
    }
    if extra:
        payload.update(extra)
    return payload


def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values.to_numpy(dtype=float))


def classify_universes(frame: pd.DataFrame) -> pd.DataFrame:
    """Label base vs coupling OOF eligibility and exclusion reasons."""
    result = frame.copy()
    minutes_ok = (
        _finite(result["minutes_hat"])
        if "minutes_hat" in result
        else np.zeros(len(result), dtype=bool)
    )
    points_ok = (
        _finite(result["points_hat"])
        if "points_hat" in result
        else np.zeros(len(result), dtype=bool)
    )
    g_ok = (
        _finite(result["g_hat"])
        if "g_hat" in result
        else np.zeros(len(result), dtype=bool)
    )
    beta_ok = (
        _finite(result["beta_hat"])
        if "beta_hat" in result
        else np.zeros(len(result), dtype=bool)
    )
    fold = (
        pd.to_numeric(result["base_fold"], errors="coerce")
        if "base_fold" in result
        else pd.Series(np.nan, index=result.index)
    )

    result["base_oof_eligible"] = minutes_ok & points_ok
    result["coupling_oof_eligible"] = (
        result["base_oof_eligible"] & g_ok & beta_ok
    )

    exclusion = np.full(len(result), "", dtype=object)
    missing_minutes = ~minutes_ok
    missing_points = ~points_ok
    missing_g = ~g_ok
    missing_beta = ~beta_ok
    first_hat = fold.eq(FIRST_HAT_FOLD).to_numpy()
    no_fold = ~np.isfinite(fold.to_numpy(dtype=float)) | fold.eq(0).to_numpy()

    warmup = (~result["base_oof_eligible"].to_numpy()) & (
        no_fold | missing_minutes | missing_points
    )
    coupling_warmup = (
        result["base_oof_eligible"].to_numpy()
        & first_hat
        & (~result["coupling_oof_eligible"].to_numpy())
    )

    exclusion[warmup] = "warmup"
    exclusion[missing_minutes & ~warmup] = "missing_minutes_hat"
    exclusion[missing_points & ~warmup & minutes_ok] = (
        "missing_points_hat"
    )
    exclusion[coupling_warmup] = "coupling_warmup"
    missing_overlay = (
        result["base_oof_eligible"].to_numpy()
        & missing_g
        & ~coupling_warmup
        & ~warmup
    )
    missing_beta_only = (
        result["base_oof_eligible"].to_numpy()
        & g_ok
        & missing_beta
        & ~coupling_warmup
        & ~warmup
    )
    exclusion[missing_overlay] = "missing_overlay_hat"
    exclusion[missing_beta_only] = "missing_beta_hat"
    exclusion[result["coupling_oof_eligible"].to_numpy()] = ""

    result["exclusion"] = exclusion
    return result


def _overlay_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "minutes_shock":
        shock = (
            pd.to_numeric(frame["minutes_hat"], errors="coerce")
            - pd.to_numeric(frame["min_mean_10"], errors="coerce")
        )
        return shock.to_numpy(dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(
        dtype=float
    )


def _overlay_design(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    names = feature_names or CURRENT_OVERLAY_FEATURES
    return np.column_stack(
        [_overlay_column(frame, name) for name in names]
    )


def _impute_and_scale(
    design: np.ndarray,
    *,
    impute_median: np.ndarray | None = None,
    scale_mean: np.ndarray | None = None,
    scale_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    medians = (
        impute_median
        if impute_median is not None
        else np.nanmedian(design, axis=0)
    )
    filled = design.copy()
    for column in range(filled.shape[1]):
        missing = ~np.isfinite(filled[:, column])
        filled[missing, column] = medians[column]
    means = (
        scale_mean
        if scale_mean is not None
        else filled.mean(axis=0)
    )
    stds = (
        scale_std
        if scale_std is not None
        else filled.std(axis=0)
    )
    stds = np.where(stds < 1e-8, 1.0, stds)
    scaled = (filled - means) / stds
    return scaled, medians, means, stds


def fit_overlay(
    frame: pd.DataFrame,
    residual: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
    feature_names: tuple[str, ...] = CURRENT_OVERLAY_FEATURES,
) -> OverlayFit:
    names = tuple(feature_names)
    target = np.asarray(residual, dtype=float)
    design = _overlay_design(frame, names)
    scaled, medians, means, stds = _impute_and_scale(design)
    n_features = scaled.shape[1]
    intercept = np.ones((len(scaled), 1))
    matrix = np.hstack([intercept, scaled])
    ridge = np.sqrt(alpha) * np.eye(n_features + 1)
    ridge[0, 0] = 0.0
    stacked = np.vstack([matrix, ridge])
    outcome = np.concatenate(
        [target, np.zeros(n_features + 1)]
    )
    lower = np.full(n_features + 1, -np.inf)
    upper = np.full(n_features + 1, np.inf)
    if "minutes_shock" in names:
        shock_idx = names.index("minutes_shock")
        lower[1 + shock_idx] = 0.0
    fit = lsq_linear(stacked, outcome, bounds=(lower, upper))
    coefficients = np.asarray(fit.x, dtype=float)
    return OverlayFit(
        coef=coefficients[1:],
        intercept=float(coefficients[0]),
        feature_names=names,
        impute_median=medians,
        scale_mean=means,
        scale_std=stds,
        alpha=alpha,
    )


def apply_overlay(
    overlay: OverlayFit,
    frame: pd.DataFrame,
) -> np.ndarray:
    design = _overlay_design(frame, overlay.feature_names)
    scaled, _, _, _ = _impute_and_scale(
        design,
        impute_median=overlay.impute_median,
        scale_mean=overlay.scale_mean,
        scale_std=overlay.scale_std,
    )
    return overlay.intercept + scaled @ overlay.coef


def _through_origin_beta(
    u: np.ndarray,
    residual: np.ndarray,
) -> float:
    denom = float(np.dot(u, u))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(u, residual) / denom)


def fit_beta(
    u: np.ndarray,
    residual: np.ndarray,
    minutes_hat: np.ndarray,
    bins: np.ndarray | None = None,
    *,
    shrinkage: float = BETA_SHRINKAGE,
    mode: str = "shrunk",
) -> BetaMap:
    if mode not in {"global", "binwise", "shrunk"}:
        raise ValueError(f"Unknown beta mode: {mode}")
    shocks = np.asarray(u, dtype=float)
    leftover = np.asarray(residual, dtype=float)
    centers = np.asarray(minutes_hat, dtype=float)
    edges = np.asarray(
        bins if bins is not None else DEFAULT_MINUTES_BINS,
        dtype=float,
    )
    finite = (
        np.isfinite(shocks)
        & np.isfinite(leftover)
        & np.isfinite(centers)
    )
    shocks = shocks[finite]
    leftover = leftover[finite]
    centers = centers[finite]
    global_beta = _through_origin_beta(shocks, leftover)
    bin_ids = np.clip(
        np.digitize(centers, edges) - 1,
        0,
        len(edges) - 2,
    )
    lo, hi = BETA_CLIP
    clipped_global = float(np.clip(global_beta, lo, hi))
    beta_by_bin: dict[int, float] = {}
    used_shrinkage = 0.0 if mode != "shrunk" else shrinkage
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        n_bin = int(mask.sum())
        raw = _through_origin_beta(
            shocks[mask], leftover[mask]
        )
        if mode == "global":
            value = clipped_global
        elif mode == "binwise":
            value = float(np.clip(raw, lo, hi))
        else:
            weight = n_bin / (n_bin + shrinkage) if n_bin else 0.0
            value = float(
                np.clip(
                    weight * raw + (1.0 - weight) * global_beta,
                    lo,
                    hi,
                )
            )
        beta_by_bin[bin_id] = value
    return BetaMap(
        beta_global=clipped_global,
        beta_by_bin=beta_by_bin,
        bins=edges,
        shrinkage=used_shrinkage,
    )


def apply_beta(
    beta_map: BetaMap,
    minutes_hat: np.ndarray,
) -> np.ndarray:
    centers = np.asarray(minutes_hat, dtype=float)
    bin_ids = np.clip(
        np.digitize(centers, beta_map.bins) - 1,
        0,
        len(beta_map.bins) - 2,
    )
    slopes = np.array(
        [
            beta_map.beta_by_bin.get(
                int(bin_id),
                beta_map.beta_global,
            )
            for bin_id in bin_ids
        ],
        dtype=float,
    )
    slopes[~np.isfinite(centers)] = np.nan
    return slopes


def run_nested_coupling(
    frame: pd.DataFrame,
    *,
    minutes_bins: np.ndarray | None = None,
    overlay_features: tuple[str, ...] = CURRENT_OVERLAY_FEATURES,
    g_enabled: bool = True,
    beta_mode: str = "shrunk",
) -> pd.DataFrame:
    """Fit overlay then β on prior base-OOF rows; first hat fold is warmup."""
    result = frame.copy()
    result["g_hat"] = np.nan
    result["beta_hat"] = np.nan
    folds = sorted(
        pd.to_numeric(result["base_fold"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    bins = (
        minutes_bins
        if minutes_bins is not None
        else DEFAULT_MINUTES_BINS
    )
    prior_index: list = []
    for fold in folds:
        current = result["base_fold"].eq(fold)
        hats = (
            np.isfinite(
                pd.to_numeric(
                    result["minutes_hat"], errors="coerce"
                )
            )
            & np.isfinite(
                pd.to_numeric(
                    result["points_hat"], errors="coerce"
                )
            )
        )
        current_eligible = current & hats
        if fold <= FIRST_HAT_FOLD or not prior_index:
            prior_index.extend(
                result.index[current_eligible].tolist()
            )
            continue
        train = result.loc[prior_index]
        v = (
            pd.to_numeric(train["pts"], errors="coerce")
            - pd.to_numeric(train["points_hat"], errors="coerce")
        ).to_numpy(dtype=float)
        if g_enabled:
            overlay = fit_overlay(
                train, v, feature_names=overlay_features
            )
            result.loc[current, "g_hat"] = apply_overlay(
                overlay, result.loc[current]
            )
            leftover = v - apply_overlay(overlay, train)
        else:
            result.loc[current, "g_hat"] = 0.0
            leftover = v
        u = (
            pd.to_numeric(train["minutes"], errors="coerce")
            - pd.to_numeric(
                train["minutes_hat"], errors="coerce"
            )
        ).to_numpy(dtype=float)
        beta_map = fit_beta(
            u,
            leftover,
            train["minutes_hat"].to_numpy(dtype=float),
            bins,
            mode=beta_mode,
        )
        result.loc[current, "beta_hat"] = apply_beta(
            beta_map,
            result.loc[current, "minutes_hat"].to_numpy(
                dtype=float
            ),
        )
        prior_index.extend(
            result.index[current_eligible].tolist()
        )
    return result


def coupling_residual(frame: pd.DataFrame) -> np.ndarray:
    """ε = pts - (points_hat + g) - β * u. Never standalone residuals."""
    u = (
        pd.to_numeric(frame["minutes"], errors="coerce")
        - pd.to_numeric(frame["minutes_hat"], errors="coerce")
    ).to_numpy(dtype=float)
    points_hat = pd.to_numeric(frame["points_hat"], errors="coerce")
    pts = pd.to_numeric(frame["pts"], errors="coerce")
    if "g_hat" in frame.columns:
        g = pd.to_numeric(frame["g_hat"], errors="coerce").fillna(0.0)
    else:
        g = 0.0
    if "beta_hat" in frame.columns:
        beta = pd.to_numeric(
            frame["beta_hat"], errors="coerce"
        ).fillna(0.0)
    else:
        beta = 0.0
    return (pts - (points_hat + g) - beta * u).to_numpy(dtype=float)


def _role_ids(start_rate: np.ndarray) -> np.ndarray:
    rates = np.asarray(start_rate, dtype=float)
    roles = np.full(len(rates), -1, dtype=int)
    known = np.isfinite(rates)
    roles[known & (rates >= 0.5)] = ROLE_STARTER
    roles[known & (rates < 0.5)] = ROLE_BENCH
    return roles


def _adjust_pool(
    pool: np.ndarray,
    *,
    overlay_enabled: bool,
    shrinkage: float = LOCATION_SHRINKAGE,
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(pool, dtype=float)
    mean = float(values.mean())
    raw_std = float(values.std())
    if overlay_enabled:
        return values - mean, mean, raw_std
    n_pool = len(values)
    weight = n_pool / (n_pool + shrinkage) if n_pool else 0.0
    removed = mean * (1.0 - weight)
    return values - removed, removed, raw_std


def build_epsilon_pools(
    frame: pd.DataFrame,
    *,
    bins: np.ndarray | None = None,
    min_pool_size: int = MIN_POOL_SIZE,
    winsor: tuple[float, float] = WINSOR_QUANTILES,
    overlay_enabled: bool = True,
    role_aware: bool = False,
    global_epsilon: bool = False,
    epsilon_source: str | None = None,
    residuals=None,
) -> EpsilonPools:
    source = epsilon_source or COUPLING_EPSILON_SOURCE
    if source == STANDALONE_EPSILON_SOURCE or residuals is not None:
        raise ValueError(
            "epsilon must be built from coupling residuals"
        )
    labeled = (
        frame
        if "coupling_oof_eligible" in frame.columns
        else classify_universes(frame)
    )
    eligible = labeled.loc[labeled["coupling_oof_eligible"]].copy()
    if eligible.empty:
        raise ValueError("No coupling_oof_eligible rows for epsilon pools")

    epsilon = coupling_residual(eligible)
    g_values = (
        pd.to_numeric(eligible["g_hat"], errors="coerce").fillna(0.0)
        if "g_hat" in eligible.columns
        else 0.0
    )
    mu_hat = np.maximum(
        0.0,
        pd.to_numeric(eligible["points_hat"], errors="coerce")
        + g_values,
    ).to_numpy(dtype=float)

    finite = np.isfinite(epsilon) & np.isfinite(mu_hat)
    epsilon = epsilon[finite]
    mu_hat = mu_hat[finite]
    start_rate = None
    if "start_rate_10" in eligible.columns:
        start_rate = pd.to_numeric(
            eligible["start_rate_10"], errors="coerce"
        ).to_numpy(dtype=float)[finite]
    lower, upper = np.quantile(epsilon, list(winsor))
    epsilon = np.clip(epsilon, lower, upper)

    edges = np.asarray(
        bins if bins is not None else EPSILON_BINS,
        dtype=float,
    )
    n_bins = len(edges) - 1
    bin_ids = np.clip(
        np.digitize(mu_hat, edges) - 1,
        0,
        n_bins - 1,
    )
    raw_pools: dict[int, np.ndarray] = {}
    for bin_id in range(n_bins):
        pool = epsilon[bin_ids == bin_id]
        if len(pool) > 0:
            raw_pools[bin_id] = pool

    fallback = (
        np.concatenate(list(raw_pools.values()))
        if raw_pools
        else epsilon
    )
    if global_epsilon:
        adjusted, removed, raw_std = _adjust_pool(
            fallback, overlay_enabled=overlay_enabled
        )
        pools = {bin_id: adjusted for bin_id in range(n_bins)}
        removed_means = {bin_id: removed for bin_id in range(n_bins)}
        raw_stds = {bin_id: raw_std for bin_id in range(n_bins)}
    else:
        pools = {}
        removed_means = {}
        raw_stds = {}
        for bin_id in range(n_bins):
            pool = raw_pools.get(bin_id, fallback)
            if len(pool) < min_pool_size:
                pool = fallback
            adjusted, removed, raw_std = _adjust_pool(
                pool, overlay_enabled=overlay_enabled
            )
            pools[bin_id] = adjusted
            removed_means[bin_id] = removed
            raw_stds[bin_id] = raw_std

    role_pools = None
    if role_aware:
        role_pools = {}
        if start_rate is not None:
            roles = _role_ids(start_rate)
            for role_id in (ROLE_BENCH, ROLE_STARTER):
                for bin_id in range(n_bins):
                    mask = (roles == role_id) & (bin_ids == bin_id)
                    pool = epsilon[mask]
                    if len(pool) < min_pool_size:
                        continue
                    adjusted, _, _ = _adjust_pool(
                        pool, overlay_enabled=overlay_enabled
                    )
                    key = int(role_id) * n_bins + int(bin_id)
                    role_pools[key] = adjusted

    return EpsilonPools(
        pools=pools,
        bins=edges,
        removed_means=removed_means,
        raw_stds=raw_stds,
        winsor=winsor,
        min_pool_size=min_pool_size,
        centered=overlay_enabled,
        overlay_enabled=overlay_enabled,
        role_aware=role_aware,
        role_pools=role_pools,
        epsilon_source=source,
    )


@dataclass
class JointCalibration:
    schema_version: int = 1
    holdout_season: str = HOLDOUT_SEASON
    fitted_through: str = "2024-25"
    fingerprints: dict = None
    appearance_key_columns: tuple[str, ...] = APPEARANCE_KEY_COLUMNS
    random_seed: int = 42
    base_oof_max_training_date: str | None = None
    final_calibration_max_date: str | None = None
    fold_records: list = None
    minutes_bins: np.ndarray | None = None
    overlay: OverlayFit | None = None
    beta: BetaMap | None = None
    epsilon: EpsilonPools | None = None
    n_preholdout_appearances: int = 0
    n_base_oof_eligible: int = 0
    n_coupling_oof_eligible: int = 0
    oof_coverage: float = 0.0
    exclusions: dict = None
    epsilon_source: str = "cross_fitted"
    variant: str = PRODUCTION_JOINT_VARIANT

    def __post_init__(self) -> None:
        if self.fingerprints is None:
            self.fingerprints = {}
        if self.fold_records is None:
            self.fold_records = []
        if self.exclusions is None:
            self.exclusions = {}
        if self.epsilon_source == STANDALONE_EPSILON_SOURCE:
            raise ValueError(
                "epsilon_source cannot be standalone_points"
            )


def save_joint_calibration(
    artifact: JointCalibration,
    path: str | Path,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, dest)
    return dest


def load_joint_calibration(
    path: str | Path,
    *,
    minutes_mean_path: str | Path,
    minutes_dist_path: str | Path,
    points_path: str | Path,
) -> JointCalibration:
    artifact = joblib.load(path)
    expected = artifact.fingerprints
    observed = {
        "minutes_mean": content_hash(minutes_mean_path),
        "minutes_distribution": content_hash(minutes_dist_path),
        "points_mean": content_hash(points_path),
    }
    for key, digest in expected.items():
        if observed.get(key) != _stored_hash(digest):
            raise ValueError(
                f"Incompatible {key} artifact hash; "
                "joint calibration load failed closed"
            )
    return artifact


def _stored_hash(value) -> str:
    if isinstance(value, dict):
        return str(value.get("content_hash", ""))
    return str(value)


def _unwrap_artifact(path: str | Path) -> tuple[dict, object | None]:
    loaded = joblib.load(path)
    if isinstance(loaded, dict):
        return loaded, loaded.get("model")
    return {}, loaded


def _minutes_bins_from_artifact(path: str | Path) -> np.ndarray:
    try:
        payload, model = _unwrap_artifact(path)
    except Exception:
        return DEFAULT_MINUTES_BINS
    if "minute_bins" in payload:
        return np.asarray(payload["minute_bins"], dtype=float)
    bins = getattr(
        getattr(model, "regressor", None),
        "residual_pool_bins",
        None,
    )
    if bins is None:
        return DEFAULT_MINUTES_BINS
    return np.asarray(bins, dtype=float)


def _metadata_from_artifact(path: str | Path) -> dict:
    payload, model = _unwrap_artifact(path)
    regressor = getattr(model, "regressor", None)
    booster = getattr(regressor, "model", None)
    features = payload.get("feature_columns")
    if features is None:
        features = list(getattr(model, "feature_columns", []) or [])
    cutoff = payload.get("training_cutoff")
    if cutoff is None:
        cutoff = getattr(regressor, "training_cutoff", None)
    n_estimators = payload.get("n_estimators")
    if n_estimators is None:
        n_estimators = getattr(booster, "n_estimators", None)
    return {
        "path": str(Path(path)),
        "feature_columns": list(features),
        "training_cutoff": None if cutoff is None else str(cutoff),
        "n_estimators": (
            None if n_estimators is None else int(n_estimators)
        ),
        "league": payload.get("league") or getattr(model, "league", "nba"),
        "config": getattr(regressor, "config", None),
        "objective": getattr(
            regressor, "objective", "reg:squarederror"
        ),
    }


def _fingerprint_input(path: str | Path) -> dict:
    meta = _metadata_from_artifact(path)
    return fingerprint_artifact(
        path,
        extra={
            "feature_columns": meta["feature_columns"],
            "training_cutoff": meta["training_cutoff"],
            "n_estimators": meta["n_estimators"],
        },
    )


def _assign_base_folds(
    frame: pd.DataFrame,
    splits: list[tuple[pd.Index, pd.Index]],
) -> pd.DataFrame:
    result = frame.copy()
    result["base_fold"] = 0
    for fold, (_, valid_idx) in enumerate(splits, start=1):
        result.loc[valid_idx, "base_fold"] = fold
    return result


def train_joint_calibration(
    frame: pd.DataFrame,
    *,
    minutes_mean_path: str | Path,
    minutes_dist_path: str | Path,
    points_path: str | Path,
    minutes_model_factory=None,
    points_model_factory=None,
    splits: list[tuple[pd.Index, pd.Index]] | None = None,
    date_column: str = "game_date",
    folds: int = 5,
    minimum_training_dates: int = 60,
    minutes_bins: np.ndarray | None = None,
    random_seed: int = 42,
    return_panel: bool = False,
    variant: str = PRODUCTION_JOINT_VARIANT,
):
    from src.models.xgboost_models.minutes import XGBoostMinutesModel
    from src.models.xgboost_models.points import (
        XGBoostPointsModel,
        add_predicted_minutes_oof,
        add_predicted_points_oof,
    )
    from src.models.xgboost_models.tuning import (
        expanding_window_splits,
    )

    assert_preholdout(frame)
    result = frame.copy()
    result[date_column] = pd.to_datetime(
        result[date_column], errors="coerce"
    )
    n_preholdout = len(result)
    fingerprints = {
        "minutes_mean": _fingerprint_input(minutes_mean_path),
        "minutes_distribution": _fingerprint_input(minutes_dist_path),
        "points_mean": _fingerprint_input(points_path),
    }
    if minutes_model_factory is None:
        minutes_settings = _metadata_from_artifact(minutes_mean_path)
        minutes_model_factory = lambda s=minutes_settings: (
            XGBoostMinutesModel(
                league=s["league"],
                feature_columns=s["feature_columns"] or None,
                config=s["config"],
            )
        )
    if points_model_factory is None:
        points_settings = _metadata_from_artifact(points_path)
        points_model_factory = lambda s=points_settings: (
            XGBoostPointsModel(
                league=s["league"],
                feature_columns=s["feature_columns"] or None,
                config=s["config"],
                objective=s["objective"],
            )
        )
    if minutes_bins is None:
        minutes_bins = _minutes_bins_from_artifact(minutes_dist_path)
    bins = np.asarray(minutes_bins, dtype=float)
    if splits is None:
        splits = expanding_window_splits(
            result,
            date_column=date_column,
            folds=folds,
            minimum_training_dates=minimum_training_dates,
        )
    result = _assign_base_folds(result, list(splits))
    result = add_predicted_minutes_oof(
        result,
        minutes_model_factory=minutes_model_factory,
        splits=splits,
        date_column=date_column,
    )
    result["minutes_hat"] = result["predicted_minutes_oof"]
    result = add_predicted_points_oof(
        result,
        points_model_factory=points_model_factory,
        splits=splits,
        date_column=date_column,
    )
    result["points_hat"] = result["predicted_points_oof"]
    overlay, beta, pools, meta = fit_joint_variant(
        result,
        variant,
        minutes_bins=bins,
    )
    labeled = meta.get("panel")
    if labeled is None:
        labeled = classify_universes(result)
    base = labeled.loc[labeled["base_oof_eligible"]].copy()
    if base.empty:
        raise ValueError("No base_oof_eligible rows")
    last_train = splits[-1][0]
    base_oof_max_training_date = str(
        result.loc[last_train, date_column].max()
    )
    final_calibration_max_date = str(base[date_column].max())
    n_base = int(labeled["base_oof_eligible"].sum())
    n_coupling = int(labeled["coupling_oof_eligible"].sum())
    exclusion_counts = (
        labeled.loc[labeled["exclusion"].ne(""), "exclusion"]
        .value_counts()
        .to_dict()
    )
    fold_records = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        fold_records.append(
            {
                "fold": fold,
                "coupling_warmup": fold == FIRST_HAT_FOLD,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "valid_min_date": str(
                    result.loc[valid_idx, date_column].min()
                ),
                "valid_max_date": str(
                    result.loc[valid_idx, date_column].max()
                ),
            }
        )
    artifact = JointCalibration(
        fingerprints=fingerprints,
        random_seed=random_seed,
        base_oof_max_training_date=base_oof_max_training_date,
        final_calibration_max_date=final_calibration_max_date,
        fold_records=fold_records,
        minutes_bins=bins,
        overlay=overlay,
        beta=beta,
        epsilon=pools,
        n_preholdout_appearances=n_preholdout,
        n_base_oof_eligible=n_base,
        n_coupling_oof_eligible=n_coupling,
        oof_coverage=(
            n_coupling / n_preholdout if n_preholdout else 0.0
        ),
        exclusions=exclusion_counts,
        variant=variant,
    )
    if return_panel:
        return artifact, labeled
    return artifact


_VARIANT_CONFIG = {
    "residual_around_p": {
        "g_enabled": False,
        "overlay_features": None,
        "beta_mode": None,
        "overlay_enabled": False,
        "role_aware": False,
        "global_epsilon": True,
        "fit_beta": False,
    },
    "current": {
        "g_enabled": True,
        "overlay_features": CURRENT_OVERLAY_FEATURES,
        "beta_mode": "shrunk",
        "overlay_enabled": True,
        "role_aware": False,
        "global_epsilon": False,
        "fit_beta": True,
    },
    "g_off_global": {
        "g_enabled": False,
        "overlay_features": None,
        "beta_mode": "global",
        "overlay_enabled": False,
        "role_aware": False,
        "global_epsilon": True,
        "fit_beta": True,
    },
    "g_off_shrunk_role": {
        "g_enabled": False,
        "overlay_features": None,
        "beta_mode": "shrunk",
        "overlay_enabled": False,
        "role_aware": True,
        "global_epsilon": False,
        "fit_beta": True,
    },
    "g_on_shrunk_role": {
        "g_enabled": True,
        "overlay_features": PREGAME_OVERLAY_FEATURES,
        "beta_mode": "shrunk",
        "overlay_enabled": True,
        "role_aware": True,
        "global_epsilon": False,
        "fit_beta": True,
    },
}


def fit_joint_variant(
    frame: pd.DataFrame,
    variant: str,
    minutes_bins: np.ndarray | None = None,
) -> tuple[OverlayFit | None, BetaMap | None, EpsilonPools, dict]:
    if variant not in _VARIANT_CONFIG:
        raise ValueError(f"Unknown joint variant: {variant}")
    cfg = _VARIANT_CONFIG[variant]
    bins = (
        minutes_bins
        if minutes_bins is not None
        else DEFAULT_MINUTES_BINS
    )
    result = frame.copy()
    if variant == "residual_around_p":
        result["g_hat"] = 0.0
        result["beta_hat"] = 0.0
    else:
        result = run_nested_coupling(
            result,
            minutes_bins=bins,
            overlay_features=(
                cfg["overlay_features"] or CURRENT_OVERLAY_FEATURES
            ),
            g_enabled=cfg["g_enabled"],
            beta_mode=cfg["beta_mode"] or "shrunk",
        )
    labeled = classify_universes(result)
    pools = build_epsilon_pools(
        labeled,
        overlay_enabled=cfg["overlay_enabled"],
        role_aware=cfg["role_aware"],
        global_epsilon=cfg["global_epsilon"],
    )
    overlay = None
    beta = None
    base = labeled.loc[labeled["base_oof_eligible"]].copy()
    if not base.empty:
        residual = (
            pd.to_numeric(base["pts"], errors="coerce")
            - pd.to_numeric(base["points_hat"], errors="coerce")
        ).to_numpy(dtype=float)
        leftover = residual
        if cfg["g_enabled"]:
            overlay = fit_overlay(
                base,
                residual,
                feature_names=cfg["overlay_features"],
            )
            leftover = residual - apply_overlay(overlay, base)
        if cfg["fit_beta"]:
            u = (
                pd.to_numeric(base["minutes"], errors="coerce")
                - pd.to_numeric(base["minutes_hat"], errors="coerce")
            ).to_numpy(dtype=float)
            beta = fit_beta(
                u,
                leftover,
                base["minutes_hat"].to_numpy(dtype=float),
                bins,
                mode=cfg["beta_mode"],
            )
    meta = {
        "variant": variant,
        "g_enabled": cfg["g_enabled"],
        "beta_mode": cfg["beta_mode"],
        "overlay_features": cfg["overlay_features"],
        "role_aware": cfg["role_aware"],
        "overlay_enabled": cfg["overlay_enabled"],
        "epsilon_source": pools.epsilon_source,
        "panel": labeled,
    }
    return overlay, beta, pools, meta


def _coverage_ok(candidate: float, baseline: float) -> bool:
    if COVERAGE_TIGHT[0] <= candidate <= COVERAGE_TIGHT[1]:
        return True
    closer = abs(candidate - TARGET_COVERAGE) < abs(
        baseline - TARGET_COVERAGE
    )
    return closer and COVERAGE_WIDE[0] <= candidate <= COVERAGE_WIDE[1]


def _pit_ok(candidate: float, baseline: float) -> bool:
    if PIT_TIGHT[0] <= candidate <= PIT_TIGHT[1]:
        return True
    closer = abs(candidate - TARGET_PIT) < abs(baseline - TARGET_PIT)
    not_worse = abs(candidate - TARGET_PIT) <= (
        abs(baseline - TARGET_PIT) + PIT_WORSE_MARGIN
    )
    return closer and not_worse


def _hist_scores(hist) -> tuple[float, float]:
    counts = np.asarray(hist, dtype=float)
    total = float(counts.sum())
    if total <= 0 or counts.size == 0:
        return 0.0, 0.0
    expected = total / counts.size
    tv = 0.5 * float(
        np.abs(counts / total - 1.0 / counts.size).sum()
    )
    max_abs = float(np.abs(counts - expected).max())
    return tv, max_abs


def _hist_no_worse(candidate, baseline) -> bool:
    cand_tv, cand_max = _hist_scores(candidate)
    base_tv, base_max = _hist_scores(baseline)
    return cand_tv <= base_tv or cand_max <= base_max


def distribution_gate(candidate: dict, baseline: dict) -> bool:
    cand_nll = float(candidate["nll"])
    base_nll = float(baseline["nll"])
    if cand_nll + NLL_ATOL >= base_nll:
        return False
    if not _coverage_ok(
        float(candidate["coverage_80"]),
        float(baseline["coverage_80"]),
    ):
        return False
    role_keys = ("starter_coverage_80", "bench_coverage_80")
    if all(key in candidate and key in baseline for key in role_keys):
        cand_gap = abs(
            float(candidate["starter_coverage_80"])
            - float(candidate["bench_coverage_80"])
        )
        base_gap = abs(
            float(baseline["starter_coverage_80"])
            - float(baseline["bench_coverage_80"])
        )
        if cand_gap >= base_gap:
            return False
    if not _pit_ok(
        float(candidate["pit_mean"]),
        float(baseline["pit_mean"]),
    ):
        return False
    if float(candidate["width_80"]) > (
        float(baseline["width_80"]) * WIDTH_ALLOWANCE
    ):
        return False
    if "pit_hist" in candidate and "pit_hist" in baseline:
        if not _hist_no_worse(
            candidate["pit_hist"], baseline["pit_hist"]
        ):
            return False
    return True


def evaluate_distribution_gate(
    fold_pairs: list[tuple[dict, dict]],
) -> bool:
    if not fold_pairs:
        return False
    outcomes = [
        distribution_gate(candidate, baseline)
        for candidate, baseline in fold_pairs
    ]
    majority = sum(outcomes) * 2 > len(outcomes)
    nll_ok = all(
        float(candidate["nll"])
        <= float(baseline["nll"]) + NLL_FOLD_MARGIN
        for candidate, baseline in fold_pairs
    )
    return majority and nll_ok


def select_joint_variant(
    fold_metrics: dict[str, list[dict]],
) -> str:
    if "residual_around_p" in fold_metrics:
        baseline_name = "residual_around_p"
    else:
        baseline_name = "current"
    baseline_folds = fold_metrics.get(baseline_name)
    if not baseline_folds:
        return PRODUCTION_JOINT_VARIANT
    for variant in JOINT_VARIANTS:
        if variant == baseline_name or variant not in fold_metrics:
            continue
        pairs = list(
            zip(fold_metrics[variant], baseline_folds)
        )
        if evaluate_distribution_gate(pairs):
            return variant
    return PRODUCTION_JOINT_VARIANT


def residual_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = actual - predicted
    finite = np.isfinite(residual)
    residual = residual[finite]
    if residual.size == 0:
        return {
            "n": 0,
            "bias": float("nan"),
            "mae": float("nan"),
            "width_80": float("nan"),
        }
    return {
        "n": int(residual.size),
        "bias": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "width_80": float(
            np.quantile(residual, 0.90)
            - np.quantile(residual, 0.10)
        ),
    }


MAE_TOLERANCE = 0.05
BIAS_TOLERANCE = 0.05
MIN_SLICE_N = 200


def overlay_gate(
    on: dict,
    off: dict,
    *,
    mae_tol: float = MAE_TOLERANCE,
    bias_tol: float = BIAS_TOLERANCE,
    min_n: int = MIN_SLICE_N,
) -> bool | None:
    """Return True/False for eligible slices, None if n is too small."""
    n = int(on.get("n", 0))
    if n < min_n:
        return None
    return (
        float(on["mae"]) <= float(off["mae"]) + mae_tol
        and abs(float(on["bias"]))
        <= abs(float(off["bias"])) + bias_tol
    )


def evaluate_overlay_gate(
    pairs: list[tuple[dict, dict]],
    *,
    mae_tol: float = MAE_TOLERANCE,
    bias_tol: float = BIAS_TOLERANCE,
    min_n: int = MIN_SLICE_N,
) -> bool:
    outcomes = [
        overlay_gate(
            on,
            off,
            mae_tol=mae_tol,
            bias_tol=bias_tol,
            min_n=min_n,
        )
        for on, off in pairs
    ]
    eligible = [value for value in outcomes if value is not None]
    if not eligible:
        return True
    return all(eligible)




