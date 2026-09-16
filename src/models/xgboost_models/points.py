"""XGBoost player-points model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.points import (
    CURRENT_PLUS_MIN_MEAN_10,
    CURRENT_PLUS_PTS_MEAN_10,
    CURRENT_POINTS_FEATURES,
    DIRECT_POINTS_FEATURES,
    LEAN_POINTS_FEATURES,
    TIER1_POINTS_FEATURES,
    add_stacked_interactions,
)
from src.models.settlement import assert_bet_settles
from src.models.xgboost_models.tuning import (
    expanding_window_splits,
)

from .core import (
    CalibratedXGBoostRegressor,
    XGBoostConfig,
)
from .minutes import XGBoostMinutesModel

DEFAULT_POINTS_FEATURES = list(
    CURRENT_POINTS_FEATURES
)
POINTS_FEATURE_CONTRACTS = {
    "current41": list(CURRENT_POINTS_FEATURES),
    "lean37": list(LEAN_POINTS_FEATURES),
    "current_plus_pts_mean_10": list(
        CURRENT_PLUS_PTS_MEAN_10
    ),
    "current_plus_min_mean_10": list(
        CURRENT_PLUS_MIN_MEAN_10
    ),
    "tier1": list(TIER1_POINTS_FEATURES),
    "direct": list(DIRECT_POINTS_FEATURES),
}

EXPECTED_STARTER_THRESHOLD = 0.5
_ROLE_STARTER = 1
_ROLE_BENCH = 0
_ROLE_MISSING = -1
_ANY_BIN = -1
_GLOBAL_KEY = (_ROLE_MISSING, _ANY_BIN)

POINTS_HAT_P_BINS = np.array(
    [0.0, 8.0, 14.0, 20.0, 28.0, np.inf]
)
POINTS_HAT_M_BINS = np.array(
    [0.0, 12.0, 18.0, 24.0, 30.0, 36.0, 64.0]
)

__all__ = [
    "DEFAULT_POINTS_FEATURES",
    "DIRECT_POINTS_FEATURES",
    "EXPECTED_STARTER_THRESHOLD",
    "POINTS_FEATURE_CONTRACTS",
    "POINTS_HAT_M_BINS",
    "POINTS_HAT_P_BINS",
    "PointsLinePrediction",
    "PointsPrediction",
    "PointsResidualPools",
    "XGBoostPointsModel",
    "add_predicted_minutes_oof",
    "add_predicted_points_oof",
    "build_points_residual_pools",
    "expected_role",
]

MinutesModelFactory = Callable[[], object]


def expected_role(start_rate_10) -> np.ndarray:
    """Expected starter (1), bench (0), or missing (-1)."""
    rate = np.asarray(start_rate_10, dtype=float)
    role = np.full(np.shape(rate), _ROLE_MISSING, dtype=int)
    finite = np.isfinite(rate)
    role[finite] = (
        rate[finite] >= EXPECTED_STARTER_THRESHOLD
    ).astype(int)
    return role


def _feature_contract_name(
    columns: Sequence[str],
) -> str:
    as_list = list(columns)
    for name, contract in POINTS_FEATURE_CONTRACTS.items():
        if as_list == contract:
            return name
    return "custom"


@dataclass
class PointsResidualPools:
    """Standalone points residual pools (not joint ε).

    Stored residuals stay raw signed calibrator values.
    Soft-role simulation mixes starter and bench pools
    with P(starter)=clip(start_rate_10, 0, 1). Crossed
    schemes back off (role, bin) → (role,) → (bin,) →
    global when a cell is smaller than min_pool_size.
    Leaf scale shrinks toward the parent std with weight
    n/(n+shrinkage) without shifting leaf location.
    """

    pools: dict[tuple[int, int], np.ndarray]
    bins: np.ndarray
    bin_by: str
    min_pool_size: int
    shrinkage: float
    backoff_order: tuple[str, ...]
    scheme: str = "soft_role"


def build_points_residual_pools(
    *,
    residuals: np.ndarray,
    start_rate_10: np.ndarray,
    hat_p: np.ndarray,
    hat_m: np.ndarray | None = None,
    scheme: str = "soft_role",
    min_pool_size: int = 40,
    shrinkage: float = 20.0,
) -> PointsResidualPools:
    """Build role-aware standalone residual pools.

    ``pts_std_10`` is not required. Pools keep raw signed
    residuals; they are not recentered. Soft-role assigns
    observed start_rate rows to starter/bench by the 0.5
    threshold at build time, then simulates with a
    continuous start_rate mixture.
    """
    leftover = np.asarray(residuals, dtype=float).reshape(-1)
    start_rate = np.asarray(
        start_rate_10,
        dtype=float,
    ).reshape(-1)
    points_hat = np.asarray(hat_p, dtype=float).reshape(-1)
    if not (
        leftover.size == start_rate.size == points_hat.size
    ):
        raise ValueError(
            "residuals, start_rate_10, and hat_p "
            "must have the same length"
        )

    minutes_hat = None
    if hat_m is not None:
        minutes_hat = np.asarray(
            hat_m,
            dtype=float,
        ).reshape(-1)
        if minutes_hat.size != leftover.size:
            raise ValueError(
                "hat_m must have the same length "
                "as residuals"
            )

    keep = np.isfinite(leftover)
    leftover = leftover[keep]
    start_rate = start_rate[keep]
    points_hat = points_hat[keep]
    if minutes_hat is not None:
        minutes_hat = minutes_hat[keep]
    if leftover.size == 0:
        raise ValueError("no finite residuals to pool")

    roles = expected_role(start_rate)
    pools: dict[tuple[int, int], np.ndarray] = {
        _GLOBAL_KEY: leftover.copy(),
    }

    if scheme == "soft_role":
        starter_mask = roles == _ROLE_STARTER
        bench_mask = roles == _ROLE_BENCH
        if starter_mask.any():
            pools[(_ROLE_STARTER, _ANY_BIN)] = leftover[
                starter_mask
            ]
        if bench_mask.any():
            pools[(_ROLE_BENCH, _ANY_BIN)] = leftover[
                bench_mask
            ]
        return PointsResidualPools(
            pools=pools,
            bins=np.array([0.0, np.inf]),
            bin_by="none",
            min_pool_size=min_pool_size,
            shrinkage=shrinkage,
            backoff_order=("role", "global"),
            scheme=scheme,
        )

    if scheme == "role_x_hat_p":
        edges = POINTS_HAT_P_BINS
        hat = points_hat
        bin_by = "hat_p"
    elif scheme == "role_x_hat_m":
        if minutes_hat is None:
            raise ValueError(
                "hat_m is required for "
                "scheme='role_x_hat_m'"
            )
        edges = POINTS_HAT_M_BINS
        hat = minutes_hat
        bin_by = "hat_m"
    else:
        raise ValueError(
            f"Unknown residual scheme: {scheme!r}"
        )

    bin_ids = _assign_hat_bins(hat, edges)
    n_bins = len(edges) - 1
    for role in (_ROLE_BENCH, _ROLE_STARTER):
        role_mask = roles == role
        if role_mask.any():
            pools[(role, _ANY_BIN)] = leftover[role_mask]
        for bin_id in range(n_bins):
            cell = role_mask & (bin_ids == bin_id)
            if cell.any():
                pools[(role, bin_id)] = leftover[cell]
    for bin_id in range(n_bins):
        bin_mask = bin_ids == bin_id
        if bin_mask.any():
            pools[(_ROLE_MISSING, bin_id)] = leftover[
                bin_mask
            ]

    return PointsResidualPools(
        pools=pools,
        bins=edges,
        bin_by=bin_by,
        min_pool_size=min_pool_size,
        shrinkage=shrinkage,
        backoff_order=(
            "role_bin",
            "role",
            "bin",
            "global",
        ),
        scheme=scheme,
    )


def _assign_hat_bins(
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    assigned = np.clip(
        np.digitize(values, edges) - 1,
        0,
        len(edges) - 2,
    ).astype(int)
    return np.where(np.isfinite(values), assigned, -1)


def _resolve_points_pool(
    pools: PointsResidualPools,
    role: int,
    bin_id: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    store = pools.pools
    min_n = pools.min_pool_size
    global_pool = store[_GLOBAL_KEY]
    if role < 0:
        return global_pool, None

    cell = store.get((role, bin_id))
    role_pool = store.get((role, _ANY_BIN))
    bin_pool = (
        store.get((_ROLE_MISSING, bin_id))
        if bin_id >= 0
        else None
    )

    if (
        cell is not None
        and bin_id >= 0
        and len(cell) >= min_n
    ):
        parent = _first_usable(
            (role_pool, min_n),
            (bin_pool, min_n),
            (global_pool, 1),
        )
        return cell, parent

    if role_pool is not None and len(role_pool) >= min_n:
        parent = _first_usable(
            (bin_pool, min_n),
            (global_pool, 1),
        )
        return role_pool, parent

    if bin_pool is not None and len(bin_pool) >= min_n:
        return bin_pool, global_pool

    return global_pool, None


def _first_usable(
    *candidates: tuple[np.ndarray | None, int],
) -> np.ndarray | None:
    for pool, min_n in candidates:
        if pool is not None and len(pool) >= min_n:
            return pool
    return None


def _shrink_residual_draws(
    draws: np.ndarray,
    leaf: np.ndarray,
    parent: np.ndarray | None,
    shrinkage: float,
) -> np.ndarray:
    if parent is None or shrinkage <= 0 or len(leaf) == 0:
        return draws
    leaf_std = float(np.std(leaf))
    if leaf_std <= 0:
        return draws
    parent_std = float(np.std(parent))
    weight = len(leaf) / (len(leaf) + shrinkage)
    target_std = (
        weight * leaf_std + (1.0 - weight) * parent_std
    )
    location = float(np.mean(leaf))
    return location + (draws - location) * (
        target_std / leaf_std
    )


def _simulate_points_residuals(
    rows: pd.DataFrame,
    *,
    means: np.ndarray,
    pools: PointsResidualPools,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_rows = len(means)
    if "start_rate_10" in rows.columns:
        start_rate = np.asarray(
            rows["start_rate_10"],
            dtype=float,
        )
    else:
        start_rate = np.full(n_rows, np.nan)

    if pools.scheme == "soft_role":
        return _draw_soft_role_residuals(
            start_rate,
            pools=pools,
            sample_count=sample_count,
            rng=rng,
        )

    if pools.bin_by == "hat_m":
        if "predicted_minutes_oof" in rows.columns:
            hat = np.asarray(
                rows["predicted_minutes_oof"],
                dtype=float,
            )
        else:
            hat = np.full(n_rows, np.nan)
    else:
        hat = means

    roles = expected_role(start_rate)
    bin_ids = _assign_hat_bins(hat, pools.bins)
    draws = np.empty((n_rows, sample_count), dtype=float)
    pairs = np.column_stack([roles, bin_ids])
    for role, bin_id in np.unique(pairs, axis=0):
        mask = (roles == role) & (bin_ids == bin_id)
        leaf, parent = _resolve_points_pool(
            pools,
            int(role),
            int(bin_id),
        )
        sampled = rng.choice(
            leaf,
            size=(int(mask.sum()), sample_count),
            replace=True,
        )
        draws[mask] = _shrink_residual_draws(
            sampled,
            leaf,
            parent,
            pools.shrinkage,
        )
    return draws


def _draw_soft_role_residuals(
    start_rate: np.ndarray,
    *,
    pools: PointsResidualPools,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_rows = len(start_rate)
    global_pool = pools.pools[_GLOBAL_KEY]
    starter = pools.pools.get((_ROLE_STARTER, _ANY_BIN))
    bench = pools.pools.get((_ROLE_BENCH, _ANY_BIN))
    if starter is None or len(starter) == 0:
        starter = global_pool
    if bench is None or len(bench) == 0:
        bench = global_pool

    rate = np.asarray(start_rate, dtype=float)
    probability = np.clip(rate, 0.0, 1.0)
    missing = ~np.isfinite(rate)
    uniforms = rng.random((n_rows, sample_count))
    use_starter = (
        ~missing[:, None]
    ) & (uniforms < probability[:, None])
    use_global = missing[:, None]

    starter_draws = rng.choice(
        starter,
        size=(n_rows, sample_count),
        replace=True,
    )
    bench_draws = rng.choice(
        bench,
        size=(n_rows, sample_count),
        replace=True,
    )
    global_draws = rng.choice(
        global_pool,
        size=(n_rows, sample_count),
        replace=True,
    )
    mixed = np.where(
        use_starter,
        starter_draws,
        bench_draws,
    )
    return np.where(use_global, global_draws, mixed)


@dataclass(frozen=True)
class PointsPrediction:
    projected_points: float
    median_points: float
    standard_deviation: float
    lower_80: float
    upper_80: float
    availability_probability: float


@dataclass(frozen=True)
class PointsLinePrediction:
    line: float
    projected_points: float
    over_probability: float
    under_probability: float
    push_probability: float


class XGBoostPointsModel:
    """Appearance-conditional stacked XGBoost points model.

    Predictions are points given that the player plays.
    Default features include chronological out-of-fold
    ``predicted_minutes_oof``. Actual Game N minutes are
    never a feature. Current-game box scores are omitted.
    """

    def __init__(
        self,
        *,
        league: str,
        feature_columns: list[str] | None = None,
        config: XGBoostConfig | None = None,
        objective: str = "reg:squarederror",
    ) -> None:
        if league not in {"nba", "wnba"}:
            raise ValueError(
                f"Unsupported league: {league!r}"
            )

        self.league = league
        self.feature_columns = (
            list(feature_columns)
            if feature_columns is not None
            else DEFAULT_POINTS_FEATURES.copy()
        )
        self.feature_contract_name = _feature_contract_name(
            self.feature_columns
        )
        self.points_residual_pools: (
            PointsResidualPools | None
        ) = None
        residual_scaling = (
            "sqrt_mean"
            if "poisson" in objective
            else "raw"
        )

        self.regressor = CalibratedXGBoostRegressor(
            self.feature_columns,
            config=config,
            objective=objective,
            residual_scaling=residual_scaling,
            minimum_prediction=0.0,
        )

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str = "pts",
        date_column: str = "game_date",
        calibrate_residuals: bool = True,
    ) -> XGBoostPointsModel:
        self.regressor.fit(
            frame,
            target_column=target_column,
            date_column=date_column,
            calibrate_residuals=calibrate_residuals,
        )
        return self

    def predict_mean(
        self,
        rows: pd.DataFrame,
    ) -> np.ndarray:
        return self.regressor.predict(rows)

    def predict(
        self,
        rows: pd.DataFrame,
        *,
        availability_probability: float = 1.0,
        is_dnp: bool = False,
    ) -> list[PointsPrediction]:
        settlement_probability = assert_bet_settles(
            availability_probability=availability_probability,
            is_dnp=is_dnp,
        )
        samples = self.simulate(rows)

        return [
            PointsPrediction(
                projected_points=float(
                    player_samples.mean()
                ),
                median_points=float(
                    np.median(player_samples)
                ),
                standard_deviation=float(
                    player_samples.std()
                ),
                lower_80=float(
                    np.quantile(
                        player_samples,
                        0.10,
                    )
                ),
                upper_80=float(
                    np.quantile(
                        player_samples,
                        0.90,
                    )
                ),
                availability_probability=(
                    settlement_probability
                ),
            )
            for player_samples in samples
        ]

    def predict_line(
        self,
        rows: pd.DataFrame,
        *,
        line: float,
        availability_probability: float = 1.0,
        is_dnp: bool = False,
    ) -> list[PointsLinePrediction]:
        assert_bet_settles(
            availability_probability=availability_probability,
            is_dnp=is_dnp,
        )
        samples = self.simulate(rows)

        return [
            PointsLinePrediction(
                line=line,
                projected_points=float(
                    player_samples.mean()
                ),
                over_probability=float(
                    np.mean(player_samples > line)
                ),
                under_probability=float(
                    np.mean(player_samples < line)
                ),
                push_probability=float(
                    np.mean(player_samples == line)
                ),
            )
            for player_samples in samples
        ]

    def set_residual_pools(
        self,
        pools: PointsResidualPools,
    ) -> XGBoostPointsModel:
        """Attach standalone role-aware residual pools."""
        self.points_residual_pools = pools
        return self

    def simulate(
        self,
        rows: pd.DataFrame,
        *,
        availability_probability: float = 1.0,
        simulations: int | None = None,
    ) -> np.ndarray:
        """Simulate points conditional on the player playing."""
        if not 0 <= availability_probability <= 1:
            raise ValueError(
                "availability_probability must be "
                "between zero and one"
            )

        if self.points_residual_pools is None:
            return self.regressor.simulate(
                rows,
                simulations=simulations,
            )

        means = np.asarray(
            self.predict_mean(rows),
            dtype=float,
        )
        sample_count = (
            simulations
            or self.regressor.config.simulations
        )
        rng = np.random.default_rng(
            self.regressor.config.random_seed
        )
        residuals = _simulate_points_residuals(
            rows,
            means=means,
            pools=self.points_residual_pools,
            sample_count=sample_count,
            rng=rng,
        )
        return self.regressor._clip(
            means[:, None] + residuals
        )


def add_predicted_minutes_oof(
    frame: pd.DataFrame,
    *,
    minutes_model_factory: MinutesModelFactory | None = None,
    splits: Sequence[tuple[pd.Index, pd.Index]] | None = None,
    date_column: str = "game_date",
    folds: int = 5,
    minimum_training_dates: int = 60,
    minutes_features: list[str] | None = None,
    target_column: str = "minutes",
    league: str = "nba",
    config: XGBoostConfig | None = None,
) -> pd.DataFrame:
    """Fill chronological out-of-fold minute predictions.

    Each fold trains only on earlier rows, then predicts
    the next block. Actual Game N minutes are never copied
    into ``predicted_minutes_oof``.
    """
    result = frame.copy()
    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )
    if splits is None:
        splits = expanding_window_splits(
            result,
            date_column=date_column,
            folds=folds,
            minimum_training_dates=(
                minimum_training_dates
            ),
        )

    factory = minutes_model_factory or (
        lambda: XGBoostMinutesModel(
            league=league,
            feature_columns=minutes_features,
            config=config,
        )
    )
    result["predicted_minutes_oof"] = np.nan

    for train_idx, valid_idx in splits:
        model = factory()
        _fit_mean_only(
            model,
            result.loc[train_idx],
            target_column=target_column,
            date_column=date_column,
        )
        result.loc[valid_idx, "predicted_minutes_oof"] = (
            _predict_minutes_mean(
                model,
                result.loc[valid_idx],
            )
        )

    return add_stacked_interactions(result)


def add_predicted_points_oof(
    frame: pd.DataFrame,
    *,
    points_model_factory: MinutesModelFactory | None = None,
    splits: Sequence[tuple[pd.Index, pd.Index]] | None = None,
    date_column: str = "game_date",
    folds: int = 5,
    minimum_training_dates: int = 60,
    feature_columns: list[str] | None = None,
    target_column: str = "pts",
    league: str = "nba",
    config: XGBoostConfig | None = None,
) -> pd.DataFrame:
    """Fill chronological out-of-fold point means.

    Actual Game N points are never copied into the hat.
    """
    result = frame.copy()
    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )
    if splits is None:
        splits = expanding_window_splits(
            result,
            date_column=date_column,
            folds=folds,
            minimum_training_dates=(
                minimum_training_dates
            ),
        )

    factory = points_model_factory or (
        lambda: XGBoostPointsModel(
            league=league,
            feature_columns=feature_columns,
            config=config,
        )
    )
    result["predicted_points_oof"] = np.nan

    for train_idx, valid_idx in splits:
        model = factory()
        _fit_mean_only(
            model,
            result.loc[train_idx],
            target_column=target_column,
            date_column=date_column,
        )
        result.loc[valid_idx, "predicted_points_oof"] = (
            _predict_minutes_mean(
                model,
                result.loc[valid_idx],
            )
        )

    return result


def _fit_mean_only(
    model: object,
    frame: pd.DataFrame,
    *,
    target_column: str,
    date_column: str,
) -> object:
    """Fit throwaway OOF mean models without residual OOF."""
    try:
        return model.fit(
            frame,
            target_column=target_column,
            date_column=date_column,
            calibrate_residuals=False,
        )
    except TypeError:
        return model.fit(
            frame,
            target_column=target_column,
            date_column=date_column,
        )


def _predict_minutes_mean(
    model: object,
    rows: pd.DataFrame,
) -> np.ndarray:
    predict_mean = getattr(model, "predict_mean", None)
    if callable(predict_mean):
        return np.asarray(
            predict_mean(rows),
            dtype=float,
        )

    regressor = getattr(model, "regressor", None)
    if regressor is not None:
        return np.asarray(
            regressor.predict(rows),
            dtype=float,
        )

    raise TypeError(
        "Minutes model must implement predict_mean() "
        "or expose regressor.predict()."
    )
