"""XGBoost player-minutes model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .core import (
    CalibratedXGBoostRegressor,
    XGBoostConfig,
)
from src.features.minutes import (
    CURRENT_MINUTES_FEATURES,
    LEAN_MINUTES_FEATURES,
    ROLE_TAIL_MINUTES_FEATURES,
    TIER1_MINUTES_FEATURES,
)
from src.models.settlement import (
    assert_bet_settles,
    maximum_minutes,
)

DEFAULT_MINUTES_FEATURES = list(
    CURRENT_MINUTES_FEATURES
)

MINUTES_FEATURE_CONTRACTS = {
    "current37": list(CURRENT_MINUTES_FEATURES),
    "role_tail": list(ROLE_TAIL_MINUTES_FEATURES),
    "lean": list(LEAN_MINUTES_FEATURES),
    "tier1": list(TIER1_MINUTES_FEATURES),
}

DEFAULT_MINUTES_BINS = np.array(
    [0.0, 12.0, 18.0, 24.0, 30.0, 36.0, 64.0]
)

EXPECTED_STARTER_THRESHOLD = 0.5

MINUTES_DISTRIBUTION_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "models"
    / "minutes"
    / "xgboost_minutes_distribution.joblib"
)


def _minutes_contract_name(columns: list[str]) -> str:
    for name, contract in MINUTES_FEATURE_CONTRACTS.items():
        if list(columns) == list(contract):
            return name
    return "custom"


def expected_role(start_rate_10: np.ndarray) -> np.ndarray:
    """Map pregame start rate to 1 starter, 0 bench, -1 missing."""
    rates = np.asarray(start_rate_10, dtype=float)
    roles = np.full(rates.shape, -1, dtype=int)
    finite = np.isfinite(rates)
    roles[finite & (rates >= EXPECTED_STARTER_THRESHOLD)] = 1
    roles[finite & (rates < EXPECTED_STARTER_THRESHOLD)] = 0
    return roles


def _bin_ids(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    edges = np.asarray(bins, dtype=float)
    return np.clip(
        np.digitize(values, edges) - 1,
        0,
        len(edges) - 2,
    )


def _shrink_leaf_std(
    leaf: np.ndarray,
    parent: np.ndarray,
    *,
    shrinkage: float,
) -> np.ndarray:
    if len(leaf) == 0:
        return leaf.copy()
    std_leaf = float(np.std(leaf))
    if std_leaf <= 0.0 or len(parent) == 0:
        return leaf.copy()
    std_parent = float(np.std(parent))
    weight = len(leaf) / (len(leaf) + shrinkage)
    scale = (
        weight * std_leaf + (1.0 - weight) * std_parent
    ) / std_leaf
    mean = float(np.mean(leaf))
    return mean + (leaf - mean) * scale


@dataclass(frozen=True)
class HierarchicalMinutePools:
    bins: np.ndarray
    leaf_pools: dict[tuple[int, int], np.ndarray]
    parent_pools: dict[int, np.ndarray]
    global_pool: np.ndarray
    min_pool_size: int = 40
    shrinkage: float = 20.0

    def select_pool(
        self,
        role: int,
        bin_id: int,
    ) -> np.ndarray:
        if role in {0, 1}:
            leaf = self.leaf_pools.get((int(role), int(bin_id)))
            if leaf is not None and len(leaf) >= self.min_pool_size:
                return leaf
        parent = self.parent_pools.get(int(bin_id))
        if parent is not None and len(parent) >= self.min_pool_size:
            return parent
        return self.global_pool


def build_hierarchical_minute_pools(
    *,
    residuals: np.ndarray,
    minutes_hat: np.ndarray,
    start_rate_10: np.ndarray,
    bins: np.ndarray,
    min_pool_size: int = 40,
    shrinkage: float = 20.0,
) -> HierarchicalMinutePools:
    residuals = np.asarray(residuals, dtype=float)
    minutes_hat = np.asarray(minutes_hat, dtype=float)
    start_rate_10 = np.asarray(start_rate_10, dtype=float)
    edges = np.asarray(bins, dtype=float)
    if residuals.shape != minutes_hat.shape:
        raise ValueError(
            "residuals and minutes_hat must align"
        )
    if residuals.shape != start_rate_10.shape:
        raise ValueError(
            "residuals and start_rate_10 must align"
        )

    finite = np.isfinite(residuals) & np.isfinite(minutes_hat)
    residuals = residuals[finite]
    minutes_hat = minutes_hat[finite]
    start_rate_10 = start_rate_10[finite]
    if len(residuals) == 0:
        raise ValueError("residuals must not be empty")

    roles = expected_role(start_rate_10)
    bin_ids = _bin_ids(minutes_hat, edges)
    global_pool = residuals.copy()

    parent_pools: dict[int, np.ndarray] = {}
    for bin_id in np.unique(bin_ids):
        parent = residuals[bin_ids == bin_id]
        if len(parent) >= min_pool_size:
            parent_pools[int(bin_id)] = parent.copy()

    leaf_pools: dict[tuple[int, int], np.ndarray] = {}
    for role in (0, 1):
        for bin_id in np.unique(bin_ids):
            mask = (roles == role) & (bin_ids == bin_id)
            leaf = residuals[mask]
            if len(leaf) < min_pool_size:
                continue
            parent = parent_pools.get(
                int(bin_id),
                global_pool,
            )
            leaf_pools[(role, int(bin_id))] = _shrink_leaf_std(
                leaf,
                parent,
                shrinkage=shrinkage,
            )

    return HierarchicalMinutePools(
        bins=edges,
        leaf_pools=leaf_pools,
        parent_pools=parent_pools,
        global_pool=global_pool,
        min_pool_size=min_pool_size,
        shrinkage=shrinkage,
    )


@dataclass(frozen=True)
class MinutesPrediction:
    projected_minutes: float
    median_minutes: float
    standard_deviation: float
    lower_80: float
    upper_80: float
    availability_probability: float


@dataclass(frozen=True)
class MinutesLinePrediction:
    line: float
    projected_minutes: float
    over_probability: float
    under_probability: float
    push_probability: float


class XGBoostMinutesModel:
    """Appearance-conditional XGBoost minutes model.

    Predictions are minutes given that the player plays.
    Default features are the causal current37 production
    contract. Current-game box, starter, and availability
    fields are not included.
    """

    def __init__(
        self,
        *,
        league: str,
        feature_columns: list[str] | None = None,
        config: XGBoostConfig | None = None,
    ) -> None:
        if league not in {"nba", "wnba"}:
            raise ValueError(
                f"Unsupported league: {league!r}"
            )

        self.league = league
        if feature_columns is None:
            self.feature_columns = (
                DEFAULT_MINUTES_FEATURES.copy()
            )
            self.feature_contract_name = "current37"
        else:
            self.feature_columns = list(feature_columns)
            self.feature_contract_name = (
                _minutes_contract_name(self.feature_columns)
            )

        maximum_minutes_allowed = maximum_minutes(league)

        # n_estimators default 750 lives on XGBoostConfig;
        # M-T2 should check whether that cap binds.
        self.regressor = CalibratedXGBoostRegressor(
            self.feature_columns,
            config=config,
            objective="reg:squarederror",
            residual_scaling="raw",
            minimum_prediction=0.0,
            maximum_prediction=maximum_minutes_allowed,
        )
        self.hierarchical_pools: (
            HierarchicalMinutePools | None
        ) = None

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str = "minutes",
        date_column: str = "game_date",
        calibrate_residuals: bool = True,
    ) -> XGBoostMinutesModel:
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
    ) -> list[MinutesPrediction]:
        settlement_probability = assert_bet_settles(
            availability_probability=availability_probability,
            is_dnp=is_dnp,
        )
        samples = self.simulate(rows)

        return [
            MinutesPrediction(
                projected_minutes=float(
                    player_samples.mean()
                ),
                median_minutes=float(
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
    ) -> list[MinutesLinePrediction]:
        assert_bet_settles(
            availability_probability=availability_probability,
            is_dnp=is_dnp,
        )
        samples = self.simulate(rows)

        return [
            MinutesLinePrediction(
                line=line,
                projected_minutes=float(
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

    def simulate(
        self,
        rows: pd.DataFrame,
        *,
        availability_probability: float = 1.0,
        simulations: int | None = None,
    ) -> np.ndarray:
        """Simulate minutes conditional on the player playing."""
        if not 0 <= availability_probability <= 1:
            raise ValueError(
                "availability_probability must be "
                "between zero and one"
            )

        if self.hierarchical_pools is not None:
            return self._simulate_hierarchical(
                rows,
                simulations=simulations,
            )

        return self.regressor.simulate(
            rows,
            simulations=simulations,
        )

    def _simulate_hierarchical(
        self,
        rows: pd.DataFrame,
        *,
        simulations: int | None = None,
    ) -> np.ndarray:
        pools = self.hierarchical_pools
        if pools is None:
            raise RuntimeError(
                "hierarchical pools are not attached"
            )

        centers = np.asarray(
            self.predict_mean(rows),
            dtype=float,
        )
        sample_count = (
            simulations or self.regressor.config.simulations
        )
        rng = np.random.default_rng(
            self.regressor.config.random_seed
        )
        bin_ids = _bin_ids(centers, pools.bins)
        if "start_rate_10" in rows.columns:
            roles = expected_role(
                rows["start_rate_10"].to_numpy()
            )
        else:
            roles = np.full(len(rows), -1, dtype=int)

        draws = np.empty(
            (len(centers), sample_count),
            dtype=float,
        )
        for role in np.unique(roles):
            for bin_id in np.unique(bin_ids):
                mask = (roles == role) & (bin_ids == bin_id)
                if not np.any(mask):
                    continue
                pool = pools.select_pool(int(role), int(bin_id))
                sampled = rng.choice(
                    pool,
                    size=(int(mask.sum()), sample_count),
                    replace=True,
                )
                draws[mask] = centers[mask, None] + sampled
        return self.regressor._clip(draws)

    def set_stratified_residual_pools(
        self,
        pools: dict[int, np.ndarray],
        bins: np.ndarray,
    ) -> XGBoostMinutesModel:
        """Attach the production minutes sampler for points."""
        self.regressor.set_residual_pools(pools, bins)
        return self

    def set_hierarchical_residual_pools(
        self,
        pools: HierarchicalMinutePools,
    ) -> XGBoostMinutesModel:
        """Attach role × minute-bin residual pools."""
        self.hierarchical_pools = pools
        return self

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the minutes distribution for the points module."""
        artifact = Path(path) if path is not None else (
            MINUTES_DISTRIBUTION_ARTIFACT
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, artifact)
        return artifact

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
    ) -> XGBoostMinutesModel:
        """Load the production minutes sampler.

        Accepts a dumped ``XGBoostMinutesModel`` or the dict payload
        written by the mean-model notebook.
        """
        artifact = Path(path) if path is not None else (
            MINUTES_DISTRIBUTION_ARTIFACT
        )
        loaded = joblib.load(artifact)
        if isinstance(loaded, cls):
            return loaded
        if isinstance(loaded, dict) and isinstance(
            loaded.get("model"), cls
        ):
            model = loaded["model"]
            pools = loaded.get("stratified_pools")
            bins = loaded.get("minute_bins")
            if pools is not None and bins is not None:
                model.set_stratified_residual_pools(pools, bins)
            hierarchical = loaded.get("hierarchical_pools")
            if hierarchical is not None:
                model.set_hierarchical_residual_pools(
                    hierarchical
                )
            return model
        raise TypeError(
            f"Unsupported minutes artifact at {artifact}"
        )
