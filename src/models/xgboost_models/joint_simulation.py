"""Appearance-conditional joint minutes → points simulator."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from src.models.settlement import maximum_minutes
from src.models.xgboost_models.artifact_bundle import (
    JointPointsArtifactBundle,
    require_pool,
)
from src.models.xgboost_models.joint_calibration import apply_overlay

DEFAULT_SEED = 42
DEFAULT_DRAWS = 10_000


@dataclass(frozen=True)
class JointSimulation:
    player_id: object
    game_id: object
    hat_m: float
    expected_minutes: float
    hat_p: float
    g: float
    mu: float
    beta: float
    beta_bin: int
    minutes_bin: int
    epsilon_bin: int
    minute_draws: np.ndarray
    point_draws: np.ndarray
    epsilon_draws: np.ndarray
    minutes_clip_rate: float
    points_clip_rate: float
    seed: int
    n_draws: int
    bundle_hash: str
    conditional_on_appearance: bool = True


def select_bin(value: float, edges: np.ndarray) -> int:
    """Left-closed, right-open bin index; last bin is exhaustive."""
    bounds = np.asarray(edges, dtype=float)
    return int(
        np.clip(
            np.digitize(value, bounds) - 1,
            0,
            len(bounds) - 2,
        )
    )


def expected_minutes(
    hat_m: float,
    pool: np.ndarray,
    *,
    lower: float = 0.0,
    upper: float | None = None,
) -> float:
    """Post-clip E[M] over the full residual pool, not the MC sample."""
    cap = maximum_minutes("nba") if upper is None else float(upper)
    return float(
        np.mean(
            np.clip(
                float(hat_m) + np.asarray(pool, dtype=float),
                lower,
                cap,
            )
        )
    )


def build_points_inference_features(
    row: pd.Series | pd.DataFrame | dict,
    hat_m: float | np.ndarray,
    *,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Stack hatM into the points contract. Never pass minute draws."""
    frame = _as_frame(row)
    minutes = np.asarray(hat_m, dtype=float)
    if minutes.ndim == 0 or minutes.size == 1:
        minutes = np.full(len(frame), float(np.reshape(minutes, -1)[0]))
    if len(minutes) != len(frame):
        raise ValueError("hat_m must be scalar or one value per row")

    out = pd.DataFrame(index=frame.index)
    pts_rate = pd.to_numeric(frame.get("pts_per_min_10"), errors="coerce")
    attempt_rate = pd.to_numeric(frame.get("fga_per_min_10"), errors="coerce")
    for column in feature_columns:
        if column == "predicted_minutes_oof":
            out[column] = minutes
        elif column == "expected_points_rate":
            out[column] = minutes * pts_rate.to_numpy(dtype=float)
        elif column == "expected_attempt_volume":
            out[column] = minutes * attempt_rate.to_numpy(dtype=float)
        else:
            out[column] = frame[column]
    return out


class JointPointsSimulator:
    """Draw points from saved minutes pools, g, β, and ε.

    Predictions are conditional on appearance. Minute *draws* never
    enter the points booster; only scalar hatM does.
    """

    def __init__(
        self,
        bundle: JointPointsArtifactBundle,
        *,
        seed: int = DEFAULT_SEED,
        n_draws: int = DEFAULT_DRAWS,
    ) -> None:
        self.bundle = bundle
        self.seed = int(seed)
        self.n_draws = int(n_draws)
        self._minutes_upper = float(
            bundle.minutes_mean.regressor.maximum_prediction
            or maximum_minutes("nba")
        )

    def predict_minutes_means(
        self,
        rows: pd.Series | pd.DataFrame | dict | list,
    ) -> np.ndarray:
        frame = _minutes_features(_as_frame(rows), self.bundle)
        raw = np.asarray(
            self.bundle.minutes_mean.predict_mean(frame),
            dtype=float,
        )
        return np.clip(raw, 0.0, self._minutes_upper)

    def predict_minutes_mean(
        self,
        row: pd.Series | pd.DataFrame | dict,
    ) -> float:
        means = self.predict_minutes_means(row)
        if len(means) != 1:
            raise ValueError("predict_minutes_mean expects a single row")
        return float(means[0])

    def predict_points_means(
        self,
        rows: pd.Series | pd.DataFrame | dict | list,
        *,
        hat_m: float | np.ndarray,
    ) -> np.ndarray:
        features = build_points_inference_features(
            rows,
            hat_m,
            feature_columns=list(self.bundle.points_mean.feature_columns),
        )
        return np.asarray(
            self.bundle.points_mean.predict_mean(features),
            dtype=float,
        )

    def predict_points_mean(
        self,
        row: pd.Series | pd.DataFrame | dict,
        *,
        hat_m: float,
    ) -> float:
        means = self.predict_points_means(row, hat_m=hat_m)
        if len(means) != 1:
            raise ValueError("predict_points_mean expects a single row")
        return float(means[0])

    def simulate(
        self,
        rows: pd.Series | pd.DataFrame | dict | list,
        *,
        n_draws: int | None = None,
    ) -> JointSimulation | list[JointSimulation]:
        many = isinstance(rows, (list, pd.DataFrame))
        frame = _as_frame(rows)
        draws = self.n_draws if n_draws is None else int(n_draws)
        results = [
            self._simulate_row(frame.iloc[index], draws)
            for index in range(len(frame))
        ]
        if many:
            return results
        return results[0]

    def simulate_points(
        self,
        rows: pd.Series | pd.DataFrame | dict | list,
        *,
        n_draws: int | None = None,
    ) -> np.ndarray:
        """Return point draws shaped ``(rows, n_draws)`` as float32.

        Does not keep per-row ``JointSimulation`` objects, so a full
        holdout with 2,000 draws stays in the low hundreds of MB.
        """
        frame = _as_frame(rows)
        draws = self.n_draws if n_draws is None else int(n_draws)
        samples = np.empty((len(frame), draws), dtype=np.float32)
        for index in range(len(frame)):
            simulated = self._simulate_row(frame.iloc[index], draws)
            samples[index] = simulated.point_draws
        return samples

    def _simulate_row(
        self,
        row: pd.Series,
        n_draws: int,
    ) -> JointSimulation:
        hat_m = self.predict_minutes_mean(row)
        hat_p = self.predict_points_mean(row, hat_m=hat_m)
        overlay = self.bundle.joint_calibration.overlay
        if overlay is None:
            g = 0.0
        else:
            g = float(
                apply_overlay(
                    overlay,
                    _overlay_frame(row, hat_m),
                )[0]
            )
        mu = float(max(0.0, hat_p + g))

        minutes_bins = self.bundle.minutes_distribution.regressor.residual_pool_bins
        minutes_bin = select_bin(hat_m, minutes_bins)
        minutes_pool = require_pool(
            self.bundle.minutes_distribution.regressor.residual_pools,
            minutes_bin,
            name="minutes",
        )
        mean_m = expected_minutes(
            hat_m,
            minutes_pool,
            upper=self._minutes_upper,
        )
        beta_map = self.bundle.joint_calibration.beta
        if beta_map is None:
            beta = 0.0
            beta_bin = select_bin(hat_m, minutes_bins)
        else:
            beta_bin = select_bin(hat_m, beta_map.bins)
            beta = float(
                beta_map.beta_by_bin.get(
                    beta_bin, beta_map.beta_global
                )
            )

        epsilon = self.bundle.joint_calibration.epsilon
        epsilon_bin = select_bin(mu, epsilon.bins)
        epsilon_pool = _select_epsilon_pool(epsilon, epsilon_bin, row)

        player_id = row.get("player_id")
        game_id = row.get("game_id")
        minutes_rng = self._rng(player_id, game_id, "minutes")
        epsilon_rng = self._rng(player_id, game_id, "epsilon")
        minute_residuals = minutes_rng.choice(
            minutes_pool,
            size=n_draws,
            replace=True,
        )
        epsilon_draws = epsilon_rng.choice(
            epsilon_pool,
            size=n_draws,
            replace=True,
        )
        raw_minutes = hat_m + minute_residuals
        minute_draws = np.clip(raw_minutes, 0.0, self._minutes_upper)
        shock_center = float(minute_draws.mean())
        raw_points = (
            hat_p
            + g
            + beta * (minute_draws - shock_center)
            + epsilon_draws
        )
        point_draws = np.maximum(0.0, raw_points)
        return JointSimulation(
            player_id=player_id,
            game_id=game_id,
            hat_m=hat_m,
            expected_minutes=mean_m,
            hat_p=hat_p,
            g=g,
            mu=mu,
            beta=beta,
            beta_bin=beta_bin,
            minutes_bin=minutes_bin,
            epsilon_bin=epsilon_bin,
            minute_draws=minute_draws,
            point_draws=point_draws,
            epsilon_draws=epsilon_draws,
            minutes_clip_rate=float(
                np.mean(
                    (raw_minutes < 0.0)
                    | (raw_minutes > self._minutes_upper)
                )
            ),
            points_clip_rate=float(np.mean(raw_points < 0.0)),
            seed=self.seed,
            n_draws=n_draws,
            bundle_hash=self.bundle.bundle_hash,
        )

    def _rng(
        self,
        player_id: object,
        game_id: object,
        stream: str,
    ) -> np.random.Generator:
        material = (
            f"{self.bundle.bundle_hash}|{self.seed}|"
            f"{player_id}|{game_id}|{stream}"
        ).encode("utf-8")
        digest = sha256(material).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _as_frame(rows: pd.Series | pd.DataFrame | dict | list) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.reset_index(drop=True)
    if isinstance(rows, pd.Series):
        return pd.DataFrame([rows])
    if isinstance(rows, dict):
        return pd.DataFrame([rows])
    return pd.DataFrame(list(rows))


def _minutes_features(
    frame: pd.DataFrame,
    bundle: JointPointsArtifactBundle,
) -> pd.DataFrame:
    return frame.loc[:, list(bundle.minutes_mean.feature_columns)]


def _role_id(start_rate) -> int | None:
    if start_rate is None:
        return None
    try:
        rate = float(start_rate)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(rate):
        return None
    return 1 if rate >= 0.5 else 0


def _select_epsilon_pool(epsilon, epsilon_bin: int, row: pd.Series) -> np.ndarray:
    role_pools = getattr(epsilon, "role_pools", None)
    if role_pools:
        role_id = _role_id(row.get("start_rate_10"))
        if role_id is not None:
            n_bins = len(np.asarray(epsilon.bins, dtype=float)) - 1
            key = int(role_id) * n_bins + int(epsilon_bin)
            if key in role_pools:
                return np.asarray(role_pools[key], dtype=float)
    return require_pool(
        epsilon.pools,
        epsilon_bin,
        name="epsilon",
    )


def _overlay_frame(row: pd.Series, hat_m: float) -> pd.DataFrame:
    frame = pd.DataFrame([row])
    frame["minutes_hat"] = hat_m
    return frame
