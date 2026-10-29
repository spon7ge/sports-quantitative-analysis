from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse.linalg import SuperLU

FEATURE_COLUMNS = (
    "min_lag_1",
    "fga_lag_1",
    "fta_lag_1",
    "rest_days",
    "opp_def_prior",
    "is_home",
)
REQUIRED_RAW_COLUMNS = {
    "season_year",
    "season_type",
    "game_date",
    "game_id",
    "player_id",
    "player_name",
    "team_id",
    "opp_team_id",
    "pos",
    "matchup",
    "minutes",
    "fga",
    "fta",
    "pts",
    "team_def_rating",
}
_NUMERIC_SOURCE_COLUMNS = ("minutes", "fga", "fta", "team_def_rating")


@dataclass(frozen=True)
class DynamicPointsConfig:
    state_drift_sd: float = 0.35
    offseason_drift_sd: float = 2.0
    player_prior_precision: float = 10.0
    team_prior_precision: float = 20.0
    position_prior_precision: float = 20.0
    slope_prior_sd: float = 2.5
    variance_prior_alpha: float = 2.0
    variance_prior_beta: float = 50.0
    train_window_seasons: int = 2
    min_group_n: int = 30

    def __post_init__(self) -> None:
        positive = (
            self.state_drift_sd,
            self.offseason_drift_sd,
            self.player_prior_precision,
            self.team_prior_precision,
            self.position_prior_precision,
            self.slope_prior_sd,
            self.variance_prior_alpha,
            self.variance_prior_beta,
        )
        if any(value <= 0 or not math.isfinite(value) for value in positive):
            raise ValueError("uncertainty and prior values must be positive and finite")
        if self.train_window_seasons < 1 or self.min_group_n < 2:
            raise ValueError("season window and minimum group size are invalid")


@dataclass(frozen=True)
class DynamicState:
    mean: float
    variance: float
    games: int
    season: str | None = None


@dataclass(frozen=True)
class KalmanUpdate:
    mean: float
    variance: float
    gain: float
    innovation: float


@dataclass(frozen=True)
class HierarchicalFit:
    center: np.ndarray
    scale: np.ndarray
    medians: np.ndarray
    coef: np.ndarray
    factor: SuperLU
    alpha: float
    beta: float
    position_map: dict[str, int]
    team_map: dict[Any, int]
    player_map: dict[Any, int]

    @property
    def df(self) -> float:
        return 2.0 * self.alpha

    @property
    def observation_variance(self) -> float:
        return self.beta / (self.alpha - 1.0)


def kalman_update(
    mean: float,
    variance: float,
    observation: float,
    base_mean: float,
    observation_variance: float,
) -> KalmanUpdate:
    variances = (variance, observation_variance)
    if any(value <= 0 or not math.isfinite(value) for value in variances):
        raise ValueError("state and observation variances must be positive and finite")
    if any(
        not math.isfinite(value) for value in (mean, observation, base_mean)
    ):
        raise ValueError("state inputs must be finite")
    innovation = observation - (base_mean + mean)
    gain = variance / (variance + observation_variance)
    return KalmanUpdate(
        mean=mean + gain * innovation,
        variance=(1.0 - gain) * variance,
        gain=gain,
        innovation=innovation,
    )


def predictive_scale(state_variance: float, observation_variance: float) -> float:
    variances = (state_variance, observation_variance)
    if any(value <= 0 or not math.isfinite(value) for value in variances):
        raise ValueError("state and observation variances must be positive and finite")
    return math.sqrt(state_variance + observation_variance)


def build_causal_frame(
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = REQUIRED_RAW_COLUMNS - set(games.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = games.copy()
    source_missing_counts: dict[str, int] = {}
    for column in _NUMERIC_SOURCE_COLUMNS:
        source = frame[column]
        source_missing_counts[f"{column}_missing_values"] = int(
            source.isna().sum()
        )
        numeric = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & (numeric.isna() | ~np.isfinite(numeric))
        invalid_count = int(invalid.sum())
        if invalid_count:
            raise ValueError(
                f"{column} contains {invalid_count} invalid numeric values"
            )
        frame[column] = numeric
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise")
    frame = frame.loc[frame["season_type"].eq("Regular Season")].copy()
    if frame.duplicated(["player_id", "game_id"]).any():
        raise ValueError("duplicate player/game rows")
    frame = frame.sort_values(
        ["player_id", "game_date", "game_id"], kind="mergesort"
    )
    for source, target in (
        ("minutes", "min_lag_1"),
        ("fga", "fga_lag_1"),
        ("fta", "fta_lag_1"),
    ):
        frame[target] = frame.groupby("player_id", sort=False)[source].shift(1)
    frame["rest_days"] = (
        frame.groupby(["player_id", "season_year"], sort=False)["game_date"]
        .diff()
        .dt.days
    )
    frame["is_home"] = (
        frame["matchup"].astype(str).str.contains("vs", case=False).astype(float)
    )
    teams = (
        frame[["team_id", "game_id", "game_date", "team_def_rating"]]
        .drop_duplicates(["team_id", "game_id"])
        .sort_values(["team_id", "game_date", "game_id"], kind="mergesort")
    )
    defense = teams["team_def_rating"]
    prior = defense.groupby(teams["team_id"], sort=False).shift(1)
    teams["opp_def_prior"] = (
        prior.groupby(teams["team_id"], sort=False)
        .rolling(10, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    opponent = teams[["game_id", "team_id", "opp_def_prior"]].rename(
        columns={"team_id": "opp_team_id"}
    )
    frame = frame.merge(
        opponent,
        on=["game_id", "opp_team_id"],
        how="left",
        validate="many_to_one",
    )
    prior_team = frame.groupby("player_id", sort=False)["team_id"].shift(1)
    frame["new_team_flag"] = prior_team.notna() & frame["team_id"].ne(prior_team)
    first_season = frame.groupby("player_id", sort=False)["season_year"].transform(
        "first"
    )
    frame["rookie_flag"] = frame["season_year"].eq(first_season)
    frame["season_game_number"] = (
        frame.groupby(["player_id", "season_year"], sort=False).cumcount() + 1
    )
    return frame, {
        "input_rows": int(len(games)),
        "non_regular_season_rows_removed": int(len(games) - len(frame)),
        "output_rows": int(len(frame)),
        **source_missing_counts,
    }
