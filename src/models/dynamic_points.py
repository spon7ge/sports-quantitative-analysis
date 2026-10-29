from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import SuperLU, splu

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


def _category_map(values: pd.Series) -> dict[Any, int]:
    unique = sorted(values.dropna().unique().tolist())
    return {value: index for index, value in enumerate(unique)}


def _one_hot(
    values: pd.Series,
    mapping: dict[Any, int],
) -> sparse.csr_matrix:
    columns = np.array([mapping[value] for value in values])
    rows = np.arange(len(values))
    return sparse.csr_matrix(
        (np.ones(len(values)), (rows, columns)),
        shape=(len(values), len(mapping)),
    )


def fit_hierarchy(
    frame: pd.DataFrame,
    config: DynamicPointsConfig,
) -> tuple[HierarchicalFit, dict[str, int]]:
    required = set(FEATURE_COLUMNS) | {
        "minutes",
        "pts",
        "player_id",
        "pos",
        "team_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing hierarchy columns: {sorted(missing)}")

    minutes = pd.to_numeric(frame["minutes"], errors="coerce")
    invalid_minutes = frame["minutes"].notna() & (
        minutes.isna() | ~np.isfinite(minutes)
    )
    if invalid_minutes.any():
        raise ValueError("training minutes must be finite")
    usable = frame.loc[minutes.gt(0)].copy()
    if usable.empty:
        raise ValueError("hierarchy requires positive-minute training rows")

    raw = usable.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    source = usable.loc[:, FEATURE_COLUMNS]
    invalid_features = source.notna() & (raw.isna() | ~np.isfinite(raw))
    if invalid_features.any().any():
        raise ValueError("hierarchy features must be finite or missing")
    missing_features = int(raw.isna().sum().sum())
    medians = raw.median().to_numpy(dtype=float)
    if not np.isfinite(medians).all():
        raise ValueError("a feature has no finite training median")
    raw = raw.fillna(dict(zip(FEATURE_COLUMNS, medians, strict=True)))
    values = raw.to_numpy(dtype=float)
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("a hierarchy feature has no variation")

    positions = usable["pos"].astype(str)
    position_map = _category_map(positions)
    team_map = _category_map(usable["team_id"])
    player_map = _category_map(usable["player_id"])
    common = sparse.csr_matrix(
        np.column_stack([np.ones(len(usable)), (values - center) / scale])
    )
    design = sparse.hstack(
        [
            common,
            _one_hot(positions, position_map),
            _one_hot(usable["team_id"], team_map),
            _one_hot(usable["player_id"], player_map),
        ],
        format="csc",
    )
    precision = np.concatenate(
        [
            [1e-6],
            np.full(len(FEATURE_COLUMNS), 1.0 / config.slope_prior_sd**2),
            np.full(len(position_map), config.position_prior_precision),
            np.full(len(team_map), config.team_prior_precision),
            np.full(len(player_map), config.player_prior_precision),
        ]
    )
    posterior_precision = (
        design.T @ design + sparse.diags(precision, format="csc")
    ).tocsc()
    factor = splu(posterior_precision)

    y = pd.to_numeric(usable["pts"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("training outcomes must be finite")
    xty = np.asarray(design.T @ y).ravel()
    coef = factor.solve(xty)
    alpha = config.variance_prior_alpha + len(usable) / 2
    beta = config.variance_prior_beta + 0.5 * float(y @ y - coef @ xty)
    if alpha <= 1 or beta <= 0 or not np.isfinite(coef).all():
        raise ValueError("hierarchical posterior is not finite")
    return (
        HierarchicalFit(
            center=center,
            scale=scale,
            medians=medians,
            coef=coef,
            factor=factor,
            alpha=alpha,
            beta=beta,
            position_map=position_map,
            team_map=team_map,
            player_map=player_map,
        ),
        {
            "candidate_rows": int(len(frame)),
            "fitted_rows": int(len(usable)),
            "zero_minute_rows_removed": int(len(frame) - len(usable)),
            "imputed_feature_cells": missing_features,
        },
    )


def _impute_row_features(row: object, medians: np.ndarray) -> np.ndarray:
    raw = np.array([getattr(row, name) for name in FEATURE_COLUMNS], dtype=float)
    missing = ~np.isfinite(raw)
    raw[missing] = medians[missing]
    return raw


def _design_row(
    fit: HierarchicalFit,
    raw: np.ndarray,
    position: object,
    team_id: object,
    player_id: object | None = None,
) -> np.ndarray:
    n_common = 1 + len(FEATURE_COLUMNS)
    n_position = len(fit.position_map)
    n_team = len(fit.team_map)
    row = np.zeros(len(fit.coef))
    row[0] = 1.0
    row[1:n_common] = (raw - fit.center) / fit.scale
    position_key = str(position)
    if position_key in fit.position_map:
        row[n_common + fit.position_map[position_key]] = 1.0
    if team_id in fit.team_map:
        row[n_common + n_position + fit.team_map[team_id]] = 1.0
    if player_id in fit.player_map:
        row[n_common + n_position + n_team + fit.player_map[player_id]] = 1.0
    return row


def _base_prediction(
    fit: HierarchicalFit,
    raw: np.ndarray,
    position: object,
    team_id: object,
) -> float:
    return float(_design_row(fit, raw, position, team_id) @ fit.coef)


def filter_training_states(
    fit: HierarchicalFit,
    frame: pd.DataFrame,
    config: DynamicPointsConfig,
) -> dict[object, DynamicState]:
    states: dict[object, DynamicState] = {}
    initial_variance = fit.observation_variance / config.player_prior_precision
    ordered = frame.sort_values(
        ["game_date", "game_id", "player_id"], kind="mergesort"
    )
    for row in ordered.itertuples(index=False):
        state = states.get(
            row.player_id,
            DynamicState(0.0, initial_variance, 0),
        )
        variance = state.variance
        if state.games:
            variance += (
                config.offseason_drift_sd**2
                if state.season != row.season_year
                else config.state_drift_sd**2
            )
        raw = _impute_row_features(row, fit.medians)
        base = _base_prediction(fit, raw, row.pos, row.team_id)
        updated = kalman_update(
            state.mean,
            variance,
            float(row.pts),
            base,
            fit.observation_variance,
        )
        states[row.player_id] = DynamicState(
            updated.mean,
            updated.variance,
            state.games + 1,
            str(row.season_year),
        )
    return states


def predict_fold(
    fit: HierarchicalFit,
    evaluation: pd.DataFrame,
    training_states: dict[object, DynamicState],
    config: DynamicPointsConfig,
) -> pd.DataFrame:
    states = dict(training_states)
    initial_variance = fit.observation_variance / config.player_prior_precision
    end_training = dict(training_states)
    seen_in_evaluation: set[object] = set()
    records: list[dict[str, object]] = []
    ordered = evaluation.sort_values(
        ["game_date", "game_id", "player_id"], kind="mergesort"
    )

    for row in ordered.itertuples(index=False):
        training_state = end_training.get(row.player_id)
        returning_player = training_state is not None
        state = states.get(
            row.player_id,
            DynamicState(0.0, initial_variance, 0),
        )
        first_evaluation_appearance = row.player_id not in seen_in_evaluation
        season_boundary = first_evaluation_appearance and returning_player
        variance = state.variance
        if state.games:
            variance += (
                config.offseason_drift_sd**2
                if season_boundary
                else config.state_drift_sd**2
            )

        raw = _impute_row_features(row, fit.medians)
        base = _base_prediction(fit, raw, row.pos, row.team_id)
        points = float(row.pts)
        scale = predictive_scale(variance, fit.observation_variance)
        predictive_mean = base + state.mean
        innovation = points - predictive_mean
        z = innovation / scale
        pit = float(
            stats.t.cdf(
                points,
                fit.df,
                loc=predictive_mean,
                scale=scale,
            )
        )
        updated = kalman_update(
            state.mean,
            variance,
            points,
            base,
            fit.observation_variance,
        )
        records.append(
            {
                "season_year": str(row.season_year),
                "game_date": row.game_date,
                "game_id": row.game_id,
                "player_id": row.player_id,
                "player_name": row.player_name,
                "team_id": row.team_id,
                "opp_team_id": row.opp_team_id,
                "position": str(row.pos),
                "points": points,
                "base_mean": base,
                "predictive_mean": predictive_mean,
                "innovation": innovation,
                "z": z,
                "pit": pit,
                "df": fit.df,
                "state_before": state.mean,
                "state_var_before": variance,
                "state_after": updated.mean,
                "state_var_after": updated.variance,
                "state_end_training": (
                    training_state.mean if training_state is not None else 0.0
                ),
                "state_var_end_training": (
                    training_state.variance
                    if training_state is not None
                    else initial_variance
                ),
                "obs_var": fit.observation_variance,
                "predictive_var": variance + fit.observation_variance,
                "kalman_gain": updated.gain,
                "games_in_model_before": state.games,
                "season_game_number": row.season_game_number,
                "returning_player": returning_player,
                "season_boundary": season_boundary,
                "new_team_flag": bool(row.new_team_flag),
                "rookie_flag": bool(row.rookie_flag),
            }
        )
        states[row.player_id] = DynamicState(
            updated.mean,
            updated.variance,
            state.games + 1,
            str(row.season_year),
        )
        seen_in_evaluation.add(row.player_id)
    return pd.DataFrame.from_records(records)


def walk_forward_trace(
    frame: pd.DataFrame,
    evaluation_seasons: tuple[str, ...],
    config: DynamicPointsConfig,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    required = {
        "season_year",
        "game_date",
        "game_id",
        "player_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing walk-forward columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("walk-forward frame must not be empty")

    season_order = (
        frame.groupby("season_year", sort=False)["game_date"]
        .min()
        .sort_values(kind="mergesort")
        .index.astype(str)
        .tolist()
    )
    fold_traces: list[pd.DataFrame] = []
    fold_audits: list[dict[str, object]] = []
    for evaluation_season in evaluation_seasons:
        if evaluation_season not in season_order:
            raise ValueError(f"unknown evaluation season: {evaluation_season}")
        evaluation_index = season_order.index(evaluation_season)
        start_index = evaluation_index - config.train_window_seasons
        if start_index < 0:
            raise ValueError(
                f"evaluation season {evaluation_season} has fewer than "
                f"{config.train_window_seasons} preceding seasons"
            )
        train_seasons = tuple(season_order[start_index:evaluation_index])
        if set(train_seasons) & {evaluation_season}:
            raise AssertionError("evaluation season leaked into training")
        train = frame.loc[frame["season_year"].astype(str).isin(train_seasons)]
        evaluation = frame.loc[
            frame["season_year"].astype(str).eq(evaluation_season)
        ]
        if train["game_date"].max() >= evaluation["game_date"].min():
            raise AssertionError("training dates must precede evaluation dates")

        fit, fit_audit = fit_hierarchy(train, config)
        training_states = filter_training_states(fit, train, config)
        fold_trace = predict_fold(fit, evaluation, training_states, config)
        fold_traces.append(fold_trace)
        fold_audits.append(
            {
                "train_seasons": train_seasons,
                "evaluation_season": evaluation_season,
                **fit_audit,
                "predicted_rows": int(len(fold_trace)),
            }
        )
    if not fold_traces:
        return pd.DataFrame(), fold_audits
    return pd.concat(fold_traces, ignore_index=True), fold_audits


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
