"""Time-aware hyperparameter tuning for XGBoost models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
)
from xgboost import XGBRegressor

from .core import XGBoostConfig

ModelType = Literal["minutes", "count"]

@dataclass(frozen=True)
class TuningConfig:
    """Configuration for expanding-window optimization."""

    trials: int = 75
    folds: int = 4
    minimum_training_dates: int = 60
    maximum_estimators: int = 2_000
    early_stopping_rounds: int = 75
    timeout_seconds: int | None = None
    random_seed: int = 42
    study_name: str | None = None

@dataclass(frozen=True)
class TuningResult:
    model_type: ModelType
    target_column: str
    best_score: float
    best_iteration: int
    best_parameters: dict[str, float | int]
    model_config: XGBoostConfig

def tune_minutes(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str = "minutes",
    date_column: str = "game_date",
    tuning_config: TuningConfig | None = None,
    base_config: XGBoostConfig | None = None,
) -> TuningResult:
    """Tune the XGBoost minutes model using validation MAE."""
    return tune_xgboost(
        frame,
        feature_columns,
        target_column=target_column,
        date_column=date_column,
        model_type="minutes",
        tuning_config=tuning_config,
        base_config=base_config,
    )

def tune_prop(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: Literal["pts", "reb", "ast"],
    date_column: str = "game_date",
    tuning_config: TuningConfig | None = None,
    base_config: XGBoostConfig | None = None,
) -> TuningResult:
    """Tune a count model using mean Poisson deviance."""
    return tune_xgboost(
        frame,
        feature_columns,
        target_column=target_column,
        date_column=date_column,
        model_type="count",
        tuning_config=tuning_config,
        base_config=base_config,
    )

def tune_xgboost(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str,
    date_column: str,
    model_type: ModelType,
    tuning_config: TuningConfig | None = None,
    base_config: XGBoostConfig | None = None,
) -> TuningResult:
    """Run time-aware hyperparameter optimization."""
    tuning = tuning_config or TuningConfig()
    base = base_config or XGBoostConfig()

    data = prepare_tuning_data(
        frame,
        feature_columns,
        target_column=target_column,
        date_column=date_column,
        model_type=model_type,
    )

    splits = expanding_window_splits(
        data,
        date_column=date_column,
        folds=tuning.folds,
        minimum_training_dates=(
            tuning.minimum_training_dates
        ),
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        parameters = suggest_parameters(
            trial,
            tuning,
        )

        fold_scores: list[float] = []
        best_iterations: list[int] = []

        for fold_number, (
            training_index,
            validation_index,
        ) in enumerate(splits):
            training = data.loc[training_index]
            validation = data.loc[validation_index]

            model = build_candidate(
                parameters,
                model_type=model_type,
                tuning=tuning,
            )

            model.fit(
                training[feature_columns],
                training[target_column],
                eval_set=[
                    (
                        validation[feature_columns],
                        validation[target_column],
                    )
                ],
                verbose=False,
            )

            predictions = model.predict(
                validation[feature_columns]
            )
            predictions = np.clip(
                predictions,
                0,
                None,
            )

            score = score_predictions(
                validation[target_column].to_numpy(),
                predictions,
                model_type=model_type,
            )

            fold_scores.append(score)

            best_iteration = getattr(
                model,
                "best_iteration",
                tuning.maximum_estimators - 1,
            )
            best_iterations.append(
                int(best_iteration) + 1
            )

            running_score = float(
                np.mean(fold_scores)
            )

            trial.report(
                running_score,
                step=fold_number,
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr(
            "best_iterations",
            best_iterations,
        )
        trial.set_user_attr(
            "recommended_estimators",
            int(np.median(best_iterations)),
        )
        trial.set_user_attr(
            "fold_scores",
            fold_scores,
        )

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(
        seed=tuning.random_seed,
        multivariate=True,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=1,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=tuning.study_name,
    )

    study.optimize(
        objective,
        n_trials=tuning.trials,
        timeout=tuning.timeout_seconds,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    best_parameters = dict(
        best_trial.params
    )

    best_iteration = int(
        best_trial.user_attrs[
            "recommended_estimators"
        ]
    )

    model_config = replace(
        base,
        n_estimators=best_iteration,
        learning_rate=float(
            best_parameters["learning_rate"]
        ),
        max_depth=int(
            best_parameters["max_depth"]
        ),
        min_child_weight=float(
            best_parameters["min_child_weight"]
        ),
        subsample=float(
            best_parameters["subsample"]
        ),
        colsample_bytree=float(
            best_parameters["colsample_bytree"]
        ),
        reg_alpha=float(
            best_parameters["reg_alpha"]
        ),
        reg_lambda=float(
            best_parameters["reg_lambda"]
        ),
        early_stopping_rounds=(
            tuning.early_stopping_rounds
        ),
        random_seed=tuning.random_seed,
    )

    return TuningResult(
        model_type=model_type,
        target_column=target_column,
        best_score=float(best_trial.value),
        best_iteration=best_iteration,
        best_parameters=best_parameters,
        model_config=model_config,
    )

def suggest_parameters(
    trial: optuna.Trial,
    tuning: TuningConfig,
) -> dict[str, float | int]:
    """Define the XGBoost search space."""
    return {
        "n_estimators": tuning.maximum_estimators,
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.10,
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            7,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            2.0,
            30.0,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.60,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.60,
            1.0,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-4,
            10.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.10,
            30.0,
            log=True,
        ),
    }

def build_candidate(
    parameters: dict[str, float | int],
    *,
    model_type: ModelType,
    tuning: TuningConfig,
) -> XGBRegressor:
    """Create one trial model."""
    objective = (
        "count:poisson"
        if model_type == "count"
        else "reg:squarederror"
    )

    evaluation_metric = (
        "poisson-nloglik"
        if model_type == "count"
        else "mae"
    )

    return XGBRegressor(
        **parameters,
        objective=objective,
        eval_metric=evaluation_metric,
        early_stopping_rounds=(
            tuning.early_stopping_rounds
        ),
        tree_method="hist",
        random_state=tuning.random_seed,
        n_jobs=-1,
    )

def score_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    model_type: ModelType,
) -> float:
    """Calculate the optimization metric."""
    if model_type == "minutes":
        return float(
            mean_absolute_error(
                actual,
                predicted,
            )
        )

    # Poisson deviance requires strictly positive predictions.
    predicted = np.clip(
        predicted,
        1e-6,
        None,
    )

    return float(
        mean_poisson_deviance(
            actual,
            predicted,
        )
    )

def prepare_tuning_data(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str,
    date_column: str,
    model_type: ModelType,
) -> pd.DataFrame:
    """Validate and normalize the tuning frame."""
    required = {
        target_column,
        date_column,
        *feature_columns,
    }
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing tuning columns: {sorted(missing)}"
        )

    data = frame[
        [
            date_column,
            target_column,
            *feature_columns,
        ]
    ].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )
    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce",
    )

    for column in feature_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=[date_column, target_column]
    )
    data = data.sort_values(
        date_column
    ).reset_index(drop=True)

    if model_type == "count":
        data = data.loc[
            data[target_column] >= 0
        ]

    if data.empty:
        raise ValueError(
            "No valid rows remain for tuning."
        )

    return data

def expanding_window_splits(
    frame: pd.DataFrame,
    *,
    date_column: str,
    folds: int,
    minimum_training_dates: int,
) -> list[tuple[pd.Index, pd.Index]]:
    """Create expanding-training, future-validation folds."""
    if folds < 2:
        raise ValueError(
            "At least two folds are required."
        )

    unique_dates = np.sort(
        frame[date_column].unique()
    )

    remaining_dates = (
        len(unique_dates)
        - minimum_training_dates
    )

    if remaining_dates < folds:
        raise ValueError(
            "Not enough dates for the requested folds. "
            f"Found {len(unique_dates)}, need at least "
            f"{minimum_training_dates + folds}."
        )

    validation_size = (
        remaining_dates // folds
    )

    splits = []

    for fold in range(folds):
        training_end = (
            minimum_training_dates
            + fold * validation_size
        )

        validation_end = (
            len(unique_dates)
            if fold == folds - 1
            else training_end + validation_size
        )

        training_dates = unique_dates[
            :training_end
        ]
        validation_dates = unique_dates[
            training_end:validation_end
        ]

        training_index = frame.index[
            frame[date_column].isin(
                training_dates
            )
        ]
        validation_index = frame.index[
            frame[date_column].isin(
                validation_dates
            )
        ]

        if (
            training_index.empty
            or validation_index.empty
        ):
            continue

        splits.append(
            (
                training_index,
                validation_index,
            )
        )

    if len(splits) != folds:
        raise ValueError(
            "Unable to create all requested folds."
        )

    return splits

def save_tuning_result(
    result: TuningResult,
    path: str | Path,
    *,
    extra_metadata: dict | None = None,
) -> Path:
    """Save selected parameters and tuning metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model_type": result.model_type,
        "target_column": result.target_column,
        "best_score": result.best_score,
        "best_iteration": result.best_iteration,
        "best_parameters": result.best_parameters,
        "model_config": asdict(
            result.model_config
        ),
        "metadata": extra_metadata or {},
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    return output_path