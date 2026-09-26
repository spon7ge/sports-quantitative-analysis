"""Save / load quantile model bundles and run inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = _REPO_ROOT / "models" / "saved_models"


def save_model_bundle(
    models_ho: dict[str, Any],
    *,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    wf_results: list[dict[str, Any]],
    ho_metrics: dict[str, Any],
    features: list[str],
    artifact_stem: str,
    naive_holdout_results: dict[str, Any] | None = None,
    save_dir: str | Path | None = None,
) -> Path:
    """Persist quantile models + metrics under ``models/saved_models/``."""
    out_dir = Path(save_dir) if save_dir is not None else DEFAULT_SAVE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    for r in wf_results:
        fold_metrics.append({
            k: v for k, v in r.items()
            if k not in ("train_mask", "val_mask")
        })

    bundle = {
        "quantile_models": models_ho,
        "feature_names": list(features),
        "fold_metrics": fold_metrics,
        "holdout_metrics": ho_metrics,
        "naive_baseline": naive_holdout_results,
        "train_end": train_df["game_date"].max(),
        "val_end": holdout_df["game_date"].max(),
    }

    save_path = out_dir / f"{artifact_stem}_{holdout_df['game_date'].max().date()}.joblib"
    joblib.dump(bundle, save_path)
    print(f"Saved to {save_path}")
    return save_path


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a saved quantile model bundle."""
    bundle = joblib.load(path)
    print(f"Loaded {path}")
    return bundle


def monotonize_quantiles(
    preds: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Sort each row so a lower quantile cannot exceed a higher one.

    The multiset of predicted values on a row is unchanged. Only their
    assignment to quantile levels is rearranged.
    """
    ordered = sorted(
        preds,
        key=lambda key: float(str(key).split("_", 1)[1]),
    )
    grid = np.column_stack(
        [np.asarray(preds[key], dtype=float) for key in ordered]
    )
    grid.sort(axis=1)
    return {key: grid[:, index] for index, key in enumerate(ordered)}


def predict_quantiles(
    models: dict[str, Any],
    feature_names: list[str],
    player_features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict every saved quantile, columns ordered from low to high."""
    assert list(player_features.columns) == list(feature_names), (
        f"Feature mismatch — expected {feature_names}"
    )
    ordered = sorted(
        models,
        key=lambda key: float(str(key).split("_", 1)[1]),
    )
    return pd.DataFrame({
        key: models[key].predict(player_features) for key in ordered
    })
