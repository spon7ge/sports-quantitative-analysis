"""Save / load quantile model bundles and run inference."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from models.shared.metrics import MINUTE_Q50_TIERS, RATE_Q50_TIERS

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
    metadata: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    save_dir: str | Path | None = None,
) -> Path:
    """Persist quantile models + metrics under ``models/saved_models/``.

    The file is ``{stem}_01.joblib``, then ``_02``, and so on. A dated
    file already in the folder does not count toward that sequence.
    """
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
        "metadata": dict(metadata) if metadata is not None else None,
    }
    if calibration is not None:
        payload = dict(calibration)
        scaled = payload.get("method") == "scaled_quantile"
        pit = payload.get("method") == "pit_cdf"
        grouped = "group_definition" in payload and "offsets" in payload
        if scaled and ("offsets" not in payload or "taus" not in payload):
            raise ValueError("scaled calibration needs offsets and taus")
        if pit and any(key not in payload for key in ("pit_x", "pit_y", "taus")):
            raise ValueError("pit calibration needs the isotonic knots and taus")
        if not scaled and not pit and not grouped:
            raise ValueError(
                "calibration must be the offset payload, including group_definition"
            )
        payload["fingerprint"] = booster_fingerprint(models_ho, features)
        bundle["calibration"] = payload

    save_path = _next_numbered_path(out_dir, artifact_stem)
    joblib.dump(bundle, save_path)
    print(f"Saved to {save_path}")
    return save_path


def _next_numbered_path(out_dir: Path, artifact_stem: str) -> Path:
    """Next ``{stem}_01.joblib`` that is not already in ``out_dir``."""
    pattern = re.compile(rf"{re.escape(artifact_stem)}_(\d+)$")
    highest = 0
    for path in out_dir.glob(f"{artifact_stem}_*.joblib"):
        match = pattern.fullmatch(path.stem)
        if match:
            highest = max(highest, int(match.group(1)))
    return out_dir / f"{artifact_stem}_{highest + 1:02d}.joblib"


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a saved quantile model bundle.

    A bundle that carries a calibration table is refused when the
    boosters or the tier cuts are not the ones stored with that table.
    """
    bundle = joblib.load(path)
    _check_calibration(bundle)
    print(f"Loaded {path}")
    return bundle


def booster_fingerprint(models: Mapping[str, Any], feature_names: Sequence[str]) -> str:
    """Hash the booster bytes the calibration table was stored against."""
    digest = sha256()
    digest.update("\n".join(str(name) for name in feature_names).encode())
    ordered = sorted(models, key=lambda key: float(str(key).split("_", 1)[1]))
    for key in ordered:
        raw = models[key].get_booster().save_raw()
        if isinstance(raw, str):
            raw = raw.encode()
        digest.update(str(key).encode())
        digest.update(raw)
    return digest.hexdigest()


def _expected_tiers(calibration: Mapping[str, Any]):
    target = calibration.get("target", "minutes")
    if target == "minutes":
        return tuple(MINUTE_Q50_TIERS)
    if target == "pts_per_min":
        return tuple(RATE_Q50_TIERS)
    raise ValueError(f"calibration target {target}")


def _check_calibration(bundle: Mapping[str, Any]) -> None:
    calibration = bundle.get("calibration")
    if calibration is None:
        return
    found = booster_fingerprint(bundle["quantile_models"], bundle["feature_names"])
    if found != calibration.get("fingerprint"):
        raise ValueError("booster fingerprint does not match the calibration table")
    if calibration.get("method") == "scaled_quantile":
        offsets = np.asarray(calibration["offsets"], dtype=float).reshape(-1)
        taus = np.asarray(calibration["taus"], dtype=float).reshape(-1)
        if offsets.shape != taus.shape or not np.isfinite(offsets).all():
            raise ValueError("scaled offsets")
        return
    if calibration.get("method") == "pit_cdf":
        pit_x = np.asarray(calibration["pit_x"], dtype=float).reshape(-1)
        pit_y = np.asarray(calibration["pit_y"], dtype=float).reshape(-1)
        taus = np.asarray(calibration["taus"], dtype=float).reshape(-1)
        if pit_x.size == 0 or pit_x.shape != pit_y.shape:
            raise ValueError("pit map")
        if not np.isfinite(pit_x).all() or not np.isfinite(pit_y).all():
            raise ValueError("pit map")
        if taus.size < 2 or not np.isfinite(taus).all():
            raise ValueError("pit taus")
        return
    definition = calibration.get("group_definition")
    if not isinstance(definition, Mapping) or "tiers" not in definition:
        raise ValueError("calibration payload is missing group_definition")
    stored = definition["tiers"]
    tiers = tuple(tuple(item) for item in stored)
    if tiers != _expected_tiers(calibration):
        raise ValueError("calibration tiers do not match the group definition")


_TRAINING_PARAM_KEYS = (
    "objective",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "min_child_weight",
    "n_jobs",
    "random_state",
    "early_stopping_rounds",
)


def quantile_training_params(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Fold-model hyperparameters from a saved bundle.

    ``quantile_alpha`` is omitted so each quantile can set its own level.
    """
    model = bundle["quantile_models"]["q_0.50"]
    raw = model.get_params()
    missing = [key for key in _TRAINING_PARAM_KEYS if key not in raw]
    if missing:
        raise KeyError(f"bundle is missing training params: {missing}")
    return {key: raw[key] for key in _TRAINING_PARAM_KEYS}


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


def predict_calibrated_quantiles(
    bundle: Mapping[str, Any],
    player_features: pd.DataFrame,
) -> pd.DataFrame:
    """Sort, replace cold starts, apply the stored calibration, and clip.

    A ``pit_cdf`` payload maps the whole CDF through one isotonic PIT
    recalibrator, then reads the quantile grid back off that CDF. A
    ``scaled_quantile`` payload shifts each tau by that row's q10–q90
    width. Grouped payloads still add one offset per minute or rate tier.
    Groups are read from the base q50, after the cold-start replacement.
    """
    from models.shared.conformal import (  # noqa: PLC0415
        apply_offsets,
        base_knots,
        calibration_group,
    )

    calibration = bundle["calibration"]
    names = list(bundle["feature_names"])
    raw = predict_quantiles(
        bundle["quantile_models"],
        names,
        player_features.loc[:, names],
    )
    if calibration.get("method") in {"scaled_quantile", "pit_cdf"}:
        levels = np.asarray(calibration["taus"], dtype=float)
        keys = [f"q_{level:.2f}" for level in levels]
        if "role_quantiles" in calibration:
            lag_col = str(calibration.get("lag_col", "min_lag_1"))
            if lag_col not in player_features.columns:
                raise KeyError(lag_col)
            base = base_knots(
                raw,
                player_features["starting"],
                player_features[lag_col],
                role_quantiles=calibration["role_quantiles"],
            )
        else:
            base = raw
        grid = np.column_stack([base[key].to_numpy(dtype=float) for key in keys])
        if calibration.get("method") == "pit_cdf":
            from models.shared.points_calibration import (  # noqa: PLC0415
                calibrated_quantile_grid,
                pit_map_from_thresholds,
            )

            grid.sort(axis=1)
            calibrator = pit_map_from_thresholds(
                calibration["pit_x"], calibration["pit_y"]
            )
            adjusted = calibrated_quantile_grid(grid, levels, calibrator)
        else:
            from models.shared.quantile_offsets import (  # noqa: PLC0415
                apply_offsets as apply_scaled_offsets,
            )

            adjusted = apply_scaled_offsets(grid, calibration["offsets"], levels)
        adjusted = np.clip(
            adjusted,
            float(calibration["clip_low"]),
            float(calibration["clip_high"]),
        )
        return pd.DataFrame(adjusted, index=raw.index, columns=keys)
    lag_col = str(calibration.get("lag_col", "min_lag_1"))
    if lag_col not in player_features.columns:
        raise KeyError(lag_col)
    lag = player_features[lag_col]
    stored = calibration.get("group_definition", {}).get("tiers", MINUTE_Q50_TIERS)
    tiers = tuple(tuple(item) for item in stored)
    base = base_knots(
        raw,
        player_features["starting"],
        lag,
        role_quantiles=calibration["role_quantiles"],
    )
    groups = calibration_group(base["q_0.50"], lag, tiers=tiers)
    upper = calibration.get("clip_high", calibration["max_minutes"])
    lower = calibration.get("clip_low", 1e-3)
    return apply_offsets(
        base,
        groups,
        calibration["offsets"],
        float(upper),
        lower=float(lower),
    )
