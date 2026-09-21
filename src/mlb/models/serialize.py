"""Joblib/pickle round-trip for model bundles."""

from __future__ import annotations

import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


def _to_dict(bundle: Any) -> dict:
    if isinstance(bundle, dict):
        return dict(bundle)
    if is_dataclass(bundle) and not isinstance(bundle, type):
        payload = asdict(bundle)
        payload["model_class"] = type(bundle).__name__
        return payload
    if hasattr(bundle, "__dict__"):
        return dict(bundle.__dict__)
    raise TypeError(f"Cannot serialize bundle of type {type(bundle)!r}")


def save_model(path, bundle) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_dict(bundle)
    if joblib is not None:
        joblib.dump(payload, destination)
    else:  # pragma: no cover
        destination.write_bytes(pickle.dumps(payload))


def load_model(path) -> dict:
    source = Path(path)
    if joblib is not None:
        loaded = joblib.load(source)
    else:  # pragma: no cover
        loaded = pickle.loads(source.read_bytes())
    if isinstance(loaded, dict):
        return loaded
    return _to_dict(loaded)


def as_strikeout_model(bundle: Any):
    """Rebuild a ``StrikeoutModel`` from a serialized dict or pass through."""
    from src.mlb.models.strikeouts import StrikeoutModel

    if isinstance(bundle, StrikeoutModel):
        return bundle
    if not isinstance(bundle, dict):
        raise TypeError(f"Cannot hydrate StrikeoutModel from {type(bundle)!r}")
    cov = bundle.get("cov")
    extra = dict(bundle.get("extra") or {})
    dropped = bundle.get("dropped_features") or extra.get("dropped_features") or {}
    centers = bundle.get("centers") or extra.get("centers") or {}
    scales = bundle.get("scales") or extra.get("scales") or {}
    return StrikeoutModel(
        feature_names=tuple(bundle["feature_names"]),
        coef=np.asarray(bundle["coef"], dtype=float),
        alpha=float(bundle["alpha"]),
        medians={str(key): float(value) for key, value in bundle["medians"].items()},
        method=str(bundle.get("method", "glm")),
        cov=None if cov is None else np.asarray(cov, dtype=float),
        model_version=str(bundle.get("model_version", "nb_k_v1")),
        extra=extra,
        dropped_features={str(key): str(value) for key, value in dict(dropped).items()},
        centers={str(key): float(value) for key, value in dict(centers).items()},
        scales={str(key): float(value) for key, value in dict(scales).items()},
        train_start=str(bundle.get("train_start", extra.get("train_start", ""))),
        train_end=str(bundle.get("train_end", extra.get("train_end", ""))),
        n_train=int(bundle.get("n_train", extra.get("n_train", 0)) or 0),
        matrix_rank=int(bundle.get("matrix_rank", extra.get("matrix_rank", 0)) or 0),
        condition_number=float(
            bundle.get("condition_number", extra.get("condition_number", float("nan")))
        ),
        fold=str(bundle.get("fold", extra.get("fold", ""))),
        uses_bf_offset=bool(
            bundle.get("uses_bf_offset", extra.get("uses_bf_offset", False))
        ),
    )


def as_workload_model(bundle: Any):
    """Rebuild a ``WorkloadModel`` from a serialized dict or pass through."""
    from src.mlb.models.workload import WorkloadModel

    if isinstance(bundle, WorkloadModel):
        return bundle
    if not isinstance(bundle, dict):
        raise TypeError(f"Cannot hydrate WorkloadModel from {type(bundle)!r}")
    cov = bundle.get("cov")
    extra = dict(bundle.get("extra") or {})
    dropped = bundle.get("dropped_features") or extra.get("dropped_features") or {}
    centers = bundle.get("centers") or extra.get("centers") or {}
    scales = bundle.get("scales") or extra.get("scales") or {}
    return WorkloadModel(
        feature_names=tuple(bundle["feature_names"]),
        coef=np.asarray(bundle["coef"], dtype=float),
        alpha=float(bundle["alpha"]),
        medians={str(key): float(value) for key, value in bundle["medians"].items()},
        mean_pitches_per_bf=float(bundle.get("mean_pitches_per_bf", 3.85)),
        mean_outs_per_bf=float(bundle.get("mean_outs_per_bf", 0.70)),
        logit_coef=np.asarray(bundle["logit_coef"], dtype=float),
        early_exit_bf=int(bundle.get("early_exit_bf", 15)),
        method=str(bundle.get("method", "glm")),
        cov=None if cov is None else np.asarray(cov, dtype=float),
        model_version=str(bundle.get("model_version", "nb_bf_v1")),
        extra=extra,
        dropped_features={str(key): str(value) for key, value in dict(dropped).items()},
        centers={str(key): float(value) for key, value in dict(centers).items()},
        scales={str(key): float(value) for key, value in dict(scales).items()},
        n_train=int(bundle.get("n_train", extra.get("n_train", 0)) or 0),
        matrix_rank=int(bundle.get("matrix_rank", extra.get("matrix_rank", 0)) or 0),
        condition_number=float(
            bundle.get("condition_number", extra.get("condition_number", float("nan")))
        ),
    )
