"""Ridge minutes model and residual banks."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from game_rotations.checks import _as_float, _required
from game_rotations.constants import (
    CONDITIONAL,
    MU_BINS,
    _ALPHAS,
    _FEATURES,
    _SIGMA_FLOOR,
)

class _RecentMinutes:
    def predict(self, frame):
        return _as_float(frame["min_mean_10"])


class _ConditionalMinutes:
    def __init__(self, pipeline, medians, alpha):
        self.pipeline = pipeline
        self.medians = medians
        self.alpha = float(alpha)
        self.columns = _FEATURES

    @classmethod
    def fit(cls, roles, alpha):
        medians = _medians(roles)

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=float(alpha))),
            ]
        )

        model.fit(_design(roles, medians), _as_float(roles["minutes"]))
        return cls(model, medians, alpha)

    def predict(self, frame):
        _required(frame, CONDITIONAL)

        return np.asarray(
            self.pipeline.predict(_design(frame, self.medians)),
            dtype=float,
        )


def _choose_alpha(train, validation):
    target = _as_float(validation["minutes"])
    scores = []

    for alpha in _ALPHAS:
        probe = _ConditionalMinutes.fit(train, alpha)

        predicted = np.clip(
            probe.predict(validation),
            0,
            _as_float(validation["game_minutes"]),
        )

        scores.append(np.mean((predicted - target) ** 2))

    return _ALPHAS[int(np.argmin(scores))]


def _numeric_features(frame):
    _required(frame, CONDITIONAL)

    return (
        frame.loc[:, list(CONDITIONAL)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def _medians(frame):
    return _numeric_features(frame).median().fillna(0).to_dict()


def _design(frame, medians):
    values = _numeric_features(frame).fillna(medians)
    recent = values["min_mean_10"]

    values["min_mean_x_margin"] = recent * values["team_final_margin"]
    values["min_mean_x_abs_margin"] = recent * values["abs_final_margin"]
    values["min_mean_x_blowout"] = recent * values["blowout"]
    values["min_mean_x_n_ot"] = recent * values["n_ot"]
    values["starter_x_blowout"] = values["is_starter"] * values["blowout"]
    values["starter_x_margin"] = values["is_starter"] * values["team_final_margin"]

    result = values.loc[:, list(_FEATURES)].to_numpy(dtype=float)

    if not np.isfinite(result).all():
        raise ValueError("feature interactions overflowed or have missing fills")

    return result


def _split_dates(frame, validation, calibration, holdout):
    fractions = np.asarray([validation, calibration, holdout], dtype=float)

    if (
        not np.isfinite(fractions).all()
        or (fractions <= 0).any()
        or fractions.sum() >= 1
    ):
        raise ValueError("split fractions must be positive and sum to less than one")

    dates = np.sort(frame["game_date"].unique())

    if len(dates) < 4:
        raise ValueError(
            "need four distinct dates for train, validation, calibration and holdout"
        )

    counts = np.maximum(1, np.floor(len(dates) * fractions).astype(int))

    if counts.sum() >= len(dates):
        raise ValueError("split fractions leave no training dates")

    train_end = len(dates) - counts.sum()
    tune_end = train_end + counts[0]
    cal_end = tune_end + counts[1]

    chunks = (
        dates[:train_end],
        dates[train_end:tune_end],
        dates[tune_end:cal_end],
        dates[cal_end:],
    )

    return tuple(
        frame.loc[frame["game_date"].isin(chunk)].copy()
        for chunk in chunks
    )


def _recent_mean(frame, fallback):
    if "min_mean_10" not in frame:
        return np.full(len(frame), fallback, dtype=float)

    values = _as_float(frame["min_mean_10"])

    return np.where(
        np.isfinite(values) & (values >= 0),
        values,
        fallback,
    )


def _role_bins(mu):
    return np.clip(
        np.digitize(mu, MU_BINS) - 1,
        0,
        len(MU_BINS) - 2,
    )


def _residual_banks(mu, resid, min_samples):
    mu = np.asarray(mu, dtype=float)
    resid = np.asarray(resid, dtype=float)
    good = np.isfinite(mu) & np.isfinite(resid)
    mu, resid = mu[good], resid[good]

    if not len(resid):
        raise ValueError("no finite out-of-sample residuals for calibration")

    bins = _role_bins(mu)
    pieces = [resid[bins == i] for i in range(len(MU_BINS) - 1)]

    return {
        "pooled": resid.copy(),
        "groups": [
            piece.copy() if len(piece) >= min_samples else resid.copy()
            for piece in pieces
        ],
        "counts": np.array([len(piece) for piece in pieces]),
    }


def _check_banks(bank):
    if bank is None:
        return None

    if not isinstance(bank, Mapping) or len(bank.get("groups", [])) != len(MU_BINS) - 1:
        raise ValueError("invalid residual bank structure")

    result = dict(bank)
    result["groups"] = [
        np.asarray(piece, dtype=float).reshape(-1)
        for piece in bank["groups"]
    ]

    if any(
        not len(piece) or not np.isfinite(piece).all()
        for piece in result["groups"]
    ):
        raise ValueError("residual groups must be finite and nonempty")

    return result


def _sample_residuals(bank, mu, rng):
    bins = _role_bins(mu)
    result = np.empty(np.shape(mu))

    for i, piece in enumerate(bank["groups"]):
        mask = bins == i
        result[mask] = rng.choice(piece, size=int(mask.sum()))

    return result


def _sigma_table(mu, resid):
    mu = np.asarray(mu, dtype=float)
    resid = np.asarray(resid, dtype=float)
    good = np.isfinite(mu) & np.isfinite(resid)
    mu, resid = mu[good], resid[good]

    overall = (
        max(float(np.sqrt(np.mean(resid**2))), _SIGMA_FLOOR)
        if len(resid)
        else _SIGMA_FLOOR
    )
    bins = _role_bins(mu)

    return np.array(
        [
            max(
                float(np.sqrt(np.mean(resid[bins == i] ** 2))),
                _SIGMA_FLOOR,
            )
            if (bins == i).sum() >= 40
            else overall
            for i in range(len(MU_BINS) - 1)
        ]
    )


def _check_sigma(values):
    if values is None:
        return None

    values = np.asarray(values, dtype=float).reshape(-1)

    if (
        values.shape != (len(MU_BINS) - 1,)
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        raise ValueError("sigma tables need one positive finite value per role bin")

    return values.copy()
