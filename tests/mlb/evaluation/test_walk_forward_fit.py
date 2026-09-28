import numpy as np
import pandas as pd
import pytest

from src.mlb.evaluation.walk_forward_fit import (
    UnestimatedDispersion,
    fit_walk_forward_nb2,
    predict_walk_forward_mean,
)


def test_fit_records_an_estimated_dispersion_and_predicts_positive_means():
    rng = np.random.default_rng(0)
    n = 80
    frame = pd.DataFrame(
        {
            "home_flag": rng.integers(0, 2, n),
            "rate": rng.normal(0.22, 0.03, n),
            "batters_faced": rng.poisson(22, n),
        }
    )
    model = fit_walk_forward_nb2(
        frame,
        target="batters_faced",
        features=("home_flag", "rate"),
        binary=("home_flag",),
        l2=1.0,
    )
    assert model.dispersion_estimated
    pred = predict_walk_forward_mean(model, frame)
    assert pred.shape == (n,)
    assert np.all(pred > 0)


def test_require_convergence_reaches_fit_nb2_and_defaults_off(monkeypatch):
    from src.mlb.models.workload import Nb2Fit

    seen = []

    def fake_fit(*args, **kwargs):
        seen.append(kwargs["require_convergence"])
        return Nb2Fit(
            coef=np.array([1.0, 0.0]),
            alpha=0.1,
            cov=None,
            method="glm",
            dispersion_estimated=True,
        )

    monkeypatch.setattr("src.mlb.evaluation.walk_forward_fit.fit_nb2", fake_fit)
    frame = pd.DataFrame({"home_flag": [1, 0, 1], "y": [4, 5, 6]})
    args = dict(target="y", features=("home_flag",), binary=("home_flag",), l2=1.0)
    fit_walk_forward_nb2(frame, **args)
    fit_walk_forward_nb2(frame, **args, require_convergence=True)
    assert seen == [False, True]


def test_parameter_count_fallback_raises(monkeypatch):
    from src.mlb.models.workload import Nb2Fit

    def fake_fit(*args, **kwargs):
        return Nb2Fit(
            coef=np.array([1.0]),
            alpha=1e-4,
            cov=None,
            method="glm",
            dispersion_estimated=False,
        )

    monkeypatch.setattr(
        "src.mlb.evaluation.walk_forward_fit.fit_nb2",
        fake_fit,
    )
    frame = pd.DataFrame({"home_flag": [1, 0], "y": [4, 5]})
    with pytest.raises(UnestimatedDispersion):
        fit_walk_forward_nb2(
            frame,
            target="y",
            features=("home_flag",),
            binary=("home_flag",),
            l2=1.0,
        )
