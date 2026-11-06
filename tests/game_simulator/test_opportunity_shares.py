from __future__ import annotations

import numpy as np
import pytest

from opportunity_shares import opportunity_shares


def test_shares_normalize_and_zero_unavailable():
    p = opportunity_shares(
        base_weights=np.array([0.31, 0.27, 0.19, 0.15, 0.08]),
        available=np.array([1, 1, 1, 0, 1]),
        projected_minutes=np.array([36.0, 34.0, 32.0, 28.0, 20.0]),
        reference_minutes=np.array([36.0, 34.0, 32.0, 28.0, 20.0]),
    )
    assert p.shape == (5,)
    assert p[3] == 0.0
    assert pytest.approx(p.sum(), abs=1e-12) == 1.0


def test_proj_equals_ref_shares_proportional_to_base_times_availability():
    w = np.array([0.4, 0.4, 0.2])
    p = opportunity_shares(
        base_weights=w,
        available=np.array([1.0, 1.0, 1.0]),
        projected_minutes=np.array([30.0, 30.0, 30.0]),
        reference_minutes=np.array([30.0, 30.0, 30.0]),
    )
    assert np.allclose(p, w / w.sum())


def test_higher_minute_ratio_increases_share():
    p = opportunity_shares(
        base_weights=np.array([0.5, 0.5]),
        available=np.array([1.0, 1.0]),
        projected_minutes=np.array([40.0, 20.0]),
        reference_minutes=np.array([30.0, 30.0]),
    )
    assert p[0] > p[1]


def test_all_unavailable_raises():
    with pytest.raises(ValueError, match="No available players"):
        opportunity_shares(
            base_weights=np.array([0.5, 0.5]),
            available=np.array([0.0, 0.0]),
            projected_minutes=np.array([30.0, 30.0]),
            reference_minutes=np.array([30.0, 30.0]),
        )


def test_negative_base_weights_raises():
    with pytest.raises(ValueError, match="base_weights"):
        opportunity_shares(
            base_weights=np.array([0.5, -0.1]),
            available=np.array([1.0, 1.0]),
            projected_minutes=np.array([30.0, 30.0]),
            reference_minutes=np.array([30.0, 30.0]),
        )


def test_available_outside_binary_values_raises():
    with pytest.raises(ValueError, match="available"):
        opportunity_shares(
            base_weights=np.array([0.5, 0.5]),
            available=np.array([1.0, 2.0]),
            projected_minutes=np.array([30.0, 30.0]),
            reference_minutes=np.array([30.0, 30.0]),
        )


def test_zero_reference_minutes_yields_zero_share():
    shares = opportunity_shares(
        base_weights=np.array([0.5, 0.5]),
        available=np.array([1.0, 1.0]),
        projected_minutes=np.array([30.0, 30.0]),
        reference_minutes=np.array([0.0, 30.0]),
    )
    assert np.array_equal(shares, np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_weights", np.array([0.5, np.nan])),
        ("projected_minutes", np.array([30.0, -1.0])),
        ("reference_minutes", np.array([30.0, np.inf])),
        ("role_factors", np.array([1.0, -1.0])),
        ("lineup_factors", np.array([1.0, np.nan])),
    ],
)
def test_nonnegative_finite_share_inputs_raise(field, value):
    inputs = {
        "base_weights": np.array([0.5, 0.5]),
        "available": np.array([1.0, 1.0]),
        "projected_minutes": np.array([30.0, 30.0]),
        "reference_minutes": np.array([30.0, 30.0]),
    }
    inputs[field] = value
    with pytest.raises(ValueError, match=field):
        opportunity_shares(**inputs)


@pytest.mark.parametrize("minutes_exponent", [-1.0, np.inf])
def test_invalid_minutes_exponent_raises(minutes_exponent):
    with pytest.raises(ValueError, match="minutes_exponent"):
        opportunity_shares(
            base_weights=np.array([0.5, 0.5]),
            available=np.array([1.0, 1.0]),
            projected_minutes=np.array([30.0, 30.0]),
            reference_minutes=np.array([30.0, 30.0]),
            minutes_exponent=minutes_exponent,
        )
