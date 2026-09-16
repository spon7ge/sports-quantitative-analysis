"""Over/under/push settlement from a discrete PMF."""

from __future__ import annotations

from src.mlb.models.pmf import negative_binomial_pmf, pmf_cdf
from src.mlb.models.props import prop_probabilities


def test_half_line_has_no_push_integer_line_does() -> None:
    pmf = negative_binomial_pmf(6.0, 0.1, 15, 0.001)[0]
    half = prop_probabilities(pmf, 5.5)
    integer = prop_probabilities(pmf, 5)
    assert half["p_push"] == 0.0
    assert integer["p_push"] == pmf[5]
    assert abs(integer["p_over"] + integer["p_under"] + integer["p_push"] - 1.0) < 1e-10
    assert abs(half["p_over"] + half["p_under"] - 1.0) < 1e-10


def test_prop_over_4_5_is_one_minus_f4() -> None:
    pmf = negative_binomial_pmf(6.0, 0.1, 15, 0.001)[0]
    cdf = pmf_cdf(pmf)
    props = prop_probabilities(pmf, 4.5)
    assert props["p_push"] == 0.0
    assert props["p_over"] == 1.0 - cdf[4]
    assert props["p_under"] == cdf[4]
