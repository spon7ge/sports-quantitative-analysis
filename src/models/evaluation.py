"""Proper scoring rules for simulated predictive distributions."""

from __future__ import annotations

import numpy as np


def negative_log_likelihood(
    samples: np.ndarray,
    actual: np.ndarray,
    *,
    pseudocount: float = 0.5,
) -> float:
    """Empirical NLL from an ensemble, with Laplace smoothing."""
    draws, outcomes = _align(samples, actual)
    rounded_draws = np.rint(draws)
    rounded_actual = np.rint(outcomes)
    n_draws = rounded_draws.shape[1]
    match = np.sum(
        rounded_draws == rounded_actual[:, None],
        axis=1,
    )
    ordered = np.sort(rounded_draws, axis=1)
    n_unique = np.ones(len(ordered), dtype=int)
    if n_draws > 1:
        n_unique = 1 + np.sum(
            np.diff(ordered, axis=1) != 0,
            axis=1,
        )
    numer = match + pseudocount
    denom = n_draws + pseudocount * (n_unique + 1)
    return float(np.mean(-np.log(numer / denom)))


def crps(samples: np.ndarray, actual: np.ndarray) -> float:
    """Fair sample CRPS (Gneiting and Raftery)."""
    draws, outcomes = _align(samples, actual)
    absolute_error = np.mean(
        np.abs(draws - outcomes[:, None]),
        axis=1,
    )
    pairwise = np.mean(
        np.abs(draws[:, :, None] - draws[:, None, :]),
        axis=(1, 2),
    )

    return float(np.mean(absolute_error - 0.5 * pairwise))


def interval_coverage(
    samples: np.ndarray,
    actual: np.ndarray,
    *,
    lower: float = 0.10,
    upper: float = 0.90,
) -> float:
    """Fraction of outcomes inside the predictive interval."""
    draws, outcomes = _align(samples, actual)
    lower_bound = np.quantile(draws, lower, axis=1)
    upper_bound = np.quantile(draws, upper, axis=1)

    return float(
        np.mean(
            (outcomes >= lower_bound) & (outcomes <= upper_bound)
        )
    )


def probability_integral_transform(
    samples: np.ndarray,
    actual: np.ndarray,
) -> np.ndarray:
    """Row-wise PIT with half-credit for ties."""
    draws, outcomes = _align(samples, actual)
    below = np.mean(draws < outcomes[:, None], axis=1)
    equal = np.mean(draws == outcomes[:, None], axis=1)
    return below + 0.5 * equal


def randomized_probability_integral_transform(
    samples: np.ndarray,
    actual: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Discrete PIT: F(y⁻) + U·(F(y) − F(y⁻)).

    Half-credit PIT has the same row-wise expectation. Randomization
    only spreads atoms (zeros, integer ties) so the histogram can be
    Uniform when the predictive CDF is calibrated.
    """
    draws, outcomes = _align(samples, actual)
    lower = np.mean(draws < outcomes[:, None], axis=1)
    equal = np.mean(draws == outcomes[:, None], axis=1)
    generator = (
        rng if rng is not None else np.random.default_rng()
    )
    unit = generator.uniform(size=len(outcomes))
    return lower + unit * equal


def brier_score(
    samples: np.ndarray,
    actual: np.ndarray,
    line: float,
) -> float:
    """Brier score for P(outcome > line)."""
    draws, outcomes = _align(samples, actual)
    probability = np.mean(draws > line, axis=1)
    indicator = (outcomes > line).astype(float)

    return float(np.mean(np.square(probability - indicator)))


def score_distribution(
    samples: np.ndarray,
    actual: np.ndarray,
    *,
    line: float | None = None,
    include_crps: bool = True,
) -> dict[str, float]:
    """Return the metrics used to compare probabilistic models.

    Pairwise CRPS allocates ``(n, draws, draws)`` and will kill a
    kernel on a full holdout with 2,000 simulations. Pass
    ``include_crps=False`` for large ensembles.
    """
    scores = {
        "negative_log_likelihood": negative_log_likelihood(
            samples,
            actual,
        ),
        "coverage_80": interval_coverage(samples, actual),
    }
    if include_crps:
        scores["crps"] = crps(samples, actual)

    if line is not None:
        scores["brier"] = brier_score(samples, actual, line)

    return scores


def interval_width(
    samples: np.ndarray,
    *,
    lower: float = 0.10,
    upper: float = 0.90,
) -> float:
    """Mean predictive interval width across rows."""
    draws = np.asarray(samples, dtype=float)
    if draws.ndim == 1:
        draws = draws[None, :]
    widths = (
        np.quantile(draws, upper, axis=1)
        - np.quantile(draws, lower, axis=1)
    )
    return float(np.mean(widths))


def pit_histogram(
    pit_values: np.ndarray,
    *,
    bins: int = 10,
) -> np.ndarray:
    """Equal-width PIT counts on [0, 1]."""
    values = np.asarray(pit_values, dtype=float)
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(
        values,
        bins=bins,
        range=(0.0, 1.0),
    )
    return counts


def pit_shape_penalty(
    pit_values: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Total variation distance vs a Uniform PIT histogram."""
    counts = pit_histogram(pit_values, bins=bins)
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probabilities = counts.astype(float) / total
    target = 1.0 / bins
    return float(
        0.5 * np.abs(probabilities - target).sum()
    )


def role_coverage_gap(
    samples: np.ndarray,
    actual: np.ndarray,
    start_rate_10: np.ndarray,
    *,
    threshold: float = 0.5,
    lower: float = 0.10,
    upper: float = 0.90,
) -> dict[str, float]:
    """Starter vs bench interval coverage and absolute gap."""
    draws, outcomes = _align(samples, actual)
    rate = np.asarray(start_rate_10, dtype=float).reshape(-1)
    if len(rate) != len(outcomes):
        raise ValueError(
            "start_rate_10 must have the same number of rows"
        )

    lower_bound = np.quantile(draws, lower, axis=1)
    upper_bound = np.quantile(draws, upper, axis=1)
    inside = (outcomes >= lower_bound) & (
        outcomes <= upper_bound
    )
    finite = np.isfinite(rate)
    starter = finite & (rate >= threshold)
    bench = finite & (rate < threshold)

    starter_coverage, n_starter = _masked_mean(
        inside,
        starter,
    )
    bench_coverage, n_bench = _masked_mean(inside, bench)
    gap = abs(starter_coverage - bench_coverage)
    return {
        "starter_coverage": starter_coverage,
        "bench_coverage": bench_coverage,
        "gap": gap,
        "n_starter": n_starter,
        "n_bench": n_bench,
    }


def score_role_distribution(
    samples: np.ndarray,
    actual: np.ndarray,
    start_rate_10: np.ndarray | None = None,
    **kwargs,
) -> dict[str, float]:
    """score_distribution plus width, PIT, and role coverage."""
    scores = score_distribution(
        samples,
        actual,
        line=kwargs.get("line"),
        include_crps=kwargs.get("include_crps", True),
    )
    scores["width_80"] = interval_width(samples)
    pit = probability_integral_transform(samples, actual)
    scores["pit_mean"] = float(np.mean(pit))
    scores["pit_std"] = float(np.std(pit))
    scores["pit_shape"] = pit_shape_penalty(pit)
    if start_rate_10 is not None:
        scores.update(
            role_coverage_gap(
                samples,
                actual,
                start_rate_10,
                threshold=kwargs.get("threshold", 0.5),
                lower=kwargs.get("lower", 0.10),
                upper=kwargs.get("upper", 0.90),
            )
        )
    return scores


def _masked_mean(
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int]:
    count = int(mask.sum())
    if count == 0:
        return float("nan"), 0
    return float(np.mean(values[mask])), count


def _align(
    samples: np.ndarray,
    actual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    draws = np.asarray(samples, dtype=float)
    outcomes = np.asarray(actual, dtype=float)

    if draws.ndim == 1:
        draws = draws[None, :]

    if outcomes.ndim == 0:
        outcomes = np.array([outcomes], dtype=float)

    if len(draws) != len(outcomes):
        raise ValueError(
            "samples and actual must have the same number of rows"
        )

    return draws, outcomes
