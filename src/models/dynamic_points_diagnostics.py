"""Pure deployment diagnostics for canonical dynamic-points traces."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import stats

from src.models.evaluation import crps, probability_integral_transform

_LEVELS = (0.50, 0.80, 0.90, 0.95)
_TIER_LABELS = ("0-9", "10-29", "30-59", "60+")
_CRPS_NODES, _CRPS_WEIGHTS = np.polynomial.legendre.leggauss(2048)
_CRPS_PROBABILITIES = (_CRPS_NODES + 1.0) / 2.0
_CRPS_WEIGHTS = _CRPS_WEIGHTS / 2.0


def _validate_trace(
    trace: pd.DataFrame,
    *,
    caller: str,
    numeric: Iterable[str] = (),
    other: Iterable[str] = (),
) -> None:
    required = set(numeric) | set(other)
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"missing {caller} columns: {sorted(missing)}")
    if trace.empty:
        raise ValueError(f"{caller} trace must not be empty")
    for column in numeric:
        if not pd.api.types.is_numeric_dtype(trace[column]):
            raise ValueError(f"{caller} column {column!r} must be numeric")
        values = trace[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{caller} column {column!r} must contain finite values")
    for column in other:
        if trace[column].isna().any():
            raise ValueError(f"{caller} column {column!r} must not contain null values")


def _validate_predictive_inputs(trace: pd.DataFrame, *, caller: str) -> None:
    _validate_trace(
        trace,
        caller=caller,
        numeric=("innovation", "df", "predictive_var"),
    )
    if (trace["df"] <= 0).any():
        raise ValueError(f"{caller} column 'df' must be positive")
    if (trace["predictive_var"] <= 0).any():
        raise ValueError(f"{caller} column 'predictive_var' must be positive")
    if "z" in trace and not np.allclose(
        trace["z"],
        trace["innovation"] / np.sqrt(trace["predictive_var"]),
    ):
        raise ValueError(
            f"{caller} column 'z' must match innovation / predictive scale"
        )


def _validate_boolean(trace: pd.DataFrame, column: str, *, caller: str) -> None:
    if not pd.api.types.is_bool_dtype(trace[column]):
        raise ValueError(f"{caller} column {column!r} must be boolean")


def _sample_sd(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1)) if len(array) > 1 else float("nan")


def _standard_error(sd: float, n: int) -> float:
    return sd / np.sqrt(n) if n > 1 else float("nan")


def games_played_tier(games: pd.Series) -> pd.Categorical:
    """Assign pregame experience counts to deployment-reporting tiers."""
    numeric = pd.to_numeric(games, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("games must contain finite non-negative integers")
    return pd.cut(
        numeric,
        bins=[-1, 9, 29, 59, np.inf],
        labels=_TIER_LABELS,
    )


def _coverage(frame: pd.DataFrame, level: float) -> float:
    if frame.empty:
        return float("nan")
    quantile = stats.t.ppf(
        (1.0 + level) / 2.0,
        frame["df"].to_numpy(dtype=float),
    )
    half_width = quantile * np.sqrt(frame["predictive_var"].to_numpy(dtype=float))
    error = np.abs(frame["innovation"].to_numpy(dtype=float))
    return float(np.mean(error <= half_width))


def _coverage_fields(frame: pd.DataFrame) -> dict[str, float]:
    return {
        f"coverage_{int(level * 100)}": _coverage(frame, level) for level in _LEVELS
    }


def core_calibration(trace: pd.DataFrame) -> dict[str, object]:
    """Summarize aggregate residual, interval, and PIT calibration."""
    _validate_trace(
        trace,
        caller="core_calibration",
        numeric=("z", "innovation", "pit", "df", "predictive_var"),
        other=("season_year",),
    )
    _validate_predictive_inputs(trace, caller="core_calibration")
    if ((trace["pit"] < 0) | (trace["pit"] > 1)).any():
        raise ValueError("core_calibration column 'pit' must be in [0, 1]")

    z = trace["z"].to_numpy(dtype=float)
    innovation = trace["innovation"].to_numpy(dtype=float)
    result: dict[str, object] = {
        "n": len(trace),
        "seasons": tuple(sorted(trace["season_year"].astype(str).unique())),
        "mean_z": float(np.mean(z)),
        "sd_z": _sample_sd(z),
        "rmse": float(np.sqrt(np.mean(innovation**2))),
        "mae": float(np.mean(np.abs(innovation))),
        **_coverage_fields(trace),
        "pit_histogram": np.histogram(
            trace["pit"].to_numpy(dtype=float),
            bins=np.linspace(0.0, 1.0, 11),
        )[0],
    }
    return result


def _group_records(
    frame: pd.DataFrame,
    group_column: str,
    metric_column: str,
    metric_name: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for group, subset in frame.groupby(group_column, observed=True, sort=True):
        values = subset[metric_column].to_numpy(dtype=float)
        sd = _sample_sd(values)
        records.append(
            {
                group_column: group,
                "n": len(subset),
                metric_name: float(np.mean(values)),
                f"{metric_name}_sd": sd,
                f"{metric_name}_se": _standard_error(sd, len(subset)),
            }
        )
    return records


def variance_decomposition(trace: pd.DataFrame) -> pd.DataFrame:
    """Report squared-scale shares and calibration by experience tier.

    ``predictive_var`` is the variance of the latent Gaussian mixture and the
    squared scale of the Student-t predictive law. It is not the finite-df
    Student-t mathematical variance.
    """
    caller = "variance_decomposition"
    _validate_trace(
        trace,
        caller=caller,
        numeric=(
            "games_in_model_before",
            "state_var_before",
            "obs_var",
            "predictive_var",
            "z",
            "innovation",
            "df",
        ),
    )
    _validate_predictive_inputs(trace, caller=caller)
    if (trace[["state_var_before", "obs_var"]] <= 0).any().any():
        raise ValueError(f"{caller} variance components must be positive")
    squared_scale = trace["state_var_before"] + trace["obs_var"]
    if not np.allclose(trace["predictive_var"], squared_scale):
        raise ValueError(
            f"{caller} predictive_var must equal the Student-t squared scale"
        )

    frame = trace.copy()
    frame["games_tier"] = games_played_tier(frame["games_in_model_before"])
    records: list[dict[str, object]] = []
    for tier, subset in frame.groupby("games_tier", observed=True, sort=False):
        state_share = subset["state_var_before"] / subset["predictive_var"]
        observation_share = subset["obs_var"] / subset["predictive_var"]
        records.append(
            {
                "games_tier": tier,
                "n": len(subset),
                "state_scale_squared_share": float(state_share.mean()),
                "observation_scale_squared_share": float(
                    observation_share.mean()
                ),
                "mean_z": float(subset["z"].mean()),
                "sd_z": _sample_sd(subset["z"]),
                **_coverage_fields(subset),
            }
        )
    return pd.DataFrame.from_records(records)


def shrinkage_summary(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize residual shrinkage by experience and state decile."""
    caller = "shrinkage_summary"
    _validate_trace(
        trace,
        caller=caller,
        numeric=(
            "games_in_model_before",
            "state_before",
            "innovation",
            "z",
        ),
    )
    frame = trace.copy()
    frame["games_tier"] = games_played_tier(frame["games_in_model_before"])
    tier_records: list[dict[str, object]] = []
    for tier, subset in frame.groupby("games_tier", observed=True, sort=False):
        tier_records.append(
            {
                "games_tier": tier,
                "n": len(subset),
                "mean_z": float(subset["z"].mean()),
                "mean_innovation": float(subset["innovation"].mean()),
            }
        )

    frame["state_decile"] = pd.qcut(
        frame["state_before"],
        q=10,
        duplicates="drop",
    )
    if frame["state_decile"].isna().any():
        raise ValueError("shrinkage_summary cannot form state deciles")
    decile_records: list[dict[str, object]] = []
    for decile, subset in frame.groupby(
        "state_decile",
        observed=True,
        sort=True,
    ):
        innovation_sd = _sample_sd(subset["innovation"])
        decile_records.append(
            {
                "state_decile": decile,
                "n": len(subset),
                "mean_state": float(subset["state_before"].mean()),
                "mean_innovation": float(subset["innovation"].mean()),
                "innovation_se": _standard_error(innovation_sd, len(subset)),
            }
        )
    deciles = pd.DataFrame.from_records(decile_records)
    deciles["extreme_decile"] = False
    if not deciles.empty:
        deciles.loc[[deciles.index[0], deciles.index[-1]], "extreme_decile"] = True
    deciles["toward_zero"] = (
        (deciles["mean_state"] < 0) & (deciles["mean_innovation"] > 0)
    ) | ((deciles["mean_state"] > 0) & (deciles["mean_innovation"] < 0))
    deciles["significant"] = (
        deciles["mean_innovation"].abs() > 2.0 * deciles["innovation_se"]
    )
    deciles["under_shrinkage_flag"] = (
        deciles["extreme_decile"] & deciles["toward_zero"] & deciles["significant"]
    )
    return pd.DataFrame.from_records(tier_records), deciles


def _calibration_record(
    subset: pd.DataFrame,
    **labels: object,
) -> dict[str, object]:
    return {
        **labels,
        "n": len(subset),
        "mean_z": (float(subset["z"].mean()) if not subset.empty else float("nan")),
        "sd_z": _sample_sd(subset["z"]),
        **_coverage_fields(subset),
    }


def _regression_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    n = len(frame)
    undefined: dict[str, float | int] = {
        "n": n,
        "slope": float("nan"),
        "intercept": float("nan"),
        "correlation": float("nan"),
        "p_value": float("nan"),
        "slope_se": float("nan"),
    }
    if n < 2 or frame["state_before"].nunique() < 2:
        return undefined
    result = stats.linregress(frame["state_before"], frame["innovation"])
    return {
        "n": n,
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "correlation": float(result.rvalue),
        "p_value": float(result.pvalue),
        "slope_se": float(result.stderr),
    }


def offseason_summary(
    trace: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare overlapping early-season windows with later returner games."""
    caller = "offseason_summary"
    _validate_trace(
        trace,
        caller=caller,
        numeric=(
            "season_game_number",
            "state_before",
            "innovation",
            "z",
            "df",
            "predictive_var",
        ),
        other=("returning_player",),
    )
    _validate_predictive_inputs(trace, caller=caller)
    if (trace["season_game_number"] < 1).any():
        raise ValueError(f"{caller} season_game_number must be positive")
    _validate_boolean(trace, "returning_player", caller=caller)

    returning = trace.loc[trace["returning_player"]]
    records: list[dict[str, object]] = []
    for window in (5, 10, 20):
        early = returning.loc[returning["season_game_number"] <= window]
        rest = returning.loc[returning["season_game_number"] > window]
        records.append(_calibration_record(early, window=window, period="early"))
        records.append(_calibration_record(rest, window=window, period="rest"))
    relationship = _regression_summary(
        returning.loc[returning["season_game_number"] <= 20]
    )
    return pd.DataFrame.from_records(records), relationship


def group_calibration(
    trace: pd.DataFrame,
    group_column: str,
    min_n: int = 30,
) -> pd.DataFrame:
    """Report calibration and actionable flags for an arbitrary trace slice."""
    if (
        isinstance(min_n, bool)
        or not isinstance(min_n, Integral)
        or min_n < 2
    ):
        raise ValueError("group_calibration min_n must be an integer at least 2")
    caller = "group_calibration"
    _validate_trace(
        trace,
        caller=caller,
        numeric=("z", "innovation", "df", "predictive_var"),
        other=(group_column,),
    )
    _validate_predictive_inputs(trace, caller=caller)
    numeric_groups = trace[group_column].map(
        lambda value: isinstance(value, (float, np.floating)) and not np.isfinite(value)
    )
    if numeric_groups.any():
        raise ValueError(f"group_calibration column {group_column!r} must be finite")

    records: list[dict[str, object]] = []
    for group, subset in trace.groupby(group_column, observed=True, sort=False):
        sd_z = _sample_sd(subset["z"])
        records.append(
            {
                group_column: group,
                "n": len(subset),
                "mean_z": float(subset["z"].mean()),
                "sd_z": sd_z,
                **_coverage_fields(subset),
            }
        )
    summary = pd.DataFrame.from_records(records)
    summary["mean_z_se"] = summary["sd_z"] / np.sqrt(summary["n"])
    summary["sd_z_se"] = summary["sd_z"] / np.sqrt(2.0 * (summary["n"] - 1))
    summary["mean_flag"] = summary["mean_z"].abs() > 2.0 * summary["mean_z_se"]
    summary["sd_flag"] = (summary["sd_z"] - 1.0).abs() > 2.0 * summary["sd_z_se"]
    summary["low_sample"] = summary["n"] < min_n
    summary["actionable_flag"] = ~summary["low_sample"] & (
        summary["mean_flag"] | summary["sd_flag"]
    )
    return summary


def _gain_frame(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    records = _group_records(
        frame,
        group_column,
        "kalman_gain",
        "mean_gain",
    )
    result = pd.DataFrame.from_records(records)
    return result.rename(
        columns={
            "mean_gain_sd": "gain_sd",
            "mean_gain_se": "gain_se",
        }
    )


def gain_summary(
    trace: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize Kalman gain and flag unexplained non-boundary jumps."""
    caller = "gain_summary"
    _validate_trace(
        trace,
        caller=caller,
        numeric=("games_in_model_before", "kalman_gain"),
        other=("season_boundary",),
    )
    if ((trace["kalman_gain"] < 0) | (trace["kalman_gain"] > 1)).any():
        raise ValueError(f"{caller} kalman_gain must be in [0, 1]")
    _validate_boolean(trace, "season_boundary", caller=caller)

    frame = trace.copy()
    frame["games_tier"] = games_played_tier(frame["games_in_model_before"])
    by_tier = _gain_frame(frame, "games_tier")

    frame["games_quantile"] = pd.qcut(
        frame["games_in_model_before"],
        q=20,
        labels=False,
        duplicates="drop",
    )
    if frame["games_quantile"].isna().any():
        raise ValueError("gain_summary cannot form games quantiles")
    fine_records: list[dict[str, object]] = []
    for quantile, subset in frame.groupby(
        "games_quantile",
        observed=True,
        sort=True,
    ):
        non_boundary = subset.loc[~subset["season_boundary"]]
        gain_sd = _sample_sd(subset["kalman_gain"])
        non_boundary_sd = _sample_sd(non_boundary["kalman_gain"])
        fine_records.append(
            {
                "games_quantile": int(quantile),
                "n": len(subset),
                "games_min": int(subset["games_in_model_before"].min()),
                "games_max": int(subset["games_in_model_before"].max()),
                "mean_gain": float(subset["kalman_gain"].mean()),
                "gain_sd": gain_sd,
                "gain_se": _standard_error(gain_sd, len(subset)),
                "season_boundary_n": int(subset["season_boundary"].sum()),
                "non_boundary_n": len(non_boundary),
                "non_boundary_mean_gain": (
                    float(non_boundary["kalman_gain"].mean())
                    if not non_boundary.empty
                    else float("nan")
                ),
                "non_boundary_gain_se": _standard_error(
                    non_boundary_sd,
                    len(non_boundary),
                ),
            }
        )
    by_games = pd.DataFrame.from_records(fine_records)
    by_games["gain_jump"] = by_games["non_boundary_mean_gain"].diff()
    previous_se = by_games["non_boundary_gain_se"].shift()
    by_games["gain_jump_se"] = np.sqrt(
        by_games["non_boundary_gain_se"] ** 2 + previous_se**2
    )
    by_games["unexplained_jump_flag"] = (
        by_games["gain_jump"] > 2.0 * by_games["gain_jump_se"]
    )
    by_boundary = _gain_frame(frame, "season_boundary")
    return by_tier, by_games, by_boundary


def _finite_statistic(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
) -> float:
    result = statistic(values)
    if not np.isfinite(result):
        raise ValueError("posterior_draw_summary produced a non-finite statistic")
    return float(result)


def _student_t_row_crps(
    points: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    dfs: np.ndarray,
) -> np.ndarray:
    """Deterministic Student-t CRPS via 2,048-node quantile integration.

    CRPS equals twice the integral of pinball loss over quantile levels.
    Gauss-Legendre nodes avoid the infinite Student-t endpoint quantiles while
    providing deterministic accuracy without pretending synthetic values are
    posterior draws.
    """
    quantiles = means[:, None] + scales[:, None] * stats.t.ppf(
        _CRPS_PROBABILITIES[None, :],
        dfs[:, None],
    )
    errors = points[:, None] - quantiles
    losses = np.where(
        errors >= 0,
        _CRPS_PROBABILITIES[None, :] * errors,
        (_CRPS_PROBABILITIES[None, :] - 1.0) * errors,
    )
    return 2.0 * (losses @ _CRPS_WEIGHTS)


def posterior_draw_summary(
    trace: pd.DataFrame,
    draws: np.ndarray | None,
) -> dict[str, object]:
    """Compare posterior draws with Gaussian and analytic Student-t models."""
    if draws is None:
        return {
            "status": "skipped",
            "reason": "posterior draws unavailable",
        }
    caller = "posterior_draw_summary"
    _validate_trace(
        trace,
        caller=caller,
        numeric=(
            "points",
            "predictive_mean",
            "predictive_var",
            "z",
            "df",
        ),
    )
    if (trace["predictive_var"] <= 0).any():
        raise ValueError(f"{caller} predictive_var must be positive")
    if (trace["df"] <= 1).any():
        raise ValueError(f"{caller} df must exceed 1 for finite CRPS")
    samples = np.asarray(draws, dtype=float)
    if (
        samples.ndim != 2
        or samples.shape[0] != len(trace)
        or samples.shape[1] == 0
        or not np.isfinite(samples).all()
    ):
        raise ValueError(
            "draws must be a finite two-dimensional array shaped (len(trace), n_draws)"
        )

    points = trace["points"].to_numpy(dtype=float)
    z = trace["z"].to_numpy(dtype=float)
    sigma = np.sqrt(trace["predictive_var"].to_numpy(dtype=float))
    dfs = trace["df"].to_numpy(dtype=float)
    expected_z = (
        points - trace["predictive_mean"].to_numpy(dtype=float)
    ) / sigma
    if not np.allclose(z, expected_z):
        raise ValueError(
            f"{caller} column 'z' must match points and predictive moments"
        )
    gaussian_row_crps = sigma * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    draw_pit = probability_integral_transform(samples, points)
    gaussian_pit = stats.norm.cdf(z)
    student_t_pit = stats.t.cdf(z, dfs)
    draw_crps = crps(samples, points)
    gaussian_crps = _finite_statistic(gaussian_row_crps, np.mean)
    student_t_row_crps = _student_t_row_crps(
        points,
        trace["predictive_mean"].to_numpy(dtype=float),
        sigma,
        dfs,
    )
    student_t_crps = _finite_statistic(student_t_row_crps, np.mean)
    draw_pit_mean = _finite_statistic(draw_pit, np.mean)
    gaussian_pit_mean = _finite_statistic(gaussian_pit, np.mean)
    student_t_pit_mean = _finite_statistic(student_t_pit, np.mean)
    draw_pit_sd = _sample_sd(draw_pit)
    gaussian_pit_sd = _sample_sd(gaussian_pit)
    student_t_pit_sd = _sample_sd(student_t_pit)
    return {
        "n": len(trace),
        "n_draws": samples.shape[1],
        "draw_crps": draw_crps,
        "gaussian_crps": gaussian_crps,
        "crps_difference": draw_crps - gaussian_crps,
        "student_t_crps": student_t_crps,
        "draw_vs_student_t_crps_difference": draw_crps - student_t_crps,
        "draw_pit_mean": draw_pit_mean,
        "draw_pit_sd": draw_pit_sd,
        "gaussian_pit_mean": gaussian_pit_mean,
        "gaussian_pit_sd": gaussian_pit_sd,
        "student_t_pit_mean": student_t_pit_mean,
        "student_t_pit_sd": student_t_pit_sd,
        "pit_mean_difference": draw_pit_mean - gaussian_pit_mean,
        "pit_sd_difference": draw_pit_sd - gaussian_pit_sd,
        "draw_vs_student_t_pit_mean_difference": (
            draw_pit_mean - student_t_pit_mean
        ),
        "draw_vs_student_t_pit_sd_difference": draw_pit_sd - student_t_pit_sd,
    }
