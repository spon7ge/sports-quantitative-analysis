"""Leakage-safe baseline strikeout predictors for chronological folds."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.mlb.config import MlbConfig
from src.mlb.schemas import PMF_COLUMNS

# Poisson PMF fallback ONLY inside baselines.py when Agent 2
# negative_binomial_pmf is not importable.
_POISSON_FALLBACK_COMMENT = True


def _import_nb_pmf() -> Callable | None:
    for module_name in (
        "src.mlb.models",
        "src.mlb.models.pmf",
        "src.mlb.models.strikeouts",
        "src.mlb.models.nb",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fn = getattr(module, "negative_binomial_pmf", None)
        if callable(fn):
            return fn
    return None


def _moment_alpha(mean: float, variance: float) -> float:
    if mean <= 0:
        return 1e-6
    alpha = (variance - mean) / (mean**2)
    return float(max(alpha, 1e-6))


def _poisson_pmf(mu: np.ndarray, k_max: int) -> np.ndarray:
    """Poisson PMF fallback ONLY inside baselines.py (see module comment)."""
    from scipy.special import gammaln

    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    mu = np.clip(mu, 1e-8, None)
    ks = np.arange(k_max + 1, dtype=float)
    log_p = ks[None, :] * np.log(mu)[:, None] - mu[:, None] - gammaln(ks + 1)[None, :]
    pmf = np.exp(log_p)
    pmf = np.clip(pmf, 0.0, None)
    mass = pmf.sum(axis=1, keepdims=True)
    tail = np.clip(1.0 - mass, 0.0, 1.0)
    out = np.concatenate([pmf, tail], axis=1)
    denom = out.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return out / denom


def _nb_or_poisson_pmf(
    mu: np.ndarray,
    alpha: float | np.ndarray,
    config: MlbConfig,
) -> np.ndarray:
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    nb_pmf = _import_nb_pmf()
    if nb_pmf is not None:
        try:
            return np.asarray(
                nb_pmf(mu, alpha, config.k_max, config.tail_mass_threshold),
                dtype=float,
            )
        except Exception:
            pass
    # Poisson PMF fallback ONLY inside baselines.py when Agent 2 NB PMF is unavailable.
    assert _POISSON_FALLBACK_COMMENT
    return _poisson_pmf(mu, config.k_max)


def _line_name(line: float) -> str:
    text = str(line).replace(".", "_")
    return text


def _p_over_under(
    pmf: np.ndarray, line: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_max = pmf.shape[1] - 2
    support = np.arange(k_max + 1, dtype=float)
    support = np.concatenate([support, np.array([k_max + 1], dtype=float)])
    over = (support > line).astype(float)
    under = (support < line).astype(float)
    push = (support == line).astype(float)
    p_over = pmf @ over
    p_under = pmf @ under
    p_push = pmf @ push
    return p_over, p_under, p_push


def _interval(pmf: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    cdf = np.cumsum(pmf, axis=1)
    lower_q = (1.0 - mass) / 2.0
    upper_q = 1.0 - lower_q
    k_max = pmf.shape[1] - 2
    support = np.arange(k_max + 1)
    support = np.concatenate([support, np.array([k_max + 1])])
    lower = support[np.argmax(cdf >= lower_q, axis=1)]
    upper = support[np.argmax(cdf >= upper_q, axis=1)]
    return lower.astype(float), upper.astype(float)


def prediction_frame_from_mu(
    test: pd.DataFrame,
    mu: np.ndarray,
    alpha: float | np.ndarray,
    config: MlbConfig,
    *,
    model_version: str,
) -> pd.DataFrame:
    """Build a prediction-like frame with expected_k and a coherent PMF."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    if len(mu) != len(test):
        raise ValueError("mu length must match test rows")
    mu = np.clip(mu, 1e-8, None)
    pmf = _nb_or_poisson_pmf(mu, alpha, config)
    if pmf.shape[0] != len(test):
        raise ValueError("PMF row count must match test rows")
    alpha_arr = np.full(len(mu), float(np.mean(np.atleast_1d(alpha))), dtype=float)
    if np.ndim(alpha) > 0 and np.size(alpha) == len(mu):
        alpha_arr = np.asarray(alpha, dtype=float)
    variance = mu + alpha_arr * mu**2
    lower, upper = _interval(pmf, float(config.prediction_interval))
    frame = pd.DataFrame(index=test.index)
    for column in (
        "pitcher_id",
        "game_pk",
        "prediction_cutoff_utc",
        "game_date",
        "season",
    ):
        if column in test.columns:
            frame[column] = test[column].values
    if "forecast_horizon_hours" in test.columns:
        frame["forecast_horizon_hours"] = test["forecast_horizon_hours"].values
    else:
        frame["forecast_horizon_hours"] = config.forecast_horizon_hours
    for name, default in (
        ("expected_bf", "expected_bf_oof"),
        ("expected_pitches", "expected_pitches_oof"),
        ("expected_outs", "expected_outs_oof"),
    ):
        if default in test.columns:
            frame[name] = pd.to_numeric(test[default], errors="coerce").values
        elif name in test.columns:
            frame[name] = pd.to_numeric(test[name], errors="coerce").values
        else:
            frame[name] = np.nan
    frame["expected_innings"] = frame["expected_outs"] / 3.0
    frame["expected_k"] = mu
    frame["variance_k"] = variance
    frame["pi_lower"] = lower
    frame["pi_upper"] = upper
    for i, name in enumerate(PMF_COLUMNS):
        frame[name] = pmf[:, i]
    for line in config.lines:
        p_over, p_under, _p_push = _p_over_under(pmf, float(line))
        key = _line_name(float(line))
        frame[f"p_over_{key}"] = p_over
        frame[f"p_under_{key}"] = p_under
    frame["model_version"] = model_version
    frame["feature_version"] = config.feature_set_version
    frame["workload_model_version"] = config.workload_model_version
    if "source_snapshot_ids_json" in test.columns:
        frame["source_snapshot_ids_json"] = test["source_snapshot_ids_json"].values
    else:
        frame["source_snapshot_ids_json"] = "[]"
    frame["missing_data_flags_json"] = "[]"
    frame["restriction_flags_json"] = "[]"
    frame["market_p_over"] = np.nan
    frame["market_p_under"] = np.nan
    frame["market_disagreement"] = np.nan
    if "strikeouts" in test.columns:
        frame["strikeouts"] = test["strikeouts"].values
    return frame.reset_index(drop=True)


def _rate(
    successes: pd.Series | np.ndarray, trials: pd.Series | np.ndarray | None
) -> float:
    if trials is None:
        return float("nan")
    s = pd.to_numeric(pd.Series(successes), errors="coerce")
    t = pd.to_numeric(pd.Series(trials), errors="coerce")
    denom = float(t.sum(skipna=True))
    if denom <= 0:
        return float("nan")
    return float(s.sum(skipna=True) / denom)


def _combined_history(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    pieces = [frame for frame in (train, test) if frame is not None and len(frame)]
    if not pieces:
        return pd.DataFrame()
    history = pd.concat(pieces, ignore_index=True, sort=False)
    if {"pitcher_id", "game_pk"}.issubset(history.columns):
        history = history.drop_duplicates(["pitcher_id", "game_pk"], keep="first")
    sort_cols = [
        c
        for c in ("pitcher_id", "game_date", "scheduled_start_utc", "game_pk")
        if c in history.columns
    ]
    if sort_cols:
        history = history.sort_values(sort_cols, kind="mergesort")
    return history.reset_index(drop=True)


def _prior_expanding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    history = _combined_history(train, test)
    if history.empty:
        out = test.copy()
        for column in columns:
            out[f"_prior_sum_{column}"] = np.nan
        out["_prior_n"] = 0
        return out
    present = [c for c in columns if c in history.columns]
    pitcher_ids = history["pitcher_id"]
    for column in present:
        numeric = pd.to_numeric(history[column], errors="coerce")
        history[f"_prior_sum_{column}"] = numeric.groupby(
            pitcher_ids, sort=False
        ).transform(lambda s: s.cumsum().shift(1))
    history["_prior_n"] = pitcher_ids.groupby(pitcher_ids, sort=False).cumcount()
    keys = [c for c in ("pitcher_id", "game_pk") if c in history.columns]
    extra = [f"_prior_sum_{c}" for c in present] + ["_prior_n"]
    mapped = test.merge(
        history[keys + extra].drop_duplicates(keys),
        on=keys,
        how="left",
    )
    return mapped


def _col_or_default(
    frame: pd.DataFrame, name: str, default: float | np.ndarray
) -> np.ndarray:
    if name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        fill = default if np.isscalar(default) else np.asarray(default, dtype=float)
        return np.where(np.isfinite(values), values, fill)
    if np.isscalar(default):
        return np.full(len(frame), float(default), dtype=float)
    return np.asarray(default, dtype=float)


def _has_quote_columns(test: pd.DataFrame) -> bool:
    return {"over_price", "under_price", "line"}.issubset(test.columns) and test[
        "over_price"
    ].notna().any()


def _pitcher_train_means(
    train: pd.DataFrame, test: pd.DataFrame, column: str, league: float
) -> np.ndarray:
    if column not in train.columns or "pitcher_id" not in train.columns:
        return np.full(len(test), league, dtype=float)
    means = (
        train.assign(_val=pd.to_numeric(train[column], errors="coerce"))
        .groupby("pitcher_id")["_val"]
        .mean()
    )
    mapped = (
        test["pitcher_id"].map(means)
        if "pitcher_id" in test.columns
        else pd.Series(np.nan, index=test.index)
    )
    return mapped.fillna(league).to_numpy(dtype=float)


def _window_kbf(
    history: pd.DataFrame,
    pitcher_id: int,
    game_date: str,
    season: int | None,
    *,
    days: int | None = None,
    prior_seasons: int | None = None,
) -> tuple[float, float]:
    if history.empty or "pitcher_id" not in history.columns:
        return float("nan"), 0.0
    rows = history.loc[history["pitcher_id"] == pitcher_id]
    if rows.empty:
        return float("nan"), 0.0
    dates = pd.to_datetime(rows["game_date"], errors="coerce")
    asof = pd.to_datetime(game_date)
    mask = dates < asof
    if days is not None:
        mask &= dates >= asof - pd.Timedelta(days=int(days))
    if prior_seasons is not None and "season" in rows.columns and season is not None:
        mask &= rows["season"].between(
            int(season) - int(prior_seasons), int(season) - 1
        )
    picked = rows.loc[mask]
    if (
        picked.empty
        or "strikeouts" not in picked.columns
        or "batters_faced" not in picked.columns
    ):
        return float("nan"), 0.0
    bf = float(pd.to_numeric(picked["batters_faced"], errors="coerce").sum())
    k = float(pd.to_numeric(picked["strikeouts"], errors="coerce").sum())
    if bf <= 0:
        return float("nan"), 0.0
    return k / bf, bf


def _opp_k_rate(
    train: pd.DataFrame, test: pd.DataFrame, league_kbf: float
) -> np.ndarray:
    if (
        "opp_k_rate_vs_hand_shrunk" in test.columns
        and test["opp_k_rate_vs_hand_shrunk"].notna().any()
    ):
        return _col_or_default(test, "opp_k_rate_vs_hand_shrunk", league_kbf)
    if (
        "opponent_team_id" not in train.columns
        or "opponent_team_id" not in test.columns
    ):
        return np.full(len(test), league_kbf, dtype=float)
    tmp = train.copy()
    tmp["_k"] = pd.to_numeric(tmp.get("strikeouts"), errors="coerce")
    tmp["_bf"] = pd.to_numeric(tmp.get("batters_faced"), errors="coerce")
    grouped = tmp.groupby("opponent_team_id").agg(k=("_k", "sum"), bf=("_bf", "sum"))
    grouped["rate"] = np.where(grouped["bf"] > 0, grouped["k"] / grouped["bf"], np.nan)
    mapped = test["opponent_team_id"].map(grouped["rate"])
    return mapped.fillna(league_kbf).to_numpy(dtype=float)


def _market_mu(
    test: pd.DataFrame,
    alpha: float,
    config: MlbConfig,
    league_mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    from src.mlb.markets.odds import no_vig
    from src.mlb.markets.quotes import implied_over_price, implied_under_price

    mu = np.full(len(test), np.nan, dtype=float)
    keep = np.zeros(len(test), dtype=bool)
    for i, row in enumerate(test.itertuples(index=False)):
        over_price = getattr(row, "over_price", np.nan)
        under_price = getattr(row, "under_price", np.nan)
        line = getattr(row, "line", np.nan)
        if (
            not np.isfinite(over_price)
            or not np.isfinite(under_price)
            or not np.isfinite(line)
        ):
            continue
        fmt = getattr(row, "price_format", "american")
        try:
            p_over_raw = implied_over_price(float(over_price), fmt)
            p_under_raw = implied_under_price(float(under_price), fmt)
            p_over, _p_under = no_vig(p_over_raw, p_under_raw)
        except (TypeError, ValueError):
            continue
        keep[i] = True

        def _objective(
            candidate: float, p_target: float = float(p_over), ln: float = float(line)
        ) -> float:
            pmf = _nb_or_poisson_pmf(np.array([candidate]), alpha, config)
            est, _, _ = _p_over_under(pmf, ln)
            return float(est[0]) - p_target

        lo, hi = 0.05, 25.0
        try:
            f_lo, f_hi = _objective(lo), _objective(hi)
            if np.sign(f_lo) == np.sign(f_hi):
                mu[i] = league_mu
            else:
                from scipy.optimize import brentq

                mu[i] = float(brentq(_objective, lo, hi, maxiter=64))
        except Exception:
            mu[i] = league_mu
    return mu, keep


def fit_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: MlbConfig,
) -> dict[str, pd.DataFrame]:
    """Fit leakage-safe baselines; each value is a test-row prediction frame."""
    if test.empty:
        empty = {
            name: test.copy()
            for name in (
                "league_nb",
                "rolling_k",
                "k9_workload",
                "shrunk_kbf",
                "pitcher_opp",
                "marcel",
            )
        }
        return empty

    train_k = (
        pd.to_numeric(train["strikeouts"], errors="coerce")
        if "strikeouts" in train.columns
        else pd.Series(dtype=float)
    )
    league_mu = float(train_k.mean()) if train_k.notna().any() else 5.0
    league_var = float(train_k.var(ddof=1)) if train_k.notna().sum() > 1 else league_mu
    alpha = _moment_alpha(league_mu, league_var)
    league_kbf = _rate(train.get("strikeouts"), train.get("batters_faced"))
    if not np.isfinite(league_kbf):
        league_kbf = float(league_mu / 24.0)
    league_bf = (
        float(pd.to_numeric(train["batters_faced"], errors="coerce").mean())
        if "batters_faced" in train.columns
        else 24.0
    )
    if not np.isfinite(league_bf):
        league_bf = 24.0
    league_outs = (
        float(pd.to_numeric(train["outs"], errors="coerce").mean())
        if "outs" in train.columns
        else 18.0
    )
    if not np.isfinite(league_outs):
        league_outs = 18.0
    strength = float(config.pitcher_k_prior_strength)

    priors = _prior_expanding(train, test, ("strikeouts", "batters_faced", "outs"))
    prior_k = pd.to_numeric(
        priors.get("_prior_sum_strikeouts"), errors="coerce"
    ).to_numpy(dtype=float)
    prior_bf = pd.to_numeric(
        priors.get("_prior_sum_batters_faced"), errors="coerce"
    ).to_numpy(dtype=float)
    prior_outs = pd.to_numeric(priors.get("_prior_sum_outs"), errors="coerce").to_numpy(
        dtype=float
    )
    prior_n = (
        pd.to_numeric(priors.get("_prior_n"), errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
    )

    rolling = np.where(prior_n > 0, prior_k / np.clip(prior_n, 1.0, None), league_mu)
    rolling = np.where(np.isfinite(rolling), rolling, league_mu)

    k9 = np.where(prior_outs > 0, (prior_k / prior_outs) * 27.0, np.nan)
    outs_hat = _col_or_default(
        test,
        "expected_outs_oof",
        _pitcher_train_means(train, test, "outs", league_outs),
    )
    k9_mu = np.where(np.isfinite(k9), k9 * outs_hat / 27.0, league_mu)
    k9_mu = np.where(np.isfinite(k9_mu), k9_mu, league_mu)

    if "k_bf_shrunk_365" in test.columns and test["k_bf_shrunk_365"].notna().any():
        shrunk_rate = _col_or_default(test, "k_bf_shrunk_365", league_kbf)
    else:
        raw_rate = np.where(prior_bf > 0, prior_k / prior_bf, league_kbf)
        shrunk_rate = (prior_bf * raw_rate + strength * league_kbf) / (
            prior_bf + strength
        )
    bf_hat = _col_or_default(
        test,
        "expected_bf_oof",
        _pitcher_train_means(train, test, "batters_faced", league_bf),
    )
    shrunk_mu = np.clip(shrunk_rate * bf_hat, 1e-8, None)

    opp_rate = _opp_k_rate(train, test, league_kbf)
    pitcher_opp_mu = 0.5 * (shrunk_rate + opp_rate) * bf_hat
    pitcher_opp_mu = np.where(np.isfinite(pitcher_opp_mu), pitcher_opp_mu, league_mu)

    if all(
        c in test.columns
        for c in ("k_bf_shrunk_prior2", "k_bf_shrunk_365", "k_bf_shrunk_60")
    ):
        r2 = _col_or_default(test, "k_bf_shrunk_prior2", league_kbf)
        r365 = _col_or_default(test, "k_bf_shrunk_365", league_kbf)
        r60 = _col_or_default(test, "k_bf_shrunk_60", league_kbf)
        n_eff = _col_or_default(test, "n_eff_k_bf_365", 0.0)
        raw = (5.0 * r2 + 4.0 * r365 + 3.0 * r60) / 12.0
        marcel_rate = (n_eff * raw + strength * league_kbf) / (n_eff + strength)
    else:
        history = _combined_history(train, test)
        marcel_rate = np.full(len(test), league_kbf, dtype=float)
        for i, row in enumerate(test.itertuples(index=False)):
            pitcher_id = int(getattr(row, "pitcher_id"))
            game_date = str(getattr(row, "game_date"))
            season = (
                int(getattr(row, "season"))
                if hasattr(row, "season") and pd.notna(getattr(row, "season"))
                else None
            )
            r2, bf2 = _window_kbf(
                history, pitcher_id, game_date, season, prior_seasons=2
            )
            r365, bf365 = _window_kbf(history, pitcher_id, game_date, season, days=365)
            r60, bf60 = _window_kbf(history, pitcher_id, game_date, season, days=60)
            r2 = league_kbf if not np.isfinite(r2) else r2
            r365 = league_kbf if not np.isfinite(r365) else r365
            r60 = league_kbf if not np.isfinite(r60) else r60
            raw = (5.0 * r2 + 4.0 * r365 + 3.0 * r60) / 12.0
            n_eff = (bf2 or 0.0) + (bf365 or 0.0) + (bf60 or 0.0)
            marcel_rate[i] = (n_eff * raw + strength * league_kbf) / (n_eff + strength)
    marcel_mu = np.clip(marcel_rate * bf_hat, 1e-8, None)

    out: dict[str, pd.DataFrame] = {
        "league_nb": prediction_frame_from_mu(
            test,
            np.full(len(test), league_mu),
            alpha,
            config,
            model_version="baseline_league_nb",
        ),
        "rolling_k": prediction_frame_from_mu(
            test, rolling, alpha, config, model_version="baseline_rolling_k"
        ),
        "k9_workload": prediction_frame_from_mu(
            test, k9_mu, alpha, config, model_version="baseline_k9_workload"
        ),
        "shrunk_kbf": prediction_frame_from_mu(
            test, shrunk_mu, alpha, config, model_version="baseline_shrunk_kbf"
        ),
        "pitcher_opp": prediction_frame_from_mu(
            test, pitcher_opp_mu, alpha, config, model_version="baseline_pitcher_opp"
        ),
        "marcel": prediction_frame_from_mu(
            test, marcel_mu, alpha, config, model_version="baseline_marcel"
        ),
    }

    if _has_quote_columns(test):
        market_mu, keep = _market_mu(test, alpha, config, league_mu)
        if keep.any():
            market_test = test.loc[keep].copy()
            out["market"] = prediction_frame_from_mu(
                market_test,
                market_mu[keep],
                alpha,
                config,
                model_version="baseline_market",
            )
    return out
