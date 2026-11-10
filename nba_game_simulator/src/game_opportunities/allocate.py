"""Allocate one roster from Layer 1 possessions and Layer 2 minutes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from game_opportunities.checks import (
    _availability,
    _check_minutes,
    _dates,
    _float_vector,
    _frame,
    _key,
    _normalize_rows,
    _required,
    _rng,
    as_nonneg_int_vector,
)
from game_opportunities.constants import (
    _MAX_POOL,
    _PROBABILITIES,
    LIVE_TOV_PRIOR,
    OPPORTUNITY_RATES,
)
from game_opportunities.counts import (
    _beta_binomial,
    _conditional_makes,
    _shoot_pools,
    align_zone_makes,
    multinomial_rows,
    sample_negbin,
    split_turnovers,
)
from game_opportunities.model import _model


def allocate_opportunities(
    model,
    profile,
    minutes,
    possessions,
    rng,
    *,
    team_points=None,
    availability=None,
    game_minutes=None,
    max_score_resamples=50,
    segments=None,
    interval_boundaries=None,
    return_intervals=False,
    game_date=None,
):
    """Use existing Layer 1 possessions and Layer 2 minutes for one roster.

    Fractional possessions remain expected exposure. Only the turnover cap
    rounds them to a count. Makes are conditioned on fixed sampled attempts.
    Set max_score_resamples=0 to require the first sampled pools to work.

    With segments, events stay inside the active lineup. Optional interval
    boundaries align both teams without changing those schedules.
    """
    _model(model)
    _frame(profile, "profile")
    _required(profile, ["player_id"])
    _rng(rng)

    if (
        isinstance(max_score_resamples, (bool, np.bool_))
        or not isinstance(
            max_score_resamples,
            (int, np.integer),
        )
        or max_score_resamples < 0
    ):
        raise ValueError("max_score_resamples must be a nonnegative integer")

    ids = [_key(value) for value in profile["player_id"]]

    if len(set(ids)) != len(ids):
        raise ValueError("profile contains duplicate player IDs")

    if "team_id" in profile and len({_key(value) for value in profile["team_id"]}) != 1:
        raise ValueError("profile must describe one team")

    index = minutes.index if isinstance(minutes, pd.DataFrame) else None

    if isinstance(minutes, pd.DataFrame):
        if [_key(value) for value in minutes.columns] != ids:
            raise ValueError("minute columns must match profile player order")

        for value in (
            possessions,
            team_points,
            game_minutes,
            game_date,
        ):
            if isinstance(value, pd.Series) and not value.index.equals(index):
                raise ValueError(
                    "environment series and minute rows must have the same index"
                )

    played = np.asarray(minutes, dtype=float)

    if played.ndim != 2 or not played.shape[0] or played.shape[1] != len(profile):
        raise ValueError("minutes must be (n_simulations, n_players)")

    n, k = played.shape

    fixed_available = _availability(
        profile.get("available"),
        n,
        k,
    )
    available = (
        fixed_available if availability is None else _availability(availability, n, k)
    )

    if np.any(available & ~fixed_available):
        raise ValueError("scenario makes a roster-unavailable player available")

    _check_minutes(
        played,
        available,
        game_minutes,
        strict_duration=True,
    )

    owned = _float_vector(
        possessions,
        n,
        "possessions",
        nonnegative=True,
    )

    if (owned > _MAX_POOL).any():
        raise ValueError("possession exposure is outside the supported NBA range")

    dates = None

    if "game_date" in profile:
        dates = _dates(profile["game_date"])

        if dates.nunique() != 1:
            raise ValueError("profile must describe one game date")

    if game_date is not None:
        supplied_dates = _dates(pd.Series(np.broadcast_to(np.asarray(game_date), (n,))))

        if supplied_dates.nunique() != 1 or (
            dates is not None and not supplied_dates.eq(dates.iloc[0]).all()
        ):
            raise ValueError("game_date must describe the profile's one game date")

        dates = supplied_dates

    if model.training_cutoff:
        if dates is None:
            raise ValueError(
                "supply profile game_date or game_date to enforce the model cutoff"
            )

        if dates.le(pd.Timestamp(model.training_cutoff)).any():
            raise ValueError("game_date must follow the model's training cutoff")

    rates = {
        name: _filled_rate(
            profile,
            name,
            float(getattr(model, name)),
        )
        for name in OPPORTUNITY_RATES
    }
    _attach_zone_rates(profile, rates)
    _attach_live_share(profile, rates)

    if segments is not None:
        from game_opportunities.segments import _allocate_segment_opportunities

        return _allocate_segment_opportunities(
            model,
            ids,
            rates,
            played,
            owned,
            available,
            game_minutes,
            segments,
            interval_boundaries,
            team_points,
            rng,
            max_score_resamples,
            return_intervals,
            index,
        )

    if interval_boundaries is not None or return_intervals:
        raise ValueError("interval boundaries/output require Layer 3 segments")

    fga_rate = _shared_rate(
        rates["fga_per_poss"],
        "fga_per_poss",
    )
    tov_rate = _shared_rate(
        rates["tov_per_poss"],
        "tov_per_poss",
    )
    fta_rate = _shared_rate(
        rates["fta_per_poss"],
        "fta_per_poss",
    )

    shares = _weights(
        rates["fga_per_min"],
        played,
    )
    ft_shares = _weights(
        rates["fta_per_min"],
        played,
    )
    tov_shares = _weights(
        rates["tov_per_min"],
        played,
    )

    team_fga = sample_negbin(
        owned * fga_rate,
        model.alpha,
        rng,
    )
    team_fta = sample_negbin(
        owned * fta_rate,
        model.alpha,
        rng,
    )
    team_tov = _beta_binomial(
        _round_possessions(owned, n),
        tov_rate,
        model.tov_rho,
        rng,
    )

    target = _whole_score(team_points, n) if team_points is not None else None

    stats = _shoot_pools(
        team_fga,
        team_fta,
        shares,
        ft_shares,
        rates["p_3pa"],
        rates["p_3p"],
        rates["p_2p"],
        rates["ft_pct"],
        rng,
        zone_share=rates.get("zone_share"),
        zone_make=rates.get("zone_make"),
    )

    resamples = np.zeros(n, dtype=np.int64)

    if target is not None:
        for i in range(n):
            no_scoring_probability = (
                not np.any(rates["p_2p"] > 0)
                and not np.any(rates["p_3p"] > 0)
                and not np.any(rates["ft_pct"] > 0)
            )

            if target[i] > 0 and (owned[i] == 0 or no_scoring_probability):
                raise ValueError(
                    "a positive score has no possible scoring opportunities"
                )

            for attempt in range(max_score_resamples + 1):
                one_shot = {
                    name: stats[name][i : i + 1]
                    for name in (
                        "FGA",
                        "2PA",
                        "2PM",
                        "3PA",
                        "3PM",
                        "FGM",
                    )
                }

                try:
                    conditioned, made_ft = _conditional_makes(
                        one_shot,
                        stats["FTA"][i : i + 1],
                        rates["p_2p"],
                        rates["p_3p"],
                        rates["ft_pct"],
                        target[i : i + 1],
                        rng,
                    )

                except ValueError as exc:
                    if attempt == max_score_resamples:
                        raise ValueError(
                            f"row {i}: no feasible score after "
                            f"{attempt} pool redraws; {exc}"
                        ) from exc

                    team_fga[i] = sample_negbin(
                        owned[i] * fga_rate,
                        model.alpha,
                        rng,
                    )
                    team_fta[i] = sample_negbin(
                        owned[i] * fta_rate,
                        model.alpha,
                        rng,
                    )

                    redrawn = _shoot_pools(
                        team_fga[i : i + 1],
                        team_fta[i : i + 1],
                        shares[i : i + 1],
                        ft_shares[i : i + 1],
                        rates["p_3pa"],
                        rates["p_3p"],
                        rates["p_2p"],
                        rates["ft_pct"],
                        rng,
                        zone_share=rates.get("zone_share"),
                        zone_make=rates.get("zone_make"),
                    )

                    for name, values in redrawn.items():
                        stats[name][i] = values[0]

                    resamples[i] += 1
                    continue

                stats["2PM"][i] = conditioned["2PM"][0]
                stats["3PM"][i] = conditioned["3PM"][0]
                stats["FGM"][i] = conditioned["FGM"][0]
                stats["FTM"][i] = made_ft[0]
                stats["ZONE_FGM"][i] = align_zone_makes(
                    stats["ZONE_FGA"][i : i + 1],
                    stats["2PM"][i : i + 1],
                    stats["3PM"][i : i + 1],
                    rng,
                )[0]
                stats["PTS"][i] = (
                    2 * (stats["ZONE_FGM"][i, :, 0] + stats["ZONE_FGM"][i, :, 1])
                    + 3 * (stats["ZONE_FGM"][i, :, 2] + stats["ZONE_FGM"][i, :, 3])
                    + stats["FTM"][i]
                )
                break

    stats["TOV"] = multinomial_rows(
        team_tov,
        tov_shares,
        rng,
    )
    stats["LIVE_TOV"], stats["DEAD_TOV"] = split_turnovers(
        stats["TOV"],
        rates["p_live_tov"],
        rng,
    )
    stats["MISS_BY_ZONE"] = (stats["ZONE_FGA"] - stats["ZONE_FGM"]).sum(axis=1)

    return {
        **stats,
        "team_fga": team_fga,
        "team_fta": team_fta,
        "team_tov": team_tov,
        "shares": shares,
        "score_pool_resamples": resamples,
    }


def _filled_rate(profile, name, fallback):
    if name not in profile:
        return np.full(
            len(profile),
            fallback,
        )

    values = pd.to_numeric(
        profile[name],
        errors="coerce",
    ).to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    bad_probability = (
        values > 1 if name in _PROBABILITIES else np.zeros(values.shape, dtype=bool)
    )

    if ((values < 0) | bad_probability).any():
        raise ValueError(f"invalid {name}")

    return np.where(
        np.isfinite(values),
        values,
        fallback,
    )


def _shared_rate(values, name):
    if not np.allclose(
        values,
        values[0],
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError(f"teammates must share one {name}")

    return float(values[0])


def _weights(rate, minutes):
    return _normalize_rows(minutes * rate[None, :])


_ZONES = ("rim", "mid", "corner3", "atb3")


def _attach_zone_rates(profile, rates) -> None:
    """Four-zone shares from the possession table, when the profile has them."""

    share_columns = [f"{zone}_share" for zone in _ZONES]
    if any(column not in profile.columns for column in share_columns):
        return

    share = (
        profile[share_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    make_columns = [f"{zone}_fg_pct" for zone in _ZONES]
    if any(column not in profile.columns for column in make_columns):
        make = np.column_stack(
            [rates["p_2p"], rates["p_2p"], rates["p_3p"], rates["p_3p"]]
        )
    else:
        make = (
            profile[make_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

    missing = ~np.isfinite(share).all(axis=1)
    if missing.any():
        p3 = rates["p_3pa"]
        zeros = np.zeros(int(missing.sum()))
        share[missing] = np.column_stack([1 - p3[missing], zeros, zeros, p3[missing]])
    missing_make = ~np.isfinite(make).all(axis=1)
    if missing_make.any():
        make[missing_make] = np.column_stack(
            [
                rates["p_2p"][missing_make],
                rates["p_2p"][missing_make],
                rates["p_3p"][missing_make],
                rates["p_3p"][missing_make],
            ]
        )

    totals = share.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    rates["zone_share"] = share / totals
    rates["zone_make"] = np.clip(make, 0.0, 1.0)


def _attach_live_share(profile, rates) -> None:
    """Live-ball share of each player's turnovers. Missing values use the prior."""

    if "p_live_tov" not in profile.columns:
        rates["p_live_tov"] = np.full(len(profile), LIVE_TOV_PRIOR)
        return

    values = pd.to_numeric(profile["p_live_tov"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if np.any(finite & ((values < 0.0) | (values > 1.0))):
        raise ValueError("p_live_tov must be in [0, 1]")
    rates["p_live_tov"] = np.where(finite, values, LIVE_TOV_PRIOR)


def _whole_score(value, n):
    values = _float_vector(
        value,
        n,
        "team_points",
        nonnegative=True,
    )

    if (values > _MAX_POOL).any():
        raise ValueError("team score is outside the supported NBA range")

    return np.floor(values + 0.5).astype(np.int64)


def _round_possessions(value, n):
    values = _float_vector(
        value,
        n,
        "possessions",
        nonnegative=True,
    )

    if (values > _MAX_POOL).any():
        raise ValueError("possession count is outside the supported range")

    return np.floor(values + 0.5).astype(np.int64)


def _pool(value, n, name):
    result = as_nonneg_int_vector(
        value,
        n,
        name,
    )

    if (result > _MAX_POOL).any():
        raise ValueError(f"{name} exceeds the supported NBA pool size")

    return result
