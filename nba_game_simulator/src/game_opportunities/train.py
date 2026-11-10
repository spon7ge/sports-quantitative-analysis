"""Fit fallback rates and build training or live rate rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from game_opportunities.checks import _number, _positive_int
from game_opportunities.constants import (
    _DEFAULTS,
    _IDS,
    _PLAYER_SPEC,
    _PROBABILITIES,
    _SHOTS,
    _TEAM_RATES,
    _TEAM_SPEC,
    MODEL_VERSION,
    OPPORTUNITY_RATES,
)
from game_opportunities.history import (
    _historical_rates,
    _paired_sums,
    _prepare_opportunity_logs,
    _prepare_slate,
    _team_games,
)
from game_opportunities.model import OpportunityModel, _model
from pbp_guard import possessions_for_slate
from possession_rates import possession_player_rates


def training_opportunities(
    logs,
    *,
    window=10,
    prior_minutes=100.0,
    prior_attempts=40.0,
    prior_possessions=500.0,
):
    """Historical rate rows using only games before the target date."""
    frame = _prepare_opportunity_logs(logs)
    teams = _team_games(frame)

    player = _historical_rates(
        frame,
        "player_id",
        _PLAYER_SPEC,
        window,
        prior_minutes,
        prior_attempts,
    )
    team = _historical_rates(
        teams,
        "team_id",
        _TEAM_SPEC,
        window,
        prior_possessions,
        prior_attempts,
    )

    result = player.merge(
        team[["game_id", "team_id", *_TEAM_RATES]],
        on=["game_id", "team_id"],
        validate="many_to_one",
    )

    return result.loc[
        :,
        [*_IDS, *OPPORTUNITY_RATES],
    ].reset_index(drop=True)


def fit_opportunities(logs, alpha=0.03, *, oreb_rate=0.25):
    """Fit production fallbacks from raw logs, pooling events and exposures.

    Pass raw player logs, not the rolling-rate output of training_opportunities.
    Team fallbacks count each team-game once, including partial player extracts.
    """
    alpha = _number(
        alpha,
        "alpha",
        nonnegative=True,
    )

    frame = _prepare_opportunity_logs(logs)
    teams = _team_games(frame)
    fallback = {}

    for data, spec in (
        (frame, _PLAYER_SPEC),
        (teams, _TEAM_SPEC),
    ):
        for name, (numerator, denominator) in spec.items():
            num, den = _paired_sums(
                data,
                numerator,
                denominator,
            )
            fallback[name] = num / den if den > 0 else _DEFAULTS[name]

    if teams["team_fta"].notna().sum() == 0:
        fallback["fta_per_poss"] = fallback["fga_per_poss"] * fallback["fta_per_fga"]

    trials = np.floor(teams["team_poss"].to_numpy(dtype=float) + 0.5)
    observed = teams["team_tov"].to_numpy(dtype=float)

    good = np.isfinite(trials) & np.isfinite(observed) & (trials > 0)

    n = trials[good]
    y = observed[good]
    p = fallback["tov_per_poss"]

    denominator = np.sum(n * np.maximum(n - 1, 0) * p * (1 - p))

    if denominator > 0:
        numerator = np.sum((y - n * p) ** 2 - n * p * (1 - p))
        tov_rho = float(
            np.clip(
                numerator / denominator,
                0,
                0.25,
            )
        )
    else:
        tov_rho = 0.0

    metadata = {
        "version": MODEL_VERSION,
        "player_games": len(frame),
        "team_games": len(teams),
        "team_fta_games": int(teams["team_fta"].notna().sum()),
        "alpha_source": ("caller supplied; not chronologically tuned"),
        "rate_estimator": ("sum(events) / sum(exposure)"),
        "tov_rho_source": ("pooled method of moments; not a holdout evaluation"),
        "oreb_rate_source": ("caller supplied field-goal rebound proxy; not fitted"),
    }

    ast_sum = 0.0
    fgm_sum = 0.0
    complete_keys = set(
        zip(
            teams.loc[teams.complete_roster, "game_id"],
            teams.loc[teams.complete_roster, "team_id"],
            strict=True,
        )
    )

    for key, rows in frame.groupby(["game_id", "team_id"], sort=False):
        if key in complete_keys and rows[["ast", "fgm"]].notna().all().all():
            if rows["ast"].sum() > rows["fgm"].sum():
                raise ValueError("complete team-game assists exceed field goals made")

            ast_sum += float(rows["ast"].sum())
            fgm_sum += float(rows["fgm"].sum())

    assist_rate = float(np.clip(ast_sum / fgm_sum, 0, 1)) if fgm_sum else 0.55
    metadata["assist_rate_source"] = (
        "complete-roster official AST / FGM; no PBP assist replay"
    )

    return OpportunityModel(
        **fallback,
        alpha=alpha,
        tov_rho=tov_rho,
        assist_rate=assist_rate,
        oreb_rate=oreb_rate,
        training_cutoff=(frame["game_date"].max().date().isoformat()),
        metadata=metadata,
    )


def live_opportunities(
    logs,
    slate,
    *,
    model=None,
    window=10,
    prior_minutes=100.0,
    prior_attempts=40.0,
    prior_possessions=500.0,
    possessions=None,
):
    """Upcoming rates through the latest completed observed appearance.

    ``possessions`` is a possession table. Only games strictly before the
    slate date are used, and a game whose shots, free throws, or turnovers
    do not match the official logs is left out.
    """
    prepared = _prepare_opportunity_logs(logs)
    upcoming = _prepare_slate(slate)
    _positive_int(window, "window")

    fallback_weights = {
        name: (
            prior_possessions
            if name in _TEAM_RATES
            else prior_attempts
            if name in _PROBABILITIES or name == "fta_per_fga"
            else prior_minutes
        )
        for name in OPPORTUNITY_RATES
    }

    for name, value in fallback_weights.items():
        _number(
            value,
            f"prior exposure for {name}",
            nonnegative=True,
        )

    parts = []

    for date, rows in upcoming.groupby(
        "game_date",
        sort=False,
    ):
        if model is not None:
            _model(model)

            if model.training_cutoff and date <= pd.Timestamp(model.training_cutoff):
                raise ValueError("live date must follow the model's training cutoff")

        prior = prepared.loc[prepared["game_date"] < date]
        teams = _team_games(prior)

        kept = [
            "player_id",
            "game_date",
            "team_id",
            *[column for column in ("available", "is_home") if column in rows],
        ]
        result = rows.loc[:, kept].copy()

        for data, entity, spec in (
            (prior, "player_id", _PLAYER_SPEC),
            (teams, "team_id", _TEAM_SPEC),
        ):
            for name, (numerator, denominator) in spec.items():
                numerator_sum, denominator_sum = _paired_sums(
                    data,
                    numerator,
                    denominator,
                )

                league = (
                    numerator_sum / denominator_sum
                    if denominator_sum > 0
                    else _DEFAULTS[name]
                )

                if model is not None:
                    league = float(getattr(model, name))

                rates = {}
                history_data = data

                if entity == "player_id":
                    observed = (data["minutes"] > 0) | (
                        data[list(_SHOTS)].fillna(0).sum(axis=1) > 0
                    )
                    history_data = data.loc[observed]

                for key, history in history_data.groupby(
                    entity,
                    sort=False,
                ):
                    num, den = _paired_sums(
                        history.tail(window),
                        numerator,
                        denominator,
                    )
                    strength = fallback_weights[name]

                    rates[key] = (
                        (num + league * strength) / (den + strength)
                        if den + strength > 0
                        else league
                    )

                result[name] = result[entity].map(rates).fillna(league)

        if possessions is not None:
            scoped = possessions_for_slate(possessions, date, prepared)
            result = _overlay_possession_rates(result, scoped)

        parts.append(result)

    return pd.concat(parts, ignore_index=True)


def _overlay_possession_rates(result: pd.DataFrame, possessions) -> pd.DataFrame:
    if not isinstance(possessions, pd.DataFrame) or possessions.empty:
        return result

    rates = possession_player_rates(possessions, result["player_id"].unique())
    if rates.empty:
        return result

    indexed = rates.assign(
        _key=pd.to_numeric(rates["player_id"], errors="coerce")
    ).set_index("_key")
    overlaid = result.copy()
    keys = pd.to_numeric(overlaid["player_id"], errors="coerce")
    for column in rates.columns:
        if column == "player_id":
            continue
        incoming = keys.map(indexed[column])
        if column in overlaid.columns:
            overlaid[column] = incoming.where(incoming.notna(), overlaid[column])
        else:
            overlaid[column] = incoming
    return overlaid
