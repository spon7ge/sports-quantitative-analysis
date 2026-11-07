"""On-court intervals and a shared timeline for both teams."""

from __future__ import annotations

import numpy as np
import pandas as pd
from game_opportunities.allocate import (
    _round_possessions,
    _shared_rate,
    _weights,
    _whole_score,
)
from game_opportunities.checks import (
    _availability,
    _dates,
    _float_vector,
    _frame,
    _key,
    _normalize_rows,
    _number,
    _required,
    _rng,
)
from game_opportunities.constants import _TOL, MODEL_VERSION
from game_opportunities.counts import (
    _assists,
    _beta_binomial,
    _beta_probability,
    _conditional_makes,
    multinomial_rows,
    sample_negbin,
)
from game_opportunities.model import _model


def _rotation_schedules(segments, n):
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError(
            "segments must contain one complete schedule or one per simulation"
        )

    if isinstance(segments[0], dict):
        return [segments] * n

    if len(segments) != n:
        raise ValueError("one Layer 3 schedule is required per simulation")

    return list(segments)


def _rotation_intervals(ids, schedule, cap, available, minutes):
    if not isinstance(schedule, (list, tuple)) or not schedule:
        raise ValueError("a Layer 3 schedule cannot be empty")

    overtime = (cap - 48) / 5

    if overtime < 0 or abs(overtime - round(overtime)) > _TOL or overtime > 5:
        raise ValueError(
            "segments require 48 minutes plus zero to five overtime periods"
        )

    n_periods = 4 + int(round(overtime))
    lookup = {player: index for index, player in enumerate(ids)}
    grouped = {period: [] for period in range(1, n_periods + 1)}

    for item in schedule:
        if not isinstance(item, dict):
            raise ValueError("each segment must be a dictionary")

        period = _number(item.get("period"), "segment period")

        if period != round(period) or int(period) not in grouped:
            raise ValueError("segment period is outside the supplied game")

        period = int(period)
        length = 12.0 if period <= 4 else 5.0
        start = _number(item.get("start_min"), "segment start")
        end = _number(item.get("end_min"), "segment end")

        if start < -_TOL or end > length + _TOL or end <= start:
            raise ValueError("segment has invalid duration or period bounds")

        players = item.get("players")

        if not isinstance(players, (list, tuple)) or len(players) != 5:
            raise ValueError("each segment must name exactly five players")

        keys = [_key(player) for player in players]

        if len(set(keys)) != 5 or any(player not in lookup for player in keys):
            raise ValueError("segment players must be distinct roster members")

        positions = np.array([lookup[player] for player in keys])

        if not available[positions].all():
            raise ValueError("an unavailable player appears in a segment")

        active = np.zeros(len(ids), dtype=bool)
        active[positions] = True
        grouped[period].append((max(0.0, start), min(length, end), active))

    intervals = []
    exposure = np.zeros(len(ids))

    for period, spans in grouped.items():
        length = 12.0 if period <= 4 else 5.0
        offset = 12 * (period - 1) if period <= 4 else 48 + 5 * (period - 5)
        cursor = 0.0

        for start, end, active in sorted(spans, key=lambda span: span[0]):
            if abs(start - cursor) > _TOL or end <= cursor:
                raise ValueError("rotation segments have a gap or overlap")

            exposure[active] += end - start
            intervals.append(
                {
                    "period": period,
                    "start": start + offset,
                    "end": end + offset,
                    "active": active,
                }
            )
            cursor = end

        if abs(cursor - length) > _TOL:
            raise ValueError("rotation segments do not cover every period")

    if not np.allclose(exposure, minutes, rtol=0, atol=_TOL):
        raise ValueError(
            "Layer 3 segment exposure must match the existing Layer 2 minutes"
        )

    return intervals


def _refine_intervals(intervals, boundaries, cap):
    own = np.array(
        [
            0.0,
            cap,
            *[phase["start"] for phase in intervals],
            *[phase["end"] for phase in intervals],
        ]
    )

    if boundaries is not None:
        extra = np.asarray(boundaries, dtype=float)

        if (
            extra.ndim != 1
            or not np.isfinite(extra).all()
            or np.any(extra < 0)
            or np.any(extra > cap)
        ):
            raise ValueError("interval boundaries must be finite times within the game")

        own = np.r_[own, extra]

    edges = np.unique(own)
    result = []
    pointer = 0

    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end - start <= 1e-10:
            continue

        midpoint = (start + end) / 2

        while pointer + 1 < len(intervals) and midpoint >= intervals[pointer]["end"]:
            pointer += 1

        base = intervals[pointer]

        if midpoint < base["start"] - _TOL or midpoint > base["end"] + _TOL:
            raise ValueError("interval boundary has no active lineup")

        result.append(
            {
                "period": base["period"],
                "start": float(start),
                "end": float(end),
                "active": base["active"],
            }
        )

    return result


def _hypergeom_split(counts, total, rng):
    """Place a fixed number of successes among fixed interval trial counts."""
    counts = np.asarray(counts, dtype=np.int64)
    remaining = int(total)
    population = int(counts.sum())

    if remaining < 0 or remaining > population:
        raise ValueError("successes exceed the supplied attempt/trial counts")

    result = np.zeros_like(counts)

    for index in range(len(counts) - 1):
        if remaining and counts[index]:
            result[index] = rng.hypergeometric(
                int(counts[index]),
                population - int(counts[index]),
                remaining,
            )
            remaining -= int(result[index])

        population -= int(counts[index])

    result[-1] = remaining
    return result


def _segment_shots(team_fga, team_fta, active, duration, rates, target, rng):
    fga_weights = duration[:, None] * active * rates["fga_per_min"][None, :]
    fta_weights = duration[:, None] * active * rates["fta_per_min"][None, :]
    fga = multinomial_rows(
        np.array([team_fga]),
        _normalize_rows(fga_weights.reshape(1, -1)),
        rng,
    )[0].reshape(active.shape)
    fta = multinomial_rows(
        np.array([team_fta]),
        _normalize_rows(fta_weights.reshape(1, -1)),
        rng,
    )[0].reshape(active.shape)
    three_pa = rng.binomial(fga, rates["p_3pa"][None, :])
    two_pa = fga - three_pa

    if target is None:
        two_pm = rng.binomial(two_pa, rates["p_2p"][None, :])
        three_pm = rng.binomial(three_pa, rates["p_3p"][None, :])
        ftm = rng.binomial(fta, rates["ft_pct"][None, :])
    else:
        shot = {
            "FGA": fga.sum(axis=0, keepdims=True),
            "2PA": two_pa.sum(axis=0, keepdims=True),
            "3PA": three_pa.sum(axis=0, keepdims=True),
        }
        conditional, made_ft = _conditional_makes(
            shot,
            fta.sum(axis=0, keepdims=True),
            rates["p_2p"],
            rates["p_3p"],
            rates["ft_pct"],
            np.array([target]),
            rng,
        )
        two_pm = np.zeros_like(fga)
        three_pm = np.zeros_like(fga)
        ftm = np.zeros_like(fga)

        for player in range(fga.shape[1]):
            two_pm[:, player] = _hypergeom_split(
                two_pa[:, player],
                conditional["2PM"][0, player],
                rng,
            )
            three_pm[:, player] = _hypergeom_split(
                three_pa[:, player],
                conditional["3PM"][0, player],
                rng,
            )
            ftm[:, player] = _hypergeom_split(
                fta[:, player],
                made_ft[0, player],
                rng,
            )

    return {
        "FGA": fga,
        "FGM": two_pm + three_pm,
        "2PA": two_pa,
        "2PM": two_pm,
        "3PA": three_pa,
        "3PM": three_pm,
        "FTA": fta,
        "FTM": ftm,
        "PTS": 2 * two_pm + 3 * three_pm + ftm,
    }


def _allocate_segment_opportunities(
    model,
    ids,
    rates,
    played,
    owned,
    available,
    game_minutes,
    segments,
    boundaries,
    team_points,
    rng,
    max_score_resamples,
    return_intervals,
    index,
):
    n, n_players = played.shape
    caps = (
        played.sum(axis=1) / 5
        if game_minutes is None
        else _float_vector(game_minutes, n, "game_minutes", nonnegative=True)
    )
    schedules = _rotation_schedules(segments, n)

    if boundaries is not None and (
        not isinstance(boundaries, (list, tuple)) or len(boundaries) != n
    ):
        raise ValueError(
            "interval_boundaries must contain one time vector per simulation"
        )

    intervals = [
        _refine_intervals(
            _rotation_intervals(
                ids,
                schedules[row],
                caps[row],
                available[row],
                played[row],
            ),
            None if boundaries is None else boundaries[row],
            caps[row],
        )
        for row in range(n)
    ]
    fga_rate = _shared_rate(rates["fga_per_poss"], "fga_per_poss")
    fta_rate = _shared_rate(rates["fta_per_poss"], "fta_per_poss")
    tov_rate = _shared_rate(rates["tov_per_poss"], "tov_per_poss")
    team_fga = sample_negbin(owned * fga_rate, model.alpha, rng)
    team_fta = sample_negbin(owned * fta_rate, model.alpha, rng)
    trials = _round_possessions(owned, n)
    team_tov = _beta_binomial(trials, tov_rate, model.tov_rho, rng)
    target = _whole_score(team_points, n) if team_points is not None else None
    names = (
        "FGA",
        "FGM",
        "2PA",
        "2PM",
        "3PA",
        "3PM",
        "FTA",
        "FTM",
        "PTS",
        "AST",
        "OREB",
        "TOV",
    )
    totals = {name: np.zeros((n, n_players), dtype=np.int64) for name in names}
    exposure = np.zeros((n, n_players))
    resamples = np.zeros(n, dtype=np.int64)
    details = []

    for row, phases in enumerate(intervals):
        active = np.array([phase["active"] for phase in phases])
        duration = np.array([phase["end"] - phase["start"] for phase in phases])
        time_share = duration / duration.sum()
        shot_support = np.sum(played[row] * rates["fga_per_min"])
        ft_support = np.sum(played[row] * rates["fta_per_min"])

        if owned[row] * fga_rate > 0 and shot_support == 0:
            raise ValueError("positive FGA pool has no on-court shot-rate support")

        if owned[row] * fta_rate > 0 and ft_support == 0:
            raise ValueError(
                "positive FTA pool has no on-court free-throw-rate support"
            )

        possible_fg = fga_rate > 0 and np.any(
            (played[row] > 0)
            & (rates["fga_per_min"] > 0)
            & (
                (rates["p_3pa"] * rates["p_3p"] + (1 - rates["p_3pa"]) * rates["p_2p"])
                > 0
            )
        )
        possible_ft = fta_rate > 0 and np.any(
            (played[row] > 0) & (rates["fta_per_min"] > 0) & (rates["ft_pct"] > 0)
        )

        if target is not None and target[row] > 0 and not (possible_fg or possible_ft):
            raise ValueError(
                "positive score has no possible on-court scoring opportunities"
            )

        if target is not None and target[row] > 0 and owned[row] == 0:
            raise ValueError("a positive score has no possession exposure")

        for attempt in range(max_score_resamples + 1):
            try:
                stats = _segment_shots(
                    team_fga[row],
                    team_fta[row],
                    active,
                    duration,
                    rates,
                    None if target is None else int(target[row]),
                    rng,
                )
                break
            except ValueError as exc:
                if target is None or attempt == max_score_resamples:
                    raise ValueError(
                        f"row {row}: no feasible interval opportunity pools "
                        f"after {attempt} redraws; {exc}"
                    ) from exc

                team_fga[row] = sample_negbin(owned[row] * fga_rate, model.alpha, rng)
                team_fta[row] = sample_negbin(owned[row] * fta_rate, model.alpha, rng)
                resamples[row] += 1

        interval_trials = rng.multinomial(int(trials[row]), time_share)
        interval_tov = _hypergeom_split(interval_trials, int(team_tov[row]), rng)
        tov_weights = active * rates["tov_per_min"][None, :]
        # Empty propensity banks borrow equally across the active five.
        tov_weights = np.where(
            tov_weights.sum(axis=1, keepdims=True) > 0,
            tov_weights,
            active,
        )
        stats["TOV"] = multinomial_rows(
            interval_tov,
            _normalize_rows(tov_weights),
            rng,
        )
        ast_probability = float(
            _beta_probability(1, model.assist_rate, model.assist_rho, rng)[0]
        )
        stats["AST"] = _assists(
            stats["FGM"],
            active * rates["ast_per_min"][None, :],
            ast_probability,
            0.0,
            rng,
        )
        misses = stats["FGA"].sum(axis=1) - stats["FGM"].sum(axis=1)
        oreb_probability = float(
            _beta_probability(1, model.oreb_rate, model.oreb_rho, rng)[0]
        )
        rebounds = rng.binomial(misses, oreb_probability)
        oreb_weights = active * rates["oreb_per_min"][None, :]
        oreb_weights = np.where(
            oreb_weights.sum(axis=1, keepdims=True) > 0,
            oreb_weights,
            active,
        )
        stats["OREB"] = multinomial_rows(
            rebounds,
            _normalize_rows(oreb_weights),
            rng,
        )
        possession_exposure = owned[row] * time_share
        exposure[row] = (possession_exposure[:, None] * active).sum(axis=0)

        for name in names:
            totals[name][row] = stats[name].sum(axis=0)

            if np.any(stats[name][~active]):
                raise ValueError("an off-court player received an interval event")

        if return_intervals:
            offsets = np.array(
                [
                    12 * (phase["period"] - 1)
                    if phase["period"] <= 4
                    else 48 + 5 * (phase["period"] - 5)
                    for phase in phases
                ]
            )
            details.append(
                {
                    "simulation_index": row if index is None else index[row],
                    "player_ids": tuple(ids),
                    "period": np.array([phase["period"] for phase in phases]),
                    "start_min": (
                        np.array([phase["start"] for phase in phases]) - offsets
                    ),
                    "end_min": np.array([phase["end"] for phase in phases]) - offsets,
                    "active": active,
                    "possession_exposure": possession_exposure,
                    "possession_trials": interval_trials,
                    "stats": stats,
                }
            )

    result = {
        **totals,
        "team_fga": team_fga,
        "team_fta": team_fta,
        "team_tov": team_tov,
        "shares": _weights(rates["fga_per_min"], played),
        "score_pool_resamples": resamples,
        "player_possession_exposure": exposure,
        "metadata": {
            "segments_used": True,
            "model_version": MODEL_VERSION,
            "possession_timing": "uniform expected pace; not observed PBP possessions",
            "shot_interval_weights": "prior player rates times on-court duration",
            "assists_exclude_shooter": True,
            "oreb_uses_field_goal_misses_only": True,
            "opportunity_pools_approximate": True,
        },
    }

    if return_intervals:
        result["intervals"] = details

    return result


def allocate_game_opportunities(
    model,
    home_profile,
    away_profile,
    home_minutes,
    away_minutes,
    environment,
    rng,
    *,
    home_segments,
    away_segments,
    availability=None,
    condition_on_score=True,
    return_intervals=False,
    max_score_resamples=50,
):
    """Allocate both teams on one aligned Layer 1 and Layer 3 timeline."""
    from game_opportunities.allocate import allocate_opportunities

    _model(model)
    _rng(rng)
    _frame(environment, "environment")
    _required(
        environment,
        ["is_home", "team_poss", "opp_poss", "game_minutes", "n_ot"],
    )
    _frame(home_profile, "home_profile")
    _frame(away_profile, "away_profile")
    _required(home_profile, ["player_id"])
    _required(away_profile, ["player_id"])
    home_ids = [_key(player) for player in home_profile.player_id]
    away_ids = [_key(player) for player in away_profile.player_id]

    if set(home_ids) & set(away_ids):
        raise ValueError("a player cannot appear on both rosters")

    for profile, expected in ((home_profile, 1), (away_profile, 0)):
        if "is_home" in profile and not np.all(
            _availability(profile.is_home.to_numpy(), 1, len(profile)) == bool(expected)
        ):
            raise ValueError(
                "home/away profile is_home flags have the wrong orientation"
            )

        if "game_date" in profile and "game_date" in environment:
            profile_dates = _dates(profile.game_date)
            environment_dates = _dates(environment.game_date)

            if (
                profile_dates.nunique() != 1
                or not environment_dates.eq(profile_dates.iloc[0]).all()
            ):
                raise ValueError("profile and environment dates disagree")

    n = len(environment)

    for minutes in (home_minutes, away_minutes):
        if isinstance(minutes, pd.DataFrame) and not minutes.index.equals(
            environment.index
        ):
            raise ValueError("minute rows must retain the Layer 1 environment index")

    caps = _float_vector(
        environment.game_minutes.to_numpy(),
        n,
        "game_minutes",
        nonnegative=True,
    )
    ot = _float_vector(environment.n_ot.to_numpy(), n, "n_ot", nonnegative=True)

    if (
        np.any(ot != np.floor(ot))
        or np.any(ot > 5)
        or not np.allclose(caps, 48 + 5 * ot, rtol=0, atol=_TOL)
    ):
        raise ValueError("environment game duration and overtime disagree")

    side = _availability(environment.is_home.to_numpy(), 1, n)[0]

    if availability is not None and (
        not isinstance(availability, dict) or set(availability) != {"home", "away"}
    ):
        raise ValueError("availability must contain the shared home and away masks")

    masks = availability or {}
    home_schedules = _rotation_schedules(home_segments, n)
    away_schedules = _rotation_schedules(away_segments, n)
    boundaries = []

    for row in range(n):
        edges = [0.0, caps[row]]

        for schedule in (home_schedules[row], away_schedules[row]):
            if not isinstance(schedule, (list, tuple)) or not schedule:
                raise ValueError("each Layer 3 schedule must be a nonempty list")

            for segment in schedule:
                if not isinstance(segment, dict):
                    raise ValueError("each Layer 3 segment must be a dictionary")

                period = _number(segment.get("period"), "segment period")
                offset = 12 * (period - 1) if period <= 4 else 48 + 5 * (period - 5)
                edges.extend(
                    [
                        offset + _number(segment.get("start_min"), "segment start"),
                        offset + _number(segment.get("end_min"), "segment end"),
                    ]
                )

        boundaries.append(np.unique(edges))

    team_poss = _float_vector(
        environment.team_poss.to_numpy(),
        n,
        "team_poss",
        nonnegative=True,
    )
    opp_poss = _float_vector(
        environment.opp_poss.to_numpy(),
        n,
        "opp_poss",
        nonnegative=True,
    )
    home_poss = pd.Series(np.where(side, team_poss, opp_poss), index=environment.index)
    away_poss = pd.Series(np.where(side, opp_poss, team_poss), index=environment.index)
    home_score = None
    away_score = None

    if condition_on_score:
        _required(environment, ["team_pts", "opp_pts"])
        team_pts = _float_vector(
            environment.team_pts.to_numpy(),
            n,
            "team_pts",
            nonnegative=True,
        )
        opp_pts = _float_vector(
            environment.opp_pts.to_numpy(),
            n,
            "opp_pts",
            nonnegative=True,
        )
        home_score = pd.Series(
            np.where(side, team_pts, opp_pts),
            index=environment.index,
        )
        away_score = pd.Series(
            np.where(side, opp_pts, team_pts),
            index=environment.index,
        )

    home = allocate_opportunities(
        model,
        home_profile,
        home_minutes,
        home_poss,
        rng,
        team_points=home_score,
        availability=masks.get("home"),
        game_minutes=environment.game_minutes,
        max_score_resamples=max_score_resamples,
        segments=home_schedules,
        interval_boundaries=boundaries,
        return_intervals=return_intervals,
        game_date=environment.get("game_date"),
    )
    away = allocate_opportunities(
        model,
        away_profile,
        away_minutes,
        away_poss,
        rng,
        team_points=away_score,
        availability=masks.get("away"),
        game_minutes=environment.game_minutes,
        max_score_resamples=max_score_resamples,
        segments=away_schedules,
        interval_boundaries=boundaries,
        return_intervals=return_intervals,
        game_date=environment.get("game_date"),
    )
    return {
        "home": home,
        "away": away,
        "simulation_index": environment.index.copy(),
    }
