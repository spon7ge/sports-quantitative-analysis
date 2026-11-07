"""Official logs, team games, and rates from prior exposure."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from game_opportunities.checks import (
    _aliases,
    _dates,
    _frame,
    _game_key,
    _key,
    _minute,
    _nonnegative,
    _number,
    _positive_int,
    _required,
)
from game_opportunities.constants import _DEFAULTS, _IDS, _SHOTS, _TEAM


def _historical_rates(
    data,
    entity,
    spec,
    window,
    prior_exposure,
    prior_attempts,
):
    _positive_int(window, "window")
    _number(
        prior_exposure,
        "prior_exposure",
        nonnegative=True,
    )
    _number(
        prior_attempts,
        "prior_attempts",
        nonnegative=True,
    )

    result = (
        data.sort_values(
            [entity, "game_date", "game_id"],
            kind="stable",
        )
        .reset_index(drop=True)
        .copy()
    )

    for name, (numerator, denominator) in spec.items():
        good = _paired_mask(
            result,
            numerator,
            denominator,
        )

        num = result[numerator].where(good, 0.0)
        den = result[denominator].where(good, 0.0)

        daily = (
            pd.DataFrame(
                {
                    "game_date": result["game_date"],
                    "num": num,
                    "den": den,
                }
            )
            .groupby("game_date")
            .sum()
            .sort_index()
        )

        prior_daily = daily.cumsum().shift(1).fillna(0)
        league = (
            prior_daily["num"]
            .div(prior_daily["den"].replace(0, np.nan))
            .fillna(_DEFAULTS[name])
        )
        league_values = result["game_date"].map(league).to_numpy()

        rolling_num = np.zeros(len(result))
        rolling_den = np.zeros(len(result))
        game_dates = result["game_date"].to_numpy()
        num_values = num.to_numpy()
        den_values = den.to_numpy()

        if entity == "player_id":
            appearance = (result["minutes"].to_numpy() > 0) | (
                result[list(_SHOTS)].fillna(0).sum(axis=1).to_numpy() > 0
            )
        else:
            appearance = np.ones(
                len(result),
                dtype=bool,
            )

        for positions in result.groupby(
            entity,
            sort=False,
        ).indices.values():
            history_positions = positions[appearance[positions]]

            if not len(history_positions):
                continue

            through_num = (
                pd.Series(num_values[history_positions])
                .rolling(window, min_periods=1)
                .sum()
                .to_numpy()
            )
            through_den = (
                pd.Series(den_values[history_positions])
                .rolling(window, min_periods=1)
                .sum()
                .to_numpy()
            )

            previous = (
                np.searchsorted(
                    game_dates[history_positions],
                    game_dates[positions],
                    side="left",
                )
                - 1
            )
            has_previous = previous >= 0

            rolling_num[positions[has_previous]] = through_num[previous[has_previous]]
            rolling_den[positions[has_previous]] = through_den[previous[has_previous]]

        strength = (
            prior_attempts
            if name
            in {
                "p_3pa",
                "p_3p",
                "p_2p",
                "ft_pct",
                "fta_per_fga",
            }
            else prior_exposure
        )

        divisor = rolling_den + strength

        result[name] = np.divide(
            rolling_num + league_values * strength,
            divisor,
            out=league_values.copy(),
            where=divisor > 0,
        )

    return result


def _prepare_opportunity_logs(logs):
    _frame(logs, "logs")
    frame = _aliases(logs)

    _required(
        frame,
        [*_IDS, "minutes", *_SHOTS, *_TEAM],
    )

    frame["game_date"] = _dates(frame["game_date"])

    for name in ("player_id", "team_id"):
        frame[name] = [_key(value) for value in frame[name]]

    frame["game_id"] = [_game_key(value) for value in frame["game_id"]]
    frame["minutes"] = frame["minutes"].map(_minute)

    for name in (
        *_SHOTS,
        *_TEAM,
        "ast",
        "oreb",
        "team_fta",
        "game_minutes",
    ):
        if name not in frame:
            frame[name] = np.nan

        frame[name] = _nonnegative(
            frame[name],
            name,
        )

        if name not in {"team_poss", "game_minutes"}:
            present = frame[name].notna()

            if (frame.loc[present, name] != np.floor(frame.loc[present, name])).any():
                raise ValueError(f"{name} must contain whole counts")

    for made, attempts in (
        ("fgm", "fga"),
        ("fg3_a", "fga"),
        ("fg3_m", "fg3_a"),
        ("fg3_m", "fgm"),
        ("ftm", "fta"),
    ):
        if (frame[made] > frame[attempts]).fillna(False).any():
            raise ValueError(f"{made} exceeds {attempts}")

    frame["fg2_a"] = frame["fga"] - frame["fg3_a"]
    frame["fg2_m"] = frame["fgm"] - frame["fg3_m"]

    if (frame["fg2_m"] > frame["fg2_a"]).fillna(False).any():
        raise ValueError("two-point makes exceed two-point attempts")

    keys = ["game_id", "player_id", "team_id"]
    numeric = [
        "minutes",
        *_SHOTS,
        *_TEAM,
        "ast",
        "oreb",
        "team_fta",
        "game_minutes",
    ]

    if frame.groupby(keys)[numeric].nunique(dropna=True).gt(1).any().any():
        raise ValueError("conflicting repeated player-game statistics")

    if frame.groupby("game_id")["game_date"].nunique().gt(1).any():
        raise ValueError("a game ID has conflicting dates")

    if (
        frame.groupby(["game_id", "team_id"])[[*_TEAM, "team_fta", "game_minutes"]]
        .nunique(dropna=True)
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError("conflicting repeated team-game statistics")

    frame = frame.groupby(
        keys,
        sort=False,
        as_index=False,
    ).first()

    if (frame["team_poss"].dropna() <= 0).any():
        raise ValueError("historical possessions must be positive")

    positive_stats = frame[list(_SHOTS)].fillna(0).sum(axis=1) > 0

    if ((frame["minutes"] == 0) & positive_stats).any():
        warnings.warn(
            "zero-minute rows have positive statistics; "
            "per-minute denominators exclude them",
            stacklevel=2,
        )

    return frame.sort_values(
        ["game_date", "game_id", "player_id"],
        kind="stable",
    ).reset_index(drop=True)


def _team_games(frame):
    columns = [
        "game_id",
        "team_id",
        "game_date",
        *_TEAM,
        "team_fta",
        "complete_roster",
    ]

    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows = []

    for (game, team), data in frame.groupby(
        ["game_id", "team_id"],
        sort=False,
    ):
        item = {
            "game_id": game,
            "team_id": team,
            "game_date": data["game_date"].iloc[0],
        }

        for name in (*_TEAM, "team_fta"):
            known = data[name].dropna()
            item[name] = float(known.iloc[0]) if len(known) else np.nan

        minutes = data["minutes"].sum(min_count=len(data))

        duration = (
            48
            + 5
            * max(
                0,
                round((minutes / 5 - 48) / 5),
            )
            if np.isfinite(minutes)
            else np.nan
        )

        minute_tolerance = 1 + 0.5 * int((data["minutes"] > 0).sum())

        complete = (
            np.isfinite(minutes)
            and abs(minutes - 5 * duration) <= minute_tolerance
            and data["fga"].notna().all()
            and data["tov"].notna().all()
            and np.isclose(
                data["fga"].sum(),
                item["team_fga"],
            )
            and data["tov"].sum() <= item["team_tov"]
        )

        if not np.isfinite(item["team_fta"]) and complete and data["fta"].notna().all():
            item["team_fta"] = float(data["fta"].sum())

        item["complete_roster"] = bool(complete)
        rows.append(item)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["team_id", "game_date", "game_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _prepare_slate(slate):
    _frame(slate, "slate")
    out = _aliases(slate)

    _required(
        out,
        ["player_id", "game_date", "team_id"],
    )

    out["game_date"] = _dates(out["game_date"])

    for name in ("player_id", "team_id"):
        out[name] = [_key(value) for value in out[name]]

    if out.duplicated(["player_id", "game_date"]).any():
        raise ValueError("slate contains repeated players on the same date")

    return out.reset_index(drop=True)


def _paired_mask(data, numerator, denominator):
    if denominator in {"minutes", "team_poss"}:
        valid_exposure = data[denominator] > 0
    else:
        valid_exposure = data[denominator] >= 0

    return data[numerator].notna() & data[denominator].notna() & valid_exposure


def _paired_sums(data, numerator, denominator):
    good = _paired_mask(
        data,
        numerator,
        denominator,
    )

    return (
        float(data.loc[good, numerator].sum()),
        float(data.loc[good, denominator].sum()),
    )
