"""Play-by-play rotation templates that arrange existing Layer 2 minute draws.

Historical reconstruction uses official player logs and resolves uncertain
identities conservatively. Scheduling samples a past pattern, then solves for
the supplied minute totals. It does not draw a new minute vector.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from role_context import BLOWOUT_MARGIN, OT_MIN, REG_MIN
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

VERSION = 1
ROLE_BINS = (12.0, 26.0)
TOL = 1e-6

_ALIASES = {
    "game_id": ("gameId", "GAME_ID"),
    "player_id": ("personId", "PLAYER_ID", "playerId"),
    "team_id": ("teamId", "TEAM_ID"),
    "game_date": ("GAME_DATE", "gameDate"),
    "minutes": ("MIN", "min"),
    "player_name": ("PLAYER_NAME", "playerName", "name"),
    "is_starter": ("starter", "STARTER", "starter_flag"),
    "action_number": ("actionNumber",),
    "action_id": ("actionId",),
    "action_type": ("actionType",),
    "sub_type": ("subType",),
    "score_home": ("scoreHome",),
    "score_away": ("scoreAway",),
}


@dataclass
class Reconstruction:
    segments: pd.DataFrame
    comparisons: pd.DataFrame
    issues: pd.DataFrame


@dataclass
class PBPRotationModel:
    templates: list[dict]
    training_cutoff: str
    diagnostics: dict
    version: int = VERSION


def _frame(frame, required, name):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame")

    out = frame.copy()

    for target, aliases in _ALIASES.items():
        if target not in out:
            for alias in aliases:
                if alias in out:
                    out[target] = out[alias]
                    break

    missing = [column for column in required if column not in out]
    if missing:
        raise ValueError(f"{name} needs {missing}")

    return out


def _id(value):
    if isinstance(value, (bool, np.bool_)) or pd.isna(value):
        raise ValueError("missing or invalid ID")

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        raise ValueError("missing or invalid ID")

    try:
        number = Decimal(text)
    except InvalidOperation:
        return text

    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("numeric IDs must be finite integers")

    return str(int(number))


def _game_id(value):
    if pd.isna(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError("game_id is missing")

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "<na>"}:
        raise ValueError("game_id is missing")

    return text


def _text(value):
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _name(value):
    text = unicodedata.normalize("NFKD", _text(value))
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    text = re.sub(r"[^a-z0-9 ]", "", text)
    words = text.split()

    if words and words[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        words.pop()

    return " ".join(words)


def _clock(value):
    text = str(value).strip()
    match = re.fullmatch(
        r"PT(?:(\d+(?:\.\d+)?)M)?(\d+(?:\.\d+)?)S",
        text,
        re.I,
    )

    if match:
        return 60 * float(match[1] or 0) + float(match[2])

    if ":" in text:
        minute, second = text.split(":", 1)

        if not 0 <= float(second) < 60:
            raise ValueError("clock seconds outside [0, 60)")

        return 60 * float(minute) + float(second)

    return float(value)


def _minute(value):
    if pd.isna(value) or not str(value).strip():
        return np.nan

    text = str(value).strip()

    if ":" in text or text.upper().startswith("PT"):
        return _clock(text) / 60

    return float(value)


def _binary(values, name):
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(numeric).all() or np.any((numeric != 0) & (numeric != 1)):
        raise ValueError(f"{name} must contain 0 or 1")

    return numeric.astype(bool)


def _logs(logs):
    out = _frame(
        logs,
        ("game_id", "player_id", "team_id", "game_date", "minutes"),
        "official logs",
    )
    out["game_id"] = out.game_id.map(_game_id)

    for column in ("player_id", "team_id"):
        out[column] = out[column].map(_id)

    out["game_date"] = pd.to_datetime(out.game_date, errors="raise").dt.normalize()

    if out.game_date.isna().any():
        raise ValueError("official game dates cannot be missing")

    out["minutes"] = out.minutes.map(_minute)
    present = out.minutes.notna()

    if (
        not np.isfinite(out.loc[present, "minutes"]).all()
        or (out.loc[present, "minutes"] < 0).any()
    ):
        raise ValueError("official minutes must be nonnegative and finite when present")

    if "is_starter" not in out and "START_POSITION" in out:
        out["is_starter"] = out.START_POSITION.fillna("").astype(str).str.strip().ne("")

    if "is_starter" in out:
        out["is_starter"] = _binary(out.is_starter, "official is_starter")

    keys = ["game_id", "team_id", "player_id"]
    check = ["minutes", "game_date"]

    if "is_starter" in out:
        check.append("is_starter")

    if (out.groupby(keys)[check].nunique(dropna=False) > 1).any().any():
        raise ValueError("conflicting official player-game labels")

    if (out.groupby("game_id").game_date.nunique() > 1).any():
        raise ValueError("one game has multiple official dates")

    if (out.groupby(["game_id", "player_id"]).team_id.nunique() > 1).any():
        raise ValueError("a player belongs to both teams in one game")

    out = (
        out.drop_duplicates(keys)
        .sort_values(["game_date", "game_id", "player_id"])
        .reset_index(drop=True)
    )

    # Updates after an entire date prevent same-date target observations leaking.
    histories = {}
    recent = np.full(len(out), 18.0)

    for _, rows in out.groupby("game_date", sort=True):
        for index, row in rows.iterrows():
            prior = histories.get(row.player_id, [])

            if prior:
                recent[index] = np.mean(prior[-10:])

        for row in rows.itertuples():
            if pd.notna(row.minutes) and row.minutes > 0:
                histories.setdefault(row.player_id, []).append(float(row.minutes))

    out["prior_minutes"] = recent
    return out


def _names(roster, game):
    aliases = {}
    valid = set(roster.player_id)
    sources = []

    if "player_name" in roster:
        sources.extend(zip(roster.player_id, roster.player_name, strict=True))

    for column in ("playerName", "playerNameI", "player_name"):
        if column in game:
            sources.extend(zip(game.player_id, game[column], strict=True))

    for player, name in sources:
        if player not in valid or not _name(name):
            continue

        aliases.setdefault(_name(name), set()).add(player)

    return aliases


def _resolve(name, aliases):
    key = _name(name)
    matches = aliases.get(key, set())

    if not matches:
        words = key.split()

        if words:
            for full, players in aliases.items():
                parts = full.split()

                if (
                    parts
                    and parts[-1] == words[-1]
                    and (
                        len(words) == 1
                        or (len(words[0]) == 1 and words[0][0] == parts[0][0])
                    )
                ):
                    matches = matches | players

    if len(matches) != 1:
        raise ValueError(f"ambiguous or unknown substitution name: {name!r}")

    return next(iter(matches))


def _sub(description, aliases):
    text = "" if pd.isna(description) else str(description).strip()
    patterns = (
        r"^SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$",
        r"^(.+?)\s+enters the game for\s+(.+?)\s*$",
        r"^(.+?)\s+substitutes for\s+(.+?)\s*$",
    )

    for pattern in patterns:
        match = re.match(pattern, text, re.I)

        if match:
            incoming, outgoing = (_resolve(name, aliases) for name in match.groups())

            if incoming == outgoing:
                raise ValueError("a substitution names the same player twice")

            return incoming, outgoing

    raise ValueError(f"unrecognized substitution description: {text!r}")


def _actor(row):
    kind, sub = row.action_type, row.sub_type

    if kind in {"made shot", "missed shot", "turnover", "rebound"}:
        return True

    if kind == "free throw":
        return "technical" not in sub and "flagrant" not in sub and bool(sub)

    return kind == "foul" and sub in {
        "personal",
        "shooting",
        "offensive",
        "offensive charge",
        "loose ball",
        "personal take",
        "transition take",
        "double personal",
    }


def _events(rows, team, roster_keys, aliases):
    events = []

    for time, group in rows.groupby("elapsed", sort=False):
        group = group.reset_index(drop=True)
        sub_positions = [
            index
            for index, row in group.iterrows()
            if row.action_type == "substitution" and row.team_id == team
        ]
        first, last = (
            (min(sub_positions), max(sub_positions))
            if sub_positions
            else (len(group), -1)
        )
        actors_before = []
        actors_after = []
        incoming = []
        outgoing = []

        for index, row in group.iterrows():
            if row.team_id != team:
                continue

            if row.action_type == "substitution":
                inc, out = _sub(row.get("description", ""), aliases)
                incoming.append(inc)
                outgoing.append(out)
            elif _actor(row) and row.player_id not in {None, "0"}:
                if row.player_id not in roster_keys:
                    raise ValueError("on-court actor is absent from official roster")

                if first < index < last:
                    raise ValueError(
                        "actor between simultaneous substitutions has uncertain lineup"
                    )

                if index < first:
                    actors_before.append(row.player_id)
                else:
                    actors_after.append(row.player_id)

        for player in actors_before:
            events.append((float(time), "actor", player))

        if incoming:
            if (
                len(set(incoming)) != len(incoming)
                or len(set(outgoing)) != len(outgoing)
                or set(incoming) & set(outgoing)
            ):
                raise ValueError("simultaneous substitutions are ambiguous")

            events.append((float(time), "sub", (incoming, outgoing)))

        for player in actors_after:
            events.append((float(time), "actor", player))

    return events


def _period_segments(rows, team, roster, aliases, period):
    length = 12.0 if period <= 4 else 5.0
    keys = set(roster.player_id)
    events = _events(rows, team, keys, aliases)
    initial = {}
    state = {}

    if period == 1 and "is_starter" in roster:
        starters = set(roster.loc[roster.is_starter, "player_id"])

        if len(starters) == 5:
            initial = {player: player in starters for player in keys}
            state = initial.copy()
        elif starters:
            raise ValueError("official starter flags do not identify five players")

    for _, kind, value in events:
        if kind == "actor":
            requirements = [(value, True)]
        else:
            requirements = [(player, False) for player in value[0]] + [
                (player, True) for player in value[1]
            ]

        for player, wanted in requirements:
            if player in state and state[player] != wanted:
                raise ValueError("actor/substitution contradicts the lineup")

            if player not in state:
                initial[player] = wanted
                state[player] = wanted

        if kind == "sub":
            for player in value[0]:
                state[player] = True

            for player in value[1]:
                state[player] = False

    forced = {player for player, flag in initial.items() if flag}
    possible = {player for player in keys if initial.get(player, True)}

    if len(forced) > 5 or len(possible) < 5:
        raise ValueError("period-start lineup cannot contain exactly five players")

    if len(forced) == 5:
        lineup = forced
    elif len(possible) == 5:
        lineup = possible
    else:
        raise ValueError("quiet players leave the period-start lineup unresolved")

    result = []
    cursor = 0.0

    for time, kind, value in events:
        if kind == "actor":
            if value not in lineup:
                raise ValueError("actor is not on court")

            continue

        if time > cursor + TOL:
            result.append(
                {
                    "period": period,
                    "start_min": cursor,
                    "end_min": time,
                    "players": sorted(lineup),
                }
            )

        incoming, outgoing = map(set, value)
        if not outgoing <= lineup or incoming & lineup:
            raise ValueError("invalid outgoing/incoming lineup membership")

        lineup = (lineup - outgoing) | incoming

        if len(lineup) != 5:
            raise ValueError("substitution does not leave five players")

        cursor = time

    if cursor < length - TOL:
        result.append(
            {
                "period": period,
                "start_min": cursor,
                "end_min": length,
                "players": sorted(lineup),
            }
        )

    return result


def _home_team(game, labels, games):
    candidates = set()

    if games is not None:
        match = games.loc[games.game_id == game.game_id.iloc[0]]

        if not match.empty:
            candidates.update(match.home_team_id.map(_id))

    if "is_home" in labels:
        flags = _binary(labels.is_home, "official is_home")
        candidates.update(labels.loc[flags, "team_id"])

    for column in ("MATCHUP", "matchup"):
        if column in labels:
            candidates.update(
                labels.loc[
                    labels[column].astype(str).str.contains(" vs", regex=False),
                    "team_id",
                ]
            )

    if "location" in game:
        for row in game.itertuples():
            if _text(row.location) in {"h", "home"} and row.team_id not in {None, "0"}:
                candidates.add(row.team_id)

    if len(candidates) != 1 or not candidates <= set(labels.team_id):
        raise ValueError("cannot resolve one consistent home team")

    return next(iter(candidates))


def script_exposure(pbp, official_logs):
    """Close minutes and garbage appearances from parsed stints.

    One row per player on the floor in a parsed team-game. A stint is split
    where the score changes: the margin can cross the blowout cutoff without
    a substitution. close_min is floor time at or inside that cutoff.
    garbage_app is any positive time outside it. reached_20 is whether the
    game's score ever went outside it.
    """
    labels = _exposure_logs(official_logs)
    work = _exposure_pbp(pbp)
    grouped = {key: group for key, group in labels.groupby("game_id", sort=False)}
    records = []
    for game_id, raw in work.groupby("game_id", sort=False):
        if game_id not in grouped:
            continue
        official = grouped[game_id]
        try:
            periods, max_margin = _exposure_periods(raw)
        except (TypeError, ValueError):
            continue
        reached = bool(max_margin > BLOWOUT_MARGIN)
        date = official.game_date.iloc[0]
        for _, roster in official.groupby("team_id", sort=False):
            try:
                minutes = _team_script_minutes(periods, roster, raw)
            except (TypeError, ValueError):
                continue
            for player, (close, garbage) in minutes.items():
                if close + garbage <= TOL:
                    continue
                records.append(
                    {
                        "game_id": game_id,
                        "game_date": date,
                        "team_id": roster.team_id.iloc[0],
                        "player_id": player,
                        "close_min": close,
                        "garbage_app": garbage > TOL,
                        "reached_20": reached,
                    }
                )
    return pd.DataFrame(
        records,
        columns=[
            "game_id",
            "game_date",
            "team_id",
            "player_id",
            "close_min",
            "garbage_app",
            "reached_20",
        ],
    )


def _exposure_logs(logs):
    frame = logs.copy()
    if "is_starter" not in frame and "start_position" in frame:
        frame["is_starter"] = (
            frame["start_position"].fillna("").astype(str).str.strip().ne("")
        )
    return _logs(frame)


def _exposure_pbp(pbp):
    work = _frame(
        pbp,
        (
            "game_id",
            "action_number",
            "period",
            "clock",
            "team_id",
            "player_id",
            "action_type",
            "sub_type",
            "score_home",
            "score_away",
            "description",
        ),
        "PBP",
    )
    work["game_id"] = work.game_id.map(_game_id)
    for column in ("player_id", "team_id"):
        work[column] = work[column].map(
            lambda value: None if pd.isna(value) else _id(value)
        )
    for column in ("action_type", "sub_type"):
        work[column] = work[column].map(_text)
    work["description"] = work["description"].map(
        lambda value: "" if pd.isna(value) else str(value)
    )
    work["action_number"] = pd.to_numeric(work["action_number"], errors="raise")
    return work


def _exposure_periods(game):
    # A shot and its block share an actionNumber and have distinct action ids.
    if "action_id" in game.columns:
        ordered = game.copy()
        ordered["action_id"] = pd.to_numeric(ordered["action_id"], errors="raise")
        ordered = ordered.drop_duplicates().sort_values("action_id", kind="stable")
        if ordered.action_id.duplicated().any():
            raise ValueError("conflicting duplicate action id")
    else:
        ordered = game.drop_duplicates().sort_values("action_number", kind="stable")
        if ordered.action_number.duplicated().any():
            raise ValueError("conflicting duplicate actionNumber")
    values = pd.to_numeric(ordered.period, errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values != np.floor(values))
        or np.any(values < 1)
        or np.any(np.diff(values) < 0)
    ):
        raise ValueError("invalid period sequence")
    ordered = ordered.copy()
    ordered["period"] = values.astype(int)
    periods = {}
    margin = 0.0
    maximum = 0.0
    for period, rows in ordered.groupby("period", sort=False):
        length = 12.0 if int(period) <= 4 else 5.0
        remaining = rows.clock.map(_clock).to_numpy(dtype=float)
        if (
            not np.isfinite(remaining).all()
            or np.any(remaining < 0)
            or np.any(remaining > 60 * length + TOL)
            or np.any(np.diff(remaining) > TOL)
        ):
            raise ValueError("invalid or increasing clock")
        scored = rows.assign(elapsed=length - remaining / 60.0)
        times, margins, margin = _margin_marks(scored, margin)
        if margins:
            maximum = max(maximum, max(margins))
        periods[int(period)] = {
            "rows": scored,
            "times": times,
            "margins": margins,
        }
    return periods, maximum


def _margin_marks(rows, incoming):
    home = pd.to_numeric(rows["score_home"], errors="coerce").to_numpy(dtype=float)
    away = pd.to_numeric(rows["score_away"], errors="coerce").to_numpy(dtype=float)
    elapsed = rows["elapsed"].to_numpy(dtype=float)
    current = float(incoming)
    times = []
    margins = []
    for index in range(len(rows)):
        if np.isfinite(home[index]) and np.isfinite(away[index]):
            current = abs(float(home[index]) - float(away[index]))
        if times and abs(times[-1] - float(elapsed[index])) <= TOL:
            margins[-1] = current
        else:
            times.append(float(elapsed[index]))
            margins.append(current)
    return times, margins, current


def _exposure_aliases(roster, game):
    """Roster names plus unique shortened first names from the same roster.

    The feed writes ``Jal. Williams`` when ``J. Williams`` would be ambiguous.
    Prefixes shorter than three letters stay on the existing initial match.
    """
    aliases = _names(roster, game)
    if "player_name" not in roster.columns:
        return aliases
    buckets = {}
    for player, name in zip(roster.player_id, roster.player_name, strict=True):
        parts = _name(name).split()
        if len(parts) < 2:
            continue
        first, last = parts[0], parts[-1]
        for length in range(3, len(first)):
            buckets.setdefault(f"{first[:length]} {last}", set()).add(player)
    for alias, players in buckets.items():
        if len(players) == 1 and alias not in aliases:
            aliases[alias] = players
    return aliases


def _team_script_minutes(periods, roster, game):
    team = roster.team_id.iloc[0]
    aliases = _exposure_aliases(roster, game)
    totals = {}
    for period, part in periods.items():
        segments = _period_segments(part["rows"], team, roster, aliases, period)
        for segment in segments:
            close, garbage = _split_margin(
                segment["start_min"],
                segment["end_min"],
                part["times"],
                part["margins"],
            )
            for player in segment["players"]:
                previous = totals.get(player, (0.0, 0.0))
                totals[player] = (previous[0] + close, previous[1] + garbage)
    return totals


def _split_margin(start, end, times, margins):
    if end <= start + TOL:
        return 0.0, 0.0
    index = 0
    current = 0.0
    while index < len(times) and times[index] <= start + TOL:
        current = margins[index]
        index += 1
    close = 0.0
    garbage = 0.0
    cursor = float(start)
    limit = float(end)
    while cursor < limit - TOL:
        nxt = limit
        if index < len(times) and times[index] < limit - TOL:
            nxt = times[index]
        if current > BLOWOUT_MARGIN:
            garbage += nxt - cursor
        else:
            close += nxt - cursor
        cursor = nxt
        if index < len(times) and times[index] <= cursor + TOL:
            current = margins[index]
            index += 1
        else:
            break
    return close, garbage


_COUNTING_FOULS = frozenset(
    {
        "personal",
        "shooting",
        "offensive",
        "offensive charge",
        "loose ball",
        "personal take",
        "transition take",
        "double personal",
    }
)


def player_stints(pbp, official_logs):
    """On-court stints labeled for the foul, blowout, and exit hazards.

    A game that cannot be parsed is skipped. The margin is the absolute
    score difference at the moment the player leaves the floor. A player
    who stays through a period break keeps one stint.
    """

    labels = _exposure_logs(official_logs)
    work = _exposure_pbp(pbp)
    grouped = {key: group for key, group in labels.groupby("game_id", sort=False)}
    rows = []
    for game_id, raw in work.groupby("game_id", sort=False):
        if game_id not in grouped:
            continue
        try:
            periods, _maximum = _exposure_periods(raw)
        except (TypeError, ValueError):
            continue
        if not periods:
            continue
        last_period = max(periods)
        game_minutes = REG_MIN + OT_MIN * max(0, last_period - 4)
        official = grouped[game_id]
        for _, roster in official.groupby("team_id", sort=False):
            try:
                rows.extend(
                    _team_player_stints(periods, roster, raw, game_id, game_minutes)
                )
            except (TypeError, ValueError):
                continue
    columns = (
        "game_id",
        "team_id",
        "player_id",
        "is_starter",
        "start_elapsed",
        "end_elapsed",
        "game_minutes",
        "margin",
        "fouls",
        "timeout",
        "reason",
    )
    return pd.DataFrame(rows, columns=list(columns))


def rotation_ticks(pbp, official_logs):
    """Player-minute rows for ``fit_rotation_process``, from parsed stints."""

    from game_rotations.process import expand_stints

    stints = player_stints(pbp, official_logs)
    if stints.empty:
        raise ValueError("play-by-play produced no on-court stints")
    ticks = expand_stints(stints)
    extra = _blowout_returns(stints)
    if extra:
        ticks = pd.concat([ticks, pd.DataFrame(extra)], ignore_index=True)
    return ticks


def _team_player_stints(periods, roster, game, game_id, game_minutes):
    team = roster.team_id.iloc[0]
    aliases = _exposure_aliases(roster, game)
    starters = set()
    if "is_starter" in roster.columns:
        starters = set(roster.loc[roster.is_starter.astype(bool), "player_id"])
    fouls = _foul_times(game, team, periods)
    ejections = _ejection_times(game, team, periods)
    timeouts = _timeout_times(game, periods)
    runs = {}
    open_run = {}
    for period in sorted(periods):
        part = periods[period]
        offset = (period - 1) * 12.0 if period <= 4 else REG_MIN + (period - 5) * OT_MIN
        segments = _period_segments(part["rows"], team, roster, aliases, period)
        for segment in segments:
            start = offset + float(segment["start_min"])
            end = offset + float(segment["end_min"])
            margin = _margin_at(
                part["times"], part["margins"], float(segment["end_min"])
            )
            present = set(segment["players"])
            for player, run in list(open_run.items()):
                if player not in present or start > run["end"] + 1e-6:
                    runs.setdefault(player, []).append(run)
                    del open_run[player]
            for player in present:
                continued = (
                    player in open_run and abs(open_run[player]["end"] - start) <= 1e-6
                )
                if continued:
                    open_run[player]["end"] = end
                    open_run[player]["margin"] = margin
                    continue
                if player in open_run:
                    runs.setdefault(player, []).append(open_run.pop(player))
                open_run[player] = {
                    "start": start,
                    "end": end,
                    "margin": margin,
                }
    for player, run in open_run.items():
        runs.setdefault(player, []).append(run)

    stints = []
    for player, pieces in runs.items():
        for run in pieces:
            start = float(np.floor(run["start"] + 1e-8))
            end = float(np.ceil(run["end"] - 1e-8))
            end = min(end, float(game_minutes))
            if end <= start:
                end = min(float(game_minutes), start + 1.0)
            if end <= start:
                continue
            run = {**run, "start": start, "end": end}
            foul_count = sum(time <= end + 1e-6 for time in fouls.get(player, ()))
            ejected = any(abs(time - end) <= 1.0 for time in ejections.get(player, ()))
            if end >= game_minutes - 1e-4:
                reason = "censored"
            elif ejected:
                reason = "ejection"
            elif foul_count >= 5 or (
                foul_count >= 2 and foul_count > 6.0 * end / REG_MIN
            ):
                reason = "foul_trouble"
            elif player in starters and float(run["margin"]) >= 15.0:
                reason = "blowout"
            else:
                reason = "rotation"
            timeout = any(end - 1.0 < time <= end + 1e-6 for time in timeouts)
            stints.append(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "player_id": player,
                    "is_starter": int(player in starters),
                    "start_elapsed": float(run["start"]),
                    "end_elapsed": end,
                    "game_minutes": float(game_minutes),
                    "margin": float(run["margin"]),
                    "fouls": float(foul_count),
                    "timeout": int(timeout),
                    "reason": reason,
                }
            )
    return stints


def _elapsed_marks(periods):
    marks = {}
    for period, part in periods.items():
        offset = (period - 1) * 12.0 if period <= 4 else REG_MIN + (period - 5) * OT_MIN
        rows = part["rows"]
        elapsed = offset + rows["elapsed"].to_numpy(dtype=float)
        marks[period] = elapsed
    return marks


def _foul_times(game, team, periods):
    found = {}
    marks = _elapsed_marks(periods)
    for period, part in periods.items():
        rows = part["rows"]
        elapsed = marks[period]
        action = rows["action_type"].to_numpy()
        sub = rows["sub_type"].to_numpy()
        owner = rows["team_id"].to_numpy()
        player = rows["player_id"].to_numpy()
        for index in range(len(rows)):
            if owner[index] != team or player[index] in {None, "0"}:
                continue
            if action[index] == "foul" and sub[index] in _COUNTING_FOULS:
                found.setdefault(player[index], []).append(float(elapsed[index]))
    return found


def _ejection_times(game, team, periods):
    found = {}
    marks = _elapsed_marks(periods)
    for period, part in periods.items():
        rows = part["rows"]
        elapsed = marks[period]
        action = rows["action_type"].to_numpy()
        owner = rows["team_id"].to_numpy()
        player = rows["player_id"].to_numpy()
        for index in range(len(rows)):
            if action[index] != "ejection" or owner[index] != team:
                continue
            if player[index] in {None, "0"}:
                continue
            found.setdefault(player[index], []).append(float(elapsed[index]))
    return found


def _timeout_times(game, periods):
    marks = _elapsed_marks(periods)
    times = []
    for period, part in periods.items():
        rows = part["rows"]
        elapsed = marks[period]
        action = rows["action_type"].to_numpy()
        for index in range(len(rows)):
            if action[index] == "timeout":
                times.append(float(elapsed[index]))
    return times


def _margin_at(times, margins, minute):
    current = 0.0
    for time, margin in zip(times, margins):
        if time <= minute + 1e-6:
            current = float(margin)
        else:
            break
    return current


def _blowout_returns(stints):
    rows = []
    grouped = stints.groupby(["game_id", "team_id", "player_id"], sort=False)
    for (game_id, team_id, player_id), group in grouped:
        ordered = group.sort_values("start_elapsed")
        pieces = list(ordered.itertuples(index=False))
        for index, stint in enumerate(pieces):
            if stint.reason != "blowout":
                continue
            later = pieces[index + 1 :]
            returned = any(
                piece.start_elapsed >= stint.end_elapsed - 1e-6 for piece in later
            )
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "player_id": player_id,
                    "stint_minutes": 0.0,
                    "fatigue_minutes": float(stint.end_elapsed),
                    "timeout": 0.0,
                    "rotation_sub": 0.0,
                    "fouls": float(stint.fouls),
                    "elapsed_minutes": float(stint.end_elapsed),
                    "foul_trouble_exit": 0.0,
                    "margin": float(stint.margin),
                    "minutes_left": float(stint.game_minutes - stint.end_elapsed),
                    "is_starter": float(stint.is_starter),
                    "blowout_exit": 0.0,
                    "injury_exit": 0.0,
                    "ejection_exit": 0.0,
                    "blowout_benched": 1.0,
                    "inside_threshold": 1.0,
                    "returned": float(returned),
                }
            )
    return rows


def reconstruct_rotation_segments(
    pbp,
    official_logs,
    *,
    games=None,
    minute_tolerance=1.0,
):
    """Return historical intervals, per-player minute checks, and explicit issues.

    Only complete team-games agreeing with official minutes become templates.
    Uncertain periods remain excluded; their official labels are retained.
    """
    if not np.isfinite(minute_tolerance) or minute_tolerance < 0:
        raise ValueError("minute_tolerance must be nonnegative and finite")

    labels = _logs(official_logs)
    work = _frame(
        pbp,
        (
            "game_id",
            "action_number",
            "period",
            "clock",
            "team_id",
            "player_id",
            "action_type",
            "sub_type",
            "score_home",
            "score_away",
        ),
        "PBP",
    )
    work["game_id"] = work.game_id.map(_game_id)

    for column in ("player_id", "team_id"):
        work[column] = work[column].map(
            lambda value: None if pd.isna(value) else _id(value)
        )

    for column in ("action_type", "sub_type"):
        work[column] = work[column].map(_text)

    if games is not None:
        games = _frame(games, ("game_id", "home_team_id"), "games")
        games["game_id"] = games.game_id.map(_game_id)

    segments = []
    comparisons = []
    issues = []
    grouped_labels = {
        key: group for key, group in labels.groupby("game_id", sort=False)
    }

    for game_id, raw in work.groupby("game_id", sort=False):

        def issue(team, period, reason, current_game=game_id):
            issues.append(
                {
                    "game_id": current_game,
                    "team_id": team,
                    "period": period,
                    "reason": str(reason),
                }
            )

        try:
            if game_id not in grouped_labels:
                raise ValueError("no official game labels")

            official = grouped_labels[game_id]

            if official.team_id.nunique() != 2:
                raise ValueError("official game needs exactly two teams")

            game = raw.copy()
            game["action_number"] = pd.to_numeric(game.action_number, errors="raise")

            if (
                not np.isfinite(game.action_number).all()
                or (game.action_number != np.floor(game.action_number)).any()
            ):
                raise ValueError("invalid actionNumber")

            game = game.drop_duplicates().sort_values("action_number", kind="stable")

            if game.action_number.duplicated().any():
                raise ValueError("conflicting duplicate actionNumber")

            values = pd.to_numeric(game.period, errors="raise").to_numpy(dtype=float)

            if (
                not np.isfinite(values).all()
                or np.any(values != np.floor(values))
                or np.any(values < 1)
                or np.any(np.diff(values) < 0)
            ):
                raise ValueError("invalid period sequence")

            game["period"] = values.astype(int)
            last = int(values.max())

            if set(values) != set(range(1, last + 1)) or last < 4 or last > 9:
                raise ValueError("incomplete or unsupported period sequence")

            substitutions = game.loc[game.action_type == "substitution"]

            if not substitutions.team_id.isin(set(official.team_id)).all():
                raise ValueError("substitution has an unknown team")

            home = _home_team(game, official, games)
            score = (
                game[["score_home", "score_away"]]
                .apply(pd.to_numeric, errors="coerce")
                .ffill()
            )
            final = score.iloc[-1].to_numpy(dtype=float)

            if not np.isfinite(final).all() or np.any(final < 0):
                raise ValueError("missing final home/away scores")

            points_column = next(
                (column for column in ("PTS", "pts", "points") if column in official),
                None,
            )

            if points_column:
                for team, rows in official.groupby("team_id", sort=False):
                    points = pd.to_numeric(
                        rows[points_column], errors="coerce"
                    ).to_numpy(dtype=float)

                    if np.isfinite(points).all():
                        expected = final[0] if team == home else final[1]

                        if np.any(points < 0) or abs(points.sum() - expected) > TOL:
                            raise ValueError(
                                "PBP final score disagrees with official player points"
                            )

            home_margin = float(final[0] - final[1])
            duration = REG_MIN + OT_MIN * (last - 4)
            period_rows = {}

            for period, rows in game.groupby("period", sort=False):
                try:
                    remaining = rows.clock.map(_clock).to_numpy(dtype=float)
                    length = 12.0 if period <= 4 else 5.0

                    if (
                        not np.isfinite(remaining).all()
                        or np.any(remaining < 0)
                        or np.any(remaining > 60 * length + TOL)
                        or np.any(np.diff(remaining) > TOL)
                    ):
                        raise ValueError("invalid or increasing clock")

                    if remaining[-1] > TOL:
                        raise ValueError("period does not reach its end")

                    starts = (rows.action_type == "period") & (rows.sub_type == "start")
                    ends = (rows.action_type == "period") & (rows.sub_type == "end")

                    if np.any(
                        np.abs(remaining[starts.to_numpy()] - 60 * length) > TOL
                    ) or np.any(remaining[ends.to_numpy()] > TOL):
                        raise ValueError("period boundary clock is invalid")

                    period_rows[period] = rows.assign(elapsed=length - remaining / 60)
                except (TypeError, ValueError) as exc:
                    issue(None, period, exc)

            for team, roster in official.groupby("team_id", sort=False):
                team_segments = []
                aliases = _names(roster, game)
                complete = len(period_rows) == last

                for period, rows in period_rows.items():
                    try:
                        team_segments.extend(
                            _period_segments(rows, team, roster, aliases, period)
                        )
                    except ValueError as exc:
                        complete = False
                        issue(team, period, exc)

                exposure = {player: 0.0 for player in roster.player_id}

                for segment in team_segments:
                    for player in segment["players"]:
                        exposure[player] += segment["end_min"] - segment["start_min"]

                observed = roster.minutes.to_numpy(dtype=float)
                rebuilt = np.array([exposure[player] for player in roster.player_id])
                official_budget = (
                    np.isfinite(observed).all()
                    and abs(observed.sum() - 5 * duration)
                    <= len(roster) * minute_tolerance + TOL
                )
                minute_ok = (
                    complete
                    and official_budget
                    and np.all(np.abs(rebuilt - observed) <= minute_tolerance + TOL)
                )

                if complete and not minute_ok:
                    issue(
                        team,
                        None,
                        "reconstructed minutes disagree with official labels",
                    )

                q1 = [segment for segment in team_segments if segment["period"] == 1]
                starters = set(q1[0]["players"]) if q1 else set()
                role_bin = dict(
                    zip(
                        roster.player_id,
                        np.digitize(roster.prior_minutes, ROLE_BINS),
                        strict=True,
                    )
                )
                date = roster.game_date.iloc[0]

                for row in roster.itertuples():
                    comparisons.append(
                        {
                            "game_id": game_id,
                            "game_date": date,
                            "team_id": team,
                            "player_id": row.player_id,
                            "official_minutes": row.minutes,
                            "reconstructed_minutes": exposure[row.player_id],
                            "complete": complete,
                            "validated": minute_ok,
                            "error": (
                                exposure[row.player_id] - row.minutes
                                if complete
                                else np.nan
                            ),
                        }
                    )

                for segment in team_segments:
                    players = segment["players"]
                    signature = [
                        int(sum(role_bin[player] == role for player in players))
                        for role in range(3)
                    ] + [len(set(players) & starters)]
                    segments.append(
                        {
                            **segment,
                            "game_id": game_id,
                            "game_date": date,
                            "team_id": team,
                            "team_margin": (
                                home_margin if team == home else -home_margin
                            ),
                            "n_ot": last - 4,
                            "game_minutes": duration,
                            "signature": signature,
                            "validated": bool(minute_ok),
                        }
                    )
        except (TypeError, ValueError) as exc:
            issue(None, None, exc)

    return Reconstruction(
        pd.DataFrame(segments),
        pd.DataFrame(comparisons),
        pd.DataFrame(issues, columns=["game_id", "team_id", "period", "reason"]),
    )


def fit_pbp_rotation(reconstruction, *, holdout_fraction=0.2):
    """Fit an empirical whole-game template bank from validated team-games.

    A date-separated holdout reports retrieval distances for interval timing
    and role mixes. These are descriptive diagnostics, not forecast accuracy.
    """
    if not isinstance(reconstruction, Reconstruction):
        raise ValueError("supply the Reconstruction returned by reconstruction")

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")

    frame = reconstruction.segments

    if frame.empty or not frame.validated.any():
        raise ValueError("no complete, minute-validated team-games to fit")

    frame = frame.loc[frame.validated].copy()
    templates = []

    for (game, team), rows in frame.groupby(["game_id", "team_id"], sort=False):
        rows = rows.sort_values(["period", "start_min"])
        first = rows.iloc[0]
        templates.append(
            {
                "game_id": game,
                "team_id": team,
                "game_date": first.game_date,
                "team_margin": float(first.team_margin),
                "n_ot": int(first.n_ot),
                "periods": {
                    int(period): group[["start_min", "end_min", "signature"]].to_dict(
                        "records"
                    )
                    for period, group in rows.groupby("period", sort=True)
                },
            }
        )

    dates = sorted({template["game_date"] for template in templates})

    if len(dates) < 3:
        raise ValueError(
            "need at least three validated game dates for chronological evaluation"
        )

    cut = min(len(dates) - 1, max(2, int(len(dates) * (1 - holdout_fraction))))
    training = [
        template for template in templates if template["game_date"] < dates[cut]
    ]
    holdout = [
        template for template in templates if template["game_date"] >= dates[cut]
    ]
    timing = []
    mixes = []

    for target in holdout:
        candidate = min(
            training,
            key=lambda template: _distance(
                template,
                target["team_id"],
                target["team_margin"],
                target["n_ot"],
            ),
        )

        for period in range(1, 5):
            actual = target["periods"][period]
            prior = candidate["periods"][period]
            actual_times = np.array([segment["end_min"] for segment in actual[:-1]])
            prior_times = np.array([segment["end_min"] for segment in prior[:-1]])

            if actual_times.size:
                comparison_times = prior_times if prior_times.size else np.array([12.0])
                timing.extend(
                    np.min(
                        np.abs(actual_times[:, None] - comparison_times[None, :]),
                        axis=1,
                    )
                )

            for time in np.arange(0.5, 12.0, 1.0):
                actual_segment = next(
                    segment
                    for segment in actual
                    if segment["start_min"] <= time < segment["end_min"]
                )
                prior_segment = next(
                    segment
                    for segment in prior
                    if segment["start_min"] <= time < segment["end_min"]
                )
                mixes.append(
                    float(
                        np.mean(
                            np.abs(
                                np.asarray(actual_segment["signature"])
                                - prior_segment["signature"]
                            )
                        )
                    )
                )

    diagnostics = {
        "training_team_games": len(training),
        "holdout_team_games": len(holdout),
        "holdout_first_date": str(dates[cut].date()),
        "nearest_boundary_mae_minutes": float(np.mean(timing)) if timing else None,
        "role_signature_mae": float(np.mean(mixes)) if mixes else None,
        "metric_scope": (
            "historical template retrieval with observed final margin; "
            "not a pregame forecast"
        ),
        "validated_team_games": len(templates),
        "reconstruction_issue_count": len(reconstruction.issues),
    }
    return PBPRotationModel(templates, str(dates[-1].date()), diagnostics)


def _distance(template, team, margin, n_ot):
    return (
        (0.75 if template["team_id"] != team else 0.0)
        + min(abs(template["team_margin"] - margin) / 15.0, 4.0)
        + abs(template["n_ot"] - n_ot)
    )


def _phases(template, n_ot, rng):
    result = []

    for period in range(1, 5 + n_ot):
        if period in template["periods"]:
            rows = template["periods"][period]
            scale = 1.0
        else:
            overtime = [source for source in template["periods"] if source >= 5]
            source = int(rng.choice(overtime)) if overtime else 4
            rows = template["periods"][source]
            scale = 1.0 if source >= 5 else 5.0 / 12.0

        for row in rows:
            result.append(
                {
                    "period": period,
                    "start_min": row["start_min"] * scale,
                    "end_min": row["end_min"] * scale,
                    "signature": row["signature"],
                }
            )

    return result


def _arrange(phases, player_ids, target, recent, starter, rng, max_variables):
    original_target = target.copy()
    n_ot = max(phase["period"] for phase in phases) - 4
    phases = [dict(phase) for phase in phases]
    prefix = []
    opening = np.flatnonzero(starter)

    if len(opening) == 5 and np.all(target[opening] > 0):
        game_length = sum(phase["end_min"] - phase["start_min"] for phase in phases)
        other = np.flatnonzero(~starter)
        room = game_length - float(target[other].max()) if len(other) else game_length
        length = min(phases[0]["end_min"], float(target[opening].min()), room)

        if length > TOL:
            prefix = [
                {
                    "period": 1,
                    "start_min": 0.0,
                    "end_min": length,
                    "players": [player_ids[index] for index in opening],
                }
            ]
            target = target.copy()
            target[opening] -= length
            phases[0]["start_min"] = length

            if phases[0]["end_min"] - length <= TOL:
                phases.pop(0)

    # Tiny positive targets must still be represented in the exact allocation.
    active = np.flatnonzero(target > 0)

    if len(active) < 5:
        raise ValueError("positive-minute roster needs at least five players")

    n_combinations = comb(len(active), 5)
    n_phases = len(phases)

    if n_combinations * n_phases > max_variables:
        raise ValueError(
            "rotation optimization is too large; increase max_variables explicitly"
        )

    combos = np.array(list(combinations(active, 5)), dtype=int)
    roles = np.digitize(recent, ROLE_BINS)
    signature = np.column_stack(
        [(roles[combos] == role).sum(axis=1) for role in range(3)]
        + [starter[combos].sum(axis=1)]
    )
    desired = np.array([phase["signature"] for phase in phases])
    cost = np.abs(desired[:, None, :] - signature[None, :, :]).sum(axis=2)
    cost = cost + rng.uniform(0, 0.02, cost.shape)
    duration = np.array([phase["end_min"] - phase["start_min"] for phase in phases])

    # Every column allocates time to one phase and exactly five players.
    columns = np.arange(n_combinations * n_phases)
    row_index = np.concatenate(
        [
            np.repeat(np.arange(n_phases), n_combinations),
            np.tile(combos.reshape(-1), n_phases) + n_phases,
        ]
    )
    col_index = np.concatenate([columns, np.repeat(columns, 5)])
    matrix = coo_matrix(
        (np.ones(len(row_index)), (row_index, col_index)),
        shape=(n_phases + len(player_ids), n_combinations * n_phases),
    ).tocsr()
    solved = linprog(
        cost.reshape(-1),
        A_eq=matrix,
        b_eq=np.r_[duration, target],
        bounds=(0, None),
        method="highs",
    )

    if not solved.success:
        raise ValueError(f"could not arrange the supplied minutes: {solved.message}")

    allocations = solved.x.reshape(n_phases, n_combinations)
    result = prefix
    previous = set(opening) if prefix else set()

    for phase, values in zip(phases, allocations, strict=True):
        pending = list(np.flatnonzero(values > 1e-9))
        cursor = phase["start_min"]

        while pending:
            chosen = max(pending, key=lambda index: len(set(combos[index]) & previous))
            pending.remove(chosen)
            end = cursor + float(values[chosen])
            players = [player_ids[index] for index in combos[chosen]]

            if (
                result
                and result[-1]["period"] == phase["period"]
                and set(result[-1]["players"]) == set(players)
                and abs(result[-1]["end_min"] - cursor) < TOL
            ):
                result[-1]["end_min"] = end
            else:
                result.append(
                    {
                        "period": phase["period"],
                        "start_min": cursor,
                        "end_min": end,
                        "players": players,
                    }
                )

            previous = set(combos[chosen])
            cursor = end

    _check_schedule(result, player_ids, original_target, n_ot)
    return result


def _check_schedule(schedule, ids, target, n_ot):
    exposure = dict.fromkeys(ids, 0.0)

    for period in range(1, 5 + n_ot):
        cursor = 0.0

        for segment in [item for item in schedule if item["period"] == period]:
            if (
                abs(segment["start_min"] - cursor) > TOL
                or segment["end_min"] <= cursor
                or len(set(segment["players"])) != 5
            ):
                raise ValueError("invalid rotation coverage or lineup")

            for player in segment["players"]:
                exposure[player] += segment["end_min"] - cursor

            cursor = segment["end_min"]

        expected = 12.0 if period <= 4 else 5.0

        if abs(cursor - expected) > TOL:
            raise ValueError("rotation does not cover its periods")

    if not np.allclose([exposure[player] for player in ids], target, rtol=0, atol=TOL):
        raise ValueError("rotation changed the supplied Layer 2 minute draw")


def learned_segments_from_minutes(
    model,
    roster,
    environment,
    minute_draws,
    *,
    availability=None,
    game_date=None,
    seed=0,
    max_variables=150_000,
):
    """Sample historical patterns, then arrange exact existing minute totals.

    Returns one segment list per environment row, in the supplied row order.
    Environment margin and is_home must already use this roster's perspective.
    """
    if (
        not isinstance(model, PBPRotationModel)
        or model.version != VERSION
        or not model.templates
    ):
        raise ValueError("unsupported or empty PBP rotation model")

    role = _frame(roster, ("player_id", "team_id"), "roster")
    env = _frame(environment, ("team_margin", "n_ot", "game_minutes"), "environment")
    ids = role.player_id.map(_id).tolist()
    teams = set(role.team_id.map(_id))

    if len(set(ids)) != len(ids) or len(teams) != 1:
        raise ValueError("roster needs distinct players belonging to one team")

    if "is_home" in role and "is_home" in env:
        side = _binary(role.is_home, "roster is_home")
        env_side = _binary(env.is_home, "environment is_home")

        if len(set(side)) != 1 or np.any(env_side != side[0]):
            raise ValueError("environment must use the roster's perspective")

    dates = env.game_date if "game_date" in env else game_date

    if dates is None:
        raise ValueError("supply game_date to enforce the PBP model cutoff")

    dates = pd.to_datetime(
        pd.Series(np.broadcast_to(np.asarray(dates), (len(env),))),
        errors="raise",
    ).dt.normalize()

    if dates.isna().any() or (dates <= pd.Timestamp(model.training_cutoff)).any():
        raise ValueError("simulation dates must follow the PBP training cutoff")

    if isinstance(minute_draws, pd.DataFrame):
        if (
            not minute_draws.index.equals(env.index)
            or [_id(player) for player in minute_draws.columns] != ids
        ):
            raise ValueError(
                "minute rows/index/columns must match environment and roster order"
            )

    target = np.asarray(minute_draws, dtype=float)

    if (
        target.shape != (len(env), len(role))
        or not np.isfinite(target).all()
        or np.any(target < 0)
    ):
        raise ValueError(
            "minutes must be finite nonnegative (n_simulations, n_players)"
        )

    script = (
        env[["team_margin", "n_ot", "game_minutes"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    if not np.isfinite(script).all():
        raise ValueError("environment must be finite")

    margin, counts, cap = script.T

    if (
        np.any(counts < 0)
        or np.any(counts > 5)
        or np.any(counts != np.floor(counts))
        or not np.allclose(cap, REG_MIN + OT_MIN * counts, rtol=0, atol=TOL)
    ):
        raise ValueError("game duration must agree with integer overtime count")

    fixed = (
        _binary(role.available, "available")
        if "available" in role
        else np.ones(len(role), dtype=bool)
    )
    mask = np.broadcast_to(fixed, target.shape).copy()

    if availability is not None:
        supplied = np.asarray(availability)

        if supplied.shape != target.shape or supplied.dtype.kind != "b":
            raise ValueError("availability must be a boolean matrix aligned to minutes")

        if np.any(supplied & ~mask):
            raise ValueError(
                "availability cannot reactivate a fixed unavailable player"
            )

        mask &= supplied

    if np.any(mask.sum(axis=1) < 5) or np.any(target[~mask] != 0):
        raise ValueError(
            "unavailable players must have zero minutes "
            "and five players must be available"
        )

    if np.any(target > cap[:, None] + TOL) or not np.allclose(
        target.sum(axis=1), 5 * cap, rtol=0, atol=TOL
    ):
        raise ValueError(
            "Layer 2 minute budgets or player caps do not match the environment"
        )

    recent_column = next(
        (column for column in ("min_mean_10", "roll_min") if column in role),
        None,
    )
    recent = (
        pd.to_numeric(role[recent_column], errors="coerce").to_numpy(dtype=float)
        if recent_column
        else np.full(len(role), 18.0)
    )
    recent = np.where(np.isfinite(recent), recent, 18.0)

    if np.any(recent < 0):
        raise ValueError("prior rolling minutes cannot be negative")

    starter = (
        _binary(role.is_starter, "is_starter")
        if "is_starter" in role
        else np.zeros(len(role), dtype=bool)
    )

    if starter.sum() > 5:
        raise ValueError("roster identifies more than five starters")

    if (
        not isinstance(max_variables, int)
        or isinstance(max_variables, bool)
        or max_variables <= 0
    ):
        raise ValueError("max_variables must be a positive integer")

    rng = np.random.default_rng(seed)
    output = []

    for index in range(len(env)):
        distances = np.array(
            [
                _distance(template, next(iter(teams)), margin[index], counts[index])
                for template in model.templates
            ]
        )
        weights = np.exp(-(distances - distances.min()))
        weights /= weights.sum()
        template = model.templates[int(rng.choice(len(weights), p=weights))]
        output.append(
            _arrange(
                _phases(template, int(counts[index]), rng),
                ids,
                target[index],
                recent,
                starter,
                rng,
                max_variables,
            )
        )

    return output
