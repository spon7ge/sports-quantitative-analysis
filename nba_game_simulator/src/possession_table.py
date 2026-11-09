"""One possession row for every time a team has the ball.

A possession starts when a team gains the ball and ends on a made field goal,
a defensive rebound, a turnover, a made last free throw of a shooting foul,
or the end of the period. An offensive rebound stays on the same possession.
Heaves are ignored: they are end-of-clock throws, not shot-mix attempts.
The free-throw lane tap (``Normal Rebound``) is not a live rebound.

Lineups begin as the five starters and follow substitutions. ``starters`` is
the five on the floor at the first row. For a full game that is the opening
lineup. When starters are omitted, the first five players who act identify
them. Play-by-play row order is the play order; action numbers are not.
If a player acts and is not in the five, he replaces the teammate who does
not appear again in that period. When more than one teammate could have
left, the quietest one is replaced and the rest of that period is flagged
``lineup_changed`` so rate estimates can drop it.
"""

from __future__ import annotations

import re
from math import hypot

import pandas as pd
from game_state.replay import name_keys

ZONES = ("rim", "mid", "corner3", "atb3")

# Restricted area. A two inside this distance is a rim attempt; the rest are mid.
RIM_DISTANCE_FEET = 4.0

# stats.nba.com yLegacy is tenths of a foot from the basket toward half court.
# The three-point line breaks at 9.2 feet, which is the corner.
CORNER_THREE_Y = 92.0

# A lead is garbage when it is larger than the points a team can erase at
# 0.7 per minute of game time, plus a 10-point cushion. Overtime uses only
# that period's clock, because another overtime is not guaranteed.
GARBAGE_CUSHION = 10.0
GARBAGE_POINTS_PER_MINUTE = 0.7

REQUIRED_COLUMNS = (
    "gameId",
    "actionNumber",
    "clock",
    "period",
    "teamId",
    "personId",
    "playerName",
    "teamTricode",
    "xLegacy",
    "yLegacy",
    "shotDistance",
    "scoreHome",
    "scoreAway",
    "location",
    "description",
    "actionType",
    "subType",
    "shotValue",
)

POSSESSION_COLUMNS = (
    "game_id",
    "possession_id",
    "period",
    "seconds_remaining",
    "offense_team_id",
    "defense_team_id",
    "off_1",
    "off_2",
    "off_3",
    "off_4",
    "off_5",
    "def_1",
    "def_2",
    "def_3",
    "def_4",
    "def_5",
    "margin",
    "garbage",
    "lineup_changed",
    "terminal",
    "oreb",
    "fga",
    "fgm",
    "rim_fga",
    "rim_fgm",
    "mid_fga",
    "mid_fgm",
    "corner3_fga",
    "corner3_fgm",
    "atb3_fga",
    "atb3_fgm",
    "shot_player_ids",
    "shot_zones",
    "shot_made",
    "tov",
    "tov_player_id",
    "shooting2_trips",
    "shooting3_trips",
    "and1_trips",
    "and1_zone",
    "bonus_trips",
    "fta_shooting",
    "ftm_shooting",
    "fta_and1",
    "ftm_and1",
    "fta_bonus",
    "ftm_bonus",
    "ft_kinds",
    "ft_player_ids",
    "ft_fta",
    "ft_ftm",
    "ft_zones",
)

_CLOCK = re.compile(r"PT(?:(\d+(?:\.\d+)?)M)?(\d+(?:\.\d+)?)S$", re.IGNORECASE)
_SUB = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)$", re.IGNORECASE)
_TIP = re.compile(r"Tip to (.+)$", re.IGNORECASE)
_FREE_THROW = re.compile(r"(\d+)\s+of\s+(\d+)", re.IGNORECASE)
_REBOUND = re.compile(r"Off:(\d+)\s+Def:(\d+)")
_ACTORS = {
    "made shot",
    "missed shot",
    "rebound",
    "turnover",
    "free throw",
    "foul",
    "heave",
}
_BONUS_FOULS = {
    "personal",
    "loose ball",
    "away from play",
    "personal take",
    "transition take",
    "double personal",
}


def shot_zone(shot_value, shot_distance, x_legacy, y_legacy) -> str:
    """Rim, mid, corner three, or above-the-break three.

    A shot distance of 0 means this feed left the distance blank. Two-point
    attempts then fall back to the shot coordinates, which are tenths of a foot.
    """

    if pd.isna(shot_value):
        raise ValueError("shot_value is missing")

    value = int(shot_value)
    if value not in (2, 3):
        raise ValueError(f"shot_value must be 2 or 3, got {shot_value!r}")

    if value == 3:
        if pd.isna(y_legacy):
            raise ValueError("a three-point shot needs yLegacy to separate the corner")
        return "corner3" if float(y_legacy) <= CORNER_THREE_Y else "atb3"

    distance = _distance_feet(shot_distance, x_legacy, y_legacy)
    return "rim" if distance <= RIM_DISTANCE_FEET else "mid"


def garbage_time(period: int, seconds_remaining: float, margin: int) -> bool:
    """True when the lead is too large, given the clock, for a competitive rate."""

    period = int(period)
    remaining = float(seconds_remaining)
    if period < 1:
        raise ValueError(f"period must be positive, got {period}")
    if remaining < 0:
        raise ValueError(f"seconds_remaining must be nonnegative, got {remaining}")

    if period >= 5:
        minutes_left = remaining / 60
    else:
        minutes_left = ((4 - period) * 720 + remaining) / 60

    lead = abs(int(margin))
    return lead > GARBAGE_CUSHION + GARBAGE_POINTS_PER_MINUTE * minutes_left


def possession_table(
    pbp: pd.DataFrame,
    starters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Parse play-by-play into one row per possession.

    ``starters`` has ``game_id``, ``team_id``, and ``player_id``. Five players
    per team. ``player_name`` is optional and is used to read substitutions.
    """

    frame = _prepare(pbp)
    starter_rows = _prepare_starters(starters)
    records: list[dict] = []

    for game_id, game in frame.groupby("gameId", sort=False):
        records.extend(_parse_game(str(game_id), game, starter_rows))

    if not records:
        return pd.DataFrame(columns=list(POSSESSION_COLUMNS))

    return pd.DataFrame.from_records(records, columns=list(POSSESSION_COLUMNS))


def _distance_feet(shot_distance, x_legacy, y_legacy) -> float:
    if not pd.isna(shot_distance) and float(shot_distance) > 0:
        return float(shot_distance)

    if pd.isna(x_legacy) or pd.isna(y_legacy):
        raise ValueError("a two-point shot needs shotDistance or coordinates")

    return hypot(float(x_legacy), float(y_legacy)) / 10


def _prepare(pbp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(pbp, pd.DataFrame) or pbp.empty:
        raise ValueError("play-by-play must be a non-empty DataFrame")

    missing = [column for column in REQUIRED_COLUMNS if column not in pbp.columns]
    if missing:
        raise ValueError(f"play-by-play needs {missing}")

    columns = list(REQUIRED_COLUMNS)
    if "playerNameI" in pbp.columns:
        # Box-score names are often just the last name. The initial form
        # ("T. Johnson") is what substitution text uses when two teammates share it.
        columns.append("playerNameI")

    frame = pbp.loc[:, columns].copy()
    frame["gameId"] = frame["gameId"].map(_game_id)
    frame["period"] = frame["period"].map(lambda value: int(value))
    frame["actionNumber"] = frame["actionNumber"].map(lambda value: int(value))
    # actionNumber is not play order. A substitution during free throws is
    # often numbered after the shot that follows it. The feed row order is
    # the order the plays happened.
    frame["_ord"] = range(len(frame))
    return frame.sort_values(["gameId", "_ord"], kind="mergesort").reset_index(
        drop=True
    )


def _prepare_starters(starters: pd.DataFrame | None) -> pd.DataFrame | None:
    if starters is None:
        return None

    if not isinstance(starters, pd.DataFrame) or starters.empty:
        raise ValueError("starters must be a non-empty DataFrame")

    frame = starters.copy()
    aliases = {
        "game_id": ("gameId", "GAME_ID"),
        "team_id": ("teamId", "TEAM_ID"),
        "player_id": ("personId", "PLAYER_ID", "playerId"),
        "player_name": ("playerName", "PLAYER_NAME"),
    }
    for target, names in aliases.items():
        if target not in frame.columns:
            for name in names:
                if name in frame.columns:
                    frame[target] = frame[name]
                    break

    required = ("game_id", "team_id", "player_id")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"starters need {missing}")

    frame["game_id"] = frame["game_id"].map(_game_id)
    frame["team_id"] = frame["team_id"].map(_as_int)
    frame["player_id"] = frame["player_id"].map(_as_int)
    if "player_name" not in frame.columns:
        frame["player_name"] = ""

    if frame["team_id"].isna().any() or frame["player_id"].isna().any():
        raise ValueError("starters have a missing team or player id")

    return frame


def _parse_game(
    game_id: str,
    game: pd.DataFrame,
    starters: pd.DataFrame | None,
) -> list[dict]:
    rows = list(game.itertuples(index=False))
    teams = _teams(rows, starters, game_id)
    home = _home_team(rows, teams, game_id)
    aliases = _aliases(rows, starters, game_id, teams)
    lineups = _opening_lineups(rows, starters, game_id, teams, aliases)
    by_period = _period_lineups(rows, teams, aliases, game_id, lineups)
    parser = _Parser(game_id, teams, home, lineups, aliases, by_period, rows)

    for index, row in enumerate(rows):
        parser.cursor = index
        parser.consume(row)

    parser.finish()
    return [
        _record(game_id, index, possession)
        for index, possession in enumerate(parser.done, start=1)
    ]


def _teams(rows, starters, game_id: str) -> set[int]:
    found: set[int] = set()

    if starters is not None:
        mine = starters.loc[starters.game_id == game_id]
        found.update(int(team) for team in mine.team_id)

    for row in rows:
        team = _as_int(row.teamId)
        if team is not None and _text(row.location).lower() in {"h", "v"}:
            found.add(team)

    if len(found) != 2:
        raise ValueError(f"{game_id} expected two teams, found {sorted(found)}")

    return found


def _home_team(rows, teams: set[int], game_id: str) -> int:
    homes = {
        team
        for row in rows
        if _text(row.location).lower() == "h" and (team := _as_int(row.teamId)) in teams
    }
    if len(homes) != 1:
        raise ValueError(f"{game_id} cannot resolve the home team")

    return next(iter(homes))


def _aliases(
    rows,
    starters,
    game_id: str,
    teams: set[int],
) -> dict[tuple[int, str], set[int]]:
    aliases: dict[tuple[int, str], set[int]] = {}

    def add(team: int, player: int, name: str) -> None:
        if team not in teams or not name:
            return
        for key in name_keys(name):
            aliases.setdefault((team, key), set()).add(player)

    for row in rows:
        player = _named_player(row)
        team = _as_int(row.teamId)
        if player is not None and team is not None:
            add(team, player, _text(row.playerName))
            add(team, player, _text(getattr(row, "playerNameI", "")))

    if starters is not None:
        mine = starters.loc[starters.game_id == game_id]
        columns = ["team_id", "player_id", "player_name"]
        for team, player, name in mine[columns].itertuples(index=False, name=None):
            add(int(team), int(player), _text(name))

    return aliases


def _opening_lineups(rows, starters, game_id, teams, aliases) -> dict[int, set[int]]:
    if starters is not None:
        mine = starters.loc[starters.game_id == game_id]
        if not mine.empty:
            lineups: dict[int, set[int]] = {}
            for team, players in mine.groupby("team_id"):
                ids = {int(player) for player in players.player_id}
                if len(ids) != 5:
                    raise ValueError(
                        f"{game_id} starters must be five players per team"
                    )
                lineups[int(team)] = ids

            if set(lineups) != teams:
                raise ValueError(f"{game_id} starters must cover both teams")

            return lineups

    return _infer_lineups(rows, teams, aliases, game_id)


def _period_lineups(
    rows,
    teams,
    aliases,
    game_id,
    opening,
) -> dict[int, dict[int, set[int]] | None]:
    """Lineup at the start of each period.

    Quarter breaks often have no substitution rows. Each later period is
    inferred on its own. If that period does not identify five players, the
    walk keeps whoever was on the floor.
    """

    periods = sorted({int(row.period) for row in rows})
    found: dict[int, dict[int, set[int]] | None] = {periods[0]: opening}

    for period in periods[1:]:
        period_rows = [row for row in rows if int(row.period) == period]
        try:
            found[period] = _infer_period(period_rows, teams, aliases, game_id)
        except ValueError:
            found[period] = None

    return found


def _infer_lineups(rows, teams, aliases, game_id: str) -> dict[int, set[int]]:
    """Opening five from who acts before checking in."""

    first_period = min(int(row.period) for row in rows)
    period_rows = [row for row in rows if int(row.period) == first_period]
    return _infer_period(period_rows, teams, aliases, game_id)


def _infer_period(period_rows, teams, aliases, game_id: str) -> dict[int, set[int]]:
    """Players substituted out, or who act before checking in, started the period."""

    lineups: dict[int, set[int]] = {}

    for team in teams:
        entered: set[int] = set()
        started: set[int] = set()

        for row in period_rows:
            if _as_int(row.teamId) != team:
                continue

            kind = _text(row.actionType).lower()
            sub = _text(row.subType).lower()
            if kind == "substitution":
                outgoing = _named_player(row)
                if (
                    outgoing is not None
                    and outgoing not in entered
                    and outgoing not in started
                ):
                    # A sixth name is someone who checked in, not a starter.
                    if len(started) < 5:
                        started.add(outgoing)
                    else:
                        entered.add(outgoing)
                # A call-up can appear only in the substitution text.
                incoming = _incoming_player(row, aliases, team)
                if incoming is not None:
                    entered.add(incoming)
                continue

            if "technical" in sub or "3 second" in sub:
                continue

            player = _named_player(row)
            if player is None or player in entered or kind not in _ACTORS:
                continue
            if player in started:
                continue
            if len(started) >= 5:
                entered.add(player)
                continue

            started.add(player)

        if len(started) != 5:
            raise ValueError(f"{game_id} could not infer five players for team {team}")

        lineups[team] = started

    return lineups


class _Possession:
    def __init__(
        self,
        offense,
        defense,
        clock,
        period,
        margin,
        offense_lineup,
        defense_lineup,
    ):
        self.offense = offense
        self.defense = defense
        self.seconds_remaining = clock
        self.period = period
        self.margin = margin
        self.offense_lineup = tuple(sorted(offense_lineup))
        self.defense_lineup = tuple(sorted(defense_lineup))
        self.shots: list[tuple[int, str, bool]] = []
        self.free_throws: list[tuple[str, int, int, int, str | None]] = []
        self.oreb = 0
        self.turnover = False
        self.tov_player: int | None = None
        self.lineup_changed = False
        self.terminal: str | None = None

    @property
    def has_events(self) -> bool:
        return bool(self.shots or self.free_throws or self.turnover or self.oreb)


class _Parser:
    def __init__(self, game_id, teams, home, lineups, aliases, period_lineups, rows):
        self.game_id = game_id
        self.teams = teams
        self.home = home
        self.lineups = {team: set(players) for team, players in lineups.items()}
        self.aliases = aliases
        self.period_lineups = period_lineups
        self.rows = rows
        self.cursor = 0
        self.last_seen: dict[int, int] = {}
        self.uncertain_period: int | None = None
        self.tracked_period: int | None = None
        self.score_home = 0
        self.score_away = 0
        self.possession: _Possession | None = None
        self.pending_make = False
        self.pending_clock = 0.0
        self.pending_period = 1
        self.foul_side: str | None = None
        self.and1_zone: str | None = None
        self.free_throw: dict | None = None
        self.pending_subs: list = []
        self.await_rebound = False
        self.rebound_totals: dict[int, tuple[int, int]] = {}
        self.done: list[_Possession] = []
        self.current = None

    def consume(self, row) -> None:
        self.current = row
        self._enter_period(int(row.period))
        kind = _text(row.actionType).lower()
        if kind not in {"made shot", "free throw"} or _missed(row):
            self._adopt_score(row)

        if not kind:
            return

        sub = _text(row.subType).lower()
        if kind == "substitution":
            self._flush_make()
            self._substitute(row)
        elif kind == "foul" and self._is_and_one(row):
            self._mark_and_one()
        elif kind == "foul":
            self._flush_make()
            self._foul(row, sub)
        elif kind in {"made shot", "missed shot"}:
            self._flush_make()
            self._shot(row, made=kind == "made shot")
        elif kind == "free throw":
            self._free_throw(row, sub)
        elif kind == "rebound":
            self._flush_make()
            self._rebound(row, sub)
        elif kind == "turnover":
            self._flush_make()
            self._turnover(row)
        elif kind == "jump ball":
            self._flush_make()
            self._jump(row)
        elif kind == "period" and sub == "end":
            self._period_end(row)
        elif kind in {"timeout", "violation", "instant replay", "ejection", "heave"}:
            self._flush_make()

    def finish(self) -> None:
        if self.pending_make:
            self._flush_make()
        if self.possession is not None and self.possession.has_events:
            self._close(
                "period_end",
                None,
                self.possession.seconds_remaining,
                self.possession.period,
            )

    def _shot(self, row, *, made: bool) -> None:
        self._abandon_free_throw()
        self._break_unfinished_foul()
        # The rebound row is not always present. The next shot says who has the ball.
        self.await_rebound = False

        team = self._require_team(row)
        player = self._require_player(row, "shot")
        self._ensure(team, row)
        self._on_court(team, player, "shooter")
        zone = shot_zone(row.shotValue, row.shotDistance, row.xLegacy, row.yLegacy)
        assert self.possession is not None
        self.possession.shots.append((player, zone, made))

        if made:
            self._adopt_score(row)
            self.pending_make = True
            self.pending_clock = _clock_seconds(row.clock)
            self.pending_period = int(row.period)
        else:
            self.await_rebound = True

    def _foul(self, row, sub: str) -> None:
        self._abandon_free_throw()
        # Bench technicals are credited to the club, not to a player on the floor.
        if "technical" in sub or "flagrant" in sub or "bench" in sub:
            return
        if sub in {"offensive", "offensive charge"}:
            return

        team = self._require_team(row)
        if self.possession is None:
            if sub == "shooting":
                self._start(self._other(team), row)
                self.foul_side = "shooting"
            return

        if team == self.possession.offense:
            return

        if sub == "shooting":
            self.foul_side = "shooting"
            self.and1_zone = None
        elif sub in _BONUS_FOULS:
            self.foul_side = "bonus"

    def _is_and_one(self, row) -> bool:
        if not self.pending_make or self.possession is None:
            return False
        if _text(row.subType).lower() != "shooting":
            return False

        team = _as_int(row.teamId)
        return team is not None and team != self.possession.offense

    def _mark_and_one(self) -> None:
        assert self.possession is not None
        self.pending_make = False
        self.and1_zone = self.possession.shots[-1][1]
        self.foul_side = "shooting"

    def _free_throw(self, row, sub: str) -> None:
        if "technical" in sub:
            self._adopt_score(row)
            return

        parsed = _FREE_THROW.search(sub)
        if parsed is None:
            raise ValueError(f"{self.game_id} unrecognized free throw {sub!r}")

        number, total = int(parsed.group(1)), int(parsed.group(2))
        team = self._require_team(row)
        player = self._require_player(row, "free throw")
        self._ensure(team, row)
        self._on_court(team, player, "free throw shooter")
        kind = self._free_throw_kind(sub, total)
        zone = self.and1_zone if kind == "and1" else None

        if self.free_throw is None:
            self.free_throw = {
                "kind": kind,
                "player": player,
                "fta": 0,
                "ftm": 0,
                "zone": zone,
            }
        elif self.free_throw["player"] != player or self.free_throw["kind"] != kind:
            raise ValueError(f"{self.game_id} free-throw trip changed shooters")

        made = not _missed(row)
        self.free_throw["fta"] += 1
        if made:
            self.free_throw["ftm"] += 1

        self._adopt_score(row)
        if number != total:
            return

        self._finish_free_throw(row, made)

    def _free_throw_kind(self, sub: str, total: int) -> str:
        if "flagrant" in sub or "clear path" in sub:
            return "other"
        if self.and1_zone is not None and total == 1:
            return "and1"
        if self.foul_side == "shooting" and total == 2:
            return "shooting2"
        if self.foul_side == "shooting" and total == 3:
            return "shooting3"
        if self.foul_side == "bonus":
            return "bonus"
        if self.and1_zone is not None:
            return "and1"
        return "other"

    def _finish_free_throw(self, row, made: bool) -> None:
        assert self.possession is not None and self.free_throw is not None
        trip = self.free_throw
        self.possession.free_throws.append(
            (trip["kind"], trip["player"], trip["fta"], trip["ftm"], trip["zone"])
        )
        self.free_throw = None
        self.foul_side = None
        self.and1_zone = None
        subs_changed = self._apply_pending_subs()
        kind = trip["kind"]
        shooting_make = kind in {"shooting2", "shooting3"} and made
        ends = shooting_make or (kind == "and1" and made)

        if ends:
            terminal = "make" if kind == "and1" else "shooting_ft"
            self._close(
                terminal,
                self._other(self.possession.offense),
                _clock_seconds(row.clock),
                int(row.period),
            )
            return

        if subs_changed and self.possession is not None and self.possession.has_events:
            self.possession.lineup_changed = True

        if not made:
            # A missed last free throw is live. The rebound decides who keeps it.
            self.await_rebound = True

    def _rebound(self, row, sub: str) -> None:
        if sub == "normal rebound":
            return

        self._abandon_free_throw()
        self._break_unfinished_foul()
        side = self._rebound_side(row)
        if side == "ignore":
            return
        if self.possession is None:
            raise ValueError(f"{self.game_id} rebound with nobody on offense")

        self.await_rebound = False
        if side == "oreb":
            player = _named_player(row)
            if player is not None:
                self._on_court(self.possession.offense, player, "offensive rebounder")
            self.possession.oreb += 1
            return

        self._close(
            "miss",
            self._rebound_team(row),
            _clock_seconds(row.clock),
            int(row.period),
        )

    def _rebound_side(self, row) -> str:
        player = _named_player(row)
        description = _text(row.description)
        match = _REBOUND.search(description) if player is not None else None

        if player is not None and match is not None:
            offensive = int(match.group(1))
            defensive = int(match.group(2))
            previous = self.rebound_totals.get(player, (0, 0))
            self.rebound_totals[player] = (offensive, defensive)
            if offensive <= previous[0] and defensive <= previous[1]:
                # Repeated line, or the feed corrected the running total downward.
                return "ignore"
            if offensive > previous[0] and defensive == previous[1]:
                return "oreb"
            if defensive > previous[1] and offensive == previous[0]:
                return "dreb"
            # Both counters moved. One of them is a correction. The player who
            # is listed on the rebound plays for the team that grabbed it.
            team = _as_int(row.teamId)
            if self.possession is not None and team == self.possession.offense:
                return "oreb"
            if self.possession is not None and team == self.possession.defense:
                return "dreb"
            raise ValueError(
                f"{self.game_id} rebound totals for {player} did not move cleanly"
            )

        if self.possession is None:
            return "ignore"

        team = self._rebound_team(row)
        if team == self.possession.offense:
            return "oreb"
        if team == self.possession.defense:
            return "dreb"

        raise ValueError(f"{self.game_id} rebound team {team} is not on the floor")

    def _rebound_team(self, row) -> int:
        person = _as_int(row.personId)
        team = _as_int(row.teamId)
        # Team rebounds store the club id in personId and leave teamId at 0.
        if person in self.teams:
            return int(person)
        if team in self.teams:
            return int(team)

        raise ValueError(f"{self.game_id} rebound has no team")

    def _turnover(self, row) -> None:
        self._abandon_free_throw()
        self._break_unfinished_foul()
        team = self._turnover_team(row)
        if team is None:
            raise ValueError(f"{self.game_id} turnover without a team")

        self._ensure(team, row)
        player = _named_player(row)
        if player is not None:
            self._on_court(int(team), player, "turnover")

        assert self.possession is not None
        self.possession.turnover = True
        self.possession.tov_player = player
        self._close(
            "turnover",
            self._other(team),
            _clock_seconds(row.clock),
            int(row.period),
        )

    def _turnover_team(self, row) -> int | None:
        team = _as_int(row.teamId)
        if team in self.teams:
            return int(team)
        # Shot-clock and five-second turnovers leave teamId at 0 and store
        # the club id in personId, the same way a team rebound does.
        person = _as_int(row.personId)
        if person in self.teams:
            return int(person)
        if self.possession is not None:
            return self.possession.offense
        return None

    def _jump(self, row) -> None:
        self._abandon_free_throw()
        self._break_unfinished_foul()
        match = _TIP.search(_text(row.description))
        if match is None:
            return

        found = _resolve_name(match.group(1).strip(), self.aliases)
        if found is None:
            return

        winner, _player = found
        if self.possession is not None and self.possession.offense == winner:
            return

        self._close(
            "jump",
            winner,
            _clock_seconds(row.clock),
            int(row.period),
        )

    def _substitute(self, row) -> None:
        if self.free_throw is not None:
            # Subs between free throws take the floor for the next live ball.
            self.pending_subs.append(row)
            return

        self._apply_sub(row)
        self._sync_open_lineup()

    def _period_end(self, row) -> None:
        self._abandon_free_throw()
        if self.pending_make:
            self._flush_make()
        if self.possession is not None and self.possession.has_events:
            self._close("period_end", None, _clock_seconds(row.clock), int(row.period))
        else:
            self.possession = None
            self.pending_make = False

    def _flush_make(self) -> None:
        if not self.pending_make or self.possession is None:
            self.pending_make = False
            return

        self.pending_make = False
        self._close(
            "make",
            self._other(self.possession.offense),
            self.pending_clock,
            self.pending_period,
        )

    def _close(self, terminal: str, nxt: int | None, clock: float, period: int) -> None:
        current = self.possession
        self.possession = None
        self.pending_make = False
        self.await_rebound = False
        if current is not None and current.has_events:
            current.terminal = terminal
            self.done.append(current)

        if nxt is not None:
            self._start_at(nxt, clock, period)

    def _ensure(self, team: int, row) -> None:
        if self.possession is None:
            self._start(team, row)
            return

        if self.possession.offense != team:
            # A made bonus free throw should be followed by an inbound for the
            # same team. The feed sometimes skips the turnover that gives the
            # ball away. The event in hand is the evidence of who has it.
            if self.possession.has_events:
                self._close("change", None, _clock_seconds(row.clock), int(row.period))
            else:
                self.possession = None
            self._start(team, row)

    def _enter_period(self, period: int) -> None:
        if self.tracked_period == period:
            return

        self._abandon_free_throw()
        if self.pending_make:
            self._flush_make()
        if self.possession is not None and self.possession.has_events:
            self._close("period_end", None, 0.0, period)
        else:
            self.possession = None
            self.pending_make = False

        inferred = self.period_lineups.get(period)
        if inferred:
            self.lineups = {
                team: set(players) for team, players in inferred.items()
            }

        self.uncertain_period = None
        self.last_seen = {}
        self.tracked_period = period

    def _start(self, team: int, row) -> None:
        self._start_at(team, _clock_seconds(row.clock), int(row.period))

    def _start_at(self, team: int, clock: float, period: int) -> None:
        if self.possession is not None:
            raise RuntimeError(
                f"{self.game_id} opened a possession before the previous one closed"
            )

        defense = self._other(team)
        offense_lineup = self.lineups[team]
        defense_lineup = self.lineups[defense]
        if len(offense_lineup) != 5 or len(defense_lineup) != 5:
            raise ValueError(f"{self.game_id} lineup does not have five players")

        self.possession = _Possession(
            team,
            defense,
            clock,
            period,
            self._margin(team),
            offense_lineup,
            defense_lineup,
        )
        if self.uncertain_period == period:
            self.possession.lineup_changed = True
        self.await_rebound = False

    def _sync_open_lineup(self) -> None:
        current = self.possession
        if current is None:
            return
        if current.has_events:
            current.lineup_changed = True
            return

        current.offense_lineup = tuple(sorted(self.lineups[current.offense]))
        current.defense_lineup = tuple(sorted(self.lineups[current.defense]))

    def _apply_pending_subs(self) -> bool:
        subs = self.pending_subs
        self.pending_subs = []
        for row in subs:
            self._apply_sub(row)
        return bool(subs)

    def _apply_sub(self, row) -> None:
        team = self._require_team(row)
        outgoing = self._require_player(row, "substitution")
        match = _SUB.search(_text(row.description))
        if match is None:
            raise ValueError(
                f"{self.game_id} unrecognized substitution {_text(row.description)!r}"
            )

        found = _resolve_name(match.group(1).strip(), self.aliases, team)
        if found is None:
            return

        incoming = found[1]
        lineup = self.lineups[team]
        if outgoing not in lineup or incoming in lineup:
            return

        lineup.remove(outgoing)
        lineup.add(incoming)

    def _break_unfinished_foul(self) -> None:
        if self.free_throw is not None:
            return
        self.foul_side = None
        self.and1_zone = None

    def _abandon_free_throw(self) -> None:
        """Keep the free throws already shot when the rest of the trip is missing.

        A lane violation, or the next live play, ends the trip without the
        last free throw. The attempts still belong to this possession.
        """

        trip = self.free_throw
        if trip is None:
            return

        current = self.possession
        if current is not None:
            current.free_throws.append(
                (trip["kind"], trip["player"], trip["fta"], trip["ftm"], trip["zone"])
            )
        self.free_throw = None
        self.foul_side = None
        self.and1_zone = None
        if self._apply_pending_subs() and current is not None and current.has_events:
            current.lineup_changed = True
        else:
            self._sync_open_lineup()

    def _on_court(self, team: int, player: int, role: str) -> None:
        if player not in self.lineups[team]:
            self._seat(team, player, role)
        self.last_seen[player] = self.cursor

    def _seat(self, team: int, player: int, role: str) -> None:
        """Put an acting player on the floor.

        The feed sometimes omits the substitution. The teammate who never
        appears again in the period is the one who left. If several teammates
        are quiet, the one who has gone longest without an event leaves, and
        the rest of the period is marked so rates can ignore it.
        """

        lineup = self.lineups[team]
        gone = [mate for mate in lineup if not self._returns(team, mate)]
        uncertain = len(gone) != 1
        pool = gone if gone else list(lineup)
        removed = (
            gone[0]
            if not uncertain
            else min(pool, key=lambda mate: (self.last_seen.get(mate, -1), mate))
        )
        if uncertain:
            self.uncertain_period = int(self.current.period)
        lineup.remove(removed)
        lineup.add(player)
        self._rewrite_open_lineup(removed, uncertain=uncertain)

    def _returns(self, team: int, player: int) -> bool:
        period = int(self.current.period)
        for row in self.rows[self.cursor + 1 :]:
            if int(row.period) != period:
                break
            if _as_int(row.teamId) != team:
                continue

            kind = _text(row.actionType).lower()
            if _named_player(row) != player:
                continue
            if kind == "substitution":
                return True
            sub = _text(row.subType).lower()
            if kind in _ACTORS and "technical" not in sub and "3 second" not in sub:
                return True

        return False

    def _rewrite_open_lineup(self, removed: int, *, uncertain: bool) -> None:
        current = self.possession
        if current is None:
            return

        actors = {player for player, _zone, _made in current.shots}
        actors.update(trip[1] for trip in current.free_throws)
        if current.tov_player is not None:
            actors.add(current.tov_player)
        if uncertain or removed in actors:
            current.lineup_changed = True

        current.offense_lineup = tuple(sorted(self.lineups[current.offense]))
        current.defense_lineup = tuple(sorted(self.lineups[current.defense]))

    def _adopt_score(self, row) -> None:
        parsed = _score(row)
        if parsed is not None:
            self.score_home, self.score_away = parsed

    def _margin(self, offense: int) -> int:
        if offense == self.home:
            return self.score_home - self.score_away
        return self.score_away - self.score_home

    def _other(self, team: int) -> int:
        remaining = self.teams - {team}
        if len(remaining) != 1:
            raise ValueError(f"{self.game_id} cannot find the opponent of {team}")
        return next(iter(remaining))

    def _require_team(self, row) -> int:
        team = _as_int(row.teamId)
        if team not in self.teams:
            raise ValueError(f"{self.game_id} event has no team")
        return int(team)

    def _require_player(self, row, role: str) -> int:
        player = _named_player(row)
        if player is None:
            raise ValueError(f"{self.game_id} {role} has no player")
        return player


def _record(game_id: str, possession_id: int, possession: _Possession) -> dict:
    shots = possession.shots
    zones = [zone for _player, zone, _made in shots]
    makes = [made for _player, _zone, made in shots]
    trips = possession.free_throws
    and_ones = [trip for trip in trips if trip[0] == "and1"]
    row = {
        "game_id": game_id,
        "possession_id": possession_id,
        "period": possession.period,
        "seconds_remaining": possession.seconds_remaining,
        "offense_team_id": possession.offense,
        "defense_team_id": possession.defense,
        "margin": possession.margin,
        "garbage": garbage_time(
            possession.period,
            possession.seconds_remaining,
            possession.margin,
        ),
        "lineup_changed": possession.lineup_changed,
        "terminal": possession.terminal,
        "oreb": possession.oreb,
        "fga": len(shots),
        "fgm": sum(makes),
        "shot_player_ids": tuple(player for player, _zone, _made in shots),
        "shot_zones": tuple(zones),
        "shot_made": tuple(makes),
        "tov": int(possession.turnover),
        "tov_player_id": possession.tov_player,
        "shooting2_trips": _trip_count(trips, "shooting2"),
        "shooting3_trips": _trip_count(trips, "shooting3"),
        "and1_trips": len(and_ones),
        "and1_zone": and_ones[0][4] if len(and_ones) == 1 else None,
        "bonus_trips": _trip_count(trips, "bonus"),
        "fta_shooting": _trip_sum(trips, {"shooting2", "shooting3"}, 2),
        "ftm_shooting": _trip_sum(trips, {"shooting2", "shooting3"}, 3),
        "fta_and1": _trip_sum(trips, {"and1"}, 2),
        "ftm_and1": _trip_sum(trips, {"and1"}, 3),
        "fta_bonus": _trip_sum(trips, {"bonus"}, 2),
        "ftm_bonus": _trip_sum(trips, {"bonus"}, 3),
        "ft_kinds": tuple(trip[0] for trip in trips),
        "ft_player_ids": tuple(trip[1] for trip in trips),
        "ft_fta": tuple(trip[2] for trip in trips),
        "ft_ftm": tuple(trip[3] for trip in trips),
        "ft_zones": tuple(trip[4] for trip in trips),
    }

    for index, player in enumerate(possession.offense_lineup, start=1):
        row[f"off_{index}"] = player
    for index, player in enumerate(possession.defense_lineup, start=1):
        row[f"def_{index}"] = player

    for zone in ZONES:
        row[f"{zone}_fga"] = sum(
            1 for name in zones if name == zone
        )
        row[f"{zone}_fgm"] = sum(
            1
            for name, made in zip(zones, makes, strict=True)
            if name == zone and made
        )

    return row


def _trip_count(trips, kind: str) -> int:
    return sum(1 for trip in trips if trip[0] == kind)


def _trip_sum(trips, kinds: set[str], field: int) -> int:
    return sum(trip[field] for trip in trips if trip[0] in kinds)


def _incoming_player(row, aliases, team: int) -> int | None:
    match = _SUB.search(_text(row.description))
    if match is None:
        return None
    found = _resolve_name(match.group(1).strip(), aliases, team)
    if found is None:
        return None
    return found[1]


def _resolve_name(
    name: str,
    aliases,
    team: int | None = None,
) -> tuple[int, int] | None:
    for key in name_keys(name):
        hits: list[tuple[int, int]] = []
        for (alias_team, alias_key), players in aliases.items():
            if alias_key != key or (team is not None and alias_team != team):
                continue
            hits.extend((alias_team, player) for player in players)

        unique = list(dict.fromkeys(hits))
        if len(unique) == 1:
            return unique[0]

    return None


def _named_player(row) -> int | None:
    if not _text(row.playerName):
        return None
    return _as_int(row.personId)


def _missed(row) -> bool:
    return _text(row.description).lstrip().upper().startswith("MISS")


def _score(row) -> tuple[int, int] | None:
    home = _text(row.scoreHome)
    away = _text(row.scoreAway)
    if not home or not away:
        return None
    return int(float(home)), int(float(away))


def _clock_seconds(value) -> float:
    text = _text(value)
    match = _CLOCK.fullmatch(text)
    if match:
        return 60 * float(match.group(1) or 0) + float(match.group(2))

    if ":" in text:
        minute, second = text.split(":", 1)
        return 60 * float(minute) + float(second)

    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"unrecognized clock {value!r}") from error


def _game_id(value) -> str:
    text = _text(value)
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(10)
    return text


def _as_int(value) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text or text in {"0", "nan", "None"}:
        return None

    return int(text)


def _text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()
