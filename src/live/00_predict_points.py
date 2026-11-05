import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.minutes import add_starter_features
from src.features.points import add_points_features
from src.features.points.pregame import OUTCOME_COLUMNS
from src.live.helpers import findOpp
from src.live.nba_lineup import projected_lineups
from src.pipeline.silver.positions import player_name_keys
from src.pipeline.silver.rotowire import normalize_team_abbreviation
from src.scrapers.arbitrage import canonical_team

# Same list as PTS_FEATURES in notebooks/nba/points/00_points_xgboost.ipynb.
FEATURES = [
    # point context
    "pts_mean_10",
    "pts_std_10",
    "pts_lag_1",
    "pts_trend",
    "pts_rank_10",

    # how they score
    "two_pa_rolling_10",
    "two_pct_smoothed",

    "fg3a_rolling_10",
    "three_pct_smoothed",

    "fta_rolling_10",
    "ft_pct_smoothed",

    # minutes context
    "min_mean_10",
    "min_mean_3_minus_10",
    "min_rank_10",
    "min_sd_10",

    # efficiency context
    "ts_x_usg",
    "usg_rolling_10",
    "usg_trend",
    "usg_rank_10",

    # game context
    "starting",
    "starter_rank",
    "start_rate_10",

    "opp_def_rating_mean_10",
    "expected_pace",
    "implied_team_total",
    "player_team_spread",
]

_REQUIRED = (
    "player_id",
    "player_name",
    "team_id",
    "team_abbreviation",
    "game_id",
    "game_date",
    "season_year",
    "minutes",
)
# The slate row carries these in. The feature builders recompute the rest.
_SLATE_INPUTS = frozenset({"starting", "player_team_spread"})
_TEXT_BLANK = frozenset({"start_position", "comment"})

_PINNACLE_DIR = (Path(__file__).resolve().parents[2] / "data" / "props" / "pinnacle" / "nba")
_LINEUP_DIR = (Path(__file__).resolve().parents[2] / "data" / "props" / "rotowire_nba_lineup")
log = logging.getLogger(__name__)


def points_pipeline(data, name, date, games=None, lineups=None):
    """One feature row for ``name`` on ``date``, in ``FEATURES`` order.

    Thin wrapper over ``build_points_live`` for a single player.
    """
    live = build_points_live(
        data,
        [name],
        date,
        games=games,
        lineups=lineups,
        extras=(),
    )
    return live.loc[name, FEATURES]


def build_points_live(
    history,
    names,
    date,
    *,
    games=None,
    lineups=None,
    extras=("pts_lag_1",),
):
    """Pregame feature rows for every named player on one slate date.

    Live betting path: history strictly before ``date``, opponent and
    team totals from the Pinnacle board when that game is posted,
    ``starting`` from the posted five, and one shared ``game_id`` per
    matchup so teammate ranks and opponent pace/defense resolve. The
    schedule API is only a fallback when Pinnacle has no participants.

    The feature builders shift by one game. The blank slate row takes
    that shift, so its windows cover every game already played,
    including the most recent one.
    """
    board_names = list(dict.fromkeys(str(name) for name in names))
    if not board_names:
        raise ValueError("names")
    day = str(date)[:10]
    full = _games_before(history, date)
    missing = [column for column in _REQUIRED if column not in full.columns]
    if missing:
        raise KeyError(f"player history needs {missing}")
    full = full.sort_values(
        ["player_id", "game_date", "game_id"],
        kind="mergesort",
    )
    if games is None:
        games = pinnacle_games_on(day)
        if not games:
            games = _load_pinnacle_games()
    if lineups is None:
        lineups = lineups_on(day)
        if not lineups:
            lineups = projected_lineups()

    last = full.groupby("player_id", sort=False).tail(1)
    specs = []
    for board_name in board_names:
        hist_name = _history_player_name(full, board_name)
        if hist_name is None:
            log.info("no history for board name %r on %s; skipping", board_name, day)
            continue
        specs.append(
            _player_slate_spec(
                full,
                last,
                hist_name,
                day,
                games,
                board_name=board_name,
            )
        )
    if not specs:
        raise ValueError(f"no history for any of {board_names!r} before {day}")
    panel = _append_matchup_slates(full, last, specs, day=day, lineups=lineups)
    featured = _live_features(panel)
    columns = list(FEATURES) + [column for column in extras if column not in FEATURES]
    rows = []
    for spec in specs:
        slate = featured.loc[
            featured["closing_row"].eq(1)
            & featured["player_id"].eq(spec["player_id"])
        ]
        if slate.empty:
            raise ValueError(f"no feature row for {spec['board_name']!r} on {day}")
        row = slate.iloc[-1]
        missing_cols = [column for column in columns if column not in row.index]
        if missing_cols:
            raise KeyError(f"feature builder missing {missing_cols}")
        rows.append(
            pd.Series(
                {column: row[column] for column in columns},
                name=spec["board_name"],
            )
        )
    return pd.DataFrame(rows)[columns]


def _games_before(frame, date):
    """Gamelog rows strictly before ``date``. The slate itself stays out."""
    cutoff = pd.Timestamp(str(date)[:10]).normalize()
    played = pd.to_datetime(frame["game_date"], errors="coerce", utc=True)
    played = played.dt.tz_convert(None).dt.normalize()
    kept = frame.loc[played < cutoff]
    dropped = len(frame) - len(kept)
    if dropped:
        log.info(
            "left out %s gamelog rows on or after %s",
            dropped,
            cutoff.date().isoformat(),
        )
    return kept


def pinnacle_games_on(date, directory=None):
    """Team-market snapshots saved on ``date``, newest file first.

    ``points_pipeline`` keeps the first spread it finds for a team, so
    the newest file is the line that wins.
    """
    folder = _PINNACLE_DIR if directory is None else Path(directory)
    day = str(date)[:10]
    files = sorted(folder.glob(f"pinnacle_nba_{day}_*_team.json"))
    games = []
    for path in reversed(files):
        payload = json.loads(path.read_text())
        games.extend(payload.get("games") or [])
    return games


def lineups_on(date, directory=None):
    """Starting fives from RotoWire files saved on ``date``.

    The newest file wins for a team. A team that has already tipped
    drops off the later page, so an earlier file that day still fills
    that five.
    """
    folder = _LINEUP_DIR if directory is None else Path(directory)
    day = str(date)[:10]
    files = sorted(folder.glob(f"rotowire_nba_{day}_*.json"))
    lineups = {}
    for path in reversed(files):
        payload = json.loads(path.read_text())
        for game in payload.get("games") or []:
            for side in ("away", "home"):
                block = game.get(side) or {}
                team = block.get("team")
                names = [
                    str(player["name"])
                    for player in block.get("lineup") or []
                    if player.get("name")
                ]
                if not team or not names:
                    continue
                abbr = normalize_team_abbreviation(team)
                lineups.setdefault(abbr, names)
    return lineups


def _season_label(date):
    year, month, _day = (int(part) for part in str(date)[:10].split("-"))
    start = year if month >= 10 else year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _history_player_name(history, board_name):
    """Map a book name onto the gamelog name (Jr./III suffixes, etc.)."""
    names = history["player_name"]
    if names.eq(board_name).any():
        return board_name
    wanted = set(player_name_keys(board_name))
    if not wanted:
        return None
    for hist_name in names.dropna().unique():
        if wanted & set(player_name_keys(str(hist_name))):
            return str(hist_name)
    return None


def _player_slate_spec(history, last, name, date, games, *, board_name=None):
    player = history.loc[history["player_name"] == name]
    if player.empty:
        raise ValueError(f"no games for {name!r} before {date}")
    team_abbr = normalize_team_abbreviation(player["team_abbreviation"].iloc[-1])
    team_id = player["team_id"].iloc[-1]
    player_id = player["player_id"].iloc[-1]
    matchup = _matchup_from_games(games, team_abbr, history)
    if matchup is None:
        opp_abbr, _is_home = findOpp(name, player, date, max_days_ahead=3, league="nba")
        spread, total = _pinnacle_lines(games, team_abbr)
    else:
        opp_abbr, _is_home, spread, total = matchup
    if pd.isna(spread) or pd.isna(total):
        log.info("no Pinnacle spread or total for %s on %s", team_abbr, date)
    opp_team_id = _team_id_for(history, opp_abbr)
    if opp_team_id is None:
        log.info(
            "no opponent id for %s on %s; opponent features stay missing",
            board_name or name,
            date,
        )
    if last.loc[last["team_id"].eq(team_id)].empty:
        raise ValueError(f"no roster rows for team {team_id!r}")
    return {
        "board_name": board_name or name,
        "player_name": name,
        "player_id": player_id,
        "team_id": team_id,
        "team_abbr": team_abbr,
        "opp_team_id": opp_team_id,
        "spread": spread,
        "total": total,
    }


def _matchup_from_games(games, team_abbr, history):
    """Opponent, home flag, spread, and total from a Pinnacle team board."""
    team_can = canonical_team(team_abbr, "nba")
    for game in games:
        parts = game.get("participants") or []
        if len(parts) < 2:
            continue
        labels = [canonical_team(part, "nba") for part in parts]
        if team_can not in labels:
            continue
        markets = game.get("team_markets") or {}
        spread = _main_line(
            markets.get("spread") or [],
            team_can,
            side=None,
            participants=parts,
        )
        total = _main_line(
            markets.get("total") or [],
            team_id=None,
            side="over",
            participants=parts,
        )
        is_home = 1.0 if labels[1] == team_can else 0.0
        opp_can = labels[1] if labels[0] == team_can else labels[0]
        opp_abbr = _abbr_for_canonical(history, opp_can)
        if opp_abbr is None:
            continue
        return opp_abbr, is_home, spread, total
    return None


def _abbr_for_canonical(frame, team_can):
    if "team_abbreviation" not in frame.columns:
        return None
    for abbr in frame["team_abbreviation"].dropna().unique():
        normalized = normalize_team_abbreviation(abbr)
        if canonical_team(normalized, "nba") == team_can:
            return normalized
    return None


def _append_matchup_slates(history, last, specs, *, day, lineups):
    """Blank slate games grouped by matchup; both sides share one game id."""
    prepared = _drop_stale_features(_ensure_starting(history))
    prepared["closing_row"] = 0
    ids = pd.to_numeric(prepared["game_id"], errors="coerce")
    next_id = 1 if ids.isna().all() else int(ids.max()) + 1
    groups: dict[frozenset, list[dict]] = {}
    for spec in specs:
        if spec["opp_team_id"] is None or pd.isna(spec["opp_team_id"]):
            key = frozenset({("team", spec["team_id"]), ("solo", spec["player_id"])})
        else:
            key = frozenset({spec["team_id"], spec["opp_team_id"]})
        groups.setdefault(key, []).append(spec)

    blocks = []
    for group in groups.values():
        game_id = next_id
        next_id += 1
        teams: dict[object, dict] = {}
        for spec in group:
            teams.setdefault(
                spec["team_id"],
                {
                    "team_abbr": spec["team_abbr"],
                    "opp_team_id": spec["opp_team_id"],
                    "spread": spec["spread"],
                    "total": spec["total"],
                },
            )
        for team_id, info in teams.items():
            mates = last.loc[last["team_id"].eq(team_id)]
            five = _posted_five(lineups, info["team_abbr"])
            if five is None:
                log.info(
                    "no posted five for %s on %s; starting stays missing",
                    info["team_abbr"],
                    day,
                )
            blocks.append(
                _slate_block(
                    mates,
                    game_id=game_id,
                    date=day,
                    opp_team_id=info["opp_team_id"],
                    spread=info["spread"],
                    total=info["total"],
                    starting=_starting_flags(mates, five),
                )
            )
        # One foe row when only one side of the board was requested.
        for team_id, info in list(teams.items()):
            opp_id = info["opp_team_id"]
            if opp_id is None or pd.isna(opp_id) or opp_id in teams:
                continue
            foes = last.loc[last["team_id"].eq(opp_id)]
            if foes.empty:
                continue
            blocks.append(
                _slate_block(
                    foes.iloc[[0]],
                    game_id=game_id,
                    date=day,
                    opp_team_id=team_id,
                    spread=np.nan,
                    total=np.nan,
                    starting=np.nan,
                )
            )
    if not blocks:
        raise ValueError("no slate rows")
    slate = pd.concat(blocks, ignore_index=True)
    slate["closing_row"] = 1
    return pd.concat([prepared, slate], ignore_index=True)


def _slate_block(rows, *, game_id, date, opp_team_id, spread, total, starting):
    blanked = _blank_outcomes(rows)
    blanked["game_id"] = game_id
    # Naive calendar date; feature builders normalize mixed history formats.
    blanked["game_date"] = pd.Timestamp(str(date)[:10])
    blanked["season_year"] = _season_label(date)
    blanked["opp_team_id"] = opp_team_id
    blanked["player_team_spread"] = spread
    blanked["game_total"] = total
    blanked["starting"] = starting
    return blanked


def _blank_outcomes(rows):
    result = rows.copy()
    for column in OUTCOME_COLUMNS:
        if column not in result.columns:
            continue
        if column in _TEXT_BLANK:
            result[column] = ""
        else:
            result[column] = np.nan
    return result


def _ensure_starting(frame):
    result = frame.copy()
    if "start_position" in result.columns:
        derived = (
            result["start_position"]
            .astype("string")
            .str.strip()
            .isin(["G", "F", "C"])
            .astype(float)
        )
    else:
        derived = pd.Series(np.nan, index=result.index)
    if "starting" not in result.columns:
        result["starting"] = derived
        return result
    current = pd.to_numeric(result["starting"], errors="coerce")
    result["starting"] = current.where(current.notna(), derived)
    return result


def _drop_stale_features(frame):
    stale = [
        column
        for column in FEATURES
        if column in frame.columns and column not in _SLATE_INPUTS
    ]
    if not stale:
        return frame
    return frame.drop(columns=stale)


def _starting_flags(rows, five):
    if five is None:
        return np.nan
    posted = set()
    for starter in five:
        posted.update(player_name_keys(starter))
    return [
        1.0 if set(player_name_keys(player)) & posted else 0.0
        for player in rows["player_name"]
    ]


def _posted_five(lineups, team_abbr):
    for key, names in lineups.items():
        if normalize_team_abbreviation(key) == team_abbr:
            return names
    return None


def _team_id_for(frame, abbreviation):
    if abbreviation is None or (isinstance(abbreviation, float) and pd.isna(abbreviation)):
        return None
    if "team_abbreviation" not in frame.columns or "team_id" not in frame.columns:
        return None
    wanted = normalize_team_abbreviation(abbreviation)
    labels = frame["team_abbreviation"].map(normalize_team_abbreviation)
    matched = frame.loc[labels.eq(wanted)].sort_values(
        ["game_date", "game_id"],
        kind="mergesort",
    )
    if matched.empty:
        return None
    return matched["team_id"].iloc[-1]


def _live_features(panel):
    normalized = panel.copy()
    normalized["game_date"] = (
        pd.to_datetime(
            normalized["game_date"],
            format="mixed",
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(None)
    )
    featured = add_points_features(add_starter_features(normalized))
    if "closing_row" not in featured.columns:
        raise KeyError("feature builder dropped closing_row")
    return featured


def _load_pinnacle_games():
    latest = max(_PINNACLE_DIR.glob("pinnacle_nba_*_team.json"))
    payload = json.loads(latest.read_text())
    return payload["games"]


def _pinnacle_lines(games, team_abbr):
    """Full-game main spread and total for ``team_abbr``."""
    team_id = canonical_team(team_abbr, "nba")
    for game in games:
        markets = game.get("team_markets") or {}
        participants = game.get("participants") or []
        spread = _main_line(
            markets.get("spread") or [],
            team_id,
            side=None,
            participants=participants,
        )
        if spread is None:
            continue
        total = _main_line(
            markets.get("total") or [],
            team_id=None,
            side="over",
            participants=participants,
        )
        return spread, total
    return float("nan"), float("nan")


def _line_team_label(line, participants):
    """Team name on a line, or away/home participant when ``team`` is absent."""
    team = line.get("team")
    if team:
        return team
    if len(participants) < 2:
        return None
    # Pinnacle participants are always [away, home].
    side = line.get("side")
    if side == "away":
        return participants[0]
    if side == "home":
        return participants[1]
    return None


def _main_line(blocks, team_id, side, participants=None):
    participants = participants or []
    for block in blocks:
        if block.get("period") not in (0, None) or block.get("is_alternate"):
            continue
        for line in block.get("lines") or []:
            if side is not None and line.get("side") != side:
                continue
            if team_id is not None:
                label = _line_team_label(line, participants)
                if canonical_team(label, "nba") != team_id:
                    continue
            if line.get("points") is None:
                continue
            return float(line["points"])
    return None
