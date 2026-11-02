from datetime import datetime, timedelta

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2
from nba_api.stats.library.parameters import LeagueID

_LEAGUE_ID = {"nba": LeagueID.nba, "wnba": LeagueID.wnba}
_gameCache: dict = {}


def _infer_schedule_season(league: str, game_date: str) -> str:
    """NBA uses 'YYYY-YY' (season starts in October); WNBA uses the calendar year."""
    d = datetime.strptime(game_date, "%Y-%m-%d")
    if league == "wnba":
        return str(d.year)
    start_year = d.year if d.month >= 10 else d.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _pick_column(df, *candidates: str) -> str:
    """Return the first candidate column name that exists in ``df``."""
    return next(c for c in candidates if c in df.columns)


def getUpcomingGamesCached(date, league: str = "nba", season: str | None = None):
    """Fetch (and cache) the full season schedule for ``league``.

    ``date`` is only used to infer the season when ``season`` is omitted.
    One cached schedule per (league, season) covers every game date.
    """
    league = league.lower()
    if league not in _LEAGUE_ID:
        raise ValueError(f"Unknown league {league!r}; expected 'nba' or 'wnba'")

    season = season or _infer_schedule_season(league, date)
    cache_key = (league, season)

    if cache_key not in _gameCache:
        schedule = scheduleleaguev2.ScheduleLeagueV2(
            league_id=_LEAGUE_ID[league], season=season
        ).get_data_frames()[0]
        schedule["gameDate"] = pd.to_datetime(schedule["gameDate"]).dt.strftime("%Y-%m-%d")
        _gameCache[cache_key] = schedule

    return _gameCache[cache_key]


def findOpp(
    playerName,
    players_df,
    gameDate,
    max_days_ahead=3,
    league: str = "nba",
    season: str | None = None,
):
    """Return ``(opp_tricode, is_home)`` for the player's next game, or ``(None, None)``."""
    # 1. Find the player's current team (last row wins).
    name_col = _pick_column(players_df, "PLAYER_NAME", "player_name")
    team_col = _pick_column(players_df, "TEAM_ABBREVIATION", "team_abbreviation")
    team_rows = players_df.loc[players_df[name_col] == playerName, team_col]
    if team_rows.empty:
        print(f"No rows for player {playerName!r}")
        return None, None
    team = team_rows.iloc[-1]

    # 2. Find that team's games in [gameDate, gameDate + max_days_ahead].
    last_date = datetime.strptime(gameDate, "%Y-%m-%d") + timedelta(days=max_days_ahead)
    schedule = getUpcomingGamesCached(gameDate, league=league, season=season)

    in_window = schedule["gameDate"].between(gameDate, last_date.strftime("%Y-%m-%d"))
    is_home = schedule["homeTeam_teamTricode"] == team
    is_away = schedule["awayTeam_teamTricode"] == team
    games = schedule[in_window & (is_home | is_away)].sort_values("gameDate", kind="stable")

    if games.empty:
        print(f"No game found for {team} within {max_days_ahead} days from {gameDate}")
        return None, None

    # 3. The earliest game is the "next" one.
    game = games.iloc[0]
    if game["homeTeam_teamTricode"] == team:
        return game["awayTeam_teamTricode"], 1
    return game["homeTeam_teamTricode"], 0