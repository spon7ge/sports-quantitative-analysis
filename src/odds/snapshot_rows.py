"""Map scraper JSON projections/picks to odds table row dicts."""

from __future__ import annotations

import re
from datetime import datetime


def parse_american_price(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    return int(text.replace("+", ""))


def _parse_line_updated_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def prizepicks_projections_to_rows(
    projections: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    rows: list[dict] = []
    league_key = league.lower()

    for projection in projections:
        player_name = projection.get("player")
        stat_type = projection.get("stat_type")
        line_score = projection.get("line_score")

        if not player_name or not stat_type or line_score is None:
            continue

        rows.append(
            {
                "league": league_key,
                "player_name": player_name,
                "stat_type": stat_type,
                "line_score": line_score,
                "odds_type": projection.get("odds_type") or "standard",
                "line_updated_at": _parse_line_updated_at(projection.get("updated_at")),
                "scraped_at": scraped_at,
            }
        )

    return rows


def underdog_picks_to_rows(
    picks: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    rows: list[dict] = []
    league_key = league.lower()

    for pick in picks:
        player_name = pick.get("full_name")
        stat_name = pick.get("stat_name")
        stat_value = pick.get("stat_value")
        side = pick.get("choice")

        if not player_name or not stat_name or stat_value is None or not side:
            continue

        rows.append(
            {
                "league": league_key,
                "player_name": player_name,
                "stat_name": stat_name,
                "line_score": stat_value,
                "side": side,
                "american_price": parse_american_price(pick.get("american_price")),
                "payout_multiplier": pick.get("payout_multiplier"),
                "line_updated_at": _parse_line_updated_at(pick.get("updated_at")),
                "scraped_at": scraped_at,
            }
        )

    return rows


_FANDUEL_SIDE_SUFFIX = re.compile(r"\s+(over|under)$", re.IGNORECASE)
_FANDUEL_STAT_TO_MARKET = {
    "hits": "player_hits",
    "extra_base_hits": "player_extra_base_hits",
    "home_runs": "player_home_runs",
    "total_bases": "player_total_bases",
    "rbis": "player_rbis",
    "runs": "player_runs",
    "stolen_bases": "player_stolen_bases",
    "strikeouts": "player_strikeouts",
    "hits_allowed": "player_hits_allowed",
    "walks": "player_walks",
    "walks_allowed": "player_walks_allowed",
    "earned_runs_allowed": "player_earned_runs",
    "earned_runs": "player_earned_runs",
    "pitching_outs": "player_outs",
    "outs_recorded": "player_outs",
    "hits_runs_rbis": "player_hits_runs_rbis",
    "singles": "player_singles",
    "doubles": "player_doubles",
    "triples": "player_triples",
}


def _fanduel_player_and_side(full_name: str, choice: str) -> tuple[str | None, str | None]:
    """Player name plus over/under, recovering the side from a trailing label."""
    player = str(full_name or "").strip()
    side = str(choice or "").strip().lower()
    match = _FANDUEL_SIDE_SUFFIX.search(player)
    if match:
        suffix_side = match.group(1).lower()
        player = player[: match.start()].strip()
        if side not in _VALID_SIDES or side != suffix_side:
            side = suffix_side
    if not player or side not in _VALID_SIDES:
        return None, None
    return player, side


def fanduel_picks_to_rows(
    picks: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map FanDuel scraper JSON picks to odds.mlb_fanduel row dicts."""
    rows: list[dict] = []
    league_key = league.lower()

    for pick in picks:
        player, side = _fanduel_player_and_side(
            str(pick.get("full_name") or ""),
            str(pick.get("choice") or ""),
        )
        stat_name = str(pick.get("stat_name") or "").strip()
        market_type = _FANDUEL_STAT_TO_MARKET.get(stat_name)
        line_raw = pick.get("stat_value")
        if not player or not side or not market_type or line_raw is None:
            continue
        try:
            line_score = float(line_raw)
            american_price = int(pick["american_price"])
        except (KeyError, TypeError, ValueError):
            continue

        rows.append(
            {
                "league": league_key,
                "player_name": player,
                "market_type": market_type,
                "stat_category": stat_name,
                "side": side,
                "line_score": line_score,
                "american_price": american_price,
                "scraped_at": scraped_at,
            }
        )

    return rows


def draftkings_picks_to_rows(
    picks: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map DraftKings scraper JSON picks to odds.mlb_draftkings row dicts."""
    rows: list[dict] = []
    league_key = league.lower()

    for pick in picks:
        player = str(pick.get("full_name") or "").strip()
        side = str(pick.get("choice") or "").strip().lower()
        stat_name = str(pick.get("stat_name") or "").strip()
        market_type = _FANDUEL_STAT_TO_MARKET.get(stat_name)
        line_raw = pick.get("stat_value")
        if not player or side not in _VALID_SIDES or not market_type or line_raw is None:
            continue
        try:
            line_score = float(line_raw)
            american_price = int(pick["american_price"])
        except (KeyError, TypeError, ValueError):
            continue

        rows.append(
            {
                "league": league_key,
                "player_name": player,
                "market_type": market_type,
                "stat_category": stat_name,
                "side": side,
                "line_score": line_score,
                "american_price": american_price,
                "scraped_at": scraped_at,
            }
        )

    return rows


_VALID_SIDES = frozenset({"over", "under"})
_SHARP_BOOKS = frozenset({"fanduel", "draftkings"})
_PARLAY_BOOKS = frozenset(
    {
        "fanduel",
        "draftkings",
        "caesars",
        "betmgm",
        "pinnacle",
        "bet365",
        "prizepicks",
        "underdog",
        "betr",
        "novig",
        "sleeper",
        "betrivers",
        "kalshi",
        "fliff",
    }
)


def _sharp_player_name(row: dict) -> str | None:
    name = row.get("player_name") or row.get("selection")
    if not name:
        return None
    text = str(name).strip()
    return text or None


def sharp_props_to_book_rows(
    rows: list[dict],
    *,
    sportsbook: str,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map Sharp prop API rows to odds.wnba_{fanduel|draftkings} row dicts."""
    book = sportsbook.lower().strip()
    if book not in _SHARP_BOOKS:
        raise ValueError(f"unsupported sportsbook: {sportsbook}")

    out: list[dict] = []
    league_key = league.lower()

    for row in rows:
        if not row.get("is_main_line", False):
            continue
        if str(row.get("sportsbook") or "").lower() != book:
            continue
        market = str(row.get("market_type") or "")
        if not market.startswith("player_"):
            continue
        side = str(row.get("selection_type") or "").lower()
        if side not in _VALID_SIDES:
            continue
        player = _sharp_player_name(row)
        if not player:
            continue
        line_raw = row.get("line")
        odds_raw = row.get("odds_american")
        if line_raw is None or odds_raw is None:
            continue
        try:
            line_score = float(line_raw)
            american_price = int(odds_raw)
        except (TypeError, ValueError):
            continue

        stat_category = row.get("stat_category")
        if stat_category is not None:
            stat_category = str(stat_category).strip() or None

        out.append(
            {
                "league": league_key,
                "player_name": player,
                "market_type": market,
                "stat_category": stat_category,
                "side": side,
                "line_score": line_score,
                "american_price": american_price,
                "scraped_at": scraped_at,
            }
        )

    return out


def parlay_props_to_book_rows(
    rows: list[dict],
    *,
    sportsbook: str,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map Parlay prop API rows for one book (main lines only), with sportsbook set."""
    from src.odds.parlay_main_lines import is_parlay_player_market, select_parlay_main_lines

    book = sportsbook.lower().strip()
    if book not in _PARLAY_BOOKS:
        raise ValueError(f"unsupported sportsbook: {sportsbook}")

    out: list[dict] = []
    league_key = league.lower()

    for row in select_parlay_main_lines(rows, books=frozenset({book})):
        player = str(row.get("player") or "").strip()
        market = str(row.get("market_key") or "").strip()
        if not player or not is_parlay_player_market(market):
            continue
        try:
            line_score = float(row["line"])
        except (KeyError, TypeError, ValueError):
            continue

        sides: list[tuple[str, int]] = []
        for side, raw in (
            ("over", row.get("over_price")),
            ("under", row.get("under_price")),
        ):
            if raw is None:
                continue
            try:
                sides.append((side, int(raw)))
            except (TypeError, ValueError):
                continue
        if not sides:
            continue

        market_label = str(row.get("market") or "").strip() or None
        for side, price in sides:
            out.append(
                {
                    "sportsbook": book,
                    "league": league_key,
                    "player_name": player,
                    "market_type": market,
                    "stat_category": market_label,
                    "side": side,
                    "line_score": line_score,
                    "american_price": price,
                    "scraped_at": scraped_at,
                }
            )

    return out


def parlay_props_to_api_odds_rows(
    rows: list[dict],
    *,
    league: str,
    scraped_at: datetime,
    books: tuple[str, ...] | list[str] | frozenset[str],
) -> list[dict]:
    """Map Parlay props for many books into unified Parlay API odds row dicts."""
    out: list[dict] = []
    for book in books:
        out.extend(
            parlay_props_to_book_rows(
                rows, sportsbook=book, league=league, scraped_at=scraped_at
            )
        )
    return out


_SELENIUM_STAT_TO_MARKET = {
    # Basketball (WNBA / NBA)
    "points": "player_points",
    "assists": "player_assists",
    "rebounds": "player_rebounds",
    "points_rebounds_assists": "player_pts_rebs_asts",
    # Baseball (MLB) — selenium JSON ``stat`` → odds.mlb_pinnacle.market_type
    "hits": "player_hits",
    "home_runs": "player_home_runs",
    "total_bases": "player_total_bases",
    "rbis": "player_rbis",
    "runs": "player_runs",
    "stolen_bases": "player_stolen_bases",
    "strikeouts": "player_strikeouts",
    "hits_allowed": "player_hits_allowed",
    "walks": "player_walks",
    "earned_runs": "player_earned_runs",
    "outs_recorded": "player_outs",
    "pitcher_outs": "player_outs",
}


def selenium_pinnacle_props_to_rows(
    games: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map Selenium Pinnacle game dicts to odds.wnba_pinnacle row dicts."""
    rows: list[dict] = []
    league_key = league.lower()

    for game in games:
        for prop in game.get("props") or []:
            player = str(prop.get("player") or "").strip()
            stat = str(prop.get("stat") or "").strip()
            market_type = _SELENIUM_STAT_TO_MARKET.get(stat)
            line_raw = prop.get("line")
            if not player or not market_type or line_raw is None:
                continue
            try:
                line_score = float(line_raw)
            except (TypeError, ValueError):
                continue

            stat_category = stat or None
            for side, price_key in (
                ("over", "american_over"),
                ("under", "american_under"),
            ):
                price_raw = prop.get(price_key)
                if price_raw is None:
                    continue
                try:
                    american_price = int(price_raw)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "league": league_key,
                        "player_name": player,
                        "market_type": market_type,
                        "stat_category": stat_category,
                        "side": side,
                        "line_score": line_score,
                        "american_price": american_price,
                        "scraped_at": scraped_at,
                    }
                )

    return rows


def selenium_pinnacle_team_to_rows(
    games: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map Selenium Pinnacle game dicts to odds.wnba_pinnacle_team row dicts."""
    rows: list[dict] = []
    league_key = league.lower()

    for game in games:
        participants = game.get("participants") or []
        if len(participants) < 2:
            continue
        away_team = str(participants[0]).strip()
        home_team = str(participants[1]).strip()
        if not away_team or not home_team:
            continue

        matchup_id = game.get("matchup_id")
        start_time = _parse_line_updated_at(game.get("start_time"))
        team_markets = game.get("team_markets") or {}

        for market_type, blocks in team_markets.items():
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                period = block.get("period", 0)
                is_alternate = bool(block.get("is_alternate", False))
                for line in block.get("lines") or []:
                    side = str(line.get("side") or "").strip()
                    american_raw = line.get("american")
                    if not side or american_raw is None:
                        continue
                    try:
                        american_price = int(american_raw)
                    except (TypeError, ValueError):
                        continue

                    points_raw = line.get("points")
                    points: float | None
                    if points_raw is None:
                        points = None
                    else:
                        try:
                            points = float(points_raw)
                        except (TypeError, ValueError):
                            continue

                    team = line.get("team")
                    team_name = str(team).strip() if team else None

                    decimal_raw = line.get("decimal")
                    decimal_price: float | None
                    if decimal_raw is None:
                        decimal_price = None
                    else:
                        try:
                            decimal_price = float(decimal_raw)
                        except (TypeError, ValueError):
                            decimal_price = None

                    rows.append(
                        {
                            "league": league_key,
                            "matchup_id": matchup_id,
                            "away_team": away_team,
                            "home_team": home_team,
                            "start_time": start_time,
                            "market_type": market_type,
                            "period": period,
                            "is_alternate": is_alternate,
                            "side": side,
                            "team": team_name,
                            "points": points,
                            "american_price": american_price,
                            "decimal_price": decimal_price,
                            "scraped_at": scraped_at,
                        }
                    )

    return rows


def prophetx_home_away(competitors: list[dict]) -> tuple[str | None, str | None]:
    """Return (away_team, home_team). ProphetX seq 0 = home, seq 1 = away."""
    home = away = None
    for c in competitors or []:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or c.get("displayName") or "").strip() or None
        seq = c.get("seq")
        if seq == 0:
            home = name
        elif seq == 1:
            away = name
    return away, home


def _prophetx_competitor_ids(competitors: list[dict]) -> tuple[int | None, int | None]:
    """Return (away_id, home_id). ProphetX seq 0 = home, seq 1 = away."""
    home_id = away_id = None
    for c in competitors or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        seq = c.get("seq")
        if seq == 0:
            home_id = cid
        elif seq == 1:
            away_id = cid
    return away_id, home_id


def _prophetx_team_side(
    *,
    market_type: str,
    name: str | None,
    competitor_id: int | None,
    home_id: int | None,
    away_id: int | None,
) -> str:
    display = str(name or "").strip()
    market = market_type.lower()
    name_lower = display.lower()

    if market == "total" or name_lower.startswith("over") or name_lower.startswith("under"):
        if name_lower.startswith("over"):
            return "over"
        if name_lower.startswith("under"):
            return "under"

    if competitor_id is not None:
        if home_id is not None and competitor_id == home_id:
            return "home"
        if away_id is not None and competitor_id == away_id:
            return "away"

    return display


def prophetx_props_to_rows(
    games: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    rows: list[dict] = []
    league_key = league.lower()
    for game in games:
        if not isinstance(game, dict):
            continue
        away, home = prophetx_home_away(game.get("competitors") or [])
        start_time = _parse_line_updated_at(game.get("scheduled"))
        event_id = game.get("event_id")
        for prop in game.get("props") or []:
            if not isinstance(prop, dict):
                continue
            player = prop.get("player")
            stat = prop.get("stat")
            line = prop.get("line")
            if not player or not stat or line is None:
                continue
            is_main = prop.get("is_main")
            if not isinstance(is_main, bool):
                is_main = True
            for side in ("over", "under"):
                payload = prop.get(side)
                if not isinstance(payload, dict):
                    continue
                american = parse_american_price(payload.get("american"))
                if american is None:
                    continue
                stake = payload.get("stake")
                rows.append(
                    {
                        "league": league_key,
                        "event_id": event_id,
                        "away_team": away,
                        "home_team": home,
                        "start_time": start_time,
                        "player_name": player,
                        "stat_name": stat,
                        "line_score": line,
                        "side": side,
                        "american_price": american,
                        "stake": stake,
                        "market_id": prop.get("market_id"),
                        "sub_type": prop.get("sub_type"),
                        "is_main": is_main,
                        "scraped_at": scraped_at,
                    }
                )
    return rows


def prophetx_team_to_rows(
    games: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    rows: list[dict] = []
    league_key = league.lower()
    for game in games:
        if not isinstance(game, dict):
            continue
        competitors = game.get("competitors") or []
        away, home = prophetx_home_away(competitors)
        if not away or not home:
            continue
        away_id, home_id = _prophetx_competitor_ids(competitors)
        start_time = _parse_line_updated_at(game.get("scheduled"))
        event_id = game.get("event_id")
        for market_type, sides in (game.get("team_markets") or {}).items():
            if not isinstance(sides, list):
                continue
            for side_row in sides:
                if not isinstance(side_row, dict):
                    continue
                american = parse_american_price(side_row.get("american"))
                if american is None:
                    continue
                name = side_row.get("name")
                rows.append(
                    {
                        "league": league_key,
                        "event_id": event_id,
                        "away_team": away,
                        "home_team": home,
                        "start_time": start_time,
                        "market_type": str(market_type),
                        "side": _prophetx_team_side(
                            market_type=str(market_type),
                            name=name,
                            competitor_id=side_row.get("competitor_id"),
                            home_id=home_id,
                            away_id=away_id,
                        ),
                        "team": name,
                        "points": side_row.get("line"),
                        "american_price": american,
                        "stake": side_row.get("stake"),
                        "scraped_at": scraped_at,
                    }
                )
    return rows


def novig_props_to_rows(
    games: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map Novig props JSON games → odds.mlb_novig rows (UUID event/market ids)."""
    # Same game shape as ProphetX; keep a named entry point for Novig loaders/tests.
    return prophetx_props_to_rows(games, league=league, scraped_at=scraped_at)


def novig_team_to_rows(
    games: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map Novig team JSON games → odds.mlb_novig_team rows."""
    return prophetx_team_to_rows(games, league=league, scraped_at=scraped_at)
