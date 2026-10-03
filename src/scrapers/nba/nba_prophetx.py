"""ProphetX NBA scraper — public API (team markets + player props)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.prophetx.co"
NBA_TOURNAMENT_ID = 132
MARKET_BATCH_SIZE = 20
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "X-Currency": "cash",
}

T = TypeVar("T")

# Repo root on sys.path so `python nba_prophetx.py` from scrapers/ can import src.*
try:
    from .paths import ensure_repo_on_path
except ImportError:  # python nba_prophetx.py from this directory
    from paths import ensure_repo_on_path

_ROOT = str(ensure_repo_on_path(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "data", "props", "prophetx", "nba")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")


def output_filename(league: str, now: datetime, *, kind: str) -> str:
    stamp = now.astimezone(_OUTPUT_TZ).strftime("%Y-%m-%d_%H%M%S")
    return f"prophetx_{league.strip().lower()}_{stamp}_{kind}.json"


def team_output_path(props_path: str) -> str:
    if props_path.endswith("_props.json"):
        return props_path[: -len("_props.json")] + "_team.json"
    root, ext = os.path.splitext(props_path)
    return f"{root}_team{ext or '.json'}"


def resolve_props_output_path(*, now: datetime | None = None) -> str:
    when = now or datetime.now(_OUTPUT_TZ)
    env_file = os.environ.get("PROPHETX_OUTPUT", "").strip()
    env_dir = os.environ.get("PROPHETX_OUTPUT_DIR", "").strip()
    name = output_filename("nba", when, kind="props")
    if env_file:
        expanded = os.path.expanduser(env_file)
        if expanded.lower().endswith(".json") and not os.path.isdir(expanded):
            return expanded
        os.makedirs(expanded, exist_ok=True)
        return os.path.join(expanded, name)
    base = env_dir or _DEFAULT_OUTPUT_DIR
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)


def pick_main_market_line(market: dict[str, Any]) -> dict[str, Any] | None:
    lines = [ln for ln in (market.get("marketLines") or []) if isinstance(ln, dict)]
    if not lines:
        return None
    favourites = [ln for ln in lines if ln.get("favourite") is True]
    if len(favourites) == 1:
        return favourites[0]
    if len(favourites) > 1:
        return favourites[0]
    if len(lines) == 1:
        return lines[0]
    return None


def best_selection(side: list[dict[str, Any]]) -> dict[str, Any] | None:
    for sel in side:
        if isinstance(sel, dict):
            return sel
    return None


def american_and_stake(sel: dict[str, Any]) -> tuple[int | None, float | None]:
    raw = sel.get("odds")
    american: int | None
    try:
        american = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        american = None
    stake_raw = sel.get("stake")
    try:
        stake = float(stake_raw) if stake_raw is not None else None
    except (TypeError, ValueError):
        stake = None
    return american, stake


TEAM_SUBTYPE_TO_KEY: dict[str, str] = {
    "moneyline": "moneyline",
    "spread": "spread",
    "total": "total",
    "first_half_moneyline": "first_half_moneyline",
    "first_half_spread": "first_half_spread",
    "first_half_total": "first_half_total",
}

# Top-level selections fallback applies only to moneyline-style keys.
_MONEYLINE_OUTPUT_KEYS = frozenset({"moneyline", "first_half_moneyline"})

PERIOD_MARKET_SUBTYPES = (
    "first_half_moneyline",
    "first_half_spread",
    "first_half_total",
)


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    competitors = []
    for c in event.get("competitors") or []:
        if not isinstance(c, dict):
            continue
        competitors.append(
            {
                "id": c.get("id"),
                "name": c.get("name") or c.get("displayName"),
                "abbreviation": c.get("abbreviation"),
                "seq": c.get("seq"),
            }
        )
    return {
        "event_id": event.get("id"),
        "name": event.get("name") or event.get("displayName"),
        "scheduled": event.get("scheduled"),
        "status": event.get("status"),
        "competitors": competitors,
    }


def _sides_from_book(book: dict[str, Any]) -> list[list[dict[str, Any]]]:
    sels = book.get("selections")
    if isinstance(sels, list) and sels:
        return [s for s in sels if isinstance(s, list)]
    return []


def _side_rows(sides: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in sides:
        best = best_selection(side)
        if not best:
            continue
        american, stake = american_and_stake(best)
        line = best.get("line")
        rows.append(
            {
                "name": best.get("name") or best.get("displayName"),
                "competitor_id": best.get("competitorId"),
                "american": american,
                "line": None if line in (0, 0.0, None) and "over" not in str(best.get("name", "")).lower() else line,
                "stake": stake,
            }
        )
    return rows


def extract_team_markets(markets: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for market in markets:
        if not isinstance(market, dict):
            continue
        sub = str(market.get("subType") or market.get("type") or "")
        key = TEAM_SUBTYPE_TO_KEY.get(sub)
        if not key:
            continue
        book: dict[str, Any] | None
        if key in _MONEYLINE_OUTPUT_KEYS and market.get("selections"):
            book = market
        else:
            book = pick_main_market_line(market)
        if not book:
            continue
        rows = _side_rows(_sides_from_book(book))
        if rows:
            out[key] = rows
    return out


PROP_SUBTYPE_TO_STAT: dict[str, str] = {
    "player_total_points": "points",
    "player_total_rebounds": "rebounds",
    "player_total_assists": "assists",
    "player_total_three_pointers_made": "three_pointers_made",
    "player_total_points_rebounds": "points_rebounds",
    "player_total_points_assists": "points_assists",
    "player_total_rebounds_assists": "rebounds_assists",
    "player_total_points_rebounds_assists": "points_rebounds_assists",
}

# Longest suffixes first so combo markets are not trimmed by " Total Points".
_PROP_NAME_SUFFIXES = (
    " Total Points, Rebounds & Assists",
    " Total Points & Rebounds",
    " Total Points & Assists",
    " Total Rebounds & Assists",
    " Total Three Pointers Made",
    " Total Points",
    " Total Rebounds",
    " Total Assists",
)


def player_name_from_market(market: dict[str, Any]) -> str:
    name = str(market.get("name") or "").strip()
    for suffix in _PROP_NAME_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def _prop_row_from_book(
    market: dict[str, Any],
    book: dict[str, Any],
    *,
    stat: str,
    sub: str,
    is_main: bool,
) -> dict[str, Any] | None:
    sides = _sides_from_book(book)
    over = under = None
    line: float | None = None
    for side in sides:
        best = best_selection(side)
        if not best:
            continue
        american, stake = american_and_stake(best)
        side_name = str(best.get("name") or "").lower()
        payload = {"american": american, "stake": stake}
        if best.get("line") is not None:
            try:
                line = float(best["line"])
            except (TypeError, ValueError):
                pass
        if side_name.startswith("over"):
            over = payload
        elif side_name.startswith("under"):
            under = payload
    if over is None and under is None:
        return None
    return {
        "player": player_name_from_market(market),
        "stat": stat,
        "line": line,
        "over": over,
        "under": under,
        "market_id": market.get("id"),
        "sub_type": sub,
        "is_main": is_main,
    }


def extract_props(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        sub = str(market.get("subType") or "")
        stat = PROP_SUBTYPE_TO_STAT.get(sub)
        if not stat:
            continue
        lines = [ln for ln in (market.get("marketLines") or []) if isinstance(ln, dict)]
        if not lines:
            continue
        favourites = [ln for ln in lines if ln.get("favourite") is True]
        if len(favourites) > 1:
            logger.debug(
                "ProphetX prop market %s has %s favourite lines; marking first as is_main",
                market.get("id"),
                len(favourites),
            )
        if favourites:
            main_book = favourites[0]
        elif len(lines) == 1:
            main_book = lines[0]
        else:
            main_book = None
        for book in lines:
            is_main = book is main_book
            row = _prop_row_from_book(market, book, stat=stat, sub=sub, is_main=is_main)
            if row is not None:
                rows.append(row)
    return rows


def chunked(items: list[T], size: int) -> list[list[T]]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_json(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    retries: int = 3,
) -> Any:
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=60, headers=DEFAULT_HEADERS)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_err = exc
            if attempt + 1 >= retries:
                break
            time.sleep(0.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def fetch_nba_events(session: requests.Session) -> list[dict[str, Any]]:
    path = f"/trade/public/api/v1/tournaments/{NBA_TOURNAMENT_ID}/events"
    events: list[dict[str, Any]] = []
    nxt: str | int | None = None
    while True:
        params = {"next": nxt} if nxt is not None else None
        payload = fetch_json(session, path, params=params)
        chunk = payload.get("data") or []
        if isinstance(chunk, list):
            events.extend([e for e in chunk if isinstance(e, dict)])
        nxt = payload.get("next")
        if not nxt or not chunk:
            break
    max_events = os.environ.get("PROPHETX_MAX_EVENTS", "").strip()
    if max_events.isdigit():
        events = events[: int(max_events)]
    return events


def fetch_markets_for_events(
    session: requests.Session,
    event_ids: list[int],
    *,
    market_types: str | None = None,
    market_sub_types: str | None = None,
    batch_size: int = MARKET_BATCH_SIZE,
) -> list[dict[str, Any]]:
    path = "/partner/v3/public/get_multiple_markets"
    out: list[dict[str, Any]] = []
    for batch in chunked(event_ids, batch_size):
        params: dict[str, Any] = {"event_ids": ",".join(str(i) for i in batch)}
        if market_types:
            params["market_types"] = market_types
        if market_sub_types:
            params["market_sub_types"] = market_sub_types
        payload = fetch_json(session, path, params=params)
        data = payload.get("data") or []
        if isinstance(data, list):
            out.extend([row for row in data if isinstance(row, dict)])
    return out


def build_game_snapshots(
    events: list[dict[str, Any]],
    team_market_rows: list[dict[str, Any]],
    prop_market_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    team_by_id = {
        int(row["eventId"]): row.get("markets") or []
        for row in team_market_rows
        if row.get("eventId") is not None
    }
    props_by_id = {
        int(row["eventId"]): row.get("markets") or []
        for row in prop_market_rows
        if row.get("eventId") is not None
    }
    props_games: list[dict[str, Any]] = []
    team_games: list[dict[str, Any]] = []
    for event in events:
        base = normalize_event(event)
        event_id = base.get("event_id")
        if event_id is None:
            continue
        event_id_int = int(event_id)
        props_games.append(
            {**base, "props": extract_props(props_by_id.get(event_id_int, []))}
        )
        team_games.append(
            {
                **base,
                "team_markets": extract_team_markets(
                    team_by_id.get(event_id_int, [])
                ),
            }
        )
    return props_games, team_games


def _payload_base(*, fetched_at: str) -> dict[str, Any]:
    return {
        "source": "prophetx",
        "fetched_at": fetched_at,
        "league": "nba",
        "tournament_id": NBA_TOURNAMENT_ID,
    }


def write_snapshots(
    props_games: list[dict[str, Any]],
    team_games: list[dict[str, Any]],
    *,
    props_path: str,
) -> tuple[str, str]:
    fetched_at = datetime.now(_OUTPUT_TZ).isoformat(timespec="seconds")
    base = _payload_base(fetched_at=fetched_at)
    props_payload = {**base, "snapshot_kind": "props", "games": props_games}
    team_payload = {**base, "snapshot_kind": "team", "games": team_games}
    team_path = team_output_path(props_path)
    for path, payload in ((props_path, props_payload), (team_path, team_payload)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return props_path, team_path


def load_supabase_snapshots(
    props_games: list[dict[str, Any]],
    team_games: list[dict[str, Any]],
    *,
    scraped_at: datetime | None = None,
    props_path: str | None = None,
    team_path: str | None = None,
) -> None:
    """Upsert snapshot games to odds.nba_prophetx / odds.nba_prophetx_team."""
    try:
        from src.odds.load_snapshots import (
            load_prophetx_props_snapshot,
            load_prophetx_team_snapshot,
        )

        when = scraped_at or datetime.now(timezone.utc)
        n_props = load_prophetx_props_snapshot(
            props_games, league="nba", scraped_at=when
        )
        n_team = load_prophetx_team_snapshot(
            team_games, league="nba", scraped_at=when
        )
        logger.info(
            "Supabase ProphetX upserted props=%s team=%s%s%s",
            n_props,
            n_team,
            f" props_path={props_path}" if props_path else "",
            f" team_path={team_path}" if team_path else "",
        )
    except Exception as exc:
        logger.error("Supabase ProphetX load failed (JSON kept): %s", exc)


def _merge_market_rows(
    primary: list[dict[str, Any]],
    extra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    markets_by_event: dict[int, list[dict[str, Any]]] = {}
    for row in primary + extra:
        event_id = row.get("eventId")
        if event_id is None:
            continue
        markets_by_event.setdefault(int(event_id), []).extend(
            row.get("markets") or []
        )
    return [
        {"eventId": event_id, "markets": markets}
        for event_id, markets in markets_by_event.items()
    ]


def run() -> None:
    logging.basicConfig(
        level=getattr(
            logging,
            os.environ.get("LOG_LEVEL", "INFO").upper(),
            logging.INFO,
        ),
        format="[%(levelname)-8s] %(name)s: %(message)s",
    )
    session = requests.Session()
    events = fetch_nba_events(session)
    event_ids = [int(event["id"]) for event in events if event.get("id") is not None]
    team_rows = fetch_markets_for_events(
        session,
        event_ids,
        market_types="moneyline,spread,total",
    )
    period_rows = fetch_markets_for_events(
        session,
        event_ids,
        market_sub_types=",".join(PERIOD_MARKET_SUBTYPES),
    )
    prop_rows = fetch_markets_for_events(
        session,
        event_ids,
        market_sub_types=",".join(PROP_SUBTYPE_TO_STAT),
    )
    props_games, team_games = build_game_snapshots(
        events,
        _merge_market_rows(team_rows, period_rows),
        prop_rows,
    )
    props_path = resolve_props_output_path()
    props_path, team_path = write_snapshots(
        props_games, team_games, props_path=props_path
    )
    logger.info(
        "Wrote ProphetX snapshots: props_games=%s team_games=%s props=%s team=%s",
        len(props_games),
        len(team_games),
        props_path,
        team_path,
    )
    load_supabase_snapshots(
        props_games,
        team_games,
        props_path=props_path,
        team_path=team_path,
    )


if __name__ == "__main__":
    run()
