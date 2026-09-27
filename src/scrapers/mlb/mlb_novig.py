"""Novig MLB scraper — public REST events + allowlisted GraphQL markets.

Hasura rejects ad-hoc queries (`query is not allowed`). Event IDs come from
`GET /nbx/v1/trading/MLB/page`; markets use the website's EventMarkets_Query.

After writing JSON snapshots, upserts to odds.mlb_novig / odds.mlb_novig_team
unless NOVIG_SKIP_DB is set.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.novig.us/v1/graphql"
REST_BASE = "https://api.novig.us/nbx/v1"
TRADING_PAGE_URL = f"{REST_BASE}/trading/MLB/page"
EVENT_MARKETS_OPERATION = "EventMarkets_Query"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.7778.280 "
        "Safari/537.36"
    ),
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://novig.com",
    "Referer": "https://novig.com/",
}

# Exact document the novig.com app POSTs. Hasura allowlists this query;
# custom GetMlbEvents / GetEventMarkets documents return "query is not allowed".
_EVENT_MARKETS_QUERY = """query EventMarkets_Query($eventId: uuid, $marketVisibleWhere: market_bool_exp) @cached(ttl: 5) {
  event(where: {id: {_eq: $eventId}}) {
    ...MarketSelector_Frag
    ...EventData_Frag
    ...EventScoreboard_Frag
    ...CustomEvent_Frag
    id
    description
    type
    league
    scheduled_start
    status
    tournament_markets: markets {
      type
      __typename
    }
    game {
      id
      awayTeam {
        id
        name
        symbol
        primary_color
        secondary_color
        swish_id
        optic_odds_id
        __typename
      }
      homeTeam {
        id
        name
        symbol
        primary_color
        secondary_color
        swish_id
        optic_odds_id
        __typename
      }
      end_date
      away_score
      home_score
      time_remaining
      sport
      __typename
    }
    __typename
  }
}

fragment MarketData_Frag on market {
  id
  type
  strike
  status
  re_settled_at
  competitor {
    id
    name
    country
    __typename
  }
  market_locks(
    order_by: {created_at: desc}
    where: {deleted_at: {_is_null: true}}
  ) {
    id
    market_id
    created_at
    minute_duration
    __typename
  }
  player {
    id
    full_name
    __typename
  }
  __typename
}

fragment Outcome_Frag on outcome {
  id
  available
  altAvailable
  index
  description
  competitor {
    id
    symbol
    __typename
  }
  __typename
}

fragment StrikePriceSelector_Frag on market {
  id
  description
  eventId
  strike
  ...MarketData_Frag
  outcomes {
    id
    index
    description
    ...Outcome_Frag
    __typename
  }
  __typename
}

fragment TennisScoreboard_Frag on game {
  events {
    id
    status
    __typename
  }
  home_set_1
  home_set_2
  home_set_3
  home_set_4
  home_set_5
  away_set_1
  away_set_2
  away_set_3
  away_set_4
  away_set_5
  possession
  home_game_score
  away_game_score
  homeTeam {
    id
    name
    __typename
  }
  awayTeam {
    id
    name
    __typename
  }
  __typename
}

fragment BaseEvent_Frag on event {
  id
  status
  type
  description
  league
  scheduled_start
  event_locks(order_by: {created_at: desc}, where: {deleted_at: {_is_null: true}}) {
    id
    event_id
    minute_duration
    created_at
    __typename
  }
  __typename
}

fragment MarketSelector_Frag on event {
  id
  type
  league
  markets(where: $marketVisibleWhere) {
    is_consensus
    id
    strike
    type
    description
    status
    volume
    market_detail {
      question
      __typename
    }
    ...MarketData_Frag
    ...StrikePriceSelector_Frag
    outcomes {
      id
      index
      description
      available
      ...Outcome_Frag
      __typename
    }
    market_locks(
      order_by: {created_at: desc}
      where: {deleted_at: {_is_null: true}}
    ) {
      id
      market_id
      minute_duration
      created_at
      __typename
    }
    player {
      id
      full_name
      logo
      jersey_number
      player_competitors {
        competitor {
          id
          name
          short_name
          symbol
          swish_id
          optic_odds_id
          primary_color
          secondary_color
          country
          __typename
        }
        __typename
      }
      __typename
    }
    competitor {
      id
      name
      short_name
      symbol
      swish_id
      optic_odds_id
      primary_color
      secondary_color
      country
      __typename
    }
    __typename
  }
  event_locks(order_by: {created_at: desc}, where: {deleted_at: {_is_null: true}}) {
    id
    event_id
    minute_duration
    created_at
    __typename
  }
  __typename
}

fragment EventData_Frag on event {
  id
  type
  status
  scheduled_start
  league
  game {
    id
    league
    sport
    scheduled_start
    period
    time_remaining
    home_score
    away_score
    awayTeam {
      id
      name
      symbol
      short_name
      mascot
      primary_color
      secondary_color
      swish_id
      optic_odds_id
      wins
      losses
      draws
      country
      __typename
    }
    homeTeam {
      id
      name
      symbol
      short_name
      mascot
      primary_color
      secondary_color
      swish_id
      optic_odds_id
      wins
      losses
      draws
      country
      __typename
    }
    __typename
  }
  __typename
}

fragment EventScoreboard_Frag on event {
  id
  type
  status
  description
  league
  scheduled_start
  game {
    id
    ...TennisScoreboard_Frag
    status
    home_score
    away_score
    period
    time_remaining
    down
    distance
    yardLine
    yardline_territory
    league
    sport
    homeTeam {
      id
      name
      short_name
      mascot
      symbol
      primary_color
      secondary_color
      wins
      losses
      draws
      swish_id
      optic_odds_id
      ranking
      country
      __typename
    }
    awayTeam {
      id
      name
      short_name
      mascot
      symbol
      primary_color
      secondary_color
      wins
      losses
      draws
      swish_id
      optic_odds_id
      ranking
      country
      __typename
    }
    __typename
  }
  parent_event {
    id
    description
    __typename
  }
  scoreboard_markets: markets(where: {type: {_eq: "MONEY"}}, limit: 1) {
    id
    type
    outcomes {
      id
      index
      status
      __typename
    }
    __typename
  }
  custom_markets: markets(where: {type: {_eq: "CUSTOM"}}, limit: 1) {
    id
    type
    __typename
  }
  __typename
}

fragment CustomEvent_Frag on event {
  ...BaseEvent_Frag
  custom_event_markets_aggregate: markets_aggregate(
    where: {type: {_eq: "CUSTOM"}, status: {_eq: "OPEN"}}
  ) {
    aggregate {
      count
      __typename
    }
    __typename
  }
  custom_event_volume_aggregate: markets_aggregate(
    where: {type: {_eq: "CUSTOM"}, status: {_eq: "OPEN"}}
  ) {
    aggregate {
      sum {
        volume
        __typename
      }
      __typename
    }
    __typename
  }
  custom_event_markets: markets(
    where: {type: {_eq: "CUSTOM"}, status: {_eq: "OPEN"}}
    order_by: {volume: desc}
  ) {
    ...MarketData_Frag
    description
    volume
    market_detail {
      question
      __typename
    }
    outcomes {
      ...Outcome_Frag
      __typename
    }
    __typename
  }
  __typename
}
"""

_MARKET_VISIBLE_WHERE: dict[str, Any] = {
    "_and": [
        {"status": {"_eq": "OPEN"}},
        {
            "_or": [
                {"is_consensus": {"_eq": True}},
                {"outcomes": {"available": {"_is_null": False}}},
            ]
        },
        {
            "_and": [
                {
                    "_not": {
                        "market_locks": {
                            "_and": [{"deleted_at": {"_is_null": True}}]
                        }
                    }
                },
                {
                    "event": {
                        "_not": {
                            "event_locks": {
                                "_and": [{"deleted_at": {"_is_null": True}}]
                            }
                        }
                    }
                },
            ]
        },
    ]
}

try:
    from .paths import ensure_repo_on_path
except ImportError:  # python mlb_novig.py from this directory
    from paths import ensure_repo_on_path

_ROOT = str(ensure_repo_on_path(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "data", "props", "novig", "mlb")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")


def output_filename(league: str, now: datetime, *, kind: str) -> str:
    if kind not in ("props", "team"):
        raise ValueError(f"kind must be 'props' or 'team', got {kind!r}")
    stamp = now.astimezone(_OUTPUT_TZ).strftime("%Y-%m-%d_%H%M%S")
    return f"novig_{league.strip().lower()}_{stamp}_{kind}.json"


def team_output_path(props_path: str) -> str:
    if props_path.endswith("_props.json"):
        return props_path[: -len("_props.json")] + "_team.json"
    root, ext = os.path.splitext(props_path)
    return f"{root}_team{ext or '.json'}"


def resolve_props_output_path(*, now: datetime | None = None) -> str:
    when = now or datetime.now(_OUTPUT_TZ)
    env_file = os.environ.get("NOVIG_OUTPUT", "").strip()
    env_dir = os.environ.get("NOVIG_OUTPUT_DIR", "").strip()
    name = output_filename("mlb", when, kind="props")
    if env_file:
        expanded = os.path.expanduser(env_file)
        if expanded.lower().endswith(".json") and not os.path.isdir(expanded):
            return expanded
        os.makedirs(expanded, exist_ok=True)
        return os.path.join(expanded, name)
    base = env_dir or _DEFAULT_OUTPUT_DIR
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)


def probability_to_american(prob: float) -> int | None:
    if prob <= 0.0 or prob >= 1.0:
        return None
    if abs(prob - 0.5) < 1e-12:
        return -100
    if prob > 0.5:
        return int(round(-100.0 * prob / (1.0 - prob)))
    return int(round(100.0 * (1.0 - prob) / prob))


def qty_cents_to_stake_dollars(qty: float | int | None) -> float | None:
    if qty is None:
        return None
    try:
        cents = float(qty)
    except (TypeError, ValueError):
        return None
    if cents <= 0:
        return None
    return cents / 100.0


def outcome_quote(
    outcome: dict[str, Any],
    opposite: dict[str, Any] | None,
) -> dict[str, int | float | None] | None:
    raw = outcome.get("available")
    if raw is None:
        return None
    try:
        prob = float(raw)
    except (TypeError, ValueError):
        return None
    american = probability_to_american(prob)
    if american is None:
        return None
    stake: float | None = None
    if opposite:
        total = 0.0
        for order in opposite.get("orders") or []:
            if not isinstance(order, dict):
                continue
            if str(order.get("status") or "OPEN").upper() != "OPEN":
                continue
            part = qty_cents_to_stake_dollars(order.get("qty"))
            if part is not None:
                total += part
        if total > 0:
            stake = total
    return {"american": american, "stake": stake}


PROP_TYPE_TO_STAT: dict[str, str] = {
    "HITS": "hits",
    "HOME_RUNS": "home_runs",
    "RBIS": "rbis",
    "RUNS": "runs",
    "TOTAL_BASES": "total_bases",
    "STOLEN_BASES": "stolen_bases",
    "SINGLES": "singles",
    "DOUBLES": "doubles",
    "HITS_ALLOWED": "hits_allowed",
    "PITCHER_STRIKEOUTS": "strikeouts",
}


def _outcome_side(description: str) -> str | None:
    lower = str(description).strip().lower()
    if lower.startswith("over"):
        return "over"
    if lower.startswith("under"):
        return "under"
    return None


def _prop_evenness_score(
    over_outcome: dict[str, Any] | None,
    under_outcome: dict[str, Any] | None,
) -> float:
    over_avail = _outcome_available(over_outcome) if over_outcome else None
    under_avail = _outcome_available(under_outcome) if under_outcome else None
    if over_avail is None:
        over_avail = 1.0
    if under_avail is None:
        under_avail = 1.0
    return abs(over_avail - 0.5) + abs(under_avail - 0.5)


def extract_props(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, str], list[tuple[dict[str, Any], float]]] = {}

    for market in markets:
        if not isinstance(market, dict):
            continue
        player = market.get("player")
        if not isinstance(player, dict):
            continue
        stat = PROP_TYPE_TO_STAT.get(str(market.get("type") or ""))
        if stat is None:
            continue
        try:
            line = float(market.get("strike"))
        except (TypeError, ValueError):
            continue

        over_outcome: dict[str, Any] | None = None
        under_outcome: dict[str, Any] | None = None
        for outcome in _market_outcomes(market):
            side = _outcome_side(str(outcome.get("description") or ""))
            if side == "over":
                over_outcome = outcome
            elif side == "under":
                under_outcome = outcome

        over_quote = (
            outcome_quote(over_outcome, under_outcome) if over_outcome else None
        )
        under_quote = (
            outcome_quote(under_outcome, over_outcome) if under_outcome else None
        )
        if over_quote is None and under_quote is None:
            continue

        player_key = player.get("id") or player.get("name") or player.get("full_name")
        row: dict[str, Any] = {
            "player": str(player.get("name") or player.get("full_name") or ""),
            "stat": stat,
            "line": line,
            "over": over_quote,
            "under": under_quote,
            "market_id": str(market.get("id") or ""),
            "sub_type": str(market.get("type") or "").lower(),
            "is_main": False,
        }
        rows.append(row)
        score = _prop_evenness_score(over_outcome, under_outcome)
        grouped.setdefault((player_key, stat), []).append((row, score))

    for group in grouped.values():
        if len(group) == 1:
            group[0][0]["is_main"] = True
            continue
        best_row, _ = min(group, key=lambda item: item[1])
        best_row["is_main"] = True

    return rows


_STATUS_MAP = {
    "OPEN_INGAME": "live",
    "OPEN_PREGAME": "not_started",
}
_SPREAD_LINE_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*$")


def _map_status(raw: str) -> str:
    if raw in _STATUS_MAP:
        return _STATUS_MAP[raw]
    return raw.lower() if raw else ""


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    game = event.get("game") if isinstance(event.get("game"), dict) else {}
    home = game.get("homeTeam") if isinstance(game.get("homeTeam"), dict) else {}
    away = game.get("awayTeam") if isinstance(game.get("awayTeam"), dict) else {}
    competitors: list[dict[str, Any]] = []
    if home:
        competitors.append(
            {
                "id": home.get("id"),
                "name": home.get("name"),
                "seq": 0,
            }
        )
    if away:
        competitors.append(
            {
                "id": away.get("id"),
                "name": away.get("name"),
                "seq": 1,
            }
        )
    return {
        "event_id": event.get("id"),
        "name": event.get("description"),
        "scheduled": game.get("scheduled_start"),
        "status": _map_status(str(event.get("status") or "")),
        "competitors": competitors,
    }


def _outcome_available(outcome: dict[str, Any]) -> float | None:
    raw = outcome.get("available")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _market_outcomes(market: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = market.get("outcomes") or []
    return [o for o in outcomes if isinstance(o, dict)]


def _both_sides_available(market: dict[str, Any]) -> bool:
    outcomes = _market_outcomes(market)
    if len(outcomes) < 2:
        return False
    return all(_outcome_available(o) is not None for o in outcomes[:2])


def _evenness_score(market: dict[str, Any]) -> float:
    outcomes = _market_outcomes(market)
    if len(outcomes) < 2:
        return float("inf")
    a = _outcome_available(outcomes[0])
    b = _outcome_available(outcomes[1])
    if a is None or b is None:
        return float("inf")
    return abs(a - 0.5) + abs(b - 0.5)


def pick_main_spread(markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        m
        for m in markets
        if isinstance(m, dict) and m.get("type") == "SPREAD" and _both_sides_available(m)
    ]
    if not candidates:
        return None
    for market in candidates:
        try:
            strike = abs(float(market.get("strike", 0)))
        except (TypeError, ValueError):
            continue
        if abs(strike - 1.5) < 1e-9:
            return market
    return min(candidates, key=_evenness_score)


def pick_main_total(markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        m
        for m in markets
        if isinstance(m, dict) and m.get("type") == "TOTAL" and _both_sides_available(m)
    ]
    if not candidates:
        return None
    return min(candidates, key=_evenness_score)


def _competitor_id_for_outcome(
    description: str,
    competitors: list[dict[str, Any]] | None,
) -> Any:
    if not competitors:
        return None
    for comp in competitors:
        name = str(comp.get("name") or "")
        if description == name:
            return comp.get("id")
    return None


def _line_from_spread_outcome(description: str, strike: float | None) -> float | None:
    match = _SPREAD_LINE_RE.search(str(description).strip())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    if strike is not None:
        return float(strike)
    return None


def _side_row(
    outcome: dict[str, Any],
    opposite: dict[str, Any],
    *,
    line: float | None = None,
    competitors: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    quote = outcome_quote(outcome, opposite)
    if quote is None:
        return None
    description = str(outcome.get("description") or "")
    return {
        "name": description,
        "competitor_id": _competitor_id_for_outcome(description, competitors),
        "american": quote["american"],
        "line": line,
        "stake": quote["stake"],
    }


def _rows_from_outcomes(
    market: dict[str, Any],
    *,
    line_for_outcome: Any | None = None,
    competitors: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    outcomes = _market_outcomes(market)
    if len(outcomes) < 2:
        return []
    rows: list[dict[str, Any]] = []
    for idx, outcome in enumerate(outcomes):
        opposite = outcomes[1 - idx]
        line: float | None = None
        if line_for_outcome is not None:
            line = line_for_outcome(outcome, market)
        row = _side_row(outcome, opposite, line=line, competitors=competitors)
        if row is not None:
            rows.append(row)
    return rows


def _moneyline_rows(market: dict[str, Any]) -> list[dict[str, Any]]:
    return _rows_from_outcomes(market)


def _spread_rows(market: dict[str, Any]) -> list[dict[str, Any]]:
    strike_raw = market.get("strike")
    strike: float | None
    try:
        strike = float(strike_raw) if strike_raw is not None else None
    except (TypeError, ValueError):
        strike = None

    def line_for_outcome(outcome: dict[str, Any], _market: dict[str, Any]) -> float | None:
        return _line_from_spread_outcome(str(outcome.get("description") or ""), strike)

    return _rows_from_outcomes(market, line_for_outcome=line_for_outcome)


def _total_rows(market: dict[str, Any]) -> list[dict[str, Any]]:
    strike_raw = market.get("strike")
    try:
        line = float(strike_raw) if strike_raw is not None else None
    except (TypeError, ValueError):
        line = None

    def line_for_outcome(_outcome: dict[str, Any], _market: dict[str, Any]) -> float | None:
        return line

    return _rows_from_outcomes(market, line_for_outcome=line_for_outcome)


def extract_team_markets(markets: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for market in markets:
        if not isinstance(market, dict):
            continue
        if market.get("type") == "MONEY" and _both_sides_available(market):
            rows = _moneyline_rows(market)
            if rows:
                out["moneyline"] = rows
            break

    spread = pick_main_spread(markets)
    if spread:
        rows = _spread_rows(spread)
        if rows:
            out["run_line"] = rows

    total = pick_main_total(markets)
    if total:
        rows = _total_rows(total)
        if rows:
            out["total"] = rows

    return out


def _max_events_cap() -> int | None:
    raw = os.environ.get("NOVIG_MAX_EVENTS", "").strip()
    if raw.isdigit():
        return int(raw)
    return None


def _graphql_post(
    session: requests.Session,
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    retries: int = 3,
    operation_name: str | None = None,
) -> Any:
    payload: dict[str, Any] = {"query": query}
    if operation_name:
        payload["operationName"] = operation_name
    if variables is not None:
        payload["variables"] = variables
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.post(
                GRAPHQL_URL,
                json=payload,
                headers=DEFAULT_HEADERS,
                timeout=60,
            )
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


def graphql(
    session: requests.Session,
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    retries: int = 3,
    operation_name: str | None = None,
) -> Any:
    body = _graphql_post(
        session,
        query,
        variables,
        retries=retries,
        operation_name=operation_name,
    )
    if body.get("errors"):
        messages = [
            str(err.get("message") or err)
            for err in body["errors"]
            if isinstance(err, dict)
        ]
        if body.get("data"):
            logger.warning(
                "GraphQL partial errors: %s",
                "; ".join(messages or ["unknown error"]),
            )
        else:
            raise RuntimeError(
                "GraphQL errors: " + "; ".join(messages or ["unknown error"])
            )
    return body


def _events_from_graphql_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    events = data.get("event") or []
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _event_from_game_card(card: dict[str, Any]) -> dict[str, Any]:
    home = card.get("homeTeam") if isinstance(card.get("homeTeam"), dict) else {}
    away = card.get("awayTeam") if isinstance(card.get("awayTeam"), dict) else {}
    scoreboard = (
        card.get("scoreboard") if isinstance(card.get("scoreboard"), dict) else {}
    )
    is_live = bool(card.get("isLive"))
    status = str(
        scoreboard.get("status")
        or ("OPEN_INGAME" if is_live else "OPEN_PREGAME")
    )
    title = str(scoreboard.get("title") or "").strip()
    if not title:
        away_name = str(away.get("name") or "").strip()
        home_name = str(home.get("name") or "").strip()
        if away_name and home_name:
            title = f"{away_name} @ {home_name}"
    return {
        "id": card.get("eventId"),
        "description": title,
        "status": status,
        "game": {
            "scheduled_start": card.get("scheduledStart"),
            "league": card.get("league") or "MLB",
            "homeTeam": {"id": home.get("id"), "name": home.get("name")},
            "awayTeam": {"id": away.get("id"), "name": away.get("name")},
        },
    }


def events_from_trading_page(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    events: list[dict[str, Any]] = []
    for section in payload.get("sections") or []:
        if not isinstance(section, dict):
            continue
        content = section.get("content")
        if not isinstance(content, dict):
            continue
        for card in content.get("components") or []:
            if isinstance(card, dict) and card.get("type") == "game_event_card":
                events.append(_event_from_game_card(card))
    return events


def fetch_mlb_events(session: requests.Session) -> list[dict[str, Any]]:
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(
                TRADING_PAGE_URL,
                headers=DEFAULT_HEADERS,
                timeout=60,
            )
            if resp.status_code in (429, 500, 502, 503, 504) and attempt + 1 < 3:
                time.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            events = events_from_trading_page(resp.json())
            cap = _max_events_cap()
            if cap is not None:
                events = events[:cap]
            return events
        except (requests.RequestException, ValueError) as exc:
            last_err = exc
            if attempt + 1 >= 3:
                break
            time.sleep(0.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def fetch_event_markets(
    session: requests.Session,
    event_id: str,
) -> list[dict[str, Any]]:
    payload = graphql(
        session,
        _EVENT_MARKETS_QUERY,
        {
            "eventId": event_id,
            "marketVisibleWhere": _MARKET_VISIBLE_WHERE,
        },
        operation_name=EVENT_MARKETS_OPERATION,
    )
    events = _events_from_graphql_payload(payload)
    if not events:
        return []
    markets = events[0].get("markets") or []
    if not isinstance(markets, list):
        return []
    return [market for market in markets if isinstance(market, dict)]


def build_game_snapshots(
    events: list[dict[str, Any]],
    markets_by_event_id: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    props_games: list[dict[str, Any]] = []
    team_games: list[dict[str, Any]] = []
    for event in events:
        base = normalize_event(event)
        event_id = base.get("event_id")
        if event_id is None:
            continue
        markets = markets_by_event_id.get(str(event_id), [])
        props_games.append({**base, "props": extract_props(markets)})
        team_games.append(
            {**base, "team_markets": extract_team_markets(markets)}
        )
    return props_games, team_games


def _payload_base(*, fetched_at: str) -> dict[str, Any]:
    return {
        "source": "novig",
        "fetched_at": fetched_at,
        "league": "mlb",
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


def _count_usable_quotes(
    props_games: list[dict[str, Any]],
    team_games: list[dict[str, Any]],
) -> tuple[int, int]:
    n_props = sum(len(g.get("props") or []) for g in props_games)
    n_team = sum(len(g.get("team_markets") or {}) for g in team_games)
    return n_props, n_team


def selenium_fallback_enabled() -> bool:
    return os.environ.get("NOVIG_ALLOW_SELENIUM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def fetch_via_selenium() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    raise RuntimeError(
        "Novig Selenium fallback is not implemented yet; "
        "public REST /nbx/v1/trading/MLB/page plus allowlisted "
        "EventMarkets_Query should work without auth. "
        "Set NOVIG_ALLOW_SELENIUM only after implementing CDP capture."
    )


def _fetch_graphql_snapshots(
    session: requests.Session,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events = fetch_mlb_events(session)
    markets_by_event_id: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        event_id = event.get("id")
        if event_id is None:
            continue
        markets_by_event_id[str(event_id)] = fetch_event_markets(
            session, str(event_id)
        )
    props_games, team_games = build_game_snapshots(events, markets_by_event_id)
    return events, props_games, team_games


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
    events: list[dict[str, Any]] = []
    props_games: list[dict[str, Any]] = []
    team_games: list[dict[str, Any]] = []
    graphql_failed = False

    try:
        events, props_games, team_games = _fetch_graphql_snapshots(session)
    except Exception as exc:
        graphql_failed = True
        logger.error("GraphQL fetch failed: %s", exc)

    n_props, n_team = _count_usable_quotes(props_games, team_games)
    needs_fallback = graphql_failed or (
        bool(events) and n_props == 0 and n_team == 0
    )

    if needs_fallback:
        if selenium_fallback_enabled():
            logger.warning("Attempting Selenium fallback for Novig MLB...")
            try:
                events, markets_by_event_id = fetch_via_selenium()
                props_games, team_games = build_game_snapshots(
                    events, markets_by_event_id
                )
                n_props, n_team = _count_usable_quotes(props_games, team_games)
            except Exception as exc:
                logger.error("Selenium fallback failed: %s", exc)
                sys.exit(1)
            if n_props == 0 and n_team == 0:
                logger.error("Selenium fallback returned no usable quotes")
                sys.exit(1)
        else:
            if graphql_failed:
                logger.error(
                    "Novig GraphQL fetch failed and NOVIG_ALLOW_SELENIUM is not set"
                )
            else:
                logger.error(
                    "Novig GraphQL returned %s events but zero usable quotes; "
                    "set NOVIG_ALLOW_SELENIUM to attempt browser fallback",
                    len(events),
                )
            sys.exit(1)

    props_path = resolve_props_output_path()
    props_path, team_path = write_snapshots(
        props_games, team_games, props_path=props_path
    )
    logger.info(
        "Wrote Novig snapshots: props_games=%s team_games=%s props_quotes=%s "
        "team_markets=%s props=%s team=%s",
        len(props_games),
        len(team_games),
        n_props,
        n_team,
        props_path,
        team_path,
    )
    load_supabase_snapshots(
        props_games,
        team_games,
        props_path=props_path,
        team_path=team_path,
    )


def load_supabase_snapshots(
    props_games: list[dict[str, Any]],
    team_games: list[dict[str, Any]],
    *,
    scraped_at: datetime | None = None,
    props_path: str | None = None,
    team_path: str | None = None,
) -> None:
    """Upsert snapshot games to odds.mlb_novig / odds.mlb_novig_team."""
    try:
        from src.odds.load_snapshots import (
            load_novig_props_snapshot,
            load_novig_team_snapshot,
        )

        when = scraped_at or datetime.now(timezone.utc)
        n_props = load_novig_props_snapshot(
            props_games, league="mlb", scraped_at=when
        )
        n_team = load_novig_team_snapshot(
            team_games, league="mlb", scraped_at=when
        )
        logger.info(
            "Supabase Novig upserted props=%s team=%s%s%s",
            n_props,
            n_team,
            f" props_path={props_path}" if props_path else "",
            f" team_path={team_path}" if team_path else "",
        )
    except Exception as exc:
        logger.error("Supabase Novig load failed (JSON kept): %s", exc)


if __name__ == "__main__":
    run()
