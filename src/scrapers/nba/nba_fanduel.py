"""
FanDuel Sportsbook prop odds scraper — NBA edition.

Fetches NBA player prop markets from the FanDuel Sportsbook API and exports clean JSON.

Includes detailed logging at each step to make debugging easy.

Usage:

    python -m src.scrapers.nba.nba_fanduel

Environment variables:

    FANDUEL_OUTPUT: Override default output path (must end in .json);
                    sport slug is appended when saving multiple leagues
    FANDUEL_REGION: Sportsbook region subdomain, e.g. ny, nj, pa (default: ny)
    FANDUEL_MAX_EVENTS: Only scrape the first N events (handy while debugging)
    FANDUEL_DUMP_RAW: Set to 1 to write raw API payloads under <output_dir>/raw
    LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import requests

# ============================================================================
# Configuration
# ============================================================================

try:
    from .paths import ensure_repo_on_path
except ImportError:  # python nba_fanduel.py from this directory
    from paths import ensure_repo_on_path

_ROOT = str(ensure_repo_on_path(__file__))

_DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "data", "props", "fanduel")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

_DEFAULT_CONFIG: dict[str, Any] = {
    "sport_allowlist": ["NBA"],
    "fd_region": "ny",
    # Public web-client key. FanDuel rotates it occasionally; override in config.json.
    "fd_api_key": "FhMFpcPWXMeyZxOx",
    "fd_page_id": "nba",
    "fd_competition_names": ["NBA"],
    # event-page only returns the markets for the requested tab, so props need their own pass.
    "fd_event_tabs": [
        "player-points",
        "player-rebounds",
        "player-assists",
        "player-threes",
        "player-combos",
        "player-defense",
    ],
    "fd_request_delay": 0.4,
    "fd_max_retries": 3,
    "fd_timeout": 30,
    "headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://sportsbook.fanduel.com/",
        "Origin": "https://sportsbook.fanduel.com",
    },
}

# Query params the sportsbook web client sends on every request.
_BASE_PARAMS: dict[str, str] = {
    "betexRegion": "GBR",
    "capiJurisdiction": "intl",
    "currencyCode": "USD",
    "exchangeLocale": "en_GB",
    "includePrices": "true",
    "language": "en",
    "regionCode": "NAMERICA",
    "timezone": "America/New_York",
}

# "Yes" on a milestone market is the same wager as over 0.5, so mapping
# yes/no onto over/under makes FanDuel rows directly comparable to Underdog rows.
_CHOICE_MAP = {"over": "over", "under": "under", "yes": "over", "no": "under"}

# Order matters — first match wins, so combo stats precede the components they contain.
_STAT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?:points?|pts)\s*\+\s*(?:rebounds?|rebs?)\s*\+\s*(?:assists?|asts?)"
        ),
        "points_rebounds_assists",
    ),
    (re.compile(r"(?:points?|pts)\s*\+\s*(?:rebounds?|rebs?)"), "points_rebounds"),
    (re.compile(r"(?:points?|pts)\s*\+\s*(?:assists?|asts?)"), "points_assists"),
    (re.compile(r"(?:rebounds?|rebs?)\s*\+\s*(?:assists?|asts?)"), "rebounds_assists"),
    (re.compile(r"steals?\s*\+\s*blocks?"), "steals_blocks"),
    (
        re.compile(
            r"three[- ]?point(?:er)?s?(?:\s+made)?|3[- ]?point(?:er)?s?(?:\s+made)?"
            r"|made\s+threes|\bthrees\b"
        ),
        "three_pointers_made",
    ),
    (re.compile(r"\bturnovers?\b"), "turnovers"),
    (re.compile(r"\bsteals?\b"), "steals"),
    (re.compile(r"\bblocks?\b"), "blocks"),
    (re.compile(r"\brebounds?\b|\brebs?\b"), "rebounds"),
    (re.compile(r"\bassists?\b|\basts?\b"), "assists"),
    (re.compile(r"\bpoints?\b|\bpts\b"), "points"),
)

# Game-level markets that would otherwise satisfy a stat pattern.
# Player markets include a name ("Jokic - Total Points"), so only bare totals match.
_GAME_MARKET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"money\s*line|\bspread\b|handicap|point\s+spread"),
    re.compile(r"^(?:alternate\s+)?(?:total\s+points|game\s+total)(?:\s*\([^)]*\))?$"),
    re.compile(r"team\s+total|over/under"),
    re.compile(r"\bto\s+win\b|winning\s+margin|\bseries\b|\bdraw\b"),
    re.compile(r"\bquarter\b|\b1st\s+half\b|\bfirst\s+half\b|\bhalftime\b"),
    re.compile(r"both\s+teams|either\s+team|\brace\s+to\b|\bmargin\b"),
)

_NAME_SPLIT = re.compile(r"\s+[-–|]\s+")
_TEAM_SPLIT = re.compile(r"\s+(?:@|v|vs\.?|at)\s+", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

_DUMP_RAW = os.environ.get("FANDUEL_DUMP_RAW", "").strip().lower() in ("1", "true", "yes")

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging with timestamp and level indicators."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="[%(levelname)-8s] %(name)s: %(message)s",
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Pick:
    """A single prop price with all public fields."""

    full_name: str
    stat_name: str
    stat_value: float | None
    updated_at: str | None
    choice: str
    american_price: int | None
    decimal_price: float | None
    implied_probability: float | None
    market_name: str = ""
    event_name: str = ""
    start_time: str | None = None
    sport_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization (omit internal sport_id)."""
        data = asdict(self)
        data.pop("sport_id", None)
        return data

@dataclass
class Event:
    """One scheduled game to pull markets for."""

    event_id: str
    name: str
    start_time: str | None
    competition: str

@dataclass
class ExportData:
    """Complete export payload."""

    source: str = "FanDuel Sportsbook"
    fetched_at: str = ""
    count: int = 0
    picks: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.picks is None:
            self.picks = []
        self.count = len(self.picks) if self.picks else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "fetched_at": self.fetched_at,
            "count": self.count,
            "picks": self.picks or [],
        }

# ============================================================================
# Configuration Loading
# ============================================================================

def load_config() -> dict[str, Any]:
    """Load and merge default + user config."""
    cfg = {**_DEFAULT_CONFIG}
    # Copy the headers dict so a config load never mutates the module-level defaults.
    cfg["headers"] = {**_DEFAULT_CONFIG["headers"]}

    if os.path.isfile(_CONFIG_PATH):
        logger.info(f"Loading config from: {_CONFIG_PATH}")
        try:
            with open(_CONFIG_PATH, encoding="utf-8-sig") as f:
                user_cfg = json.load(f)
            logger.debug(f"User config keys: {list(user_cfg.keys())}")

            user_headers = user_cfg.pop("headers", None)
            cfg.update(user_cfg)
            if isinstance(user_headers, dict):
                cfg["headers"].update(user_headers)

            logger.info("Config merged successfully")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
    else:
        logger.info(f"No config file found at {_CONFIG_PATH}. Using defaults.")

    region = os.environ.get("FANDUEL_REGION", "").strip()
    if region:
        logger.info(f"Overriding region from FANDUEL_REGION: {region}")
        cfg["fd_region"] = region

    return cfg

def get_sport_allowlist(cfg: dict[str, Any]) -> frozenset[str] | None:
    """
    Parse sport allowlist from config.

    Returns:
        Frozenset of sport IDs to keep, or None to disable filtering.
    """
    raw = cfg.get("sport_allowlist", ["NBA"])
    if raw is None:
        logger.info("Sport allowlist is None — will keep all sports")
        return None

    result = frozenset(str(x) for x in raw)
    logger.info(f"Sport allowlist: {result}")
    return result

# ============================================================================
# Payload Helpers
# ============================================================================

def _dig(obj: Any, *path: str) -> Any:
    """Walk nested dict keys, returning None as soon as the path breaks."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj

def _iter_records(node: Any) -> Iterator[dict[str, Any]]:
    """Yield dict records from a FanDuel attachment, which may be a dict or a list."""
    if isinstance(node, dict):
        values: Any = node.values()
    elif isinstance(node, list):
        values = node
    else:
        return
    for value in values:
        if isinstance(value, dict):
            yield value


def _competition_names(payload: dict[str, Any]) -> dict[str, str]:
    """Map competitionId → name from the page attachments (events omit the name)."""
    names: dict[str, str] = {}
    for rec in _iter_records(_dig(payload, "attachments", "competitions")):
        cid = rec.get("competitionId") or rec.get("id")
        name = rec.get("name") or rec.get("competitionName")
        if cid is not None and name:
            names[str(cid)] = str(name)
    return names


def _event_competition_name(rec: dict[str, Any], competitions: dict[str, str]) -> str:
    """Prefer the event's own competitionName; otherwise join via competitionId."""
    name = rec.get("competitionName")
    if name:
        return str(name)
    cid = rec.get("competitionId")
    if cid is not None:
        return competitions.get(str(cid), "")
    return ""

def _to_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _first_number(text: str) -> float | None:
    match = _NUMBER.search(text or "")
    return float(match.group()) if match else None

def _maybe_dump(payload: dict[str, Any], name: str) -> None:
    """Write a raw payload to disk when FANDUEL_DUMP_RAW is set."""
    if not _DUMP_RAW:
        return
    raw_dir = os.path.join(_DEFAULT_OUTPUT_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.debug(f"Dumped raw payload: {path}")

def american_to_decimal(american: int) -> float | None:
    """Convert American odds to decimal (total return per unit staked)."""
    if not american:
        return None
    if american > 0:
        return round(1.0 + american / 100.0, 4)
    return round(1.0 + 100.0 / abs(american), 4)

def decimal_to_american(decimal: float) -> int | None:
    """Convert decimal odds to American odds."""
    if decimal <= 1.0:
        return None
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    return int(round(-100.0 / (decimal - 1.0)))

# ============================================================================
# API Fetching
# ============================================================================

def _api_base(region: str) -> str:
    return f"https://sbapi.{(region or 'ny').strip().lower()}.sportsbook.fanduel.com/api"

def build_session(headers: dict[str, str]) -> requests.Session:
    session = requests.Session()
    session.headers.update(headers)
    return session

def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, str],
    *,
    timeout: int,
    max_retries: int,
    label: str,
) -> dict[str, Any]:
    """
    GET a FanDuel endpoint and return parsed JSON, retrying transient failures.

    Raises:
        requests.RequestException: On HTTP error after the final attempt
        ValueError: On invalid JSON after the final attempt
    """
    logger.info(f"Fetching {label}: {url}")
    last_error: Exception | None = None

    for attempt in range(1, max(1, max_retries) + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(
                    f"HTTP {response.status_code} (transient)", response=response
                )
            response.raise_for_status()
            data = response.json()
            logger.debug(f"{label}: status={response.status_code}, keys={list(data.keys())}")
            return data
        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt >= max(1, max_retries):
                break
            backoff = 2 ** (attempt - 1)
            logger.warning(
                f"{label}: attempt {attempt}/{max_retries} failed ({e}); retrying in {backoff}s"
            )
            time.sleep(backoff)

    logger.error(f"{label}: giving up after {max_retries} attempt(s): {last_error}")
    assert last_error is not None
    raise last_error

def fetch_events(session: requests.Session, cfg: dict[str, Any]) -> list[Event]:
    """Discover today's NBA events from the sportsbook's managed NBA page."""
    params = {
        **_BASE_PARAMS,
        "page": "CUSTOM",
        "customPageId": str(cfg.get("fd_page_id") or "nba"),
        "pbHorizontal": "false",
        "_ak": str(cfg.get("fd_api_key") or ""),
    }
    payload = fetch_json(
        session,
        f"{_api_base(cfg['fd_region'])}/content-managed-page",
        params,
        timeout=int(cfg.get("fd_timeout", 30)),
        max_retries=int(cfg.get("fd_max_retries", 3)),
        label=f"NBA page ({cfg.get('fd_page_id')})",
    )
    _maybe_dump(payload, f"page_{cfg.get('fd_page_id')}")

    records = list(_iter_records(_dig(payload, "attachments", "events")))
    markets = list(_iter_records(_dig(payload, "attachments", "markets")))
    competitions = _competition_names(payload)
    logger.info(f"Payload shape: events={len(records)}, markets={len(markets)}")

    allow = {str(c).upper() for c in (cfg.get("fd_competition_names") or [])}
    events: list[Event] = []
    skipped_competition = 0

    for rec in records:
        event_id = rec.get("eventId") or rec.get("id")
        if not event_id:
            continue
        competition = _event_competition_name(rec, competitions)
        if allow and competition.upper() not in allow:
            skipped_competition += 1
            continue
        events.append(
            Event(
                event_id=str(event_id),
                name=str(rec.get("name") or ""),
                start_time=rec.get("openDate") or rec.get("startTime") or rec.get("eventTime"),
                competition=competition or "NBA",
            )
        )

    events.sort(key=lambda e: (e.start_time or "", e.name))
    logger.info(f"Discovered {len(events)} events (skipped_competition={skipped_competition})")

    cap = os.environ.get("FANDUEL_MAX_EVENTS", "").strip()
    if cap.isdigit() and int(cap) > 0:
        logger.info(f"FANDUEL_MAX_EVENTS set — limiting to first {cap} event(s)")
        events = events[: int(cap)]

    return events

def fetch_event_markets(
    session: requests.Session,
    cfg: dict[str, Any],
    event: Event,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Fetch one event's markets, merged across every configured event-page tab.

    Returns:
        (markets, event_record) — markets deduped by market ID
    """
    url = f"{_api_base(cfg['fd_region'])}/event-page"
    timeout = int(cfg.get("fd_timeout", 30))
    max_retries = int(cfg.get("fd_max_retries", 3))
    delay = float(cfg.get("fd_request_delay", 0.4) or 0)

    markets: dict[str, dict[str, Any]] = {}
    event_record: dict[str, Any] = {}
    tabs: list[str | None] = list(cfg.get("fd_event_tabs") or []) or [None]

    for tab in tabs:
        params = {**_BASE_PARAMS, "eventId": event.event_id, "_ak": str(cfg.get("fd_api_key") or "")}
        if tab:
            params["tab"] = tab
        try:
            payload = fetch_json(
                session,
                url,
                params,
                timeout=timeout,
                max_retries=max_retries,
                label=f"event {event.event_id} tab={tab or 'default'}",
            )
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"event {event.event_id}: tab {tab!r} unavailable ({e}); continuing")
            continue

        _maybe_dump(payload, f"event_{event.event_id}_{tab or 'default'}")

        new_markets = 0
        for market in _iter_records(_dig(payload, "attachments", "markets")):
            market_id = str(market.get("marketId") or market.get("id") or "")
            if market_id and market_id not in markets:
                markets[market_id] = market
                new_markets += 1
        for rec in _iter_records(_dig(payload, "attachments", "events")):
            if str(rec.get("eventId") or rec.get("id") or "") == event.event_id:
                event_record = rec

        logger.debug(f"event {event.event_id} tab={tab or 'default'}: +{new_markets} markets")
        if delay:
            time.sleep(delay)

    return list(markets.values()), event_record

# ============================================================================
# Data Processing
# ============================================================================

def match_stat(text: str) -> str | None:
    """Map a market name fragment onto a canonical stat name."""
    lowered = (text or "").lower()
    for pattern, stat in _STAT_PATTERNS:
        if pattern.search(lowered):
            return stat
    return None

def is_game_market(market_name: str) -> bool:
    """True for game-level markets (moneyline, totals, innings) rather than props."""
    lowered = (market_name or "").lower()
    return any(pattern.search(lowered) for pattern in _GAME_MARKET_PATTERNS)

def split_market_name(market_name: str) -> tuple[str, str]:
    """
    Split a market name into (player, stat phrase).

    FanDuel writes both "Aaron Judge - Total Bases" and "Alternate Total Bases -
    Aaron Judge", so the stat side is whichever half a stat pattern matches; the
    trailing half wins ties because player names occasionally contain stat words.
    """
    parts = [p.strip() for p in _NAME_SPLIT.split(market_name or "") if p.strip()]
    if len(parts) < 2:
        return "", (market_name or "").strip()

    candidates = (
        (parts[0], " ".join(parts[1:])),
        (parts[-1], " ".join(parts[:-1])),
    )
    for player, stat in candidates:
        if match_stat(stat) and not match_stat(player):
            return player, stat
    for player, stat in candidates:
        if match_stat(stat):
            return player, stat
    return "", (market_name or "").strip()

def parse_runner_selection(runner_name: str, handicap: Any) -> tuple[str, float | None]:
    """
    Derive (choice, stat_value) from a runner.

    An empty choice means the runner names a player instead of a side, which is how
    FanDuel models player-listed markets such as "To Record a Double Double".
    """
    text = (runner_name or "").strip()
    lowered = text.lower()
    line = _to_float(handicap)

    if lowered in ("yes", "no"):
        # A yes/no prop is a 0.5 line: "yes" pays if the player records at least one.
        return _CHOICE_MAP[lowered], line if line is not None else 0.5

    for token in ("over", "under"):
        if (
            lowered == token
            or lowered.startswith(f"{token} ")
            or lowered.endswith(f" {token}")
        ):
            value = _first_number(text)
            return _CHOICE_MAP[token], value if value is not None else line

    # "2+ Hits" style alternates: the equivalent line is one half below the threshold.
    threshold = re.fullmatch(r"(\d+(?:\.\d+)?)\s*\+", text)
    if threshold:
        return "over", float(threshold.group(1)) - 0.5

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return "exact", float(text)

    return "", line

def extract_prices(runner: dict[str, Any]) -> tuple[int | None, float | None, float | None]:
    """
    Pull (american, decimal, implied_probability) from a runner's odds block.

    The implied probability is vig-inclusive — it is the book's price, not a fair
    probability, so it sits above the true number by roughly the hold.
    """
    odds = runner.get("winRunnerOdds") or {}

    american = _to_float(_dig(odds, "americanDisplayOdds", "americanOddsInt"))
    if american is None:
        american = _to_float(_dig(odds, "trueOdds", "americanDisplayOdds", "americanOddsInt"))
    if american is None:
        american = _first_number(str(_dig(odds, "americanDisplayOdds", "americanOdds") or ""))

    decimal = _to_float(_dig(odds, "trueOdds", "decimalOdds", "decimalOdds"))
    if decimal is None:
        decimal = _to_float(_dig(odds, "decimalOdds", "decimalOdds"))
    if decimal is None:
        decimal = _to_float(odds.get("decimalOdds"))

    american_int = int(american) if american is not None else None
    if decimal is None and american_int is not None:
        decimal = american_to_decimal(american_int)
    if american_int is None and decimal is not None:
        american_int = decimal_to_american(decimal)

    implied = round(1.0 / decimal, 6) if decimal and decimal > 1.0 else None
    return american_int, decimal, implied

def team_names(event_name: str) -> set[str]:
    """Team names from an event title, used to reject team runners in prop markets."""
    return {p.strip().lower() for p in _TEAM_SPLIT.split(event_name or "") if p.strip()}

def looks_like_person(name: str, teams: set[str]) -> bool:
    """Reject team names, side labels and other non-player runner text."""
    cleaned = (name or "").strip()
    if len(cleaned) < 3 or any(c.isdigit() for c in cleaned):
        return False
    if cleaned.lower() in teams:
        return False
    if any(cleaned.lower() in team for team in teams):
        return False
    return 2 <= len(cleaned.split()) <= 5

def extract_event_picks(
    markets: list[dict[str, Any]],
    event: Event,
    event_record: dict[str, Any],
    fetched_at: str,
    sport_id: str,
    counters: dict[str, int],
    unmatched: set[str],
) -> list[Pick]:
    """Flatten one event's prop markets into individual picks."""
    event_name = str(event_record.get("name") or event.name or "")
    start_time = (
        event_record.get("openDate")
        or event_record.get("startTime")
        or event.start_time
    )
    teams = team_names(event_name)
    picks: list[Pick] = []

    for market in markets:
        market_name = str(market.get("marketName") or "")

        if str(market.get("marketStatus") or "").upper() in ("SUSPENDED", "CLOSED", "RESULTED"):
            counters["suspended_markets"] += 1
            continue
        if is_game_market(market_name):
            counters["game_markets"] += 1
            continue

        market_player, stat_phrase = split_market_name(market_name)
        stat_name = match_stat(stat_phrase)
        if not stat_name:
            counters["no_stat"] += 1
            unmatched.add(market_name)
            continue

        updated_at = market.get("lastUpdated") or market.get("marketTime") or fetched_at

        for runner in _iter_records(market.get("runners")):
            if str(runner.get("runnerStatus") or "").upper() in ("SUSPENDED", "CLOSED", "REMOVED"):
                counters["suspended_runners"] += 1
                continue

            runner_name = str(runner.get("runnerName") or "")
            choice, stat_value = parse_runner_selection(runner_name, runner.get("handicap"))

            if choice == "exact":
                counters["exact_markets"] += 1
                continue

            if choice:
                full_name = market_player
            else:
                # Player-listed market: the runner is the player and the bet is "yes".
                full_name = runner_name
                choice = "over"
                if stat_value is None:
                    stat_value = 0.5

            if not looks_like_person(full_name, teams):
                counters["no_player"] += 1
                continue

            american, decimal, implied = extract_prices(runner)
            if american is None and decimal is None:
                counters["no_price"] += 1
                continue

            picks.append(
                Pick(
                    full_name=full_name,
                    stat_name=stat_name,
                    stat_value=stat_value,
                    updated_at=updated_at,
                    choice=choice,
                    american_price=american,
                    decimal_price=decimal,
                    implied_probability=implied,
                    market_name=market_name,
                    event_name=event_name,
                    start_time=start_time,
                    sport_id=sport_id,
                )
            )
            counters["added"] += 1

    return picks

def dedupe_picks(picks: list[Pick]) -> list[Pick]:
    """
    Collapse duplicate prices for the same wager.

    Standard and "Alternate" markets overlap — over 27.5 points is also listed as
    28+ — so keep one row per wager, preferring the standard market's wording.
    """
    best: dict[tuple[str, str, float | None, str, str], Pick] = {}
    dropped = 0

    for pick in picks:
        key = (pick.full_name, pick.stat_name, pick.stat_value, pick.choice, pick.event_name)
        current = best.get(key)
        if current is None:
            best[key] = pick
            continue
        dropped += 1
        if "alternate" in current.market_name.lower() and "alternate" not in pick.market_name.lower():
            best[key] = pick

    if dropped:
        logger.info(f"Deduped {dropped} overlapping alternate-market price(s)")
    return list(best.values())

def extract_picks(
    session: requests.Session,
    cfg: dict[str, Any],
    events: list[Event],
    sport_allowlist: frozenset[str] | None,
) -> list[Pick]:
    """
    Fetch each event's markets and flatten them into picks.

    Args:
        events: Events discovered from the NBA page
        sport_allowlist: Set of sport IDs to keep (None = keep all)

    Returns:
        List of Pick objects
    """
    logger.info(f"Extracting picks from {len(events)} event(s)...")
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    counters = {
        "added": 0,
        "suspended_markets": 0,
        "suspended_runners": 0,
        "game_markets": 0,
        "exact_markets": 0,
        "no_stat": 0,
        "no_player": 0,
        "no_price": 0,
        "skipped_sport": 0,
    }
    unmatched: set[str] = set()
    picks: list[Pick] = []

    for index, event in enumerate(events, start=1):
        sport_id = (event.competition or "NBA").upper()
        if sport_allowlist is not None and sport_id not in sport_allowlist:
            counters["skipped_sport"] += 1
            continue

        logger.info(f"[{index}/{len(events)}] {event.name or event.event_id}")
        markets, event_record = fetch_event_markets(session, cfg, event)
        if not markets:
            logger.warning(f"event {event.event_id}: no markets returned")
            continue

        event_picks = extract_event_picks(
            markets, event, event_record, fetched_at, sport_id, counters, unmatched
        )
        logger.info(f"  markets={len(markets)}, picks={len(event_picks)}")
        picks.extend(event_picks)

    picks = dedupe_picks(picks)

    logger.info(
        f"Extraction complete: added={counters['added']}, kept={len(picks)}, "
        f"game_markets={counters['game_markets']}, exact={counters['exact_markets']}, "
        f"suspended_markets={counters['suspended_markets']}, "
        f"suspended_runners={counters['suspended_runners']}, "
        f"no_stat={counters['no_stat']}, no_player={counters['no_player']}, "
        f"no_price={counters['no_price']}, skipped_sport={counters['skipped_sport']}"
    )
    if unmatched:
        sample = sorted(unmatched)[:25]
        logger.debug(f"Market names with no stat match ({len(unmatched)} distinct): {sample}")

    return picks

# ============================================================================
# File Export
# ============================================================================

def resolve_output_path(sport: str) -> str:
    """
    Resolve output file path for a single sport.

    Checks:
    1. FANDUEL_OUTPUT env var (if .json file) — used as-is for a single sport
    2. Default: data/props/fanduel/fanduel_{sport}_YYYY-MM-DD_HHMMSS.json

    Returns:
        Absolute path to output file
    """
    env_path = os.environ.get("FANDUEL_OUTPUT", "").strip()
    sport_slug = sport.strip().lower() or "unknown"

    if env_path and env_path.lower().endswith(".json"):
        expanded = os.path.expanduser(env_path)
        if not expanded.endswith(("/", "\\")) and not os.path.isdir(expanded):
            # When env points at a single file and we have multiple sports, stamp sport in.
            root, ext = os.path.splitext(expanded)
            if sport_slug not in os.path.basename(root).lower():
                expanded = f"{root}_{sport_slug}{ext}"
            logger.info(f"Using FANDUEL_OUTPUT: {expanded}")
            return expanded

    now = datetime.now(_OUTPUT_TZ)
    filename = now.strftime(f"fanduel_{sport_slug}_%Y-%m-%d_%H%M%S.json")
    path = os.path.join(_DEFAULT_OUTPUT_DIR, filename)
    logger.info(f"Using default output path: {path}")
    return path

def group_picks_by_sport(picks: list[Pick]) -> dict[str, list[Pick]]:
    """Group picks by sport_id (empty → unknown)."""
    grouped: dict[str, list[Pick]] = {}
    for pick in picks:
        key = pick.sport_id or "unknown"
        grouped.setdefault(key, []).append(pick)
    return grouped

def sport_to_league(sport: str) -> str | None:
    """Map FanDuel competition name to odds league slug (nba only)."""
    normalized = sport.strip().upper()
    if normalized == "NBA":
        return "nba"
    return None

def save_picks(picks: list[Pick], path: str) -> None:
    """
    Save picks to JSON file.

    Args:
        picks: List of Pick objects
        path: Output file path
    """
    logger.info(f"Saving {len(picks)} picks to {path}...")

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
        logger.debug(f"Ensured directory exists: {parent}")

    pick_dicts = [p.to_dict() for p in picks]
    export = ExportData(picks=pick_dicts)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved successfully to {path}")
    except IOError as e:
        logger.error(f"Failed to write file: {e}")
        raise

# ============================================================================
# Main Scraper
# ============================================================================

class FanDuelScraper:
    """Main scraper orchestrator."""

    def __init__(self) -> None:
        logger.info("Initializing FanDuelScraper...")
        self.config = load_config()
        self.output_paths: list[str] = []
        self.picks: list[Pick] = []
        self.scraped_at: datetime | None = None
        logger.info("Initialization complete")

    def _load_sport_to_supabase(self, sport: str, sport_picks: list[Pick]) -> None:
        """Upsert one sport batch to odds.nba_fanduel; JSON save is already done."""
        league = sport_to_league(sport)
        if league is None:
            logger.info(f"Skipping Supabase load for unmapped sport {sport!r}")
            return
        try:
            from src.odds.load_snapshots import load_fanduel_snapshot

            n = load_fanduel_snapshot(
                [p.to_dict() for p in sport_picks],
                league=league,
                scraped_at=self.scraped_at,
            )
            logger.info(f"Supabase odds.nba_fanduel upserted {n} rows ({sport})")
        except Exception as e:
            logger.error(f"Supabase fanduel load failed (JSON kept): {e}")

    def run(self) -> None:
        """Execute the full scrape pipeline."""
        logger.info("=" * 70)
        logger.info("STARTING FANDUEL SCRAPER (NBA)")
        logger.info("=" * 70)

        try:
            # Step 1: Discover events
            logger.info("\n[Step 1/3] Discovering events...")
            session = build_session(self.config.get("headers", {}))
            events = fetch_events(session, self.config)

            # Step 2: Fetch markets per event and extract
            logger.info("\n[Step 2/3] Extracting picks...")
            sport_allowlist = get_sport_allowlist(self.config)
            self.picks = extract_picks(session, self.config, events, sport_allowlist)

            # Step 3: Save one file per sport (fanduel_nba_*, …)
            logger.info("\n[Step 3/3] Saving to file...")
            self.scraped_at = datetime.now(timezone.utc)
            grouped = group_picks_by_sport(self.picks)
            self.output_paths = []

            if not grouped:
                # Still write an empty NBA file when allowlist includes it, else first allowlist sport
                fallback = "NBA"
                if sport_allowlist:
                    fallback = sorted(sport_allowlist)[0]
                path = resolve_output_path(fallback)
                save_picks([], path)
                self._load_sport_to_supabase(fallback, [])
                self.output_paths.append(path)
            else:
                for sport, sport_picks in sorted(grouped.items()):
                    path = resolve_output_path(sport)
                    save_picks(sport_picks, path)
                    self._load_sport_to_supabase(sport, sport_picks)
                    self.output_paths.append(path)

            logger.info("=" * 70)
            logger.info(
                f"✓ SUCCESS: {len(self.picks)} picks saved across "
                f"{len(self.output_paths)} file(s)"
            )
            logger.info("=" * 70)

        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"✗ FAILED: {type(e).__name__}: {e}")
            logger.error("=" * 70)
            raise

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    scraper = FanDuelScraper()
    scraper.run()