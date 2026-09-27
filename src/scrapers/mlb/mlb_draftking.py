"""
DraftKings Sportsbook prop odds scraper — MLB edition (mirrors underdog_scraper_mlb.py).

Fetches MLB player prop markets from the DraftKings Sportsbook API and exports clean JSON.

Includes detailed logging at each step to make debugging easy.

Usage:

    python draftkings_scraper_mlb.py

Environment variables:

    DK_OUTPUT: Override default output path (must end in .json);
               sport slug is appended when saving multiple leagues
    DK_EVENT_GROUP: Override the MLB event group id (default: 84240)
    DK_LEAGUE_PAGE: Override the MLB sportsbook page used to discover prop tabs
    DK_MAX_SUBCATEGORIES: Only scrape the first N prop subcategories (debugging)
    DK_DUMP_RAW: Set to 1 to write raw API payloads under <output_dir>/raw
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
except ImportError:  # python mlb_draftking.py from this directory
    from paths import ensure_repo_on_path

_ROOT = str(ensure_repo_on_path(__file__))

_DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "data", "props", "draftkings")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

_DEFAULT_CONFIG: dict[str, Any] = {
    "sport_allowlist": ["MLB"],
    # DraftKings groups a whole league under one event group; 84240 is MLB.
    "dk_event_group": "84240",
    "dk_league_page": "https://sportsbook.draftkings.com/leagues/baseball/mlb",
    # Nash controldata BFF (v5 eventgroups on sportsbook-nash-usnj is dead / 403).
    "dk_api_base": (
        "https://sportsbook-nash.draftkings.com/sites/US-SB/api/sportscontent/controldata"
    ),
    # Categories whose subcategories hold player props. Matched case-insensitively as
    # substrings; widen this rather than the stat patterns when a whole tab is missing.
    "dk_category_allowlist": ["batter", "pitcher", "prop", "home run", "strikeout"],
    "dk_request_delay": 0.4,
    "dk_max_retries": 3,
    "dk_timeout": 30,
    "headers": {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.7778.280 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://sportsbook.draftkings.com/",
        "Origin": "https://sportsbook.draftkings.com",
    },
}

_BASE_PARAMS: dict[str, str] = {"format": "json"}

# "Yes" on a "to hit a home run" market is the same wager as over 0.5, so mapping
# yes/no onto over/under makes DraftKings rows directly comparable to Underdog rows.
_CHOICE_MAP = {"over": "over", "under": "under", "yes": "over", "no": "under"}

# Order matters — first match wins, so compound and qualified stats precede the
# components they contain ("hits allowed" before "hits", "earned runs" before "runs").
_STAT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"hits?\s*\+\s*runs?\s*\+\s*rbis?|\bh\+r\+rbi\b"), "hits_runs_rbis"),
    (re.compile(r"extra\s+base\s+hits"), "extra_base_hits"),
    (re.compile(r"total\s+bases"), "total_bases"),
    (re.compile(r"home\s*runs?|\bhomers?\b"), "home_runs"),
    (re.compile(r"stolen\s+bases?"), "stolen_bases"),
    (re.compile(r"strikeouts?|\bk'?s\b"), "strikeouts"),
    (re.compile(r"earned\s+runs?"), "earned_runs_allowed"),
    (re.compile(r"hits\s+allowed"), "hits_allowed"),
    (re.compile(r"walks\s+allowed|bases\s+on\s+balls\s+allowed"), "walks_allowed"),
    (re.compile(r"outs?\s+recorded|pitching\s+outs|innings\s+pitched|\bouts?\b"), "pitching_outs"),
    (re.compile(r"\bsingles?\b"), "singles"),
    (re.compile(r"\bdoubles?\b"), "doubles"),
    (re.compile(r"\btriples?\b"), "triples"),
    (re.compile(r"\brbis?\b|runs?\s+batted\s+in"), "rbis"),
    (re.compile(r"\bwalks?\b|bases\s+on\s+balls"), "walks"),
    (re.compile(r"\bhits?\b"), "hits"),
    (re.compile(r"\bruns?\b|\bto\s+score\b"), "runs"),
)

# Game-level markets that would otherwise satisfy a stat pattern ("total runs" → runs).
_GAME_MARKET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"money\s*line|\brun\s*line\b|\bspread\b|handicap"),
    re.compile(r"total\s+runs|team\s+total|over/under"),
    re.compile(r"\bto\s+win\b|winning\s+margin|\bseries\b|\bdraw\b"),
    re.compile(r"first\s+\d+\s+innings|\binning\b|\binnings\b(?!\s+pitched)"),
    re.compile(r"both\s+teams|either\s+team|\brace\s+to\b|\bmargin\b"),
)

_TEAM_SPLIT = re.compile(r"\s+(?:@|v|vs\.?|at)\s+", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# DraftKings sends American odds as display strings, sometimes with a unicode minus.
_MINUS_CHARS = str.maketrans({"−": "-", "–": "-", "—": "-"})

_DUMP_RAW = os.environ.get("DK_DUMP_RAW", "").strip().lower() in ("1", "true", "yes")

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
    """One scheduled game, used to attach event context to each offer."""

    event_id: str
    name: str
    start_time: str | None
    competition: str

@dataclass
class Subcategory:
    """One prop tab to fetch (category + subcategory pair)."""

    category_id: str
    subcategory_id: str
    category_name: str
    name: str

@dataclass
class ExportData:
    """Complete export payload."""

    source: str = "DraftKings Sportsbook"
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

    group = os.environ.get("DK_EVENT_GROUP", "").strip()
    if group:
        logger.info(f"Overriding event group from DK_EVENT_GROUP: {group}")
        cfg["dk_event_group"] = group

    page = os.environ.get("DK_LEAGUE_PAGE", "").strip()
    if page:
        logger.info(f"Overriding league page from DK_LEAGUE_PAGE: {page}")
        cfg["dk_league_page"] = page

    return cfg

def get_sport_allowlist(cfg: dict[str, Any]) -> frozenset[str] | None:
    """
    Parse sport allowlist from config.

    Returns:
        Frozenset of sport IDs to keep, or None to disable filtering.
    """
    raw = cfg.get("sport_allowlist", ["MLB"])
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
    """Yield dict records from a DraftKings collection, which may be a dict or a list."""
    if isinstance(node, dict):
        values: Any = node.values()
    elif isinstance(node, list):
        values = node
    else:
        return
    for value in values:
        if isinstance(value, dict):
            yield value

def _iter_offers(node: Any) -> Iterator[dict[str, Any]]:
    """
    Yield offers from a subcategory's ``offers`` field.

    DraftKings nests offers one level deeper than everything else — ``offers`` is a
    list of lists, one inner list per event — so this flattens arbitrarily.
    """
    if isinstance(node, dict):
        node = list(node.values())
    if not isinstance(node, list):
        return
    for item in node:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from _iter_offers(item)

def _to_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, str):
            value = value.translate(_MINUS_CHARS).replace(",", "").replace("+", "").strip()
            if not value:
                return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _first_number(text: str) -> float | None:
    match = _NUMBER.search((text or "").translate(_MINUS_CHARS))
    return float(match.group()) if match else None

def _maybe_dump(payload: dict[str, Any], name: str) -> None:
    """Write a raw payload to disk when DK_DUMP_RAW is set."""
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
    GET a DraftKings endpoint and return parsed JSON, retrying transient failures.

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


def fetch_text(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    max_retries: int,
    label: str,
) -> str:
    """GET a page and return the response body, retrying transient failures."""
    logger.info(f"Fetching {label}: {url}")
    last_error: Exception | None = None

    for attempt in range(1, max(1, max_retries) + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(
                    f"HTTP {response.status_code} (transient)", response=response
                )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
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


def _extract_nav_root(html: str) -> dict[str, Any]:
    """Pull the inlined sportsbook nav tree from the league HTML page."""
    marker = '{"id":"NavRoot"'
    start = html.find(marker)
    if start < 0:
        alt = html.find('"id":"NavRoot"')
        if alt < 0:
            raise ValueError("NavRoot not found in league page HTML")
        start = html.rfind("{", 0, alt)
    depth = 0
    for index, char in enumerate(html[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(html[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("NavRoot JSON is not an object")
                return parsed
    raise ValueError("NavRoot JSON unterminated")


def subcategories_from_nav(html: str, allow: list[str]) -> list[Subcategory]:
    """
    Walk the league nav tree and keep player-prop tabs under allowlisted categories.

    Game lines and innings tabs are dropped even when they have a subcategory id.
    Duplicate subcategory ids (e.g. Early Exit copies) keep the first occurrence.
    """
    nav = _extract_nav_root(html)
    allow_l = [str(token).lower() for token in allow if str(token).strip()]
    seen: set[str] = set()
    subcategories: list[Subcategory] = []

    def walk(node: dict[str, Any], category_name: str) -> None:
        title = str(node.get("title") or "")
        seo = str(node.get("seoId") or "")
        params = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
        next_category = category_name
        node_hay = f"{title} {seo}".lower()
        if not params.get("subcategoryId") and allow_l and any(token in node_hay for token in allow_l):
            next_category = title or category_name

        subcategory_id = str(params.get("subcategoryId") or "")
        category_id = str(params.get("categoryId") or "")
        tags = {str(tag).lower() for tag in (node.get("tags") or [])}
        hay = f"{next_category} {title} {seo}".lower()
        in_allow = (not allow_l) or any(token in hay for token in allow_l)
        if (
            subcategory_id
            and category_id
            and subcategory_id not in seen
            and in_allow
            and "playerprops" in tags
        ):
            seen.add(subcategory_id)
            subcategories.append(
                Subcategory(
                    category_id=category_id,
                    subcategory_id=subcategory_id,
                    category_name=next_category or title,
                    name=title,
                )
            )

        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child, next_category)

    walk(nav, "")
    return subcategories


def _selection_player(selection: dict[str, Any]) -> str:
    """Prefer the SEO player name; fall back to the display name."""
    for participant in selection.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        if str(participant.get("type") or "").lower() not in ("", "player"):
            continue
        name = str(participant.get("seoIdentifier") or participant.get("name") or "").strip()
        if name:
            return name
    return ""


def payload_to_offers(
    payload: dict[str, Any],
    *,
    competition: str = "MLB",
) -> tuple[list[dict[str, Any]], dict[str, Event]]:
    """Flatten a Nash markets payload into the offer/outcome shape extract_offer_picks expects."""
    events: dict[str, Event] = {}
    for rec in payload.get("events") or []:
        if not isinstance(rec, dict):
            continue
        event_id = rec.get("id") or rec.get("eventId")
        if not event_id:
            continue
        events[str(event_id)] = Event(
            event_id=str(event_id),
            name=str(rec.get("name") or ""),
            start_time=rec.get("startEventDate") or rec.get("startDate") or rec.get("eventDate"),
            competition=competition,
        )

    by_market: dict[str, list[dict[str, Any]]] = {}
    for selection in payload.get("selections") or []:
        if not isinstance(selection, dict):
            continue
        market_id = str(selection.get("marketId") or "")
        if market_id:
            by_market.setdefault(market_id, []).append(selection)

    offers: list[dict[str, Any]] = []
    for market in payload.get("markets") or []:
        if not isinstance(market, dict):
            continue
        market_id = str(market.get("id") or "")
        outcomes: list[dict[str, Any]] = []
        for selection in by_market.get(market_id, []):
            odds = selection.get("displayOdds") if isinstance(selection.get("displayOdds"), dict) else {}
            outcomes.append(
                {
                    "label": str(selection.get("label") or ""),
                    "line": (
                        selection.get("milestoneValue")
                        if selection.get("milestoneValue") is not None
                        else selection.get("points")
                    ),
                    "oddsAmerican": odds.get("american"),
                    "oddsDecimal": odds.get("decimal") or selection.get("trueOdds"),
                    "participant": _selection_player(selection),
                }
            )
        offers.append(
            {
                "label": str(market.get("name") or ""),
                "eventId": str(market.get("eventId") or ""),
                "outcomes": outcomes,
                "isOpen": True,
            }
        )
    return offers, events


def fetch_event_group(
    session: requests.Session,
    cfg: dict[str, Any],
) -> tuple[dict[str, Event], list[Subcategory]]:
    """
    Discover MLB prop tabs from the league page nav tree.

    Events are filled in later from each subcategory markets payload.
    """
    page = str(cfg.get("dk_league_page") or _DEFAULT_CONFIG["dk_league_page"])
    html = fetch_text(
        session,
        page,
        timeout=int(cfg.get("dk_timeout", 30)),
        max_retries=int(cfg.get("dk_max_retries", 3)),
        label="MLB league page",
    )
    allow = [str(c).lower() for c in (cfg.get("dk_category_allowlist") or [])]
    subcategories = subcategories_from_nav(html, allow)

    logger.info(
        f"League nav: events=0 (loaded with markets), prop subcategories={len(subcategories)}"
    )

    cap = os.environ.get("DK_MAX_SUBCATEGORIES", "").strip()
    if cap.isdigit() and int(cap) > 0:
        logger.info(f"DK_MAX_SUBCATEGORIES set — limiting to first {cap} subcategory(ies)")
        subcategories = subcategories[: int(cap)]

    return {}, subcategories


def fetch_subcategory_offers(
    session: requests.Session,
    cfg: dict[str, Any],
    sub: Subcategory,
    events: dict[str, Event] | None = None,
) -> list[dict[str, Any]]:
    """Fetch every offer under one prop subcategory from the Nash markets API."""
    group_id = str(cfg.get("dk_event_group") or "84240")
    url = (
        f"{str(cfg['dk_api_base']).rstrip('/')}/league/leagueSubcategory/v1/markets"
    )
    params = {
        "isBatchable": "false",
        "templateVars": f"{group_id},{sub.subcategory_id}",
        "eventsQuery": (
            f"$filter=leagueId eq '{group_id}' AND "
            f"clientMetadata/Subcategories/any(s: s/Id eq '{sub.subcategory_id}')"
        ),
        "marketsQuery": (
            f"$filter=clientMetadata/subCategoryId eq '{sub.subcategory_id}' "
            f"AND tags/all(t: t ne 'SportcastBetBuilder')"
        ),
        "include": "Events",
        "entity": "events",
    }
    try:
        payload = fetch_json(
            session,
            url,
            params,
            timeout=int(cfg.get("dk_timeout", 30)),
            max_retries=int(cfg.get("dk_max_retries", 3)),
            label=f"subcategory {sub.category_name}/{sub.name}",
        )
    except (requests.RequestException, ValueError) as e:
        logger.warning(f"subcategory {sub.name!r} unavailable ({e}); continuing")
        return []

    _maybe_dump(payload, f"sub_{sub.category_id}_{sub.subcategory_id}")
    offers, payload_events = payload_to_offers(payload, competition="MLB")
    if events is not None:
        events.update(payload_events)

    delay = float(cfg.get("dk_request_delay", 0.4) or 0)
    if delay:
        time.sleep(delay)
    return offers

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

def resolve_stat(offer_label: str, sub: Subcategory) -> str | None:
    """
    Resolve the canonical stat for an offer.

    DraftKings names the stat on the subcategory ("Total Bases") and uses the offer
    label for the player, so the subcategory wins; the offer label is only a fallback
    for tabs whose name is generic ("Batter Props").
    """
    return match_stat(sub.name) or match_stat(offer_label) or match_stat(sub.category_name)

def parse_outcome_selection(label: str, line: Any) -> tuple[str, float | None]:
    """
    Derive (choice, stat_value) from an outcome.

    An empty choice means the outcome names a player instead of a side, which is how
    DraftKings models player-listed markets such as "First Home Run Scorer".
    """
    text = (label or "").strip()
    lowered = text.lower()
    numeric_line = _to_float(line)

    if lowered in ("yes", "no"):
        # A yes/no prop is a 0.5 line: "yes" pays if the player records at least one.
        return _CHOICE_MAP[lowered], numeric_line if numeric_line is not None else 0.5

    for token in ("over", "under"):
        if lowered.startswith(token):
            value = _first_number(text)
            return _CHOICE_MAP[token], value if value is not None else numeric_line

    # "2+ Hits" style alternates: the equivalent line is one half below the threshold.
    threshold = re.match(r"(\d+(?:\.\d+)?)\s*\+", text)
    if threshold:
        return "over", float(threshold.group(1)) - 0.5

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return "exact", float(text)

    return "", numeric_line

def extract_prices(outcome: dict[str, Any]) -> tuple[int | None, float | None, float | None]:
    """
    Pull (american, decimal, implied_probability) from an outcome.

    The implied probability is vig-inclusive — it is the book's price, not a fair
    probability, so it sits above the true number by roughly the hold.
    """
    american = _to_float(outcome.get("oddsAmerican"))
    if american is None:
        american = _first_number(str(outcome.get("oddsAmericanDisplay") or ""))

    decimal = _to_float(outcome.get("oddsDecimal"))
    if decimal is None:
        decimal = _to_float(outcome.get("oddsDecimalDisplay"))

    american_int = int(american) if american is not None else None
    if decimal is None and american_int is not None:
        decimal = american_to_decimal(american_int)
    if american_int is None and decimal is not None:
        american_int = decimal_to_american(decimal)

    implied = round(1.0 / decimal, 6) if decimal and decimal > 1.0 else None
    return american_int, decimal, implied

def team_names(event_name: str) -> set[str]:
    """Team names from an event title, used to reject team outcomes in prop markets."""
    return {p.strip().lower() for p in _TEAM_SPLIT.split(event_name or "") if p.strip()}

def looks_like_person(name: str, teams: set[str]) -> bool:
    """Reject team names, side labels and other non-player outcome text."""
    cleaned = (name or "").strip()
    if len(cleaned) < 3 or any(c.isdigit() for c in cleaned):
        return False
    if cleaned.lower() in teams:
        return False
    if any(cleaned.lower() in team for team in teams):
        return False
    return 2 <= len(cleaned.split()) <= 5

def is_offer_open(offer: dict[str, Any]) -> bool:
    """
    True when an offer is actually bettable.

    DraftKings carries suspended offers in the payload with stale prices, and signals
    state three different ways depending on endpoint version.
    """
    if offer.get("isSuspended") or offer.get("isSuspendedLive"):
        return False
    if offer.get("isOpen") is False:
        return False
    status = str(offer.get("offerStatus") or offer.get("status") or "").upper()
    return status not in ("SUSPENDED", "CLOSED", "SETTLED", "RESULTED")

def extract_offer_picks(
    offers: list[dict[str, Any]],
    sub: Subcategory,
    events: dict[str, Event],
    fetched_at: str,
    counters: dict[str, int],
    unmatched: set[str],
    sport_allowlist: frozenset[str] | None,
) -> list[Pick]:
    """Flatten one subcategory's offers into individual picks."""
    picks: list[Pick] = []

    for offer in offers:
        offer_label = str(offer.get("label") or "")
        market_name = f"{sub.name} - {offer_label}".strip(" -") or sub.name

        if not is_offer_open(offer):
            counters["suspended_offers"] += 1
            continue
        if is_game_market(sub.name) or is_game_market(offer_label):
            counters["game_markets"] += 1
            continue

        stat_name = resolve_stat(offer_label, sub)
        if not stat_name:
            counters["no_stat"] += 1
            unmatched.add(f"{sub.name} | {offer_label}")
            continue

        event = events.get(str(offer.get("eventId") or ""))
        sport_id = (event.competition if event else "MLB").upper()
        if sport_allowlist is not None and sport_id not in sport_allowlist:
            counters["skipped_sport"] += 1
            continue

        event_name = event.name if event else ""
        teams = team_names(event_name)
        updated_at = offer.get("lastModifiedDate") or fetched_at

        for outcome in _iter_records(offer.get("outcomes")):
            label = str(outcome.get("label") or "")
            choice, stat_value = parse_outcome_selection(label, outcome.get("line"))

            if choice == "exact":
                counters["exact_markets"] += 1
                continue

            participant = str(outcome.get("participant") or "")
            if choice:
                # Side-labelled outcome: the player is named on the outcome when
                # present, else on the offer (a per-player offer).
                full_name = participant or offer_label
            else:
                # Player-listed market: the outcome is the player and the bet is "yes".
                full_name = participant or label
                choice = "over"
                if stat_value is None:
                    stat_value = 0.5

            if not looks_like_person(full_name, teams):
                counters["no_player"] += 1
                continue

            american, decimal, implied = extract_prices(outcome)
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
                    start_time=event.start_time if event else None,
                    sport_id=sport_id,
                )
            )
            counters["added"] += 1

    return picks

def dedupe_picks(picks: list[Pick]) -> list[Pick]:
    """
    Collapse duplicate prices for the same wager.

    Standard and "Alternate" subcategories overlap — over 1.5 total bases is also
    listed as 2+ — so keep one row per wager, preferring the standard tab's wording.
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
    events: dict[str, Event],
    subcategories: list[Subcategory],
    sport_allowlist: frozenset[str] | None,
) -> list[Pick]:
    """
    Fetch each prop subcategory and flatten it into picks.

    Args:
        events: Events discovered from the event group, keyed by event id
        subcategories: Prop tabs to fetch
        sport_allowlist: Set of sport IDs to keep (None = keep all)

    Returns:
        List of Pick objects
    """
    logger.info(f"Extracting picks from {len(subcategories)} subcategory(ies)...")
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    counters = {
        "added": 0,
        "suspended_offers": 0,
        "game_markets": 0,
        "exact_markets": 0,
        "no_stat": 0,
        "no_player": 0,
        "no_price": 0,
        "skipped_sport": 0,
    }
    unmatched: set[str] = set()
    picks: list[Pick] = []

    for index, sub in enumerate(subcategories, start=1):
        logger.info(f"[{index}/{len(subcategories)}] {sub.category_name} / {sub.name}")
        offers = fetch_subcategory_offers(session, cfg, sub, events)
        if not offers:
            logger.warning(f"subcategory {sub.name!r}: no offers returned")
            continue

        sub_picks = extract_offer_picks(
            offers, sub, events, fetched_at, counters, unmatched, sport_allowlist
        )
        logger.info(f"  offers={len(offers)}, picks={len(sub_picks)}")
        picks.extend(sub_picks)

    picks = dedupe_picks(picks)

    logger.info(
        f"Extraction complete: added={counters['added']}, kept={len(picks)}, "
        f"game_markets={counters['game_markets']}, exact={counters['exact_markets']}, "
        f"suspended_offers={counters['suspended_offers']}, "
        f"no_stat={counters['no_stat']}, no_player={counters['no_player']}, "
        f"no_price={counters['no_price']}, skipped_sport={counters['skipped_sport']}"
    )
    if unmatched:
        sample = sorted(unmatched)[:25]
        logger.debug(f"Markets with no stat match ({len(unmatched)} distinct): {sample}")

    return picks

# ============================================================================
# File Export
# ============================================================================

def resolve_output_path(sport: str) -> str:
    """
    Resolve output file path for a single sport.

    Checks:
    1. DK_OUTPUT env var (if .json file) — used as-is for a single sport
    2. Default: data/props/draftkings/draftkings_{sport}_YYYY-MM-DD_HHMMSS.json

    Returns:
        Absolute path to output file
    """
    env_path = os.environ.get("DK_OUTPUT", "").strip()
    sport_slug = sport.strip().lower() or "unknown"

    if env_path and env_path.lower().endswith(".json"):
        expanded = os.path.expanduser(env_path)
        if not expanded.endswith(("/", "\\")) and not os.path.isdir(expanded):
            # When env points at a single file and we have multiple sports, stamp sport in.
            root, ext = os.path.splitext(expanded)
            if sport_slug not in os.path.basename(root).lower():
                expanded = f"{root}_{sport_slug}{ext}"
            logger.info(f"Using DK_OUTPUT: {expanded}")
            return expanded

    now = datetime.now(_OUTPUT_TZ)
    filename = now.strftime(f"draftkings_{sport_slug}_%Y-%m-%d_%H%M%S.json")
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
    """Map DraftKings event group name to odds league slug (mlb only)."""
    if "MLB" in sport.strip().upper():
        return "mlb"
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

class DraftKingsScraper:
    """Main scraper orchestrator."""

    def __init__(self) -> None:
        logger.info("Initializing DraftKingsScraper...")
        self.config = load_config()
        self.output_paths: list[str] = []
        self.picks: list[Pick] = []
        self.scraped_at: datetime | None = None
        logger.info("Initialization complete")

    def _load_sport_to_supabase(self, sport: str, sport_picks: list[Pick]) -> None:
        """Upsert one sport batch to odds.mlb_draftkings; JSON save is already done."""
        league = sport_to_league(sport)
        if league is None:
            logger.info(f"Skipping Supabase load for unmapped sport {sport!r}")
            return
        try:
            from src.odds.load_snapshots import load_draftkings_snapshot

            n = load_draftkings_snapshot(
                [p.to_dict() for p in sport_picks],
                league=league,
                scraped_at=self.scraped_at,
            )
            logger.info(f"Supabase odds.mlb_draftkings upserted {n} rows ({sport})")
        except Exception as e:
            logger.error(f"Supabase draftkings load failed (JSON kept): {e}")

    def run(self) -> None:
        """Execute the full scrape pipeline."""
        logger.info("=" * 70)
        logger.info("STARTING DRAFTKINGS SCRAPER (MLB)")
        logger.info("=" * 70)

        try:
            # Step 1: Discover events and prop tabs
            logger.info("\n[Step 1/3] Discovering events and prop subcategories...")
            session = build_session(self.config.get("headers", {}))
            events, subcategories = fetch_event_group(session, self.config)

            # Step 2: Fetch each subcategory and extract
            logger.info("\n[Step 2/3] Extracting picks...")
            sport_allowlist = get_sport_allowlist(self.config)
            self.picks = extract_picks(
                session, self.config, events, subcategories, sport_allowlist
            )

            # Step 3: Save one file per sport (draftkings_mlb_*, …)
            logger.info("\n[Step 3/3] Saving to file...")
            self.scraped_at = datetime.now(timezone.utc)
            grouped = group_picks_by_sport(self.picks)
            self.output_paths = []

            if not grouped:
                # Still write an empty MLB file when allowlist includes it, else first allowlist sport
                fallback = "MLB"
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
    scraper = DraftKingsScraper()
    scraper.run()