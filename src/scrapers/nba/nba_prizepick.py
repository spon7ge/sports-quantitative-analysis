"""
PrizePicks NBA projections.

Fetches NBA projections and exports clean JSON.
Handles DataDome (and legacy PerimeterX) via curl_cffi, requests, then Playwright.

Install for better success rate:
    pip install curl_cffi
    pip install playwright && playwright install chromium

Optional environment variables:
    PRIZEPICKS_OUTPUT: Override default output path (must end in .json)
    PRIZEPICKS_COOKIE: Browser session cookie (from DevTools)
    PRIZEPICKS_COOKIE_FILE: Path to file containing cookie (default: an auto-managed
                             file under data/props/prizepicks/ — see below)
    PRIZEPICKS_STORAGE_STATE: Path to Playwright storage_state JSON (default:
                               data/props/prizepicks/.playwright_storage.json)
    PRIZEPICKS_PROFILE_DIR: Persistent Chrome/Chromium user-data dir (default:
                             data/props/prizepicks/.pw_profile)
    PRIZEPICKS_EXECUTABLE: Absolute path to a Chromium-based browser binary
                            (Brave/Chrome/Edge). Overrides channel auto-detect.
    PRIZEPICKS_CHANNEL: Playwright browser channel. Default: auto-detect a
                         system browser (Chrome, Brave, Edge), else chrome.
                         Set to chromium/off/0 for bundled Chromium.
    PRIZEPICKS_CDP_URL: Attach to an already-running browser via CDP
                         (e.g. http://127.0.0.1:9222). Best when DataDome
                         blocks Playwright-launched browsers — open Brave with
                         --remote-debugging-port=9222, solve captcha, then run.
    PRIZEPICKS_PROXY: Optional proxy URL for HTTP and Playwright
    PRIZEPICKS_IMPERSONATE: Comma-separated curl_cffi profiles (default: safari17_2_ios,...)
    PRIZEPICKS_HEADLESS: Set to 1/true/yes for headless Playwright (default: headed)
    PRIZEPICKS_NBA_LEAGUE_ID: Override the NBA league_id if you already know it
    LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)

Cookie / profile persistence:
    Browser fallback prefers a system browser (Brave/Chrome) with a persistent
    profile under data/props/prizepicks/.pw_profile (shared with mlb_prizepick.py).
    When Playwright-launched browsers keep getting DataDome-blocked, use CDP
    attach (PRIZEPICKS_CDP_URL / PRIZEPICKS_CDP=1) after solving the captcha in
    a manually started Brave window. Session cookies (+ storage_state) are saved
    after a successful solve so HTTP can often skip the browser until the session dies.

league_id note:
    NBA defaults to league_id=7 (same convention as MLB=2 / WNBA=3 in
    mlb_prizepick.py). Set PRIZEPICKS_NBA_LEAGUE_ID to override. Live
    /leagues lookup is opt-in via PRIZEPICKS_RESOLVE_LEAGUES=1 because that
    endpoint is usually DataDome-blocked and was poisoning runs with
    noisy 403s before the real projections fetch.

Default output (one file per league):
    data/props/prizepicks/prizepicks_{league}_YYYY-MM-DD_HHMMSS.json
    e.g. prizepicks_nba_2026-07-31_180238.json

Usage:
    python -m src.scrapers.nba.nba_prizepick                    # Fetch and save
    python -m src.scrapers.nba.nba_prizepick --head 20          # Fetch and show first 20
    python -m src.scrapers.nba.nba_prizepick --from-file data.json --head 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

import requests

# ============================================================================
# Configuration
# ============================================================================

try:
    from .paths import ensure_repo_on_path
except ImportError:  # python nba_prizepick.py from this directory
    from paths import ensure_repo_on_path

_ROOT = str(ensure_repo_on_path(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "data", "props", "prizepicks")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")

# Auto-managed cookie file. A successful Playwright DataDome solve writes the
# resulting session cookie here; later runs read it automatically (no env
# var required) so they can often skip the browser fallback entirely.
_DEFAULT_COOKIE_FILE = os.path.join(_DEFAULT_OUTPUT_DIR, ".session_cookie.txt")
_DEFAULT_STORAGE_STATE_FILE = os.path.join(
    _DEFAULT_OUTPUT_DIR, ".playwright_storage.json"
)
_DEFAULT_PROFILE_DIR = os.path.join(_DEFAULT_OUTPUT_DIR, ".pw_profile")

# Hardcoded like mlb_prizepick.py (MLB=2, WNBA=3). Calling /leagues under
# DataDome usually 403s and only adds noise before the real fetch.
# Override with PRIZEPICKS_NBA_LEAGUE_ID if PrizePicks renumbers leagues.
DEFAULT_LEAGUES: tuple[tuple[str, int], ...] = (("NBA", 7),)
DEFAULT_LEAGUE_NAMES: tuple[str, ...] = ("NBA",)

# Playwright clearance probe: use a small board (WNBA) so captcha wait does not
# have to download the full NBA payload on every poll.
_CLEARANCE_PROBE_LEAGUE_ID = 3

API_BASE = "https://api.prizepicks.com/projections"
LEAGUES_API = "https://api.prizepicks.com/leagues"


def projections_api_url(league_id: int) -> str:
    """Build projections URL for a PrizePicks league_id."""
    return f"{API_BASE}?league_id={league_id}&per_page=250&single_stat=true"


# Headers for requests library (no TLS impersonation)
HEADERS_REQUESTS: dict[str, str] = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://app.prizepicks.com",
    "Referer": "https://app.prizepicks.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

# Base headers for curl_cffi (impersonate profile provides User-Agent/sec-ch-ua)
HEADERS_CURL_BASE: dict[str, str] = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://app.prizepicks.com",
    "Referer": "https://app.prizepicks.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

# curl_cffi browser impersonation profiles to try (in order)
CURL_IMPERSONATE_DEFAULT = (
    "safari17_2_ios",  # Often works against PerimeterX
    "safari17_0",
    "chrome136",
    "chrome131",
    "chrome124",
    "chrome120",
)

# Origin variants to try (sometimes bot protection checks Origin header)
ORIGIN_VARIANTS = [
    {"name": "prizepicks.com", "Origin": "https://app.prizepicks.com", "Referer": "https://app.prizepicks.com/"},
    {"name": "www.prizepicks.com", "Origin": "https://www.prizepicks.com", "Referer": "https://www.prizepicks.com/"},
]

APP_URL = "https://app.prizepicks.com/"
PLAYWRIGHT_WAIT_SECONDS = 180
PLAYWRIGHT_POLL_INTERVAL_SECONDS = 2.0

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
class Projection:
    """A single player projection."""
    player: str
    stat_type: str
    line_score: float | int | None
    odds_type: str
    updated_at: str
    league: str = "NBA"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ExportData:
    """Complete export payload."""
    source: str = "PrizePicks"
    league: str = "NBA"
    fetched_at: str = ""
    raw_snapshot: str | None = None
    count: int = 0
    projections: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.projections is None:
            self.projections = []
        self.count = len(self.projections) if self.projections else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "league": self.league,
            "fetched_at": self.fetched_at,
            "raw_snapshot": self.raw_snapshot,
            "count": self.count,
            "projections": self.projections or [],
        }


# ============================================================================
# Utility Functions
# ============================================================================

def league_slug(league: str) -> str:
    """Normalize league label for filenames (e.g. NBA → nba)."""
    return (league or "mixed").strip().lower().replace("+", "_").replace(" ", "_")


def resolve_output_path(
    league: str | None = None,
    *,
    when: datetime | None = None,
) -> str:
    """
    Resolve output file path.

    Checks:
    1. PRIZEPICKS_OUTPUT env var (if .json file) — used as-is when set
    2. Default: data/props/prizepicks/prizepicks_{league}_YYYY-MM-DD_HHMMSS.json
       e.g. prizepicks_nba_2026-07-31_180238.json

    Args:
        league: League label embedded in the default filename (nba)
        when: Timestamp for the filename (defaults to now in America/Los_Angeles)

    Returns:
        Absolute path to output file
    """
    env_path = os.environ.get("PRIZEPICKS_OUTPUT", "").strip()

    if env_path and env_path.lower().endswith(".json"):
        expanded = os.path.expanduser(env_path)
        if not expanded.endswith(("/", "\\")) and not os.path.isdir(expanded):
            # When saving per-league under an explicit path, insert slug before .json
            if league:
                root, ext = os.path.splitext(expanded)
                slug = league_slug(league)
                if not root.lower().endswith(f"_{slug}"):
                    expanded = f"{root}_{slug}{ext}"
            logger.info(f"Using PRIZEPICKS_OUTPUT: {expanded}")
            return expanded

    stamp = when or datetime.now(_OUTPUT_TZ)
    slug = league_slug(league) if league else "mixed"
    filename = stamp.strftime(f"prizepicks_{slug}_%Y-%m-%d_%H%M%S.json")
    path = os.path.join(_DEFAULT_OUTPUT_DIR, filename)

    logger.info(f"Using default output path: {path}")
    return path


def cookie_file_path() -> str:
    """
    Resolve which file to read/write for the persisted session cookie.

    Uses PRIZEPICKS_COOKIE_FILE if explicitly set, otherwise the
    auto-managed default under data/props/prizepicks/.
    """
    explicit = os.environ.get("PRIZEPICKS_COOKIE_FILE", "").strip()
    if explicit:
        return os.path.expanduser(explicit)
    return _DEFAULT_COOKIE_FILE



def storage_state_path() -> str:
    """Resolve Playwright storage_state JSON path (env override or default)."""
    explicit = os.environ.get("PRIZEPICKS_STORAGE_STATE", "").strip()
    if explicit:
        return os.path.expanduser(explicit)
    return _DEFAULT_STORAGE_STATE_FILE


def profile_dir_path() -> str:
    """Resolve persistent browser profile directory (env override or default)."""
    explicit = os.environ.get("PRIZEPICKS_PROFILE_DIR", "").strip()
    if explicit:
        return os.path.expanduser(explicit)
    return _DEFAULT_PROFILE_DIR


# Prefer real installed browsers over Playwright's bundled Chromium (DataDome).
# Windows entries use env vars so Program Files / AppData layouts resolve.
_SYSTEM_CHROMIUM_CANDIDATES: tuple[str, ...] = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
    r"%PROGRAMFILES%\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"%PROGRAMFILES(X86)%\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe",
    r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe",
    r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe",
    r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe",
)


def detect_system_chromium_executable() -> str | None:
    """Return the first installed Chromium-based browser binary, if any."""
    for raw in _SYSTEM_CHROMIUM_CANDIDATES:
        path = os.path.expandvars(os.path.expanduser(raw))
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def playwright_channel_from_env() -> str | None:
    """
    Explicit Playwright channel from PRIZEPICKS_CHANNEL, if set.

    Empty / unset means "auto" (see resolve_playwright_browser). Values
    chromium / off / 0 / false / bundled force bundled Chromium (no channel).
    """
    raw = os.environ.get("PRIZEPICKS_CHANNEL", "").strip().lower()
    if raw == "":
        return None  # auto
    if raw in {"0", "off", "false", "chromium", "bundled"}:
        return None
    return raw


def resolve_playwright_browser() -> tuple[str | None, str | None]:
    """
    Choose Playwright launch browser.

    Returns:
        (channel, executable_path) — at most one is set. Both None → bundled Chromium.

    Priority:
    1. PRIZEPICKS_EXECUTABLE
    2. Explicit PRIZEPICKS_CHANNEL (non-auto)
    3. Auto-detected system Chrome / Brave / Edge / Chromium
    4. channel=chrome (Playwright-managed Chrome)
    """
    explicit_exe = os.environ.get("PRIZEPICKS_EXECUTABLE", "").strip()
    if explicit_exe:
        return None, os.path.expanduser(explicit_exe)

    raw_channel = os.environ.get("PRIZEPICKS_CHANNEL", "").strip().lower()
    if raw_channel in {"0", "off", "false", "chromium", "bundled"}:
        return None, None
    if raw_channel:
        return raw_channel, None

    detected = detect_system_chromium_executable()
    if detected:
        return None, detected

    return "chrome", None


def cdp_url_from_env() -> str | None:
    """
    Optional CDP endpoint for attaching to a manually started browser.

    PRIZEPICKS_CDP_URL wins. PRIZEPICKS_CDP=1/true/yes defaults to
    http://127.0.0.1:9222.
    """
    explicit = os.environ.get("PRIZEPICKS_CDP_URL", "").strip()
    if explicit:
        return explicit
    raw = os.environ.get("PRIZEPICKS_CDP", "").strip().lower()
    if raw in {"1", "true", "yes"}:
        return "http://127.0.0.1:9222"
    return None


def storage_state_for_context(path: str | None = None) -> str | None:
    """Return storage_state path if the file exists and is valid JSON; else None.

    Playwright aborts new_context() on corrupt storage_state. Invalid files
    should fall back to an empty context instead of failing the whole run.
    """
    target = path or storage_state_path()
    if not os.path.isfile(target):
        return None
    try:
        with open(target, encoding="utf-8") as f:
            json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning(
            f"Ignoring invalid Playwright storage_state at {target}; "
            "using empty context"
        )
        return None
    return target


def proxy_from_env() -> str | None:
    """Optional proxy URL for HTTP and Playwright (`PRIZEPICKS_PROXY`)."""
    raw = os.environ.get("PRIZEPICKS_PROXY", "").strip()
    return raw or None


def playwright_proxy_kwargs(proxy_url: str) -> dict[str, str]:
    """Playwright proxy dict; split user:pass@host into username/password keys."""
    parsed = urlparse(proxy_url)
    if not parsed.hostname:
        return {"server": proxy_url}
    server = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        server = f"{server}:{parsed.port}"
    config: dict[str, str] = {"server": server}
    if parsed.username:
        config["username"] = unquote(parsed.username)
    if parsed.password:
        config["password"] = unquote(parsed.password)
    return config


def get_cookie_for_request() -> str:
    """
    Load optional browser cookie.

    Checks (in order):
    1. PRIZEPICKS_COOKIE env var
    2. Cookie file — PRIZEPICKS_COOKIE_FILE if set, else the auto-managed
       default file (data/props/prizepicks/.session_cookie.txt), which is
       written automatically after a successful Playwright DataDome solve.

    Returns:
        Cookie string, or empty string if none found
    """
    # Try env var first
    c = os.environ.get("PRIZEPICKS_COOKIE", "").strip()
    if c:
        logger.debug("Using PRIZEPICKS_COOKIE from environment")
        return c

    # Try file (explicit or auto-managed default)
    path = cookie_file_path()
    if os.path.isfile(path):
        logger.debug(f"Loading cookie from: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                cookie = f.read().strip()
                if cookie:
                    return cookie
        except IOError as e:
            logger.warning(f"Failed to read cookie file: {e}")

    logger.debug("No cookie found (PRIZEPICKS_COOKIE not set, no cookie file yet)")
    return ""


def save_session_cookie(cookie_header: str, path: str | None = None) -> None:
    """
    Persist a session cookie string to disk for reuse by future runs.

    Args:
        cookie_header: A "name=value; name2=value2" cookie header string
        path: Override target path (default: cookie_file_path())
    """
    target = path or cookie_file_path()
    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(cookie_header.strip() + "\n")
        logger.info(f"✓ Saved session cookie to {target} (reused automatically by future runs)")
    except IOError as e:
        logger.warning(f"Failed to save session cookie: {e}")


def _build_cookie_header_from_playwright(cookies: list[dict[str, Any]]) -> str:
    """Build a 'name=value; ...' header string from Playwright context cookies."""
    parts = []
    for c in cookies:
        domain = c.get("domain") or ""
        name = c.get("name")
        value = c.get("value")
        if "prizepicks.com" not in domain or not name or value is None:
            continue
        parts.append(f"{name}={value}")
    parts.sort()
    return "; ".join(parts)


def add_headers_to_dict(base: dict[str, str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Merge base headers with optional overrides and cookie."""
    h = dict(base)

    cookie = get_cookie_for_request()
    if cookie:
        h["Cookie"] = cookie

    if overrides:
        h.update(overrides)

    return h


def is_perimeterx_challenge(body: str) -> bool:
    """Check if response body indicates a PerimeterX challenge."""
    if not body:
        return False
    return any(marker in body for marker in ["px-cloud", '"appId":"PX', "PerimeterX"])


def is_datadome_challenge(
    body: str,
    headers: dict[str, str] | None = None,
) -> bool:
    """Check if response indicates a DataDome challenge."""
    if headers:
        for key, value in headers.items():
            if key.lower() == "x-datadome" and value:
                return True
    if not body:
        return False
    lowered = body.lower()
    return any(
        marker in lowered
        for marker in (
            "captcha-delivery.com",
            "geo.captcha-delivery.com",
            "datadome",
        )
    )


def is_bot_challenge(
    body: str,
    headers: dict[str, str] | None = None,
) -> bool:
    """True if response looks like DataDome or PerimeterX bot protection."""
    return is_datadome_challenge(body, headers) or is_perimeterx_challenge(body)


def headless_from_env() -> bool:
    """Return True when PRIZEPICKS_HEADLESS is a truthy value (default: headed)."""
    raw = os.environ.get("PRIZEPICKS_HEADLESS", "").strip().lower()
    return raw in {"1", "true", "yes"}


def build_fetch_failure_message() -> str:
    """Human-readable troubleshooting when all fetch paths fail."""
    return (
        "Failed to fetch PrizePicks projections.\n"
        "\n"
        "DataDome is blocking automated HTTP clients.\n"
        "\n"
        "Troubleshooting:\n"
        "1. Browser fallback (recommended when HTTP returns 403):\n"
        "   pip install playwright && playwright install chromium\n"
        "   Prefer CDP attach when Playwright-launched browsers stay blocked:\n"
        "     1) Start Brave with remote debugging (separate terminal):\n"
        "        python nba_prizepick.py --print-brave-cdp-cmd\n"
        "        # or run the printed Brave command yourself\n"
        "     2) Solve any captcha at app.prizepicks.com in that window\n"
        "     3) PRIZEPICKS_CDP=1 python nba_prizepick.py\n"
        "   Otherwise: headed Brave/Chrome persistent profile (.pw_profile).\n"
        "   Use PRIZEPICKS_HEADLESS=1 only when no interactive captcha is expected.\n"
        "   If hard-restricted: wait 30–60m, rm .session_cookie.txt, retry.\n"
        "2. Optional browser session cookie for the HTTP path:\n"
        "   PRIZEPICKS_COOKIE='...' python nba_prizepick.py\n"
        "   Or: PRIZEPICKS_COOKIE_FILE=~/.prizepicks_cookie python ...\n"
        "   Auto-managed files (written after a successful Playwright solve):\n"
        "   data/props/prizepicks/.session_cookie.txt\n"
        "   data/props/prizepicks/.playwright_storage.json\n"
        "   Override with PRIZEPICKS_COOKIE_FILE / PRIZEPICKS_STORAGE_STATE /\n"
        "   PRIZEPICKS_PROFILE_DIR / PRIZEPICKS_CHANNEL.\n"
        "3. curl_cffi impersonation profiles:\n"
        "   pip install curl_cffi\n"
        "   PRIZEPICKS_IMPERSONATE='safari17_0,chrome131' python ...\n"
        "4. Optional proxy (sticky residential IPs if challenges remain frequent):\n"
        "   PRIZEPICKS_PROXY='http://user:pass@host:port' python ...\n"
        "\n"
        "Run with LOG_LEVEL=DEBUG for detailed diagnostics."
    )


def _normalize_response_headers(headers: Any) -> dict[str, str]:
    """Best-effort convert response header mapping to str→str."""
    if not headers:
        return {}
    try:
        return {str(k): str(v) for k, v in dict(headers).items()}
    except Exception:
        return {}


def _log_non_success_response(status_code: int, body: str, headers: Any) -> None:
    """Log why an HTTP response was rejected (DataDome / status)."""
    hdrs = _normalize_response_headers(headers)
    if is_datadome_challenge(body, hdrs):
        logger.debug(f"    → Status {status_code}, DataDome challenge")
    elif is_perimeterx_challenge(body):
        logger.debug(f"    → Status {status_code}, PerimeterX challenge")
    elif status_code == 403:
        logger.debug(f"    → Status {status_code} Forbidden (bot protection)")
    else:
        logger.debug(f"    → Status {status_code}, skipping")


def get_curl_impersonate_list() -> tuple[str, ...]:
    """
    Get list of curl_cffi impersonation profiles to try.

    Returns:
        Tuple of profile names in order of preference
    """
    env_list = os.environ.get("PRIZEPICKS_IMPERSONATE", "").strip()

    if env_list:
        profiles = tuple(p.strip() for p in env_list.split(",") if p.strip())
        logger.info(f"Using custom impersonate profiles: {profiles}")
        return profiles

    logger.debug(f"Using default impersonate profiles: {CURL_IMPERSONATE_DEFAULT}")
    return CURL_IMPERSONATE_DEFAULT


# ============================================================================
# League ID Resolution
# ============================================================================

def _extract_league_id_map(payload: dict[str, Any]) -> dict[str, int]:
    """Parse a /leagues API payload into {NAME_UPPER: league_id}."""
    out: dict[str, int] = {}
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in ("league", None):
            continue
        attrs = item.get("attributes") or {}
        name = attrs.get("name") or attrs.get("display_name") or attrs.get("code")
        lid = item.get("id")
        if not name or lid is None:
            continue
        try:
            out[str(name).strip().upper()] = int(lid)
        except (TypeError, ValueError):
            continue
    return out


def fetch_league_id_map(headers: dict[str, str]) -> dict[str, int]:
    """
    Fetch the live league name → league_id mapping from PrizePicks.

    Returns:
        Dict of uppercased league name → numeric id. Empty dict on failure.
    """
    try:
        resp = requests.get(LEAGUES_API, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch /leagues for id resolution: {e}")
        return {}

    id_map = _extract_league_id_map(data)
    logger.debug(f"Resolved {len(id_map)} league ids from /leagues")
    return id_map


def resolve_leagues(
    names: tuple[str, ...] = DEFAULT_LEAGUE_NAMES,
) -> tuple[tuple[str, int], ...]:
    """
    Resolve league_id for each requested league name.

    Order of precedence per league:
    1. PRIZEPICKS_{NAME}_LEAGUE_ID env var (e.g. PRIZEPICKS_NBA_LEAGUE_ID)
    2. Hardcoded DEFAULT_LEAGUES (NBA=7)
    3. Optional live /leagues lookup only when PRIZEPICKS_RESOLVE_LEAGUES=1
       (skipped by default — that endpoint is usually DataDome-blocked)

    Returns:
        Tuple of (league_name, league_id) pairs. Leagues that can't be
        resolved are skipped (with an error logged).
    """
    defaults = {n.upper(): lid for n, lid in DEFAULT_LEAGUES}
    id_map: dict[str, int] = {}
    if os.environ.get("PRIZEPICKS_RESOLVE_LEAGUES", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        id_map = fetch_league_id_map(HEADERS_REQUESTS)

    resolved: list[tuple[str, int]] = []
    for name in names:
        upper = name.strip().upper()

        env_key = f"PRIZEPICKS_{upper}_LEAGUE_ID"
        env_override = os.environ.get(env_key, "").strip()
        if env_override:
            try:
                resolved.append((name, int(env_override)))
                logger.info(f"Using {env_key}={env_override} for {name}")
                continue
            except ValueError:
                logger.warning(f"{env_key}={env_override!r} is not a valid int, ignoring")

        lid = id_map.get(upper)
        if lid is not None:
            logger.info(f"Resolved {name} → league_id={lid} via /leagues")
            resolved.append((name, lid))
            continue

        default_lid = defaults.get(upper)
        if default_lid is not None:
            resolved.append((name, default_lid))
            continue

        logger.error(f"Could not resolve league_id for {name}; skipping")

    return tuple(resolved)


# ============================================================================
# API Fetching
# ============================================================================

def _parse_projections_json(text: str) -> dict[str, Any] | None:
    """Parse API JSON; return dict only when it has a 'data' key."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "data" in data:
        return data
    return None


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    get_fn: Any,
    get_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Shared GET → JSON helper for curl_cffi / requests."""
    kwargs = dict(get_kwargs or {})
    response = get_fn(url, headers=headers, timeout=timeout, **kwargs)
    body = getattr(response, "text", "") or ""
    if response.status_code != 200:
        _log_non_success_response(
            response.status_code,
            body,
            getattr(response, "headers", None),
        )
        return None
    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.debug("    → Invalid JSON, skipping")
        return None
    if isinstance(data, dict) and "data" in data:
        return data
    if is_bot_challenge(body, _normalize_response_headers(getattr(response, "headers", None))):
        logger.debug("    → Bot challenge payload (no 'data' field), skipping")
    else:
        logger.debug("    → Response missing 'data' field, skipping")
    return None


def try_fetch_with_curl_cffi(
    leagues: tuple[tuple[str, int], ...],
) -> dict[str, dict[str, Any]] | None:
    """
    Try to fetch projections for each league using curl_cffi browser impersonation.

    Returns:
        Mapping league name → API payload, or None if no league succeeded
    """
    try:
        import curl_cffi.requests as curl_requests  # type: ignore[import-untyped]
    except ImportError:
        logger.info("curl_cffi not installed — skipping browser impersonation")
        return None

    logger.info("Attempting fetch with curl_cffi (browser impersonation)...")

    profiles = get_curl_impersonate_list()
    cookie = get_cookie_for_request()
    proxy = proxy_from_env()

    for i, profile in enumerate(profiles, 1):
        for origin_var in ORIGIN_VARIANTS:
            origin_name = origin_var["name"]
            headers = add_headers_to_dict(HEADERS_CURL_BASE, {
                "Origin": origin_var["Origin"],
                "Referer": origin_var["Referer"],
            })

            logger.debug(
                f"  Try {i}/{len(profiles)}: profile={profile}, "
                f"origin={origin_name}, cookie={'yes' if cookie else 'no'}"
            )

            try:
                out: dict[str, dict[str, Any]] = {}
                for league_name, league_id in leagues:
                    url = projections_api_url(league_id)
                    get_kwargs: dict[str, Any] = {"impersonate": profile}
                    if proxy:
                        get_kwargs["proxy"] = proxy
                    data = _http_get_json(
                        url,
                        headers=headers,
                        timeout=45,
                        get_fn=curl_requests.get,
                        get_kwargs=get_kwargs,
                    )
                    if data is not None:
                        out[league_name] = data
                        logger.info(
                            f"✓ curl_cffi {league_name} ok "
                            f"(profile={profile}, rows={len(data.get('data') or [])})"
                        )
                if out:
                    return out
            except Exception as e:
                logger.debug(f"    → Error: {type(e).__name__}: {e}")
                continue

    logger.info("curl_cffi attempts exhausted or no valid response")
    return None


def try_fetch_with_requests(
    leagues: tuple[tuple[str, int], ...],
) -> dict[str, dict[str, Any]] | None:
    """
    Try to fetch projections for each league using stdlib requests.

    Often fails against DataDome without a browser session cookie.

    Returns:
        Mapping league name → API payload, or None if no league succeeded
    """
    logger.info("Attempting fetch with requests (stdlib)...")

    cookie = get_cookie_for_request()
    proxy = proxy_from_env()

    for origin_var in ORIGIN_VARIANTS:
        origin_name = origin_var["name"]
        headers = add_headers_to_dict(HEADERS_REQUESTS, {
            "Origin": origin_var["Origin"],
            "Referer": origin_var["Referer"],
        })

        logger.debug(f"  Try: origin={origin_name}, cookie={'yes' if cookie else 'no'}")

        try:
            out: dict[str, dict[str, Any]] = {}
            for league_name, league_id in leagues:
                url = projections_api_url(league_id)
                get_kwargs: dict[str, Any] = {}
                if proxy:
                    get_kwargs["proxies"] = {"http": proxy, "https": proxy}
                data = _http_get_json(
                    url,
                    headers=headers,
                    timeout=30,
                    get_fn=requests.get,
                    get_kwargs=get_kwargs,
                )
                if data is not None:
                    out[league_name] = data
                    logger.info(
                        f"✓ requests {league_name} ok "
                        f"(rows={len(data.get('data') or [])})"
                    )
            if out:
                return out
        except requests.RequestException as e:
            logger.debug(f"    → Request error: {type(e).__name__}: {e}")
            continue

    logger.info("requests attempts exhausted")
    return None


def _playwright_fetch_url(page: Any, api_url: str) -> tuple[int, str]:
    """In-page fetch; returns (status, body_text)."""
    result = page.evaluate(
        """async (apiUrl) => {
            try {
                const r = await fetch(apiUrl, {
                    credentials: 'include',
                    headers: { 'Accept': 'application/json' },
                });
                const text = await r.text();
                return { ok: r.ok, status: r.status, text };
            } catch (e) {
                return { ok: false, status: 0, text: String(e) };
            }
        }""",
        api_url,
    )
    return int(result.get("status") or 0), (result.get("text") or "")


def brave_cdp_launch_command(*, port: int = 9222) -> str:
    """Shell command to start Brave with remote debugging on the scraper profile."""
    exe = detect_system_chromium_executable() or os.path.expandvars(
        r"%PROGRAMFILES%\BraveSoftware\Brave-Browser\Application\brave.exe"
        if os.name == "nt"
        else "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    )
    profile = profile_dir_path()
    return (
        f'"{exe}" '
        f"--remote-debugging-port={port} "
        f'--user-data-dir="{profile}" '
        f'"{APP_URL}"'
    )


def _playwright_poll_until_cleared(
    page: Any,
    *,
    probe_url: str,
    deadline: float,
) -> bool:
    """Poll in-page API until projections JSON returns or deadline hits."""
    captcha_logged = False
    while time.monotonic() < deadline:
        page_content = ""
        try:
            page_content = page.content()
        except Exception:
            pass

        if is_datadome_challenge(page_content) and not captcha_logged:
            logger.info(
                "DataDome captcha detected — solve it in the browser window"
            )
            captcha_logged = True

        try:
            status, text = _playwright_fetch_url(page, probe_url)
        except Exception as e:
            logger.debug(f"  Playwright evaluate error: {type(e).__name__}: {e}")
            time.sleep(PLAYWRIGHT_POLL_INTERVAL_SECONDS)
            continue

        data = _parse_projections_json(text) if status == 200 else None
        if data is not None:
            return True

        if is_datadome_challenge(text) and not captcha_logged:
            logger.info(
                "DataDome captcha detected — solve it in the browser window"
            )
            captcha_logged = True
        else:
            logger.debug(
                f"  Playwright poll: status={status}, "
                f"datadome={is_datadome_challenge(text)}"
            )

        time.sleep(PLAYWRIGHT_POLL_INTERVAL_SECONDS)
    return False


def _persist_playwright_session(context: Any, *, profile: str | None = None) -> None:
    """Save cookie header + storage_state after a successful clearance."""
    try:
        cookie_header = _build_cookie_header_from_playwright(context.cookies())
        if cookie_header:
            save_session_cookie(cookie_header)
        state_path = storage_state_path()
        parent = os.path.dirname(os.path.abspath(state_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        context.storage_state(path=state_path)
        extra = f" (profile also at {profile})" if profile else ""
        logger.info(f"✓ Saved Playwright storage_state to {state_path}{extra}")
    except Exception as e:
        logger.warning(f"Could not persist session/storage: {e}")


def _fetch_leagues_with_page(
    page: Any,
    league_urls: list[tuple[str, str]],
) -> dict[str, dict[str, Any]] | None:
    """Fetch each league URL via in-page fetch; return mapping or None."""
    out: dict[str, dict[str, Any]] = {}
    for league_name, url in league_urls:
        try:
            status, text = _playwright_fetch_url(page, url)
        except Exception as e:
            logger.debug(
                f"  Playwright {league_name} error: {type(e).__name__}: {e}"
            )
            continue
        data = _parse_projections_json(text) if status == 200 else None
        if data is None:
            logger.warning(
                f"Playwright {league_name} failed "
                f"(status={status}, datadome={is_datadome_challenge(text)})"
            )
            continue
        out[league_name] = data
        logger.info(
            f"✓ Playwright {league_name} ok "
            f"(rows={len(data.get('data') or [])})"
        )
    return out or None


def try_fetch_with_playwright(
    leagues: tuple[tuple[str, int], ...],
) -> dict[str, dict[str, Any]] | None:
    """
    Last-resort fetch via Playwright: CDP attach (preferred when set) or
    launch a persistent system browser profile, clear DataDome, fetch leagues.

    Set PRIZEPICKS_CDP_URL / PRIZEPICKS_CDP=1 to attach to a manually started
    Brave/Chrome with --remote-debugging-port (best clearance success rate).

    Returns:
        Mapping league name → API payload, or None if Playwright is unavailable / times out.
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        logger.info(
            "playwright not installed — skipping browser fallback "
            "(pip install playwright && playwright install chromium)"
        )
        return None

    cdp_url = cdp_url_from_env()
    headless = headless_from_env()
    profile = profile_dir_path()
    os.makedirs(profile, exist_ok=True)

    league_urls = [(name, projections_api_url(lid)) for name, lid in leagues]
    # Probe a small board during captcha wait — not the full NBA URL.
    probe_url = projections_api_url(_CLEARANCE_PROBE_LEAGUE_ID)
    deadline = time.monotonic() + PLAYWRIGHT_WAIT_SECONDS

    try:
        with sync_playwright() as p:
            browser = None
            context: Any
            owns_browser = False

            if cdp_url:
                logger.info(
                    f"Attempting fetch with Playwright CDP attach ({cdp_url}; "
                    f"wait up to {PLAYWRIGHT_WAIT_SECONDS}s)..."
                )
                logger.info(
                    "Solve any captcha in the already-open browser; "
                    "the scraper will keep polling the API."
                )
                try:
                    browser = p.chromium.connect_over_cdp(cdp_url)
                except Exception as cdp_err:
                    logger.info(
                        f"CDP attach failed ({type(cdp_err).__name__}: {cdp_err}). "
                        "Nothing is listening on that port — start Brave first, "
                        "leave it open, then re-run with PRIZEPICKS_CDP=1:\n"
                        f"  {brave_cdp_launch_command()}"
                    )
                    return None
                context = (
                    browser.contexts[0]
                    if browser.contexts
                    else browser.new_context()
                )
                page = context.pages[0] if context.pages else context.new_page()
                page.set_default_timeout(120_000)
                try:
                    page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
                except Exception as e:
                    logger.debug(f"CDP goto note: {type(e).__name__}: {e}")
            else:
                channel, executable = resolve_playwright_browser()
                if executable:
                    browser_label = executable
                elif channel:
                    browser_label = f"channel={channel}"
                else:
                    browser_label = "bundled-chromium"
                logger.info(
                    "Attempting fetch with Playwright "
                    f"({browser_label}, {'headless' if headless else 'headed'}; "
                    f"profile={profile}; wait up to {PLAYWRIGHT_WAIT_SECONDS}s)..."
                )
                if not headless:
                    logger.info(
                        "If a captcha appears, solve it in the browser window; "
                        "the scraper will keep polling the API."
                    )
                    logger.info(
                        "Tip: if captchas stay unsolvable, start Brave with CDP "
                        f"instead:\n  {brave_cdp_launch_command()}\n"
                        "Then: PRIZEPICKS_CDP=1 python nba_prizepick.py"
                    )

                proxy = proxy_from_env()
                launch_kwargs: dict[str, Any] = {
                    "user_data_dir": profile,
                    "headless": headless,
                    "viewport": {"width": 1280, "height": 800},
                    "locale": "en-US",
                    "args": ["--disable-blink-features=AutomationControlled"],
                }
                if executable:
                    launch_kwargs["executable_path"] = executable
                elif channel:
                    launch_kwargs["channel"] = channel
                if proxy:
                    launch_kwargs["proxy"] = playwright_proxy_kwargs(proxy)
                    logger.info("Using PRIZEPICKS_PROXY for Playwright")

                try:
                    context = p.chromium.launch_persistent_context(**launch_kwargs)
                except Exception as first_err:
                    used_system = (
                        "channel" in launch_kwargs
                        or "executable_path" in launch_kwargs
                    )
                    if not used_system:
                        raise
                    launch_kwargs.pop("channel", None)
                    failed_exe = launch_kwargs.pop("executable_path", None)
                    alt = detect_system_chromium_executable()
                    if alt and alt != failed_exe:
                        launch_kwargs["executable_path"] = alt
                        logger.warning(
                            f"Playwright browser launch failed "
                            f"({type(first_err).__name__}: {first_err}); "
                            f"retrying with {alt}"
                        )
                        try:
                            context = p.chromium.launch_persistent_context(
                                **launch_kwargs
                            )
                        except Exception as second_err:
                            launch_kwargs.pop("executable_path", None)
                            logger.warning(
                                f"Alternate system browser failed "
                                f"({type(second_err).__name__}: {second_err}); "
                                "retrying with bundled Chromium"
                            )
                            context = p.chromium.launch_persistent_context(
                                **launch_kwargs
                            )
                    else:
                        logger.warning(
                            f"Playwright system browser failed "
                            f"({type(first_err).__name__}: {first_err}); "
                            "retrying with bundled Chromium"
                        )
                        context = p.chromium.launch_persistent_context(**launch_kwargs)

                owns_browser = True
                page = context.pages[0] if context.pages else context.new_page()
                # NBA payloads are large; default 30s evaluate timeout is too tight.
                page.set_default_timeout(120_000)
                page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)

            try:
                cleared = _playwright_poll_until_cleared(
                    page, probe_url=probe_url, deadline=deadline
                )
                if not cleared:
                    logger.info(
                        "Playwright wait exhausted without a valid projections payload"
                    )
                    return None

                _persist_playwright_session(
                    context, profile=None if cdp_url else profile
                )
                return _fetch_leagues_with_page(page, league_urls)
            finally:
                if owns_browser:
                    context.close()
                elif browser is not None:
                    # CDP: disconnect only — leave the user's browser running.
                    browser.close()
    except Exception as e:
        logger.info(f"Playwright fallback failed: {type(e).__name__}: {e}")
        return None



def fetch_projections_payloads(
    leagues: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Fetch NBA projections from PrizePicks API.

    Strategy:
    1. Resolve league_id(s) if not provided
    2. Try curl_cffi with browser impersonation
    3. Fall back to requests (often needs cookie against DataDome)
    4. Fall back to Playwright (headed; solve captcha if needed)
    5. If all fail, provide helpful error message

    Returns:
        Mapping league name → API response with 'data' / 'included'

    Raises:
        requests.HTTPError: If all fetch attempts fail
    """
    if leagues is None:
        leagues = resolve_leagues(DEFAULT_LEAGUE_NAMES)

    if not leagues:
        raise ValueError(
            "No leagues resolved — could not determine a league_id to fetch. "
            "Set PRIZEPICKS_NBA_LEAGUE_ID to override."
        )

    logger.info("=" * 70)
    logger.info(
        "FETCHING PRIZEPICKS PROJECTIONS "
        f"({', '.join(f'{n}={i}' for n, i in leagues)})"
    )
    logger.info("=" * 70)

    result = try_fetch_with_curl_cffi(leagues)
    if result is not None:
        return result

    result = try_fetch_with_requests(leagues)
    if result is not None:
        return result

    result = try_fetch_with_playwright(leagues)
    if result is not None:
        return result

    logger.error("=" * 70)
    logger.error("✗ FETCH FAILED: All attempts exhausted")
    logger.error("=" * 70)

    raise requests.HTTPError(build_fetch_failure_message())


# ============================================================================
# Data Processing
# ============================================================================

def build_player_lookup(data: dict[str, Any]) -> dict[str, str]:
    """
    Build player ID → name mapping from API response.

    Args:
        data: API response dict

    Returns:
        Dict mapping player IDs to full names
    """
    out: dict[str, str] = {}

    for elem in data.get("included") or []:
        if elem.get("type") != "new_player":
            continue

        eid = elem.get("id")
        if eid is None:
            continue

        name = (elem.get("attributes") or {}).get("name")
        if name is not None:
            out[eid] = name

    logger.debug(f"Built player lookup: {len(out)} players")
    return out


def extract_projections(
    data: dict[str, Any],
    *,
    league: str = "NBA",
) -> list[Projection]:
    """
    Extract projections from API response.

    Flattens nested structure into clean Projection objects.

    Args:
        data: API response dict
        league: League label stamped onto each projection

    Returns:
        List of Projection objects
    """
    logger.info(f"Extracting {league} projections...")

    player_names = build_player_lookup(data)
    projections: list[Projection] = []

    raw_projections = data.get("data") or []
    logger.debug(f"Processing {len(raw_projections)} raw projections...")

    skipped_wrong_type = 0
    skipped_no_line_score = 0
    skipped_no_player = 0
    added = 0

    for proj in raw_projections:
        if proj.get("type") != "projection":
            skipped_wrong_type += 1
            continue

        attrs = proj.get("attributes") or {}
        line_score = attrs.get("line_score")

        if line_score is None:
            skipped_no_line_score += 1
            continue

        rel = proj.get("relationships") or {}
        np_ref = (rel.get("new_player") or {}).get("data") or {}
        pid = np_ref.get("id")

        player = ""
        if pid is not None:
            player = player_names.get(pid, "")

        if not player:
            player = attrs.get("description") or ""

        if not player:
            skipped_no_player += 1
            continue

        projection = Projection(
            player=player,
            stat_type=attrs.get("stat_type") or "",
            line_score=line_score,
            odds_type=attrs.get("odds_type") or "",
            updated_at=attrs.get("updated_at") or "",
            league=league,
        )
        projections.append(projection)
        added += 1

    logger.info(
        f"Extraction complete ({league}): added={added}, skipped_type={skipped_wrong_type}, "
        f"skipped_no_line={skipped_no_line_score}, skipped_no_player={skipped_no_player}"
    )

    return projections


def extract_projections_from_payloads(
    payloads: dict[str, dict[str, Any]],
) -> list[Projection]:
    """Extract and concatenate projections for each league payload."""
    rows: list[Projection] = []
    for league_name, payload in payloads.items():
        rows.extend(extract_projections(payload, league=league_name))
    return rows


# ============================================================================
# File I/O
# ============================================================================

def save_projections(
    projections: list[Projection],
    path: str,
    *,
    source_path: str | None = None,
    league_label: str | None = None,
) -> None:
    """
    Save projections to JSON file.

    Args:
        projections: List of Projection objects
        path: Output file path
        source_path: Optional path to source file (for raw_snapshot field)
        league_label: Top-level league string (default derived from rows)
    """
    logger.info(f"Saving {len(projections)} projections to {path}...")

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
        logger.debug(f"Ensured directory exists: {parent}")

    if league_label is None:
        leagues = sorted({p.league for p in projections if p.league})
        league_label = "+".join(leagues) if leagues else "NBA"

    proj_dicts = [p.to_dict() for p in projections]
    export = ExportData(
        league=league_label,
        projections=proj_dicts,
        raw_snapshot=os.path.basename(source_path) if source_path else None,
    )

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved successfully to {path}")
    except IOError as e:
        logger.error(f"Failed to write file: {e}")
        raise


def load_projections_from_file(path: str) -> list[Projection]:
    """
    Load projections from either flat export or legacy API JSON.

    Args:
        path: File path

    Returns:
        List of Projection objects
    """
    logger.info(f"Loading projections from: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load file: {e}")
        raise

    if not isinstance(obj, dict):
        logger.warning("Loaded JSON is not a dict, returning empty list")
        return []

    projs = obj.get("projections")
    if isinstance(projs, list) and projs and isinstance(projs[0], dict):
        if "player" in projs[0]:
            logger.debug("Recognized as flat export format")
            default_league = obj.get("league") or "NBA"
            if isinstance(default_league, str) and "+" in default_league:
                default_league = "NBA"
            projections = [
                Projection(
                    player=p.get("player") or "",
                    stat_type=p.get("stat_type") or "",
                    line_score=p.get("line_score"),
                    odds_type=p.get("odds_type") or "",
                    updated_at=p.get("updated_at") or "",
                    league=p.get("league") or default_league,
                )
                for p in projs
            ]
            logger.info(f"Loaded {len(projections)} projections from flat export")
            return projections

    logger.debug("Treating as legacy API JSON format")
    league = obj.get("league") if isinstance(obj.get("league"), str) else "NBA"
    if "+" in str(league):
        league = "NBA"
    projections = extract_projections(obj, league=league)
    logger.info(f"Loaded {len(projections)} projections from API format")
    return projections


# ============================================================================
# Display
# ============================================================================

def print_projections_tsv(projections: list[Projection], limit: int | None = None) -> None:
    """
    Print projections as TSV to stdout.

    Args:
        projections: List of Projection objects
        limit: Maximum number to print (None = all)
    """
    print("league\tplayer\tstat_type\tline_score\todds_type\tupdated_at")

    slice_projs = projections if limit is None else projections[:limit]
    for proj in slice_projs:
        print(
            f"{proj.league}\t{proj.player}\t{proj.stat_type}\t"
            f"{proj.line_score}\t{proj.odds_type}\t{proj.updated_at}"
        )


# ============================================================================
# Main Scraper
# ============================================================================

class PrizePicks_Scraper:
    """Main scraper orchestrator."""

    def __init__(self) -> None:
        logger.info("Initializing PrizePicks_Scraper...")
        self.output_paths: list[str] = []
        self.projections: list[Projection] = []
        self.scraped_at: datetime | None = None
        logger.info("Initialization complete")

    @property
    def output_path(self) -> str:
        """Most recent output path (last league file written), for CLI --save-export."""
        if self.output_paths:
            return self.output_paths[-1]
        return resolve_output_path()

    def _save_by_league(self, projections: list[Projection]) -> list[str]:
        """Write one prizepicks_{league}_*.json file per league present."""
        when = datetime.now(_OUTPUT_TZ)
        by_league: dict[str, list[Projection]] = {}
        for row in projections:
            by_league.setdefault(row.league or "NBA", []).append(row)

        if not by_league:
            # No rows — still emit an empty file for the configured league(s)
            for league_name in DEFAULT_LEAGUE_NAMES:
                by_league[league_name] = []

        scraped_at = self.scraped_at or datetime.now(timezone.utc)
        self.scraped_at = scraped_at

        paths: list[str] = []
        for league_name in sorted(by_league.keys()):
            rows = by_league[league_name]
            path = resolve_output_path(league_name, when=when)
            save_projections(rows, path, league_label=league_name)
            try:
                from src.odds.load_snapshots import load_prizepicks_snapshot

                n = load_prizepicks_snapshot(
                    [p.to_dict() for p in rows],
                    league=league_slug(league_name),
                    scraped_at=scraped_at,
                )
                logger.info(
                    f"Supabase odds.nba_prizepicks upserted {n} rows ({league_name})"
                )
            except Exception as e:
                logger.error(f"Supabase prizepicks load failed (JSON kept): {e}")
            paths.append(path)
        self.output_paths = paths
        return paths

    def run(self) -> None:
        """Execute the full scrape pipeline for NBA."""
        try:
            logger.info("\n[Step 1/2] Fetching data...")
            payloads = fetch_projections_payloads()

            logger.info("\n[Step 2/2] Extracting and saving...")
            self.projections = extract_projections_from_payloads(payloads)
            self.scraped_at = datetime.now(timezone.utc)
            paths = self._save_by_league(self.projections)

            by_league: dict[str, int] = {}
            for p in self.projections:
                by_league[p.league] = by_league.get(p.league, 0) + 1
            summary = ", ".join(f"{k}={v}" for k, v in sorted(by_league.items())) or "none"

            logger.info("=" * 70)
            logger.info(
                f"✓ SUCCESS: {len(self.projections)} projections saved ({summary})"
            )
            for path in paths:
                logger.info(f"  → {path}")
            logger.info("=" * 70)

        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"✗ FAILED: {type(e).__name__}: {e}")
            logger.error("=" * 70)
            raise

    def load_from_file(self, path: str) -> None:
        """Load projections from file."""
        try:
            expanded = os.path.expanduser(path)
            self.projections = load_projections_from_file(expanded)
            logger.info(f"Loaded {len(self.projections)} projections")
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            raise


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="PrizePicks NBA projections scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           Fetch and save projections
  %(prog)s --head 20                 Fetch and show first 20
  %(prog)s --from-file data.json     Load from file and show sample
  %(prog)s --from-file data.json --tsv-all  Load and show all as TSV

Environment variables:
  LOG_LEVEL                          DEBUG, INFO, WARNING, or ERROR
  PRIZEPICKS_COOKIE                  Browser session cookie
  PRIZEPICKS_COOKIE_FILE             Path to cookie file (default: auto-managed,
                                      written automatically after a Playwright solve)
  PRIZEPICKS_IMPERSONATE             Comma-separated curl_cffi profiles
  PRIZEPICKS_HEADLESS                1/true/yes for headless Playwright (default headed)
  PRIZEPICKS_NBA_LEAGUE_ID           Override the resolved NBA league_id
        """,
    )

    parser.add_argument(
        "--head",
        type=int,
        default=12,
        metavar="N",
        help="Print first N rows as TSV (default 12, use 0 to skip)",
    )
    parser.add_argument(
        "--tsv-all",
        action="store_true",
        help="Print all rows as TSV (can be large)",
    )
    parser.add_argument(
        "--from-file",
        metavar="PATH",
        help="Load projections from JSON file instead of fetching",
    )
    parser.add_argument(
        "--print-brave-cdp-cmd",
        action="store_true",
        help="Print a Brave launch command with --remote-debugging-port and exit",
    )
    parser.add_argument(
        "--save-export",
        action="store_true",
        help="With --from-file: also save as timestamped export",
    )

    args = parser.parse_args()

    if args.print_brave_cdp_cmd:
        print(brave_cdp_launch_command())
        print()
        print("# After Brave opens and you clear any captcha:")
        print("PRIZEPICKS_CDP=1 python nba_prizepick.py")
        return

    if args.from_file:
        scraper = PrizePicks_Scraper()
        scraper.load_from_file(args.from_file)

        if args.save_export:
            logger.info("\nSaving export...")
            scraper._save_by_league(scraper.projections)

        if args.tsv_all:
            print()
            print_projections_tsv(scraper.projections, limit=None)
        elif args.head > 0:
            print(f"\nShowing first {args.head} rows (TSV):")
            print()
            print_projections_tsv(scraper.projections, limit=args.head)

    else:
        logger.info("Starting PrizePicks NBA scraper...")
        scraper = PrizePicks_Scraper()
        scraper.run()

        if args.tsv_all:
            print()
            print_projections_tsv(scraper.projections, limit=None)
        elif args.head > 0:
            print(f"\nShowing first {args.head} rows (TSV):")
            print()
            print_projections_tsv(scraper.projections, limit=args.head)


if __name__ == "__main__":
    main()