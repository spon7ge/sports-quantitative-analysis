"""Find same-line arbitrage across Novig, ProphetX, FanDuel, and DraftKings.

Reads the newest snapshot written for each book. A bet is an arbitrage when the
best price on each side of one market sums to less than 100% implied probability.
Player props match on player, stat, and line. Game lines match on the two teams,
the slate date, and the number: moneyline, full-game spread (MLB run line), and
full-game total. First-half and quarter markets stay out so they cannot pair
with the full game.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.models.odds import american_to_decimal, american_to_implied_probability

BOOKS: tuple[str, ...] = ("draftkings", "fanduel", "novig", "prophetx")
_EDGE_FLOOR = 1e-6
_DISPLAY_LIMIT = 40
_NY = ZoneInfo("America/New_York")

_EVENT_SPLIT = re.compile(r"\s+(?:@|at|vs\.?|v)\s+", re.IGNORECASE)
_TRAILING_LINE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*$")
_PAREN = re.compile(r"\([^)]*\)")
_NON_TEAM = re.compile(r"[^a-z0-9]+")
_NON_PLAYER = re.compile(r"[^a-z0-9]+")
_FRACTION = re.compile(r"(\.\d+)")
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

_PATTERNS: dict[str, tuple[str, ...]] = {
    "novig": ("novig_*_props.json", "novig_*_team.json"),
    "prophetx": ("prophetx_*_props.json", "prophetx_*_team.json"),
    "fanduel": ("fanduel_*.json",),
    "draftkings": ("draftkings_*.json",),
}


@dataclass(frozen=True)
class _Teams:
    display: dict[str, str]
    aliases: dict[str, str]
    abbrevs: dict[str, str]
    nicknames: tuple[tuple[str, str], ...]


def _norm_team_text(text: str) -> str:
    cleaned = _PAREN.sub(" ", text)
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    cleaned = _NON_TEAM.sub(" ", cleaned.lower().replace("'", ""))
    return " ".join(cleaned.split())


def _build_teams(
    rows: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]],
) -> _Teams:
    display: dict[str, str] = {}
    aliases: dict[str, str] = {}
    abbrevs: dict[str, str] = {}
    nicknames: list[tuple[str, str]] = []
    for team_id, name, abbrev_list, nick_list in rows:
        display[team_id] = name
        aliases[_norm_team_text(name)] = team_id
        for abbrev in abbrev_list:
            token = abbrev.lower()
            abbrevs[token] = team_id
            aliases[token] = team_id
        for nick in nick_list:
            normalized = _norm_team_text(nick)
            nicknames.append((normalized, team_id))
            aliases[normalized] = team_id
    nicknames.sort(key=lambda item: len(item[0]), reverse=True)
    return _Teams(display, aliases, abbrevs, tuple(nicknames))


_NBA = _build_teams(
    [
        ("atl", "Atlanta Hawks", ("atl",), ("hawks",)),
        ("bos", "Boston Celtics", ("bos",), ("celtics",)),
        ("bkn", "Brooklyn Nets", ("bkn", "brk"), ("nets",)),
        ("cha", "Charlotte Hornets", ("cha", "cho"), ("hornets",)),
        ("chi", "Chicago Bulls", ("chi",), ("bulls",)),
        ("cle", "Cleveland Cavaliers", ("cle",), ("cavaliers", "cavs")),
        ("dal", "Dallas Mavericks", ("dal",), ("mavericks", "mavs")),
        ("den", "Denver Nuggets", ("den",), ("nuggets",)),
        ("det", "Detroit Pistons", ("det",), ("pistons",)),
        ("gsw", "Golden State Warriors", ("gsw", "gs"), ("warriors",)),
        ("hou", "Houston Rockets", ("hou",), ("rockets",)),
        ("ind", "Indiana Pacers", ("ind",), ("pacers",)),
        ("lac", "LA Clippers", ("lac",), ("clippers",)),
        ("lal", "Los Angeles Lakers", ("lal",), ("lakers",)),
        ("mem", "Memphis Grizzlies", ("mem",), ("grizzlies",)),
        ("mia", "Miami Heat", ("mia",), ("heat",)),
        ("mil", "Milwaukee Bucks", ("mil",), ("bucks",)),
        ("min", "Minnesota Timberwolves", ("min",), ("timberwolves", "wolves")),
        ("nop", "New Orleans Pelicans", ("nop", "no"), ("pelicans",)),
        ("nyk", "New York Knicks", ("nyk", "ny"), ("knicks",)),
        ("okc", "Oklahoma City Thunder", ("okc",), ("thunder",)),
        ("orl", "Orlando Magic", ("orl",), ("magic",)),
        ("phi", "Philadelphia 76ers", ("phi",), ("76ers", "sixers")),
        ("phx", "Phoenix Suns", ("phx", "pho"), ("suns",)),
        ("por", "Portland Trail Blazers", ("por",), ("trail blazers", "blazers")),
        ("sac", "Sacramento Kings", ("sac",), ("kings",)),
        ("sas", "San Antonio Spurs", ("sas", "sa"), ("spurs",)),
        ("tor", "Toronto Raptors", ("tor",), ("raptors",)),
        ("uta", "Utah Jazz", ("uta", "utah"), ("jazz",)),
        ("was", "Washington Wizards", ("was", "wsh"), ("wizards",)),
    ]
)

_MLB = _build_teams(
    [
        ("ari", "Arizona Diamondbacks", ("ari", "az"), ("diamondbacks", "d-backs")),
        ("atl", "Atlanta Braves", ("atl",), ("braves",)),
        ("bal", "Baltimore Orioles", ("bal",), ("orioles",)),
        ("bos", "Boston Red Sox", ("bos",), ("red sox",)),
        ("chc", "Chicago Cubs", ("chc",), ("cubs",)),
        ("cws", "Chicago White Sox", ("cws", "chw"), ("white sox",)),
        ("cin", "Cincinnati Reds", ("cin",), ("reds",)),
        ("cle", "Cleveland Guardians", ("cle",), ("guardians",)),
        ("col", "Colorado Rockies", ("col",), ("rockies",)),
        ("det", "Detroit Tigers", ("det",), ("tigers",)),
        ("hou", "Houston Astros", ("hou",), ("astros",)),
        ("kc", "Kansas City Royals", ("kc", "kcr"), ("royals",)),
        ("laa", "Los Angeles Angels", ("laa",), ("angels",)),
        ("lad", "Los Angeles Dodgers", ("lad",), ("dodgers",)),
        ("mia", "Miami Marlins", ("mia",), ("marlins",)),
        ("mil", "Milwaukee Brewers", ("mil",), ("brewers",)),
        ("min", "Minnesota Twins", ("min",), ("twins",)),
        ("nym", "New York Mets", ("nym",), ("mets",)),
        ("nyy", "New York Yankees", ("nyy",), ("yankees",)),
        ("ath", "Athletics", ("ath", "oak"), ("athletics",)),
        ("phi", "Philadelphia Phillies", ("phi",), ("phillies",)),
        ("pit", "Pittsburgh Pirates", ("pit",), ("pirates",)),
        ("sd", "San Diego Padres", ("sd", "sdp"), ("padres",)),
        ("sf", "San Francisco Giants", ("sf", "sfg"), ("giants",)),
        ("sea", "Seattle Mariners", ("sea",), ("mariners",)),
        ("stl", "St. Louis Cardinals", ("stl",), ("cardinals",)),
        ("tb", "Tampa Bay Rays", ("tb", "tbr"), ("rays",)),
        ("tex", "Texas Rangers", ("tex",), ("rangers",)),
        ("tor", "Toronto Blue Jays", ("tor",), ("blue jays",)),
        ("wsh", "Washington Nationals", ("wsh", "was"), ("nationals",)),
    ]
)

_LEAGUE_TEAMS = {"nba": _NBA, "mlb": _MLB}


def canonical_team(text: str | None, league: str) -> str | None:
    """Map a book-specific team label onto one id for that league."""
    table = _LEAGUE_TEAMS.get(league)
    if table is None or not text:
        return None
    norm = _norm_team_text(text)
    if not norm:
        return None
    if norm in table.aliases:
        return table.aliases[norm]
    padded = f" {norm} "
    for nick, team_id in table.nicknames:
        if f" {nick} " in padded:
            return team_id
    for token in norm.split():
        hit = table.abbrevs.get(token)
        if hit:
            return hit
    return None


def team_display(team_id: str, league: str) -> str:
    table = _LEAGUE_TEAMS.get(league)
    if table is None:
        return team_id
    return table.display.get(team_id, team_id)


def event_teams(name: str, league: str) -> frozenset[str]:
    cleaned = _PAREN.sub(" ", name or "")
    parts = _EVENT_SPLIT.split(cleaned.strip())
    if len(parts) != 2:
        return frozenset()
    left = canonical_team(parts[0], league)
    right = canonical_team(parts[1], league)
    if not left or not right or left == right:
        return frozenset()
    return frozenset({left, right})


def player_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _NON_PLAYER.sub(" ", text.lower().replace("'", ""))
    tokens = [tok for tok in text.split() if tok]
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    folded: list[str] = []
    initials = ""
    for tok in tokens:
        if len(tok) == 1:
            initials += tok
            continue
        if initials:
            folded.append(initials)
            initials = ""
        folded.append(tok)
    if initials:
        folded.append(initials)
    return " ".join(folded)


def slate_of(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = _FRACTION.sub(lambda match: match.group(1)[:7], text, count=1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed.astimezone(_NY).date().isoformat()


def parse_american(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().translate({ord("−"): "-", ord("–"): "-", ord("—"): "-"})
        cleaned = cleaned.replace("+", "")
        if not cleaned:
            return None
        value = cleaned
    try:
        odds = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            odds = int(float(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    if odds == 0 or abs(odds) > 100_000:
        return None
    return odds


def parse_line(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        line = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if line != line:  # NaN
        return None
    return line


def format_number(value: float) -> str:
    rounded = round(float(value), 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def format_signed(value: float) -> str:
    text = format_number(abs(value))
    if value > 0:
        return f"+{text}"
    if value < 0:
        return f"-{text}"
    return "0"


def format_american(odds: int) -> str:
    if odds > 0:
        return f"+{odds}"
    return str(odds)


@dataclass(frozen=True)
class Quote:
    book: str
    league: str
    kind: str
    market_label: str
    teams: frozenset[str]
    slate: str
    player_key: str
    player_label: str
    stat: str
    line: float | None
    side: str
    side_label: str
    american: int
    event_label: str


@dataclass(frozen=True)
class Leg:
    book: str
    american: int
    side_label: str


@dataclass(frozen=True)
class Arb:
    league: str
    market: str
    description: str
    edge: float
    left: Leg
    right: Leg


def _split_label_line(name: str) -> tuple[str, float | None]:
    match = _TRAILING_LINE.search(name.strip())
    if not match:
        return name.strip(), None
    line = parse_line(match.group(1))
    if line is None:
        return name.strip(), None
    return name[: match.start()].strip(), line


def _over_under(name: str) -> str | None:
    token = (name or "").strip().lower().split(" ", 1)[0]
    if token in {"over", "under"}:
        return token
    return None


def _quote(
    *,
    book: str,
    league: str,
    kind: str,
    market_label: str,
    teams: frozenset[str],
    slate: str,
    player_key_text: str,
    player_label: str,
    stat: str,
    line: float | None,
    side: str,
    side_label: str,
    american: int,
    event_label: str,
) -> Quote | None:
    try:
        decimal = american_to_decimal(american)
    except ValueError:
        return None
    if decimal <= 1:
        return None
    return Quote(
        book=book,
        league=league,
        kind=kind,
        market_label=market_label,
        teams=teams,
        slate=slate,
        player_key=player_key_text,
        player_label=player_label,
        stat=stat,
        line=line,
        side=side,
        side_label=side_label,
        american=american,
        event_label=event_label,
    )


def _teams_from_game(game: dict, league: str) -> frozenset[str]:
    found = event_teams(str(game.get("name") or ""), league)
    if found:
        return found
    ids: list[str] = []
    for comp in game.get("competitors") or []:
        if not isinstance(comp, dict):
            continue
        team_id = canonical_team(str(comp.get("name") or ""), league)
        if team_id is None and comp.get("abbreviation"):
            team_id = canonical_team(str(comp["abbreviation"]), league)
        if team_id and team_id not in ids:
            ids.append(team_id)
    if len(ids) == 2:
        return frozenset(ids)
    return frozenset()


def _append(quotes: list[Quote], quote: Quote | None) -> None:
    if quote is not None:
        quotes.append(quote)


def _props_from_games(
    games: list[dict],
    *,
    book: str,
    league: str,
) -> list[Quote]:
    quotes: list[Quote] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        teams = _teams_from_game(game, league)
        slate = slate_of(game.get("scheduled"))
        event_label = str(game.get("name") or "")
        for prop in game.get("props") or []:
            if not isinstance(prop, dict):
                continue
            label = str(prop.get("player") or "").strip()
            key = player_key(label)
            stat = str(prop.get("stat") or "").strip().lower()
            line = parse_line(prop.get("line"))
            if not key or not stat or line is None:
                continue
            for side in ("over", "under"):
                payload = prop.get(side)
                if not isinstance(payload, dict):
                    continue
                american = parse_american(payload.get("american"))
                if american is None:
                    continue
                _append(
                    quotes,
                    _quote(
                        book=book,
                        league=league,
                        kind="prop",
                        market_label="prop",
                        teams=teams,
                        slate=slate,
                        player_key_text=key,
                        player_label=label,
                        stat=stat,
                        line=line,
                        side=side,
                        side_label=side,
                        american=american,
                        event_label=event_label,
                    ),
                )
        markets = game.get("team_markets")
        if isinstance(markets, dict):
            quotes.extend(
                _game_lines_from_markets(
                    markets,
                    book=book,
                    league=league,
                    teams=teams,
                    slate=slate,
                    event_label=event_label,
                )
            )
    return quotes


def _game_lines_from_markets(
    markets: dict,
    *,
    book: str,
    league: str,
    teams: frozenset[str],
    slate: str,
    event_label: str,
) -> list[Quote]:
    if len(teams) != 2:
        return []
    quotes: list[Quote] = []
    for market_key, rows in markets.items():
        key = str(market_key or "").strip().lower()
        if key.startswith("first_half") or "quarter" in key or "inning" in key:
            continue
        if key == "moneyline":
            kind, label = "moneyline", "moneyline"
        elif key in {"spread", "run_line", "run line"}:
            kind = "spread"
            label = "run line" if key != "spread" else "spread"
        elif key == "total":
            kind, label = "total", "total"
        else:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            american = parse_american(row.get("american"))
            if american is None:
                continue
            if kind == "total":
                side = _over_under(name)
                line = parse_line(row.get("line"))
                if line is None:
                    _, line = _split_label_line(name)
                if side is None or line is None:
                    continue
                _append(
                    quotes,
                    _quote(
                        book=book,
                        league=league,
                        kind=kind,
                        market_label=label,
                        teams=teams,
                        slate=slate,
                        player_key_text="",
                        player_label="",
                        stat="total",
                        line=abs(line),
                        side=side,
                        side_label=side,
                        american=american,
                        event_label=event_label,
                    ),
                )
                continue
            team_text, parsed_line = _split_label_line(name)
            team_id = canonical_team(team_text or name, league)
            if team_id is None or team_id not in teams:
                continue
            if kind == "moneyline":
                _append(
                    quotes,
                    _quote(
                        book=book,
                        league=league,
                        kind=kind,
                        market_label=label,
                        teams=teams,
                        slate=slate,
                        player_key_text="",
                        player_label="",
                        stat="moneyline",
                        line=None,
                        side=team_id,
                        side_label=team_display(team_id, league),
                        american=american,
                        event_label=event_label,
                    ),
                )
                continue
            line = parse_line(row.get("line"))
            if line is None:
                line = parsed_line
            if line is None:
                continue
            _append(
                quotes,
                _quote(
                    book=book,
                    league=league,
                    kind=kind,
                    market_label=label,
                    teams=teams,
                    slate=slate,
                    player_key_text="",
                    player_label="",
                    stat=kind,
                    line=line,
                    side=team_id,
                    side_label=f"{team_display(team_id, league)} {format_signed(line)}",
                    american=american,
                    event_label=event_label,
                ),
            )
    return quotes


def _picks_from_payload(payload: dict, *, book: str, league: str) -> list[Quote]:
    quotes: list[Quote] = []
    for pick in payload.get("picks") or []:
        if not isinstance(pick, dict):
            continue
        stat = str(pick.get("stat_name") or "").strip().lower()
        choice = str(pick.get("choice") or "").strip().lower()
        if choice not in {"over", "under"} or not stat:
            continue
        if stat in {"moneyline", "spread", "total", "run_line", "run line"}:
            continue
        label = str(pick.get("full_name") or "").strip()
        key = player_key(label)
        line = parse_line(pick.get("stat_value"))
        american = parse_american(pick.get("american_price"))
        if not key or line is None or american is None:
            continue
        event_label = str(pick.get("event_name") or "")
        _append(
            quotes,
            _quote(
                book=book,
                league=league,
                kind="prop",
                market_label="prop",
                teams=event_teams(event_label, league),
                slate=slate_of(pick.get("start_time")),
                player_key_text=key,
                player_label=label,
                stat=stat,
                line=line,
                side=choice,
                side_label=choice,
                american=american,
                event_label=event_label,
            ),
        )
    for row in payload.get("game_lines") or []:
        if isinstance(row, dict):
            quotes.extend(_flat_game_line(row, book=book, league=league))
    return quotes


def _flat_game_line(row: dict, *, book: str, league: str) -> list[Quote]:
    market = str(row.get("market") or "").strip().lower()
    event_label = str(row.get("event_name") or "")
    teams = event_teams(event_label, league)
    if len(teams) != 2:
        return []
    american = parse_american(row.get("american"))
    if american is None:
        return []
    slate = slate_of(row.get("start_time") or row.get("scheduled"))
    side_raw = str(row.get("side") or row.get("name") or "")
    line = parse_line(row.get("line"))
    if market == "total":
        side = side_raw.strip().lower()
        if side not in {"over", "under"} or line is None:
            return []
        quote = _quote(
            book=book,
            league=league,
            kind="total",
            market_label="total",
            teams=teams,
            slate=slate,
            player_key_text="",
            player_label="",
            stat="total",
            line=abs(line),
            side=side,
            side_label=side,
            american=american,
            event_label=event_label,
        )
        return [quote] if quote else []
    if market == "moneyline":
        team_id = canonical_team(side_raw, league)
        if team_id is None or team_id not in teams:
            return []
        quote = _quote(
            book=book,
            league=league,
            kind="moneyline",
            market_label="moneyline",
            teams=teams,
            slate=slate,
            player_key_text="",
            player_label="",
            stat="moneyline",
            line=None,
            side=team_id,
            side_label=team_display(team_id, league),
            american=american,
            event_label=event_label,
        )
        return [quote] if quote else []
    if market not in {"spread", "run_line", "run line"}:
        return []
    team_id = canonical_team(side_raw, league)
    if team_id is None or team_id not in teams or line is None:
        return []
    label = "run line" if market != "spread" else "spread"
    quote = _quote(
        book=book,
        league=league,
        kind="spread",
        market_label=label,
        teams=teams,
        slate=slate,
        player_key_text="",
        player_label="",
        stat="spread",
        line=line,
        side=team_id,
        side_label=f"{team_display(team_id, league)} {format_signed(line)}",
        american=american,
        event_label=event_label,
    )
    return [quote] if quote else []


def quotes_from_payload(book: str, league: str, payload: dict) -> list[Quote]:
    """Normalize one snapshot payload into comparable quotes."""
    if not isinstance(payload, dict):
        return []
    stated = str(payload.get("league") or league).strip().lower()
    if stated != league:
        return []
    quotes: list[Quote] = []
    games = payload.get("games")
    if isinstance(games, list):
        quotes.extend(_props_from_games(games, book=book, league=league))
    quotes.extend(_picks_from_payload(payload, book=book, league=league))
    return quotes


def _line_key(line: float | None) -> float | None:
    if line is None:
        return None
    return round(float(line), 2)


def _best(quotes: list[Quote]) -> Quote:
    return max(
        quotes,
        key=lambda quote: (american_to_decimal(quote.american), quote.book),
    )


def _edge(left: Quote, right: Quote) -> float:
    return 1.0 - (
        american_to_implied_probability(left.american)
        + american_to_implied_probability(right.american)
    )


def _prefer_label(left: str, right: str) -> str:
    if len(left) != len(right):
        return left if len(left) > len(right) else right
    return min(left, right)


def _event_label(left: Quote, right: Quote) -> str:
    options = [left.event_label, right.event_label]
    with_at = [label for label in options if "@" in label]
    pool = with_at or [label for label in options if label]
    if not pool:
        return ""
    return max(pool, key=len)


def _ordered(left: Quote, right: Quote) -> tuple[Quote, Quote]:
    rank = {"over": 0, "under": 1}
    if left.side in rank and right.side in rank:
        return (left, right) if rank[left.side] <= rank[right.side] else (right, left)
    if left.side_label <= right.side_label:
        return left, right
    return right, left


def _leg(quote: Quote) -> Leg:
    return Leg(book=quote.book, american=quote.american, side_label=quote.side_label)


def _make_arb(
    left: Quote,
    right: Quote,
    *,
    description: str,
    market: str,
) -> Arb | None:
    margin = _edge(left, right)
    if margin <= _EDGE_FLOOR:
        return None
    first, second = _ordered(left, right)
    return Arb(
        league=first.league,
        market=market,
        description=description,
        edge=margin,
        left=_leg(first),
        right=_leg(second),
    )


def _market_name(left: Quote, right: Quote) -> str:
    labels = {left.market_label, right.market_label}
    if labels == {"run line"}:
        return "run line"
    if "prop" in labels:
        return "prop"
    if len(labels) == 1:
        return next(iter(labels))
    return left.kind


def find_arbs(quotes: list[Quote]) -> list[Arb]:
    """Return every positive-edge pair, best price on each side."""
    arbs: list[Arb] = []
    arbs.extend(_two_way_arbs(quotes))
    arbs.extend(_side_arbs(quotes))
    arbs.sort(key=lambda arb: (-arb.edge, arb.league, arb.market, arb.description))
    return arbs


def _two_way_arbs(quotes: list[Quote]) -> list[Arb]:
    grouped: dict[tuple, dict[str, list[Quote]]] = {}
    for quote in quotes:
        if quote.kind not in {"prop", "total"} or quote.side not in {"over", "under"}:
            continue
        if quote.kind == "total" and len(quote.teams) != 2:
            continue
        if quote.kind == "prop" and not quote.player_key:
            continue
        key = (
            quote.league,
            quote.kind,
            quote.teams,
            quote.slate,
            quote.player_key,
            quote.stat,
            _line_key(quote.line),
        )
        grouped.setdefault(key, {"over": [], "under": []})[quote.side].append(quote)
    found: list[Arb] = []
    for sides in grouped.values():
        if not sides["over"] or not sides["under"]:
            continue
        over = _best(sides["over"])
        under = _best(sides["under"])
        if over.kind == "prop":
            stat = over.stat.replace("_", " ")
            player = _prefer_label(over.player_label, under.player_label)
            line = over.line if over.line is not None else under.line
            description = f"{player} {stat} {format_number(line or 0)}"
        else:
            line = over.line if over.line is not None else under.line
            description = (
                f"{_event_label(over, under)} total {format_number(line or 0)}"
            )
        arb = _make_arb(
            over,
            under,
            description=description,
            market=_market_name(over, under),
        )
        if arb is not None:
            found.append(arb)
    return found


def _side_arbs(quotes: list[Quote]) -> list[Arb]:
    grouped: dict[tuple, list[Quote]] = {}
    for quote in quotes:
        if quote.kind not in {"moneyline", "spread"} or len(quote.teams) != 2:
            continue
        number = None if quote.kind == "moneyline" else _line_key(abs(quote.line or 0))
        key = (quote.league, quote.kind, quote.teams, quote.slate, number)
        grouped.setdefault(key, []).append(quote)
    found: list[Arb] = []
    for bucket in grouped.values():
        if bucket[0].kind == "moneyline":
            by_team: dict[str, list[Quote]] = {}
            for quote in bucket:
                by_team.setdefault(quote.side, []).append(quote)
            teams = bucket[0].teams
            if set(by_team) != set(teams) or len(by_team) != 2:
                continue
            pair = [_best(items) for items in by_team.values()]
            description = _event_label(pair[0], pair[1]) or "moneyline"
            arb = _make_arb(
                pair[0],
                pair[1],
                description=description,
                market="moneyline",
            )
            if arb is not None:
                found.append(arb)
            continue
        by_side: dict[tuple[str, int], list[Quote]] = {}
        for quote in bucket:
            if quote.line is None:
                continue
            sign = 1 if quote.line > 0 else -1 if quote.line < 0 else 0
            by_side.setdefault((quote.side, sign), []).append(quote)
        if len(by_side) != 2:
            continue
        left_key, right_key = list(by_side)
        (left_team, left_sign), (right_team, right_sign) = left_key, right_key
        opposite = left_team != right_team and (
            (left_sign == 0 and right_sign == 0)
            or left_sign == -right_sign
        )
        if not opposite:
            continue
        left = _best(by_side[left_key])
        right = _best(by_side[right_key])
        description = _event_label(left, right) or "spread"
        arb = _make_arb(
            left,
            right,
            description=description,
            market=_market_name(left, right),
        )
        if arb is not None:
            found.append(arb)
    return found


def format_report(
    arbs: list[Arb],
    *,
    books_seen: list[str],
    books_missing: list[str],
    game_line_books: list[str] | None = None,
) -> list[str]:
    lines = ["=== Arbitrage ==="]
    if books_seen:
        lines.append("  books: " + ", ".join(sorted(books_seen)))
    if game_line_books is not None:
        shown = ", ".join(game_line_books) if game_line_books else "none"
        lines.append(f"  game lines: {shown}")
    if books_missing:
        lines.append("  missing: " + ", ".join(sorted(books_missing)))
    if not arbs:
        lines.append("  none")
        return lines
    ordered = sorted(arbs, key=lambda arb: (-arb.edge, arb.league, arb.description))
    lines.append(f"  {len(ordered)} found")
    for arb in ordered[:_DISPLAY_LIMIT]:
        left_price = format_american(arb.left.american)
        right_price = format_american(arb.right.american)
        lines.append(
            f"  {arb.league} {arb.market}: {arb.description} | "
            f"{arb.left.side_label} {arb.left.book} {left_price} | "
            f"{arb.right.side_label} {arb.right.book} {right_price} | "
            f"edge {arb.edge * 100:.2f}%"
        )
    extra = len(ordered) - _DISPLAY_LIMIT
    if extra > 0:
        lines.append(f"  and {extra} more")
    return lines


def default_props_root() -> Path:
    from src.scrapers.nba.paths import repo_root

    return repo_root(__file__) / "data" / "props"


def _latest_file(
    directory: Path,
    pattern: str,
    not_before: float | None,
) -> Path | None:
    if not directory.is_dir():
        return None
    matches: list[Path] = []
    for path in directory.glob(pattern):
        team_file_on_props_glob = (
            path.name.endswith("_team.json") and pattern.endswith("_props.json")
        )
        if not path.is_file() or team_file_on_props_glob:
            continue
        if not_before is not None and path.stat().st_mtime + 1 < not_before:
            continue
        matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def _load_file(path: Path, *, book: str, league: str) -> list[Quote]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"snapshot {path.name} is not an object")
    return quotes_from_payload(book, league, payload)


def _load_book(
    root: Path,
    league: str,
    book: str,
    not_before: float | None,
) -> tuple[list[Quote], str]:
    directory = root / book / league
    patterns = _PATTERNS.get(book, ())
    paths = [
        path
        for pattern in patterns
        if (path := _latest_file(directory, pattern, not_before)) is not None
    ]
    if not paths:
        return [], "missing"
    quotes: list[Quote] = []
    loaded = False
    for path in paths:
        try:
            quotes.extend(_load_file(path, book=book, league=league))
        except (OSError, json.JSONDecodeError, UnicodeError, ValueError) as exc:
            print(f"  {book}: unreadable snapshot {path.name} ({exc})", flush=True)
            continue
        loaded = True
    if not loaded:
        return [], "missing"
    return quotes, "ok"


def report_arbitrage(
    *,
    league: str | None = None,
    not_before: float | None = None,
    root: Path | None = None,
) -> int:
    """Print arbitrage from the newest snapshots. Returns how many were found.

    ``not_before`` is an epoch timestamp. Snapshots older than that are ignored
    so a book that failed this cycle is not compared with a stale file.
    """
    leagues = (league,) if league else ("mlb", "nba")
    props_root = root or default_props_root()
    quotes: list[Quote] = []
    seen: set[str] = set()
    missing: set[str] = set()
    for league_name in leagues:
        if league_name not in _LEAGUE_TEAMS:
            print(f"  unknown league {league_name}", flush=True)
            continue
        for book in BOOKS:
            loaded, status = _load_book(props_root, league_name, book, not_before)
            if status == "ok":
                seen.add(book)
                quotes.extend(loaded)
            else:
                missing.add(book)
    arbs = find_arbs(quotes)
    game_line_books = sorted({quote.book for quote in quotes if quote.kind != "prop"})
    for line in format_report(
        arbs,
        books_seen=sorted(seen),
        books_missing=sorted(missing - seen),
        game_line_books=game_line_books,
    ):
        print(line, flush=True)
    return len(arbs)
