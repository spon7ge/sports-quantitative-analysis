import os, time, csv, json, pathlib, datetime as dt
import requests

# ---------------- CONFIG ----------------
API_KEY = os.environ.get("SGO_API_KEY", "")

BASE = "https://api.sportsgameodds.com/v2/events"

BOOKS = [
    "pinnacle",
    "draftkings", "fanduel", "betmgm", "caesars", "fanatics", "bet365",
    "prizepicks", "underdog",
    "bovada", "betonline",
]

STATS = ["points", "rebounds", "assists"]
# PLAYER_ID wildcard = this market for every player
ODD_IDS = ",".join(f"{s}-PLAYER_ID-game-ou-over" for s in STATS)

START_DATE = "2024-10-20"
END_DATE = "2025-12-29"
_HISTORICAL = pathlib.Path(__file__).resolve().parents[2] / "data" / "historical"
OUTPUT_CSV = _HISTORICAL / "nba_historical_odds.parquet"
CHECKPOINT = _HISTORICAL / "done_windows_openclose.json"
# Open/close is null on this API before 2026-01-17, so an earlier day
# cannot separate books. This date is inside the scrape window.
CHECK_DAY = "2026-01-20"
# ----------------------------------------

HEADERS = {"x-api-key": API_KEY}


def _ts(s):
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_events(start, end, books=BOOKS):
    """Yield every finalized NBA event in [start, end], following pagination."""
    cursor = None
    while True:
        params = {
            "leagueID": "NBA",
            "startsAfter": start,
            "startsBefore": end,
            "finalized": "true",
            "oddIDs": ODD_IDS,                 # if results look unfiltered, try `oddID`
            "includeOpposingOddIDs": "true",
            "includeOpenCloseOdds": "true",    # returns per-book open/close fields
            "limit": 100,                      # API may cap this lower (25-100)
        }
        if books:
            params["bookmakerID"] = ",".join(books)
        if cursor:
            params["cursor"] = cursor          # keep all other params identical between pages

        r = requests.get(BASE, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 429:               # rate limited
            time.sleep(10)
            continue
        r.raise_for_status()

        body = r.json()
        data = body.get("data", [])
        yield from data

        cursor = body.get("nextCursor")
        if not cursor or not data:
            break
        time.sleep(0.25)                       # stay well under 300 req/min


def rows(event):
    """One row per (market, bookmaker) with open and close lines."""
    players = event.get("players", {})
    tip = _ts(event["status"].get("startsAt"))

    for odd_id, odd in (event.get("odds") or {}).items():
        pid = odd.get("playerID")
        for book, b in (odd.get("byBookmaker") or {}).items():
            updated = _ts(b.get("lastUpdatedAt"))
            yield {
                "eventID": event["eventID"],
                "startsAt": event["status"]["startsAt"],
                "home": event["teams"]["home"]["teamID"],
                "away": event["teams"]["away"]["teamID"],
                "player": players.get(pid, {}).get("name", pid),
                "statID": odd.get("statID"),
                "sideID": odd.get("sideID"),
                "bookmaker": book,
                "openOdds": b.get("openOdds"),
                "openLine": b.get("openOverUnder"),
                "closeOdds": b.get("closeOdds"),
                "closeLine": b.get("closeOverUnder"),
                # audit columns
                "lastUpdatedAt": b.get("lastUpdatedAt"),
                "minsBeforeTip": (round((tip - updated).total_seconds() / 60)
                                  if tip and updated else None),
                "actual": odd.get("score"),
            }


def daterange(start, end, step_days=7):
    d = dt.date.fromisoformat(start)
    stop = dt.date.fromisoformat(end)
    while d < stop:
        nxt = min(d + dt.timedelta(days=step_days), stop)
        yield d.isoformat(), nxt.isoformat()
        d = nxt


_OPEN_FIELDS = ("openOdds", "openSpread", "openOverUnder")
_CLOSE_FIELDS = ("closeOdds", "closeSpread", "closeOverUnder")


def _populated(entry, fields):
    return any(entry.get(k) is not None for k in fields)


def market_shape(odd_id):
    """Collapse a player id so the report lists market families, not every player."""
    parts = odd_id.split("-")
    if len(parts) >= 5 and parts[1] not in ("home", "away", "all"):
        parts = [parts[0], "*", *parts[2:]]
    return "-".join(parts)


def tally_books(events):
    """Count, per book, how many quotes have an open and how many have a close."""
    tally, markets, sample = {}, set(), None
    for ev in events:
        for odd_id, odd in (ev.get("odds") or {}).items():
            markets.add(market_shape(odd_id))
            for book, entry in (odd.get("byBookmaker") or {}).items():
                counts = tally.setdefault(
                    book,
                    {
                        "with_open": 0,
                        "no_open": 0,
                        "with_close": 0,
                        "no_close": 0,
                    },
                )
                has_open = _populated(entry, _OPEN_FIELDS)
                has_close = _populated(entry, _CLOSE_FIELDS)
                counts["with_open" if has_open else "no_open"] += 1
                counts["with_close" if has_close else "no_close"] += 1
                if has_close and sample is None:
                    sample = (book, odd_id, entry)
    return tally, markets, sample


def format_check(day, tally, markets, sample, books=None):
    """Same layout as the tennis check, with open counts beside close counts."""
    if books is None:
        books = BOOKS
    lines = [f"Open/close availability on {day}:"]
    ranked = sorted(
        tally.items(),
        key=lambda item: (
            -item[1]["with_close"],
            -item[1]["with_open"],
            item[0],
        ),
    )
    for book, counts in ranked:
        flag = "" if book in books else "   (not in your BOOKS list)"
        lines.append(
            f"  {book}: with_open={counts['with_open']}"
            f"  no_open={counts['no_open']}"
            f"  with_close={counts['with_close']}"
            f"  no_close={counts['no_close']}{flag}"
        )
    missing = [book for book in books if book not in tally]
    if missing:
        lines.append("No data that day for: " + ", ".join(missing))
    lines.append("")
    shown = ", ".join(sorted(markets)) or "(none)"
    lines.append("oddIDs returned: " + shown)
    if sample:
        book, odd_id, entry = sample
        lines.append(
            f"\nSample byBookmaker entry ({book}, {odd_id}):\n"
            + json.dumps(entry, indent=2)
        )
    return "\n".join(lines)


def check(day=CHECK_DAY):
    """One-day test: per book, how many markets have an open and a close?"""
    nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
    # No book filter, so books outside BOOKS still show up and can be added.
    tally, markets, sample = tally_books(fetch_events(day, nxt, books=None))
    print(format_check(day, tally, markets, sample))


def run(start=START_DATE, end=END_DATE):
    _HISTORICAL.mkdir(parents=True, exist_ok=True)
    cp = pathlib.Path(CHECKPOINT)
    done = set(json.loads(cp.read_text())) if cp.exists() else set()
    new_file = not pathlib.Path(OUTPUT_CSV).exists()

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = None
        for s, e in daterange(start, end):
            key = f"{s}_{e}"
            if key in done:
                continue
            n = 0
            for ev in fetch_events(s, e):
                for row in rows(ev):
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=row.keys())
                        if new_file:
                            writer.writeheader()
                    writer.writerow(row)
                    n += 1
            f.flush()
            done.add(key)
            cp.write_text(json.dumps(sorted(done)))
            print(f"{key}: {n} rows")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    else:
        run()