import os, time, csv, json, pathlib, datetime as dt
import requests

# ---------------- CONFIG ----------------
API_KEY = os.environ.get("SGO_API_KEY", "")

BASE = "https://api.sportsgameodds.com/v2/events"

LEAGUES = ["ATP", "WTA"]

BOOKS = [
    "pinnacle", "caesars", "draftkings", "fanduel", "betmgm", "bet365",
    "prizepicks",                                   # DFS, totals
    "betonline", "lowvig", "bovada", "matchbook",   # all 198 in your check
    "betfairexchange",                              # exchange
]

# Tennis oddID format: {statID}-{statEntityID}-{periodID}-{betTypeID}-{sideID}
# NOTE: in tennis, `points` = SETS won (used for the match moneyline) and
#       `games` = GAMES won (used for totals and handicaps). Don't mix them up.
# includeOpposingOddIDs=true returns the other side of each market automatically.
ODD_IDS = [
    "points-home-game-ml-home",     # match winner (moneyline), settled on set score
    "games-all-game-ou-over",       # total games, full match (over/under)
    "games-home-game-sp-home",      # games handicap, full match (verify in /v2/markets if empty)
    # first-set versions (period codes 1s, 2s, 3s), uncomment to add:
    # "games-all-1s-ou-over",
    # "points-home-1s-ml-home",
]

# START_DATE = "2025-01-01"
START_DATE = "2024-01-01"
END_DATE = "2024-04-01"
STEP_DAYS = 3                       # tennis has many matches per day, so use small windows
_HISTORICAL = pathlib.Path(__file__).resolve().parents[2] / "data" / "historical"
OUTPUT_CSV = _HISTORICAL / "tennis_historical_odds.parquet"
CHECKPOINT = _HISTORICAL / "done_windows_tennis.json"
CHECK_DAY = "2026-01-20"            # a day with matches inside your range, used by `check`
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
    """Yield every finalized ATP/WTA match in [start, end], following pagination."""
    cursor = None
    while True:
        params = {
            "sportID": "TENNIS",
            "leagueID": ",".join(LEAGUES),
            "startsAfter": start,
            "startsBefore": end,
            "finalized": "true",
            "oddIDs": ",".join(ODD_IDS),       # if results look unfiltered, try `oddID`
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


def _name(team):
    names = (team or {}).get("names") or {}
    return names.get("long") or (team or {}).get("teamID")


def rows(event):
    """One row per (market, bookmaker) with open and close lines."""
    tip = _ts(event["status"].get("startsAt"))
    teams = event.get("teams", {})
    home, away = teams.get("home"), teams.get("away")

    for odd_id, odd in (event.get("odds") or {}).items():
        for book, b in (odd.get("byBookmaker") or {}).items():
            updated = _ts(b.get("lastUpdatedAt"))
            yield {
                "eventID": event["eventID"],
                "leagueID": event.get("leagueID"),
                "startsAt": event["status"]["startsAt"],
                "home": _name(home),
                "away": _name(away),
                "oddID": odd_id,
                "statID": odd.get("statID"),
                "statEntityID": odd.get("statEntityID"),
                "periodID": odd.get("periodID"),
                "betTypeID": odd.get("betTypeID"),
                "sideID": odd.get("sideID"),
                "bookmaker": book,
                "openOdds": b.get("openOdds"),
                "openSpread": b.get("openSpread"),
                "openOverUnder": b.get("openOverUnder"),
                "closeOdds": b.get("closeOdds"),
                "closeSpread": b.get("closeSpread"),
                "closeOverUnder": b.get("closeOverUnder"),
                # audit columns
                "lastUpdatedAt": b.get("lastUpdatedAt"),
                "minsBeforeStart": (round((tip - updated).total_seconds() / 60)
                                    if tip and updated else None),
                "actual": odd.get("score"),
            }


def daterange(start, end, step_days=STEP_DAYS):
    d = dt.date.fromisoformat(start)
    stop = dt.date.fromisoformat(end)
    while d < stop:
        nxt = min(d + dt.timedelta(days=step_days), stop)
        yield d.isoformat(), nxt.isoformat()
        d = nxt


def check(day=CHECK_DAY):
    """One-day test: per book, how many markets have a populated close?"""
    nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
    tally, markets, sample = {}, set(), None
    for ev in fetch_events(day, nxt, books=None):   # no book filter
        for odd_id, odd in (ev.get("odds") or {}).items():
            markets.add(odd_id)
            for book, b in (odd.get("byBookmaker") or {}).items():
                t = tally.setdefault(book, {"with_close": 0, "no_close": 0})
                has_close = any(b.get(k) is not None
                                for k in ("closeOdds", "closeSpread", "closeOverUnder"))
                if has_close:
                    t["with_close"] += 1
                    sample = sample or (book, odd_id, b)
                else:
                    t["no_close"] += 1
    print(f"Close availability on {day}:")
    for book, t in sorted(tally.items(), key=lambda x: -x[1]["with_close"]):
        flag = "" if book in BOOKS else "   (not in your BOOKS list)"
        print(f"  {book}: with_close={t['with_close']}  no_close={t['no_close']}{flag}")
    missing = [b for b in BOOKS if b not in tally]
    if missing:
        print("No data that day for:", ", ".join(missing))
    print("\noddIDs returned:", ", ".join(sorted(markets)) or "(none)")
    if sample:
        print(f"\nSample byBookmaker entry ({sample[0]}, {sample[1]}):\n",
              json.dumps(sample[2], indent=2))


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