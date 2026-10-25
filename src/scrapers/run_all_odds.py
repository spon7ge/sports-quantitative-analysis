"""Run MLB and NBA odds/props scrapers.

Usage:
  python -m src.scrapers.run_all_odds
  python -m src.scrapers.run_all_odds --league nba
  python -m src.scrapers.run_all_odds --only nba_novig,nba_underdog
  python -m src.scrapers.run_all_odds --exclude nba_prizepick,mlb_prizepick
  python -m src.scrapers.run_all_odds --fail-fast

Concurrency (env):
  ODDS_HTTP_CONCURRENCY — Novig/ProphetX/Underdog/FanDuel/DraftKings/PrizePicks HTTP (default 8)
  ODDS_PINNACLE_CONCURRENCY — Pinnacle browser jobs (default 1)
  ODDS_PRIZEPICKS_BROWSER_CONCURRENCY — PrizePicks Playwright fallback (default 1)
  ODDS_DB_LOAD_SLOTS — simultaneous change-filter/upsert slots (default 2)

Database:
  JSON under data/props/ is always written. Supabase upserts are off unless
  ODDS_WRITE_DB=1 (then book-specific *_SKIP_DB still applies).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Literal

League = Literal["mlb", "nba"]
LEAGUES: tuple[League, ...] = ("mlb", "nba")
Pool = Literal["http", "pinnacle", "prizepick_browser"]

# PrizePicks HTTP path exhausted; runner may queue a browser fallback job.
PRIZEPICKS_NEED_BROWSER_EXIT = 3


@dataclass(frozen=True)
class ScraperJob:
    name: str
    league: League
    module: str
    env: dict[str, str] | None = None


def _league_jobs(league: League) -> tuple[ScraperJob, ...]:
    """One book order per league. PrizePicks stays last inside the league."""
    package = f"src.scrapers.{league}"
    jobs = [
        ScraperJob(f"{league}_novig", league, f"{package}.{league}_novig"),
        ScraperJob(f"{league}_prophetx", league, f"{package}.{league}_prophetx"),
        ScraperJob(f"{league}_underdog", league, f"{package}.{league}_underdog"),
        ScraperJob(f"{league}_fanduel", league, f"{package}.{league}_fanduel"),
        ScraperJob(f"{league}_draftking", league, f"{package}.{league}_draftking"),
        ScraperJob(
            f"{league}_pinnacle",
            league,
            f"{package}.{league}_pinnacle",
            env={"PINNACLE_LEAGUES": league},
        ),
    ]
    if league == "nba":
        jobs.append(
            ScraperJob(
                "nba_rotowire_lineup",
                "nba",
                "src.scrapers.nba.nba_rotowire_lineup",
            )
        )
    jobs.append(
        ScraperJob(
            f"{league}_prizepick",
            league,
            f"{package}.{league}_prizepick",
            # HTTP attempt first; browser fallback is queued separately on exit 3.
            env={"PRIZEPICKS_NO_BROWSER": "1"},
        )
    )
    return tuple(jobs)


# PrizePicks is last in each league, and resolve_jobs moves every PrizePicks
# job to the end of the selected list so a captcha does not block later HTTP scrapers.
SCRAPER_JOBS: tuple[ScraperJob, ...] = _league_jobs("mlb") + _league_jobs("nba")

KNOWN_NAMES = {job.name for job in SCRAPER_JOBS}


def _unknown_names(names: list[str]) -> list[str]:
    return [name for name in names if name not in KNOWN_NAMES]


def _prizepicks_last(jobs: list[ScraperJob]) -> list[ScraperJob]:
    rest = [job for job in jobs if not job.name.endswith("prizepick")]
    prize = [job for job in jobs if job.name.endswith("prizepick")]
    return rest + prize


def classify_pool(job: ScraperJob) -> Pool:
    if job.name.endswith("_pinnacle"):
        return "pinnacle"
    if (job.env or {}).get("PRIZEPICKS_BROWSER_ONLY", "").strip() in {
        "1",
        "true",
        "yes",
    }:
        return "prizepick_browser"
    return "http"


def resolve_jobs(
    *,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    league: str | None = None,
) -> list[ScraperJob]:
    """Filter jobs by league, then allowlist, then denylist. PrizePicks runs last."""
    if league is not None and league not in LEAGUES:
        raise ValueError(f"Unknown league {league!r}; choose from {LEAGUES}")
    jobs = list(SCRAPER_JOBS)
    if league:
        jobs = [job for job in jobs if job.league == league]
    if only:
        unknown = _unknown_names(only)
        if unknown:
            raise ValueError(
                f"Unknown scraper name(s): {unknown}; choose from {sorted(KNOWN_NAMES)}"
            )
        allow = set(only)
        jobs = [j for j in jobs if j.name in allow]
    if exclude:
        unknown = _unknown_names(exclude)
        if unknown:
            raise ValueError(
                f"Unknown scraper name(s): {unknown}; choose from {sorted(KNOWN_NAMES)}"
            )
        deny = set(exclude)
        jobs = [j for j in jobs if j.name not in deny]
    return _prizepicks_last(jobs)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 1)
    except ValueError:
        return default


def _prizepick_browser_job(job: ScraperJob) -> ScraperJob:
    env = dict(job.env or {})
    env.pop("PRIZEPICKS_NO_BROWSER", None)
    env["PRIZEPICKS_BROWSER_ONLY"] = "1"
    return replace(job, name=f"{job.name}_browser", env=env)


def run_job(
    job: ScraperJob,
    *,
    python: str | None = None,
    cancel_event: threading.Event | None = None,
    proc_registry: list[subprocess.Popen] | None = None,
    registry_lock: threading.Lock | None = None,
) -> int:
    """Run one scraper module as a subprocess; return its exit code."""
    cmd = [python or sys.executable, "-m", job.module]
    env = os.environ.copy()
    if job.env:
        env.update(job.env)
    print(f"\n=== {job.name} ({job.module}) ===", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    if proc_registry is not None and registry_lock is not None:
        with registry_lock:
            proc_registry.append(proc)
    try:
        while True:
            code = proc.poll()
            if code is not None:
                return int(code)
            if cancel_event is not None and cancel_event.is_set():
                _terminate_proc(proc)
                return int(proc.wait(timeout=30) or 1)
            time.sleep(0.1)
    finally:
        if proc_registry is not None and registry_lock is not None:
            with registry_lock:
                if proc in proc_registry:
                    proc_registry.remove(proc)


def _terminate_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def cancel_running(procs: list[subprocess.Popen], lock: threading.Lock) -> None:
    with lock:
        snapshot = list(procs)
    for proc in snapshot:
        _terminate_proc(proc)


def _pool_limit(pool: Pool) -> int:
    if pool == "http":
        return _env_int("ODDS_HTTP_CONCURRENCY", 8)
    if pool == "pinnacle":
        return _env_int("ODDS_PINNACLE_CONCURRENCY", 1)
    return _env_int("ODDS_PRIZEPICKS_BROWSER_CONCURRENCY", 1)


def run_all(
    *,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    league: str | None = None,
    fail_fast: bool = False,
    python: str | None = None,
) -> int:
    """Run selected scrapers concurrently by pool. Returns 0 iff all succeeded."""
    jobs = resolve_jobs(only=only, exclude=exclude, league=league)
    if not jobs:
        print("No scrapers selected.", flush=True)
        return 1

    cancel_event = threading.Event()
    proc_registry: list[subprocess.Popen] = []
    registry_lock = threading.Lock()
    results: dict[str, int] = {}
    results_lock = threading.Lock()
    browser_queue: list[ScraperJob] = []
    browser_lock = threading.Lock()

    def _record(name: str, code: int) -> None:
        with results_lock:
            results[name] = code

    def _run(job: ScraperJob) -> int:
        if cancel_event.is_set():
            _record(job.name, 1)
            return 1
        code = run_job(
            job,
            python=python,
            cancel_event=cancel_event,
            proc_registry=proc_registry,
            registry_lock=registry_lock,
        )
        _record(job.name, code)
        if (
            job.name.endswith("prizepick")
            and (job.env or {}).get("PRIZEPICKS_NO_BROWSER") == "1"
            and code == PRIZEPICKS_NEED_BROWSER_EXIT
        ):
            with browser_lock:
                browser_queue.append(_prizepick_browser_job(job))
            print(
                f"{job.name}: HTTP blocked — queued browser fallback",
                flush=True,
            )
            # HTTP miss is expected when browser is needed; do not fail the run
            # solely for exit 3 (browser job owns the final status).
            _record(job.name, 0)
            code = 0
        if code != 0 and fail_fast:
            print(f"Stopping after failure: {job.name} exited {code}", flush=True)
            cancel_event.set()
            cancel_running(proc_registry, registry_lock)
        return code

    by_pool: dict[Pool, list[ScraperJob]] = {
        "http": [],
        "pinnacle": [],
        "prizepick_browser": [],
    }
    for job in jobs:
        by_pool[classify_pool(job)].append(job)

    # Overlap HTTP downloads + Pinnacle (own browser limit). PrizePicks browser
    # fallback runs after HTTP pool so captcha work does not starve HTTP books.
    with ThreadPoolExecutor(max_workers=_pool_limit("http")) as http_ex, ThreadPoolExecutor(
        max_workers=_pool_limit("pinnacle")
    ) as pin_ex:
        futures: list[Future[int]] = []
        for job in by_pool["http"]:
            futures.append(http_ex.submit(_run, job))
        for job in by_pool["pinnacle"]:
            futures.append(pin_ex.submit(_run, job))
        wait(futures)

    with browser_lock:
        queued = list(browser_queue)
    if queued and not (fail_fast and cancel_event.is_set()):
        with ThreadPoolExecutor(
            max_workers=_pool_limit("prizepick_browser")
        ) as browser_ex:
            list(browser_ex.map(_run, queued))

    # Preserve a stable summary order: resolved jobs, then any browser fallbacks.
    summary_names = [j.name for j in jobs]
    for name in results:
        if name not in summary_names:
            summary_names.append(name)

    print("\n=== Summary ===", flush=True)
    print(
        "(Per-book SCRAPE_REPORT lines include json_ok/db_ok and stage timings.)",
        flush=True,
    )
    failed = 0
    for name in summary_names:
        if name not in results:
            continue
        code = results[name]
        status = "ok" if code == 0 else f"FAIL ({code})"
        print(f"  {name}: {status}", flush=True)
        if code != 0:
            failed += 1
    succeeded = sum(1 for n in summary_names if results.get(n) == 0)
    total = sum(1 for n in summary_names if n in results)
    print(f"{succeeded}/{total} succeeded", flush=True)
    return 0 if failed == 0 else 1


def _parse_only(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    names = [part.strip() for part in raw.split(",") if part.strip()]
    return names or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run MLB and NBA odds/props scrapers.",
    )
    parser.add_argument(
        "--league",
        choices=LEAGUES,
        default=None,
        help="Run one league. Default runs MLB and NBA.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=f"Comma-separated scraper names. Known: {', '.join(sorted(KNOWN_NAMES))}",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated scraper names to skip (e.g. PrizePicks in Docker)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Cancel running scrapers after the first failure (default: continue)",
    )
    args = parser.parse_args(argv)
    try:
        return run_all(
            only=_parse_only(args.only),
            exclude=_parse_only(args.exclude),
            league=args.league,
            fail_fast=args.fail_fast,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
