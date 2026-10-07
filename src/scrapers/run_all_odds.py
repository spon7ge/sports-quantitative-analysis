"""Run MLB and NBA odds/props scrapers.

Usage:
  python -m src.scrapers.run_all_odds
  python -m src.scrapers.run_all_odds --league nba
  python -m src.scrapers.run_all_odds --only nba_novig,nba_underdog
  python -m src.scrapers.run_all_odds --exclude nba_prizepick,mlb_prizepick
  python -m src.scrapers.run_all_odds --fail-fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

League = Literal["mlb", "nba"]
LEAGUES: tuple[League, ...] = ("mlb", "nba")


@dataclass(frozen=True)
class ScraperJob:
    name: str
    league: League
    module: str
    env: dict[str, str] | None = None


def _league_jobs(league: League) -> tuple[ScraperJob, ...]:
    """One book order per league. PrizePicks stays last inside the league."""
    package = f"src.scrapers.{league}"
    return (
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
        ScraperJob(f"{league}_prizepick", league, f"{package}.{league}_prizepick"),
    )


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


def run_job(job: ScraperJob, *, python: str | None = None) -> int:
    """Run one scraper module as a subprocess; return its exit code."""
    cmd = [python or sys.executable, "-m", job.module]
    env = os.environ.copy()
    if job.env:
        env.update(job.env)
    print(f"\n=== {job.name} ({job.module}) ===", flush=True)
    completed = subprocess.run(cmd, env=env, check=False)
    return int(completed.returncode)


def run_all(
    *,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    league: str | None = None,
    fail_fast: bool = False,
    python: str | None = None,
) -> int:
    """Run selected scrapers sequentially. Returns 0 iff all succeeded."""
    jobs = resolve_jobs(only=only, exclude=exclude, league=league)
    if not jobs:
        print("No scrapers selected.", flush=True)
        return 1

    results: list[tuple[str, int]] = []
    for job in jobs:
        code = run_job(job, python=python)
        results.append((job.name, code))
        if code != 0 and fail_fast:
            print(f"Stopping after failure: {job.name} exited {code}", flush=True)
            break

    print("\n=== Summary ===", flush=True)
    failed = 0
    for name, code in results:
        status = "ok" if code == 0 else f"FAIL ({code})"
        print(f"  {name}: {status}", flush=True)
        if code != 0:
            failed += 1
    print(f"{len(results) - failed}/{len(results)} succeeded", flush=True)
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
        help="Stop after the first scraper failure (default: continue)",
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
