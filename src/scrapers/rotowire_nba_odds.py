"""Scrape Rotowire NBA betting data for one or more seasons.

Rotowire identifies a season by its starting year:
"2025" represents the 2025-26 NBA season.

Examples:
    python -m src.scrapers.rotowire_nba_odds --season 2025

    python -m src.scrapers.rotowire_nba_odds \
        --seasons 2021 2023 2025

    python -m src.scrapers.rotowire_nba_odds \
        --seasons 2019-2025
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PAGE_URL = "https://www.rotowire.com/betting/nba/archive.php"
API_URL = (
    "https://www.rotowire.com/betting/nba/"
    "tables/games-archive.php"
)

DEFAULT_SEASON = "2025"

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "bronze"
    / "rotowire"
)

FIELD_NAMES = (
    "Game",
    "Tipoff",
    "Season",
    "Score",
    "Over_Under",
    "Home_Line",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

async def fetch_archive(
    *,
    headless: bool = True,
) -> list[dict[str, Any]]:
    """Fetch the complete Rotowire NBA game archive."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as error:
        raise RuntimeError(
            "Playwright is required. Run: "
            "pip install playwright && playwright install chromium"
        ) from error

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=headless
        )

        try:
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1440, "height": 900},
            )
            page = await context.new_page()

            logger.info("Loading %s", PAGE_URL)
            await page.goto(
                PAGE_URL,
                wait_until="networkidle",
                timeout=60_000,
            )

            logger.info("Fetching %s", API_URL)
            response_text = await page.evaluate(
                """
                async (apiUrl) => {
                    const response = await fetch(apiUrl, {
                        credentials: "include",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest"
                        }
                    });

                    if (!response.ok) {
                        throw new Error(
                            `Rotowire API returned ${response.status}`
                        );
                    }

                    return await response.text();
                }
                """,
                API_URL,
            )
        finally:
            await browser.close()

    records = json.loads(response_text)

    if not isinstance(records, list):
        raise RuntimeError(
            "Rotowire returned an unexpected response format."
        )

    logger.info(
        "Received %d total records from Rotowire",
        len(records),
    )
    return records

def build_rows(
    records: list[dict[str, Any]],
    season: str | int,
) -> list[dict[str, Any]]:
    """Convert archive records for one season into output rows."""
    season = str(season)
    rows: list[dict[str, Any]] = []

    for record in records:
        if str(record.get("season", "")) != season:
            continue

        away = record.get("visit_team_abbrev", "")
        home = record.get("home_team_abbrev", "")

        rows.append(
            {
                "Game": f"{away} @ {home}",
                "Tipoff": record.get("tipoff", ""),
                "Season": record.get("season", ""),
                "Score": record.get("score", ""),
                "Over_Under": record.get(
                    "game_over_under",
                    "",
                ),
                "Home_Line": record.get("line", ""),
            }
        )

    return rows

def save_rows(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Save Rotowire rows to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELD_NAMES,
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Saved %d rows to %s",
        len(rows),
        output_path,
    )
    return output_path

async def run_scrape(
    *,
    season: str | None = None,
    output_file: str | Path | None = None,
    headless: bool | None = None,
) -> Path:
    """Scrape one season.

    This function remains available for the silver pipeline's
    automatic Rotowire scraping.
    """
    selected_season = str(season or DEFAULT_SEASON)
    use_headless = True if headless is None else headless

    output_path = (
        Path(output_file)
        if output_file is not None
        else DEFAULT_OUTPUT_DIR
        / f"rotowire_nba_{selected_season}.csv"
    )

    records = await fetch_archive(headless=use_headless)
    rows = build_rows(records, selected_season)

    if not rows:
        logger.warning(
            "No Rotowire games found for season %s",
            selected_season,
        )

    return save_rows(rows, output_path)

async def run_scrapes(
    seasons: list[str],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    headless: bool = True,
) -> dict[str, Path]:
    """Fetch the archive once and save one CSV per season."""
    records = await fetch_archive(headless=headless)
    output_dir = Path(output_dir)
    saved_files: dict[str, Path] = {}

    for season in seasons:
        rows = build_rows(records, season)

        if not rows:
            logger.warning(
                "No Rotowire games found for season %s",
                season,
            )

        output_path = (
            output_dir / f"rotowire_nba_{season}.csv"
        )
        saved_files[season] = save_rows(rows, output_path)

    return saved_files

def parse_seasons(values: list[str]) -> list[str]:
    """Parse individual seasons and inclusive ranges."""
    seasons: set[int] = set()

    for value in values:
        value = value.strip()

        if "-" not in value:
            seasons.add(validate_season(value))
            continue

        start_text, end_text = value.split("-", maxsplit=1)
        start_year = validate_season(start_text)
        end_year = validate_season(end_text)

        if start_year > end_year:
            raise ValueError(
                f"Invalid season range {value!r}: "
                "the start must not exceed the end"
            )

        seasons.update(range(start_year, end_year + 1))

    return [str(season) for season in sorted(seasons)]

def validate_season(value: str) -> int:
    try:
        season = int(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid season: {value!r}"
        ) from error

    if season < 1946 or season > 2100:
        raise ValueError(
            f"Season out of expected range: {season}"
        )

    return season

def parse_cli_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Rotowire NBA betting data for "
            "one or more seasons"
        )
    )

    parser.add_argument(
        "--seasons",
        "--season",
        nargs="+",
        default=[DEFAULT_SEASON],
        metavar="YEAR",
        help=(
            "Season starting years or ranges, such as "
            "'2025', '2021 2023', or '2019-2025'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for season CSV files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Custom output path; only valid for one season",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Display the browser while scraping",
    )

    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_cli_args(argv)

    try:
        seasons = parse_seasons(args.seasons)
    except ValueError as error:
        logger.error("%s", error)
        return 2

    if args.output is not None and len(seasons) != 1:
        logger.error(
            "--output can only be used when scraping one season"
        )
        return 2

    if args.output is not None:
        asyncio.run(
            run_scrape(
                season=seasons[0],
                output_file=args.output,
                headless=not args.headed,
            )
        )
    else:
        asyncio.run(
            run_scrapes(
                seasons,
                output_dir=args.output_dir,
                headless=not args.headed,
            )
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())


