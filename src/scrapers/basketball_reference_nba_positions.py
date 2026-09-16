"""
Scrape NBA player positions from Basketball Reference.

Basketball Reference identifies an NBA season by its ending year:
NBA_2026 represents the 2025-26 season.

Examples:
    python -m src.scrapers.basketball_reference_nba_positions --years 2021 2022 2023

    python -m src.scrapers.basketball_reference_nba_positions --years 2021-2026

    python -m src.scrapers.basketball_reference_nba_positions \
        --years 2021 2023-2025 \
        --output-dir data/bronze/player_positions
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

from bs4 import BeautifulSoup
from curl_cffi import requests

logger = logging.getLogger(__name__)

URL_TEMPLATE = (
    "https://www.basketball-reference.com/leagues/"
    "NBA_{year}_per_game.html"
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "bronze"
    / "player_positions"
)

def scrape_players(
    year: int,
    *,
    timeout: int = 15,
) -> list[dict[str, str]]:
    """Scrape player names, positions, and ages for one NBA season."""
    url = URL_TEMPLATE.format(year=year)
    logger.info("Fetching %s", url)

    response = requests.get(
        url,
        impersonate="chrome",
        timeout=timeout,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", id="per_game_stats")

    if table is None:
        raise RuntimeError(
            f"Could not find the player stats table for {year}. "
            "The page structure may have changed."
        )

    table_body = table.find("tbody")

    if table_body is None:
        raise RuntimeError(
            f"The player stats table for {year} has no tbody."
        )

    players: list[dict[str, str]] = []
    seen_names: set[str] = set()

    for row in table_body.find_all("tr"):
        if "thead" in (row.get("class") or []):
            continue

        name_cell = row.find("td", {"data-stat": "name_display"})

        if name_cell is None:
            continue

        name = name_cell.get_text(strip=True)

        if not name or name in seen_names:
            continue

        seen_names.add(name)

        players.append(
            {
                "name": name,
                "pos": _cell_text(row, "pos"),
                "age": _cell_text(row, "age"),
            }
        )

    if not players:
        raise RuntimeError(f"No players were found for season {year}.")

    logger.info("Found %d players for NBA season %d", len(players), year)
    return players

def scrape_seasons(
    years: list[int],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    request_delay: float = 3.0,
    timeout: int = 15,
) -> dict[int, Path]:
    """Scrape several seasons and save one CSV per season."""
    if request_delay < 0:
        raise ValueError("request_delay cannot be negative")

    output_dir = Path(output_dir)
    saved_files: dict[int, Path] = {}

    for index, year in enumerate(years):
        if index:
            logger.info(
                "Waiting %.1f seconds before the next request",
                request_delay,
            )
            time.sleep(request_delay)

        try:
            players = scrape_players(year, timeout=timeout)
            output_path = (
                output_dir / f"nba_{year}_players.csv"
            )
            save_csv(players, output_path)
            saved_files[year] = output_path
        except Exception:
            logger.exception(
                "Unable to scrape NBA season ending in %d",
                year,
            )

    return saved_files

def save_csv(
    players: list[dict[str, str]],
    path: str | Path,
) -> Path:
    """Save a season's player positions to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["name", "pos", "age"],
        )
        writer.writeheader()
        writer.writerows(players)

    logger.info(
        "Saved %d players to %s",
        len(players),
        output_path,
    )
    return output_path

def parse_years(values: list[str]) -> list[int]:
    """Parse individual years and inclusive ranges.

    Examples:
        ["2021", "2023"] -> [2021, 2023]
        ["2021-2025"] -> [2021, 2022, 2023, 2024, 2025]
    """
    years: set[int] = set()

    for value in values:
        value = value.strip()

        if "-" not in value:
            years.add(_validate_year(value))
            continue

        start_text, end_text = value.split("-", maxsplit=1)
        start_year = _validate_year(start_text)
        end_year = _validate_year(end_text)

        if start_year > end_year:
            raise ValueError(
                f"Invalid year range {value!r}: "
                "the start year must not exceed the end year"
            )

        years.update(range(start_year, end_year + 1))

    return sorted(years)

def _validate_year(value: str) -> int:
    try:
        year = int(value)
    except ValueError as error:
        raise ValueError(f"Invalid year: {value!r}") from error

    if year < 1950 or year > 2100:
        raise ValueError(f"Year out of expected range: {year}")

    return year

def _cell_text(row, data_stat: str) -> str:
    cell = row.find("td", {"data-stat": data_stat})
    return cell.get_text(strip=True) if cell else ""

def parse_cli_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape NBA player positions for one or more seasons"
        )
    )
    parser.add_argument(
        "--years",
        nargs="+",
        required=True,
        metavar="YEAR",
        help=(
            "Season ending years or ranges, such as "
            "'2021 2022' or '2021-2026'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=3.0,
        help="Seconds to wait between requests",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_cli_args(argv)

    try:
        years = parse_years(args.years)
    except ValueError as error:
        logger.error("%s", error)
        return 2

    saved_files = scrape_seasons(
        years,
        output_dir=args.output_dir,
        request_delay=args.request_delay,
        timeout=args.timeout,
    )

    logger.info(
        "Completed %d of %d seasons",
        len(saved_files),
        len(years),
    )

    return 0 if len(saved_files) == len(years) else 1

if __name__ == "__main__":
    raise SystemExit(main())