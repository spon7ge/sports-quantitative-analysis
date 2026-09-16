"""
Scrape WNBA player positions for one or more seasons.

Examples:
    python -m src.scrapers.basketball_reference_wnba_positions \
        --years 2021 2022 2024

    python -m src.scrapers.basketball_reference_wnba_positions \
        --years 2021-2025
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

from bs4 import BeautifulSoup, Comment, Tag
from curl_cffi import requests

logger = logging.getLogger(__name__)

BASE_URL = (
    "https://www.basketball-reference.com/wnba/years/"
    "{year}_per_game.html"
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "bronze"
    / "player_positions"
)

TABLE_IDS = ("per_game", "per_game_stats")

def scrape_players(
    year: int,
    *,
    timeout: int = 20,
) -> list[dict[str, str]]:
    """Scrape player names, positions, and ages for one WNBA season."""
    url = BASE_URL.format(year=year)
    logger.info("Fetching %s", url)

    response = requests.get(
        url,
        impersonate="chrome",
        timeout=timeout,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = find_table(soup)

    if table is None:
        raise RuntimeError(
            f"Could not find the WNBA per-game table for {year}. "
            f"Tried table IDs: {', '.join(TABLE_IDS)}"
        )

    table_body = table.find("tbody")

    if table_body is None:
        raise RuntimeError(
            f"The WNBA per-game table for {year} has no tbody."
        )

    players: list[dict[str, str]] = []
    seen_names: set[str] = set()

    for row in table_body.find_all("tr"):
        if "thead" in (row.get("class") or []):
            continue

        name = (
            cell_text(row, "player")
            or cell_text(row, "name_display")
        )

        if not name or name in seen_names:
            continue

        seen_names.add(name)

        players.append(
            {
                "name": name,
                "pos": cell_text(row, "pos") or "",
                "age": cell_text(row, "age") or "",
            }
        )

    if not players:
        raise RuntimeError(
            f"No WNBA players were found for season {year}."
        )

    logger.info(
        "Found %d players for WNBA season %d",
        len(players),
        year,
    )
    return players

def scrape_seasons(
    years: list[int],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    request_delay: float = 3.0,
    timeout: int = 20,
) -> dict[int, Path]:
    """Scrape multiple WNBA seasons and save one CSV per year."""
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
                output_dir / f"wnba_{year}_players.csv"
            )
            save_csv(players, output_path)
            saved_files[year] = output_path
        except Exception:
            logger.exception(
                "Unable to scrape WNBA season %d",
                year,
            )

    return saved_files

def find_table(
    soup: BeautifulSoup,
    table_ids: tuple[str, ...] = TABLE_IDS,
) -> Tag | None:
    """Find the stats table, including tables inside HTML comments."""
    for table_id in table_ids:
        table = soup.find("table", id=table_id)

        if isinstance(table, Tag):
            return table

    comments = soup.find_all(
        string=lambda text: isinstance(text, Comment)
    )

    for comment in comments:
        if not any(table_id in comment for table_id in table_ids):
            continue

        comment_soup = BeautifulSoup(comment, "html.parser")

        for table_id in table_ids:
            table = comment_soup.find("table", id=table_id)

            if isinstance(table, Tag):
                return table

    return None

def cell_text(row: Tag, data_stat: str) -> str | None:
    """Extract text from a table cell identified by data-stat."""
    cell = row.find(
        ["td", "th"],
        attrs={"data-stat": data_stat},
    )

    if cell is None:
        return None

    link = cell.find("a")
    text = (
        link.get_text(strip=True)
        if link is not None
        else cell.get_text(strip=True)
    )

    return text or None

def save_csv(
    players: list[dict[str, str]],
    path: str | Path,
) -> Path:
    """Save one WNBA season to CSV."""
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
    """Parse individual years and inclusive ranges."""
    years: set[int] = set()

    for value in values:
        value = value.strip()

        if "-" not in value:
            years.add(validate_year(value))
            continue

        start_text, end_text = value.split("-", maxsplit=1)
        start_year = validate_year(start_text)
        end_year = validate_year(end_text)

        if start_year > end_year:
            raise ValueError(
                f"Invalid range {value!r}: start exceeds end"
            )

        years.update(range(start_year, end_year + 1))

    return sorted(years)

def validate_year(value: str) -> int:
    try:
        year = int(value)
    except ValueError as error:
        raise ValueError(f"Invalid year: {value!r}") from error

    if year < 1997 or year > 2100:
        raise ValueError(
            f"WNBA year out of expected range: {year}"
        )

    return year

def parse_cli_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape WNBA player positions for one or more seasons"
        )
    )
    parser.add_argument(
        "--years",
        nargs="+",
        required=True,
        metavar="YEAR",
        help=(
            "Individual years or ranges, such as "
            "'2021 2023' or '2021-2025'"
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
        default=20,
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