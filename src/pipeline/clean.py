"""Compatibility facade and CLI for the silver pipeline."""

from __future__ import annotations

import argparse
import logging

from src.pipeline.bronze.config import LEAGUES
from src.pipeline.silver import (
    build_silver,
    enrich_positions,
    enrich_rotowire,
    fetch_and_build_silver,
    load_player_positions,
    load_rotowire,
    merge_gamelogs,
)

__all__ = [
    "build_silver",
    "enrich_positions",
    "enrich_rotowire",
    "fetch_and_build_silver",
    "load_player_positions",
    "load_rotowire",
    "merge_gamelogs",
]

def parse_cli_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build silver player game-log Parquet files"
    )

    parser.add_argument(
        "--league",
        choices=tuple(LEAGUES),
        default="nba",
    )
    parser.add_argument("--season")
    parser.add_argument(
        "--season-type",
        default="Regular Season",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/bronze",
    )
    parser.add_argument(
        "--silver-dir",
        default="data/silver",
    )
    parser.add_argument(
        "--positions-dir",
        default="data/bronze/player_positions",
    )
    parser.add_argument(
        "--rotowire-dir",
        default="data/bronze/rotowire",
    )
    parser.add_argument(
        "--auto-scrape-rotowire",
        action="store_true",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--start-position-delay",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--start-position-workers",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
    )

    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_cli_args(argv)
    season = (
        args.season
        or LEAGUES[args.league].default_season
    )

    common_arguments = {
        "league": args.league,
        "raw_dir": args.raw_dir,
        "silver_dir": args.silver_dir,
        "positions_dir": args.positions_dir,
        "rotowire_dir": args.rotowire_dir,
        "auto_scrape_rotowire": args.auto_scrape_rotowire,
    }

    if args.fetch:
        fetch_and_build_silver(
            season,
            args.season_type,
            **common_arguments,
            parallel=not args.sequential,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            start_position_delay=args.start_position_delay,
            start_position_workers=args.start_position_workers,
            run_all_batches=not args.one_batch,
        )
    else:
        build_silver(
            season,
            args.season_type,
            **common_arguments,
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())