"""Fetch NBA play-by-play for every game in a silver season.

Reads unique ``game_id`` values from
``data/silver/<league>/<season>/*/player_gamelogs.parquet`` and writes
PlayByPlayV3 rows to ``data/PlayByPlay/s{end}_pbp.parquet``. A rerun
skips game IDs already present in that file.

Run from the repo root:

    python -m src.pipeline.bronze.play_by_play --season 2026-27
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

from .retry import call_with_retry

logger = logging.getLogger(__name__)

_SEASON = re.compile(r"^(?P<start>\d{4})-(?P<end>\d{2})$")
_GAME_ID = re.compile(r"^\d{1,10}$")

PlayByPlayRequest = Callable[[str], pd.DataFrame]


@dataclass(frozen=True)
class PlayByPlayFetchResult:
    path: Path
    fetched: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[str, ...]


def validate_season(season: str) -> str:
    match = _SEASON.fullmatch(season.strip())
    if match is None:
        raise ValueError(f"Season must look like 2026-27, got {season!r}")

    start = int(match.group("start"))
    end = int(match.group("end"))
    if end != (start + 1) % 100:
        raise ValueError(f"Season end year must follow the start year, got {season!r}")

    return f"{start}-{end:02d}"


def normalize_game_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]

    if _GAME_ID.fullmatch(text) is None:
        raise ValueError(f"Game ID must be numeric, got {value!r}")

    return text.zfill(10)


def play_by_play_path(output_dir: str | Path, season: str) -> Path:
    end_year = validate_season(season).split("-")[1]
    return Path(output_dir) / f"s{end_year}_pbp.parquet"


def silver_game_ids(
    silver_dir: str | Path,
    season: str,
    *,
    league: str = "nba",
) -> list[str]:
    """Unique zero-padded game IDs from every season type in one season."""
    label = validate_season(season)
    root = Path(silver_dir) / league / label
    paths = sorted(root.glob("*/player_gamelogs.parquet"))

    if not paths:
        raise FileNotFoundError(f"No player_gamelogs.parquet under {root}")

    game_ids: list[str] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["game_id"])
        if "game_id" not in frame.columns:
            raise ValueError(f"{path} has no game_id column")

        game_ids.extend(
            normalize_game_id(value) for value in frame["game_id"].dropna().tolist()
        )

    if not game_ids:
        raise ValueError(f"No game IDs in {root}")

    return sorted(set(game_ids))


def completed_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    frame = pd.read_parquet(path, columns=["gameId"])
    if "gameId" not in frame.columns:
        raise ValueError(f"{path} has no gameId column")

    return {
        normalize_game_id(value) for value in frame["gameId"].dropna().unique().tolist()
    }


def request_play_by_play(game_id: str) -> pd.DataFrame:
    normalized = normalize_game_id(game_id)

    def request() -> pd.DataFrame:
        endpoint = playbyplayv3.PlayByPlayV3(
            game_id=normalized,
            timeout=60,
        )
        return endpoint.play_by_play.get_data_frame()

    return _checked_frame(
        normalized,
        call_with_retry(request, label=f"play-by-play {normalized}"),
    )


def fetch_season_play_by_play(
    season: str = "2026-27",
    *,
    silver_dir: str | Path = "data/silver",
    output_dir: str | Path = "data/PlayByPlay",
    league: str = "nba",
    delay_seconds: float = 1.0,
    request: PlayByPlayRequest | None = None,
) -> PlayByPlayFetchResult:
    """Fetch missing play-by-play and checkpoint after each successful game.

    stats.nba.com throttles bursts, and a season is hundreds of games.
    Requests run one at a time so an interrupted run can resume from the
    parquet already on disk.
    """
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be >= 0")

    game_ids = silver_game_ids(silver_dir, season, league=league)
    path = play_by_play_path(output_dir, season)
    already_saved = completed_game_ids(path)
    skipped = tuple(game_id for game_id in game_ids if game_id in already_saved)
    pending = [game_id for game_id in game_ids if game_id not in already_saved]
    fetcher = request if request is not None else request_play_by_play

    logger.info(
        "Play-by-play for %s: %d games, %d already saved, %d remaining",
        validate_season(season),
        len(game_ids),
        len(skipped),
        len(pending),
    )

    fetched: list[str] = []
    failed: list[str] = []

    for index, game_id in enumerate(pending, start=1):
        if delay_seconds:
            time.sleep(delay_seconds)

        try:
            frame = _checked_frame(game_id, fetcher(game_id))
        except Exception as error:
            logger.warning(
                "Play-by-play failed for game %s (%d/%d): %s",
                game_id,
                index,
                len(pending),
                error,
            )
            failed.append(game_id)
            continue

        _append_game(path, frame)
        fetched.append(game_id)
        logger.info(
            "Saved play-by-play for game %s (%d/%d, %d rows)",
            game_id,
            index,
            len(pending),
            len(frame),
        )

    if failed:
        logger.error(
            "Play-by-play missing for %d games: %s",
            len(failed),
            failed,
        )

    return PlayByPlayFetchResult(
        path=path,
        fetched=tuple(fetched),
        skipped=skipped,
        failed=tuple(failed),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args(argv)
    result = fetch_season_play_by_play(
        args.season,
        silver_dir=args.silver_dir,
        output_dir=args.output_dir,
        league=args.league,
        delay_seconds=args.delay,
    )
    return 1 if result.failed else 0


def _checked_frame(game_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"No play-by-play rows for game {game_id}")

    if "gameId" not in frame.columns:
        raise ValueError(f"Play-by-play for game {game_id} has no gameId column")

    result = frame.copy()
    result["gameId"] = result["gameId"].map(normalize_game_id)
    mismatched = sorted(set(result["gameId"]) - {game_id})
    if mismatched:
        raise ValueError(f"Play-by-play for game {game_id} returned {mismatched}")

    return result


def _append_game(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        combined = pd.concat(
            [pd.read_parquet(path), frame],
            ignore_index=True,
        )
    else:
        combined = frame

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Fetch NBA play-by-play for every game id in a silver season")
    )
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--league", default="nba")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--output-dir", default="data/PlayByPlay")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait before each request",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
