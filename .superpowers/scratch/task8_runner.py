import json
import sys
from pathlib import Path

ANALYSIS_SRC = Path("src").resolve()
if not (ANALYSIS_SRC / "line_movement").is_dir():
    ANALYSIS_SRC = Path.cwd() / "notebooks/nba/line_movement_analysis/src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from line_movement import (
    DEFAULT_FOLLOW_WINDOW,
    build_episodes,
    build_follows,
    build_market_groups,
    detect_movements,
    filter_schedule,
    load_all_books,
    normalize_quotes,
    pairwise_stats,
    save_daily_report,
    validate_quotes,
)
from line_movement.match import attach_market_keys
from line_movement.paths import analysis_root, repo_root

DATE = "2026-10-24"
EXPECTED_ARTIFACTS = {
    "coverage_by_book.png",
    "episodes.csv",
    "follows.csv",
    "market_groups.csv",
    "movements.csv",
    "movements_over_time.png",
    "pairwise_lag_heatmap.png",
    "pairwise_stats.csv",
    "quality_report.json",
    "quotes_normalized.csv",
}


def main() -> None:
    root = repo_root()
    raw = load_all_books(DATE, root=root)
    quotes = normalize_quotes(raw)
    quotes, dropped = filter_schedule(quotes, date=DATE)
    quality = validate_quotes(quotes)
    quality.dropped_by_schedule = dropped

    quotes = attach_market_keys(quotes)
    groups = build_market_groups(quotes)
    movements = detect_movements(quotes)
    episodes = build_episodes(movements, window=DEFAULT_FOLLOW_WINDOW)
    follows = build_follows(
        movements,
        episodes,
        window=DEFAULT_FOLLOW_WINDOW,
    )
    pairwise = pairwise_stats(follows, episodes)
    out = save_daily_report(
        date=DATE,
        quotes=quotes,
        groups=groups,
        movements=movements,
        episodes=episodes,
        follows=follows,
        pairwise=pairwise,
        quality=quality,
        analysis_root=analysis_root(),
    )

    artifacts = {path.name for path in out.iterdir() if path.is_file()}
    missing = sorted(EXPECTED_ARTIFACTS - artifacts)
    if missing:
        raise RuntimeError(f"missing report artifacts: {', '.join(missing)}")
    if movements.empty:
        raise RuntimeError("movement log is empty")
    if pairwise.empty:
        raise RuntimeError("pairwise table is empty")

    print(
        json.dumps(
            {
                "artifacts": sorted(artifacts),
                "counts": {
                    "raw": len(raw),
                    "quotes": len(quotes),
                    "groups": len(groups),
                    "movements": len(movements),
                    "episodes": len(episodes),
                    "follows": len(follows),
                    "pairwise": len(pairwise),
                    "dropped_by_schedule": dropped,
                },
                "prizepicks_missing_side_rate": quality.missing_rates_by_book[
                    "prizepicks"
                ]["side"],
                "report_dir": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
