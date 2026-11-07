"""Train the Layer 3 play-by-play extension and save reconstruction audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


def load(path):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            dtype={
                "gameId": "string",
                "GAME_ID": "string",
                "game_id": "string",
            },
        )

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError("input must be CSV or parquet")


def main():
    # Running this file directly puts the package directory on sys.path.
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from game_rotations.pbp import fit_pbp_rotation, reconstruct_rotation_segments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbp", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--games", help="Optional game_id/home_team_id mapping")
    parser.add_argument("--output", default="outputs/pbp_rotation")
    parser.add_argument("--minute-tolerance", type=float, default=1.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = reconstruct_rotation_segments(
        load(args.pbp),
        load(args.logs),
        games=load(args.games) if args.games else None,
        minute_tolerance=args.minute_tolerance,
    )
    result.comparisons.to_csv(output / "minute_comparisons.csv", index=False)
    result.issues.to_csv(output / "reconstruction_issues.csv", index=False)
    segments = result.segments.copy()

    for column in ("players", "signature"):
        if column in segments:
            segments[column] = segments[column].map(json.dumps)

    segments.to_csv(output / "historical_segments.csv", index=False)
    model = fit_pbp_rotation(result, holdout_fraction=args.holdout_fraction)
    joblib.dump(model, output / "pbp_rotation_model.joblib")
    metadata = {
        "version": model.version,
        "training_cutoff": model.training_cutoff,
        **model.diagnostics,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
