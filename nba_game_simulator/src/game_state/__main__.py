from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    # Running this file directly puts the package dir on sys.path.
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import joblib

    from game_state.constants import PRE, RECENCY_DECAY, REG_SECONDS, STATE
    from game_state.evaluate import evaluate_pipeline
    from game_state.features import add_pregame_features, build_player_frame
    from game_state.load import load_pbp, load_player_logs, read_files
    from game_state.tables import build_tables, starters_on_floor_table
    from game_state.train import new_model, script_library, train_models

    ap = argparse.ArgumentParser(
        description="Layer 2 conditional roster minutes"
    )
    ap.add_argument(
        "--pbp", required=True, help="PBP parquet/CSV glob"
    )
    ap.add_argument(
        "--player-logs",
        required=True,
        help="Official player-log parquet/CSV glob",
    )
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument(
        "--test-season",
        type=int,
        help="Season starting year; default is final 25%% of dates",
    )
    ap.add_argument(
        "--positive-favorite-logs", action="store_true"
    )
    ap.add_argument(
        "--layer1-draws",
        help=(
            "OOS historical draws: gameId, team_margin, "
            "is_home, n_ot, game_minutes"
        ),
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logs = load_player_logs(args.player_logs)
    if args.positive_favorite_logs:
        logs["spread"] *= -1

    states, labels, games = build_tables(
        load_pbp(args.pbp), logs
    )
    games = add_pregame_features(games)

    states = states.merge(
        games[["gameId", "prior_pace"]],
        on="gameId",
        validate="many_to_one",
    )
    states["poss_rem"] = (
        states.sec_rem / REG_SECONDS * states.prior_pace
    )

    table = starters_on_floor_table(states, games)
    d = build_player_frame(labels, games)
    bundle, test_games, _ = train_models(
        d, games, args.test_season
    )

    if args.layer1_draws:
        predictions = evaluate_pipeline(
            bundle,
            logs,
            test_games,
            read_files(args.layer1_draws),
        )
        predictions.to_csv(
            out / "minutes_backtest.csv", index=False
        )
    else:
        print(
            "Provide --layer1-draws to evaluate actual "
            "pregame minute distributions."
        )

    final_model = new_model().fit(
        d[PRE + STATE],
        d.minutes,
        sample_weight=RECENCY_DECAY ** (
            d.season.max() - d.season
        ),
    )
    bundle.update(
        model=final_model,
        script=script_library(games),
        cutoff=str(games.game_date.max().date()),
    )

    joblib.dump(bundle, out / "minutes_model.joblib")
    states.to_parquet(
        out / "game_states.parquet", index=False
    )
    games.to_parquet(
        out / "games.parquet", index=False
    )
    d.to_parquet(
        out / "player_games.parquet", index=False
    )
    table.to_csv(
        out / "starters_on_floor.csv", index=False
    )
    print(f"Saved model and tables to {out}")


if __name__ == "__main__":
    main()
