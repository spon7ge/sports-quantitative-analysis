from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Running this file directly puts the package dir on sys.path.
    # The old script lived in src, next to lineup.py.
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from game_enviroment.features import build_features
    from game_enviroment.games import build_game_table, load_gamelogs
    from game_enviroment.simulate import (
        load_model,
        print_sim_summary,
        save_model,
        simulate_game,
    )
    from game_enviroment.train import train
    from lineup import (
        attach_lineup_fill,
        build_lineup_features,
        merge_lineup_features,
    )

    ap = argparse.ArgumentParser(
        description="Layer 1 game environment model"
    )
    ap.add_argument("--data", help="Parquet file, directory, or glob")
    ap.add_argument("--model-out", default="game_env_model.json")
    ap.add_argument("--model-in")
    ap.add_argument("--test-seasons", type=int, default=1)
    ap.add_argument("--validation-season", default=None)
    ap.add_argument("--holdout-season", default=None)
    ap.add_argument("--include-playoffs", action="store_true")
    ap.add_argument("--halflife", type=float, default=12.0)
    ap.add_argument("--prior-weight", type=float, default=4.0)
    ap.add_argument("--season-carry", type=float, default=0.6)
    ap.add_argument("--league-window", type=int, default=300)
    ap.add_argument("--min-league-games", type=int, default=30)
    ap.add_argument(
        "--spread-convention",
        choices=["auto", "negative-favorite", "positive-favorite"],
        default="auto",
        help="Convention in historical logs",
    )
    ap.add_argument("--plots", action="store_true")
    ap.add_argument(
        "--simulate",
        nargs=2,
        metavar=("TEAM", "OPPONENT"),
        help="TEAM is home",
    )
    ap.add_argument(
        "--date", default=str(pd.Timestamp.today().date())
    )
    ap.add_argument("--total", type=float)
    ap.add_argument(
        "--spread", type=float,
        help="TEAM spread; negative means favored",
    )
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sim-out")
    ap.add_argument(
        "--backtest",
        action="store_true",
        help="Score the frozen simulation path on a final date holdout",
    )
    ap.add_argument("--out-dir", default="backtest_out")
    ap.add_argument("--pbp-dir", default="data/PlayByPlay")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--minute-halflife", type=float, default=8.0)
    ap.add_argument("--roster-window", type=int, default=12)
    ap.add_argument("--gtd-probability", type=float, default=0.5)
    args = ap.parse_args()

    if (
        not np.isfinite(args.halflife)
        or args.halflife <= 0
        or not np.isfinite(args.prior_weight)
        or args.prior_weight < 0
    ):
        ap.error("Require finite halflife > 0 and prior-weight >= 0")
    if (
        not 0 <= args.season_carry <= 1
        or not 1 <= args.min_league_games <= args.league_window
    ):
        ap.error(
            "Carry must be in [0, 1]; require "
            "1 <= min-league-games <= league-window"
        )
    if args.test_seasons < 1 or args.n < 1 or args.draws < 1:
        ap.error("--test-seasons, --n, and --draws must be positive")
    if not 0 < args.train_frac < 1 or not 0 < args.val_frac < 1:
        ap.error("--train-frac and --val-frac must be between 0 and 1")
    if args.train_frac + args.val_frac >= 1:
        ap.error("--train-frac and --val-frac must sum to less than 1")
    if not 0 <= args.gtd_probability <= 1:
        ap.error("--gtd-probability must be between 0 and 1")
    if args.minute_halflife <= 0 or args.roster_window < 1:
        ap.error("--minute-halflife must be positive and --roster-window at least 1")

    if args.backtest:
        if not args.data:
            ap.error("--data is required for --backtest")
        from game_enviroment.backtest import run_backtest
        run_backtest(args)
        return

    if args.model_in:
        model = load_model(args.model_in)
    else:
        if not args.data:
            ap.error("--data is required unless --model-in is supplied")

        logs = load_gamelogs(args.data)
        g = build_game_table(logs, args.include_playoffs)
        lineups, lineup_state = build_lineup_features(
            logs,
            args.halflife,
            args.prior_weight,
            args.season_carry,
            args.league_window,
            args.include_playoffs,
        )
        g = merge_lineup_features(g, lineups)
        g, state = build_features(
            g,
            args.halflife,
            args.prior_weight,
            args.season_carry,
            args.league_window,
            args.min_league_games,
        )
        model = train(g, state, args)
        model["state"]["lineup"] = attach_lineup_fill(
            lineup_state, g
        )
        save_model(model, args.model_out)

    if args.simulate:
        sim = simulate_game(
            model,
            *args.simulate,
            args.date,
            args.total,
            args.spread,
            args.n,
            args.seed,
        )
        print_sim_summary(sim, args.total, args.spread)
        if args.sim_out:
            writer = (
                sim.to_csv
                if args.sim_out.lower().endswith(".csv")
                else sim.to_parquet
            )
            writer(args.sim_out, index=False)
            print(f"Wrote {args.sim_out}")


if __name__ == "__main__":
    main()
