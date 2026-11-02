"""Score shared slopes against one player-level slope at a time.

Training history is 2024-25, 2025-26, and the start of 2026-27. The holdout
is each player's last ``--holdout-games`` in 2026-27. Those games are scored
with the joint log likelihood of minutes, attempts, and makes. A slope is
recommended as varying only when that score beats the all-shared model and
the 5% quantile of its player-level sd is above 0.01.

    .venv/bin/python scripts/compare_varying_slopes.py --max-days 28

``--max-days`` clips 2026-27 only. The two earlier seasons stay whole so
player trails see recent history. ``--only usg_stars_out`` is the default.
``--only all`` runs every eligible slope. This is not a full 2026-27 fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hierarchical_points import (  # noqa: E402
    VARYING_SLOPE_NAMES,
    build_design,
    build_model,
    expanded_covariates,
    fit,
    transform_design,
)
from src.models.slope_comparison import (  # noqa: E402
    holdout_elpd,
    modeled_holdout_mask,
    one_at_a_time_configs,
    plot_shrinkage,
    posterior_draws,
    recommend_slope,
    sampling_diagnostics,
    sd_away_from_zero,
    sd_summary,
    slope_stems,
)

HISTORY_SEASONS = ("2024-25", "2025-26", "2026-27")
HOLDOUT_SEASON = "2026-27"

HIER_COLUMNS = [
    "season_type",
    "season_year",
    "game_date",
    "game_id",
    "player_id",
    "team_id",
    "opp_team_id",
    "pos",
    "matchup",
    "minutes",
    "fga",
    "fgm",
    "fg3_a",
    "fg3_m",
    "fta",
    "ftm",
    "pts",
    "start_position",
    "game_total",
    "player_team_spread",
    "team_pace",
    "team_def_rating",
    "usg_pct",
]


def _history_frame(max_days: int | None) -> pd.DataFrame:
    """2024-25 and 2025-26 in full, plus 2026-27 through ``max_days``."""
    pieces = []
    for season in HISTORY_SEASONS:
        pattern = f"data/silver/nba/{season}/regular_season/player_gamelogs.parquet"
        matches = sorted(ROOT.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one gamelog parquet for {season}, found {matches}"
            )
        season_frame = pd.read_parquet(matches[0], columns=HIER_COLUMNS)
        season_frame = season_frame.loc[
            season_frame["season_type"].eq("Regular Season")
        ].copy()
        if max_days is not None and season == HOLDOUT_SEASON:
            dates = pd.to_datetime(season_frame["game_date"])
            last = dates.min() + pd.Timedelta(days=max_days)
            season_frame = season_frame.loc[dates.le(last)].copy()
        if season_frame.empty:
            raise ValueError(f"{season} has no rows in the requested window")
        pieces.append(season_frame)
    return pd.concat(pieces, ignore_index=True)


def _sd_names(stem: str, use_role_tier: bool) -> list[str]:
    if use_role_tier:
        return [f"{stem}_tier_sd", f"{stem}_player_sd"]
    return [f"{stem}_sd"]


def _z_names(stem: str, use_role_tier: bool) -> list[str]:
    names = [f"{stem}_z"]
    if use_role_tier:
        names.append(f"{stem}_tier_z")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-days", type=int, default=28)
    parser.add_argument("--holdout-games", type=int, default=2)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--tune", type=int, default=100)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument(
        "--only",
        default="usg_stars_out",
        choices=[*VARYING_SLOPE_NAMES, "all"],
    )
    parser.add_argument("--role-tier", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "slope_comparison",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = _history_frame(args.max_days)
    mask, split_audit = modeled_holdout_mask(
        frame,
        args.holdout_games,
        expanded_covariates(),
        season=HOLDOUT_SEASON,
    )
    if not mask.any():
        raise ValueError(
            "holdout is empty; players need more games than --holdout-games"
        )
    skipped = split_audit["unscored_trailing_rows"]
    if skipped:
        market_gap = split_audit["missing_market_trailing_rows"]
        print(
            f"Left {skipped} trailing games out of the holdout "
            "because the feature filters remove them "
            f"({market_gap} missing a total or spread)."
        )
    design = build_design(frame.loc[~mask].copy(), covariates=expanded_covariates())
    holdout, holdout_audit = transform_design(
        frame, design, keep=pd.Series(mask, index=frame.index)
    )
    print(
        f"Train rows {design.audit['output_rows']:,}; "
        f"holdout rows {holdout_audit['scoring_rows']:,}."
    )
    configs = one_at_a_time_configs(use_role_tier=args.role_tier)
    if args.only != "all":
        configs = [item for item in configs if item[0] in {"shared", args.only}]

    rows = []
    shared_elpd = None
    args.output.mkdir(parents=True, exist_ok=True)
    for name, config in configs:
        print(f"\nFitting {name}")
        model = build_model(
            design.data,
            design.n_players,
            design.n_teams,
            design.n_pos,
            design.n_blocks,
            minutes_columns=design.minutes_columns,
            rate_columns=design.rate_columns,
            slopes=config,
        )
        idata = fit(
            model,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            random_seed=args.seed,
            progressbar=True,
        )
        elpd = holdout_elpd(model, idata, holdout)
        if name == "shared":
            shared_elpd = elpd
            rows.append(
                {
                    "variant": name,
                    "elpd": elpd,
                    "elpd_minus_shared": 0.0,
                    "recommendation": "baseline",
                }
            )
            print(f"{name} holdout elpd {elpd:.2f}")
            continue
        if shared_elpd is None:
            raise RuntimeError("fit the shared variant before a player-level slope")
        stems = slope_stems(config, design.minutes_columns, design.rate_columns)
        sd_names = [sd for stem in stems for sd in _sd_names(stem, args.role_tier)]
        z_names = [z for stem in stems for z in _z_names(stem, args.role_tier)]
        diagnostics = sampling_diagnostics(idata, [*sd_names, *z_names])
        print(diagnostics.to_string(index=False))
        details = []
        decisions = []
        for sd_name in sd_names:
            samples = posterior_draws(idata, sd_name)
            summary = sd_summary(samples)
            decisions.append(recommend_slope(shared_elpd, elpd, samples))
            away = "away from zero" if sd_away_from_zero(samples) else "near zero"
            details.append(
                f"{sd_name} median={summary['median']:.4f} "
                f"q05={summary['q05']:.4f} "
                f"frac<0.01={summary['fraction_below_near_zero']:.2f} ({away})"
            )
            print(details[-1])
        for stem in stems:
            plot_shrinkage(
                posterior_draws(idata, stem),
                posterior_draws(idata, f"{stem}_mu"),
                args.output / f"{name}_{stem}.png",
                title=stem,
            )
        # A feature can have one minutes slope and one slope per attempt type.
        # Keep it varying only when every one of those sds clears the cutoff.
        every_sd_clears = bool(decisions) and all(item == "vary" for item in decisions)
        recommendation = "vary" if every_sd_clears else "share"
        rows.append(
            {
                "variant": name,
                "elpd": elpd,
                "elpd_minus_shared": elpd - shared_elpd,
                "recommendation": recommendation,
                "sd": "; ".join(details),
            }
        )
    table = pd.DataFrame(rows)
    table_path = args.output / "elpd.csv"
    table.to_csv(table_path, index=False)
    print("\nHoldout elpd")
    print(table.to_string(index=False))
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
