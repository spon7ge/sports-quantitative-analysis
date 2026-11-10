from __future__ import annotations

import numpy as np
from game_enviroment.constants import FEATURES
from game_enviroment.games import _season_label
from game_enviroment.pace import (
    _date_split,
    choose_lambda,
    fit_model,
    fit_t,
    predict_dev,
)
from game_enviroment.pbp_script import (
    choose_ot_blend,
    fit_ot_path,
    fit_pace_state,
)
from game_enviroment.script import (
    _col,
    _draw,
    _expected_possessions,
    _ot_probability,
    _score_means,
    fit_ot,
    fit_scoring,
    fit_split,
)
from lineup import LINEUP_GAME_FEATURES
from scipy import stats


def _spread_sign(df, convention):
    if convention != "auto":
        return -1.0 if convention == "negative-favorite" else 1.0

    available = df.home_spread.notna()
    x = df.loc[available, "home_spread"]
    y = df.loc[available, "margin"]
    if len(x) >= 50 and x.std() > 1e-8:
        slope = float(
            np.cov(x, y)[0, 1] / np.var(x, ddof=1)
        )
        if abs(slope) > 0.1:
            return float(np.sign(slope))

    print(
        "  WARNING: spread convention unclear; "
        "assuming negative = favorite."
    )
    return -1.0


def _season_split(df, validation_season, holdout_season):
    """Every other season trains. The named seasons validate and hold out."""
    validation = _season_label(validation_season)
    holdout = _season_label(holdout_season)
    if validation == holdout:
        raise ValueError("validation and holdout seasons must differ")
    labels = df.season_year.astype(str)
    for name in (validation, holdout):
        if name not in set(labels):
            raise ValueError(f"missing season {name}")
    validation_rows = labels.eq(validation)
    holdout_rows = labels.eq(holdout)
    return (
        df.loc[~validation_rows & ~holdout_rows].copy(),
        df.loc[validation_rows].copy(),
        df.loc[holdout_rows].copy(),
    )


def _deploy_frame(df, holdout_season):
    """The saved fit keeps the validation season and leaves the holdout out."""
    if not holdout_season:
        return df
    holdout = _season_label(holdout_season)
    return df.loc[~df.season_year.astype(str).eq(holdout)].copy()


def train(g, state, args, script=None):
    df = g.dropna(subset=["lg_poss"]).reset_index(drop=True)
    validation_season = getattr(args, "validation_season", None)
    holdout_season = getattr(args, "holdout_season", None)
    if validation_season or holdout_season:
        if not validation_season or not holdout_season:
            raise ValueError("supply both validation_season and holdout_season")
        tr, va, te = _season_split(df, validation_season, holdout_season)
    else:
        seasons = sorted(df.season_year.unique())
        if len(seasons) > args.test_seasons:
            test = df.season_year.isin(seasons[-args.test_seasons:])
            before_test, te = df[~test].copy(), df[test].copy()
        else:
            before_test, te = _date_split(df, 0.75)
        tr, va = _date_split(before_test, 0.80)
    if (
        len(tr) < max(50, len(FEATURES) + 10)
        or min(len(va), len(te)) < 20
    ):
        raise ValueError(
            "Need more historical games for training, "
            "validation, and holdout."
        )

    fill = (
        tr[list(LINEUP_GAME_FEATURES)]
        .mean()
        .fillna(0.0)
        .to_dict()
    )
    spread_sign = _spread_sign(
        tr, getattr(args, "spread_convention", "auto")
    )
    for frame in (df, tr, va, te):
        frame["home_spread"] *= -spread_sign
    for frame in (tr, va, te):
        frame[list(LINEUP_GAME_FEATURES)] = (
            frame[list(LINEUP_GAME_FEATURES)].fillna(fill)
        )

    team_ids = sorted(set(tr.home_id) | set(tr.away_id))
    lam = choose_lambda(tr, team_ids)
    candidates = {
        name: fit_model(name, tr, team_ids, lam)
        for name in ("league_prior", "ols", "ridge_team")
    }
    errors = {
        name: float(np.sqrt(np.mean(
            (
                va.y_dev.to_numpy()
                - predict_dev(candidate, va)
            ) ** 2
        )))
        for name, candidate in candidates.items()
    }
    best = min(
        errors,
        key=lambda name: (errors[name], name != "league_prior"),
    )

    print(
        f"\nGames: train {len(tr):,}, validation {len(va):,}, "
        f"holdout {len(te):,}"
    )
    if validation_season:
        print(
            "Seasons: train through "
            f"{tr.season_year.astype(str).max()}, "
            f"validation {_season_label(validation_season)}, "
            f"holdout {_season_label(holdout_season)}"
        )
    print("Validation pace RMSE:", {
        name: round(error, 3)
        for name, error in errors.items()
    })
    print(f"Selected: {best}")

    pace_resid = (
        va.y_dev.to_numpy() - predict_dev(candidates[best], va)
    )
    model = {
        "version": 3,
        "features": FEATURES,
        "model": candidates[best],
        "residual_t": fit_t(pace_resid),
        "ot": fit_ot(tr),
        "split": fit_split(tr),
        "scoring": fit_scoring(tr),
    }
    _attach_script(model, script, tr.game_id, va.game_id)

    scoring = model["scoring"]
    hp, ap = _expected_possessions(model, va)
    weight_errors = {}
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        scoring["market_weight"] = weight
        h, a = _score_means(scoring, va, hp, ap)
        weight_errors[weight] = np.mean(
            (va.home_pts.to_numpy() - h) ** 2
            + (va.away_pts.to_numpy() - a) ** 2
        )
    scoring["market_weight"] = min(
        weight_errors, key=weight_errors.get
    )

    h, a = _score_means(
        scoring,
        va,
        va.home_poss.to_numpy(),
        va.away_poss.to_numpy(),
    )
    eh = va.home_pts.to_numpy() - h
    ea = va.away_pts.to_numpy() - a
    rho = (
        np.corrcoef(eh, ea)[0, 1]
        if min(np.std(eh), np.std(ea)) > 1e-8 else 0.0
    )
    rho = float(np.nan_to_num(rho))
    scoring["residual"] = {
        **fit_t(np.r_[eh, ea]),
        "rho": float(np.clip(rho, -0.999, 0.999)),
    }

    print(
        f"Market weight: {scoring['market_weight']:.2f}; "
        f"scoring residual correlation: {rho:+.2f}"
    )
    print(
        "Historical spreads:",
        "negative = favorite" if spread_sign < 0
        else "positive = favorite",
    )
    print("Live/API spreads: negative = favorite")

    samples = _draw(
        model, te, 1000, np.random.default_rng(7)
    )
    metrics = {}
    print("\nHoldout: actual simulation path (90% interval coverage)")
    for name, observed in [
        ("pace", "poss48"),
        ("home_poss", "home_poss"),
        ("home_pts", "home_pts"),
        ("game_total", "pts_total"),
        ("margin", "margin"),
    ]:
        draws = samples[name]
        actual = te[observed].to_numpy(float)
        low, high = np.quantile(
            draws, [0.05, 0.95], axis=0
        )
        rmse = float(np.sqrt(np.mean(
            (actual - draws.mean(axis=0)) ** 2
        )))
        coverage = float(np.mean(
            (actual >= low) & (actual <= high)
        ))
        metrics[name] = {
            "rmse": rmse,
            "coverage_90": coverage,
        }
        print(
            f"  {name:<12} RMSE {rmse:6.2f}; "
            f"coverage {coverage:.1%}"
        )

    if "ot_path" in model:
        probability = (samples["n_ot"] > 0).mean(axis=0)
    else:
        probability = _ot_probability(
            model["ot"], _col(te, "home_spread", np.nan)
        )
    metrics["ot_brier"] = float(np.mean(
        (probability - te.n_ot.ge(1).to_numpy()) ** 2
    ))
    print(f"  OT Brier score: {metrics['ot_brier']:.4f}")

    deploy = _deploy_frame(df, holdout_season)
    final_fill = (
        deploy[list(LINEUP_GAME_FEATURES)]
        .mean()
        .fillna(0.0)
        .to_dict()
    )
    deploy[list(LINEUP_GAME_FEATURES)] = (
        deploy[list(LINEUP_GAME_FEATURES)].fillna(final_fill)
    )
    all_teams = sorted(set(deploy.home_id) | set(deploy.away_id))
    model["model"] = fit_model(best, deploy, all_teams, lam)

    final_scoring = fit_scoring(deploy)
    final_scoring.update(
        market_weight=scoring["market_weight"],
        residual=scoring["residual"],
    )
    model.update(
        scoring=final_scoring,
        ot=fit_ot(deploy),
        split=fit_split(deploy),
        state=state,
        lineup_fill=final_fill,
        holdout=metrics,
        trained_games=len(deploy),
        config={
            "halflife": args.halflife,
            "prior_weight": args.prior_weight,
            "season_carry": args.season_carry,
            "league_window": args.league_window,
            "historical_spread_sign": spread_sign,
            "validation_season": (
                _season_label(validation_season) if validation_season else None
            ),
            "holdout_season": (
                _season_label(holdout_season) if holdout_season else None
            ),
        },
    )
    if script is not None:
        blend = model["ot_path"]["blend"]
        _attach_script(model, script, deploy.game_id, None)
        model["ot_path"]["blend"] = blend

    if args.plots:
        _plots(pace_resid, model["residual_t"])
    return model


def _attach_script(model, script, fit_ids, blend_ids):
    """Fit pace and overtime on fit_ids. Choose the blend on blend_ids."""

    if script is None:
        return
    model["pace_state"] = fit_pace_state(script["exposure"], fit_ids)
    ot_path = fit_ot_path(script["paths"], fit_ids)
    if blend_ids is not None:
        ot_path["blend"] = choose_ot_blend(ot_path, script["paths"], blend_ids)
    model["ot_path"] = ot_path


def _plots(
    residuals, residual, path="game_env_diagnostics.png",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        residuals, bins=40, density=True, alpha=0.6
    )
    x = np.linspace(residuals.min(), residuals.max(), 300)
    if residual["scale"] > 0:
        axes[0].plot(
            x,
            stats.t.pdf(
                x, residual["nu"], scale=residual["scale"]
            ),
        )
        stats.probplot(
            residuals,
            dist=stats.t,
            sparams=(residual["nu"], 0, residual["scale"]),
            plot=axes[1],
        )
    axes[0].set_title("Validation pace residuals")
    axes[1].set_title("Student-t QQ")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")
