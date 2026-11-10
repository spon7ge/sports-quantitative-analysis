"""Chronological backtest of the shared environment draw.

Coefficients are fit on the training dates and chosen on the validation
dates. Test dates are scored once. Rolling pace and rotation inputs keep
updating after games that have already been played; those updates are in
the feature rows, not in a refit.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from game_enviroment.constants import FEATURES
from game_enviroment.features import build_features
from game_enviroment.games import build_game_table, load_gamelogs
from game_enviroment.pace import choose_lambda, fit_model, fit_t, predict_dev
from game_enviroment.rotation import (
    AVAILABILITY_SOURCE,
    DEFAULT_GTD_PROBABILITY,
    GTD_PROBABILITY_SOURCE,
    build_rotation_history,
)
from game_enviroment.pbp_script import apply_possession_counts, load_play_by_play
from game_enviroment.script import (
    _draw,
    _expected_possessions,
    _score_means,
    fit_ot,
    fit_scoring,
    fit_split,
)
from game_enviroment.train import _attach_script
from game_enviroment.train import _spread_sign
from lineup import LINEUP_GAME_FEATURES

DRAW_CHUNK = 200
_QUANTILES = {
    "p025": 0.025,
    "p05": 0.05,
    "p10": 0.10,
    "p50": 0.50,
    "p90": 0.90,
    "p95": 0.95,
    "p975": 0.975,
}
_INTERVALS = {"80": ("p10", "p90"), "90": ("p05", "p95"), "95": ("p025", "p975")}
_BASE_FEATURES = [
    name for name in FEATURES if name not in LINEUP_GAME_FEATURES
]
_ROTATION_TO_LINEUP = {
    "lineup_pace_h": "rotation_pace_h",
    "lineup_pace_a": "rotation_pace_a",
    "lineup_off_rating_h": "rotation_off_rating_h",
    "lineup_off_rating_a": "rotation_off_rating_a",
    "lineup_def_rating_h": "rotation_def_rating_h",
    "lineup_def_rating_a": "rotation_def_rating_a",
}
_MARKET_NOTE = (
    "game_total and team_spread on the player logs come from the Rotowire "
    "season archive (Over_Under, Home_Line). That file has a tipoff and a "
    "final score, and no odds timestamp, so these lines are unspecified: "
    "they are not labeled opening or closing."
)
_AVAILABILITY_NOTE = (
    "Historical starters are players with start_position G, F, or C. A blank "
    "start_position is the bench. Their minutes and ratings still come from "
    "earlier games only. There is no injury snapshot, so a player absent from "
    "the log is not marked out. "
    f"The live GTD probability {DEFAULT_GTD_PROBABILITY:.2f} is an assumption, "
    "not a rate estimated from those snapshots. This test does not measure "
    "the value of live injury information."
)


def assign_splits(games, train_frac=0.6, val_frac=0.2):
    """Put every game on one date into the same chronological split."""
    if not (
        0 < train_frac < 1
        and 0 < val_frac < 1
        and train_frac + val_frac < 1
    ):
        raise ValueError(
            "train_frac and val_frac must be positive and sum to less than 1"
        )
    out = games.copy()
    dates = np.array(sorted(pd.to_datetime(out["game_date"]).unique()))
    if len(dates) < 3:
        raise ValueError(
            "Need games on at least three dates to form train, validation, and test"
        )
    i_train = min(max(int(len(dates) * train_frac), 1), len(dates) - 2)
    i_val = min(
        max(int(len(dates) * (train_frac + val_frac)), i_train + 1),
        len(dates) - 1,
    )
    when = pd.to_datetime(out["game_date"])
    split = np.full(len(out), "test", dtype=object)
    split[when.lt(dates[i_train]).to_numpy()] = "train"
    split[
        when.ge(dates[i_train]).to_numpy() & when.lt(dates[i_val]).to_numpy()
    ] = "val"
    out["split"] = split
    out.attrs["splits"] = {
        "train_last_date": str(pd.Timestamp(dates[i_train - 1]).date()),
        "val_first_date": str(pd.Timestamp(dates[i_train]).date()),
        "val_last_date": str(pd.Timestamp(dates[i_val - 1]).date()),
        "test_first_date": str(pd.Timestamp(dates[i_val]).date()),
        "test_last_date": str(pd.Timestamp(dates[-1]).date()),
    }
    return out


def fit_frozen_model(train, val, features, script=None):
    """Fit on train, choose the pace model and market weight on validation."""
    train = train.copy()
    val = val.copy()
    columns = list(features)
    if len(train) < max(50, len(columns) + 10) or len(val) < 20:
        raise ValueError(
            "Need more games for a frozen train/validation fit"
        )
    fill = train[columns].mean()
    train[columns] = train[columns].fillna(fill)
    val[columns] = val[columns].fillna(fill)
    team_ids = sorted(set(train.home_id) | set(train.away_id))
    lam = choose_lambda(train, team_ids, features=columns)
    candidates = {
        name: fit_model(name, train, team_ids, lam, features=columns)
        for name in ("league_prior", "ols", "ridge_team")
    }
    errors = {
        name: float(np.sqrt(np.mean(
            (val.y_dev.to_numpy(float) - predict_dev(candidate, val)) ** 2
        )))
        for name, candidate in candidates.items()
    }
    best = min(errors, key=lambda name: (errors[name], name != "league_prior"))
    model = {
        "version": 3,
        "features": columns,
        "model": candidates[best],
        "residual_t": fit_t(
            val.y_dev.to_numpy(float) - predict_dev(candidates[best], val)
        ),
        "ot": fit_ot(train),
        "split": fit_split(train),
        "scoring": fit_scoring(train),
        "selection": {
            "validation_rmse": errors,
            "selected": best,
            "lambda": lam,
        },
    }
    _attach_script(model, script, train.game_id, val.game_id)
    scoring = model["scoring"]
    home_poss, away_poss = _expected_possessions(model, val)
    weight_errors = {}
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        scoring["market_weight"] = weight
        home_hat, away_hat = _score_means(
            scoring, val, home_poss, away_poss
        )
        weight_errors[weight] = float(np.mean(
            (val.home_pts.to_numpy(float) - home_hat) ** 2
            + (val.away_pts.to_numpy(float) - away_hat) ** 2
        ))
    scoring["market_weight"] = min(weight_errors, key=weight_errors.get)
    home_hat, away_hat = _score_means(
        scoring,
        val,
        val.home_poss.to_numpy(float),
        val.away_poss.to_numpy(float),
    )
    home_error = val.home_pts.to_numpy(float) - home_hat
    away_error = val.away_pts.to_numpy(float) - away_hat
    if min(np.std(home_error), np.std(away_error)) > 1e-8:
        rho = float(np.corrcoef(home_error, away_error)[0, 1])
    else:
        rho = 0.0
    scoring["residual"] = {
        **fit_t(np.r_[home_error, away_error]),
        "rho": float(np.clip(np.nan_to_num(rho), -0.999, 0.999)),
    }
    model["selection"]["market_weight"] = scoring["market_weight"]
    return model


def team_view(samples, side):
    """Read one side of a home/away draw without drawing again."""
    side = str(side).lower()
    if side not in {"home", "away"}:
        raise ValueError("side must be home or away")
    own, other = ("home", "away") if side == "home" else ("away", "home")
    sign = 1.0 if side == "home" else -1.0
    home_pts = np.asarray(samples["home_pts"], dtype=float)
    away_pts = np.asarray(samples["away_pts"], dtype=float)
    return {
        "team_poss": np.asarray(samples[f"{own}_poss"], dtype=float),
        "opp_poss": np.asarray(samples[f"{other}_poss"], dtype=float),
        "team_pts": np.asarray(samples[f"{own}_pts"], dtype=float),
        "opp_pts": np.asarray(samples[f"{other}_pts"], dtype=float),
        "game_total": home_pts + away_pts,
        "team_margin": sign * (home_pts - away_pts),
        "pace": np.asarray(samples["pace"], dtype=float),
        "n_ot": np.asarray(samples["n_ot"], dtype=float),
        "game_minutes": np.asarray(samples["game_minutes"], dtype=float),
        "is_home": 1.0 if side == "home" else 0.0,
    }


def implied_market(home_spread, total):
    """Negative home spread is the favorite. Margin is the negative of that spread."""
    spread = float(home_spread)
    line = float(total)
    return {
        "market_home_margin": -spread,
        "market_home_pts": (line - spread) / 2.0,
        "market_away_pts": (line + spread) / 2.0,
    }


def market_comparison(frame):
    """Score the model and the posted line on the same games.

    Rows missing a spread or a total stay out of this comparison. They are
    not given a synthetic market number.
    """
    spread = pd.to_numeric(frame["home_spread"], errors="coerce").to_numpy(float)
    total = pd.to_numeric(frame["total"], errors="coerce").to_numpy(float)
    ok = np.isfinite(spread) & np.isfinite(total)
    sub = frame.loc[ok]
    if sub.empty:
        return {"n": 0}
    market_margin = -sub["home_spread"].to_numpy(float)
    market_home = (
        sub["total"].to_numpy(float) - sub["home_spread"].to_numpy(float)
    ) / 2.0
    market_away = (
        sub["total"].to_numpy(float) + sub["home_spread"].to_numpy(float)
    ) / 2.0
    out = {"n": int(len(sub))}
    pairs = {
        "margin": (sub["pred_margin"], sub["actual_margin"], market_margin),
        "home_pts": (sub["pred_home_pts"], sub["actual_home_pts"], market_home),
        "away_pts": (sub["pred_away_pts"], sub["actual_away_pts"], market_away),
        "total": (sub["pred_total"], sub["actual_total"], market_home + market_away),
    }
    for name, (pred, actual, market) in pairs.items():
        out[f"model_{name}_mae"] = _mae(pred, actual)
        out[f"model_{name}_rmse"] = _rmse(pred, actual)
        out[f"market_{name}_mae"] = _mae(market, actual)
        out[f"market_{name}_rmse"] = _rmse(market, actual)
    return out


def summarize_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Final score, regulation score, and overtime count from play-by-play.

    Blank score rows are ignored. This does not count possessions.
    """
    frame = pbp.copy()
    if "gameId" in frame.columns:
        frame["game_id"] = frame["gameId"].map(_game_id)
    elif "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].map(_game_id)
    else:
        raise ValueError("play-by-play needs gameId")
    frame["period"] = pd.to_numeric(frame["period"], errors="coerce")
    frame["home"] = pd.to_numeric(frame.get("scoreHome"), errors="coerce")
    frame["away"] = pd.to_numeric(frame.get("scoreAway"), errors="coerce")
    if "actionNumber" in frame.columns:
        frame["action"] = pd.to_numeric(frame["actionNumber"], errors="coerce")
    else:
        frame["action"] = np.arange(len(frame))
    scored = frame.dropna(subset=["game_id", "home", "away", "period"])
    rows = []
    for game_id, game in scored.groupby("game_id", sort=False):
        ordered = game.sort_values("action")
        final = ordered.iloc[-1]
        regulation = ordered.loc[ordered["period"].le(4)]
        rows.append(
            {
                "game_id": game_id,
                "reg_home": (
                    float(regulation.iloc[-1]["home"]) if len(regulation) else np.nan
                ),
                "reg_away": (
                    float(regulation.iloc[-1]["away"]) if len(regulation) else np.nan
                ),
                "final_home": float(final["home"]),
                "final_away": float(final["away"]),
                "n_ot": max(0, int(ordered["period"].max()) - 4),
            }
        )
    return pd.DataFrame(rows)


def run_backtest(args):
    """Fit baseline and rotation models on the same dates and score the test set."""
    out_dir = Path(getattr(args, "out_dir", "backtest_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    draws = int(getattr(args, "draws", 5000) or 5000)
    seed = int(getattr(args, "seed", 7))
    if draws < 1:
        raise ValueError("draws must be positive")

    logs, failed_files = _load_logs(args.data)
    games = build_game_table(logs, getattr(args, "include_playoffs", False))
    script = load_play_by_play(getattr(args, "pbp_dir", None))
    if script is not None:
        games = apply_possession_counts(games, script["counts"])
    games, _end_state = build_features(
        games,
        args.halflife,
        args.prior_weight,
        args.season_carry,
        args.league_window,
        args.min_league_games,
    )
    print("Building prior-only rotations")
    rotation_players, rotation_teams = build_rotation_history(
        logs,
        halflife=float(getattr(args, "minute_halflife", 8.0)),
        prior_weight=float(args.prior_weight),
        roster_window=int(getattr(args, "roster_window", 12)),
    )
    games = _attach_rotation(games, rotation_teams)
    games = games.dropna(subset=[
        "lg_poss", "y_dev", "poss48", "home_poss", "away_poss",
        "home_pts", "away_pts", "margin",
    ]).reset_index(drop=True)
    games = assign_splits(
        games,
        train_frac=float(getattr(args, "train_frac", 0.6)),
        val_frac=float(getattr(args, "val_frac", 0.2)),
    )
    train_mask = games["split"].eq("train")
    sign = _spread_sign(
        games.loc[train_mask], getattr(args, "spread_convention", "auto")
    )
    games["home_spread"] = games["home_spread"] * -sign
    limits = games.attrs["splits"]
    print(
        "Splits:",
        {name: int((games["split"] == name).sum()) for name in ("train", "val", "test")},
    )
    print(
        f"Train through {limits['train_last_date']}, "
        f"validation {limits['val_first_date']} to {limits['val_last_date']}, "
        f"test {limits['test_first_date']} to {limits['test_last_date']}"
    )

    prepared = {
        "baseline": _prepare_baseline(games),
        "rotation": _prepare_rotation(games, train_mask),
    }
    predictions = []
    stored = {}
    selections = {}
    for name, (frame, columns) in prepared.items():
        print(f"Fitting {name}")
        model = fit_frozen_model(
            frame.loc[frame["split"].eq("train")],
            frame.loc[frame["split"].eq("val")],
            columns,
            script,
        )
        print(
            f"  selected {model['selection']['selected']}; "
            f"market weight {model['scoring']['market_weight']:.2f}"
        )
        test = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
        print(f"  drawing {draws} samples for {len(test):,} test games")
        forecast = _forecast(model, test, draws, seed)
        forecast.insert(0, "feature_set", name)
        predictions.append(forecast)
        stored[name] = forecast
        selections[name] = {
            "selected": model["selection"]["selected"],
            "market_weight": model["scoring"]["market_weight"],
            "validation_pace_rmse": model["selection"]["validation_rmse"],
        }

    predicted = pd.concat(predictions, ignore_index=True)
    pbp_note, predicted = _attach_pbp(
        predicted, getattr(args, "pbp_dir", None)
    )
    predicted.to_csv(out_dir / "predictions.csv", index=False)
    context = _test_context(rotation_players, predicted)
    context.to_csv(out_dir / "rotation_context.csv", index=False)

    metrics = {
        "draws": draws,
        "seed": seed,
        "splits": {
            **limits,
            "n_train": int(train_mask.sum()),
            "n_val": int(games["split"].eq("val").sum()),
            "n_test": int(games["split"].eq("test").sum()),
        },
        "spread_sign": (
            "negative = favorite" if sign < 0 else "positive = favorite, flipped"
        ),
        "market_timing": "unspecified",
        "market_note": _MARKET_NOTE,
        "availability_source": AVAILABILITY_SOURCE,
        "gtd_probability": DEFAULT_GTD_PROBABILITY,
        "gtd_probability_source": GTD_PROBABILITY_SOURCE,
        "availability_note": _AVAILABILITY_NOTE,
        "availability_slice": {
            "available": False,
            "reason": _AVAILABILITY_NOTE,
        },
        "unreadable_logs": failed_files,
        "pbp": pbp_note,
        "possession_target": (
            "play-by-play counts"
            if script is not None
            else "box estimate on the player logs"
        ),
        "end_of_sample_state_used_for_test": False,
        "models": {
            name: {
                **_model_metrics(frame),
                "selection": selections[name],
            }
            for name, frame in stored.items()
        },
    }
    metrics["comparison"] = _comparison(metrics["models"])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _plots(stored, out_dir)
    print(_summary_text(metrics))
    print(f"Wrote {out_dir / 'predictions.csv'}")
    print(f"Wrote {out_dir / 'metrics.json'}")
    return metrics


def _forecast(model, frame, n, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for start in range(0, len(frame), DRAW_CHUNK):
        part = frame.iloc[start:start + DRAW_CHUNK].reset_index(drop=True)
        samples = _draw(model, part, n, rng)
        parts.append(_summarize_draws(part, samples))
    return pd.concat(parts, ignore_index=True)


def _summarize_draws(frame, samples):
    out = pd.DataFrame(
        {
            "game_id": frame["game_id"].astype(str).to_numpy(),
            "game_date": pd.to_datetime(frame["game_date"]).dt.date.astype(str),
            "home_id": frame["home_id"].to_numpy(),
            "away_id": frame["away_id"].to_numpy(),
            "home_abbr": frame.get("home_abbr"),
            "away_abbr": frame.get("away_abbr"),
            "home_spread": frame["home_spread"].to_numpy(float),
            "total": frame["total"].to_numpy(float),
            "market_missing": (
                ~np.isfinite(frame["home_spread"].to_numpy(float))
                | ~np.isfinite(frame["total"].to_numpy(float))
            ),
            "lg_poss": frame["lg_poss"].to_numpy(float),
            "actual_poss48": frame["poss48"].to_numpy(float),
            "actual_home_poss": frame["home_poss"].to_numpy(float),
            "actual_away_poss": frame["away_poss"].to_numpy(float),
            "actual_home_pts": frame["home_pts"].to_numpy(float),
            "actual_away_pts": frame["away_pts"].to_numpy(float),
            "actual_total": frame["pts_total"].to_numpy(float),
            "actual_margin": frame["margin"].to_numpy(float),
            "actual_n_ot": frame["n_ot"].to_numpy(int),
            "actual_ot": frame["n_ot"].ge(1).to_numpy(float),
        }
    )
    specs = {
        "pace": samples["pace"],
        "home_poss": samples["home_poss"],
        "away_poss": samples["away_poss"],
        "home_pts": samples["home_pts"],
        "away_pts": samples["away_pts"],
        "total": samples["game_total"],
        "margin": samples["margin"],
    }
    for name, draws in specs.items():
        out[f"pred_{name}"] = draws.mean(axis=0)
        for label, quantile in _QUANTILES.items():
            out[f"pred_{name}_{label}"] = np.quantile(draws, quantile, axis=0)
    out["pred_ot"] = (samples["n_ot"] > 0).mean(axis=0)
    out["pred_gap_draw"] = samples["home_poss"][0] - samples["away_poss"][0]
    out["rotation_valid_h"] = frame.get("rotation_valid_h")
    out["rotation_valid_a"] = frame.get("rotation_valid_a")
    out["n_eligible_h"] = frame.get("n_eligible_h")
    out["n_eligible_a"] = frame.get("n_eligible_a")
    out["expected_starters_h"] = frame.get("expected_starters_h")
    out["expected_starters_a"] = frame.get("expected_starters_a")
    prices = [
        implied_market(spread, total)
        if np.isfinite(spread) and np.isfinite(total)
        else {"market_home_margin": np.nan, "market_home_pts": np.nan, "market_away_pts": np.nan}
        for spread, total in zip(out["home_spread"], out["total"], strict=True)
    ]
    out = pd.concat([out, pd.DataFrame(prices)], axis=1)
    return out


def _model_metrics(frame):
    pace = _error_stats(frame["pred_pace"], frame["actual_poss48"])
    possessions = {
        "home": _error_stats(
            frame["pred_home_poss"], frame["actual_home_poss"], bias=False
        ),
        "away": _error_stats(
            frame["pred_away_poss"], frame["actual_away_poss"], bias=False
        ),
    }
    points = {
        "home_pts": _error_stats(frame["pred_home_pts"], frame["actual_home_pts"]),
        "away_pts": _error_stats(frame["pred_away_pts"], frame["actual_away_pts"]),
        "total": _error_stats(frame["pred_total"], frame["actual_total"]),
        "margin": _error_stats(frame["pred_margin"], frame["actual_margin"]),
    }
    intervals = {}
    for target, actual, prefix in (
        ("pace", "actual_poss48", "pred_pace"),
        ("home_poss", "actual_home_poss", "pred_home_poss"),
        ("away_poss", "actual_away_poss", "pred_away_poss"),
        ("home_pts", "actual_home_pts", "pred_home_pts"),
        ("away_pts", "actual_away_pts", "pred_away_pts"),
        ("total", "actual_total", "pred_total"),
        ("margin", "actual_margin", "pred_margin"),
    ):
        intervals[target] = {
            level: _coverage(
                frame[actual],
                frame[f"{prefix}_{low}"],
                frame[f"{prefix}_{high}"],
            )
            for level, (low, high) in _INTERVALS.items()
        }
    missing = frame.loc[frame["market_missing"].astype(bool)]
    observed_gap = frame["actual_home_poss"] - frame["actual_away_poss"]
    return {
        "n": int(len(frame)),
        "possessions_per_48": pace,
        "full_game_possessions": possessions,
        "points": points,
        "intervals": intervals,
        "overtime": _ot_metrics(frame),
        "possession_gap": {
            "observed_mean": float(observed_gap.mean()),
            "observed_std": float(observed_gap.std(ddof=0)),
            "simulated_draw_mean": float(frame["pred_gap_draw"].mean()),
            "simulated_draw_std": float(frame["pred_gap_draw"].std(ddof=0)),
            "mean_forecast_bias": float(
                (frame["pred_home_poss"] - frame["pred_away_poss"] - observed_gap).mean()
            ),
        },
        "league_pace_baseline": _error_stats(frame["lg_poss"], frame["actual_poss48"]),
        "market": market_comparison(frame),
        "missing_market": {
            "n": int(len(missing)),
            "possessions_per_48": _error_stats(
                missing["pred_pace"], missing["actual_poss48"]
            ) if len(missing) else {"n": 0},
            "margin": _error_stats(
                missing["pred_margin"], missing["actual_margin"]
            ) if len(missing) else {"n": 0},
        },
        "invalid_rotation_rate": _invalid_rate(frame),
    }


def _comparison(models):
    lines = []
    improved = []
    for metric, label in (
        ("possessions_per_48", "possessions per 48"),
        ("points", "margin"),
    ):
        if metric == "possessions_per_48":
            base = models["baseline"][metric]["rmse"]
            rot = models["rotation"][metric]["rmse"]
        else:
            base = models["baseline"]["points"]["margin"]["rmse"]
            rot = models["rotation"]["points"]["margin"]["rmse"]
            label = "home margin"
        if base is None or rot is None:
            continue
        change = rot - base
        lines.append(
            f"{label} RMSE baseline {base:.3f}, rotation {rot:.3f}, change {change:+.3f}"
        )
        improved.append(change < -0.05)
    base_total = models["baseline"]["points"]["total"]["rmse"]
    rot_total = models["rotation"]["points"]["total"]["rmse"]
    lines.append(
        f"game total RMSE baseline {base_total:.3f}, rotation {rot_total:.3f}, "
        f"change {rot_total - base_total:+.3f}"
    )
    if all(improved) and rot_total < base_total - 0.05:
        verdict = "Rotation features lowered the held-out pace and scoring errors."
    elif any(change < -0.05 for change in (
        models["rotation"]["possessions_per_48"]["rmse"]
        - models["baseline"]["possessions_per_48"]["rmse"],
        models["rotation"]["points"]["margin"]["rmse"]
        - models["baseline"]["points"]["margin"]["rmse"],
        rot_total - base_total,
    )):
        verdict = (
            "Rotation features helped at least one held-out target. "
            "They did not improve every target."
        )
    else:
        verdict = (
            "Held-out results do not show a clear accuracy gain from rotation features."
        )
    return {"lines": lines, "verdict": verdict}


def _summary_text(metrics):
    splits = metrics["splits"]
    lines = [
        "",
        "Backtest summary",
        (
            f"Train {splits['n_train']} games through {splits['train_last_date']}; "
            f"validation {splits['n_val']} ({splits['val_first_date']} to {splits['val_last_date']}); "
            f"test {splits['n_test']} ({splits['test_first_date']} to {splits['test_last_date']})."
        ),
        f"Draws {metrics['draws']}, seed {metrics['seed']}.",
        f"Market timing: {metrics['market_timing']}. {metrics['market_note']}",
        metrics["availability_note"],
        f"Unreadable log files: {len(metrics['unreadable_logs'])}.",
        f"Play-by-play: {metrics['pbp']}",
    ]
    for name, block in metrics["models"].items():
        pace = block["possessions_per_48"]
        margin = block["points"]["margin"]
        total = block["points"]["total"]
        market_n = block["market"].get("n", 0)
        lines.append(
            f"{name}: pace model {block['selection']['selected']}, "
            f"market weight {block['selection']['market_weight']:.2f}; "
            f"pace RMSE {pace['rmse']:.2f} (bias {pace['bias']:+.2f}); "
            f"margin RMSE {margin['rmse']:.2f}; total RMSE {total['rmse']:.2f}; "
            f"OT Brier {block['overtime']['brier']:.4f}; "
            f"market games {market_n}; missing-market games {block['missing_market']['n']}; "
            f"invalid rotation rate {block['invalid_rotation_rate']}."
        )
    lines.extend(metrics["comparison"]["lines"])
    lines.append(metrics["comparison"]["verdict"])
    return "\n".join(lines)


def _plots(stored, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    levels = ["80", "90", "95"]
    nominal = [0.80, 0.90, 0.95]
    for ax, target in zip(axes, ("pace", "total", "margin"), strict=True):
        width = 0.35
        for offset, name in ((-width / 2, "baseline"), (width / 2, "rotation")):
            coverage = [
                _coverage_value(stored[name], target, level) or 0.0
                for level in levels
            ]
            ax.bar(
                np.arange(len(levels)) + offset,
                coverage,
                width=width,
                label=name,
            )
        ax.plot(np.arange(len(levels)), nominal, color="black", marker="o", linestyle="none")
        ax.set_xticks(np.arange(len(levels)), [f"{level}%" for level in levels])
        ax.set_ylim(0, 1)
        ax.set_title(target)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "coverage.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, frame in stored.items():
        centers, predicted, observed = _reliability(frame["pred_ot"], frame["actual_ot"])
        ax.plot(predicted, observed, marker="o", label=name)
        del centers
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_xlabel("Predicted overtime frequency")
    ax.set_ylabel("Observed overtime frequency")
    ax.set_title("Overtime reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ot_reliability.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, ("baseline", "rotation"), strict=True):
        frame = stored[name]
        observed = frame["actual_home_poss"] - frame["actual_away_poss"]
        ax.hist(observed, bins=30, alpha=0.55, label="Observed", density=True)
        ax.hist(frame["pred_gap_draw"], bins=30, alpha=0.55, label="One draw", density=True)
        ax.set_title(name)
        ax.set_xlabel("Home possessions minus away possessions")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "possession_gap.png", dpi=120)
    plt.close(fig)


def _coverage_value(frame, target, level):
    prefix = {
        "pace": "pred_pace",
        "total": "pred_total",
        "margin": "pred_margin",
    }[target]
    actual = {
        "pace": "actual_poss48",
        "total": "actual_total",
        "margin": "actual_margin",
    }[target]
    low, high = _INTERVALS[level]
    return _coverage(frame[actual], frame[f"{prefix}_{low}"], frame[f"{prefix}_{high}"])["coverage"]


def _reliability(predicted, actual, bins=8):
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.linspace(0, max(predicted.max(), 0.05), bins + 1)
    centers, pred_rate, obs_rate = [], [], []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (predicted >= low) & (predicted <= high if high == edges[-1] else predicted < high)
        if mask.sum() < 5:
            continue
        centers.append(float((low + high) / 2))
        pred_rate.append(float(predicted[mask].mean()))
        obs_rate.append(float(actual[mask].mean()))
    return centers, pred_rate, obs_rate


def _prepare_baseline(games):
    frame = games.copy()
    for name in LINEUP_GAME_FEATURES:
        frame[name] = 0.0
    return frame, list(_BASE_FEATURES)


def _prepare_rotation(games, train_mask):
    frame = games.copy()
    for dest, source in _ROTATION_TO_LINEUP.items():
        values = pd.to_numeric(frame[source], errors="coerce") if source in frame else np.nan
        train_mean = pd.to_numeric(frame.loc[train_mask, source], errors="coerce").mean()
        if not np.isfinite(train_mean):
            train_mean = 0.0
        frame[dest] = values.fillna(train_mean)
    return frame, list(FEATURES)


def _attach_rotation(games, teams):
    columns = [
        "game_id", "team_id", "rotation_pace", "rotation_off_rating",
        "rotation_def_rating", "n_eligible", "rotation_valid", "expected_starters",
    ]
    slim = teams[columns].copy()
    slim["game_id"] = slim["game_id"].astype(str)
    slim["team_id"] = pd.to_numeric(slim["team_id"]).astype("int64")
    out = games.copy()
    out["game_id"] = out["game_id"].astype(str)
    for side, key in (("h", "home_id"), ("a", "away_id")):
        part = slim.rename(columns={
            "team_id": key,
            "rotation_pace": f"rotation_pace_{side}",
            "rotation_off_rating": f"rotation_off_rating_{side}",
            "rotation_def_rating": f"rotation_def_rating_{side}",
            "n_eligible": f"n_eligible_{side}",
            "rotation_valid": f"rotation_valid_{side}",
            "expected_starters": f"expected_starters_{side}",
        })
        out[key] = pd.to_numeric(out[key]).astype("int64")
        out = out.merge(part, on=["game_id", key], how="left")
    return out


def _test_context(players, predicted):
    if players.empty:
        return players
    test_ids = set(predicted["game_id"].astype(str))
    context = players.loc[players["game_id"].astype(str).isin(test_ids)].copy()
    keep = [
        "game_id", "game_date", "team_id", "player_id",
        "projected_reg_minutes", "status", "is_expected_starter",
        "availability_source",
    ]
    return context[keep]


def _attach_pbp(predicted, pbp_dir):
    folder = Path(pbp_dir) if pbp_dir else Path("data/PlayByPlay")
    if not folder.exists():
        return {"used": False, "reason": f"No play-by-play directory at {folder}"}, predicted
    frames = []
    failed = []
    for path in sorted(folder.glob("s*_pbp.parquet")):
        try:
            raw = pd.read_parquet(
                path,
                columns=["gameId", "actionNumber", "period", "scoreHome", "scoreAway"],
            )
        except (OSError, ValueError) as exc:
            failed.append({"path": str(path), "error": str(exc)})
            continue
        frames.append(summarize_pbp(raw))
    if not frames:
        return {"used": False, "failed": failed}, predicted
    summary = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    out = predicted.merge(summary, on="game_id", how="left", suffixes=("", "_pbp"))
    matched = out["final_home"].notna()
    score_ok = matched & (out["final_home"] - out["actual_home_pts"]).abs().le(1) & (
        out["final_away"] - out["actual_away_pts"]
    ).abs().le(1)
    ot_ok = matched & out["n_ot"].eq(out["actual_n_ot"])
    note = {
        "used": True,
        "failed": failed,
        "matched_games": int(matched.sum() // max(out["feature_set"].nunique(), 1)),
        "final_score_agreement": float(score_ok[matched].mean()) if matched.any() else None,
        "overtime_count_agreement": float(ot_ok[matched].mean()) if matched.any() else None,
        "target": (
            "Model scores use full-game points. Possessions per 48 use "
            "play-by-play counts when a game parsed, and the box estimate otherwise. "
            "Play-by-play also checks the final score, the regulation score, and the overtime count."
        ),
    }
    return note, out


def _load_logs(spec):
    from pyarrow.lib import ArrowInvalid

    pattern = os.path.join(spec, "*.parquet") if os.path.isdir(spec) else spec
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files found for {spec!r}")
    frames = []
    failed = []
    for path in paths:
        try:
            frames.append(load_gamelogs(path))
        except (OSError, ValueError, ArrowInvalid) as exc:
            failed.append({"path": path, "error": str(exc)})
            print(f"  skipped unreadable log {path}: {exc}")
    if not frames:
        raise FileNotFoundError(f"No readable gamelogs in {spec!r}")
    return pd.concat(frames, ignore_index=True), failed


def _error_stats(pred, actual, bias=True):
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(actual)
    if not ok.any():
        return {"n": 0, "rmse": None, "mae": None, "bias": None}
    error = pred[ok] - actual[ok]
    out = {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }
    if bias:
        out["bias"] = float(np.mean(error))
    return out


def _mae(pred, actual):
    return _error_stats(pred, actual)["mae"]


def _rmse(pred, actual):
    return _error_stats(pred, actual)["rmse"]


def _coverage(actual, low, high):
    actual = np.asarray(actual, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    ok = np.isfinite(actual) & np.isfinite(low) & np.isfinite(high)
    if not ok.any():
        return {"n": 0, "coverage": None, "width": None}
    return {
        "n": int(ok.sum()),
        "coverage": float(np.mean((actual[ok] >= low[ok]) & (actual[ok] <= high[ok]))),
        "width": float(np.mean(high[ok] - low[ok])),
    }


def _ot_metrics(frame):
    predicted = frame["pred_ot"].to_numpy(float)
    actual = frame["actual_ot"].to_numpy(float)
    return {
        "n": int(len(frame)),
        "brier": float(np.mean((predicted - actual) ** 2)),
        "predicted_frequency": float(np.mean(predicted)),
        "observed_frequency": float(np.mean(actual)),
    }


def _invalid_rate(frame):
    if "rotation_valid_h" not in frame or frame["rotation_valid_h"].isna().all():
        return None
    valid = frame["rotation_valid_h"].fillna(False).astype(bool) & frame[
        "rotation_valid_a"
    ].fillna(False).astype(bool)
    return float(1.0 - valid.mean())


def _game_id(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text
