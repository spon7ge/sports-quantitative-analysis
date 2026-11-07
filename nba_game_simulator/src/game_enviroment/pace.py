from __future__ import annotations

import numpy as np
from game_enviroment.constants import FEATURES
from scipy import stats


def design(df, team_ids=None, features=None):
    columns = list(FEATURES if features is None else features)
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    x = np.column_stack([
        np.ones(len(df)),
        df[columns].to_numpy(float),
    ])
    if team_ids:
        effects = np.column_stack([
            (
                df.home_id.eq(t).astype(float)
                + df.away_id.eq(t).astype(float)
            ).to_numpy()
            for t in team_ids
        ])
        x = np.column_stack([x, effects])
    return x


def fit_model(name, df, team_ids, lam=0.0, features=None):
    columns = list(FEATURES if features is None else features)
    tids = team_ids if name == "ridge_team" else []
    if name == "league_prior":
        return {
            "name": name,
            "team_ids": [],
            "beta": [],
            "lam": 0.0,
            "features": columns,
        }

    x, y = design(df, tids, columns), df.y_dev.to_numpy(float)
    if tids:
        penalty = np.zeros((len(tids), x.shape[1]))
        penalty[:, -len(tids):] = np.eye(len(tids)) * np.sqrt(lam)
        x = np.vstack([x, penalty])
        y = np.r_[y, np.zeros(len(tids))]

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return {
        "name": name,
        "team_ids": tids,
        "beta": beta.tolist(),
        "lam": float(lam),
        "features": columns,
    }


def predict_dev(model, df):
    if model["name"] == "league_prior":
        return np.zeros(len(df))
    return design(
        df, model["team_ids"], model.get("features")
    ) @ np.asarray(model["beta"])


def _date_split(df, fraction):
    dates = sorted(df.game_date.unique())
    if len(dates) < 2:
        raise ValueError(
            "Need games on multiple dates for chronological splits."
        )
    cutoff = dates[
        min(len(dates) - 1, max(1, int(len(dates) * fraction)))
    ]
    return (
        df[df.game_date < cutoff].copy(),
        df[df.game_date >= cutoff].copy(),
    )


def choose_lambda(
    df, team_ids, grid=(1, 3, 10, 30, 100, 300, 1000),
    features=None,
):
    tr, va = _date_split(df, 0.75)
    return float(min(
        grid,
        key=lambda lam: np.mean(
            (
                va.y_dev.to_numpy()
                - predict_dev(
                    fit_model(
                        "ridge_team", tr, team_ids, lam, features
                    ),
                    va,
                )
            ) ** 2
        ),
    ))


def fit_t(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        raise ValueError(
            "Not enough finite residuals to calibrate uncertainty."
        )
    if np.std(values) < 1e-8:
        return {
            "nu": 30.0,
            "scale": float(np.sqrt(np.mean(values ** 2))),
        }

    raw_nu, _, scale = stats.t.fit(values, floc=0.0)
    nu = float(np.clip(raw_nu, 2.5, 100.0))
    if nu != raw_nu:
        _, _, scale = stats.t.fit(values, f0=nu, floc=0.0)
    return {"nu": nu, "scale": float(scale)}
