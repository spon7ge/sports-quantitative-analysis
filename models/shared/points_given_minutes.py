"""Points given realized minutes, and its link to the minutes PIT.

Points are NegBin (NB2) with

    log mu    = log_kappa + beta * log m + beta2 * log(m / M0)**2 + log r_hat + gamma * z
    log alpha = log_alpha + alpha_minutes * log(m / M0) + alpha_cv * log(cv)

``m`` is realized minutes, ``r_hat`` and ``cv`` the rate sampler's mean and
coefficient of variation, ``z`` an optional minutes-surprise covariate
(``Phi^-1(u_m)``), and ``M0`` a fixed centring constant. Variance is
``mu + alpha * mu**2``.

A bivariate copula on ``(u_m, v)`` couples the minutes PIT with the
randomized PIT of points under this model. The conditional points PMF
for a minutes draw ``u`` is ``h(F(k) | u) - h(F(k - 1) | u)``, where
``h`` is the copula's ``hfunc1``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy import optimize, stats

RATE_FLOOR = 1e-4
CV_FLOOR = 1e-3
MINUTES_CENTER = 24.0
_H_EPS = 1e-12


@dataclass(frozen=True)
class CountModel:
    log_kappa: float
    beta: float
    log_alpha: float
    beta2: float = 0.0
    alpha_minutes: float = 0.0
    alpha_cv: float = 0.0
    gamma: float = 0.0

    def mean(self, minutes, rate_hat, surprise=None) -> np.ndarray:
        m = np.asarray(minutes, dtype=float)
        r = np.maximum(np.asarray(rate_hat, dtype=float), RATE_FLOOR)
        centred = np.log(m / MINUTES_CENTER)
        log_mu = self.log_kappa + self.beta * np.log(m) + self.beta2 * centred**2 + np.log(r)
        if surprise is not None:
            log_mu = log_mu + self.gamma * np.asarray(surprise, dtype=float)
        return np.exp(log_mu)

    def alpha(self, minutes, rate_cv=None) -> np.ndarray:
        centred = np.log(np.asarray(minutes, dtype=float) / MINUTES_CENTER)
        log_alpha = self.log_alpha + self.alpha_minutes * centred
        if rate_cv is not None:
            cv = np.maximum(np.asarray(rate_cv, dtype=float), CV_FLOOR)
            log_alpha = log_alpha + self.alpha_cv * np.log(cv)
        return np.exp(log_alpha)

    def to_dict(self) -> dict:
        return {**asdict(self), "minutes_center": MINUTES_CENTER}


def _nb_args(mu, alpha):
    size = 1.0 / np.asarray(alpha, dtype=float)
    return size, size / (size + np.asarray(mu, dtype=float))


def nb_cdf(k, mu, alpha) -> np.ndarray:
    size, p = _nb_args(mu, alpha)
    return stats.nbinom.cdf(k, size, p)


def nb_logpmf(k, mu, alpha) -> np.ndarray:
    size, p = _nb_args(mu, alpha)
    return stats.nbinom.logpmf(k, size, p)


def fit_count_model(
    minutes,
    points,
    rate_hat,
    *,
    rate_cv=None,
    surprise=None,
    flexible: bool = True,
) -> CountModel:
    """Maximum likelihood.

    ``flexible`` adds ``beta2`` and ``alpha_minutes``. ``alpha_cv`` is fit
    when ``rate_cv`` is given, and ``gamma`` when ``surprise`` is given.
    """
    m = np.asarray(minutes, dtype=float)
    y = np.asarray(points, dtype=float)
    r = np.asarray(rate_hat, dtype=float)
    if np.any(m <= 0) or not np.isfinite(m).all():
        raise ValueError("minutes must be positive")
    if np.any(y < 0) or np.any(np.abs(y - np.rint(y)) > 1e-9):
        raise ValueError("points must be non-negative integers")
    cv = None if rate_cv is None else np.asarray(rate_cv, dtype=float)
    z = None if surprise is None else np.asarray(surprise, dtype=float)

    free = ["log_kappa", "beta", "log_alpha"]
    if flexible:
        free += ["beta2", "alpha_minutes"]
    if cv is not None:
        free.append("alpha_cv")
    if z is not None:
        free.append("gamma")
    start = {"log_kappa": 0.0, "beta": 1.0, "log_alpha": np.log(0.1)}

    def unpack(theta) -> CountModel:
        values = {name: 0.0 for name in (f.name for f in fields(CountModel))}
        values.update(dict(zip(free, (float(t) for t in theta), strict=True)))
        return CountModel(**values)

    def nll(theta) -> float:
        model = unpack(theta)
        mu = model.mean(m, r, z)
        return -float(np.sum(nb_logpmf(y, mu, model.alpha(m, cv))))

    result = optimize.minimize(nll, [start.get(name, 0.0) for name in free], method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"count model fit failed: {result.message}")
    return unpack(result.x)


def count_pit(points, mu, alpha, atom_u) -> np.ndarray:
    """Randomized PIT ``F(y - 1) + w * p(y)`` with caller-supplied ``w``."""
    y = np.asarray(points, dtype=float)
    w = np.asarray(atom_u, dtype=float)
    left = nb_cdf(y - 1, mu, alpha)
    right = nb_cdf(y, mu, alpha)
    return left + w * (right - left)


def corner_lifts(u, v, levels) -> list[dict]:
    """Joint exceedance counts and lift over independence in all four corners."""
    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    n = a.shape[0]
    rows = []
    for q in levels:
        masks = {
            "low_u_low_v": (a < q) & (b < q),
            "low_u_high_v": (a < q) & (b > 1 - q),
            "high_u_low_v": (a > 1 - q) & (b < q),
            "high_u_high_v": (a > 1 - q) & (b > 1 - q),
        }
        for corner, mask in masks.items():
            count = int(mask.sum())
            rows.append(
                {
                    "q": float(q),
                    "corner": corner,
                    "count": count,
                    "expected_indep": float(n * q * q),
                    "lift": count / (n * q * q),
                }
            )
    return rows


def copula_corner_lifts(bicop, levels) -> list[dict]:
    """Fitted joint exceedance lift in all four corners from the copula CDF."""
    rows = []
    for q in levels:
        c = bicop.cdf(np.array([[q, q], [q, 1 - q], [1 - q, q], [1 - q, 1 - q]]))
        probs = {
            "low_u_low_v": c[0],
            "low_u_high_v": q - c[1],
            "high_u_low_v": q - c[2],
            "high_u_high_v": 1 - 2 * (1 - q) + c[3],
        }
        for corner, prob in probs.items():
            rows.append({"q": float(q), "corner": corner, "lift": float(prob) / (q * q)})
    return rows


def conditional_points_pmf(
    minutes_at_u,
    u_grid,
    rate_hat,
    model: CountModel,
    *,
    kmax: int,
    rate_cv=None,
    bicop=None,
    use_surprise: bool = False,
) -> np.ndarray:
    """Points PMF per row, averaging over a quadrature grid of minutes PITs.

    ``minutes_at_u`` is ``(rows, J)``: each row's minutes quantile at
    ``u_grid``. With ``bicop`` the points PIT is drawn from the copula's
    conditional given ``u``; without it the two are independent. The last
    bin holds the mass above ``kmax``.
    """
    minutes = np.maximum(np.asarray(minutes_at_u, dtype=float), 1e-6)
    u = np.asarray(u_grid, dtype=float)
    n_rows, n_grid = minutes.shape
    if u.shape != (n_grid,):
        raise ValueError("u_grid")
    z = stats.norm.ppf(u)[None, :] if use_surprise else None
    mu = model.mean(minutes, np.asarray(rate_hat, dtype=float)[:, None], z)
    cv = None if rate_cv is None else np.asarray(rate_cv, dtype=float)[:, None]
    alpha = model.alpha(minutes, cv)
    ks = np.arange(kmax, dtype=float)
    cdf = nb_cdf(ks[None, None, :], mu[:, :, None], alpha[:, :, None])
    if bicop is not None:
        flat_v = np.clip(cdf.reshape(-1), _H_EPS, 1 - _H_EPS)
        flat_u = np.broadcast_to(u[None, :, None], cdf.shape).reshape(-1)
        cdf = bicop.hfunc1(np.column_stack([flat_u, flat_v])).reshape(cdf.shape)
    cdf = np.concatenate([cdf, np.ones((n_rows, n_grid, 1))], axis=2)
    pmf = np.diff(cdf, axis=2, prepend=0.0)
    pmf = np.clip(pmf, 0.0, None).mean(axis=1)
    return pmf / pmf.sum(axis=1, keepdims=True)
