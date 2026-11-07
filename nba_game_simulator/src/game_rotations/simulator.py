"""Draw roster minutes inside one Layer 1 environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from game_rotations.checks import (
    _as_float,
    _binary,
    _frame,
    _number,
    _one_game,
    _one_team,
    _positive_integer,
    _required,
)
from game_rotations.constants import _SCRIPT_INPUT, _TOL, MODEL_VERSION, PREGAME
from game_rotations.minutes import constrain, scale_mu, sim_gaussian
from game_rotations.model import (
    _check_banks,
    _check_sigma,
    _recent_mean,
    _role_bins,
    _sample_residuals,
)
from game_rotations.participation import draw_availability, draw_script_availability
from game_rotations.roster import _availability_matrix, _roster_keys
from game_rotations.script import _attach_script, _script_arrays
from game_rotations.segments import _segment_minutes, _target_array
from role_context import OT_MIN, REG_MIN


def allocate_rotation(
    simulator: RotationSimulator,
    role: pd.DataFrame,
    environment: pd.DataFrame,
    rng: np.random.Generator,
    *,
    method: str = "state",
    regulation_minutes: float = REG_MIN,
    ot_minutes: float = OT_MIN,
    segments=None,
    availability=None,
    target_minutes=None,
) -> np.ndarray:
    """One roster; environment margin must already use that team's perspective.

    availability is bool[n_simulations, n_players] in roster row order.
    target_minutes is only for segments and can be the corresponding Layer 2
    DataFrame; its index and player columns must match exactly.
    """
    if not isinstance(simulator, RotationSimulator):
        raise ValueError("simulator must be a RotationSimulator")

    _frame(environment, "environment")
    _frame(role, "roster")
    _required(environment, _SCRIPT_INPUT)

    if "is_home" in environment and "is_home" in role:
        env_side = _binary(environment["is_home"], "environment is_home")
        roster_side = _binary(role["is_home"], "roster is_home")

        if np.unique(env_side).size != 1 or np.unique(roster_side).size != 1:
            raise ValueError("one roster and one environment need a single perspective")

        if env_side[0] != roster_side[0]:
            raise ValueError("flip team_margin and is_home to the roster's perspective")

    _one_game(environment, cutoff=simulator.metadata.get("training_cutoff"))

    target_minutes = _target_array(role, environment, target_minutes)

    return simulator.simulate(
        role,
        method=method,
        n=len(environment),
        n_ot=_as_float(environment["n_ot"]),
        margin=_as_float(environment["team_margin"]),
        team_total=5.0 * _as_float(environment["game_minutes"]),
        cap=_as_float(environment["game_minutes"]),
        rng=rng,
        regulation_minutes=regulation_minutes,
        ot_minutes=ot_minutes,
        segments=segments,
        availability=availability,
        target_minutes=target_minutes,
    )


class RotationSimulator:
    """Minutes conditional on a supplied game; shared availability is fixed."""

    def __init__(
        self,
        pre_model=None,
        cond_model=None,
        sigma_pre=None,
        sigma_cond=None,
        alpha=None,
        margin_resid=None,
        p_ot=None,
        fallback_mu=24.0,
        fallback_sd=6.0,
        *,
        residual_pre=None,
        residual_cond=None,
        metadata=None,
    ):
        if any(value is not None for value in (alpha, margin_resid, p_ot)):
            raise ValueError(
                "legacy margin/OT parameters were removed; supply Layer 1 draws"
            )

        self.fb_mu = _number(fallback_mu, "fallback_mu", nonnegative=True)
        self.fb_sd = _number(fallback_sd, "fallback_sd", positive=True)
        self.pre = pre_model
        self.cond = cond_model
        self.sig_pre = _check_sigma(sigma_pre)
        self.sig_cond = _check_sigma(sigma_cond)
        self.res_pre = _check_banks(residual_pre)
        self.res_cond = _check_banks(residual_cond)
        self.metadata = dict(metadata or {})
        self.version = MODEL_VERSION

    def pregame_mu(self, roster: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        _frame(roster, "roster")

        mu = _recent_mean(roster, self.fb_mu)
        known = self._known(roster)

        if self.pre is not None and known.any():
            prediction = np.asarray(
                self.pre.predict(roster.loc[known]),
                dtype=float,
            ).reshape(-1)

            if (
                prediction.shape != (int(known.sum()),)
                or not np.isfinite(prediction).all()
            ):
                raise ValueError("pregame model returned invalid minute predictions")

            mu[known] = prediction

        return np.maximum(mu, 0), known

    def simulate(
        self,
        roster: pd.DataFrame,
        method: str = "state",
        n: int = 2000,
        seed: int = 0,
        *,
        n_ot=None,
        margin=None,
        team_total=None,
        cap=None,
        rng: np.random.Generator | None = None,
        regulation_minutes: float = REG_MIN,
        ot_minutes: float = OT_MIN,
        segments=None,
        availability=None,
        target_minutes=None,
    ) -> np.ndarray:
        _frame(roster, "roster")
        _roster_keys(roster)
        _one_team(roster)

        if method not in {"state", "gaussian", "segments"}:
            raise ValueError("method must be state, gaussian, or segments")

        if (
            segments is not None or target_minutes is not None
        ) and method != "segments":
            raise ValueError("segments and target_minutes require method='segments'")

        if segments is not None and target_minutes is not None:
            raise ValueError("supply either segments or target_minutes")

        _positive_integer(n, "n")
        n = int(n)

        if getattr(self, "version", None) != MODEL_VERSION:
            raise ValueError("unsupported rotation model version; retrain this model")

        if self.metadata:
            for key, value in (
                ("regulation_minutes", regulation_minutes),
                ("ot_minutes", ot_minutes),
            ):
                if key in self.metadata and not np.isclose(
                    self.metadata[key],
                    value,
                    rtol=0,
                    atol=_TOL,
                ):
                    raise ValueError("simulation duration rules differ from training")

        if rng is None:
            rng = np.random.default_rng(seed)

        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy Generator")

        n_ot, ot, margin, total, cap = _script_arrays(
            n,
            n_ot,
            margin,
            team_total,
            cap,
            regulation_minutes=regulation_minutes,
            ot_minutes=ot_minutes,
        )

        active = _availability_matrix(roster, availability, n)

        if np.any(active.sum(axis=1) < 5):
            raise ValueError("every scenario needs at least five available players")

        if method == "segments" and (
            segments is not None or target_minutes is not None
        ):
            return _segment_minutes(
                roster,
                n=n,
                n_ot=n_ot,
                margin=margin,
                cap=cap,
                segments=segments,
                regulation_minutes=regulation_minutes,
                ot_minutes=ot_minutes,
                availability=active,
                target_minutes=target_minutes,
            )

        mu, _ = self.pregame_mu(roster)
        means = np.broadcast_to(mu, (n, len(roster))).copy()
        use_conditional = method in {"state", "segments"} and self.cond is not None

        if use_conditional:
            _required(roster, PREGAME)

            means = self._conditional(
                roster,
                margin,
                n_ot,
                ot,
                cap,
                regulation_minutes=regulation_minutes,
                ot_minutes=ot_minutes,
            )

        if not np.isfinite(means).all():
            raise ValueError("minute model returned nonfinite predictions")

        table = self.sig_cond if use_conditional else self.sig_pre
        spread = self._sigma(np.broadcast_to(mu, means.shape), table)
        bank = self.res_cond if use_conditional else self.res_pre
        # Banks store actual minus the raw prediction. Noise has to land
        # on that prediction; the team budget is applied once after it.
        if method != "gaussian" and bank is not None:
            weights = np.maximum(
                means
                + _sample_residuals(
                    bank,
                    np.broadcast_to(mu, means.shape),
                    rng,
                ),
                0,
            )
        else:
            weights = np.maximum(means, 0)

        self.participation_scale_ = None
        if "close_rate" in roster.columns or "blowout_rate" in roster.columns:
            active = draw_script_availability(roster, margin, active, rng)
            covered = (weights * active).sum(axis=1)
            self.participation_scale_ = np.divide(
                total,
                covered,
                out=np.full(total.shape, np.inf),
                where=covered > _TOL,
            )
        elif "play_rate" in roster.columns:
            active, self.participation_scale_ = draw_availability(
                roster,
                weights,
                active,
                total,
                rng,
            )

        minutes = constrain(
            scale_mu(weights, total, availability=active),
            spread**2,
            total,
            cap,
            availability=active,
        )

        if method == "gaussian" or bank is None:
            minutes = sim_gaussian(
                minutes,
                spread,
                total,
                cap,
                rng,
                availability=active,
            )

        if method == "segments":
            return _segment_minutes(
                roster,
                n=n,
                n_ot=n_ot,
                margin=margin,
                cap=cap,
                segments=None,
                regulation_minutes=regulation_minutes,
                ot_minutes=ot_minutes,
                availability=active,
                target_minutes=minutes,
            )

        return minutes

    def _conditional(
        self,
        roster,
        margin,
        n_ot,
        ot,
        cap,
        *,
        regulation_minutes=REG_MIN,
        ot_minutes=OT_MIN,
    ) -> np.ndarray:
        k = len(roster)
        result = np.empty((len(margin), k))
        batch_size = max(1, 50_000 // k)

        for start in range(0, len(margin), batch_size):
            end = min(start + batch_size, len(margin))
            frame = pd.concat([roster] * (end - start), ignore_index=True)

            frame["team_final_margin"] = np.repeat(margin[start:end], k)
            frame["n_ot"] = np.repeat(n_ot[start:end], k)
            frame["game_minutes"] = np.repeat(cap[start:end], k)

            _attach_script(
                frame,
                regulation_minutes=regulation_minutes,
                ot_minutes=ot_minutes,
            )

            predicted = np.asarray(self.cond.predict(frame), dtype=float).reshape(-1)

            if predicted.size != (end - start) * k or not np.isfinite(predicted).all():
                raise ValueError(
                    "conditional model returned invalid minute predictions"
                )

            result[start:end] = predicted.reshape(end - start, k)

        return result

    def _known(self, roster: pd.DataFrame) -> np.ndarray:
        if "min_mean_10" not in roster:
            return np.zeros(len(roster), dtype=bool)

        values = _as_float(roster["min_mean_10"])
        return np.isfinite(values) & (values >= 0)

    def _sigma(self, mu: np.ndarray, table) -> np.ndarray:
        if table is None:
            return np.full(np.shape(mu), self.fb_sd)

        return np.asarray(table)[_role_bins(mu)]
