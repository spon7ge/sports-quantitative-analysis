from __future__ import annotations

import numpy as np
from role_context import (
    OPP_HISTORY,
    PLAYER_HISTORY,
    SCRIPT,
    SLATE_KNOWN,
    TEAM_HISTORY,
)

MODEL_VERSION = 2
MU_BINS = np.array([0.0, 8.0, 16.0, 24.0, 30.0, np.inf])
PREGAME = tuple(
    dict.fromkeys((*PLAYER_HISTORY, *TEAM_HISTORY, *OPP_HISTORY, *SLATE_KNOWN))
)
CONDITIONAL = tuple(dict.fromkeys((*PREGAME, *SCRIPT)))
_INTERACTIONS = (
    "min_mean_x_margin",
    "min_mean_x_abs_margin",
    "min_mean_x_blowout",
    "min_mean_x_n_ot",
    "starter_x_blowout",
    "starter_x_margin",
)
_FEATURES = (*CONDITIONAL, *_INTERACTIONS)
_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
_SIGMA_FLOOR = 0.05
_SCRIPT_INPUT = ("team_margin", "n_ot", "game_minutes")
_TOL = 1e-6
# Players under this played-minute average can sit. Measured appearance
# rates from 2019-10-22 through 2025-04-13 rise from about 0.25 below
# 4 minutes to about 0.51 at 8–12, so the draw stops at a rotation role.
LOW_MINUTE_CUTOFF = 12.0
APPEARANCE_WINDOW = 10
PLAY_PRIOR_STRENGTH = 4.0
# Floor time in a non-blowout counts as a close appearance. A script that
# draws fewer than this many players adds the next most likely names.
CLOSE_MINUTES = 3.0
MIN_ACTIVE = 7
_PRIOR_MINUTES = np.array([0.0, 2.5, 6.0, 10.0, 12.0])
_PRIOR_RATE = np.array([0.20, 0.25, 0.37, 0.51, 0.58])
