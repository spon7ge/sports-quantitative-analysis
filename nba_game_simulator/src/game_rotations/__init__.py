"""Conditional NBA minutes and feasible five-player rotation segments.

Use existing Layer 1 draws. To turn Layer 2 minutes into segments, supply
target_minutes; this preserves those draws instead of sampling minutes again.

segments_from_minutes packs those totals into a feasible five. The play-by-play
templates arrange the same totals into a historical pattern.
"""

from __future__ import annotations

from game_rotations.constants import CONDITIONAL, MODEL_VERSION, MU_BINS, PREGAME
from game_rotations.history import live_minutes, training_minutes
from game_rotations.minutes import constrain, draw_minutes, scale_mu, sim_gaussian
from game_rotations.participation import (
    play_prior,
    play_rates,
    script_rates,
    shrink_play_rate,
)
from game_rotations.pbp import (
    PBPRotationModel,
    Reconstruction,
    fit_pbp_rotation,
    learned_segments_from_minutes,
    reconstruct_rotation_segments,
    script_exposure,
)
from game_rotations.segments import (
    build_rotation_segments,
    invalid_segment_count,
    segments_from_minutes,
)
from game_rotations.simulator import RotationSimulator, allocate_rotation
from game_rotations.train import fit_rotation

__all__ = [
    "CONDITIONAL",
    "MODEL_VERSION",
    "MU_BINS",
    "PREGAME",
    "PBPRotationModel",
    "Reconstruction",
    "RotationSimulator",
    "allocate_rotation",
    "build_rotation_segments",
    "constrain",
    "draw_minutes",
    "fit_pbp_rotation",
    "fit_rotation",
    "invalid_segment_count",
    "learned_segments_from_minutes",
    "live_minutes",
    "play_prior",
    "play_rates",
    "script_exposure",
    "script_rates",
    "shrink_play_rate",
    "reconstruct_rotation_segments",
    "scale_mu",
    "segments_from_minutes",
    "sim_gaussian",
    "training_minutes",
]
