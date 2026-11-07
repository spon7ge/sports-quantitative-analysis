"""Layer 6: one box score from a single pass through layers 1–5.

Layer 1 draws the shared game. Layer 2 draws each player's total minutes
from that game. Layer 3 arranges those minutes into five-player segments.
Layer 4 turns possessions, minutes, and segments into shots, free throws,
turnovers, points, assists, and offensive rebounds, matched to the Layer 1
score. Layer 5 adds defensive rebounds, steals, and blocks from those misses
and turnovers, and reuses Layer 4's assists and offensive rebounds.

``assemble_box_score`` stacks those arrays and checks that they describe one
coherent game. It does not draw anything again. Row ``k`` of every array is
the same simulated game, so a joint prop is the share of rows where every
leg wins.
"""

from __future__ import annotations

from game_boxscore.assemble import assemble_box_score
from game_boxscore.constants import LAYER_MODULES, SIDES
from game_boxscore.props import (
    format_outcome,
    player_summary,
    portfolio_risk,
    price_over,
    prop_probability,
    stat_values,
)
from game_boxscore.simulate import simulate_box_scores
from game_boxscore.types import (
    BoxScoreSet,
    OverPrice,
    PipelineModels,
    PortfolioRisk,
    Prop,
    PropProbability,
    TeamBoxScore,
    TeamInputs,
)

__all__ = [
    "LAYER_MODULES",
    "SIDES",
    "BoxScoreSet",
    "OverPrice",
    "PipelineModels",
    "PortfolioRisk",
    "Prop",
    "PropProbability",
    "TeamBoxScore",
    "TeamInputs",
    "assemble_box_score",
    "format_outcome",
    "player_summary",
    "portfolio_risk",
    "price_over",
    "prop_probability",
    "simulate_box_scores",
    "stat_values",
]
