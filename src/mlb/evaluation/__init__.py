"""Chronological evaluation, baselines, and market comparison."""

from src.mlb.evaluation.backtest import run_backtest
from src.mlb.evaluation.baselines import fit_baselines
from src.mlb.evaluation.folds import chronological_folds
from src.mlb.evaluation.market import compare_market
from src.mlb.evaluation.report import write_daily_report

__all__ = [
    "chronological_folds",
    "compare_market",
    "fit_baselines",
    "run_backtest",
    "write_daily_report",
]
