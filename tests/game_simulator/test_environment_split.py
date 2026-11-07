"""Layer 1 season roles: earlier files train, one season validates, one holds out."""

from __future__ import annotations

import pandas as pd
import pytest
from game_enviroment.train import _deploy_frame, _season_split


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season_year": [
                "2019-20",
                "2024-25",
                "2025-26",
                "2025-26",
                "2026-27",
            ],
            "game_id": range(5),
        }
    )


def test_named_seasons_leave_every_earlier_file_in_training():
    train, validation, holdout = _season_split(_frame(), "2025-26", "2026-27")
    assert set(train.season_year) == {"2019-20", "2024-25"}
    assert len(validation) == 2
    assert validation.season_year.eq("2025-26").all()
    assert len(holdout) == 1
    assert holdout.season_year.eq("2026-27").all()


def test_saved_fit_keeps_validation_and_leaves_out_the_holdout():
    deployed = _deploy_frame(_frame(), "2026-27")
    assert set(deployed.season_year) == {"2019-20", "2024-25", "2025-26"}


def test_missing_season_is_rejected():
    with pytest.raises(ValueError, match="2027-28"):
        _season_split(_frame(), "2025-26", "2027-28")
