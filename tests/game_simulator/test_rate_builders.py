import numpy as np
import pandas as pd
import pytest

from rate_builders import build_player_rates_from_pbp


def test_build_player_rates_from_synthetic_pbp():
    rows = [
        dict(
            teamTricode="BOS",
            personId=1,
            isFieldGoal=1,
            shotValue=3,
            shotResult="Made",
            actionType="Made Shot",
            subType="Jump Shot",
        ),
        dict(
            teamTricode="BOS",
            personId=1,
            isFieldGoal=1,
            shotValue=2,
            shotResult="Missed",
            actionType="Missed Shot",
            subType="Layup",
        ),
        dict(
            teamTricode="BOS",
            personId=2,
            isFieldGoal=1,
            shotValue=2,
            shotResult="Missed",
            actionType="Missed Shot",
            subType="Jump Shot",
        ),
    ]
    rates = build_player_rates_from_pbp(pd.DataFrame(rows), team_tricode="BOS")
    assert set(rates["player_ids"].tolist()) == {1, 2}
    assert pytest.approx(float(rates["base_weights"].sum()), abs=1e-9) == 1.0
    player_1 = int(np.flatnonzero(rates["player_ids"] == 1)[0])
    assert rates["p_3pa"][player_1] == pytest.approx(0.5)
    assert rates["p_3p"][player_1] == pytest.approx(1.0)
    assert rates["p_2p"][player_1] == pytest.approx(0.0)


def test_build_player_rates_missing_columns_raises():
    df = pd.DataFrame([dict(teamTricode="BOS", personId=1)])
    with pytest.raises(ValueError, match="required columns"):
        build_player_rates_from_pbp(df, team_tricode="BOS")
