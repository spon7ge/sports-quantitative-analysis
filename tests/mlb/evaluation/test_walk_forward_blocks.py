import pandas as pd

from src.mlb.evaluation.walk_forward_blocks import assign_walk_forward_blocks


def test_blocks_reset_each_season_and_drop_postseason():
    calendar = pd.DataFrame(
        {
            "season": [2019, 2020],
            "regular_season_open_date": ["2019-03-28", "2020-07-23"],
            "regular_season_close_date": ["2019-04-30", "2020-09-27"],
        }
    )
    starts = pd.DataFrame(
        {
            "pitcher_id": [1, 1, 1, 1],
            "game_pk": [10, 11, 12, 13],
            "season": [2019, 2019, 2019, 2020],
            "game_date": ["2019-03-28", "2019-04-25", "2019-05-02", "2020-07-23"],
        }
    )
    out = assign_walk_forward_blocks(starts, calendar)
    assert out.loc[0, "model_eligible"]
    assert out.loc[0, "walk_forward_block"] == "2019-0"
    assert out.loc[1, "walk_forward_block"] == "2019-1"
    assert not out.loc[2, "model_eligible"]
    assert out.loc[3, "walk_forward_block"] == "2020-0"
    assert out.loc[out["game_date"] == "2019-03-28", "walk_forward_block"].nunique() == 1
