import numpy as np
import pandas as pd
import pytest

from src.mlb.evaluation.strikeout_walk_forward import walk_blocks
from src.mlb.evaluation.walk_forward_snapshot import SnapshotStore
from src.mlb.pipeline.walk_forward_features import NB_RATE_FEATURES

KEY = ["pitcher_id", "game_pk"]


def _start(rng, game_pk, date, hour=17):
    bf = int(rng.integers(14, 32))
    return {
        "pitcher_id": 1,
        "game_pk": game_pk,
        "game_date": date,
        "scheduled_start_utc": pd.Timestamp(f"{date} {hour:02d}:05", tz="UTC"),
        "season": int(date[:4]),
        "is_home": bool(rng.integers(0, 2)),
        "opponent_team_id": 100,
        "strikeouts": int(rng.integers(1, max(2, bf // 3))),
        "batters_faced": bf,
        "outs": int(rng.integers(9, 21)),
        "pitches": int(rng.integers(60, 110)),
    }


def _starts(postseason=False):
    rng = np.random.default_rng(11)
    rows = []
    pk = 5000
    for day in range(1, 31):
        rows.append(_start(rng, pk, f"2018-04-{day:02d}"))
        pk += 1
    if postseason:
        rows.append(_start(rng, 9999, "2018-10-05"))
    for day in (1, 4, 11, 16):
        rows.append(_start(rng, pk, f"2019-04-{day:02d}"))
        pk += 1
    for day in (1, 3, 10):
        rows.append(_start(rng, pk, f"2025-04-{day:02d}"))
        pk += 1
    return pd.DataFrame(rows)


CALENDAR = pd.DataFrame(
    [
        {"season": 2018, "regular_season_open_date": "2018-03-29", "regular_season_close_date": "2018-09-30"},
        {"season": 2019, "regular_season_open_date": "2019-03-28", "regular_season_close_date": "2019-09-29"},
        {"season": 2025, "regular_season_open_date": "2025-03-27", "regular_season_close_date": "2025-09-28"},
    ]
)


def _walk(starts, through_season=2025):
    return walk_blocks(
        starts,
        CALENDAR,
        m=50,
        kappa=3,
        rest_cap=30,
        through_season=through_season,
        score_strikeouts=False,
    )


def _row(frame, game_pk):
    rows = frame.loc[frame["game_pk"] == game_pk]
    assert len(rows) == 1
    return rows.iloc[0]


def _pk_on(starts, date):
    pks = starts.loc[starts["game_date"] == date, "game_pk"]
    assert len(pks) == 1
    return int(pks.iloc[0])


def test_postseason_start_does_not_feed_the_next_regular_season_start():
    without = _walk(_starts(postseason=False), through_season=2019)
    with_post = _walk(_starts(postseason=True), through_season=2019)
    assert 9999 not in set(with_post["game_pk"])
    next_pk = _pk_on(_starts(), "2019-04-01")
    before = _row(without, next_pk)
    after = _row(with_post, next_pk)
    assert after["pitcher_prior_regular_starts"] == before["pitcher_prior_regular_starts"]
    assert after["league_k_per_bf_prior"] == before["league_k_per_bf_prior"]


def test_strikeout_features_ignore_the_same_start_workload_and_result():
    for column in ("batters_faced", "outs", "pitches"):
        assert column not in NB_RATE_FEATURES
    starts = _starts()
    target = _pk_on(starts, "2019-04-04")
    baseline = _row(_walk(starts), target)

    mutated = starts.copy()
    hit = mutated["game_pk"] == target
    mutated.loc[hit, "batters_faced"] += 10
    mutated.loc[hit, "outs"] += 6
    mutated.loc[hit, "pitches"] += 40
    mutated.loc[hit, "strikeouts"] += 5
    changed = _row(_walk(mutated), target)

    for column in NB_RATE_FEATURES:
        assert changed[column] == baseline[column], column
    assert changed["predicted_bf_oof"] == baseline["predicted_bf_oof"]


def test_second_snapshot_write_of_the_same_walk_is_rejected():
    frame = _walk(_starts())
    store = SnapshotStore()
    store.write(frame)
    with pytest.raises(ValueError):
        store.write(frame)


def test_first_2025_block_prior_excludes_its_own_strikeouts():
    starts = _starts()
    first = _walk(starts)
    block_2025 = first.loc[first["season"] == 2025]
    block_label = block_2025["walk_forward_block"].min()
    block = first.loc[first["walk_forward_block"] == block_label]
    block_open = block["game_date"].min()

    earlier = first.loc[first["game_date"] < block_open]
    expected = earlier["strikeouts"].sum() / earlier["batters_faced"].sum()
    np.testing.assert_allclose(block["league_k_per_bf_prior"], expected)
    including = (earlier["strikeouts"].sum() + block["strikeouts"].sum()) / (
        earlier["batters_faced"].sum() + block["batters_faced"].sum()
    )
    assert not np.isclose(including, expected)

    mutated = starts.copy()
    target = int(block["game_pk"].iloc[0])
    mutated.loc[mutated["game_pk"] == target, "strikeouts"] += 7
    second = _walk(mutated)
    before = block.set_index(KEY)["league_k_per_bf_prior"].sort_index()
    after = (
        second.loc[second["walk_forward_block"] == block_label]
        .set_index(KEY)["league_k_per_bf_prior"]
        .sort_index()
    )
    pd.testing.assert_series_equal(before, after)


def test_earlier_date_in_a_block_feeds_the_later_date_but_not_the_league_prior():
    starts = _starts()
    frame = _walk(starts)
    early_pk = _pk_on(starts, "2019-04-01")
    late_pk = _pk_on(starts, "2019-04-04")
    early = _row(frame, early_pk)
    late = _row(frame, late_pk)
    assert early["walk_forward_block"] == late["walk_forward_block"]
    assert late["pitcher_prior_regular_starts"] > early["pitcher_prior_regular_starts"]
    assert late["league_k_per_bf_prior"] == early["league_k_per_bf_prior"]
    assert np.isfinite(early["predicted_bf_oof"]) and np.isfinite(late["predicted_bf_oof"])

    mutated = starts.copy()
    mutated.loc[mutated["game_pk"] == early_pk, "batters_faced"] += 12
    changed = _walk(mutated)
    assert _row(changed, early_pk)["predicted_bf_oof"] == early["predicted_bf_oof"]
    assert _row(changed, late_pk)["predicted_bf_oof"] != late["predicted_bf_oof"]
    assert _row(changed, late_pk)["league_k_per_bf_prior"] == late["league_k_per_bf_prior"]
