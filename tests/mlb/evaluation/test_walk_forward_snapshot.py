import pandas as pd
import pytest

from src.mlb.evaluation.walk_forward_snapshot import SnapshotStore


def test_duplicate_start_in_one_write_raises():
    store = SnapshotStore()
    frame = pd.DataFrame(
        [
            {"pitcher_id": 1, "game_pk": 10, "predicted_bf_oof": 21.0},
            {"pitcher_id": 1, "game_pk": 10, "predicted_bf_oof": 22.0},
        ]
    )
    with pytest.raises(ValueError, match="game_pk="):
        store.write(frame)
    assert store.frame().empty


def test_second_write_of_the_same_start_raises():
    store = SnapshotStore()
    row = pd.DataFrame([{"pitcher_id": 1, "game_pk": 10, "predicted_bf_oof": 21.0}])
    store.write(row)
    with pytest.raises(ValueError, match="game_pk=10"):
        store.write(row)


def test_store_returns_the_original_feature_value():
    store = SnapshotStore()
    store.write(pd.DataFrame([{"pitcher_id": 1, "game_pk": 10, "league_k_per_bf_prior": 0.22}]))
    assert store.frame().iloc[0]["league_k_per_bf_prior"] == 0.22
