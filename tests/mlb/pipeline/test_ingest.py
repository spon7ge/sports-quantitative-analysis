"""Ingest and snapshot tests — never hit the network."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
from src.mlb.config import load_config
from src.mlb.pipeline.http import HttpClient
from src.mlb.pipeline.ingest import (
    ingest_chadwick,
    ingest_schedule,
    ingest_statcast,
    snapshot_raw,
)
from src.mlb.schemas import ID_MAP_COLUMNS, PITCH_EVENT_COLUMNS


def _tmp_config(tmp_path):
    cfg = load_config()
    return replace(cfg, data_dir=tmp_path / "data", artifact_dir=tmp_path / "artifacts")


def test_snapshot_raw_writes_payload_and_hash(tmp_path) -> None:
    config = _tmp_config(tmp_path)
    payload = b"hello-statcast"
    params = {"start_date": "2017-04-01"}
    fetched_at = datetime(2017, 4, 1, 12, tzinfo=UTC)
    snapshot_id = snapshot_raw(config, "statcast", params, payload, fetched_at)

    expected_id = hashlib.sha256(
        b"statcast" + json.dumps(params, sort_keys=True).encode("utf-8") + payload
    ).hexdigest()[:16]
    assert snapshot_id == expected_id
    payload_path = config.raw_dir / "statcast" / snapshot_id
    assert payload_path.read_bytes() == payload
    expected_hash = hashlib.sha256(payload).hexdigest()
    snapshots = pd.read_parquet(config.table_dir / "raw_snapshots" / "part-0.parquet")
    assert snapshots["payload_hash"].iloc[0] == expected_hash
    assert snapshots["snapshot_id"].iloc[0] == snapshot_id


def test_ingest_statcast_loads_fixture_when_http_is_none(tmp_path) -> None:
    config = _tmp_config(tmp_path)
    pitches = ingest_statcast(config, start_date="2017-04-01", end_date="2017-04-08")
    for column in PITCH_EVENT_COLUMNS:
        assert column in pitches.columns
    assert len(pitches) > 0
    written = list((config.raw_dir / "statcast").iterdir())
    assert written


def test_ingest_statcast_uses_injected_http(tmp_path) -> None:
    config = _tmp_config(tmp_path)
    fixture = (config.fixture_dir / "raw" / "statcast_sample.csv").read_bytes()
    calls: list[tuple[str, dict | None]] = []

    def http(url: str, params: dict | None) -> bytes:
        calls.append((url, params))
        return fixture

    pitches = ingest_statcast(
        config, start_date="2017-04-01", end_date="2017-04-08", http=http
    )
    assert calls
    assert "statcast_search" in calls[0][0]
    assert len(pitches) > 0


def test_ingest_schedule_loads_fixture(tmp_path) -> None:
    config = _tmp_config(tmp_path)
    versions = ingest_schedule(config, game_date="2017-04-07")
    assert not versions.empty
    assert "game_pk" in versions.columns
    assert versions["game_pk"].nunique() >= 1


def test_ingest_chadwick_maps_mlbam_to_mlb_id(tmp_path) -> None:
    config = _tmp_config(tmp_path)
    people = ingest_chadwick(config)
    for column in ID_MAP_COLUMNS:
        assert column in people.columns
    assert (people["mlb_id"] == people["key_mlbam"]).all()
    assert 111001 in set(people["mlb_id"].astype(int))


def test_http_client_sleeps_and_sends_user_agent(tmp_path) -> None:
    config = replace(_tmp_config(tmp_path), rate_limit_seconds=0.0)
    client = HttpClient(config)
    mock_response = MagicMock()
    mock_response.read.return_value = b"ok"
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False
    with (
        patch("src.mlb.pipeline.http.time.sleep") as sleep,
        patch("src.mlb.pipeline.http.urlopen", return_value=mock_response) as urlopen,
    ):
        body = client("https://example.test/csv", {"a": "1"})
    assert body == b"ok"
    sleep.assert_called_once_with(0.0)
    request = urlopen.call_args[0][0]
    assert (
        config.user_agent in request.headers.values()
        or request.get_header("User-agent") == config.user_agent
    )
