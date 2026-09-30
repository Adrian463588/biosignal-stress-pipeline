from __future__ import annotations

import json
from pathlib import Path

import pytest

from PipelinePython.common.live_ppg_capture_verifier import (
    load_captured_live_session,
    verify_live_ppg_capture,
    watch_payload_sha256,
)


def _batch() -> dict[str, object]:
    return {
        "schemaVersion": 2,
        "source": "acknowledged_watch_transport",
        "channel": "ppg_green",
        "sessionId": "session-actual",
        "sessionNonce": "nonce-actual",
        "protocolVersion": 1,
        "peerNodeId": "watch-node",
        "generation": 1,
        "batchId": "batch-1",
        "sequenceNumber": 1,
        "samplingRateHz": 25,
        "elapsedStartMillis": 0,
        "elapsedEndMillis": 40,
        "timestampsMillis": [1_000, 1_040],
        "rawValues": [111, 112],
    }


def _write_capture(root: Path) -> None:
    root.mkdir()
    batch = _batch()
    batch["payloadSha256"] = watch_payload_sha256(batch)
    batch["encodedBatchSha256"] = "not_used_by_python_oracle"
    (root / "ppg_green_1.json").write_text(json.dumps(batch), encoding="utf-8")
    (root / "window_evidence.json").write_text(
        json.dumps([{"channel": "ppg_green", "windowId": "window-1"}]),
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": 2,
        "sessionId": "session-actual",
        "sessionNonce": "nonce-actual",
        "source": "acknowledged_watch_transport",
        "verificationStatus": "PENDING_PYTHON_ORACLE",
        "actualSensorAcquisition": False,
        "sensorPathParityStatus": "PENDING_PYTHON_ORACLE",
        "batchFilesByChannel": {"ppg_green": 1},
        "droppedWriteCount": 0,
        "inferenceTraceFile": None,
        "windowEvidenceFile": "window_evidence.json",
        "finalized": True,
    }
    (root / "capture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_acknowledged_capture_requires_matching_payload_checksum(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture"
    _write_capture(capture)

    loaded = load_captured_live_session(capture)

    assert loaded.batch_timestamps_by_channel["ppg_green"] == frozenset({1_000, 1_040})
    assert loaded.batch_raw_values_by_channel_timestamp["ppg_green"] == {
        1_000: 111,
        1_040: 112,
    }
    batch_path = capture / "ppg_green_1.json"
    tampered = json.loads(batch_path.read_text(encoding="utf-8"))
    tampered["rawValues"] = [111, 999]
    batch_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="capture_batch_payload_checksum_mismatch"):
        load_captured_live_session(capture)


def test_capture_rejects_conflicting_duplicate_timestamp(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _write_capture(capture)
    batch = _batch()
    batch["batchId"] = "batch-2"
    batch["sequenceNumber"] = 2
    batch["rawValues"] = [999, 112]
    batch["payloadSha256"] = watch_payload_sha256(batch)
    (capture / "ppg_green_2.json").write_text(json.dumps(batch), encoding="utf-8")
    manifest = json.loads(
        (capture / "capture_manifest.json").read_text(encoding="utf-8")
    )
    manifest["batchFilesByChannel"] = {"ppg_green": 2}
    (capture / "capture_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="capture_timestamp_raw_value_conflict"):
        load_captured_live_session(capture)


def test_failed_capture_never_writes_sensor_parity_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "live_ppg_parity_receipt.json"

    report = verify_live_ppg_capture(
        tmp_path,
        "20260916T132736Z",
        tmp_path / "missing-capture",
        receipt_path=receipt,
    )

    assert report["status"] == "NO_GO"
    assert report["actual_sensor_acquisition"] is False
    assert not receipt.exists()
