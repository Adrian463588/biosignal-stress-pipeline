from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

from PipelinePython.common.live_ppg_capture_verifier import (
    CapturedLiveSession,
    _canonical_cleaned_core,
    _detect,
    _load_run_artifacts,
    load_captured_live_session,
    verify_live_ppg_capture,
    watch_payload_sha256,
)
from PipelinePython.common.preprocessing import SignalFilter


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


def _algorithm_capture(tmp_path: Path) -> tuple[CapturedLiveSession, dict]:
    # Non-physiological numeric ramp for filter/coverage tests only, not device evidence.
    times = np.arange(6000, dtype=np.int64) * 40 + 30_000
    raw = {int(time): index for index, time in enumerate(times)}
    windows = tuple(
        {
            "channel": "ppg_green",
            "timestampsMillis": times[start : start + 4500].tolist(),
            "filteredValues": [0.0] * 4500,
        }
        for start in range(0, 1501, 250)
    )
    capture = CapturedLiveSession(
        capture_directory=tmp_path,
        manifest={},
        manifest_sha256="",
        batch_timestamps_by_channel={"ppg_green": frozenset(raw)},
        batch_raw_values_by_channel_timestamp={"ppg_green": raw},
        batch_ids_by_channel_timestamp={},
        batch_checksums_by_channel_timestamp={},
        recording_rates_by_channel={"ppg_green": 25.0},
        batch_elapsed_by_channel_timestamp={
            "ppg_green": {int(time): int(time) for time in times}
        },
        window_evidence=windows,
    )
    artifacts = {
        "ppg_green": {
            "manifest": {
                "window_seconds": 180,
                "stride_seconds": 10,
                "filter_low_cut_hz": 0.5,
                "filter_high_cut_hz": 5.0,
                "filter_order": 4,
            }
        }
    }
    return capture, artifacts


def test_filter_is_recomputed_from_raw_not_supplied_filtered_values(
    tmp_path: Path,
) -> None:
    capture, artifacts = _algorithm_capture(tmp_path)

    cleaned = _canonical_cleaned_core(capture, artifacts)["ppg_green"]
    expected = SignalFilter.bandpass(np.arange(6000, dtype=float), 25.0, 0.5, 5.0, 4)

    np.testing.assert_array_equal(list(cleaned.values()), expected)
    assert any(value != 0.0 for value in cleaned.values())


def test_partial_windows_cannot_claim_full_core_parity(tmp_path: Path) -> None:
    from dataclasses import replace

    capture, artifacts = _algorithm_capture(tmp_path)
    partial = replace(capture, window_evidence=capture.window_evidence[:-1])

    with pytest.raises(ValueError, match="canonical_core_window_coverage_incomplete"):
        _canonical_cleaned_core(partial, artifacts)


def test_window_union_cannot_drop_actual_core_samples(tmp_path: Path) -> None:
    from dataclasses import replace

    capture, artifacts = _algorithm_capture(tmp_path)
    windows = tuple(
        {
            **window,
            "timestampsMillis": [
                time for time in window["timestampsMillis"] if time != 80_000
            ],
        }
        for window in capture.window_evidence
    )

    with pytest.raises(ValueError, match="canonical_core_raw_coverage_mismatch"):
        _canonical_cleaned_core(replace(capture, window_evidence=windows), artifacts)


def test_shifted_windows_fail_independent_session_elapsed_selection(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    capture, artifacts = _algorithm_capture(tmp_path)
    elapsed = {
        time: time + 10_000
        for time in capture.batch_raw_values_by_channel_timestamp["ppg_green"]
    }

    with pytest.raises(
        ValueError,
        match="canonical_core_duration_incomplete|canonical_core_raw_coverage_mismatch",
    ):
        _canonical_cleaned_core(
            replace(capture, batch_elapsed_by_channel_timestamp={"ppg_green": elapsed}),
            artifacts,
        )


def test_green_integer_detector_guard_matches_frozen_python_rule() -> None:
    with pytest.raises(ValueError, match="msptd_integer_sampling_rate_required"):
        _detect(
            np.arange(20, dtype=float),
            25.3,
            {"detector_method": "neurokit2_ppg_charlton_msptdfast"},
        )


def test_sampling_policy_is_not_assumed_for_an_unverified_future_run(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match="live_ppg_sampling_policy_not_verified_for_run"
    ):
        _load_run_artifacts(tmp_path, "20260917T000000Z")
