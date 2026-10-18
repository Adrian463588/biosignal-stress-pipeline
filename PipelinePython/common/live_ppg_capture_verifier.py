"""Fail-closed verification for opt-in Android captures of real Watch PPG."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from .features_v4 import extract_ppg_features
from .peak_detection import PpgPeakDetection, msptd_fast_boundary_safe
from .preprocessing import SignalFilter

CAPTURE_SCHEMA_VERSION = 2
FEATURE_ATOL = 1.0e-4
FEATURE_RTOL = 1.0e-5
PROBABILITY_ATOL = 1.0e-5
FILTER_ATOL = 1.0e-5
SAMPLING_RATE_POLICY = "recording_filter_observed_window_v1"
VERIFIED_SAMPLING_POLICY_RUN = "20260916T132736Z"
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class CapturedLiveSession:
    """Private capture evidence required before raw-to-feature parity can run."""

    capture_directory: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    batch_timestamps_by_channel: Mapping[str, frozenset[int]]
    batch_raw_values_by_channel_timestamp: Mapping[str, Mapping[int, int]]
    batch_ids_by_channel_timestamp: Mapping[str, Mapping[int, str]]
    batch_checksums_by_channel_timestamp: Mapping[str, Mapping[int, str]]
    recording_rates_by_channel: Mapping[str, float]
    batch_elapsed_by_channel_timestamp: Mapping[str, Mapping[int, int]]
    window_evidence: tuple[Mapping[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def watch_payload_sha256(batch: Mapping[str, Any]) -> str:
    """Match Android's canonical batch payload digest, without decoding wire bytes."""

    parts = (
        "channel",
        "sessionId",
        "sessionNonce",
        "protocolVersion",
        "peerNodeId",
        "generation",
        "batchId",
        "sequenceNumber",
        "samplingRateHz",
        "elapsedStartMillis",
        "elapsedEndMillis",
    )
    try:
        payload = "".join(f"{batch[name]}\n" for name in parts)
        payload += "".join(f"{int(value)}," for value in batch["timestampsMillis"])
        payload += "\n"
        payload += "".join(f"{int(value)}," for value in batch["rawValues"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("capture_batch_payload_invalid") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def kotlin_index_checksum(indices: NDArray[np.int64]) -> str:
    """Match `PpgDetectorTraceChecksum.indices` exactly."""

    value = 0xCBF29CE484222325
    mask = (1 << 64) - 1
    for index in np.asarray(indices, dtype=np.int64):
        current = int(index)
        for _ in range(4):
            value ^= current & 0xFF
            value = (value * 0x100000001B3) & mask
            current >>= 8
    return f"{value:016x}"


def load_captured_live_session(capture_directory: Path) -> CapturedLiveSession:
    """Read only a finalized, untampered, acknowledged Watch capture."""

    root = capture_directory.resolve()
    manifest_path = root / "capture_manifest.json"
    manifest = _read_object(manifest_path, "capture_manifest_missing")
    if manifest.get("schemaVersion") != CAPTURE_SCHEMA_VERSION:
        raise ValueError("capture_schema_mismatch")
    if manifest.get("source") != "acknowledged_watch_transport":
        raise ValueError("capture_source_not_acknowledged_watch_transport")
    if manifest.get("actualSensorAcquisition") is not False:
        raise ValueError("capture_claims_sensor_parity_before_python_oracle")
    if manifest.get("sensorPathParityStatus") != "PENDING_PYTHON_ORACLE":
        raise ValueError("capture_sensor_path_status_invalid")
    if manifest.get("verificationStatus") != "PENDING_PYTHON_ORACLE":
        raise ValueError("capture_not_finalized_for_python_oracle")
    if manifest.get("finalized") is not True:
        raise ValueError("capture_not_finalized")
    if int(manifest.get("droppedWriteCount", -1)) != 0:
        raise ValueError("capture_write_loss_detected")

    session_id = _required_string(manifest, "sessionId")
    session_nonce = _required_string(manifest, "sessionNonce")
    batch_files = sorted(
        path
        for path in root.glob("*.json")
        if path.name
        not in {
            "capture_manifest.json",
            "inference_traces.json",
            "window_evidence.json",
        }
    )
    if not batch_files:
        raise ValueError("capture_contains_no_acknowledged_batches")

    observed_by_channel: dict[str, set[int]] = {}
    raw_values_by_channel: dict[str, dict[int, int]] = {}
    batch_ids_by_channel: dict[str, dict[int, str]] = {}
    batch_checksums_by_channel: dict[str, dict[int, str]] = {}
    observed_counts: dict[str, int] = {}
    recording_rates: dict[str, float] = {}
    elapsed_by_channel: dict[str, dict[int, int]] = {}
    for path in batch_files:
        batch = _read_object(path, "capture_batch_unreadable")
        if batch.get("schemaVersion") != CAPTURE_SCHEMA_VERSION:
            raise ValueError("capture_batch_schema_mismatch")
        if batch.get("source") != "acknowledged_watch_transport":
            raise ValueError("capture_batch_source_invalid")
        if (
            batch.get("sessionId") != session_id
            or batch.get("sessionNonce") != session_nonce
        ):
            raise ValueError("capture_batch_session_mismatch")
        timestamps = _int_list(batch, "timestampsMillis")
        raw_values = _int_list(batch, "rawValues")
        if not timestamps or len(timestamps) != len(raw_values):
            raise ValueError("capture_batch_shape_mismatch")
        if batch.get("payloadSha256") != watch_payload_sha256(batch):
            raise ValueError("capture_batch_payload_checksum_mismatch")
        channel = _required_string(batch, "channel")
        rate = float(batch.get("samplingRateHz", 0))
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("capture_recording_rate_invalid")
        if channel in recording_rates and recording_rates[channel] != rate:
            raise ValueError("capture_recording_rate_conflict")
        recording_rates[channel] = rate
        elapsed_start = int(batch["elapsedStartMillis"])
        elapsed_end = int(batch["elapsedEndMillis"])
        if elapsed_start < 0 or elapsed_end < elapsed_start:
            raise ValueError("capture_batch_elapsed_invalid")
        if (
            elapsed_start > 0
            and elapsed_end - elapsed_start != timestamps[-1] - timestamps[0]
        ):
            raise ValueError("capture_batch_elapsed_span_mismatch")
        channel_elapsed = elapsed_by_channel.setdefault(channel, {})
        for timestamp in timestamps:
            channel_elapsed[timestamp] = (
                elapsed_end
                if len(timestamps) == 1
                else elapsed_start + timestamp - timestamps[0]
            )
        batch_id = _required_string(batch, "batchId")
        payload_checksum = str(batch["payloadSha256"])
        observed_by_channel.setdefault(channel, set()).update(timestamps)
        channel_values = raw_values_by_channel.setdefault(channel, {})
        channel_batch_ids = batch_ids_by_channel.setdefault(channel, {})
        channel_batch_checksums = batch_checksums_by_channel.setdefault(channel, {})
        for timestamp, raw_value in zip(timestamps, raw_values, strict=True):
            previous = channel_values.get(timestamp)
            if previous is not None and previous != raw_value:
                raise ValueError("capture_timestamp_raw_value_conflict")
            previous_batch_id = channel_batch_ids.get(timestamp)
            if previous_batch_id is not None and previous_batch_id != batch_id:
                raise ValueError("capture_timestamp_batch_conflict")
            channel_values[timestamp] = raw_value
            channel_batch_ids[timestamp] = batch_id
            channel_batch_checksums[timestamp] = payload_checksum
        observed_counts[channel] = observed_counts.get(channel, 0) + 1

    declared_counts = manifest.get("batchFilesByChannel")
    if (
        not isinstance(declared_counts, Mapping)
        or {str(channel): int(count) for channel, count in declared_counts.items()}
        != observed_counts
    ):
        raise ValueError("capture_batch_file_count_mismatch")

    evidence_name = _required_string(manifest, "windowEvidenceFile")
    evidence_payload = _read_json(
        root / evidence_name, "capture_window_evidence_missing"
    )
    if not isinstance(evidence_payload, list) or not evidence_payload:
        raise ValueError("capture_window_evidence_empty")
    evidence = tuple(item for item in evidence_payload if isinstance(item, Mapping))
    if len(evidence) != len(evidence_payload):
        raise ValueError("capture_window_evidence_invalid")
    return CapturedLiveSession(
        capture_directory=root,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        batch_timestamps_by_channel={
            channel: frozenset(timestamps)
            for channel, timestamps in observed_by_channel.items()
        },
        batch_raw_values_by_channel_timestamp={
            channel: dict(values) for channel, values in raw_values_by_channel.items()
        },
        batch_ids_by_channel_timestamp={
            channel: dict(values) for channel, values in batch_ids_by_channel.items()
        },
        batch_checksums_by_channel_timestamp={
            channel: dict(values)
            for channel, values in batch_checksums_by_channel.items()
        },
        window_evidence=evidence,
        recording_rates_by_channel=recording_rates,
        batch_elapsed_by_channel_timestamp=elapsed_by_channel,
    )


def verify_live_ppg_capture(
    project_root: Path,
    run_id: str,
    capture_directory: Path,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Compare Android live-window evidence with a frozen model contract.

    This function never fits, trains, alters a model, or writes into the private
    Android capture. A successful report is the only condition under which an
    optional, run-scoped physical-capture receipt is atomically published.
    """

    root = project_root.resolve()
    try:
        captured = load_captured_live_session(capture_directory)
        artifacts = _load_run_artifacts(root, run_id)
        cleaned = _canonical_cleaned_core(captured, artifacts)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return _failed_report(run_id, capture_directory, str(error))

    channel_reports: dict[str, list[dict[str, Any]]] = {}
    for evidence in captured.window_evidence:
        report = _verify_window(evidence, captured, artifacts, cleaned)
        channel = str(evidence.get("channel", "unknown"))
        channel_reports.setdefault(channel, []).append(report)

    active_channels = sorted(artifacts)
    eligible_reports = [
        report
        for channel in active_channels
        for report in channel_reports.get(channel, [])
    ]
    missing_channels = [
        channel for channel in active_channels if not channel_reports.get(channel)
    ]
    passed = (
        bool(active_channels)
        and not missing_channels
        and all(report["status"] == "GO" for report in eligible_reports)
    )
    replay = captured.manifest.get("verificationExecution") == "REPLAY"
    report: dict[str, Any] = {
        "schema_version": "live-ppg-parity-v3",
        "run_id": run_id,
        "verification_scope": "watch_raw_to_canonical_cleaning_features_onnx",
        "sampling_rate_policy": SAMPLING_RATE_POLICY,
        "preprocessing_verified": passed,
        "capture_directory": str(captured.capture_directory),
        "capture_manifest_sha256": captured.manifest_sha256,
        "capture_source": "acknowledged_watch_transport",
        "verification_execution": "REPLAY" if replay else "ACQUISITION",
        "processing_apk_sha256": captured.manifest.get("processingApkSha256"),
        "actual_sensor_acquisition": passed and not replay,
        "sensor_path_parity_status": ("REPLAY_VERIFIED" if replay else "GO")
        if passed
        else "NO_GO",
        "active_channels": active_channels,
        "raw_lineage_verified": passed,
        "missing_active_channels": missing_channels,
        "status": "GO" if passed else "NO_GO",
        "feature_tolerance": {"atol": FEATURE_ATOL, "rtol": FEATURE_RTOL},
        "probability_tolerance": PROBABILITY_ATOL,
        "filter_tolerance": {"atol": FILTER_ATOL, "rtol": 1.0e-9},
        "channels": {
            channel: {
                "status": (
                    "NOT_APPLICABLE"
                    if channel not in artifacts
                    else "GO"
                    if channel_reports.get(channel)
                    and all(item["status"] == "GO" for item in channel_reports[channel])
                    else "NO_GO"
                ),
                "window_count": len(channel_reports.get(channel, [])),
                "windows": channel_reports.get(channel, []),
            }
            for channel in sorted(set(channel_reports) | set(active_channels))
        },
    }
    if passed and receipt_path is not None:
        _atomic_new_json(receipt_path.resolve(), report)
        report["receipt_path"] = str(receipt_path.resolve())
    return report


def _verify_window(
    evidence: Mapping[str, Any],
    captured: CapturedLiveSession,
    artifacts: Mapping[str, Mapping[str, Any]],
    cleaned: Mapping[str, Mapping[int, float]],
) -> dict[str, Any]:
    channel = str(evidence.get("channel", "unknown"))
    window_id = str(evidence.get("windowId", "unknown"))
    reasons: list[str] = []
    trace = evidence.get("inferenceTrace")
    if not isinstance(trace, Mapping):
        return _window_report(channel, window_id, reasons=["inference_trace_missing"])
    artifact = artifacts.get(channel)
    if artifact is None:
        return _window_report(
            channel,
            window_id,
            reasons=["diagnostic_channel_not_in_model_contract"],
            status="NOT_APPLICABLE",
        )

    values = _float_vector(evidence, "filteredValues", reasons)
    raw_values = _int_vector(evidence, "rawValues", reasons)
    timestamp_batch_ids = _string_vector(evidence, "timestampBatchIds", reasons)
    timestamp_batch_checksums = _string_vector(
        evidence, "timestampBatchChecksums", reasons
    )
    timestamps = _int_vector(evidence, "timestampsMillis", reasons)
    observed_mask = _bool_vector(evidence, "observedMask", reasons)
    repaired_mask = _bool_vector(evidence, "repairedShortGapMask", reasons)
    grid_mask = _bool_vector(evidence, "gridResampledMask", reasons)
    if (
        not values
        or len(timestamps) != len(values)
        or len(raw_values) != len(timestamps)
    ):
        reasons.append("window_shape_mismatch")
    if not raw_values:
        reasons.append("window_raw_values_missing")
    if len(timestamp_batch_ids) != len(timestamps) or len(
        timestamp_batch_checksums
    ) != len(timestamps):
        reasons.append("window_batch_lineage_shape_mismatch")
    if (
        len(observed_mask) != len(values)
        or len(repaired_mask) != len(values)
        or len(grid_mask) != len(values)
    ):
        reasons.append("window_provenance_shape_mismatch")
    if any(not np.isfinite(value) for value in values):
        reasons.append("window_non_finite")
    if any(
        next_value <= value
        for value, next_value in zip(timestamps, timestamps[1:], strict=False)
    ):
        reasons.append("window_timestamp_not_strictly_monotonic")
    if not all(observed_mask) or any(repaired_mask) or any(grid_mask):
        reasons.append("window_contains_derived_samples")
    if (
        int(evidence.get("trueGapCount", -1)) != 0
        or int(evidence.get("unresolvedSampleCount", -1)) != 0
    ):
        reasons.append("window_contains_unresolved_gap")
    if not set(timestamps).issubset(
        captured.batch_timestamps_by_channel.get(channel, frozenset())
    ):
        reasons.append("window_timestamps_not_backed_by_watch_batches")
    captured_values = captured.batch_raw_values_by_channel_timestamp.get(channel, {})
    captured_batch_ids = captured.batch_ids_by_channel_timestamp.get(channel, {})
    captured_batch_checksums = captured.batch_checksums_by_channel_timestamp.get(
        channel, {}
    )
    for timestamp, raw_value, batch_id, batch_checksum in zip(
        timestamps,
        raw_values,
        timestamp_batch_ids,
        timestamp_batch_checksums,
        strict=False,
    ):
        if timestamp not in captured_values:
            reasons.append("window_raw_values_not_backed_by_watch_batches")
            break
        if captured_values[timestamp] != raw_value:
            reasons.append("window_raw_value_mismatch")
            break
        if captured_batch_ids.get(timestamp) != batch_id:
            reasons.append("window_batch_id_mismatch")
            break
        if captured_batch_checksums.get(timestamp) != batch_checksum:
            reasons.append("window_batch_checksum_mismatch")
            break
    if _raw_values_sha256(raw_values) != evidence.get("rawValuesSha256"):
        reasons.append("window_raw_values_checksum_mismatch")

    contract = artifact["contract"]
    manifest = artifact["manifest"]
    # Model-bound Gold records endpoint cadence. The run's Git base differs
    # from its dirty executable; stored training artifacts decide this policy.
    rate = (
        (len(timestamps) - 1) * 1000.0 / max(1, timestamps[-1] - timestamps[0])
        if timestamps
        else 0.0
    )
    _check_trace_contract(trace, contract, manifest, values, reasons)
    if trace.get("samplingRatePolicy") != SAMPLING_RATE_POLICY:
        reasons.append("trace_sampling_rate_policy_mismatch")
    if not _close(
        trace.get("sourceRateHz"), captured.recording_rates_by_channel[channel], 1.0e-9
    ):
        reasons.append("trace_recording_rate_mismatch")
    if not _close(trace.get("modelRateHz"), rate, 1.0e-9):
        reasons.append("trace_feature_window_rate_mismatch")
    expected_values = np.asarray(
        [cleaned[channel][time] for time in timestamps], dtype=np.float64
    )
    if len(values) == len(expected_values) and not np.allclose(
        values, expected_values, atol=FILTER_ATOL, rtol=1.0e-9
    ):
        reasons.append("canonical_filter_mismatch")
    if reasons:
        return _window_report(channel, window_id, reasons=reasons)

    try:
        values_array = expected_values
        detection = _detect(values_array, rate, contract)
        features, _, _ = extract_ppg_features(
            values_array,
            rate,
            channel=channel,
            detection=detection,
            window_seconds=int(manifest["window_seconds"]),
        )
        feature_order = tuple(str(value) for value in manifest["feature_order"])
        expected_features = np.asarray(
            [features[name] for name in feature_order], dtype=np.float32
        )
        observed_features = np.asarray(trace["featureValues"], dtype=np.float32)
        _compare_detector_trace(trace, detection, rate, reasons)
        if observed_features.shape != expected_features.shape or not np.allclose(
            observed_features,
            expected_features,
            rtol=FEATURE_RTOL,
            atol=FEATURE_ATOL,
        ):
            reasons.append("feature_vector_mismatch")
        if (
            trace.get("featureChecksum")
            != hashlib.sha256(observed_features.astype("<f4").tobytes()).hexdigest()
        ):
            reasons.append("feature_checksum_mismatch")
        raw_probability = _onnx_probability(
            artifact["model_path"], manifest, expected_features
        )
        calibrated_probability = _calibrate(
            raw_probability, manifest["probability_calibration"]
        )
        if not _close(trace.get("rawProbability"), raw_probability, PROBABILITY_ATOL):
            reasons.append("raw_probability_mismatch")
        if not _close(
            trace.get("calibratedProbability"), calibrated_probability, PROBABILITY_ATOL
        ):
            reasons.append("calibrated_probability_mismatch")
    except (ImportError, KeyError, ValueError, RuntimeError) as error:
        reasons.append(f"python_oracle_error:{type(error).__name__}:{error}")
    return _window_report(
        channel,
        window_id,
        reasons=reasons,
        peak_count=int(trace.get("peakCount", 0)),
        valid_ibi_count=int(trace.get("validIbiCount", 0)),
        adjacent_ibi_count=int(trace.get("adjacentIbiCount", 0)),
    )


def _canonical_cleaned_core(
    captured: CapturedLiveSession, artifacts: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[int, float]]:
    """Reconstruct the full observed core, never trust Android filtered values."""
    output: dict[str, dict[int, float]] = {}
    for channel, artifact in artifacts.items():
        windows = [
            item for item in captured.window_evidence if item.get("channel") == channel
        ]
        manifest = artifact["manifest"]
        duration = int(manifest["window_seconds"])
        stride = int(manifest["stride_seconds"])
        expected_windows = (240 - duration) // stride + 1
        if len(windows) != expected_windows:
            raise ValueError(f"canonical_core_window_coverage_incomplete:{channel}")
        window_times = [tuple(_int_list(item, "timestampsMillis")) for item in windows]
        starts = sorted(times[0] for times in window_times if times)
        elapsed = captured.batch_elapsed_by_channel_timestamp[channel]
        observed = captured.batch_raw_values_by_channel_timestamp[channel]
        timestamps = sorted(
            time for time in observed if 30_000 <= elapsed[time] < 270_000
        )
        if len(starts) != expected_windows or len(timestamps) < 2:
            raise ValueError(f"canonical_core_timestamps_missing:{channel}")
        rate = captured.recording_rates_by_channel[channel]
        interval = 1000.0 / rate
        if any(
            abs(right - left - stride * 1000) > 2 * interval
            for left, right in zip(starts, starts[1:])
        ):
            raise ValueError(f"canonical_core_window_stride_mismatch:{channel}")
        if abs(timestamps[-1] - timestamps[0] - 240_000) > 2 * interval:
            raise ValueError(f"canonical_core_duration_incomplete:{channel}")
        if {time for times in window_times for time in times} != set(timestamps):
            raise ValueError(f"canonical_core_raw_coverage_mismatch:{channel}")
        for index, times in enumerate(sorted(window_times, key=lambda times: times[0])):
            start = 30_000 + index * stride * 1000
            end = start + duration * 1000
            expected = tuple(
                time for time in timestamps if start <= elapsed[time] < end
            )
            if times != expected:
                raise ValueError(f"canonical_model_window_selection_mismatch:{channel}")
        if np.max(np.diff(timestamps)) > 4 * interval:
            raise ValueError(f"canonical_core_true_gap:{channel}")
        raw = np.asarray([observed[time] for time in timestamps], dtype=np.float64)
        filtered = SignalFilter.bandpass(
            raw,
            rate,
            float(manifest["filter_low_cut_hz"]),
            float(manifest["filter_high_cut_hz"]),
            int(manifest["filter_order"]),
        )
        output[channel] = dict(zip(timestamps, filtered.tolist(), strict=True))
    return output


def _load_run_artifacts(project_root: Path, run_id: str) -> dict[str, dict[str, Any]]:
    # The dirty run's model-bound Gold was independently checked. Do not
    # silently apply that sampling policy to a future, unverified training run.
    if run_id != VERIFIED_SAMPLING_POLICY_RUN:
        raise ValueError("live_ppg_sampling_policy_not_verified_for_run")
    run_root = project_root / "data" / "artifacts" / "runs" / run_id
    index = _read_object(
        run_root / "export" / "internal" / "internal_model_index.json",
        "internal_model_index_missing",
    )
    contract_bundle = _read_object(
        run_root / "export" / "internal" / "ppg_detector_contract.json",
        "detector_contract_missing",
    )
    if (
        index.get("schema_version") != "internal-android-v2"
        or index.get("run_id") != run_id
    ):
        raise ValueError("internal_model_index_contract_mismatch")
    if (
        contract_bundle.get("schema_version") != "ppg-detector-contract-v1"
        or contract_bundle.get("run_id") != run_id
    ):
        raise ValueError("detector_contract_run_mismatch")
    contracts = {
        str(item.get("sensor_channel")): item
        for item in contract_bundle.get("contracts", [])
        if isinstance(item, Mapping)
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for entry in index.get("models", []):
        if not isinstance(entry, Mapping):
            continue
        channel = str(entry.get("sensor_channel", ""))
        contract = contracts.get(channel)
        files = entry.get("files")
        if contract is None or not isinstance(files, Mapping):
            raise ValueError("internal_model_index_entry_invalid")
        model_path = _resolve_run_path(project_root, files.get("model"))
        manifest_path = _resolve_run_path(project_root, files.get("manifest"))
        if sha256_file(model_path) != str(files["model"].get("sha256", "")):
            raise ValueError("onnx_model_checksum_mismatch")
        manifest = _read_object(manifest_path, "model_manifest_missing")
        if sha256_file(manifest_path) != str(files["manifest"].get("sha256", "")):
            raise ValueError("model_manifest_checksum_mismatch")
        if (
            manifest.get("sensor_channel") != channel
            or manifest.get("model_version") != contract.get("model_version")
            or manifest.get("candidate_bundle_sha256")
            != contract.get("candidate_artifact_fingerprint")
            or manifest.get("onnx_sha256") != contract.get("onnx_sha256")
            or sha256_file(model_path) != contract.get("onnx_sha256")
        ):
            raise ValueError("model_detector_contract_mismatch")
        artifacts[channel] = {
            "contract": contract,
            "manifest": manifest,
            "model_path": model_path,
        }
    if not artifacts:
        raise ValueError("no_run_scoped_internal_models")
    return artifacts


def _check_trace_contract(
    trace: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    values: list[float],
    reasons: list[str],
) -> None:
    checks = {
        "sensorChannel": contract["sensor_channel"],
        "detectorMethod": contract["detector_method"],
        "detectorFingerprint": contract["detector_config_fingerprint"],
        "filterContractFingerprint": contract["filter_contract_fingerprint"],
        "manifestFingerprint": manifest["candidate_bundle_sha256"],
        "featureVersion": manifest["feature_version"],
    }
    for name, expected in checks.items():
        if trace.get(name) != expected:
            reasons.append(f"trace_{name}_mismatch")
    if tuple(trace.get("featureOrder", ())) != tuple(manifest["feature_order"]):
        reasons.append("trace_feature_order_mismatch")
    if int(trace.get("featureCount", -1)) != len(manifest["feature_order"]):
        reasons.append("trace_feature_count_mismatch")
    if int(trace.get("sampleCount", -1)) != len(values):
        reasons.append("trace_sample_count_mismatch")
    if int(trace.get("observedSampleCount", -1)) != len(values):
        reasons.append("trace_observed_count_mismatch")
    if any(
        int(trace.get(name, -1)) != 0
        for name in (
            "repairedShortGapSampleCount",
            "trueGapCount",
            "unresolvedSampleCount",
            "gridResampledSampleCount",
        )
    ):
        reasons.append("trace_contains_derived_or_gap_samples")
    if not _rate_compatible(
        trace.get("sourceRateHz"), float(contract["source_rate_hz"])
    ):
        reasons.append("trace_source_rate_mismatch")
    if not _rate_compatible(trace.get("modelRateHz"), float(contract["model_rate_hz"])):
        reasons.append("trace_model_rate_mismatch")


def _rate_compatible(observed: Any, expected: float) -> bool:
    return (
        isinstance(observed, (int, float))
        and np.isfinite(observed)
        and observed > 0
        and abs(observed - expected) / expected <= 0.20
    )


def _compare_detector_trace(
    trace: Mapping[str, Any],
    detection: PpgPeakDetection,
    sampling_rate_hz: float,
    reasons: list[str],
) -> None:
    all_ibi = np.diff(detection.peaks).astype(np.float64) * 1000.0 / sampling_rate_hz
    physiological = (all_ibi >= 300.0) & (all_ibi <= 1500.0)
    expected = {
        "peakChecksum": kotlin_index_checksum(detection.peaks),
        "troughChecksum": kotlin_index_checksum(detection.troughs),
        "peakCount": int(detection.peaks.size),
        "validIbiCount": int(np.sum(physiological)),
        "adjacentIbiCount": int(np.sum(physiological[:-1] & physiological[1:])),
    }
    for name, expected_value in expected.items():
        if trace.get(name) != expected_value:
            reasons.append(f"detector_{name}_mismatch")
    expected_ratio = float(np.mean(physiological)) if physiological.size else 0.0
    if not _close(trace.get("physiologicalIbiRatio"), expected_ratio, FEATURE_ATOL):
        reasons.append("detector_physiological_ibi_ratio_mismatch")


def _detect(
    values: NDArray[np.float64],
    sampling_rate_hz: float,
    contract: Mapping[str, Any],
) -> PpgPeakDetection:
    method = str(contract["detector_method"])
    if method == "neurokit2_ppg_charlton_msptdfast":
        detector_rate = round(sampling_rate_hz)
        if not np.isclose(sampling_rate_hz, detector_rate, rtol=0.01, atol=0.0):
            raise ValueError("msptd_integer_sampling_rate_required")
        peaks, _ = msptd_fast_boundary_safe(
            values,
            float(detector_rate),
        )
        _, troughs = _scipy_detect(values, sampling_rate_hz, contract)
    elif method == "scipy_find_peaks_adaptive":
        peaks, troughs = _scipy_detect(values, sampling_rate_hz, contract)
    else:
        raise ValueError("unsupported_detector_method")
    if peaks.size < int(contract["min_peak_count"]):
        raise ValueError(f"insufficient_peaks:{peaks.size}")
    ratio = _physiological_ratio(peaks, sampling_rate_hz)
    if ratio < float(contract["min_physiological_ibi_ratio"]):
        raise ValueError("insufficient_physiological_ibi_ratio")
    return PpgPeakDetection(
        peaks=peaks,
        troughs=troughs,
        method=method,
        fallback_reason="",
        physiological_ibi_ratio=ratio,
        config_fingerprint=str(contract["detector_config_fingerprint"]),
    )


def _scipy_detect(
    values: NDArray[np.float64],
    sampling_rate_hz: float,
    contract: Mapping[str, Any],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    centered = values - np.median(values)
    prominence = max(
        np.finfo(float).eps,
        float(contract["prominence_mad_multiplier"])
        * float(np.median(np.abs(centered))),
    )
    distance = max(
        1, int(round(float(contract["min_distance_seconds"]) * sampling_rate_hz))
    )
    width_min = contract.get("width_min_samples")
    width_max = contract.get("width_max_samples")
    width = None if width_min is None and width_max is None else (width_min, width_max)
    peaks, _ = find_peaks(
        centered, distance=distance, prominence=prominence, width=width
    )
    troughs, _ = find_peaks(
        -centered, distance=distance, prominence=prominence, width=width
    )
    return peaks.astype(np.int64), troughs.astype(np.int64)


def _physiological_ratio(peaks: NDArray[np.int64], sampling_rate_hz: float) -> float:
    if peaks.size < 2:
        return 0.0
    ibi = np.diff(peaks).astype(np.float64) * 1000.0 / sampling_rate_hz
    return float(np.mean((ibi >= 300.0) & (ibi <= 1500.0)))


def _onnx_probability(
    model_path: Path,
    manifest: Mapping[str, Any],
    features: NDArray[np.float32],
) -> float:
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - environment dependency
        raise ImportError("onnxruntime_not_installed") from error
    scaled = features.copy()
    means = manifest.get("scaler_mean")
    scales = manifest.get("scaler_scale")
    if means is not None or scales is not None:
        if (
            not isinstance(means, list)
            or not isinstance(scales, list)
            or len(means) != len(features)
            or len(scales) != len(features)
        ):
            raise ValueError("manifest_scaler_contract_invalid")
        for index, (mean, scale) in enumerate(zip(means, scales, strict=True)):
            if float(scale) > 0.0:
                scaled[index] = np.float32((scaled[index] - float(mean)) / float(scale))
    input_name = str(manifest["onnx_input"]["name"])
    output_name = str(manifest["onnx_output"]["name"])
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output = session.run([output_name], {input_name: scaled.reshape(1, -1)})[0]
    value = float(np.asarray(output, dtype=np.float32).reshape(-1)[0])
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("onnx_probability_invalid")
    return value


def _calibrate(raw_probability: float, calibration: Mapping[str, Any]) -> float:
    if calibration.get("method") != "platt":
        raise ValueError("unsupported_probability_calibration")
    bounded = np.clip(raw_probability, 1.0e-7, 1.0 - 1.0e-7)
    logit = np.log(bounded / (1.0 - bounded))
    value = 1.0 / (
        1.0
        + np.exp(
            -(float(calibration["slope"]) * logit + float(calibration["intercept"]))
        )
    )
    return float(np.clip(value, 0.0, 1.0))


def _window_report(
    channel: str,
    window_id: str,
    *,
    reasons: list[str],
    status: str | None = None,
    peak_count: int | None = None,
    valid_ibi_count: int | None = None,
    adjacent_ibi_count: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "channel": channel,
        "window_id": window_id,
        "status": status or ("GO" if not reasons else "NO_GO"),
        "reasons": sorted(set(reasons)),
    }
    if peak_count is not None:
        report["peak_count"] = peak_count
    if valid_ibi_count is not None:
        report["valid_ibi_count"] = valid_ibi_count
    if adjacent_ibi_count is not None:
        report["adjacent_ibi_count"] = adjacent_ibi_count
    return report


def _failed_report(run_id: str, capture_directory: Path, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "live-ppg-parity-v3",
        "preprocessing_verified": False,
        "run_id": run_id,
        "capture_directory": str(capture_directory),
        "actual_sensor_acquisition": False,
        "sensor_path_parity_status": "NO_GO",
        "raw_lineage_verified": False,
        "status": "NO_GO",
        "reasons": [reason],
    }


def _read_json(path: Path, missing_reason: str) -> Any:
    if not path.is_file():
        raise ValueError(missing_reason)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_object(path: Path, missing_reason: str) -> Mapping[str, Any]:
    payload = _read_json(path, missing_reason)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{missing_reason}_object_required")
    return payload


def _required_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"capture_{name}_missing")
    return value


def _int_list(payload: Mapping[str, Any], name: str) -> list[int]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise ValueError(f"capture_{name}_invalid")
    return [int(item) for item in value]


def _float_vector(
    payload: Mapping[str, Any], name: str, reasons: list[str]
) -> list[float]:
    try:
        return [float(item) for item in payload.get(name, [])]
    except (TypeError, ValueError):
        reasons.append(f"window_{name}_invalid")
        return []


def _int_vector(payload: Mapping[str, Any], name: str, reasons: list[str]) -> list[int]:
    try:
        return [int(item) for item in payload.get(name, [])]
    except (TypeError, ValueError):
        reasons.append(f"window_{name}_invalid")
        return []


def _string_vector(
    payload: Mapping[str, Any], name: str, reasons: list[str]
) -> list[str]:
    value = payload.get(name, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        reasons.append(f"window_{name}_invalid")
        return []
    return value


def _raw_values_sha256(values: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _bool_vector(
    payload: Mapping[str, Any], name: str, reasons: list[str]
) -> list[bool]:
    value = payload.get(name, [])
    if not isinstance(value, list) or any(not isinstance(item, bool) for item in value):
        reasons.append(f"window_{name}_invalid")
        return []
    return value


def _resolve_run_path(project_root: Path, entry: Any) -> Path:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
        raise ValueError("run_artifact_path_missing")
    path = Path(entry["path"])
    resolved = path if path.is_absolute() else project_root / path
    if not resolved.is_file():
        raise ValueError("run_artifact_missing")
    return resolved


def _close(observed: Any, expected: float, tolerance: float) -> bool:
    try:
        return bool(np.isclose(float(observed), expected, rtol=0.0, atol=tolerance))
    except (TypeError, ValueError):
        return False


def _atomic_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("live_ppg_parity_receipt_already_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
