"""Run-scoped readiness checks for the Android research and production bundles.

This module is deliberately a reporting boundary. It never promotes a model,
opens a holdout cohort, or reads the legacy global export directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .android_vectors import DeviceParityReceiptVerifier

RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")
RESEARCH_CHANNELS = ("ppg_green", "ppg_ir")
ALL_CHANNELS = (*RESEARCH_CHANNELS, "ppg_red")
PARITY_TOLERANCE = 1.0e-5
MIN_PARITY_VECTORS = 100


def _aggregate_research_status(statuses: Iterable[str]) -> str:
    """Return channel-independent research status without hiding failures."""

    return "GO" if any(status == "GO" for status in statuses) else "NO_GO"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _safe_run_path(project_root: Path, run_root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = (
        candidate if candidate.is_absolute() else project_root / candidate
    ).resolve()
    if run_root not in resolved.parents:
        raise RuntimeError(f"artifact_path_outside_active_run:{value}")
    return resolved


def _descriptor_path(
    project_root: Path,
    run_root: Path,
    entry: dict[str, Any],
    role: str,
) -> tuple[Path, str]:
    descriptor = entry.get("files", {}).get(role)
    if not isinstance(descriptor, dict):
        raise RuntimeError(f"research_artifact_descriptor_missing:{role}")
    path_value = str(descriptor.get("path", "")).strip()
    checksum = str(descriptor.get("sha256", "")).lower().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise RuntimeError(f"research_artifact_checksum_invalid:{role}")
    path = _safe_run_path(project_root, run_root, path_value)
    if not path.is_file() or _sha256(path) != checksum:
        raise RuntimeError(f"research_artifact_checksum_mismatch:{role}")
    return path, checksum


def _finite_probability(value: Any) -> float:
    probability = float(value)
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise RuntimeError("parity_probability_invalid")
    return probability


def _validate_host_onnx(
    model_path: Path,
    manifest: dict[str, Any],
    parity: dict[str, Any],
    vectors: dict[str, Any],
) -> dict[str, Any]:
    """Independently check the current ONNX graph against native probabilities."""

    cases = vectors.get("cases")
    if not isinstance(cases, list) or len(cases) < MIN_PARITY_VECTORS:
        raise RuntimeError("real_parity_vectors_fewer_than_100")
    cases = cases[:MIN_PARITY_VECTORS]
    case_ids = [
        str(case.get("case_id", "")) for case in cases if isinstance(case, dict)
    ]
    if len(case_ids) != MIN_PARITY_VECTORS or len(set(case_ids)) != MIN_PARITY_VECTORS:
        raise RuntimeError("real_parity_vector_ids_invalid")
    if vectors.get("synthetic_data_used") is True:
        raise RuntimeError("synthetic_parity_vectors_forbidden")
    if not all(
        str(case.get("participant_id", "")).strip().startswith("ID-")
        and str(case.get("participant_id", ""))
        in {f"ID-{index:02d}" for index in range(1, 29)}
        for case in cases
    ):
        raise RuntimeError("parity_vectors_must_be_development_participants")

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("host_onnx_runtime_unavailable") from exc

    graph = onnx.load(str(model_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("onnx_io_count_invalid")

    manifest_inputs = manifest.get("inputs")
    manifest_output = manifest.get("output")
    if not isinstance(manifest_inputs, list) or len(manifest_inputs) != 1:
        raise RuntimeError("manifest_input_contract_invalid")
    declared_input = manifest_inputs[0]
    if not isinstance(declared_input, dict) or not isinstance(manifest_output, dict):
        raise RuntimeError("manifest_io_contract_invalid")
    if inputs[0].name != declared_input.get("name"):
        raise RuntimeError("onnx_input_name_mismatch")
    if outputs[0].name != manifest_output.get("name"):
        raise RuntimeError("onnx_output_name_mismatch")
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise RuntimeError("onnx_io_dtype_mismatch")

    actual: list[float] = []
    native: list[float] = []
    recorded_onnx: list[float] = []
    expected_features = len(manifest.get("feature_order", []))
    if expected_features <= 0:
        raise RuntimeError("manifest_feature_order_missing")
    for case in cases:
        if not isinstance(case, dict):
            raise RuntimeError("parity_case_invalid")
        values = np.asarray(case.get("input"), dtype=np.float32)
        if values.shape != (expected_features,) or not np.isfinite(values).all():
            raise RuntimeError("parity_input_contract_invalid")
        native_probability = case.get("native_probability")
        if native_probability is None:
            raise RuntimeError("native_probability_missing")
        native.append(_finite_probability(native_probability))
        if case.get("onnx_probability") is not None:
            recorded_onnx.append(_finite_probability(case["onnx_probability"]))
        result = session.run(
            [outputs[0].name],
            {inputs[0].name: values.reshape(1, -1)},
        )[0]
        flattened = np.asarray(result, dtype=np.float64).reshape(-1)
        if flattened.size != 1:
            raise RuntimeError("onnx_probability_output_invalid")
        actual.append(_finite_probability(flattened[0]))

    differences = np.abs(np.asarray(actual) - np.asarray(native))
    recorded_differences = (
        np.abs(np.asarray(actual) - np.asarray(recorded_onnx))
        if len(recorded_onnx) == MIN_PARITY_VECTORS
        else np.asarray([], dtype=np.float64)
    )
    max_abs_diff = float(differences.max())
    if max_abs_diff >= PARITY_TOLERANCE:
        raise RuntimeError(f"host_parity_failed:{max_abs_diff}")
    declared_max = float(
        parity.get("max_abs_diff", parity.get("parity_max_diff", float("nan")))
    )
    if not np.isfinite(declared_max) or declared_max >= PARITY_TOLERANCE:
        raise RuntimeError("declared_host_parity_failed")
    return {
        "onnx_checker_passed": True,
        "onnx_runtime_passed": True,
        "vector_count": MIN_PARITY_VECTORS,
        "max_abs_diff_native": max_abs_diff,
        "max_abs_diff_recorded": (
            float(recorded_differences.max()) if recorded_differences.size else None
        ),
        "declared_max_abs_diff": declared_max,
        "input_name": inputs[0].name,
        "output_name": outputs[0].name,
        "input_dtype": inputs[0].type,
        "output_dtype": outputs[0].type,
    }


def _validate_research_entry(
    project_root: Path,
    run_root: Path,
    index: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    channel = str(entry.get("sensor_channel", ""))
    if channel not in RESEARCH_CHANNELS:
        raise RuntimeError(f"research_channel_not_supported:{channel}")
    if entry.get("artifact_status") != "research_candidate":
        raise RuntimeError("research_artifact_status_invalid")
    if entry.get("evaluation_scope") != "development_only":
        raise RuntimeError("research_evaluation_scope_invalid")
    if entry.get("release_status") != "blocked_existing_holdout_contract_mismatch":
        raise RuntimeError("research_release_status_invalid")
    if (
        entry.get("deployment_scope") != "research"
        or entry.get("android_ready") is not False
    ):
        raise RuntimeError("research_deployment_contract_invalid")
    if str(entry.get("model_version")) != str(index.get("model_version")):
        raise RuntimeError("research_model_version_mismatch")

    model_path, model_checksum = _descriptor_path(
        project_root, run_root, entry, "model"
    )
    manifest_path, manifest_checksum = _descriptor_path(
        project_root, run_root, entry, "manifest"
    )
    parity_path, parity_checksum = _descriptor_path(
        project_root, run_root, entry, "parity"
    )
    vectors_path, vectors_checksum = _descriptor_path(
        project_root, run_root, entry, "test_vectors"
    )
    manifest = _read_json(manifest_path)
    parity = _read_json(parity_path)
    vectors = _read_json(vectors_path)
    if manifest.get("schema_version") != "onnx-research-v1":
        raise RuntimeError("research_manifest_schema_invalid")
    if manifest.get("model_version") != entry.get("model_version"):
        raise RuntimeError("research_manifest_model_mismatch")
    if manifest.get("sensor_channel") != channel:
        raise RuntimeError("research_manifest_channel_mismatch")
    if manifest.get("holdout_evaluated") is not False:
        raise RuntimeError("research_manifest_holdout_state_invalid")
    if any(key in manifest for key in ("holdout_metrics", "holdout_raw_metrics")):
        raise RuntimeError("research_manifest_contains_holdout_metrics")
    if manifest.get("clinical_validation_status") != "external_validation_required":
        raise RuntimeError("research_clinical_status_invalid")
    development_cohort = {f"ID-{index_value:02d}" for index_value in range(1, 29)}
    declared_cohort = set(manifest.get("validation_cohort_ids", []))
    if not declared_cohort or not declared_cohort <= development_cohort:
        raise RuntimeError("research_manifest_cohort_invalid")
    if parity.get("passed") is not True:
        raise RuntimeError("declared_host_parity_not_passed")
    host = _validate_host_onnx(model_path, manifest, parity, vectors)
    return {
        "sensor_channel": channel,
        "model_version": str(entry["model_version"]),
        "model_path": str(model_path),
        "model_sha256": model_checksum,
        "manifest_sha256": manifest_checksum,
        "parity_sha256": parity_checksum,
        "vectors_sha256": vectors_checksum,
        "host_validation": host,
        "development_cohort": sorted(development_cohort),
    }


def _receipt_by_channel(
    receipt_paths: Iterable[Path],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    receipts: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for receipt_path in receipt_paths:
        try:
            payload = _read_json(receipt_path)
            channel = str(payload.get("sensor_channel", ""))
            if channel not in ALL_CHANNELS:
                raise RuntimeError("receipt_channel_unsupported")
            if channel in receipts:
                raise RuntimeError("duplicate_receipt_channel")
            receipts[channel] = {"path": str(receipt_path), "payload": payload}
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            errors.append(f"{receipt_path}:{exc}")
    return receipts, errors


def _sensor_path_receipt(run_root: Path) -> dict[str, Any]:
    """Read the separate Watch sensor-path evidence without merging it with ONNX parity."""

    receipt_path = (
        run_root
        / "export"
        / "internal"
        / "device_parity"
        / "sensor_path"
        / "live_ppg_sensor_parity_receipt.json"
    ).resolve()
    base = {
        "path": str(receipt_path),
        "status": "PENDING_DEVICE_CAPTURE",
        "actual_sensor_acquisition": False,
        "sha256": None,
        "reason": "sensor_path_receipt_missing",
    }
    if not receipt_path.is_file():
        return base

    try:
        receipt = _read_json(receipt_path)
        if receipt.get("schema_version") != "live-ppg-parity-v3":
            raise RuntimeError("sensor_path_receipt_schema_invalid")
        if receipt.get("run_id") != run_root.name:
            raise RuntimeError("sensor_path_receipt_run_id_mismatch")
        if receipt.get("status") != "GO":
            raise RuntimeError("sensor_path_receipt_not_go")
        if receipt.get("sensor_path_parity_status") != "GO":
            raise RuntimeError("sensor_path_parity_not_go")
        if receipt.get("actual_sensor_acquisition") is not True:
            raise RuntimeError("sensor_path_receipt_not_actual")
        if receipt.get("raw_lineage_verified") is not True:
            raise RuntimeError("sensor_path_raw_lineage_not_verified")
        if receipt.get("preprocessing_verified") is not True:
            raise RuntimeError("sensor_path_preprocessing_not_verified")
        if (
            receipt.get("verification_scope")
            != "watch_raw_to_canonical_cleaning_features_onnx"
        ):
            raise RuntimeError("sensor_path_verification_scope_invalid")
        if receipt.get("sampling_rate_policy") != "recording_filter_observed_window_v1":
            raise RuntimeError("sensor_path_sampling_rate_policy_invalid")
        active_channels = set(receipt.get("active_channels", []))
        if active_channels != set(RESEARCH_CHANNELS):
            raise RuntimeError("sensor_path_active_channels_incomplete")
        channels = receipt.get("channels")
        if not isinstance(channels, dict):
            raise RuntimeError("sensor_path_channel_results_missing")
        for channel in RESEARCH_CHANNELS:
            channel_result = channels.get(channel)
            if not isinstance(channel_result, dict):
                raise RuntimeError(f"sensor_path_channel_result_missing:{channel}")
            if channel_result.get("status") != "GO":
                raise RuntimeError(f"sensor_path_channel_not_go:{channel}")
            if int(channel_result.get("window_count", 0)) != 7:
                raise RuntimeError(f"sensor_path_channel_window_missing:{channel}")
        return {
            **base,
            "status": "GO",
            "actual_sensor_acquisition": True,
            "sha256": _sha256(receipt_path),
            "reason": "watch_raw_to_feature_python_parity_verified",
            "active_channels": sorted(active_channels),
            "capture_manifest_sha256": receipt.get("capture_manifest_sha256"),
            "channels": {
                channel: {
                    "status": channels[channel].get("status"),
                    "window_count": int(channels[channel].get("window_count", 0)),
                }
                for channel in RESEARCH_CHANNELS
            },
        }
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": "NO_GO",
            "reason": str(exc),
            "sha256": _sha256(receipt_path),
        }


def _verify_research_receipt(
    receipt_record: dict[str, Any],
    *,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    model_path: Path,
    apk_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    receipt_path = Path(receipt_record["path"]).resolve()
    if run_root not in receipt_path.parents:
        raise RuntimeError("receipt_outside_active_run")
    receipt = receipt_record["payload"]
    if receipt.get("schema_version") != "1.0.0":
        raise RuntimeError("receipt_schema_invalid")
    if receipt.get("actual_sensor_acquisition") is not False:
        raise RuntimeError("research_receipt_must_declare_no_sensor_capture")
    if receipt.get("model_version") != entry.get("model_version"):
        raise RuntimeError("receipt_model_version_mismatch")
    verified = DeviceParityReceiptVerifier(max_abs_diff=PARITY_TOLERANCE).verify(
        receipt,
        manifest=manifest,
        model_path=model_path,
        apk_path=apk_path,
    )
    if int(receipt.get("vector_count", 0)) < MIN_PARITY_VECTORS:
        raise RuntimeError("receipt_vector_count_below_100")
    return {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "device_fingerprint": verified["device_fingerprint"],
        "apk_sha256": verified["apk_sha256"],
        "vector_count": int(verified["vector_count"]),
        "max_abs_diff": float(verified["max_abs_diff"]),
        "actual_sensor_acquisition": bool(verified["actual_sensor_acquisition"]),
    }


def _production_status(
    channel: str,
    run_id: str,
    readiness: dict[str, Any],
    production_index: dict[str, Any] | None,
) -> dict[str, str]:
    """Classify production without consulting the global legacy index."""

    current = str(readiness.get("model_release_status_by_channel", {}).get(channel, ""))
    if current == "blocked_insufficient_ppg_coverage":
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": current,
        }
    if production_index is None:
        return {
            "status": "NO_GO",
            "reason": current or "run_scoped_production_index_missing",
        }
    entries = [
        item
        for item in production_index.get("models", [])
        if isinstance(item, dict) and item.get("sensor_channel") == channel
    ]
    if len(entries) != 1:
        return {"status": "NO_GO", "reason": "production_channel_entry_missing"}
    entry = entries[0]
    required = (
        production_index.get("run_id") == run_id,
        production_index.get("release_status") == "accepted",
        entry.get("release_status") == "accepted",
        entry.get("android_ready") is True,
        entry.get("holdout_evaluated") is True,
        bool(entry.get("holdout_contract_fingerprint")),
    )
    if not all(required):
        return {
            "status": "NO_GO",
            "reason": "current_holdout_or_device_contract_missing",
        }
    return {"status": "GO", "reason": "accepted_run_scoped_production_evidence"}


def evaluate_android_readiness(
    project_root: Path,
    run_id: str,
    *,
    apk_path: Path | None,
    receipt_paths: Iterable[Path] = (),
    device_connection_status: str,
) -> dict[str, Any]:
    """Evaluate current-run research and production readiness.

    The function intentionally returns explicit per-channel failure states so a
    failed Red channel cannot hide or block Green/IR research evidence.
    """

    project_root = project_root.resolve()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must use UTC format YYYYMMDDTHHMMSSZ")
    if device_connection_status not in {"READY", "OFFLINE", "INSTRUMENTATION_FAILED"}:
        raise ValueError("unsupported_device_connection_status")
    run_root = (project_root / "data" / "artifacts" / "runs" / run_id).resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"active_run_missing:{run_root}")
    research_index_path = run_root / "export" / "research" / "research_model_index.json"
    if not research_index_path.is_file():
        raise FileNotFoundError(f"research_index_missing:{research_index_path}")
    research_index = _read_json(research_index_path)
    if research_index.get("run_id") != run_id:
        raise RuntimeError("research_index_run_id_mismatch")
    if (
        research_index.get("release_status") != "research_candidate"
        or research_index.get("artifact_status") != "research_candidate"
        or research_index.get("evaluation_scope") != "development_only"
        or research_index.get("deployment_scope") != "research"
        or research_index.get("android_ready") is not False
    ):
        raise RuntimeError("research_index_contract_invalid")

    readiness_path = run_root / "gold" / "data_readiness.json"
    readiness = _read_json(readiness_path) if readiness_path.is_file() else {}
    production_index_path = run_root / "export" / "model_index.json"
    production_index = (
        _read_json(production_index_path) if production_index_path.is_file() else None
    )
    receipts, receipt_errors = _receipt_by_channel(
        Path(path).resolve() for path in receipt_paths
    )
    apk = apk_path.resolve() if apk_path is not None else None
    apk_error = None if apk and apk.is_file() else "research_apk_missing"
    artifact_by_channel: dict[str, dict[str, Any]] = {}
    research_by_channel: dict[str, dict[str, Any]] = {}
    production_by_channel: dict[str, dict[str, str]] = {}
    for channel in ALL_CHANNELS:
        production_by_channel[channel] = _production_status(
            channel, run_id, readiness, production_index
        )

    for channel in RESEARCH_CHANNELS:
        entry = next(
            (
                item
                for item in research_index.get("models", [])
                if isinstance(item, dict) and item.get("sensor_channel") == channel
            ),
            None,
        )
        result: dict[str, Any] = {
            "device_connection_status": device_connection_status,
            "research_model_status": "NO_GO",
            "reason": "research_artifact_not_verified",
            "run_id": run_id,
        }
        try:
            if entry is None:
                raise RuntimeError("research_channel_entry_missing")
            artifact = _validate_research_entry(
                project_root, run_root, research_index, entry
            )
            artifact_by_channel[channel] = artifact
            if device_connection_status != "READY":
                raise RuntimeError(
                    f"device_connection_{device_connection_status.lower()}"
                )
            if apk_error:
                raise RuntimeError(apk_error)
            receipt_record = receipts.get(channel)
            if receipt_record is None:
                raise RuntimeError("research_device_parity_receipt_missing")
            manifest_path, _ = _descriptor_path(
                project_root, run_root, entry, "manifest"
            )
            manifest = _read_json(manifest_path)
            model_path = Path(artifact["model_path"])
            receipt = _verify_research_receipt(
                receipt_record,
                entry=entry,
                manifest=manifest,
                model_path=model_path,
                apk_path=apk,
                run_root=run_root,
            )
            result.update(
                {
                    "research_model_status": "GO",
                    "reason": "run_scoped_host_and_android_parity_verified",
                    "artifact": artifact,
                    "device_parity": receipt,
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            result["reason"] = str(exc)
            if channel in artifact_by_channel:
                result["artifact"] = artifact_by_channel[channel]
        research_by_channel[channel] = result

    for channel in ALL_CHANNELS:
        if channel not in research_by_channel:
            research_by_channel[channel] = {
                "device_connection_status": device_connection_status,
                "research_model_status": "NO_GO",
                "reason": "research_not_deployable_channel",
                "run_id": run_id,
            }

    research_statuses = [
        research_by_channel[channel]["research_model_status"]
        for channel in RESEARCH_CHANNELS
    ]
    production_statuses = [
        production_by_channel[channel]["status"] for channel in ALL_CHANNELS
    ]
    # Research readiness is channel-independent: one verified PPG channel is
    # enough for the research flavor, while each channel keeps its own status.
    research_status = _aggregate_research_status(research_statuses)
    if "GO" in production_statuses:
        production_status = "GO"
    elif all(status == "INSUFFICIENT_EVIDENCE" for status in production_statuses):
        production_status = "INSUFFICIENT_EVIDENCE"
    else:
        production_status = "NO_GO"

    run_contract_path = run_root / "run_contract_fingerprint.json"
    run_contract = _read_json(run_contract_path) if run_contract_path.is_file() else {}
    report = {
        "schema_version": "android-readiness-v1",
        "run_id": run_id,
        "device_connection_status": device_connection_status,
        "research_model_status": research_status,
        "research_model_status_by_channel": research_by_channel,
        "production_model_status": production_status,
        "production_model_status_by_channel": production_by_channel,
        "production_index_used": str(production_index_path)
        if production_index is not None
        else None,
        "global_production_index_used": False,
        "apk_path": str(apk) if apk else None,
        "apk_sha256": _sha256(apk) if apk and apk.is_file() else None,
        "receipt_errors": receipt_errors,
        "run_contract": {
            "source_tree_sha256": run_contract.get("source_tree_sha256"),
            "config_fingerprint": run_contract.get("config_fingerprint"),
            "run_contract_sha256": (
                _sha256(run_contract_path) if run_contract_path.is_file() else None
            ),
        },
        "production_policy": {
            "holdout_reopened": False,
            "holdout_ledger_reused": False,
            "research_status_does_not_promote_production": True,
        },
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    return report


def write_android_readiness_report(
    project_root: Path,
    run_id: str,
    *,
    apk_path: Path | None,
    receipt_paths: Iterable[Path] = (),
    device_connection_status: str,
) -> tuple[dict[str, Any], Path, Path]:
    """Evaluate and atomically write the run-scoped JSON and Markdown report."""

    report = evaluate_android_readiness(
        project_root,
        run_id,
        apk_path=apk_path,
        receipt_paths=receipt_paths,
        device_connection_status=device_connection_status,
    )
    project_root = project_root.resolve()
    artifact_path = (
        project_root
        / "data"
        / "artifacts"
        / "runs"
        / run_id
        / "export"
        / "research"
        / "research_device_readiness.json"
    )
    report_path = (
        project_root
        / "reports"
        / "runs"
        / run_id
        / "android"
        / "research_device_readiness.json"
    )
    markdown_path = report_path.with_suffix(".md")
    _atomic_write_json(artifact_path, report)
    _atomic_write_json(report_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Android readiness — {run_id}",
        "",
        f"- Device connection: `{report['device_connection_status']}`",
        f"- Research models: `{report['research_model_status']}`",
        f"- Production models: `{report['production_model_status']}`",
        "- Production index: run-scoped only; global index was not used.",
        "",
        "| Channel | Research | Production | Reason |",
        "|---|---|---|---|",
    ]
    for channel in ALL_CHANNELS:
        research = report["research_model_status_by_channel"].get(channel, {})
        production = report["production_model_status_by_channel"].get(channel, {})
        lines.append(
            f"| {channel} | {research.get('research_model_status', 'NO_GO')} | "
            f"{production.get('status', 'INSUFFICIENT_EVIDENCE')} | "
            f"{research.get('reason') or production.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "Research status requires current-run host validation, the exact APK, and a device receipt.",
            "Production remains fail-closed until current-contract holdout evidence and production device parity exist.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, artifact_path, report_path


def _validate_internal_entry(
    project_root: Path,
    run_root: Path,
    index: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Validate one internal-v2 channel before checking a device receipt."""

    channel = str(entry.get("sensor_channel", ""))
    if channel not in RESEARCH_CHANNELS:
        raise RuntimeError(f"internal_channel_not_supported:{channel}")
    required = {
        "artifact_status": "internal_candidate",
        "evaluation_scope": "internal_32_participants",
        "validation_scope": "internal_32_participants",
        "claims_policy": "decision_support_only",
        "release_status": "internal_candidate",
        "hrv_prv_status": "GO",
        "training_admission_status": "GO",
        "development_model_status": "GO",
        "internal_test_status": "GO",
        "onnx_export_status": "GO",
        "android_device_parity_status": "PENDING",
        "internal_android_status": "PENDING_DEVICE_PARITY",
    }
    if any(entry.get(key) != value for key, value in required.items()):
        raise RuntimeError("internal_index_channel_contract_invalid")
    if any(
        key in entry
        for key in ("android_ready", "onnx_published", "clinical_validation_status")
    ):
        raise RuntimeError("internal_index_contains_legacy_status")
    model_path, model_checksum = _descriptor_path(
        project_root, run_root, entry, "model"
    )
    manifest_path, manifest_checksum = _descriptor_path(
        project_root, run_root, entry, "manifest"
    )
    parity_path, parity_checksum = _descriptor_path(
        project_root, run_root, entry, "parity"
    )
    vectors_path, vectors_checksum = _descriptor_path(
        project_root, run_root, entry, "test_vectors"
    )
    manifest = _read_json(manifest_path)
    if any(
        key in manifest
        for key in ("android_ready", "onnx_published", "clinical_validation_status")
    ):
        raise RuntimeError("internal_manifest_contains_legacy_status")
    manifest_required = {
        "schema_version": "internal-android-v2",
        "artifact_status": "internal_candidate",
        "evaluation_scope": "internal_32_participants",
        "validation_scope": "internal_32_participants",
        "claims_policy": "decision_support_only",
        "sensor_channel": channel,
        "hrv_prv_status": "GO",
        "training_admission_status": "GO",
        "development_model_status": "GO",
        "internal_test_status": "GO",
        "onnx_export_status": "GO",
        "android_device_parity_status": "PENDING",
        "internal_android_status": "PENDING_DEVICE_PARITY",
    }
    if any(manifest.get(key) != value for key, value in manifest_required.items()):
        raise RuntimeError("internal_manifest_contract_invalid")
    if manifest.get("model_version") != entry.get("model_version"):
        raise RuntimeError("internal_manifest_model_version_mismatch")
    if manifest.get("internal_test_contract_fingerprint") != entry.get(
        "internal_test_contract_fingerprint"
    ):
        raise RuntimeError("internal_manifest_test_fingerprint_mismatch")
    for metric in ("balanced_accuracy", "f1", "roc_auc"):
        value = float(manifest.get(f"{metric}_internal_test", float("nan")))
        gate = float(manifest.get(f"{metric}_gate", float("nan")))
        if not np.isfinite(value) or not np.isfinite(gate) or value < gate:
            raise RuntimeError(f"internal_test_metric_failed:{metric}")
    parity = _read_json(parity_path)
    vectors = _read_json(vectors_path)
    if parity.get("passed") is not True:
        raise RuntimeError("internal_host_parity_not_passed")
    host = _validate_host_onnx(model_path, manifest, parity, vectors)
    return {
        "sensor_channel": channel,
        "model_version": str(entry["model_version"]),
        "model_path": str(model_path),
        "model_sha256": model_checksum,
        "manifest_sha256": manifest_checksum,
        "parity_sha256": parity_checksum,
        "vectors_sha256": vectors_checksum,
        "host_validation": host,
    }


def _verify_internal_receipt(
    receipt_record: dict[str, Any],
    *,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    model_path: Path,
    apk_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    receipt_path = Path(receipt_record["path"]).resolve()
    if run_root not in receipt_path.parents:
        raise RuntimeError("internal_receipt_outside_active_run")
    receipt = receipt_record["payload"]
    verified = DeviceParityReceiptVerifier(max_abs_diff=PARITY_TOLERANCE).verify(
        receipt,
        manifest=manifest,
        model_path=model_path,
        apk_path=apk_path,
    )
    if int(receipt.get("vector_count", 0)) < MIN_PARITY_VECTORS:
        raise RuntimeError("internal_receipt_vector_count_below_100")
    if receipt.get("model_version") != entry.get("model_version"):
        raise RuntimeError("internal_receipt_model_version_mismatch")
    return {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "device_fingerprint": verified["device_fingerprint"],
        "apk_sha256": verified["apk_sha256"],
        "vector_count": int(verified["vector_count"]),
        "max_abs_diff": float(verified["max_abs_diff"]),
        "actual_sensor_acquisition": bool(verified["actual_sensor_acquisition"]),
    }


def evaluate_internal_android_readiness(
    project_root: Path,
    run_id: str,
    *,
    apk_path: Path | None,
    receipt_paths: Iterable[Path] = (),
    device_connection_status: str,
) -> dict[str, Any]:
    """Evaluate internal-v2 Android readiness without opening any data cohort."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id_format_invalid")
    project_root = project_root.resolve()
    run_root = (project_root / "data" / "artifacts" / "runs" / run_id).resolve()
    index_path = run_root / "export" / "internal" / "internal_model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"internal_index_missing:{index_path}")
    index = _read_json(index_path)
    expected_index = {
        "schema_version": "internal-android-v2",
        "run_id": run_id,
        "validation_scope": "internal_32_participants",
        "claims_policy": "decision_support_only",
    }
    if any(index.get(key) != value for key, value in expected_index.items()):
        raise RuntimeError("internal_index_contract_invalid")
    if any(
        key in index
        for key in ("android_ready", "onnx_published", "clinical_validation_status")
    ):
        raise RuntimeError("internal_index_contains_legacy_status")
    if index.get("development_participants") != [
        f"ID-{index_value:02d}" for index_value in range(1, 29)
    ] or index.get("internal_test_participants") != [
        f"ID-{index_value:02d}" for index_value in range(29, 33)
    ]:
        raise RuntimeError("internal_index_cohort_contract_invalid")
    entries = index.get("models")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("internal_index_models_missing")
    indexed_channels = [str(entry.get("sensor_channel", "")) for entry in entries]
    if len(indexed_channels) != len(set(indexed_channels)):
        raise RuntimeError("internal_index_duplicate_channels")
    receipts, receipt_errors = _receipt_by_channel(
        Path(path).resolve() for path in receipt_paths
    )
    sensor_path = _sensor_path_receipt(run_root)
    apk = apk_path.resolve() if apk_path is not None else None
    apk_error = None if apk and apk.is_file() else "internal_apk_missing"
    by_channel: dict[str, dict[str, Any]] = {}
    for entry in entries:
        channel = str(entry.get("sensor_channel", ""))
        result: dict[str, Any] = {
            "device_connection_status": device_connection_status,
            "hrv_prv_status": entry.get("hrv_prv_status"),
            "training_admission_status": entry.get("training_admission_status"),
            "development_model_status": entry.get("development_model_status"),
            "internal_test_status": entry.get("internal_test_status"),
            "onnx_export_status": entry.get("onnx_export_status"),
            "android_device_parity_status": "NO_GO",
            "sensor_path_parity_status": sensor_path["status"],
            "internal_android_status": "NO_GO",
            "reason": "internal_artifact_not_verified",
        }
        try:
            artifact = _validate_internal_entry(project_root, run_root, index, entry)
            result["artifact"] = artifact
            if device_connection_status != "READY":
                raise RuntimeError("device_connection_not_ready")
            if apk_error is not None:
                raise RuntimeError(apk_error)
            receipt_record = receipts.get(channel)
            if receipt_record is None:
                raise RuntimeError("internal_device_parity_receipt_missing")
            manifest_path, _ = _descriptor_path(
                project_root, run_root, entry, "manifest"
            )
            receipt = _verify_internal_receipt(
                receipt_record,
                entry=entry,
                manifest=_read_json(manifest_path),
                model_path=Path(artifact["model_path"]),
                apk_path=apk,
                run_root=run_root,
            )
            result.update(
                {
                    "android_device_parity_status": "GO",
                    "internal_android_status": "GO",
                    "reason": "run_scoped_internal_host_and_device_parity_verified",
                    "device_parity": receipt,
                }
            )
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            result["reason"] = str(exc)
        by_channel[channel] = result
    for channel in ALL_CHANNELS:
        by_channel.setdefault(
            channel,
            {
                "device_connection_status": device_connection_status,
                "hrv_prv_status": "NO_GO",
                "training_admission_status": "NO_GO",
                "development_model_status": "NO_GO",
                "internal_test_status": "NO_GO",
                "onnx_export_status": "NO_GO",
                "android_device_parity_status": "NO_GO",
                "sensor_path_parity_status": "NOT_APPLICABLE",
                "internal_android_status": "NO_GO",
                "reason": "channel_not_exported",
            },
        )
    statuses = [entry["internal_android_status"] for entry in by_channel.values()]
    aggregate = "GO" if "GO" in statuses else "NO_GO"
    return {
        "schema_version": "internal-android-v2",
        "run_id": run_id,
        "validation_scope": "internal_32_participants",
        "claims_policy": "decision_support_only",
        "device_connection_status": device_connection_status,
        "internal_android_status": aggregate,
        "sensor_path_parity_status": sensor_path["status"],
        "actual_sensor_acquisition": bool(sensor_path["actual_sensor_acquisition"]),
        "sensor_path_parity": sensor_path,
        "internal_android_status_by_channel": by_channel,
        "eligible_channels": [
            channel
            for channel, result in by_channel.items()
            if result["internal_android_status"] == "GO"
        ],
        "global_production_index_used": False,
        "apk_path": str(apk) if apk else None,
        "apk_sha256": _sha256(apk) if apk and apk.is_file() else None,
        "receipt_errors": receipt_errors,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def write_internal_android_readiness_report(
    project_root: Path,
    run_id: str,
    *,
    apk_path: Path | None,
    receipt_paths: Iterable[Path] = (),
    device_connection_status: str,
) -> tuple[dict[str, Any], Path, Path]:
    """Write internal-v2 readiness artifacts without changing model manifests."""

    report = evaluate_internal_android_readiness(
        project_root,
        run_id,
        apk_path=apk_path,
        receipt_paths=receipt_paths,
        device_connection_status=device_connection_status,
    )
    project_root = project_root.resolve()
    artifact_path = (
        project_root
        / "data"
        / "artifacts"
        / "runs"
        / run_id
        / "export"
        / "internal"
        / "internal_android_readiness.json"
    )
    report_path = (
        project_root
        / "reports"
        / "runs"
        / run_id
        / "android"
        / "internal_android_readiness.json"
    )
    _atomic_write_json(artifact_path, report)
    _atomic_write_json(report_path, report)
    markdown_path = report_path.with_suffix(".md")
    lines = [
        f"# Internal Android readiness — {run_id}",
        "",
        "- Validation scope: `internal_32_participants`",
        "- Claims policy: `decision_support_only`",
        f"- Device connection: `{report['device_connection_status']}`",
        f"- Aggregate internal Android status: `{report['internal_android_status']}`",
        f"- ONNX vector parity: `{report['internal_android_status']}` (100 real development vectors)",
        f"- Watch sensor-path parity: `{report['sensor_path_parity_status']}`",
        f"- Actual sensor acquisition: `{report['actual_sensor_acquisition']}`",
        "",
        "| Channel | HRV/PRV | Development | Internal test | ONNX | Vector parity | Sensor path | Android | Reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for channel, status in sorted(report["internal_android_status_by_channel"].items()):
        lines.append(
            f"| {channel} | {status['hrv_prv_status']} | "
            f"{status['development_model_status']} | {status['internal_test_status']} | "
            f"{status['onnx_export_status']} | {status['android_device_parity_status']} | "
            f"{status['sensor_path_parity_status']} | "
            f"{status['internal_android_status']} | {status['reason']} |"
        )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, artifact_path, report_path
