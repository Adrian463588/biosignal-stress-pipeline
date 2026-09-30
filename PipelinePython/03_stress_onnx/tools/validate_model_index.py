"""Locally validate the canonical accepted-model index and referenced artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPORT_ROOT = (ROOT / "data" / "artifacts" / "export").resolve()
INDEX_PATH = EXPORT_ROOT / "model_index.json"
REQUIRED_FILES = {
    "model",
    "manifest",
    "parity",
    "test_vectors",
    "device_parity_receipt",
}
RESEARCH_REQUIRED_FILES = {"model", "manifest", "parity", "test_vectors"}
PARITY_POOL = {"ID-29", "ID-30", "ID-31", "ID-32"}
DEVELOPMENT_POOL = {f"ID-{participant:02d}" for participant in range(1, 29)}
METRIC_NAMES = ("balanced_accuracy", "f1", "roc_auc")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(
    relative: str,
    base_root: Path | None = None,
    allowed_root: Path | None = None,
) -> Path:
    root = (base_root or EXPORT_ROOT).resolve()
    boundary = (allowed_root or root).resolve()
    path = (root / relative).resolve()
    if boundary != path and boundary not in path.parents:
        raise RuntimeError(f"artifact path escapes export root: {relative}")
    if not path.is_file():
        raise RuntimeError(f"missing indexed artifact: {relative}")
    return path


def _decimal(value: object, name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:
        raise RuntimeError(f"{name} is not numeric") from exc


def _validate_performance_policy(
    candidate: dict[str, object], manifest: dict[str, object], identity: str
) -> None:
    policy = manifest.get("performance_policy")
    if not isinstance(policy, dict):
        raise RuntimeError(f"{identity}: performance policy is missing")
    minimum = policy.get("minimum_research")
    preferred = policy.get("preferred")
    if not isinstance(minimum, dict) or not isinstance(preferred, dict):
        raise RuntimeError(f"{identity}: performance thresholds are missing")
    policy_version = policy.get("version")
    policy_sha256 = manifest.get("performance_policy_sha256")
    if (
        not policy_version
        or not isinstance(policy_sha256, str)
        or len(policy_sha256) != 64
        or policy.get("accuracy_is_gate") is not False
        or candidate.get("performance_policy_version") != policy_version
        or candidate.get("performance_policy_sha256") != policy_sha256
        or manifest.get("performance_policy_version") != policy_version
    ):
        raise RuntimeError(f"{identity}: performance policy contract is invalid")

    holdout = {
        name: _decimal(candidate.get(f"{name}_holdout"), f"holdout {name}")
        for name in METRIC_NAMES
    }
    conservative = {
        name: _decimal(candidate.get(f"{name}_conservative"), f"conservative {name}")
        for name in METRIC_NAMES
    }
    minimum_passed = all(
        holdout[name] >= _decimal(minimum.get(name), f"minimum {name}")
        and conservative[name] >= _decimal(minimum.get(name), f"minimum {name}")
        for name in METRIC_NAMES
    )
    preferred_passed = all(
        conservative[name] >= _decimal(preferred.get(name), f"preferred {name}")
        for name in METRIC_NAMES
    )
    expected_tier = (
        "below_minimum"
        if not minimum_passed
        else "preferred"
        if preferred_passed
        else "research_usable"
    )
    expected_scope = {
        "below_minimum": "none",
        "research_usable": "research",
        "preferred": "standard",
    }[expected_tier]
    if (
        not minimum_passed
        or candidate.get("performance_tier") != expected_tier
        or candidate.get("deployment_scope") != expected_scope
        or manifest.get("performance_tier") != expected_tier
        or manifest.get("deployment_scope") != expected_scope
    ):
        raise RuntimeError(f"{identity}: performance policy gate failed")
    if expected_tier == "research_usable" and (
        candidate.get("research_disclaimer") != policy.get("research_disclaimer")
        or manifest.get("research_disclaimer") != policy.get("research_disclaimer")
        or manifest.get("disclaimer_required") is not True
    ):
        raise RuntimeError(f"{identity}: research disclaimer contract is invalid")


def _validate_device_receipt(
    candidate: dict[str, object],
    manifest: dict[str, object],
    receipt_path: Path,
    model_path: Path,
    identity: str,
) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    sensor_channel = candidate.get("sensor_channel")
    if (
        manifest.get("device_parity_receipt_sha256") != receipt.get("receipt_sha256")
        or receipt.get("model_version") != candidate.get("model_version")
        or receipt.get("sensor_channel") != sensor_channel
        or receipt.get("onnx_sha256") != manifest.get("onnx_sha256")
        or receipt.get("onnx_sha256") != _sha256(model_path)
        or receipt.get("vector_count") != 100
        or _decimal(receipt.get("max_abs_diff"), "device parity max_abs_diff")
        >= Decimal("0.00001")
        or (
            sensor_channel == "ppg_red"
            and receipt.get("actual_sensor_acquisition") is not True
        )
    ):
        raise RuntimeError(f"{identity}: device parity receipt mismatch")


def _index_context(
    index_path: Path | None, index: dict[str, object]
) -> tuple[Path, Path]:
    """Return the descriptor root and safety boundary for the selected index."""

    if index_path is None:
        return EXPORT_ROOT.resolve(), EXPORT_ROOT.resolve()

    selected = index_path.resolve()
    runs_root = (ROOT / "data" / "artifacts" / "runs").resolve()
    if runs_root != selected and runs_root not in selected.parents:
        raise RuntimeError("explicit model index must be under data/artifacts/runs")
    run_parts = selected.relative_to(runs_root).parts
    if len(run_parts) < 2 or not run_parts[0].endswith("Z"):
        raise RuntimeError("explicit model index must include a run directory")
    run_id = run_parts[0]
    if index.get("run_id") != run_id:
        raise RuntimeError("model index run_id does not match its directory")
    run_root = (runs_root / run_id).resolve()
    return ROOT.resolve(), run_root


def _resolve_index_artifacts(
    index: dict[str, object],
    descriptor_root: Path,
    artifact_boundary: Path,
    required_files: set[str],
    identity: str,
) -> dict[str, Path]:
    files = index.get("files")
    if not isinstance(files, dict) or set(files) != required_files:
        raise RuntimeError(f"{identity}: incomplete or unexpected artifact set")

    resolved: dict[str, Path] = {}
    for kind, descriptor in files.items():
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"{identity}: invalid {kind} descriptor")
        path = _resolve(
            str(descriptor.get("path", "")),
            descriptor_root,
            artifact_boundary,
        )
        expected = str(descriptor.get("sha256", ""))
        if len(expected) != 64 or _sha256(path) != expected:
            raise RuntimeError(f"{identity}: checksum failed for {path.name}")
        resolved[kind] = path
    return resolved


def _validate_research_manifest(
    candidate: dict[str, object],
    manifest: dict[str, object],
    identity: str,
) -> None:
    required = {
        "schema_version": "onnx-research-v1",
        "artifact_status": "research_candidate",
        "evaluation_scope": "development_only",
        "deployment_scope": "research",
        "clinical_validation_status": "external_validation_required",
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"{identity}: research manifest contract is invalid")
    if (
        manifest.get("release_status") != candidate.get("release_status")
        or manifest.get("android_ready") is not False
        or manifest.get("holdout_evaluated") is not False
        or any(key in manifest for key in ("holdout_metrics", "holdout_raw_metrics"))
    ):
        raise RuntimeError(
            f"{identity}: research manifest contains release/holdout evidence"
        )

    policy = manifest.get("performance_policy")
    development = manifest.get("development_metrics")
    minimum = policy.get("minimum_research") if isinstance(policy, dict) else None
    if not isinstance(development, dict) or not isinstance(minimum, dict):
        raise RuntimeError(f"{identity}: research performance policy is incomplete")
    for metric in METRIC_NAMES:
        value = _decimal(development.get(metric), f"development {metric}")
        gate = _decimal(minimum.get(metric), f"minimum {metric}")
        if value < gate:
            raise RuntimeError(
                f"{identity}: research performance gate failed for {metric}"
            )

    validation_cohort = manifest.get("validation_cohort_ids")
    if (
        not isinstance(validation_cohort, list)
        or len(validation_cohort) < 26
        or len(set(validation_cohort)) != len(validation_cohort)
        or set(validation_cohort) - DEVELOPMENT_POOL
    ):
        raise RuntimeError(f"{identity}: research cohort is not development-only")
    if (
        manifest.get("analysis_seconds") != 240
        or manifest.get("window_seconds") != 180
        or manifest.get("stride_seconds") != 10
        or manifest.get("expected_windows_per_session") != 7
    ):
        raise RuntimeError(f"{identity}: research window contract is not canonical")

    sensor_channel = candidate.get("sensor_channel")
    if manifest.get("sensor_channel") != sensor_channel:
        raise RuntimeError(f"{identity}: research sensor channel mismatch")
    for rate_name in ("source_rate_hz", "model_rate_hz"):
        if (
            not isinstance(manifest.get(rate_name), (int, float))
            or manifest[rate_name] <= 0
        ):
            raise RuntimeError(f"{identity}: research {rate_name} must be positive")

    inputs = manifest.get("inputs")
    output = manifest.get("output")
    if (
        not isinstance(inputs, list)
        or len(inputs) != 1
        or not isinstance(inputs[0], dict)
        or inputs[0].get("dtype") != "float32"
        or not isinstance(inputs[0].get("shape"), list)
        or len(inputs[0]["shape"]) != 2
        or inputs[0]["shape"][0] != 1
        or not isinstance(inputs[0]["shape"][1], int)
        or inputs[0]["shape"][1] <= 0
        or not isinstance(output, dict)
        or output.get("name") != "stress_probability"
        or output.get("dtype") != "float32"
        or output.get("shape") != [1]
    ):
        raise RuntimeError(f"{identity}: research tensor contract is invalid")


def _validate_research_vectors(
    candidate: dict[str, object],
    manifest: dict[str, object],
    resolved: dict[str, Path],
    identity: str,
) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    parity = json.loads(resolved["parity"].read_text(encoding="utf-8"))
    max_diff = parity.get("max_abs_diff", parity.get("max_diff"))
    if (
        parity.get("passed") is not True
        or parity.get("onnx_checker_passed") is not True
        or parity.get("real_batch1_inference_passed") is not True
        or int(parity.get("checked_samples", 0)) < 100
        or _decimal(max_diff, "research parity max_abs_diff") >= Decimal("0.00001")
    ):
        raise RuntimeError(f"{identity}: research ONNX parity evidence is incomplete")

    model = onnx.load(str(resolved["model"]))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        str(resolved["model"]), providers=["CPUExecutionProvider"]
    )
    input_spec = manifest["inputs"][0]
    input_name = str(input_spec["name"])
    output_name = str(manifest["output"]["name"])
    if [item.name for item in session.get_inputs()] != [
        input_name
    ] or output_name not in {item.name for item in session.get_outputs()}:
        raise RuntimeError(f"{identity}: research ONNX I/O names differ from manifest")

    vectors = json.loads(resolved["test_vectors"].read_text(encoding="utf-8"))
    cases = vectors.get("cases")
    if (
        vectors.get("evaluation_scope") != "development_only"
        or vectors.get("synthetic_data_used") is not False
        or int(vectors.get("count", 0)) < 100
        or not isinstance(cases, list)
        or len(cases) < 100
        or vectors.get("model_version") not in (None, candidate.get("model_version"))
        or vectors.get("sensor_channel") not in (None, candidate.get("sensor_channel"))
    ):
        raise RuntimeError(
            f"{identity}: research vectors are not real development vectors"
        )

    expected_shape = tuple(int(value) for value in input_spec["shape"])
    for case in cases[:100]:
        if case.get("participant_id") not in DEVELOPMENT_POOL:
            raise RuntimeError(
                f"{identity}: research vector includes holdout participant"
            )
        raw_input = case.get("input")
        expected_native = case.get("native_probability")
        expected_onnx = case.get("onnx_probability")
        if raw_input is None or expected_native is None or expected_onnx is None:
            raise RuntimeError(f"{identity}: research vector is incomplete")
        value = np.asarray(raw_input, dtype=np.float32)
        if value.size != int(np.prod(expected_shape)) or not np.isfinite(value).all():
            raise RuntimeError(f"{identity}: research vector input is invalid")
        native = float(expected_native)
        recorded_onnx = float(expected_onnx)
        if not np.isfinite(native) or not np.isfinite(recorded_onnx):
            raise RuntimeError(f"{identity}: research vector probability is non-finite")
        if abs(native - recorded_onnx) >= 1e-5:
            raise RuntimeError(f"{identity}: recorded research parity failed")
        actual = float(
            np.asarray(
                session.run([output_name], {input_name: value.reshape(expected_shape)})[
                    0
                ]
            ).reshape(-1)[0]
        )
        if not np.isfinite(actual) or abs(actual - recorded_onnx) >= 1e-5:
            raise RuntimeError(f"{identity}: research vector replay failed")


def _validate_research_index(
    index: dict[str, object],
    descriptor_root: Path,
    artifact_boundary: Path,
) -> None:
    if (
        index.get("schema_version") != "1.0.0"
        or index.get("artifact_status") != "research_candidate"
        or index.get("evaluation_scope") != "development_only"
        or index.get("release_status") != "research_candidate"
        or index.get("deployment_scope") != "research"
        or index.get("android_ready") is not False
        or not isinstance(index.get("models"), list)
        or not index["models"]
    ):
        raise RuntimeError("invalid research model index contract")

    for candidate in index["models"]:
        identity = f"{candidate.get('model_version')}/{candidate.get('sensor_channel')}"
        if (
            candidate.get("artifact_status") != "research_candidate"
            or candidate.get("evaluation_scope") != "development_only"
            or candidate.get("release_status")
            != "blocked_existing_holdout_contract_mismatch"
            or candidate.get("deployment_scope") != "research"
            or candidate.get("android_ready") is not False
        ):
            raise RuntimeError(f"{identity}: research candidate contract is invalid")
        resolved = _resolve_index_artifacts(
            candidate,
            descriptor_root,
            artifact_boundary,
            RESEARCH_REQUIRED_FILES,
            identity,
        )
        manifest = json.loads(resolved["manifest"].read_text(encoding="utf-8"))
        _validate_research_manifest(candidate, manifest, identity)
        _validate_research_vectors(candidate, manifest, resolved, identity)


def validate(index_path: Path | None = None, mode: str = "production") -> None:
    selected_index = (index_path or INDEX_PATH).resolve()
    index = json.loads(selected_index.read_text(encoding="utf-8"))
    descriptor_root, artifact_boundary = _index_context(index_path, index)
    if mode == "research":
        _validate_research_index(index, descriptor_root, artifact_boundary)
        return
    if mode != "production":
        raise ValueError(f"unsupported validation mode: {mode}")
    if index.get("schema_version") != "1.0.0" or not isinstance(
        index.get("models"), list
    ):
        raise RuntimeError("invalid canonical model index schema")
    models = index["models"]
    if not models and not str(index.get("release_status", "")).startswith("blocked_"):
        raise RuntimeError(
            "an empty accepted index must declare a blocked release status"
        )

    for candidate in models:
        identity = f"{candidate.get('model_version')}/{candidate.get('sensor_channel')}"
        if candidate.get("release_status") != "accepted":
            raise RuntimeError(f"{identity}: non-accepted candidate is listed")
        if candidate.get("android_ready") is not True:
            raise RuntimeError(f"{identity}: Android release evidence is not complete")
        if (
            not isinstance(candidate.get("source_rate_hz"), (int, float))
            or candidate["source_rate_hz"] <= 0
            or not isinstance(candidate.get("model_rate_hz"), (int, float))
            or candidate["model_rate_hz"] <= 0
        ):
            raise RuntimeError(f"{identity}: positive source/model rates are required")
        if set(candidate.get("cohort_ids", [])) != PARITY_POOL:
            raise RuntimeError(f"{identity}: accepted evidence is not ID-29..ID-32")
        resolved = _resolve_index_artifacts(
            candidate, descriptor_root, artifact_boundary, REQUIRED_FILES, identity
        )

        manifest = json.loads(resolved["manifest"].read_text(encoding="utf-8"))
        if manifest.get("release_status") != "accepted":
            raise RuntimeError(f"{identity}: manifest is not accepted")
        if manifest.get("android_ready") is not True:
            raise RuntimeError(f"{identity}: manifest is not Android-ready")
        _validate_performance_policy(candidate, manifest, identity)
        _validate_device_receipt(
            candidate,
            manifest,
            resolved["device_parity_receipt"],
            resolved["model"],
            identity,
        )
        inputs = manifest.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise RuntimeError(f"{identity}: manifest has no named inputs")
        manifest_input_shapes: dict[str, tuple[int, ...]] = {}
        for tensor in inputs:
            name = tensor.get("name")
            shape = tensor.get("shape")
            if not name or not isinstance(shape, list) or shape[0] != 1:
                raise RuntimeError(
                    f"{identity}: inputs must be named with static batch size 1"
                )
            manifest_input_shapes[str(name)] = tuple(
                int(dimension) for dimension in shape
            )

        parity = json.loads(resolved["parity"].read_text(encoding="utf-8"))
        max_diff = parity.get("max_diff", parity.get("parity_max_diff"))
        if _decimal(max_diff, "parity max_diff") >= Decimal("0.00001"):
            raise RuntimeError(f"{identity}: parity max_diff must be < 1e-5")
        if parity.get("onnx_checker_passed") is not True:
            raise RuntimeError(f"{identity}: ONNX checker evidence is missing")
        if parity.get("real_batch1_inference_passed") is not True:
            raise RuntimeError(
                f"{identity}: real batch-1 inference evidence is missing"
            )

        import onnx
        import numpy as np
        import onnxruntime as ort

        onnx.checker.check_model(onnx.load(str(resolved["model"])))
        vectors = json.loads(resolved["test_vectors"].read_text(encoding="utf-8"))
        if vectors.get("python_vectors_passed") != 100:
            raise RuntimeError(f"{identity}: Python vector evidence is incomplete")
        if vectors.get("python_vector_status") != "passed":
            raise RuntimeError(f"{identity}: Python vector status is not passed")
        if vectors.get("android_instrumentation_status") != "passed":
            raise RuntimeError(
                f"{identity}: Android instrumentation evidence is not passed"
            )
        if vectors.get("android_ready") is not True:
            raise RuntimeError(f"{identity}: vector artifact is not Android-ready")
        cases = vectors.get("cases")
        if not isinstance(cases, list) or len(cases) != 100:
            raise RuntimeError(
                f"{identity}: exactly 100 deterministic vectors are required"
            )
        session = ort.InferenceSession(
            str(resolved["model"]), providers=["CPUExecutionProvider"]
        )
        metadata = {item.name: item for item in session.get_inputs()}
        vector_input_name = str(vectors.get("input_name", ""))
        output_name = str(vectors.get("output_name", ""))
        if set(metadata) != set(manifest_input_shapes):
            raise RuntimeError(
                f"{identity}: manifest input names differ from ONNX inputs"
            )
        if vector_input_name and vector_input_name not in metadata:
            raise RuntimeError(
                f"{identity}: vector input name differs from ONNX inputs"
            )
        if output_name not in {item.name for item in session.get_outputs()}:
            raise RuntimeError(
                f"{identity}: vector output name differs from ONNX outputs"
            )
        for case in cases:
            kind = str(case.get("kind", ""))
            if kind in {
                "quality_rejection",
                "timestamp_rejection",
                "schema_rejection",
                "model_rejection",
            }:
                if not isinstance(case.get("expected_status"), str):
                    raise RuntimeError(
                        f"{identity}: rejection vector has no expected status"
                    )
                if any(
                    case.get(name) is not None
                    for name in (
                        "expected_probability",
                        "onnx_probability",
                        "expected_stress_probability",
                    )
                ):
                    raise RuntimeError(
                        f"{identity}: rejection vector must not contain a probability"
                    )
                continue
            raw_inputs = case.get("inputs")
            single_input_shorthand = raw_inputs is None and bool(vector_input_name)
            if single_input_shorthand:
                raw_inputs = {vector_input_name: case.get("input")}
            if not isinstance(raw_inputs, dict):
                raise RuntimeError(f"{identity}: vector case has no named inputs")
            if set(raw_inputs) != set(metadata):
                raise RuntimeError(
                    f"{identity}: vector input names differ from ONNX inputs"
                )
            feeds = {
                name: np.asarray(raw_inputs[name], dtype=np.float32)
                for name in metadata
            }
            if single_input_shorthand and len(feeds) == 1:
                name = next(iter(feeds))
                expected_shape = manifest_input_shapes[name]
                if feeds[name].size == int(np.prod(expected_shape)):
                    feeds[name] = feeds[name].reshape(expected_shape)
            for name, value in feeds.items():
                if value.shape != manifest_input_shapes[name]:
                    raise RuntimeError(
                        f"{identity}: vector input shape mismatch for {name}"
                    )
                if not np.isfinite(value).all():
                    raise RuntimeError(
                        f"{identity}: vector input contains non-finite values for {name}"
                    )
            actual = float(
                np.asarray(session.run([output_name], feeds)[0]).reshape(-1)[0]
            )
            expected_value = case.get(
                "expected_probability",
                case.get("onnx_probability", case.get("expected_stress_probability")),
            )
            if expected_value is None:
                raise RuntimeError(
                    f"{identity}: vector case has no expected probability"
                )
            expected = float(expected_value)
            if not np.isfinite(actual) or abs(actual - expected) >= 1e-5:
                raise RuntimeError(f"{identity}: deterministic vector replay failed")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("index_path", nargs="?")
        parser.add_argument(
            "--mode", choices=("production", "research"), default="production"
        )
        arguments = parser.parse_args()
        selected = Path(arguments.index_path) if arguments.index_path else None
        validate(selected, mode=arguments.mode)
        print(
            f"{arguments.mode.capitalize()} model index and referenced artifacts are valid"
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"MODEL INDEX CHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
