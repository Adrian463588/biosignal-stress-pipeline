from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

from PipelinePython.common.android_vectors import (
    AndroidVectorGenerator,
    DeviceParityPromoter,
    DeviceParityReceiptVerifier,
    _assert_promotable_vectors,
    _assert_vector_contract,
    _receipt_sha256,
)


def _vectors(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "python_vectors_passed": 100,
        "python_vector_status": "passed",
        "android_instrumentation_status": "passed",
        "android_ready": True,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"python_vectors_passed": 99}, "Python vector evidence"),
        ({"python_vector_status": "pending"}, "Python vector status"),
        ({"android_instrumentation_status": "pending"}, "instrumentation"),
        ({"android_ready": False}, "Android-ready"),
    ],
)
def test_pending_vector_evidence_cannot_promote_index(
    mutation: dict[str, object], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _assert_promotable_vectors(_vectors(**mutation), "v4/ppg_green")


def test_passed_vector_evidence_is_promotable() -> None:
    _assert_promotable_vectors(_vectors(), "v4/ppg_green")


def test_gold_contract_rejects_unknown_participant_and_legacy_phase(tmp_path) -> None:
    generator = object.__new__(AndroidVectorGenerator)
    generator.root = tmp_path.resolve()
    generator.paths = type(
        "Paths", (), {"gold_features_path": tmp_path / "gold.parquet"}
    )()
    frame = pd.DataFrame(
        {
            "participant_id": ["ID-33"],
            "phase": ["routine"],
            "sensor_channel": ["ppg_green"],
            "window_id": ["w0"],
            "window_start_utc": ["2026-01-01T00:00:00Z"],
            "window_end_utc": ["2026-01-01T00:03:00Z"],
            "sampling_rate_hz": [25.0],
            "source_file": ["silver.parquet"],
            "source_checksum": ["0" * 64],
            "source_silver_path": ["silver.parquet"],
            "source_silver_sha256": ["0" * 64],
            "quality_tier": ["good"],
        }
    )
    frame.to_parquet(generator.paths.gold_features_path, index=False)
    with pytest.raises(RuntimeError, match="unknown participants"):
        generator._load_gold("ppg_green", {})


def test_gold_contract_does_not_fallback_to_legacy_table(tmp_path) -> None:
    generator = object.__new__(AndroidVectorGenerator)
    generator.root = tmp_path.resolve()
    generator.paths = type(
        "Paths", (), {"gold_features_path": tmp_path / "run" / "gold.parquet"}
    )()
    with pytest.raises(FileNotFoundError, match="legacy Gold"):
        generator._load_gold("ppg_green", {})


def test_window_loader_requires_canonical_clean_value(tmp_path: Path) -> None:
    generator = object.__new__(AndroidVectorGenerator)
    generator.root = tmp_path.resolve()
    generator.config = type("Config", (), {"artifacts": {}})()
    source = tmp_path / "silver.parquet"
    pd.DataFrame(
        {
            "timestamp_raw_utc": pd.date_range(
                "2026-01-01T00:00:00Z", periods=10, freq="40ms"
            ),
            "raw_value": range(10),
        }
    ).to_parquet(source, index=False)
    row = pd.Series(
        {
            "source_file": str(source),
            "source_silver_path": str(source),
            "source_silver_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "participant_id": "ID-01",
            "phase": "relax",
            "sensor_channel": "ppg_green",
            "window_start_utc": "2026-01-01T00:00:00Z",
            "window_end_utc": "2026-01-01T00:03:00Z",
            "sampling_rate_hz": 25.0,
            "window_id": "w0",
        }
    )

    with pytest.raises(RuntimeError, match="clean_value"):
        generator._load_window(row, {})


def test_window_loader_never_falls_back_to_raw_source_file(tmp_path: Path) -> None:
    generator = object.__new__(AndroidVectorGenerator)
    generator.root = tmp_path.resolve()
    generator.config = type("Config", (), {"artifacts": {}})()
    raw = tmp_path / "raw.csv"
    raw.write_text("timestamp,value\n0,1\n", encoding="utf-8")
    row = pd.Series(
        {
            "source_file": str(raw),
            "participant_id": "ID-01",
            "phase": "relax",
            "sensor_channel": "ppg_green",
            "window_start_utc": "2026-01-01T00:00:00Z",
            "window_end_utc": "2026-01-01T00:03:00Z",
            "sampling_rate_hz": 25.0,
            "window_id": "w0",
        }
    )

    with pytest.raises(RuntimeError, match="source_silver_path"):
        generator._load_window(row, {})


def test_vector_contract_requires_exactly_94_real_and_6_rejections() -> None:
    cases = [
        {"case_id": f"real-{index}", "kind": "valid_raw_window"} for index in range(94)
    ] + [
        {"case_id": f"reject-{index}", "kind": "quality_rejection"}
        for index in range(6)
    ]
    _assert_vector_contract(cases, valid_count=94, rejection_count=6)

    with pytest.raises(RuntimeError, match="exactly 100"):
        _assert_vector_contract(cases[:-1], valid_count=94, rejection_count=6)


def _receipt(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "device_fingerprint": "samsung/test/device:1",
        "apk_sha256": "a" * 64,
        "onnx_sha256": "b" * 64,
        "model_version": "stress-v4-test",
        "sensor_channel": "ppg_green",
        "vector_count": 100,
        "max_abs_diff": 1.0e-6,
        "timestamp_utc": "2026-08-09T12:00:00Z",
        "actual_sensor_acquisition": False,
    }
    payload.update(overrides)
    payload["receipt_sha256"] = _receipt_sha256(payload)
    return payload


def test_device_receipt_verifier_checks_hash_and_apk(tmp_path: Path) -> None:
    apk = tmp_path / "app.apk"
    apk.write_bytes(b"apk")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    receipt = _receipt(
        apk_sha256=hashlib.sha256(apk.read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    )
    manifest = {
        "model_version": receipt["model_version"],
        "sensor_channel": receipt["sensor_channel"],
        "onnx_sha256": receipt["onnx_sha256"],
        "android_ready": False,
    }

    verified = DeviceParityReceiptVerifier(max_abs_diff=1.0e-5).verify(
        receipt, manifest=manifest, model_path=model, apk_path=apk
    )

    assert verified["receipt_sha256"] == receipt["receipt_sha256"]
    tampered = dict(receipt, vector_count=99)
    with pytest.raises(RuntimeError, match="receipt_sha256"):
        DeviceParityReceiptVerifier(max_abs_diff=1.0e-5).verify(
            tampered, manifest=manifest, model_path=model, apk_path=apk
        )


def test_red_receipt_requires_actual_red_acquisition(tmp_path: Path) -> None:
    apk = tmp_path / "app.apk"
    apk.write_bytes(b"apk")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    receipt = _receipt(
        sensor_channel="ppg_red",
        apk_sha256=hashlib.sha256(apk.read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        actual_sensor_acquisition=False,
    )
    manifest = {
        "model_version": receipt["model_version"],
        "sensor_channel": "ppg_red",
        "onnx_sha256": receipt["onnx_sha256"],
        "android_ready": False,
    }

    with pytest.raises(RuntimeError, match="actual Red acquisition"):
        DeviceParityReceiptVerifier(max_abs_diff=1.0e-5).verify(
            receipt, manifest=manifest, model_path=model, apk_path=apk
        )


def test_physical_receipt_promotes_copy_not_candidate_manifest(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    version = "stress-v4-test"
    channel = "ppg_green"
    base = export_dir / version / channel
    base.mkdir(parents=True)
    model = base / f"stress_{channel}_fp32.onnx"
    model.write_bytes(b"onnx")
    apk = tmp_path / "app.apk"
    apk.write_bytes(b"apk")
    manifest = {
        "schema_version": "1.0.0",
        "model_version": version,
        "sensor_channel": channel,
        "model_family": "extra_trees",
        "release_status": "release_candidate",
        "gate_passed": True,
        "android_ready": False,
        "onnx_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "preprocessing_version": "v4.0.0",
        "feature_version": "v4.0.0",
        "analysis_seconds": 240,
        "window_seconds": 180,
        "stride_seconds": 10,
        "expected_windows_per_session": 7,
        "source_rate_hz": 25.0,
        "model_rate_hz": 25.0,
        "feature_order": ["median_hr_bpm"],
        "probability_threshold": 0.5,
        "onnx_input": {"name": "float_input"},
        "onnx_output": {"name": "stress_probability"},
        "balanced_accuracy_holdout": 0.85,
        "f1_holdout": 0.82,
        "roc_auc_holdout": 0.87,
        "conservative_metrics": {
            "balanced_accuracy": 0.85,
            "f1": 0.82,
            "roc_auc": 0.87,
        },
        "calibration_version": "platt-v1-dev-oof",
        "performance_policy_version": "ml_dl_tiered_v1",
        "performance_policy_sha256": "c" * 64,
        "performance_tier": "preferred",
        "deployment_scope": "standard",
        "research_disclaimer": "Model penelitian — validasi internal terbatas. Bukan diagnosis klinis.",
        "disclaimer_required": False,
        "validation_cohort_ids": ["ID-29", "ID-30", "ID-31", "ID-32"],
    }
    from PipelinePython.common.onnx_v3 import _manifest_sha256

    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    (base / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (base / "onnx_parity.json").write_text("{}", encoding="utf-8")
    cases = [
        {"case_id": f"real-{index}", "kind": "valid_raw_window"} for index in range(94)
    ] + [
        {"case_id": f"reject-{index}", "kind": "quality_rejection"}
        for index in range(6)
    ]
    (base / "test_vectors.json").write_text(
        json.dumps(
            {
                "python_vectors_passed": 100,
                "python_vector_status": "passed",
                "android_instrumentation_status": "pending",
                "android_ready": False,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    receipt = _receipt(
        apk_sha256=hashlib.sha256(apk.read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    promoter = object.__new__(DeviceParityPromoter)
    promoter.root = tmp_path
    promoter.export_dir = export_dir
    promoter.config = SimpleNamespace(
        model_version=version,
        onnx=SimpleNamespace(max_abs_diff=1.0e-5),
        android=SimpleNamespace(valid_vector_count=94, edge_vector_count=6),
    )
    promoter.verifier = DeviceParityReceiptVerifier(max_abs_diff=1.0e-5)

    result = promoter.promote(receipt_path, apk)

    assert result["status"] == "android_ready"
    candidate = json.loads((base / "model_manifest.json").read_text())
    production = json.loads((base / "production_model_manifest.json").read_text())
    index = json.loads((export_dir / "model_index.json").read_text())
    assert candidate["android_ready"] is False
    assert production["android_ready"] is True
    assert index["models"][0]["sensor_channel"] == channel
    assert index["models"][0]["calibration_version"] == "platt-v1-dev-oof"
    assert index["models"][0]["performance_tier"] == "preferred"
    assert index["models"][0]["balanced_accuracy_conservative"] == 0.85
