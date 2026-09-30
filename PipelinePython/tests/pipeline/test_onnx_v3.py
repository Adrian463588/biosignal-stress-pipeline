from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from PipelinePython.common.onnx_v3 import (
    OnnxBundleExporter,
    OnnxExportMode,
    _enforce_public_io,
    _manifest_window_contract,
    _positive_probability_output,
)
from PipelinePython.common.settings import load_pipeline_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_onnx_contract_is_single_probability_vector() -> None:
    features = np.asarray(
        [[0.0, 0.0], [0.2, 0.1], [0.8, 0.9], [1.0, 1.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1])
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    ).fit(features, labels)
    classifier = estimator.steps[-1][1]
    converted = convert_sklearn(
        estimator,
        initial_types=[("float_input", FloatTensorType([None, 2]))],
        target_opset=17,
        options={id(classifier): {"zipmap": False}},
    )
    model = _enforce_public_io(
        converted,
        _positive_probability_output(converted),
    )
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    output = session.run(
        ["stress_probability"],
        {"float_input": features},
    )[0]

    assert session.get_inputs()[0].name == "float_input"
    assert session.get_outputs()[0].name == "stress_probability"
    assert output.shape == (4,)
    np.testing.assert_allclose(
        output,
        estimator.predict_proba(features)[:, 1],
        atol=1.0e-5,
    )


def test_static_batch_one_contract_rejects_multirow_inference() -> None:
    features = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 1])
    estimator = LogisticRegression(random_state=42).fit(features, labels)
    converted = convert_sklearn(
        estimator,
        initial_types=[("float_input", FloatTensorType([1, 2]))],
        target_opset=17,
        options={id(estimator): {"zipmap": False}},
    )
    model = _enforce_public_io(
        converted,
        _positive_probability_output(converted),
        batch_size=1,
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    output = session.run(["stress_probability"], {"float_input": features[:1]})[0]

    assert session.get_inputs()[0].shape == [1, 2]
    assert session.get_outputs()[0].shape == [1]
    assert output.shape == (1,)


def test_manifest_window_contract_is_dynamic_and_complete() -> None:
    contract = _manifest_window_contract(
        {
            "selected_window_seconds": 120,
            "selected_stride_seconds": 30,
        },
        analysis_seconds=240,
    )

    assert contract == {
        "analysis_seconds": 240,
        "window_seconds": 120,
        "stride_seconds": 30,
        "expected_windows_per_session": 5,
    }


def test_candidate_export_uses_minimum_policy_not_legacy_preferred_alias(
    tmp_path: Path,
) -> None:
    config = load_pipeline_config("PipelinePython/configs/data.yaml")
    exporter = OnnxBundleExporter(tmp_path, config)
    manifest = {
        "model_version": config.model_version,
        "sensor_channel": "ppg_ir",
        "development_metrics": {
            "balanced_accuracy": 0.70,
            "f1": 0.70,
            "roc_auc": 0.70,
        },
        "holdout_metrics": {
            "balanced_accuracy": 0.70,
            "f1": 0.70,
            "roc_auc": 0.70,
        },
        "development_participants": [f"ID-{index:02d}" for index in range(1, 29)],
        "expected_development_participants": [
            f"ID-{index:02d}" for index in range(1, 29)
        ],
        "represented_development_participants": [
            f"ID-{index:02d}" for index in range(1, 29)
        ],
        "missing_development_participants": [],
        "holdout_participants": [f"ID-{index:02d}" for index in range(29, 33)],
    }

    exporter._validate_holdout_manifest(manifest, "ppg_ir")
    assessment = exporter._conservative_assessment(manifest)

    assert assessment["assessment"]["performance_tier"] == "research_usable"
    assert assessment["assessment"]["deployment_scope"] == "research"


def test_candidate_export_rejects_overstated_development_provenance(
    tmp_path: Path,
) -> None:
    config = load_pipeline_config("PipelinePython/configs/data.yaml")
    exporter = OnnxBundleExporter(tmp_path, config)
    manifest = {
        "model_version": config.model_version,
        "sensor_channel": "ppg_green",
        "development_metrics": {
            "balanced_accuracy": 0.80,
            "f1": 0.80,
            "roc_auc": 0.80,
        },
        "development_participants": [f"ID-{index:02d}" for index in range(1, 29)],
    }

    with pytest.raises(
        RuntimeError, match="represented development provenance missing"
    ):
        exporter._validate_development_manifest(manifest, "ppg_green")


def test_research_export_uses_development_bundle_without_holdout_metrics(
    tmp_path: Path,
) -> None:
    config = load_pipeline_config("PipelinePython/configs/data.yaml")
    exporter = OnnxBundleExporter(tmp_path, config, mode=OnnxExportMode.RESEARCH)
    manifest = {
        "model_version": config.model_version,
        "sensor_channel": "ppg_green",
        "development_metrics": {
            "balanced_accuracy": 0.80,
            "f1": 0.80,
            "roc_auc": 0.80,
        },
        "development_participants": [f"ID-{index:02d}" for index in range(1, 29)],
        "expected_development_participants": [
            f"ID-{index:02d}" for index in range(1, 29)
        ],
        "represented_development_participants": [
            f"ID-{index:02d}" for index in range(1, 29)
        ],
        "missing_development_participants": [],
    }

    exporter._validate_development_manifest(manifest, "ppg_green")
    with pytest.raises(RuntimeError, match="internal-test metrics"):
        exporter._validate_holdout_manifest(manifest, "ppg_green")
    bundle, _, _ = exporter._candidate_paths("ppg_green")
    assert bundle.name == "development_candidate"
    assert "research" in exporter._published_export_dir("ppg_green").parts


def test_internal_export_uses_development_candidate_and_run_scoped_internal_root(
    tmp_path: Path,
) -> None:
    config = load_pipeline_config("PipelinePython/configs/data.yaml")
    exporter = OnnxBundleExporter(tmp_path, config, mode=OnnxExportMode.INTERNAL)

    bundle, _, _ = exporter._candidate_paths("ppg_green")

    assert bundle.name == "development_candidate"
    assert "internal" in exporter._published_export_dir("ppg_green").parts


def test_research_export_resolves_exact_run_scoped_training_variant(
    tmp_path: Path,
) -> None:
    loaded = load_pipeline_config("PipelinePython/configs/data.yaml")
    run_root = tmp_path / "data" / "artifacts" / "runs" / "RUN-1"
    artifacts = dict(loaded.artifacts)
    artifacts.update(
        {
            "run_id": "RUN-1",
            "artifacts_root": str(run_root),
            "gold_dir": str(run_root / "gold"),
            "gold_features": str(run_root / "gold" / "gold_features_v3.parquet"),
        }
    )
    exporter = OnnxBundleExporter(
        tmp_path,
        replace(loaded, artifacts=artifacts),
        mode=OnnxExportMode.RESEARCH,
    )
    selected = run_root / "gold" / "training_variants" / "w240_s240" / "gold.parquet"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"selected-real-gold-fixture")
    manifest = {
        "gold_contract": {
            "features_path": str(selected),
            "features_sha256": _sha256(selected),
        }
    }

    assert exporter._candidate_gold_path(manifest) == selected.resolve()


def test_research_export_rejects_candidate_gold_outside_current_run(
    tmp_path: Path,
) -> None:
    loaded = load_pipeline_config("PipelinePython/configs/data.yaml")
    run_root = tmp_path / "data" / "artifacts" / "runs" / "RUN-1"
    artifacts = dict(loaded.artifacts)
    artifacts.update(
        {
            "run_id": "RUN-1",
            "artifacts_root": str(run_root),
            "gold_dir": str(run_root / "gold"),
            "gold_features": str(run_root / "gold" / "gold_features_v3.parquet"),
        }
    )
    exporter = OnnxBundleExporter(
        tmp_path,
        replace(loaded, artifacts=artifacts),
        mode=OnnxExportMode.RESEARCH,
    )
    stale = tmp_path / "stale" / "gold.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="outside_current_run"):
        exporter._candidate_gold_path(
            {
                "gold_contract": {
                    "features_path": str(stale),
                    "features_sha256": _sha256(stale),
                }
            }
        )
