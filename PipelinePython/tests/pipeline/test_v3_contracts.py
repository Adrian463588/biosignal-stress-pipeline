from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PipelinePython.common.contracts import (
    CandidateBundleManifest,
    GoldWindowContract,
    ManifestRecord,
    OnnxParityResult,
    SignalFrameContract,
    assert_feature_tensor,
    assert_stress_probability,
)
from PipelinePython.common.local_pipeline import LocalBiosignalPipeline
from PipelinePython.common.settings import (
    load_pipeline_config,
    pipeline_paths_from_config,
)


def test_local_config_is_frozen_and_uses_sensor_specific_sources() -> None:
    config = load_pipeline_config()

    assert config.raw_source.renamed_dir.name == "DataRenamedPartisipan"
    assert config.raw_source.trimmed_dir.name == "DataTrimmedPartisipan"
    assert config.raw_source.ecg_source == "renamed_complete"
    assert config.raw_source.ppg_source == "trimmed_phase"
    assert config.raw_source.trimmed_ecg_role == "audit_only"
    assert config.feature_version == "v4.0.0_robust_channel_independent"
    assert config.synchronization.validation_window_seconds == 240
    assert config.window.source_phase_duration_seconds == 300
    assert config.window.edge_trim_start_seconds == 30
    assert config.window.edge_trim_end_seconds == 30
    assert config.window.analysis_phase_duration_seconds == 240
    assert config.window.validation_window_seconds == 240
    assert config.window.gold_window_seconds == 180
    assert config.window.gold_stride_seconds == 10
    assert config.window.size_seconds == 180
    assert config.window.step_seconds == 10
    assert config.training.training_channels == ("ppg_green", "ppg_ir", "ppg_red")
    assert set(config.android.deployable_sensors) == {
        "ppg_green",
        "ppg_ir",
        "ppg_red",
    }
    assert config.android.channel_priority == ("ppg_green", "ppg_ir", "ppg_red")
    assert config.training.development_participants == 28
    assert config.training.parity_pool_participants == 4
    assert config.training.parity_pool_participant_ids == tuple(
        f"ID-{index:02d}" for index in range(29, 33)
    )
    assert config.training.min_balanced_accuracy == 0.80
    assert config.training.min_f1 == 0.78
    assert config.training.min_roc_auc == 0.83
    with pytest.raises(FrozenInstanceError):
        config.model_version = "changed"  # type: ignore[misc]


def test_run_overlay_keeps_canonical_policy_and_changes_only_artifacts(
    tmp_path: Path,
) -> None:
    source = Path("PipelinePython/configs/data.yaml")
    overlay = tmp_path / "rerun.yaml"
    content = source.read_text(encoding="utf-8").replace(
        "bronze_dir: data/bronze",
        "bronze_dir: data/artifacts/runs/test/bronze",
    )
    overlay.write_text(content, encoding="utf-8")

    config = load_pipeline_config(overlay)

    assert config.artifacts["bronze_dir"] == "data/artifacts/runs/test/bronze"
    assert config.synchronization.alpha_min == 0.98
    assert config.training.min_balanced_accuracy == 0.80
    assert pipeline_paths_from_config(Path.cwd(), config).bronze_dir.name == "bronze"


def test_signal_contract_rejects_routine_and_non_monotonic_time() -> None:
    base = {
        "participant_id": ["ID-01", "ID-01"],
        "session_id": ["ID-01", "ID-01"],
        "recording_id": ["ID-01_full", "ID-01_full"],
        "phase": ["relax", "relax"],
        "sensor_channel": ["ppg_ir", "ppg_ir"],
        "timestamp_raw_utc": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"],
        "timestamp_corrected_utc": [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:01Z",
        ],
        "timestamp_local": [
            "2026-01-01T07:00:00+07:00",
            "2026-01-01T07:00:01+07:00",
        ],
        "elapsed_seconds": [0.0, 1.0],
        "sampling_rate_hz": [1.0, 1.0],
        "raw_value": [1.0, 2.0],
        "clean_value": [1.0, 2.0],
        "quality_score": [1.0, 1.0],
        "quality_tier": ["excellent", "excellent"],
        "quality_flags": ["", ""],
        "source_file": ["raw.csv", "raw.csv"],
        "source_checksum": ["0" * 64, "0" * 64],
        "exclusion_reason": ["", ""],
        "preprocessing_version": ["v3.0.0", "v3.0.0"],
    }
    frame = pd.DataFrame(base)
    SignalFrameContract().assert_valid(frame)

    routine = frame.assign(phase="routine")
    with pytest.raises(AssertionError):
        SignalFrameContract().assert_valid(routine)


def test_manifest_preserves_unknown_audit_phase_but_gold_rejects_it() -> None:
    manifest = ManifestRecord(
        participant_id="ID-01",
        session_id="S-01",
        phase="unknown",
        sensor_channel="ppg_ir",
        source_file="raw.csv",
        source_checksum="0" * 64,
        sampling_rate_hz=25.0,
        start_timestamp_utc=None,
        end_timestamp_utc=None,
        row_count=0,
        status="ok",
    )
    assert manifest.phase == "unknown"

    with pytest.raises(AssertionError):
        GoldWindowContract(
            participant_id="ID-01",
            session_id="S-01",
            phase="unknown",
            sensor_channel="ppg_ir",
            window_id="W-01",
            quality_score=1.0,
            quality_tier="good",
            source_file="raw.csv",
            source_checksum="0" * 64,
            preprocessing_version="v3.0.0",
            feature_version="v3.0.0_core240s",
        )


def test_short_gap_interpolation_returns_a_waveform_mask() -> None:
    timestamps = pd.date_range("2026-01-01", periods=18_001, freq="10ms", tz="UTC")
    values = pd.Series(np.sin(np.arange(18_001, dtype=float) / 10.0))
    values.iloc[500:503] = np.nan

    _, uniform, interpolated_mask = LocalBiosignalPipeline._uniform_window(
        pd.Series(timestamps),
        values,
        timestamps[0],
        timestamps[-1] + pd.Timedelta(milliseconds=10),
        100.0,
    )

    assert np.isfinite(uniform).all()
    assert interpolated_mask.any()
    assert interpolated_mask.sum() >= 3


def test_onnx_tensor_and_manifest_contracts() -> None:
    tensor = assert_feature_tensor(np.ones((3, 2)), n_features=2)
    probability = assert_stress_probability([0.1, 0.5, 0.9], batch_size=3)

    assert tensor.dtype == np.float32
    assert probability.shape == (3,)
    CandidateBundleManifest(
        model_version="stress_v3.0.0",
        preprocessing_version="v3.0.0",
        feature_version="v3.0.0_180s",
        sensor_channel="ppg_ir",
        feature_order=("mean_ibi_ms", "rmssd_ms"),
        label_order=("relax", "task"),
        probability_threshold=0.5,
        model_sha256="0" * 64,
        development_participants=tuple(f"ID-{i:02d}" for i in range(1, 29)),
        holdout_participants=tuple(f"ID-{i:02d}" for i in range(29, 33)),
        native_model_path="candidate/model.joblib",
        onnx_model_path="candidate/model.onnx",
    )
    result = OnnxParityResult(100, 9.0e-6, 1.0e-6, 1024, 0.5, True)
    assert result.passed
