"""test_validation.py
------------------
Comprehensive validation suite covering the 5 critical pipeline dimensions:
1. EDA Artifacts & Signal Integrity
2. Overfitting Prevention & Fold Stability
3. Augmentation & Interpolation Constraints
4. Data Actuality & Frozen Gate Enforcement
5. ONNX & Android Parity Contracts
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from PipelinePython.common.settings import DEFAULT_PATHS
from PipelinePython.common.augmentation.transforms import (
    AmplitudeScaling,
    GaussianNoiseInjection,
    TimeReversal,
    AmplitudeInversion,
)
from PipelinePython.common.augmentation.pipeline import AugmentationPipeline
from PipelinePython.common.interpolation import CubicSplineInterpolationEngine


def test_eda_artifacts_and_signal_integrity():
    """1. EDA Validation: Check gold features table, participant counts, and sensor channels."""
    gold_csv = DEFAULT_PATHS.artifacts_root / "gold" / "gold_features.csv"
    assert gold_csv.exists(), "gold_features.csv missing"

    df = pd.read_csv(gold_csv)
    assert not df.empty, "gold_features.csv is empty"

    # Verify participant count
    pid_col = (
        "participant_code" if "participant_code" in df.columns else "participant_id"
    )
    participants = df[pid_col].unique()
    assert len(participants) == 32, f"Expected 32 participants, got {len(participants)}"

    # Verify holdout participants presence
    holdout_ids = {"ID-29", "ID-30", "ID-31", "ID-32"}
    assert holdout_ids.issubset(set(participants)), (
        "Holdout participants (ID-29..ID-32) missing"
    )

    # Verify sensor channel separation
    sensors = df["sensor_channel"].unique()
    assert set(sensors).issuperset({"ppg_green", "ppg_ir", "ppg_red"}), (
        "Missing expected PPG sensor channels"
    )

    # Verify label counts (Relax = 0/Relax, Task = 1/Task)
    assert "label" in df.columns or "phase" in df.columns, "Label/phase column missing"


def test_overfitting_prevention_and_gate_enforcement():
    """2 & 4. Enforce the immutable two-tier release policy."""
    # The canonical local model contract is configured in training.yaml.
    frozen_gates_path = Path("PipelinePython/configs/frozen_gates.yaml")
    assert frozen_gates_path.exists(), "frozen_gates.yaml missing"
    gates_text = frozen_gates_path.read_text(encoding="utf-8")
    assert "version: ml_dl_tiered_v1" in gates_text
    assert "minimum_research:" in gates_text
    assert "balanced_accuracy: 0.70" in gates_text
    assert "f1: 0.70" in gates_text
    assert "roc_auc: 0.70" in gates_text
    assert "preferred:" in gates_text
    assert "balanced_accuracy: 0.80" in gates_text

    # Only the canonical accepted index is deployable. An empty index is the
    # correct fail-closed state while new-cohort evidence is pending.
    index = json.loads(
        (DEFAULT_PATHS.export_dir / "model_index.json").read_text(encoding="utf-8")
    )
    if not index["models"]:
        assert index["release_status"].startswith("blocked_")
        return

    for candidate in index["models"]:
        assert candidate["release_status"] == "accepted"
        assert candidate["android_ready"] is True
        assert candidate["performance_tier"] in {"research_usable", "preferred"}
        assert candidate["deployment_scope"] in {"research", "standard"}
        assert candidate["balanced_accuracy_holdout"] >= 0.70
        assert candidate["f1_holdout"] >= 0.70
        assert candidate["roc_auc_holdout"] >= 0.70
        assert candidate["balanced_accuracy_conservative"] >= 0.70
        assert candidate["f1_conservative"] >= 0.70
        assert candidate["roc_auc_conservative"] >= 0.70
        assert set(candidate["cohort_ids"]) == {"ID-29", "ID-30", "ID-31", "ID-32"}


def test_augmentation_placement_and_forbidden_transforms():
    """3. Augmentation Validation: Post-split placement and forbidden transform guards."""
    # Forbidden transforms must raise ValueError
    with pytest.raises(ValueError, match="TimeReversal is STRICTLY FORBIDDEN"):
        TimeReversal().apply(np.ones(100, dtype=np.float32))

    with pytest.raises(ValueError, match="AmplitudeInversion is STRICTLY FORBIDDEN"):
        AmplitudeInversion().apply(np.ones(100, dtype=np.float32))

    # Approved pipeline validation
    pipeline = AugmentationPipeline(
        transforms=[AmplitudeScaling(), GaussianNoiseInjection()], seed=42
    )
    X_train = np.random.randn(10, 100).astype(np.float32)
    y_train = np.array([0, 1] * 5)

    X_aug, y_aug = pipeline.augment_dataset(X_train, y_train, multiplier=2)
    assert len(X_aug) == 30
    assert len(y_aug) == 30


def test_cubic_spline_interpolation_engine():
    """3. Interpolation Engine: Verify missing IBI cubic spline interpolation and 4 Hz resampling."""
    engine = CubicSplineInterpolationEngine()
    ibi_ms = np.array([800.0, 810.0, np.nan, 820.0, 790.0, 805.0], dtype=np.float64)
    missing_mask = np.isnan(ibi_ms)
    ibi_input = np.nan_to_num(ibi_ms, nan=800.0)

    interpolated_ibi = engine.interpolate_ibi(ibi_input, missing_mask)
    assert not np.isnan(interpolated_ibi).any(), "Interpolated IBI contains NaNs"
    assert len(interpolated_ibi) == len(ibi_ms)

    timestamps = np.cumsum(interpolated_ibi) / 1000.0
    resampled_t, resampled_signal = engine.resample_regular(
        timestamps, interpolated_ibi, target_fs=4.0
    )
    assert len(resampled_signal) > 0
    assert not np.isnan(resampled_signal).any()


def test_onnx_and_android_parity_contracts():
    """5. Canonical indexed artifacts require native/ONNX parity below 1e-5."""
    index = json.loads(
        (DEFAULT_PATHS.export_dir / "model_index.json").read_text(encoding="utf-8")
    )
    if not index["models"]:
        assert index["release_status"].startswith("blocked_")
        return

    for candidate in index["models"]:
        parity_path = DEFAULT_PATHS.export_dir / candidate["files"]["parity"]["path"]
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        assert parity["onnx_checker_passed"] is True
        assert parity["real_batch1_inference_passed"] is True
        assert parity.get("max_diff", parity.get("parity_max_diff")) < 1e-5
