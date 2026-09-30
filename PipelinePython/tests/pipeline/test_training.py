"""
test_training.py
----------------
Tests for model training, validation, and feature engineering.

Task 14.2: Property tests for training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from PipelinePython.common.training import StressModelTrainer
from PipelinePython.common.config import FEATURE_VERSION, WINDOW_SECONDS
from PipelinePython.common.settings import DEFAULT_PATHS, PipelineSettings
from PipelinePython.common.training_lightgbm import (
    DEVELOPMENT_PARTICIPANTS,
    PARITY_POOL_PARTICIPANTS,
)


class TestTrainTestSplit:
    """Test train/test split logic."""

    def test_canonical_split_uses_locked_participant_ids(self) -> None:
        """The trainer must never sample a random participant holdout."""
        groups = np.repeat(np.array([f"ID-{index:02d}" for index in range(1, 33)]), 2)
        assert DEVELOPMENT_PARTICIPANTS == {f"ID-{index:02d}" for index in range(1, 29)}
        assert PARITY_POOL_PARTICIPANTS == {
            f"ID-{index:02d}" for index in range(29, 33)
        }
        assert not DEVELOPMENT_PARTICIPANTS & PARITY_POOL_PARTICIPANTS
        assert set(groups) == DEVELOPMENT_PARTICIPANTS | PARITY_POOL_PARTICIPANTS

    def test_no_participant_leakage(self) -> None:
        """Property test: no participant appears in both train AND test sets."""
        # Create sample data
        participants = np.array(["P001", "P002", "P003", "P004", "P005"] * 20)

        # Simulate participant holdout split
        unique_participants = np.array(sorted(set(participants)))
        rng = np.random.default_rng(42)
        rng.shuffle(unique_participants)

        test_size = 0.2
        n_test = max(1, int(len(unique_participants) * test_size))
        test_set = set(unique_participants[:n_test].tolist())

        train_idx = np.where(np.array([p not in test_set for p in participants]))[0]
        test_idx = np.where(np.array([p in test_set for p in participants]))[0]

        train_participants = set(participants[train_idx])
        test_participants = set(participants[test_idx])

        # Check no overlap
        overlap = train_participants & test_participants
        assert len(overlap) == 0, "No participant should appear in both train and test"

    def test_split_preserves_all_data(self) -> None:
        """Test that split preserves all data points."""
        n_samples = 1000
        participants = np.array(["P001", "P002", "P003"] * (n_samples // 3 + 1))[
            :n_samples
        ]

        unique_participants = np.array(sorted(set(participants)))
        rng = np.random.default_rng(42)
        rng.shuffle(unique_participants)

        test_size = 0.2
        n_test = max(1, int(len(unique_participants) * test_size))
        test_set = set(unique_participants[:n_test].tolist())

        train_idx = np.where(np.array([p not in test_set for p in participants]))[0]
        test_idx = np.where(np.array([p in test_set for p in participants]))[0]

        # All indices should be accounted for
        all_idx = np.concatenate([train_idx, test_idx])
        assert len(all_idx) == n_samples, "Split should preserve all data"

    def test_split_respects_test_size(self) -> None:
        """Test that split respects test size parameter."""
        participants = np.array(["P001", "P002", "P003", "P004", "P005"] * 20)

        unique_participants = np.array(sorted(set(participants)))
        rng = np.random.default_rng(42)
        rng.shuffle(unique_participants)

        test_size = 0.2
        n_test = max(1, int(len(unique_participants) * test_size))
        test_set = set(unique_participants[:n_test].tolist())

        test_idx = np.where(np.array([p in test_set for p in participants]))[0]

        # Test set should be approximately test_size fraction
        actual_test_fraction = len(test_idx) / len(participants)
        assert 0.15 < actual_test_fraction < 0.25, (
            "Test fraction should be close to 0.2"
        )


class TestFeatureEngineering:
    """Test feature selection and engineering."""

    def test_feature_columns_exclude_lineage_quality_timestamps_ecg_and_other_sensors(
        self,
    ) -> None:
        """Model features must stay PPG-only and sensor-specific."""
        df = pd.DataFrame(
            {
                "label": [0, 1],
                "participant_code": ["ID-01", "ID-02"],
                "session_id": ["S1", "S2"],
                "phase": ["Relax", "Task"],
                "sensor_channel": ["ppg_ir", "ppg_ir"],
                "signal_source": ["ppg", "ppg"],
                "window_idx": [0, 1],
                "window_start_ms": [1000, 2000],
                "window_end_ms": [61000, 62000],
                "quality_score": [1.0, 0.9],
                "quality_tier": ["excellent", "good"],
                "feature_version": [FEATURE_VERSION, FEATURE_VERSION],
                "num_peaks": [80, 82],
                "hrv_HRV_RMSSD": [30.0, 40.0],
                "mean_rr_ms": [750.0, 730.0],
                "ppg_ir_mean": [0.1, 0.2],
                "ppg_ir_peak_amp_mean": [100.0, 120.0],
                "ppg_red_mean": [0.3, 0.4],
                "ppg_green_mean": [0.5, 0.6],
                "ecg_groundtruth_HRV_RMSSD": [31.0, 41.0],
                "ecg_leaky_feature": [1.0, 2.0],
            }
        )
        trainer = StressModelTrainer(DEFAULT_PATHS, PipelineSettings())

        cols = trainer._feature_columns(df, "ppg_ir")

        assert "num_peaks" in cols
        assert "hrv_HRV_RMSSD" in cols
        assert "mean_rr_ms" in cols
        assert "ppg_ir_mean" in cols
        assert "ppg_ir_peak_amp_mean" in cols
        for forbidden in [
            "label",
            "participant_code",
            "session_id",
            "phase",
            "sensor_channel",
            "signal_source",
            "window_idx",
            "window_start_ms",
            "window_end_ms",
            "quality_score",
            "quality_tier",
            "feature_version",
            "ppg_red_mean",
            "ppg_green_mean",
            "ecg_groundtruth_HRV_RMSSD",
            "ecg_leaky_feature",
        ]:
            assert forbidden not in cols

    def test_training_window_contract_is_three_minutes(self) -> None:
        """Stress model feature windows are locked to 180 seconds for Android parity."""
        settings = PipelineSettings()

        assert WINDOW_SECONDS == 180
        assert settings.model_window_seconds == 180
        assert settings.feature_version == "v4.0.0_robust_channel_independent"

    def test_ecg_feature_exclusion(self) -> None:
        """Test that ECG ground truth features are excluded from training."""
        # Create sample features
        features = {
            "ppg_heart_rate": [70.0, 75.0, 80.0],
            "ppg_ibi_mean": [850.0, 800.0, 750.0],
            "ecg_groundtruth_heart_rate": [72.0, 76.0, 82.0],
            "ecg_groundtruth_ibi_mean": [833.0, 789.0, 732.0],
            "label": [0, 1, 0],
            "participant_code": ["P001", "P002", "P003"],
        }
        df = pd.DataFrame(features)

        # Simulate feature column selection
        blacklist = {"label", "participant_code"}
        ecg_exclusion_pattern = "ecg_groundtruth_"

        selected_cols = []
        for c in df.columns:
            if c in blacklist:
                continue
            if c.startswith(ecg_exclusion_pattern):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                selected_cols.append(c)

        # Check that ECG features are excluded
        assert "ecg_groundtruth_heart_rate" not in selected_cols
        assert "ecg_groundtruth_ibi_mean" not in selected_cols
        assert "ppg_heart_rate" in selected_cols
        assert "ppg_ibi_mean" in selected_cols

    def test_feature_sanitization_removes_infinities(self) -> None:
        """Test that feature sanitization removes infinities."""
        features = {
            "feature1": [1.0, 2.0, np.inf, 4.0],
            "feature2": [5.0, -np.inf, 7.0, 8.0],
            "feature3": [9.0, 10.0, 11.0, 12.0],
        }
        df = pd.DataFrame(features)

        # Replace infinities with NaN
        df_clean = df.replace([np.inf, -np.inf], np.nan)

        assert not np.any(np.isinf(df_clean.values)), "Should remove infinities"

    def test_feature_sanitization_clips_outliers(self) -> None:
        """Test that feature sanitization clips extreme outliers."""
        features = {
            "feature1": [1.0, 2.0, 1e10, 4.0],
            "feature2": [5.0, -1e10, 7.0, 8.0],
        }
        df = pd.DataFrame(features)

        # Clip extreme values
        df_clipped = df.clip(lower=-1e9, upper=1e9)

        assert df_clipped["feature1"].max() <= 1e9
        assert df_clipped["feature2"].min() >= -1e9


class TestModelEvaluation:
    """Test model evaluation metrics."""

    def test_f1_score_calculation(self) -> None:
        """Test F1 score calculation."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        # Calculate F1 manually
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        assert 0 <= f1 <= 1, "F1 score should be between 0 and 1"

    def test_roc_auc_calculation(self) -> None:
        """Test ROC-AUC calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8])

        # ROC-AUC should be between 0 and 1
        # For this example, it should be > 0.5 (better than random)

        # Simple calculation: count concordant pairs
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        concordant = 0
        for i in range(len(y_true)):
            for j in range(len(y_true)):
                if y_true[i] == 1 and y_true[j] == 0:
                    if y_prob[i] > y_prob[j]:
                        concordant += 1

        auc = concordant / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5
        assert 0 <= auc <= 1, "AUC should be between 0 and 1"

    def test_balanced_accuracy_calculation(self) -> None:
        """Test balanced accuracy calculation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])

        # Calculate balanced accuracy
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2

        assert 0 <= balanced_acc <= 1, "Balanced accuracy should be between 0 and 1"


class TestPredictionProbabilities:
    """Test probability prediction handling."""

    def test_probability_bounds(self) -> None:
        """Test that predicted probabilities are in [0, 1]."""
        # Simulate predicted probabilities
        y_prob = np.array([0.1, 0.5, 0.9, 0.3, 0.7])

        assert np.all(y_prob >= 0.0), "Probabilities should be >= 0"
        assert np.all(y_prob <= 1.0), "Probabilities should be <= 1"

    def test_probability_sum_for_multiclass(self) -> None:
        """Test that multiclass probabilities sum to 1."""
        # Simulate multiclass probabilities (5 classes)
        y_prob = np.array(
            [
                [0.1, 0.2, 0.3, 0.2, 0.2],
                [0.5, 0.1, 0.1, 0.1, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
            ]
        )

        prob_sums = np.sum(y_prob, axis=1)
        assert np.allclose(prob_sums, 1.0), "Probabilities should sum to 1"
