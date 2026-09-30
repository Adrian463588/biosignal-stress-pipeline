from __future__ import annotations

import numpy as np
import pandas as pd

from PipelinePython.common.madc import MotionArtifactDetector
from PipelinePython.common.sqi import PostPeakSQA


def test_missing_accelerometer_is_explicitly_unavailable() -> None:
    signal = np.linspace(0.0, 1.0, 8)
    detector = MotionArtifactDetector()

    result = detector.process_window(pd.DataFrame({"ppg": signal}))

    assert result["motion_available"] is False
    assert result["madc_action"] == "MOTION_UNAVAILABLE"
    assert np.isnan(result["motion_score"])
    np.testing.assert_array_equal(result["cleaned_signal"], signal)
    assert result["provenance"]["signal_modified"] is False
    assert result["provenance"]["interpolated_sample_count"] == 0


def test_nonfinite_accelerometer_is_not_treated_as_zero_motion() -> None:
    signal = np.arange(8, dtype=float)
    frame = pd.DataFrame(
        {
            "ppg": signal,
            "acc_x": np.ones(signal.size),
            "acc_y": np.ones(signal.size),
            "acc_z": np.full(signal.size, np.nan),
            "accelerometer_provenance_status": "VERIFIED",
        }
    )

    result = MotionArtifactDetector().process_window(frame)

    assert result["motion_available"] is False
    assert result["motion_annotation"] == "accelerometer_nonfinite"
    assert np.isnan(result["motion_score"])
    assert not np.any(result["interpolated_mask"])
    np.testing.assert_array_equal(result["cleaned_signal"], signal)


def test_madc_window_is_sampling_rate_invariant() -> None:
    detector = MotionArtifactDetector(
        {
            "window_seconds": 2.0,
            "robust_z_threshold": 6.0,
            "heavy_motion_threshold": 0.8,
            "reject_ratio_threshold": 0.3,
            "light_motion_threshold": 0.2,
        }
    )

    def detect(sampling_rate_hz: float) -> dict[str, object]:
        elapsed = np.arange(int(12.0 * sampling_rate_hz)) / sampling_rate_hz
        burst = (elapsed >= 4.0) & (elapsed < 6.0)
        acc_x = 0.01 * np.sin(2.0 * np.pi * 0.5 * elapsed)
        acc_x += burst * np.sin(2.0 * np.pi * 4.0 * elapsed)
        frame = pd.DataFrame(
            {
                "ppg": np.sin(2.0 * np.pi * 1.2 * elapsed),
                "acc_x": acc_x,
                "acc_y": np.zeros_like(elapsed),
                "acc_z": np.ones_like(elapsed),
                "accelerometer_provenance_status": "VERIFIED",
            }
        )
        return detector.process_window(frame, sampling_rate_hz=sampling_rate_hz)

    green = detect(25.0)
    infrared = detect(100.0)

    assert green["madc_window_samples"] == 50
    assert infrared["madc_window_samples"] == 200
    assert green["madc_window_seconds"] == infrared["madc_window_seconds"] == 2.0
    assert (
        abs(float(green["heavy_motion_ratio"]) - float(infrared["heavy_motion_ratio"]))
        < 0.03
    )
    assert len(green["motion_heavy_mask"]) == 300
    assert len(infrared["motion_heavy_mask"]) == 1_200


def test_unverified_accelerometer_columns_are_unavailable() -> None:
    frame = pd.DataFrame(
        {
            "ppg": np.arange(8, dtype=float),
            "acc_x": np.ones(8),
            "acc_y": np.ones(8),
            "acc_z": np.ones(8),
            "accelerometer_provenance_status": "UNAVAILABLE",
        }
    )

    result = MotionArtifactDetector().process_window(frame, sampling_rate_hz=25.0)

    assert result["motion_available"] is False
    assert result["motion_annotation"] == "accelerometer_provenance_unverified"
    assert np.isnan(result["motion_score"])
    assert not result["motion_heavy_mask"].any()


def test_post_peak_quality_uses_observed_peak_timestamps() -> None:
    timestamps_ms = np.arange(201, dtype=float) * 10.0
    timestamps_ms[100:] = np.linspace(1_000.0, 2_600.0, 101)
    timestamps = pd.to_datetime(timestamps_ms, unit="ms", origin="unix", utc=True)
    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, len(timestamps)))

    result = PostPeakSQA(minimum_peak_count=2).assess(
        signal,
        np.array([0, 100, 200]),
        sampling_rate_hz=100.0,
        timestamps_utc=timestamps,
    )

    assert result.physiological_ibi_ratio == 0.5
    assert "implausible_ibi_ratio" in result.flags
    assert result.tier == "invalid"
