from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from PipelinePython.common.peak_detection import PpgPeakDetector
from PipelinePython.common.settings import load_pipeline_config


def test_elgendi_primary_uses_already_cleaned_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_pipeline_config().peak_detection
    config = replace(
        config,
        channels={
            **config.channels,
            "ppg_green": replace(
                config.for_channel("ppg_green"),
                selected_method="neurokit2_ppg_elgendi",
            ),
        },
    )
    detector = PpgPeakDetector(config)
    signal = np.sin(2.0 * np.pi * np.arange(2_000) / 100.0)
    expected = np.arange(50, 1_951, 100)
    captured: list[np.ndarray] = []

    def find_peaks(values: np.ndarray, **_: object) -> dict[str, np.ndarray]:
        captured.append(values)
        return {"PPG_Peaks": expected}

    monkeypatch.setattr(
        "PipelinePython.common.peak_detection.nk.ppg_findpeaks", find_peaks
    )

    result = detector.detect(signal, 100.0, "ppg_green")

    assert result.method == "neurokit2_ppg_elgendi"
    assert not result.fallback_reason
    assert np.array_equal(result.peaks, expected)
    assert np.shares_memory(captured[0], signal)


def test_detect_uses_frozen_method_per_channel() -> None:
    config = load_pipeline_config().peak_detection
    config = replace(
        config,
        channels={
            **config.channels,
            "ppg_ir": replace(
                config.for_channel("ppg_ir"),
                selected_method="scipy_find_peaks_adaptive",
            ),
        },
    )
    detector = PpgPeakDetector(config)
    signal = np.sin(2.0 * np.pi * np.arange(3_000) / 100.0)

    result = detector.detect(signal, 100.0, "ppg_ir")

    assert result.method == "scipy_find_peaks_adaptive"


def test_scipy_fallback_records_reason_and_channel_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_pipeline_config().peak_detection, allow_runtime_fallback=True)
    detector = PpgPeakDetector(config)
    samples = np.arange(3_000)
    signal = np.sin(2.0 * np.pi * samples / 100.0)

    def fail_primary(*_: object, **__: object) -> dict[str, np.ndarray]:
        raise ValueError("forced primary failure")

    monkeypatch.setattr(
        "PipelinePython.common.peak_detection.nk.ppg_findpeaks", fail_primary
    )

    result = detector.detect(signal, 100.0, "ppg_red")

    assert result.method == "scipy_find_peaks_fallback"
    assert "forced primary failure" in result.fallback_reason
    assert len(result.peaks) >= 15
    assert result.physiological_ibi_ratio >= 0.90


def test_frozen_detector_failure_does_not_mix_peak_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_pipeline_config().peak_detection
    detector = PpgPeakDetector(config)
    signal = np.sin(2.0 * np.pi * np.arange(3_000) / 100.0)

    def fail_selected(*_: object, **__: object) -> dict[str, np.ndarray]:
        raise ValueError("forced selected failure")

    monkeypatch.setattr(
        "PipelinePython.common.peak_detection.nk.ppg_findpeaks", fail_selected
    )

    result = detector.detect(signal, 100.0, "ppg_red")

    assert result.status == "UNAVAILABLE"
    assert result.peaks.size == 0
    assert result.method == "neurokit2_ppg_elgendi"
    assert "forced selected failure" in result.status_reason


def test_development_audit_can_evaluate_scipy_without_forcing_primary_failure() -> None:
    detector = PpgPeakDetector(load_pipeline_config().peak_detection)
    samples = np.arange(3_000)
    signal = np.sin(2.0 * np.pi * samples / 100.0)

    result = detector.detect_fallback(signal, 100.0, "ppg_green")

    assert result.method == "scipy_find_peaks_development_audit"
    assert not result.fallback_reason
    assert result.physiological_ibi_ratio >= 0.90


def test_legacy_green_ablation_uses_native_samples_and_edge_guard() -> None:
    detector = PpgPeakDetector(load_pipeline_config().peak_detection)
    samples = np.arange(1_000)
    signal = np.sin(2.0 * np.pi * samples / 100.0)

    result = detector.evaluate_candidate(
        signal,
        100.0,
        "ppg_green",
        "scipy_legacy_green_native",
    )

    assert result.status == "AVAILABLE"
    assert result.method == "scipy_legacy_green_native"
    assert len(result.peaks) >= 8
    assert result.peaks.min() >= 50
    assert result.peaks.max() < len(signal) - 50
    assert np.all(np.diff(result.peaks) >= 33)


def test_legacy_green_ablation_is_unavailable_for_ir_and_red() -> None:
    detector = PpgPeakDetector(load_pipeline_config().peak_detection)
    signal = np.sin(2.0 * np.pi * np.arange(1_000) / 100.0)

    for channel in ("ppg_ir", "ppg_red"):
        result = detector.evaluate_candidate(
            signal, 100.0, channel, "scipy_legacy_green_native"
        )
        assert result.status == "UNAVAILABLE"
        assert result.peaks.size == 0
        assert "not_valid_for_ir_or_red" in result.status_reason


def test_bishop_uses_bounded_observed_chunks_for_long_recordings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = PpgPeakDetector(load_pipeline_config().peak_detection)
    signal = np.sin(2.0 * np.pi * np.arange(2_000) / 100.0)
    calls: list[np.ndarray] = []

    def bishop(values: np.ndarray, **_: object) -> dict[str, np.ndarray]:
        calls.append(values)
        return {
            "PPG_Peaks": np.array([50], dtype=np.int64),
            "PPG_Onsets": np.array([25], dtype=np.int64),
        }

    monkeypatch.setattr("PipelinePython.common.peak_detection.nk.ppg_findpeaks", bishop)

    result = detector.evaluate_candidate(
        signal,
        100.0,
        "ppg_green",
        "neurokit2_ppg_bishop",
    )

    assert result.status == "AVAILABLE"
    assert calls
    assert max(len(chunk) for chunk in calls) <= 600
    assert all(np.shares_memory(chunk, signal) for chunk in calls)
    assert np.all(np.diff(result.peaks) > 0)


def test_unknown_ppg_channel_is_rejected() -> None:
    detector = PpgPeakDetector(load_pipeline_config().peak_detection)

    with pytest.raises(ValueError, match="PPG channel"):
        detector.detect(np.sin(np.arange(1_000) / 10.0), 100.0, "ppg_blue")


def test_minimum_segment_length_is_derived_from_physiological_config() -> None:
    detector = PpgPeakDetector(load_pipeline_config().peak_detection)

    assert detector.minimum_sample_count(100.0) == 450

    with pytest.raises(ValueError, match="too short"):
        detector.detect(np.ones(449), 100.0, "ppg_green")
