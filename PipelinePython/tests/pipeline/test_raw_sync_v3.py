from __future__ import annotations

import numpy as np
import pytest

from PipelinePython.common.features_v3 import extract_ppg_features
from PipelinePython.common.raw_sync import (
    _contains_long_dropout,
    affine_clock_correction,
    detect_peak_times_ms,
    validate_ecg_against_ppg,
)
from PipelinePython.common.settings import SynchronizationConfig


def test_affine_clock_correction_uses_metadata_without_changing_order() -> None:
    sample_count = 46_801
    raw_ms = np.linspace(1_000_000.0, 1_359_000.0, sample_count)
    correction = affine_clock_correction(
        raw_ms,
        {
            "stoptime": "1970-01-01T00:22:40Z",
            "duration": "360 sec",
            "samplescount": str(sample_count),
            "samplingrate": "130",
        },
        SynchronizationConfig(),
    )

    assert not correction.excluded
    assert 0.98 <= correction.alpha <= 1.02
    assert np.all(np.diff(correction.corrected_unix_ms) > 0)
    np.testing.assert_allclose(correction.corrected_unix_ms[0], 1_000_000.0)
    np.testing.assert_allclose(correction.corrected_unix_ms[-1], 1_360_000.0, atol=1.0)
    assert correction.timebase_source == "sample_index_metadata_duration_stoptime"
    assert correction.observed_sample_count == sample_count
    assert correction.declared_sample_count == sample_count
    assert correction.nominal_sample_rate_hz == 130.0
    assert correction.effective_sample_rate_hz == 130.0
    assert correction.clock_selection_basis == "metadata_only"


def test_padded_full_ecg_is_not_scaled_to_trimmed_ppg_endpoints() -> None:
    """Acquisition padding is cropped later, not compressed into PPG time."""

    ecg_sample_count = 42_901
    ecg_raw_timestamps = np.linspace(200_000.0, 700_000.0, ecg_sample_count)
    ppg_timestamps = np.linspace(1_100_000.0, 1_340_000.0, 24_001)

    correction = affine_clock_correction(
        ecg_raw_timestamps,
        {
            "duration": "330 sec",
            "samplescount": str(ecg_sample_count),
            "samplingrate": "130",
            "stoptime": "1970-01-01T00:23:20Z",
        },
        SynchronizationConfig(),
        reference_timestamps_ms=ppg_timestamps,
        phase_start_ms=1_100_000.0,
        phase_end_ms=1_340_000.0,
    )

    assert not correction.excluded
    np.testing.assert_allclose(correction.alpha, 1.0)
    np.testing.assert_allclose(correction.corrected_unix_ms[0], 1_070_000.0)
    np.testing.assert_allclose(correction.corrected_unix_ms[-1], 1_400_000.0)
    np.testing.assert_allclose(
        correction.corrected_unix_ms[-1] - correction.corrected_unix_ms[0],
        330_000.0,
    )
    assert correction.corrected_unix_ms[0] != ppg_timestamps[0]
    assert correction.corrected_unix_ms[-1] != ppg_timestamps[-1]


def test_metadata_duration_clock_accepts_rate_error_within_ten_percent() -> None:
    """Effective-rate evidence must not be reapplied as affine compression."""

    sample_count = 39_785
    raw_ms = np.linspace(100_000.0, 623_190.0, sample_count)
    correction = affine_clock_correction(
        raw_ms,
        {
            "duration": "314 sec",
            "samplescount": str(sample_count),
            "samplingrate": "130",
            "stoptime": "1970-01-01T00:21:40Z",
        },
        SynchronizationConfig(),
    )

    assert not correction.excluded
    assert correction.alpha == 1.0
    assert correction.effective_sample_rate_hz == pytest.approx(126.7006369)
    assert correction.effective_rate_relative_error == pytest.approx(0.0253797)
    np.testing.assert_allclose(
        correction.corrected_unix_ms[-1] - correction.corrected_unix_ms[0],
        314_000.0,
    )


def test_inconsistent_metadata_rate_fails_closed_with_evidence() -> None:
    sample_count = 39_001
    raw_ms = np.linspace(10_000.0, 310_000.0, sample_count)

    correction = affine_clock_correction(
        raw_ms,
        {
            "duration": "300 sec",
            "samplescount": str(sample_count),
            "samplingrate": "100",
            "stoptime": "1970-01-01T00:05:10Z",
        },
        SynchronizationConfig(),
    )

    assert correction.excluded
    assert (correction.exclusion_reason or "").startswith(
        "metadata_effective_rate_out_of_tolerance:"
    )
    np.testing.assert_array_equal(correction.corrected_unix_ms, raw_ms)
    assert correction.nominal_sample_rate_hz == 100.0
    assert correction.effective_sample_rate_hz == 130.0
    assert correction.effective_rate_relative_error == 0.30
    assert correction.clock_selection_basis == "metadata_only"


def test_sample_index_clock_ignores_corrupted_raw_timestamp_spacing() -> None:
    sample_count = 39_001
    raw_ms = np.linspace(700_000.0, 200_000.0, sample_count)

    correction = affine_clock_correction(
        raw_ms,
        {
            "duration": "300 sec",
            "samplescount": str(sample_count),
            "samplingrate": "130",
            "stoptime": "1970-01-01T00:21:40Z",
        },
        SynchronizationConfig(),
    )

    assert not correction.excluded
    np.testing.assert_allclose(correction.corrected_unix_ms[0], 1_000_000.0)
    np.testing.assert_allclose(correction.corrected_unix_ms[-1], 1_300_000.0)
    np.testing.assert_allclose(
        np.diff(correction.corrected_unix_ms),
        np.full(sample_count - 1, 300_000.0 / (sample_count - 1)),
    )


def test_sync_preserves_physiological_pat() -> None:
    duration = 240.0
    ecg_fs, ppg_fs = 130.0, 100.0
    ecg_time = np.linspace(0.0, duration, int(duration * ecg_fs) + 1)
    ppg_time = np.arange(0.0, duration, 1.0 / ppg_fs)
    ecg = np.zeros_like(ecg_time)
    ppg = np.zeros_like(ppg_time)
    beat_times = np.arange(1.0, duration - 1.0, 1.0)
    for beat in beat_times:
        ecg += np.exp(-0.5 * ((ecg_time - beat) / 0.02) ** 2)
        ppg += np.exp(-0.5 * ((ppg_time - (beat + 0.22)) / 0.05) ** 2)

    ppg_timestamps_before = (ppg_time * 1000.0).copy()
    ppg_values_before = ppg.copy()
    ecg_timestamps_before = (ecg_time * 1000.0).copy()
    ecg_values_before = ecg.copy()

    _, result = validate_ecg_against_ppg(
        participant_id="ID-01",
        recording_id="ID-01_relax_raw",
        ecg_timestamps_ms=ecg_time * 1000.0,
        ecg_values=ecg,
        ppg_timestamps_ms=ppg_time * 1000.0,
        ppg_values=ppg,
        metadata={
            "duration": "240 sec",
            "samplescount": str(len(ecg_time)),
            "samplingrate": str(ecg_fs),
            "stoptime": "1970-01-01T00:04:00Z",
        },
        config=SynchronizationConfig(),
    )

    assert not result.excluded
    assert result.median_pat_ms is not None
    assert 200.0 <= result.median_pat_ms <= 240.0
    np.testing.assert_array_equal(ppg_time * 1000.0, ppg_timestamps_before)
    np.testing.assert_array_equal(ppg, ppg_values_before)
    np.testing.assert_array_equal(ecg_time * 1000.0, ecg_timestamps_before)
    np.testing.assert_array_equal(ecg, ecg_values_before)


def test_ppg_diagnostic_failure_does_not_exclude_valid_ecg_clock() -> None:
    """One poor PPG channel must not invalidate ECG for sibling channels."""

    ecg_fs = 10.0
    ecg_time = np.arange(0.0, 330.0 + 1.0 / ecg_fs, 1.0 / ecg_fs)
    ecg = np.zeros_like(ecg_time)
    for beat in np.arange(1.0, 329.0, 1.0):
        ecg += np.exp(-0.5 * ((ecg_time - beat) / 0.04) ** 2)
    ppg_time = np.arange(30.0, 270.0 + 1.0 / ecg_fs, 1.0 / ecg_fs)
    ppg = np.zeros_like(ppg_time)

    _, result = validate_ecg_against_ppg(
        participant_id="ID-01",
        recording_id="ID-01_relax_raw",
        ecg_timestamps_ms=np.linspace(0.0, 500_000.0, len(ecg_time)),
        ecg_values=ecg,
        ppg_timestamps_ms=ppg_time * 1000.0,
        ppg_values=ppg,
        metadata={
            "duration": "330 sec",
            "samplescount": str(len(ecg_time)),
            "samplingrate": str(ecg_fs),
            "stoptime": "1970-01-01T00:05:30Z",
        },
        config=SynchronizationConfig(),
        phase_start_ms=30_000.0,
        phase_end_ms=270_000.0,
        diagnostic_reference_channel="ppg_ir",
    )

    assert not result.excluded
    assert result.exclusion_reason is None
    assert result.overlap_seconds == 240.0
    assert result.ecg_peak_count >= 30
    assert result.ppg_peak_count == 0
    assert result.reference_signal_role == "diagnostic_only"
    assert result.clock_excluded is False
    assert result.clock_exclusion_reason is None
    assert result.diagnostic_reference_channel == "ppg_ir"
    assert "insufficient_ppg_peaks" in (result.ppg_diagnostic_reason or "")
    assert "insufficient_pat_matches" in (result.ppg_diagnostic_reason or "")


def test_long_dropout_is_rejected_before_peak_interpolation() -> None:
    timestamps = np.arange(0.0, 10_000.0, 10.0)
    values = np.sin(timestamps / 100.0)
    values[300:500] = np.nan

    assert _contains_long_dropout(timestamps, values)


def test_primary_peak_detection_does_not_interpolate_waveform(monkeypatch) -> None:
    timestamps = np.arange(0.0, 240_000.0, 10.0)
    values = np.sin(2.0 * np.pi * timestamps / 1_000.0)

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "primary peak detection must not interpolate waveform samples"
        )

    monkeypatch.setattr("PipelinePython.common.raw_sync.np.interp", fail_if_called)
    peaks = detect_peak_times_ms(timestamps, values, is_ecg=False)

    assert peaks.size > 100
    assert np.all(np.isin(peaks, timestamps))


def test_broken_ecg_raw_clock_is_not_misclassified_as_waveform_dropout() -> None:
    duration = 240.0
    ecg_fs = 130.0
    ppg_fs = 100.0
    ecg_time = np.linspace(0.0, duration, int(duration * ecg_fs) + 1)
    ppg_time = np.arange(0.0, duration, 1.0 / ppg_fs)
    ecg = np.zeros_like(ecg_time)
    ppg = np.zeros_like(ppg_time)
    for beat in np.arange(1.0, duration - 1.0, 1.0):
        ecg += np.exp(-0.5 * ((ecg_time - beat) / 0.02) ** 2)
        ppg += np.exp(-0.5 * ((ppg_time - (beat + 0.2)) / 0.05) ** 2)

    _, result = validate_ecg_against_ppg(
        participant_id="ID-01",
        recording_id="ID-01_relax_raw",
        ecg_timestamps_ms=np.linspace(0.0, 400_000.0, len(ecg_time)),
        ecg_values=ecg,
        ppg_timestamps_ms=(ppg_time + 1_000.0) * 1000.0,
        ppg_values=ppg,
        metadata={
            "duration": "240 sec",
            "samplescount": str(len(ecg_time)),
            "samplingrate": str(ecg_fs),
            "stoptime": "1970-01-01T00:20:40Z",
        },
        config=SynchronizationConfig(),
    )

    assert "ecg_long_dropout_detected" not in (result.exclusion_reason or "")


def test_feature_contract_has_720_masked_ibi_values() -> None:
    sampling_rate_hz = 100.0
    time = np.arange(0.0, 180.0, 1.0 / sampling_rate_hz)
    signal = np.sin(2.0 * np.pi * 1.1 * time) + 0.1 * np.sin(2.0 * np.pi * 2.2 * time)
    features, sequence, mask = extract_ppg_features(signal, sampling_rate_hz)

    assert len(features) >= 30
    assert sequence.shape == mask.shape == (720,)
    assert mask.sum() > 0
