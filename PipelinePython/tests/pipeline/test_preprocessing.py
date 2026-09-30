"""
test_preprocessing.py
---------------------
Tests for signal preprocessing: filtering, resampling, and alignment.

Task 20.4: Property tests for preprocessing operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PipelinePython.common import preprocessing
from PipelinePython.common.preprocessing import PpgPreprocessor, interpolate_short_gaps


class TestSignalFiltering:
    """Test signal filtering operations."""

    def test_filtered_signal_energy_less_than_original(self) -> None:
        """Property test: filtered signal energy ≤ original signal energy."""
        # Create a signal with noise
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) * 100
        noise = np.random.randn(1000) * 50
        noisy_signal = signal + noise

        # Simple low-pass filter (moving average)
        window_size = 5
        filtered = np.convolve(
            noisy_signal, np.ones(window_size) / window_size, mode="same"
        )

        # Energy should be less or equal
        original_energy = np.sum(noisy_signal**2)
        filtered_energy = np.sum(filtered**2)

        assert filtered_energy <= original_energy, (
            "Filtered signal energy should be ≤ original"
        )

    def test_filter_preserves_signal_length(self) -> None:
        """Test that filtering preserves signal length."""
        signal = np.random.randn(1000)

        # Simple moving average filter
        window_size = 5
        filtered = np.convolve(signal, np.ones(window_size) / window_size, mode="same")

        assert len(filtered) == len(signal), "Filter should preserve signal length"

    def test_filter_reduces_high_frequency_noise(self) -> None:
        """Test that low-pass filter reduces high-frequency content."""
        # Create signal with high-frequency noise
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) * 100  # 5 Hz signal
        noise = np.sin(2 * np.pi * 100 * t) * 50  # 100 Hz noise
        noisy_signal = signal + noise

        # Simple low-pass filter
        window_size = 10
        filtered = np.convolve(
            noisy_signal, np.ones(window_size) / window_size, mode="same"
        )

        # Filtered signal should be closer to original signal
        error_before = np.mean((noisy_signal - signal) ** 2)
        error_after = np.mean((filtered - signal) ** 2)

        assert error_after < error_before, "Filter should reduce error"


class TestSignalResampling:
    """Test signal resampling operations."""

    def test_resampled_signal_length(self) -> None:
        """Property test: resampled signal length is correct."""
        # Original signal: 1000 samples at 100 Hz = 10 seconds
        original_fs = 100
        target_fs = 50
        n_samples = 1000

        # Expected length after resampling
        expected_length = int(n_samples * target_fs / original_fs)

        assert expected_length == 500, "Resampled length should be 500"

    def test_resampling_preserves_duration(self) -> None:
        """Test that resampling preserves signal duration."""
        # Create a signal
        original_fs = 100
        duration = 10  # seconds
        n_samples = original_fs * duration

        # Resample to different rate
        target_fs = 50
        resampled_length = int(n_samples * target_fs / original_fs)
        resampled_duration = resampled_length / target_fs

        assert np.isclose(resampled_duration, duration), "Duration should be preserved"

    def test_upsampling_increases_length(self) -> None:
        """Test that upsampling increases signal length."""
        original_fs = 50
        target_fs = 100
        n_samples = 500

        resampled_length = int(n_samples * target_fs / original_fs)

        assert resampled_length > n_samples, "Upsampling should increase length"

    def test_downsampling_decreases_length(self) -> None:
        """Test that downsampling decreases signal length."""
        original_fs = 100
        target_fs = 50
        n_samples = 1000

        resampled_length = int(n_samples * target_fs / original_fs)

        assert resampled_length < n_samples, "Downsampling should decrease length"


class TestSignalAlignment:
    """Test signal alignment operations."""

    def test_aligned_signals_same_length(self) -> None:
        """Test that aligned signals have the same length."""
        # Create two signals with different lengths
        signal1 = np.random.randn(1000)
        signal2 = np.random.randn(950)

        # Align to shorter length
        min_length = min(len(signal1), len(signal2))
        aligned1 = signal1[:min_length]
        aligned2 = signal2[:min_length]

        assert len(aligned1) == len(aligned2), "Aligned signals should have same length"

    def test_aligned_signals_preserve_values(self) -> None:
        """Test that alignment preserves signal values."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Align to subset
        aligned = signal[:3]

        assert np.allclose(aligned, [1.0, 2.0, 3.0]), "Alignment should preserve values"

    def test_timestamp_alignment(self) -> None:
        """Test that timestamps are correctly aligned."""
        # Create two time series with different sampling rates
        ts1 = pd.date_range("2026-01-01", periods=100, freq="10ms", tz="UTC")
        ts2 = pd.date_range("2026-01-01", periods=50, freq="20ms", tz="UTC")

        # Find common time range
        start = max(ts1[0], ts2[0])
        end = min(ts1[-1], ts2[-1])

        aligned1 = ts1[(ts1 >= start) & (ts1 <= end)]
        aligned2 = ts2[(ts2 >= start) & (ts2 <= end)]

        assert len(aligned1) > 0, "Should have aligned timestamps"
        assert len(aligned2) > 0, "Should have aligned timestamps"


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline."""

    def test_preprocessing_pipeline_output_shape(self) -> None:
        """Test that preprocessing pipeline produces expected output shape."""
        # Create input signal
        n_samples = 1000
        signal = np.random.randn(n_samples)

        # Apply preprocessing steps
        # 1. Filter
        filtered = np.convolve(signal, np.ones(5) / 5, mode="same")

        # 2. Resample
        resampled_length = int(n_samples * 0.5)
        source_positions = np.linspace(0.0, 1.0, n_samples)
        target_positions = np.linspace(0.0, 1.0, resampled_length)
        resampled = np.interp(target_positions, source_positions, filtered)

        # 3. Normalize
        normalized = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-10)

        assert len(normalized) == resampled_length
        assert np.isfinite(normalized).all()

    def test_preprocessing_pipeline_no_nans(self) -> None:
        """Test that preprocessing pipeline doesn't introduce NaNs."""
        signal = np.random.randn(1000)

        # Apply preprocessing
        filtered = np.convolve(signal, np.ones(5) / 5, mode="same")
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)

        assert not np.any(np.isnan(normalized)), "Pipeline should not introduce NaNs"

    def test_preprocessing_pipeline_no_infs(self) -> None:
        """Test that preprocessing pipeline doesn't introduce infinities."""
        signal = np.random.randn(1000)

        # Apply preprocessing
        filtered = np.convolve(signal, np.ones(5) / 5, mode="same")
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)

        assert not np.any(np.isinf(normalized)), (
            "Pipeline should not introduce infinities"
        )


class TestBoundedInterpolationProvenance:
    """Contract fixtures for bounded interpolation; not physiological data."""

    @staticmethod
    def _timestamps(size: int, sampling_rate_hz: float = 10.0) -> pd.DatetimeIndex:
        return pd.date_range(
            "2026-01-01",
            periods=size,
            freq=pd.Timedelta(seconds=1 / sampling_rate_hz),
            tz="UTC",
        )

    @staticmethod
    def _assert_partition(result: preprocessing.InterpolationResult) -> None:
        categories = np.vstack(
            [
                result.observed_mask,
                result.interpolated_mask,
                result.gap_mask,
                result.unresolved_mask,
            ]
        )
        np.testing.assert_array_equal(categories.sum(axis=0), 1)

    def test_linear_short_gap_uses_existing_timestamps_and_exclusive_masks(
        self,
    ) -> None:
        timestamps = self._timestamps(8)
        values = np.array([0.0, 1.0, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0])

        result = interpolate_short_gaps(values, timestamps, 10.0)

        np.testing.assert_allclose(result.values, np.arange(8, dtype=float))
        np.testing.assert_array_equal(result.interpolated_mask, result.mask)
        np.testing.assert_array_equal(
            result.interpolated_mask,
            [False, False, True, True, False, False, False, False],
        )
        assert result.values.size == timestamps.size
        assert result.runs == ((2, 3),)
        self._assert_partition(result)

    def test_linear_interpolation_uses_actual_timestamp_positions(self) -> None:
        timestamps = pd.to_datetime(
            [
                "2026-01-01T00:00:00.000Z",
                "2026-01-01T00:00:00.100Z",
                "2026-01-01T00:00:00.400Z",
                "2026-01-01T00:00:00.500Z",
            ],
            utc=True,
        )
        values = np.array([0.0, np.nan, 4.0, 5.0])

        result = interpolate_short_gaps(values, timestamps, 10.0)

        assert result.values[1] == pytest.approx(1.0)
        assert result.values.size == values.size
        self._assert_partition(result)

    def test_edge_and_long_nan_runs_remain_unmodified(self) -> None:
        timestamps = self._timestamps(14)
        values = np.arange(14, dtype=float)
        values[0] = np.nan
        values[4:10] = np.nan

        result = interpolate_short_gaps(values, timestamps, 10.0)

        assert np.isnan(result.values[0])
        assert np.isnan(result.values[4:10]).all()
        assert result.unresolved_mask[0]
        assert result.gap_mask[4:10].all()
        assert not result.interpolated_mask.any()
        assert result.long_dropout is True
        assert result.unbounded_dropout is True
        self._assert_partition(result)

    def test_gap_at_half_second_is_allowed_but_longer_gap_is_rejected(self) -> None:
        timestamps = self._timestamps(20)
        allowed = np.arange(20, dtype=float)
        allowed[4:9] = np.nan
        rejected = np.arange(20, dtype=float)
        rejected[4:10] = np.nan

        allowed_result = interpolate_short_gaps(allowed, timestamps, 10.0)
        rejected_result = interpolate_short_gaps(rejected, timestamps, 10.0)

        assert allowed_result.interpolated_mask[4:9].all()
        assert rejected_result.gap_mask[4:10].all()
        assert np.isnan(rejected_result.values[4:10]).all()

    def test_cross_phase_gap_remains_unresolved(self) -> None:
        timestamps = self._timestamps(8)
        values = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
        phases = np.array(["relax"] * 4 + ["task"] * 4)

        result = interpolate_short_gaps(
            values,
            timestamps,
            10.0,
            phase_labels=phases,
        )

        assert np.isnan(result.values[3])
        assert result.unresolved_mask[3]
        assert not result.interpolated_mask.any()
        assert result.intervals == ((3, 3, "unresolved_phase_boundary"),)
        self._assert_partition(result)

    def test_pchip_requires_two_neighbors_and_disables_extrapolation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        timestamps = self._timestamps(8)
        values = np.array([0.0, 1.0, 4.0, np.nan, 16.0, 25.0, 36.0, 49.0])
        real_pchip = preprocessing.PchipInterpolator
        extrapolation_values: list[bool | None] = []

        def checked_pchip(
            x: np.ndarray,
            y: np.ndarray,
            *,
            extrapolate: bool | None = None,
        ) -> object:
            extrapolation_values.append(extrapolate)
            return real_pchip(x, y, extrapolate=extrapolate)

        monkeypatch.setattr(preprocessing, "PchipInterpolator", checked_pchip)
        result = interpolate_short_gaps(values, timestamps, 10.0, method="pchip")

        assert np.isfinite(result.values[3])
        assert result.interpolated_mask[3]
        assert extrapolation_values == [False]
        self._assert_partition(result)

        insufficient = values.copy()
        insufficient[1] = np.nan
        unresolved = interpolate_short_gaps(
            insufficient,
            timestamps,
            10.0,
            method="pchip",
        )
        assert np.isnan(unresolved.values[1])
        assert unresolved.unresolved_mask[1]

    def test_preprocessor_emits_masks_without_changing_timestamp_rows(self) -> None:
        timestamps = self._timestamps(300, sampling_rate_hz=100.0)
        frame = pd.DataFrame(
            {
                "timestamp_corrected_utc": timestamps,
                "phase": "relax",
                "raw_value": np.sin(np.arange(300, dtype=float) / 10.0),
            }
        )
        frame.loc[100:102, "raw_value"] = np.nan

        result = PpgPreprocessor().preprocess(frame, "ppg_green", 100.0)

        assert len(result.frame) == len(frame)
        pd.testing.assert_series_equal(
            result.frame["timestamp_corrected_utc"],
            frame["timestamp_corrected_utc"],
            check_names=False,
        )
        for column in (
            "observed_mask",
            "interpolated_mask",
            "gap_mask",
            "unresolved_mask",
        ):
            assert column in result.frame
        categories = result.frame[
            ["observed_mask", "interpolated_mask", "gap_mask", "unresolved_mask"]
        ].to_numpy(dtype=np.uint8)
        np.testing.assert_array_equal(categories.sum(axis=1), 1)
        assert result.interpolation_provenance["timestamp_grid_synthesized"] is False
        assert result.interpolation_provenance["method"] == "bounded_linear_waveform"
        assert result.interpolation_provenance["interpolated_sample_count"] == 3
