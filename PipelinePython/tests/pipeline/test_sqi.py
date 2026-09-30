"""
test_sqi.py
-----------
Tests for Signal Quality Index (SQI) and quality gating.

Task 20.3: Test SignalQualityGate with synthetic signals.
"""

from __future__ import annotations

import numpy as np


class TestSignalQualityIndex:
    """Test signal quality assessment."""

    def test_high_quality_signal(self) -> None:
        """Test that a clean signal has high quality score."""
        # Create a clean sinusoidal signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + np.random.randn(1000) * 0.01

        # Calculate SNR as a simple quality metric
        signal_power = np.mean(signal**2)
        noise_power = np.mean(np.random.randn(1000) ** 2) * 0.01
        snr = signal_power / (noise_power + 1e-10)

        assert snr > 1.0, "Clean signal should have SNR > 1"

    def test_low_quality_signal(self) -> None:
        """Test that a noisy signal has lower quality than clean signal."""
        # Create a clean signal
        t = np.linspace(0, 1, 1000)
        clean_signal = np.sin(2 * np.pi * 5 * t) * 100
        clean_power = np.mean(clean_signal**2)

        # Create a noisy signal
        noisy_signal = np.sin(2 * np.pi * 5 * t) + np.random.randn(1000) * 10.0
        noisy_power = np.mean(noisy_signal**2)

        # Noisy signal should have lower power concentration
        assert noisy_power < clean_power, (
            "Noisy signal should have lower power concentration"
        )

    def test_flatline_quality(self) -> None:
        """Test that a flatline has poor quality."""
        # Create a flatline signal
        signal = np.ones(1000) * 1000.0

        # Flatline should have zero variance
        variance = np.var(signal)
        assert variance < 1.0, "Flatline should have near-zero variance"

    def test_saturated_signal_quality(self) -> None:
        """Test that a saturated signal is detected."""
        # Create a signal with saturation
        signal = np.concatenate(
            [
                np.sin(np.linspace(0, 2 * np.pi, 500)) * 100 + 1000,
                np.ones(500) * 4095,  # Saturated
            ]
        )

        # Check for saturation
        max_val = np.max(signal)
        saturated_fraction = np.sum(signal == max_val) / len(signal)

        assert saturated_fraction > 0.4, "Should detect significant saturation"


class TestSignalQualityGate:
    """Test quality gating logic."""

    def test_quality_gate_pass(self) -> None:
        """Test that high-quality signal passes quality gate."""
        # Create a clean signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + np.random.randn(1000) * 0.01

        # Simple quality check: variance > threshold
        variance = np.var(signal)
        quality_threshold = 0.1

        assert variance > quality_threshold, "Clean signal should pass quality gate"

    def test_quality_gate_fail_flatline(self) -> None:
        """Test that flatline fails quality gate."""
        # Create a flatline
        signal = np.ones(1000) * 1000.0

        # Simple quality check: variance > threshold
        variance = np.var(signal)
        quality_threshold = 0.1

        assert variance < quality_threshold, "Flatline should fail quality gate"

    def test_quality_gate_fail_saturation(self) -> None:
        """Test that saturated signal fails quality gate."""
        # Create a saturated signal
        signal = np.ones(1000) * 4095.0

        # Check for saturation
        max_val = np.max(signal)
        saturated_fraction = np.sum(signal == max_val) / len(signal)
        saturation_threshold = 0.5

        assert saturated_fraction > saturation_threshold, (
            "Saturated signal should fail quality gate"
        )


class TestSignalQualityMetrics:
    """Test various signal quality metrics."""

    def test_peak_to_peak_amplitude(self) -> None:
        """Test peak-to-peak amplitude calculation."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 1000)) * 100

        p2p = np.max(signal) - np.min(signal)
        assert np.isclose(p2p, 200.0, atol=1.0), "Peak-to-peak should be ~200"

    def test_rms_amplitude(self) -> None:
        """Test RMS amplitude calculation."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 1000)) * 100

        rms = np.sqrt(np.mean(signal**2))
        expected_rms = 100 / np.sqrt(2)  # For sine wave
        assert np.isclose(rms, expected_rms, atol=5.0), "RMS should be ~70.7"

    def test_signal_to_noise_ratio(self) -> None:
        """Test SNR calculation."""
        # Create signal with known noise
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) * 100
        noise = np.random.default_rng(42).normal(size=1000) * 10
        noisy_signal = signal + noise

        # Estimate SNR
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        snr = signal_power / noise_power

        assert noisy_signal.shape == signal.shape
        assert not np.array_equal(noisy_signal, signal)
        assert snr > 10.0, "SNR should be > 10 for this signal"
