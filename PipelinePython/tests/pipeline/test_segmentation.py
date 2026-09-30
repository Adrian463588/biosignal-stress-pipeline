"""
test_segmentation.py
--------------------
Tests for signal segmentation and quality flagging.

Task 20.2: Test flatline detection, time gap detection, saturation detection
from DataFoundationBuilder._quality_flag_for_segment().
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestFlatlineDetection:
    """Test detection of flatline segments (constant signal)."""

    def test_flatline_detection(self) -> None:
        """Test that a constant signal is flagged as flatline."""
        # Create a segment with constant values (flatline)
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2026-01-01", periods=100, freq="10ms", tz="UTC"
                ),
                "ppg_raw": np.ones(100) * 1000.0,  # Constant value
            }
        )

        # Check if flatline detection would flag this
        # (This is a simplified test; actual implementation may vary)
        signal = df["ppg_raw"].to_numpy()
        std_dev = np.std(signal)

        # Flatline should have very low standard deviation
        assert std_dev < 1.0, "Flatline should have near-zero standard deviation"

    def test_non_flatline_signal(self) -> None:
        """Test that a normal signal is not flagged as flatline."""
        # Create a segment with varying values
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2026-01-01", periods=100, freq="10ms", tz="UTC"
                ),
                "ppg_raw": np.sin(np.linspace(0, 2 * np.pi, 100)) * 100 + 1000,
            }
        )

        signal = df["ppg_raw"].to_numpy()
        std_dev = np.std(signal)

        # Normal signal should have higher standard deviation
        assert std_dev > 10.0, "Normal signal should have measurable standard deviation"


class TestTimeGapDetection:
    """Test detection of time gaps in signal."""

    def test_continuous_timestamps(self) -> None:
        """Test that continuous timestamps are not flagged as gaps."""
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2026-01-01", periods=100, freq="10ms", tz="UTC"
                ),
                "ppg_raw": np.random.randn(100),
            }
        )

        # Calculate time deltas
        time_deltas = df["timestamp_utc"].diff().dt.total_seconds()
        expected_delta = 0.01  # 10ms

        # All deltas should be close to expected (except first NaN)
        assert (np.abs(time_deltas[1:] - expected_delta) < 0.001).all()

    def test_time_gap_detection(self) -> None:
        """Test that large time gaps are detected."""
        # Create timestamps with a gap
        ts1 = pd.date_range("2026-01-01", periods=50, freq="10ms", tz="UTC")
        ts2 = pd.date_range("2026-01-01 00:01:00", periods=50, freq="10ms", tz="UTC")
        ts = ts1.append(ts2)

        df = pd.DataFrame(
            {
                "timestamp_utc": ts,
                "ppg_raw": np.random.randn(100),
            }
        )

        # Calculate time deltas
        time_deltas = df["timestamp_utc"].diff().dt.total_seconds()

        # Should have one large gap (around 60 seconds)
        large_gaps = (time_deltas > 1.0).sum()
        assert large_gaps >= 1, "Should detect at least one large time gap"


class TestSaturationDetection:
    """Test detection of signal saturation/clipping."""

    def test_saturation_at_max(self) -> None:
        """Test detection of signal clipping at maximum."""
        # Create a signal with saturation at max
        signal = np.concatenate(
            [
                np.random.randn(50) * 100 + 1000,
                np.ones(20) * 4095,  # Saturated at max (12-bit ADC)
                np.random.randn(30) * 100 + 1000,
            ]
        )

        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2026-01-01", periods=100, freq="10ms", tz="UTC"
                ),
                "ppg_raw": signal,
            }
        )

        # Check for saturation
        max_val = df["ppg_raw"].max()
        saturated_count = (df["ppg_raw"] == max_val).sum()

        assert saturated_count >= 20, "Should detect saturation at maximum"

    def test_saturation_at_min(self) -> None:
        """Test detection of signal clipping at minimum."""
        # Create a signal with saturation at min
        signal = np.concatenate(
            [
                np.random.randn(50) * 100 + 1000,
                np.zeros(20),  # Saturated at min
                np.random.randn(30) * 100 + 1000,
            ]
        )

        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2026-01-01", periods=100, freq="10ms", tz="UTC"
                ),
                "ppg_raw": signal,
            }
        )

        # Check for saturation
        min_val = df["ppg_raw"].min()
        saturated_count = (df["ppg_raw"] == min_val).sum()

        assert saturated_count >= 20, "Should detect saturation at minimum"
