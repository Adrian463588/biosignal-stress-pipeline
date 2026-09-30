from __future__ import annotations

import numpy as np

from PipelinePython.common.legacy_reproduction import (
    LegacyReproductionConfig,
    compressed_peak_intervals,
    cubic_upsample,
    legacy_hrv_features,
    optimal_ihr_lag,
)


def test_cubic_upsampling_is_derived_and_bounded_to_observed_support() -> None:
    times = np.arange(0.0, 2.0, 0.04)
    values = np.sin(2.0 * np.pi * times)

    grid, derived = cubic_upsample(times, values)

    assert len(derived) > len(values)
    assert grid[0] == times[0]
    assert grid[-1] < times[-1]
    assert np.allclose(derived[::4], values[: len(derived[::4])], atol=1e-10)


def test_compressed_adjacency_counts_rejected_segment_bridge() -> None:
    intervals, _, bridges = compressed_peak_intervals(
        [0.0, 1.0, 61.0, 62.0],
        [0, 0, 2, 2],
    )

    assert intervals.tolist() == [1000.0, 60000.0, 1000.0]
    assert bridges == 1


def test_legacy_hrv_records_but_does_not_hide_bridged_segments() -> None:
    config = LegacyReproductionConfig(minimum_intervals=3)
    peaks = np.arange(0.0, 13.0, 1.0)
    segment_ids = np.array([0] * 6 + [2] * 7)

    features, audit = legacy_hrv_features(peaks, segment_ids, config)

    assert audit["bridged_rejected_segment_count"] == 1
    assert np.isfinite(features["rmssd_ms"])


def test_ihr_lag_reproduction_recovers_known_constant_shift() -> None:
    intervals = 0.8 + np.random.default_rng(42).normal(0.0, 0.05, 120)
    ecg = np.cumsum(np.r_[0.0, intervals])
    ppg = ecg + 3.0

    lag, before, after = optimal_ihr_lag(ecg, ppg, LegacyReproductionConfig())

    assert abs(lag - 3.0) <= 0.1
    assert np.isfinite(before)
    assert abs(after) >= abs(before)


def test_diagnostic_runner_output_is_outside_canonical_run_root(tmp_path) -> None:
    from PipelinePython.common.legacy_reproduction import LegacyReproductionRunner

    runner = LegacyReproductionRunner(tmp_path, "20260828T213118Z", "20260829T020000Z")

    assert runner.output_root == (
        tmp_path
        / "data"
        / "artifacts"
        / "diagnostics"
        / "pipeline_signal_reproduction"
        / "20260829T020000Z"
    )
    assert "runs" not in runner.output_root.parts
