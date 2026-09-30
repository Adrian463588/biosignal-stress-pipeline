import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from PipelinePython.common.local_pipeline import LocalBiosignalPipeline
from PipelinePython.common.preprocessing import resample_timestamped_dataframe


def test_timestamped_resampler_is_deterministic_and_keeps_first_duplicate():
    frame = pd.DataFrame(
        {
            "unix_ms": [0, 0, 40, 80],
            "value": [1.0, 99.0, 2.0, 3.0],
        }
    )

    result = resample_timestamped_dataframe(frame, "value", target_fs=25)

    np.testing.assert_array_equal(result["unix_ms"].to_numpy(), [0, 40, 80])
    np.testing.assert_allclose(result["value"].to_numpy(), [1.0, 2.0, 3.0])


def test_timestamped_resampler_rejects_regression_and_long_gap():
    with pytest.raises(ValueError, match="timestamp_order_regression"):
        resample_timestamped_dataframe(
            pd.DataFrame({"unix_ms": [0, 40, 20], "value": [1.0, 2.0, 3.0]}),
            "value",
            target_fs=25,
        )

    with pytest.raises(ValueError, match="timestamp_gap_exceeds_policy"):
        resample_timestamped_dataframe(
            pd.DataFrame({"unix_ms": [0, 40, 700], "value": [1.0, 2.0, 3.0]}),
            "value",
            target_fs=25,
        )


def test_continuous_ir_contract_antialiases_to_source25_then_model100():
    timestamps = pd.date_range("2026-01-01", periods=401, freq="10ms", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_corrected_utc": timestamps,
            "raw_value": np.sin(np.linspace(0.0, 20.0, len(timestamps))),
            "participant_id": "ID-01",
            "phase": "relax",
            "sensor_channel": "ppg_ir",
        }
    )
    pipeline = object.__new__(LocalBiosignalPipeline)
    pipeline.config = SimpleNamespace(timezone_local="Asia/Jakarta")

    canonical, source_rate, model_rate = pipeline._continuous_ppg_frame(
        frame, "ppg_ir", 100.0
    )

    assert source_rate == 25.0
    assert model_rate == 100.0
    assert canonical["sampling_rate_hz"].eq(100.0).all()
    assert canonical["source_sampling_rate_hz"].eq(100.0).all()
    assert canonical["timestamp_corrected_utc"].is_monotonic_increasing
    assert np.isfinite(canonical["raw_value"]).all()


def test_observed_window_uses_effective_timestamp_rate_without_resampling():
    full_timestamps = pd.date_range("2026-01-01", periods=4_500, freq="40ms", tz="UTC")
    removed = np.arange(300, 4_200, 300)
    timestamps = full_timestamps.delete(removed)
    values = pd.Series(np.sin(np.arange(len(timestamps), dtype=float) / 10.0))
    start = full_timestamps[0]
    end = start + pd.Timedelta(seconds=180)

    observed_times, observed, mask, sampling_rate_hz = (
        LocalBiosignalPipeline._observed_window(
            pd.Series(timestamps), values, start, end
        )
    )

    duration_seconds = (observed_times[-1] - observed_times[0]) / 1_000.0
    expected_rate = (len(observed_times) - 1) / duration_seconds
    assert len(observed) == len(full_timestamps) - len(removed)
    assert not mask.any()
    assert sampling_rate_hz == pytest.approx(expected_rate)
    assert sampling_rate_hz < 25.0
