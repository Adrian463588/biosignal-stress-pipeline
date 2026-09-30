from __future__ import annotations

import pandas as pd
import pytest

from PipelinePython.common.anchors import AnchorWindowCoverageError, AnchorWindowSlicer
from PipelinePython.common.settings import PipelineSettings, load_pipeline_config


def _frame(start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=3001, freq="100ms", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "raw_value": range(len(timestamps)),
        }
    )


def test_slicer_returns_240_second_core_without_mutating_input() -> None:
    source = _frame()
    before = source.copy(deep=True)

    result = AnchorWindowSlicer().slice(source, "2026-01-01T00:00:00Z")

    assert result.metadata.anchor_start == pd.Timestamp("2026-01-01", tz="UTC")
    assert result.metadata.anchor_end == pd.Timestamp("2026-01-01 00:05:00", tz="UTC")
    assert result.metadata.core_start == pd.Timestamp("2026-01-01 00:00:30", tz="UTC")
    assert result.metadata.core_end == pd.Timestamp("2026-01-01 00:04:30", tz="UTC")
    assert result.metadata.analysis_phase_duration_seconds == 240
    assert result.metadata.observed_duration_seconds == pytest.approx(239.9)
    assert result.frame["timestamp_utc"].iloc[0] == result.metadata.core_start
    assert result.frame["timestamp_utc"].iloc[-1] < result.metadata.core_end
    pd.testing.assert_frame_equal(source, before)


def test_slicer_does_not_pad_a_short_core_window() -> None:
    source = _frame().iloc[:2690].copy()

    with pytest.raises(AnchorWindowCoverageError, match="below 239"):
        AnchorWindowSlicer().slice(source, "2026-01-01T00:00:00Z")


def test_source_mode_preserves_the_nominal_300_second_phase() -> None:
    result = AnchorWindowSlicer().slice(_frame(), "2026-01-01T00:00:00Z", mode="source")

    assert result.metadata.anchor_start == pd.Timestamp("2026-01-01", tz="UTC")
    assert result.metadata.anchor_end == pd.Timestamp("2026-01-01 00:05:00", tz="UTC")
    assert result.frame["timestamp_utc"].iloc[0] == result.metadata.anchor_start
    assert result.frame["timestamp_utc"].iloc[-1] < result.metadata.anchor_end
    assert result.metadata.source_phase_duration_seconds == 300
    assert result.metadata.analysis_phase_duration_seconds == 240


def test_gold_180_second_windows_with_10_second_stride_fit_core_at_most_seven() -> None:
    starts = list(range(0, 240 - 180 + 1, 10))

    assert starts == [0, 10, 20, 30, 40, 50, 60]
    assert len(starts) == 7
    assert all(start + 180 <= 240 for start in starts)


def test_slicer_rejects_non_monotonic_input_without_sorting() -> None:
    source = _frame()
    source.iloc[[100, 101]] = source.iloc[[101, 100]].to_numpy()

    with pytest.raises(ValueError, match="monotonic"):
        AnchorWindowSlicer().slice(source, "2026-01-01T00:00:00Z")


def test_legacy_pipeline_settings_keep_model_window_contract() -> None:
    settings = PipelineSettings()
    assert settings.analysis_window_seconds == 240
    assert settings.model_window_seconds == 180
    assert settings.model_window_step_seconds == 10


def test_data_and_sync_configs_expose_core_contract() -> None:
    config = load_pipeline_config()
    assert config.window.source_phase_duration_seconds == 300
    assert config.window.analysis_phase_duration_seconds == 240
    assert config.window.validation_window_seconds == 240
    assert config.window.gold_window_seconds == 180
    assert config.window.gold_stride_seconds == 10
    assert config.synchronization.validation_window_seconds == 240
    assert config.synchronization.min_overlap_seconds == 239.0
