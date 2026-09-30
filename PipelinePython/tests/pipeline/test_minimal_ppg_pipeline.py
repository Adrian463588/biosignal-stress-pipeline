from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PipelinePython.common.settings import PipelinePaths
from PipelinePython.common.settings import load_pipeline_config
from PipelinePython.common.local_pipeline import LocalBiosignalPipeline
from PipelinePython.common.madc import MotionArtifactDetector
from PipelinePython.common.preprocessing import PpgPreprocessor, interpolate_short_gaps
from PipelinePython.common.sqi import PostPeakSQA, RawSignalSQA
from PipelinePython.common.features_v3 import extract_ppg_features
import PipelinePython.common.features_v3 as features_v3
from PipelinePython.common.wrangling_eda import WranglingEDA


def _ppg_frame(fs: float = 100.0, seconds: float = 30.0) -> pd.DataFrame:
    samples = int(fs * seconds)
    timestamps = pd.date_range(
        "2026-01-01", periods=samples, freq=pd.Timedelta(seconds=1 / fs), tz="UTC"
    )
    return pd.DataFrame(
        {
            "timestamp_corrected_utc": timestamps,
            "raw_value": 1000.0
            + 100.0 * np.sin(2 * np.pi * 1.2 * np.arange(samples) / fs),
        }
    )


def test_silver_crops_core_before_ppg_preprocessing(tmp_path: Path) -> None:
    loaded = load_pipeline_config()
    segmented_dir = tmp_path / "segmented"
    signal_path = tmp_path / "source.parquet"
    config = replace(
        loaded,
        artifacts={
            **loaded.artifacts,
            "segmented_dir": str(segmented_dir),
            "silver_dir": str(tmp_path / "silver"),
            "bronze_dir": str(tmp_path / "bronze"),
            "sync_dir": str(tmp_path / "sync"),
            "gold_dir": str(tmp_path / "gold"),
            "reports_dir": str(tmp_path / "reports"),
        },
    )
    pipeline = LocalBiosignalPipeline(tmp_path, config)
    segmented_dir.mkdir(parents=True)
    anchor = pd.Timestamp("2026-01-01T00:00:00Z")
    timestamps = pd.date_range(anchor, periods=300, freq="1s")
    pd.DataFrame(
        {
            "timestamp_corrected_utc": timestamps,
            "anchor_start_utc": anchor,
            "raw_value": np.sin(np.arange(300, dtype=float)),
            "sampling_rate_hz": 1.0,
            "source_file": str(signal_path),
            "source_checksum": "a" * 64,
        }
    ).to_parquet(signal_path, index=False)
    pd.DataFrame(
        [
            {
                "participant_id": "ID-01",
                "phase": "relax",
                "sensor_channel": "ppg_green",
                "status": "ok",
                "path": str(signal_path),
            }
        ]
    ).to_parquet(segmented_dir / "segmentation_manifest_v3.parquet", index=False)

    class CoreObserved(RuntimeError):
        pass

    def observe_core(
        frame: pd.DataFrame, channel: str, sampling_rate_hz: float
    ) -> None:
        assert channel == "ppg_green"
        assert sampling_rate_hz == 1.0
        observed = pd.to_datetime(frame["timestamp_corrected_utc"], utc=True)
        assert len(frame) == 240
        assert observed.min() == anchor + pd.Timedelta(seconds=30)
        assert observed.max() == anchor + pd.Timedelta(seconds=269)
        raise CoreObserved

    pipeline._ppg_preprocessor = SimpleNamespace(preprocess=observe_core)

    with pytest.raises(CoreObserved):
        pipeline.preprocess()


def test_madc_large_uncalibrated_accelerometer_does_not_modify_signal() -> None:
    signal = np.linspace(100.0, 200.0, 500)
    accel = np.column_stack(
        [
            np.full(signal.size, 1000.0),
            np.full(signal.size, -1000.0),
            np.zeros(signal.size),
        ]
    )
    result = MotionArtifactDetector(
        {"accelerometer_scale_status": "uncalibrated"}
    ).process_window(
        pd.DataFrame(
            {
                "raw_value": signal,
                "acc_x": accel[:, 0],
                "acc_y": accel[:, 1],
                "acc_z": accel[:, 2],
                "accelerometer_provenance_status": "VERIFIED",
            }
        )
    )

    np.testing.assert_array_equal(result["cleaned_signal"], signal)
    assert result["accelerometer_scale_status"] == "uncalibrated"
    assert result["provenance"]["signal_modified"] is False
    assert not result["madc_action"].startswith("REJECT")


@pytest.mark.parametrize(
    ("channel", "highcut"),
    [("ppg_ir", 8.0), ("ppg_red", 8.0), ("ppg_green", 5.0)],
)
def test_ppg_filter_is_single_configured_filter(channel: str, highcut: float) -> None:
    result = PpgPreprocessor().preprocess(_ppg_frame(), channel, 100.0)

    assert result.filter_applied is True
    assert result.filter_spec["highcut_hz"] == highcut
    assert result.frame["clean_value"].notna().all()


def test_short_gap_is_masked_and_long_gap_is_not_reconstructed() -> None:
    frame = _ppg_frame()
    values = frame["raw_value"].to_numpy()
    values[100:105] = np.nan
    short = interpolate_short_gaps(values, frame["timestamp_corrected_utc"], 100.0)

    assert short.mask[100:105].all()
    assert np.isfinite(short.values[100:105]).all()

    values[100:250] = np.nan
    long = interpolate_short_gaps(values, frame["timestamp_corrected_utc"], 100.0)

    assert long.long_dropout is True
    assert np.isnan(long.values[100:250]).all()


def test_ppg_quality_failure_is_diagnostic_only_on_clean_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neurokit2 as nk

    def fail_quality(*args: object, **kwargs: object) -> np.ndarray:
        raise RuntimeError("test quality failure")

    monkeypatch.setattr(nk, "ppg_quality", fail_quality)
    values = np.sin(np.linspace(0.0, 30.0 * np.pi, 3_000))
    score, tier, flags = RawSignalSQA().score(
        values,
        channel="ppg_ir",
        sampling_rate_hz=100.0,
        cleaned_values=values,
    )

    assert score > 0.0
    assert tier in {"good", "excellent"}
    assert flags == ["ppg_quality_error:RuntimeError"]


def test_soft_quality_flags_can_remain_training_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neurokit2 as nk

    monkeypatch.setattr(nk, "ppg_quality", lambda signal, **_: np.zeros(len(signal)))
    values = np.sin(np.linspace(0.0, 30.0 * np.pi, 3_000))
    # Two diagnostics are intentionally present; neither is a hard failure.
    timestamps = pd.date_range(
        "2026-01-01", periods=3_000, freq=pd.Timedelta(seconds=0.01), tz="UTC"
    ).to_series(index=range(3_000))
    timestamps.iloc[100:110] = timestamps.iloc[100:110] + pd.Timedelta(seconds=0.04)
    score, tier, flags = RawSignalSQA().score(
        values,
        timestamps_utc=timestamps,
        channel="ppg_ir",
        sampling_rate_hz=100.0,
        cleaned_values=values,
    )

    assert score >= 0.60
    assert tier in {"acceptable", "good", "excellent"}
    assert "nk_low_quality" in flags


def test_no_peak_sequence_is_invalid_and_features_are_ppg_only() -> None:
    values = np.sin(np.linspace(0.0, 30.0 * np.pi, 3_000))
    post = PostPeakSQA().assess(values, [], sampling_rate_hz=100.0)

    assert post.tier == "invalid"
    long_values = np.sin(np.linspace(0.0, 360.0 * np.pi, 18_000))
    features, _, _ = extract_ppg_features(long_values, 100.0)
    assert not any(name.startswith(("ecg_", "pat_", "motion")) for name in features)


def test_non_finite_feature_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    values = np.sin(np.linspace(0.0, 360.0 * np.pi, 18_000))
    monkeypatch.setattr(features_v3, "skew", lambda *_args, **_kwargs: np.nan)

    with pytest.raises(ValueError, match="non_finite_feature"):
        extract_ppg_features(values, 100.0)


def test_eda_writes_explicit_raw_silver_gold_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EDA stages remain traceable without modifying the raw Bronze input."""

    artifacts = tmp_path / "data" / "artifacts"
    paths = PipelinePaths(
        project_root=tmp_path,
        raw_data_root=tmp_path / "data" / "raw",
        anchor_csv_path=tmp_path / "anchors.csv",
        artifacts_root=artifacts,
        bronze_dir=tmp_path / "data" / "bronze",
        silver_dir=tmp_path / "data" / "silver",
        gold_dir=tmp_path / "data" / "gold",
        gold_features_path=tmp_path / "data" / "gold" / "features.parquet",
        eda_dir=artifacts / "eda",
        validation_dir=artifacts / "validation",
        models_dir=artifacts / "models",
        export_dir=artifacts / "export",
    )
    timestamp = pd.date_range("2026-01-01", periods=1_200, freq="10ms", tz="UTC")
    raw_values = np.sin(np.arange(len(timestamp)))
    clean_values = raw_values.copy()
    clean_values[600] = np.nan
    signal = pd.DataFrame(
        {
            "participant_id": "ID-01",
            "session_id": "ID-01_relax",
            "phase": "relax",
            "sensor_channel": "ppg_ir",
            "timestamp_raw_utc": timestamp,
            "timestamp_corrected_utc": timestamp,
            "sampling_rate_hz": 100.0,
            "raw_value": raw_values,
            "clean_value": clean_values,
            "interpolated_mask": np.arange(len(timestamp)) == 600,
        }
    )
    bronze_path = paths.bronze_dir / "ID-01" / "relax" / "ppg_ir.parquet"
    silver_path = paths.silver_dir / "ID-01" / "relax" / "ppg_ir.parquet"
    bronze_path.parent.mkdir(parents=True)
    silver_path.parent.mkdir(parents=True)
    paths.gold_dir.mkdir(parents=True)
    paths.eda_dir.mkdir(parents=True)
    signal.to_parquet(bronze_path, index=False)
    signal.to_parquet(silver_path, index=False)
    pd.DataFrame(
        {
            "participant_id": ["ID-01"],
            "phase": ["relax"],
            "sensor_channel": ["ppg_green"],
            "label": [0],
            "quality_tier": ["good"],
            "feature_ppg": [0.1],
        }
    ).to_parquet(paths.gold_features_path, index=False)

    eda = WranglingEDA(paths)
    monkeypatch.setattr(eda, "_generate_ydata_profiling", lambda *_: None)
    monkeypatch.setattr(eda, "_plot_label_balance", lambda *_: None)
    monkeypatch.setattr(eda, "_plot_sampling_interval", lambda *_: None)

    def assert_waveform_stage(frame: pd.DataFrame, *_: object) -> None:
        assert "raw_value" in frame.columns

    monkeypatch.setattr(eda, "_plot_overlay_example", assert_waveform_stage)
    monkeypatch.setattr(eda, "_plot_raw_vs_filtered_with_peaks", assert_waveform_stage)

    for stage in ("raw", "silver", "gold"):
        eda.run(stage)

    assert (paths.eda_dir / "eda_report_raw.md").is_file()
    assert (paths.eda_dir / "eda_report_silver.md").is_file()
    assert (paths.eda_dir / "eda_report_gold.md").is_file()
    assert (paths.eda_dir / "eda_manifest_raw.json").is_file()
    assert (paths.eda_dir / "eda_manifest_silver.json").is_file()
    assert (paths.eda_dir / "eda_manifest_gold.json").is_file()
    assert (paths.eda_dir / "eda_stage_comparison.csv").is_file()
    raw_counts = pd.read_csv(paths.eda_dir / "raw_sample_counts.csv")
    silver_counts = pd.read_csv(paths.eda_dir / "silver_sample_counts.csv")
    silver_missing = pd.read_csv(paths.eda_dir / "silver_missingness_discontinuity.csv")
    assert raw_counts.loc[0, "sample_count"] == 1_200
    assert silver_counts.loc[0, "sample_count"] == 1_200
    assert silver_missing.loc[0, "gap_count"] == 0
    assert silver_missing.loc[0, "missing_ratio"] == pytest.approx(1 / 1_200)
    assert silver_missing.loc[0, "interpolated_mask_count"] == 1
    comparison = pd.read_csv(paths.eda_dir / "eda_stage_comparison.csv")
    assert comparison.loc[comparison["stage"] == "gold", "group_count"].item() == 1


def test_eda_disables_ydata_analytics_for_local_only_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("YDATA_PROFILING_NO_ANALYTICS", raising=False)
    eda = object.__new__(WranglingEDA)
    eda.paths = SimpleNamespace(eda_dir=tmp_path)

    eda._generate_ydata_profiling(pd.DataFrame(), stage="raw")

    assert os.environ["YDATA_PROFILING_NO_ANALYTICS"] == "1"
