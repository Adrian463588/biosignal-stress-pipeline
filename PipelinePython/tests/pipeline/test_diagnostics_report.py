from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from PipelinePython.common.diagnostics_report import SQAPeakDiagnosticsReport
from PipelinePython.common.settings import PipelinePaths


def _paths(tmp_path: Path) -> PipelinePaths:
    artifacts = tmp_path / "data" / "artifacts"
    return PipelinePaths(
        project_root=tmp_path,
        raw_data_root=tmp_path / "data" / "raw" / "DataTrimmedPartisipan",
        anchor_csv_path=tmp_path / "TimeStampAnchor" / "anchors.csv",
        artifacts_root=artifacts,
        bronze_dir=tmp_path / "data" / "bronze",
        silver_dir=tmp_path / "data" / "silver",
        gold_dir=tmp_path / "data" / "gold",
        gold_features_path=artifacts / "gold" / "gold_features.csv",
        eda_dir=artifacts / "eda",
        validation_dir=artifacts / "validation",
        models_dir=artifacts / "models",
        export_dir=artifacts / "export",
    )


def test_sqa_peak_diagnostics_writes_per_subject_outputs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    silver_dir = paths.silver_dir / "ID-01" / "Relax"
    silver_dir.mkdir(parents=True, exist_ok=True)

    fs = 25.0
    seconds = 65
    t = np.arange(0, seconds, 1 / fs)
    signal = (
        1000 + 120 * np.sin(2 * np.pi * 1.25 * t) + 20 * np.sin(2 * np.pi * 2.5 * t)
    )
    df = pd.DataFrame(
        {
            "participant_code": "ID-01",
            "session_id": "ID-01_Relaks",
            "signal_type": "PPG_GREEN",
            "timestamp_utc": pd.date_range(
                "2026-01-01", periods=len(t), freq="40ms", tz="UTC"
            ),
            "timestamp_local": pd.date_range(
                "2026-01-01 07:00:00", periods=len(t), freq="40ms", tz="Asia/Jakarta"
            ),
            "sample_index": np.arange(len(t)),
            "raw_value": signal,
            "sampling_rate_hz": fs,
            "phase": "Relax",
            "anchor_start": pd.Timestamp("2026-01-01", tz="UTC"),
            "anchor_end": pd.Timestamp("2026-01-01 00:01:05", tz="UTC"),
            "quality_flag": "PASS",
            "quality_score": 1.0,
            "quality_tier": "excellent",
            "preprocessing_version": "v2.0.0",
        }
    )
    df.to_parquet(silver_dir / "ppg_green.parquet", index=False)

    out = SQAPeakDiagnosticsReport(paths).run()
    base = out.output_dir / "per_subject" / "ID-01" / "Relax" / "ppg_green"

    required = [
        "sqa_report.csv",
        "accepted_segments.csv",
        "segments_summary.csv",
        "peaks_report.csv",
        "rr_intervals.csv",
        "sqa_plot.png",
        "peak_plot.png",
    ]
    for name in required:
        assert (base / name).exists(), name

    sqa = pd.read_csv(base / "sqa_report.csv")
    peaks = pd.read_csv(base / "peaks_report.csv")
    rr = pd.read_csv(base / "rr_intervals.csv")
    summary = pd.read_csv(out.summary_path)

    assert {"segment_idx", "quality_score", "n_peaks", "final_label"}.issubset(
        sqa.columns
    )
    assert {"peak_time_global_s", "peak_amplitude"}.issubset(peaks.columns)
    assert {"rr_interval_ms", "rr_valid"}.issubset(rr.columns)
    assert summary.loc[0, "participant_code"] == "ID-01"
    assert summary.loc[0, "sensor_channel"] == "ppg_green"
    assert out.participant_sensor_phase_count == 1
    assert out.figure_count == 2
