from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from PipelinePython.common.rerun_report import generate_preprocessing_peak_report
from PipelinePython.common.settings import (
    load_pipeline_config,
    pipeline_paths_from_config,
)


def test_rerun_report_reads_artifacts_without_claiming_release(tmp_path: Path) -> None:
    config = replace(
        load_pipeline_config(),
        artifacts={
            "artifacts_root": str(tmp_path / "artifacts"),
            "bronze_dir": str(tmp_path / "bronze"),
            "silver_dir": str(tmp_path / "silver"),
            "sync_dir": str(tmp_path / "sync"),
            "gold_dir": str(tmp_path / "gold"),
            "eda_dir": str(tmp_path / "eda"),
            "validation_dir": str(tmp_path / "validation"),
            "reports_dir": str(tmp_path / "reports"),
        },
    )
    paths = pipeline_paths_from_config(tmp_path, config)
    for directory in (
        paths.bronze_dir,
        paths.silver_dir,
        paths.gold_dir,
        paths.eda_dir,
        paths.validation_dir / "ppg_ecg",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"source_file": "missing.csv", "source_checksum": "0" * 64}]
    ).to_parquet(paths.bronze_dir / "ingestion_manifest_v3.parquet", index=False)
    pd.DataFrame([{"quality_tier": "good", "participant_id": "ID-01"}]).to_parquet(
        paths.silver_dir / "preprocessing_manifest_v3.parquet", index=False
    )
    (paths.gold_dir / "data_readiness.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "ml_ready_window_count": 1,
                "excluded_window_count": 2,
                "observed_group_count": 1,
                "expected_group_count": 192,
            }
        ),
        encoding="utf-8",
    )
    sync_dir = Path(config.artifacts["sync_dir"])
    sync_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"excluded": False, "alpha": 1.0, "beta_seconds": 0.0}]).to_parquet(
        sync_dir / "synchronization_manifest_v3.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "participant_id": "ID-01",
                "sensor_channel": "ppg_green",
                "status": "ok",
                "exclusion_reason": "",
                "beat_f1": 0.9,
                "ibi_mae_ms": 10.0,
                "ibi_median_absolute_error_ms": 8.0,
                "ibi_rmse_ms": 12.0,
                "ibi_pearson_r": 0.8,
                "ibi_spearman_r": 0.7,
                "hr_mae_bpm": 2.0,
                "bland_altman_bias_ms": 1.0,
                "pat_median_ms": 220.0,
                "interpolated_short_gap": False,
                "long_gap_detected": False,
            }
        ]
    ).to_csv(paths.validation_dir / "ppg_ecg" / "ppg_ecg_metrics.csv", index=False)
    (paths.validation_dir / "ppg_ecg" / "validation_summary.json").write_text(
        json.dumps(
            {
                "total_combinations": 192,
                "expected_combinations": 192,
                "ok_combinations": 1,
                "excluded_combinations": 191,
            }
        ),
        encoding="utf-8",
    )
    for stage in ("raw", "silver", "gold"):
        (paths.eda_dir / f"ydata_profiling_status_{stage}.json").write_text(
            json.dumps({"status": "ready"}), encoding="utf-8"
        )

    output = generate_preprocessing_peak_report(tmp_path, config)

    content = output.read_text(encoding="utf-8")
    assert "Status data: **BLOCKED**" in content
    assert "192/192" in content
    assert "ibi_rmse_ms_mean" in content
    assert "12.000" in content
    assert "ECG failure memblokir `ecg_validation_status`" in content
