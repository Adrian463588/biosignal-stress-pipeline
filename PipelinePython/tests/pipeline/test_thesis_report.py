from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from PipelinePython.common.settings import PipelinePaths
from PipelinePython.common.thesis_report import ThesisReportGenerator


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


def test_thesis_report_writes_csv_markdown_and_plot(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    for p in [
        paths.gold_features_path.parent,
        paths.eda_dir,
        paths.bronze_dir / "ID-01" / "ppg_green",
        paths.silver_dir / "ID-01" / "Relax",
        paths.gold_dir / "features" / "stress" / "ID-01" / "Relax",
        tmp_path / "reports" / "model_eval" / "v1.0.0",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    gold = pd.DataFrame(
        {
            "participant_code": ["ID-01", "ID-01", "ID-02"],
            "phase": ["Relax", "Relax", "Relax"],
            "sensor_channel": ["ppg_green", "ppg_green", "ppg_green"],
            "window_idx": [0, 1, 0],
            "quality_score": [1.0, 1.0, 1.0],
            "quality_tier": ["excellent", "excellent", "excellent"],
            "mean_hr_bpm": [70.0, 72.0, 75.0],
            "mean_rr_ms": [850.0, 830.0, 800.0],
            "rmssd_ms": [30.0, 32.0, 34.0],
            "std_rr_ms": [40.0, 42.0, 44.0],
            "num_peaks": [70, 72, 75],
            "ppg_green_peak_amp_mean": [1.1, 1.2, 1.3],
        }
    )
    gold.to_csv(paths.gold_features_path, index=False)
    gold.iloc[:2].to_parquet(
        paths.gold_dir
        / "features"
        / "stress"
        / "ID-01"
        / "Relax"
        / "ppg_green.parquet",
        index=False,
    )

    ts = pd.date_range("2026-01-01", periods=100, freq="40ms", tz="UTC")
    signal = pd.DataFrame({"timestamp_utc": ts, "raw_value": range(100)})
    signal.to_csv(paths.bronze_dir / "ID-01" / "ppg_green" / "raw.csv", index=False)
    silver = signal.assign(
        participant_code="ID-01",
        phase="Relax",
        signal_type="PPG_GREEN",
        sampling_rate_hz=25.0,
        quality_tier="excellent",
        quality_flag="PASS",
    )
    silver.to_parquet(
        paths.silver_dir / "ID-01" / "Relax" / "ppg_green.parquet", index=False
    )

    pd.DataFrame(
        {
            "participant_code": ["ID-01"],
            "session_id": ["ID-01_Relaks"],
            "phase": ["Relax"],
            "signal_type": ["PPG_GREEN"],
            "sample_count": [100],
        }
    ).to_csv(paths.eda_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        {
            "participant_code": ["ID-01"],
            "phase": ["Relax"],
            "signal_type": ["PPG_GREEN"],
            "actual_duration_s": [4.0],
            "duration_error_s": [0.0],
        }
    ).to_csv(paths.eda_dir / "duration_consistency.csv", index=False)
    pd.DataFrame(
        {
            "participant_code": ["ID-01"],
            "phase": ["Relax"],
            "signal_type": ["PPG_GREEN"],
            "missing_ratio": [0.0],
            "gap_count": [0],
        }
    ).to_csv(paths.eda_dir / "missingness_discontinuity.csv", index=False)
    pd.DataFrame(
        {
            "participant_code": ["ID-01"],
            "phase": ["Relax"],
            "signal_type": ["PPG_GREEN"],
            "interval_ms_mean": [40.0],
            "interval_ms_std": [0.1],
            "interval_ms_p95": [40.0],
        }
    ).to_csv(paths.eda_dir / "sampling_interval_summary.csv", index=False)

    metrics = {
        "ppg_green": {
            "model_name": "logreg",
            "test_participants": ["ID-01"],
            "test_metrics": {"below_threshold": False},
            "per_participant_metrics": {
                "ID-01": {
                    "accuracy": 0.75,
                    "balanced_accuracy": 0.75,
                    "specificity_relax": 0.5,
                    "f1": 0.8,
                    "roc_auc": 0.9,
                    "pr_auc": 0.91,
                    "confusion_matrix": {
                        "labels": ["Relax", "Task"],
                        "matrix": [[1, 1], [0, 2]],
                    },
                }
            },
        }
    }
    (tmp_path / "reports" / "model_eval" / "v1.0.0" / "stress_metrics.json").write_text(
        json.dumps(metrics)
    )

    out = ThesisReportGenerator(paths).run(make_plots=True)

    assert out.report_path.exists()
    assert out.figure_count >= 1
    assert (tmp_path / "reports" / "thesis" / "participant_eda_summary.csv").exists()
    cls = pd.read_csv(
        tmp_path / "reports" / "thesis" / "participant_classification_metrics.csv"
    )
    assert {
        "accuracy",
        "precision",
        "recall_sensitivity_task",
        "error_rate",
        "difficulty_rank",
    }.issubset(cls.columns)
    assert (
        cls.loc[cls["participant_code"] == "ID-02", "status"].iloc[0]
        == "not_in_test_holdout"
    )
    text = out.report_path.read_text(encoding="utf-8")
    assert "Failure Analysis" in text
    assert "EDA dan Visualisasi" in text
