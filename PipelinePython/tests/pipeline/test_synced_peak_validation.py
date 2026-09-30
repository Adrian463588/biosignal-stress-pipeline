from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from PipelinePython.common.settings import PipelinePaths
from PipelinePython.common.synced_peak_validation import SyncedPeakValidator


def _paths(tmp_path: Path) -> PipelinePaths:
    return PipelinePaths(
        project_root=tmp_path,
        raw_data_root=tmp_path / "data" / "raw" / "DataTrimmedPartisipan",
        anchor_csv_path=tmp_path / "anchors.csv",
        artifacts_root=tmp_path / "data" / "artifacts",
        bronze_dir=tmp_path / "data" / "bronze",
        silver_dir=tmp_path / "data" / "silver",
        gold_dir=tmp_path / "data" / "gold",
        gold_features_path=tmp_path
        / "data"
        / "artifacts"
        / "gold"
        / "gold_features.csv",
        eda_dir=tmp_path / "data" / "artifacts" / "eda",
        validation_dir=tmp_path / "data" / "artifacts" / "validation",
        models_dir=tmp_path / "data" / "artifacts" / "models",
        export_dir=tmp_path / "data" / "artifacts" / "export",
    )


def test_ptt_matching_recovers_known_lag(tmp_path: Path) -> None:
    validator = SyncedPeakValidator(_paths(tmp_path))
    ecg_ts = np.array([0, 1000, 2000, 3000, 4000], dtype=np.int64)
    ppg_ts = ecg_ts + 220

    result = validator._match_ptt(ecg_ts, ppg_ts)

    assert result["matched_beats"] == 5
    assert result["f1"] == 1.0
    assert result["median_ptt_ms"] == 220.0


def test_failed_row_preserves_lineage(tmp_path: Path) -> None:
    validator = SyncedPeakValidator(_paths(tmp_path))

    row = validator._failed_row(
        "ID-01", "Relaks", "Relax", "ppg_green", "missing_ppg", selected=True
    )

    assert row["participant_code"] == "ID-01"
    assert row["phase_original"] == "Relaks"
    assert row["phase"] == "Relax"
    assert row["sensor_channel"] == "ppg_green"
    assert row["is_selected_reference_channel"] is True
    assert row["status"] == "failed"


def test_session_overview_plot_created(tmp_path: Path) -> None:
    validator = SyncedPeakValidator(_paths(tmp_path))
    rows = [
        {
            "sensor_channel": "ppg_green",
            "is_selected_reference_channel": True,
            "status": "ok",
            "quality_flag": "good",
            "ptt_f1": 0.9,
            "pearson_r": 0.8,
            "ibi_mae_ms": 20.0,
        }
    ]

    validator._plot_session_overview("ID-01", "Relaks", "Relax", rows)

    assert (
        tmp_path
        / "reports"
        / "peaks"
        / "per_subject"
        / "ID-01"
        / "Relaks"
        / "session_overview.png"
    ).exists()


def test_summary_csv_columns_with_stubbed_processing(
    tmp_path: Path, monkeypatch
) -> None:
    synced = tmp_path / "data" / "raw" / "DataSyncedECGPartisipan"
    phase_dir = synced / "ID-01" / "Relaks"
    phase_dir.mkdir(parents=True)
    (phase_dir / "ID-01Relaks_ECG_2026-01-01_synced.csv").write_text(
        "0,0,0,x,0,0,0\n", encoding="utf-8"
    )
    for marker in ["PPG Green", "PPG IR", "PPG Red"]:
        (phase_dir / f"ID-01Relaks-{marker}-2026-01-01.csv").write_text(
            "0,0,x,0,0,0\n", encoding="utf-8"
        )
    pd.DataFrame(
        [
            {
                "participant_code": "ID-01",
                "phase": "Relaks",
                "selected_reference_channel": "ppg_green",
            }
        ]
    ).to_csv(synced / "sync_summary.csv", index=False)

    validator = SyncedPeakValidator(_paths(tmp_path))

    def fake_process(**kwargs):
        out = (
            validator.per_subject_dir
            / kwargs["participant"]
            / kwargs["phase_original"]
            / kwargs["channel"]
        )
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ok": 1}]).to_csv(out / "peak_metrics.csv", index=False)
        pd.DataFrame().to_csv(out / "peak_matches.csv", index=False)
        pd.DataFrame().to_csv(out / "ibi_comparison.csv", index=False)
        (out / "ppg_vs_ecg_peak_overlay.png").write_bytes(b"x")
        (out / "ibi_bland_altman.png").write_bytes(b"x")
        return {
            "participant_code": kwargs["participant"],
            "phase_original": kwargs["phase_original"],
            "phase": kwargs["phase"],
            "sensor_channel": kwargs["channel"],
            "is_selected_reference_channel": kwargs["is_selected_reference"],
            "status": "ok",
            "quality_flag": "good",
        }, 2

    monkeypatch.setattr(validator, "_process_channel", fake_process)

    outputs = validator.run()
    summary = pd.read_csv(outputs.summary_path)

    assert outputs.channel_count == 3
    assert {
        "participant_code",
        "phase_original",
        "phase",
        "sensor_channel",
        "status",
    }.issubset(summary.columns)
    assert "comparison" not in summary.columns
    assert summary["is_selected_reference_channel"].sum() == 1


def test_cohort_outputs_are_created(tmp_path: Path) -> None:
    validator = SyncedPeakValidator(_paths(tmp_path))
    summary = pd.DataFrame(
        [
            {
                "participant_code": "ID-01",
                "phase_original": "Relaks",
                "phase": "Relax",
                "sensor_channel": "ppg_green",
                "is_selected_reference_channel": True,
                "status": "ok",
                "quality_flag": "acceptable",
                "sync_quality": "acceptable",
                "ptt_f1": 0.7,
                "pearson_r": 0.6,
                "ibi_mae_ms": 42.0,
                "ptt_median_ptt_ms": 220.0,
            },
            {
                "participant_code": "ID-02",
                "phase_original": "Stress",
                "phase": "Task",
                "sensor_channel": "ppg_ir",
                "is_selected_reference_channel": True,
                "status": "ok",
                "quality_flag": "poor",
                "sync_quality": "poor",
                "ptt_f1": 0.3,
                "pearson_r": 0.1,
                "ibi_mae_ms": 200.0,
                "ptt_median_ptt_ms": 250.0,
            },
        ]
    )

    validator.out_dir.mkdir(parents=True, exist_ok=True)
    validator._write_cohort_outputs(summary)

    assert (validator.out_dir / "channel_summary.csv").exists()
    assert (validator.out_dir / "participant_summary.csv").exists()
    assert (validator.out_dir / "selected_reference_summary.csv").exists()
    assert (validator.out_dir / "cohort_channel_quality.png").exists()
    assert (validator.out_dir / "cohort_ptt_f1_by_channel.png").exists()
    assert (validator.out_dir / "cohort_ibi_mae_by_channel.png").exists()
    assert (validator.out_dir / "selected_reference_heatmap.png").exists()
    discussion = (validator.out_dir / "ppg_ecg_synced_discussion.md").read_text(
        encoding="utf-8"
    )
    for section in [
        "Ringkasan",
        "Perbandingan Sensor",
        "Failure Analysis",
        "Implikasi",
    ]:
        assert section in discussion
