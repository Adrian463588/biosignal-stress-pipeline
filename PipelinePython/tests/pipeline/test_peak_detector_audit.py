from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from PipelinePython.common.peak_detector_audit import (
    DEVELOPMENT_PARTICIPANTS,
    OUTER_GROUPED_FOLDS,
    PeakDetectorAudit,
)
from PipelinePython.common.settings import PipelinePaths, load_pipeline_config


def _paths(root: Path) -> PipelinePaths:
    artifacts = root / "artifacts"
    return PipelinePaths(
        project_root=root,
        raw_data_root=root / "raw",
        anchor_csv_path=root / "raw" / "anchors.csv",
        artifacts_root=artifacts,
        bronze_dir=artifacts / "bronze",
        silver_dir=artifacts / "silver",
        gold_dir=artifacts / "gold",
        gold_features_path=artifacts / "gold" / "gold_features_v3.parquet",
        eda_dir=artifacts / "eda",
        validation_dir=artifacts / "validation",
        models_dir=artifacts / "models",
        export_dir=artifacts / "export",
    )


def test_detector_audit_is_development_only_and_emits_explicit_exclusions(
    tmp_path: Path,
) -> None:
    loaded = load_pipeline_config()
    config = replace(
        loaded,
        raw_source=replace(
            loaded.raw_source,
            renamed_dir=tmp_path / "raw" / "renamed",
            trimmed_dir=tmp_path / "raw" / "trimmed",
            anchor_csv=tmp_path / "raw" / "anchors.csv",
        ),
        artifacts={
            **loaded.artifacts,
            "sync_dir": str(tmp_path / "sync"),
        },
    )

    payload = PeakDetectorAudit(tmp_path, config, _paths(tmp_path)).run()

    assert payload["development_participants"] == list(DEVELOPMENT_PARTICIPANTS)
    assert payload["holdout_participants_accessed"] == []
    assert payload["frozen_config_verified"] is False
    assert payload["selection_evidence_sufficient"] is False
    assert payload["config_update_required"] is True
    assert len(payload["candidates"]) == 24
    assert payload["candidate_registry"] == [
        "neurokit2_ppg_elgendi",
        "neurokit2_ppg_bishop",
        "neurokit2_ppg_charlton_msptdfast",
        "scipy_find_peaks_adaptive",
        "scipy_legacy_green_native",
    ]
    detail = pd.read_csv(
        tmp_path
        / "artifacts"
        / "validation"
        / "detector_selection"
        / "detector_selection_detail.csv"
    )
    assert len(detail) == 28 * 2 * 3 * 8
    assert not set(detail["participant_id"]) & {
        "ID-29",
        "ID-30",
        "ID-31",
        "ID-32",
    }
    assert (detail["status"] == "excluded").all()


def test_detector_ranking_uses_participant_then_outer_grouped_medians() -> None:
    rows = [
        {
            "participant_id": "ID-01",
            "status": "ok",
            "beat_f1": 0.40,
            "ibi_mae_ms": 80.0,
            "hr_mae_bpm": 4.0,
            "pat_mad_ms": 10.0,
            "ppg_physiological_ibi_ratio": 0.95,
            "ecg_reference_status": "CONSENSUS",
        },
        {
            "participant_id": "ID-01",
            "status": "ok",
            "beat_f1": 0.90,
            "ibi_mae_ms": 40.0,
            "hr_mae_bpm": 2.0,
            "pat_mad_ms": 6.0,
            "ppg_physiological_ibi_ratio": 0.98,
            "ecg_reference_status": "CONSENSUS",
        },
        {
            "participant_id": "ID-02",
            "status": "ok",
            "beat_f1": 0.70,
            "ibi_mae_ms": 60.0,
            "hr_mae_bpm": 3.0,
            "pat_mad_ms": 8.0,
            "ppg_physiological_ibi_ratio": 0.97,
            "ecg_reference_status": "CONSENSUS",
        },
    ]

    summary = PeakDetectorAudit._summarize(
        "ppg_green",
        "neurokit2_ppg_elgendi",
        None,
        rows,
    )

    assert len(OUTER_GROUPED_FOLDS) == 7
    assert summary["participant_count"] == 2
    assert summary["median_beat_f1"] == 0.675
    assert summary["outer_grouped_evaluated_folds"] == 7
    assert summary["outer_grouped_valid_fraction_median"] == 0.0
