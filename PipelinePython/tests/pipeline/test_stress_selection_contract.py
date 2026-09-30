from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PipelinePython.common.workflows.stress_onnx import (
    freeze_stress_selection_contract,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        artifacts={"reports_dir": "reports"},
        model_version="test-model",
        training=SimpleNamespace(
            training_channels=("ppg_green",),
            development_participant_ids=("ID-01",),
        ),
    )


def _paths(tmp_path: Path) -> SimpleNamespace:
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    (artifacts_root / "run_contract_fingerprint.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "source_tree_sha256": "a" * 64,
                "config_fingerprint": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        artifacts_root=artifacts_root,
        models_dir=artifacts_root / "models",
    )


def _write_report(tmp_path: Path, **overrides: object) -> None:
    report = {
        "run_id": "test-run",
        "model_name": "logistic_regression",
        "model_contract_fingerprint": "c" * 64,
        "release_status": "blocked_existing_holdout_contract_mismatch",
        "requires_dl": False,
        "holdout_evaluated": False,
        "selected_quality_cohort": "strict",
        "selected_window_seconds": 180,
        "selected_stride_seconds": 10,
        "threshold": 0.47,
        "predictor_columns": ["rmssd_ms"],
    }
    report.update(overrides)
    report_dir = tmp_path / "reports" / "model_eval" / "test-model"
    report_dir.mkdir(parents=True)
    (report_dir / "ppg_green_full.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def test_freeze_uses_canonical_selected_fields(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_report(tmp_path)

    target = freeze_stress_selection_contract(
        tmp_path,
        _config(),
        paths,
        {
            "ppg_green": {
                "gold_input_validation": {
                    "training_status": "ready",
                    "features_sha256": "d" * 64,
                }
            }
        },
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    decision = payload["channels"]["ppg_green"]
    assert payload["selection_contract_frozen"] is True
    assert decision["terminal_status"] == "FROZEN_CLASSICAL"
    assert decision["quality_cohort"] == "strict"
    assert decision["window_seconds"] == 180
    assert decision["window_stride_seconds"] == 10
    assert decision["decision_threshold"] == 0.47


def test_incomplete_classical_contract_does_not_freeze(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_report(tmp_path, threshold=None)

    target = freeze_stress_selection_contract(
        tmp_path,
        _config(),
        paths,
        {"ppg_green": {"gold_input_validation": {"training_status": "ready"}}},
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["selection_contract_frozen"] is False
    assert payload["channels"]["ppg_green"]["terminal_status"] == "CONTRACT_INVALID"
