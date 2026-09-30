from __future__ import annotations

import json
from pathlib import Path

import pytest

from PipelinePython.common.export_onnx_dl import (
    PipelineGateError,
    PyTorchONNXExporter,
    _release_evidence,
)
from PipelinePython.common.onnx_v3 import EXPORT_CHANNELS, OnnxBundleExporter


def _valid_report(project_root: Path) -> dict:
    receipt = project_root / "parity-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "completed",
                "candidate_bundle_sha256": "a" * 64,
                "cohort_access_count": 1,
            }
        ),
        encoding="utf-8",
    )
    return {
        "release_status": "candidate_ready",
        "holdout_metrics": {
            "balanced_accuracy": 0.80,
            "f1": 0.78,
            "roc_auc": 0.83,
        },
        "holdout_participants": ["ID-29", "ID-30", "ID-31", "ID-32"],
        "frozen_decision_threshold": 0.57,
        "threshold_provenance": {
            "source": "participant_grouped_development_validation",
        },
        "evaluation_provenance": {
            "cohort_access_count": 1,
            "candidate_frozen_before_cohort": True,
            "threshold_frozen_before_cohort": True,
            "candidate_bundle_sha256": "a" * 64,
            "parity_receipt": receipt.name,
        },
    }


def test_release_evidence_accepts_only_frozen_existing_parity_pool(
    tmp_path: Path,
) -> None:
    evidence = _release_evidence(_valid_report(tmp_path), "a" * 64, tmp_path)
    assert evidence["balanced_accuracy"] == pytest.approx(0.80)
    assert evidence["cohort_ids"] == ["ID-29", "ID-30", "ID-31", "ID-32"]
    assert evidence["decision_threshold"] == pytest.approx(0.57)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report.update(
                holdout_participants=["ID-25", "ID-26", "ID-27", "ID-28"]
            ),
            "ID-29",
        ),
        (
            lambda report: report["evaluation_provenance"].update(
                cohort_access_count=2
            ),
            "access_count",
        ),
        (
            lambda report: report["threshold_provenance"].update(
                source="holdout_optimized"
            ),
            "Threshold source",
        ),
        (
            lambda report: report["holdout_metrics"].update(balanced_accuracy=0.799),
            "balanced_accuracy",
        ),
        (
            lambda report: report["evaluation_provenance"].update(
                candidate_bundle_sha256="b" * 64
            ),
            "checksum",
        ),
    ],
)
def test_release_evidence_rejects_unsafe_candidates(
    mutation, message: str, tmp_path: Path
) -> None:
    report = _valid_report(tmp_path)
    mutation(report)
    with pytest.raises(PipelineGateError, match=message):
        _release_evidence(report, "a" * 64, tmp_path)


def test_canonical_index_contains_only_independent_accepted_ppg_channels() -> None:
    index = json.loads(
        open("data/artifacts/export/model_index.json", encoding="utf-8").read()
    )
    assert index["schema_version"] == "1.0.0"
    if not index["models"]:
        assert index["release_status"].startswith("blocked_")
        return
    assert index["release_status"] == "accepted"
    channels = {model["sensor_channel"] for model in index["models"]}
    assert channels <= {"ppg_green", "ppg_ir", "ppg_red"}
    assert len(channels) == len(index["models"])
    assert all(model["release_status"] == "accepted" for model in index["models"])
    assert "fusion" not in channels


def test_classical_onnx_allowlist_includes_red_without_fusion() -> None:
    assert EXPORT_CHANNELS == ("ppg_green", "ppg_ir", "ppg_red")


def test_dl_onnx_export_rejects_fusion() -> None:
    exporter = object.__new__(PyTorchONNXExporter)
    with pytest.raises(ValueError, match="Unsupported sensor channel"):
        exporter.run("fusion")


def test_red_export_failure_does_not_drop_green_or_ir(monkeypatch) -> None:
    exporter = object.__new__(OnnxBundleExporter)
    recorded: list[list[str]] = []

    def export_sensor(channel: str) -> dict:
        if channel == "ppg_red":
            raise FileNotFoundError("red_candidate_missing")
        return {"sensor_channel": channel, "onnx_parity": {"max_abs_diff": 0.0}}

    monkeypatch.setattr(exporter, "export_sensor", export_sensor)
    monkeypatch.setattr(
        exporter,
        "_mark_readiness_model_release",
        lambda manifests: recorded.append(
            [manifest["sensor_channel"] for manifest in manifests]
        ),
    )

    result = exporter.run()

    assert result["ppg_green"]["sensor_channel"] == "ppg_green"
    assert result["ppg_ir"]["sensor_channel"] == "ppg_ir"
    assert result["ppg_red"] == {
        "sensor_channel": "ppg_red",
        "status": "blocked",
        "reason": "red_candidate_missing",
    }
    assert recorded == [["ppg_green", "ppg_ir"]]
