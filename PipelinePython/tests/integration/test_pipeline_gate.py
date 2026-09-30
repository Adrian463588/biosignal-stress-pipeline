from __future__ import annotations

from pathlib import Path

import pytest

from PipelinePython.common.config import get_locked_balanced_accuracy_gate
from PipelinePython.common.export_onnx_dl import PipelineGateError, _release_evidence


ROOT = Path(__file__).resolve().parents[3]


def _candidate_report(balanced_accuracy: float) -> dict[str, object]:
    return {
        "release_status": "candidate_ready",
        "holdout_metrics": {
            "balanced_accuracy": balanced_accuracy,
            "f1": 0.78,
            "roc_auc": 0.83,
        },
        "holdout_participants": ["ID-29", "ID-30", "ID-31", "ID-32"],
        "frozen_decision_threshold": 0.57,
        "threshold_provenance": {
            "source": "participant_grouped_development_validation"
        },
        "evaluation_provenance": {
            "cohort_access_count": 1,
            "candidate_frozen_before_cohort": True,
            "threshold_frozen_before_cohort": True,
            "candidate_bundle_sha256": "a" * 64,
            "parity_receipt": "reports/parity/receipt.json",
        },
    }


def test_locked_minimum_research_gate_is_exactly_070() -> None:
    assert (
        get_locked_balanced_accuracy_gate(
            ROOT / "PipelinePython" / "configs" / "frozen_gates.yaml"
        )
        == 0.70
    )


@pytest.mark.parametrize("balanced_accuracy", [0.0, 0.799999])
def test_release_evidence_rejects_sub_gate_model(balanced_accuracy: float) -> None:
    with pytest.raises(PipelineGateError, match="balanced_accuracy"):
        _release_evidence(_candidate_report(balanced_accuracy), "a" * 64, ROOT)


def test_release_evidence_accepts_existing_parity_pool_at_gate() -> None:
    # Need to create a fake parity receipt for the test to pass if it checks the file
    receipt_dir = ROOT / "reports" / "parity"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / "receipt.json"
    import json

    receipt_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "candidate_bundle_sha256": "a" * 64,
                "cohort_access_count": 1,
            }
        )
    )
    report = _candidate_report(0.80)
    report["evaluation_provenance"]["parity_receipt"] = "reports/parity/receipt.json"
    evidence = _release_evidence(report, "a" * 64, ROOT)
    assert evidence["balanced_accuracy"] == pytest.approx(0.80)
    assert evidence["cohort_ids"] == ["ID-29", "ID-30", "ID-31", "ID-32"]
