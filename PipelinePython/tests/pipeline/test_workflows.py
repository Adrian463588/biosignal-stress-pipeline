from __future__ import annotations

from types import SimpleNamespace

import pytest

from PipelinePython.common.workflows.runner import CanonicalWorkflowRunner
from PipelinePython.common.workflows.data_preparation import DataPreparationWorkflow
from PipelinePython.common.stress_comparison import write_stress_run_comparison


def test_canonical_workflow_stage_order() -> None:
    assert CanonicalWorkflowRunner.stage_names("data") == (
        "validate_inputs",
        "ingest",
        "eda_raw",
        "sync",
        "segment",
        "preprocess",
        "eda_silver",
        "audit_peak_detector",
        "extract",
        "eda_gold",
        "validate_selection",
    )
    assert CanonicalWorkflowRunner.stage_names("peaks") == (
        "audit_peak_detector",
        "validate_selection",
    )
    assert CanonicalWorkflowRunner.stage_names("stress-onnx") == (
        "train",
        "validate_full_cohort",
        "export",
    )
    complete = CanonicalWorkflowRunner.stage_names("all")
    assert (
        complete.index("train")
        < complete.index("validate_full_cohort")
        < complete.index("export")
    )


def test_complete_smoke_stops_before_training() -> None:
    selected = CanonicalWorkflowRunner.stage_names("all", smoke=True)

    assert selected[-1] == "validate_selection"
    assert "train" not in selected
    assert "export" not in selected


def test_internal_export_defers_full_cohort_access_to_publish_command() -> None:
    stress = CanonicalWorkflowRunner.stage_names("stress-onnx", export_mode="internal")
    complete = CanonicalWorkflowRunner.stage_names("all", export_mode="internal")

    assert stress == ("train", "export")
    assert "validate_full_cohort" not in complete
    assert complete[-2:] == ("train", "export")


def test_smoke_is_restricted_to_complete_flow() -> None:
    with pytest.raises(ValueError, match="--smoke is supported only"):
        CanonicalWorkflowRunner.stage_names("peaks", smoke=True)


def test_gold_extraction_is_independent_from_ecg_detector_gate(tmp_path) -> None:
    workflow = object.__new__(DataPreparationWorkflow)
    workflow.paths = SimpleNamespace(gold_dir=tmp_path)
    workflow.data = SimpleNamespace(extract_features=lambda participant: [participant])

    result = workflow.extract(SimpleNamespace(participant="ID-01"))

    assert result.metadata == {"window_count": 1}


def test_stress_comparison_is_run_scoped(tmp_path) -> None:
    current = tmp_path / "data" / "artifacts" / "runs" / "20260907T195324Z"
    current_export = current / "export"
    current_research = current_export / "research"
    current_research.mkdir(parents=True)
    (current_export / "export_status.json").write_text(
        '{"status":"RESEARCH_READY","export_mode":"research"}',
        encoding="utf-8",
    )
    (current_research / "research_model_index.json").write_text(
        '{"run_id":"20260907T195324Z","models":[]}',
        encoding="utf-8",
    )
    for run_id in ("20260830T065109Z", "20260810T031618Z"):
        export = tmp_path / "data" / "artifacts" / "runs" / run_id / "export"
        export.mkdir(parents=True)
        (export / "export_status.json").write_text(
            '{"status":"NO_GO"}',
            encoding="utf-8",
        )

    json_path, markdown_path, report_path = write_stress_run_comparison(
        tmp_path, "20260907T195324Z"
    )

    assert json_path.is_file()
    assert markdown_path.is_file()
    assert report_path.is_file()
    assert "20260830T065109Z" in json_path.read_text(encoding="utf-8")
