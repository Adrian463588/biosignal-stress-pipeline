from __future__ import annotations

import pytest

from PipelinePython.common import cli
from PipelinePython.common.cli import _parse_args
from PipelinePython.common.settings import load_pipeline_config


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["ingest"], "ingest"),
        (["sqa"], "sqa"),
        (["eda"], "eda"),
        (["peaks"], "peaks"),
        (["features"], "features"),
        (["train"], "train"),
        (
            [
                "prepare-training-gold",
                "--source-run-id",
                "20260809T044902Z",
            ],
            "prepare-training-gold",
        ),
        (["prepare-colab-dl"], "prepare-colab-dl"),
        (["export-csv", "--run-root", "run"], "export-csv"),
        (["evaluate"], "evaluate"),
        (["export-onnx"], "export-onnx"),
        (["android-prep"], "android-prep"),
        (["run-pipeline"], "run-pipeline"),
        (["pipeline"], "pipeline"),
        (["generate-eda-report"], "generate-eda-report"),
        (["audit"], "audit"),
    ],
)
def test_v4_public_cli(argv: list[str], command: str) -> None:
    assert _parse_args(argv).command == command


def test_selected_sensor() -> None:
    assert cli._selected_sensor(None) == "ppg_green"
    assert cli._selected_sensor("ppg_red") == "ppg_red"


def test_full_run_uses_an_isolated_artifact_root() -> None:
    config = load_pipeline_config()

    isolated = cli._with_isolated_run(config, "20260801T120000Z")

    assert isolated.raw_source == config.raw_source
    assert isolated.artifacts["gold_dir"] == "data/artifacts/runs/20260801T120000Z/gold"
    assert isolated.artifacts["reports_dir"] == "reports/runs/20260801T120000Z"


@pytest.mark.parametrize(
    "phase", ["ingest", "eda", "validate", "train", "export", "all"]
)
def test_prd_phase_compatibility_interface(phase: str) -> None:
    parsed = _parse_args(
        [
            "--phase",
            phase,
            "--config",
            "PipelinePython/configs/data.yaml",
            "--run-id",
            "20260809T120000Z",
            "--resume",
        ]
    )

    assert parsed.command == "phase"
    assert parsed.phase == phase
    assert parsed.run_id == "20260809T120000Z"
    assert parsed.resume is True


def test_canonical_workflow_cli_contract() -> None:
    parsed = _parse_args(
        [
            "pipeline",
            "--flow",
            "stress-onnx",
            "--sensor",
            "ppg_ir",
            "--run-id",
            "20260821T120000Z",
            "--resume",
        ]
    )

    assert parsed.command == "pipeline"
    assert parsed.flow == "stress-onnx"
    assert parsed.sensor == "ppg_ir"
    assert parsed.run_id == "20260821T120000Z"
    assert parsed.resume is True
