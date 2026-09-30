from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_local_gate_accepts_exact_gates_with_accepted_adr() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "PipelinePython/03_stress_onnx/tools/check_frozen_gate.py",
            "--base-ref",
            "HEAD",
            "--adr-reference",
            "ADR-015",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "research 0.70/0.70/0.70" in result.stdout
    assert "preferred 0.80/0.78/0.83" in result.stdout


def test_hosted_workflow_is_disabled_outside_github_workflows() -> None:
    active = ROOT / ".github" / "workflows"
    assert not active.exists() or not any(active.iterdir())
    historical = (
        ROOT
        / "_bmad-output"
        / "specs"
        / "spec-local-biosignal-pipeline"
        / "legacy"
        / "validation.yml.disabled"
    )
    assert historical.is_file()


def test_validation_runner_and_mutating_runner_are_unambiguous() -> None:
    runner = (ROOT / "scripts" / "run_full_pipeline.ps1").read_text(encoding="utf-8")
    assert "mutating_pipeline_stages = $false" in runner
    assert "one_time_cohort_access = $false" in runner
    assert "verify_release_gate.ps1" in runner
    mutating = (ROOT / "scripts" / "run_local_training_pipeline.ps1").read_text(
        encoding="utf-8"
    )
    assert "python -m PipelinePython" in mutating
    assert "'all'" in mutating
    assert "$Resume" in mutating


def test_local_release_adr_supersedes_fabricated_cohort() -> None:
    adr = (
        ROOT
        / "_bmad-output"
        / "specs"
        / "spec-local-biosignal-pipeline"
        / "adr"
        / "ADR-015-existing-32-local-pipeline-contract.md"
    ).read_text(encoding="utf-8")
    assert "supersedes ADR-011" in adr
    assert "balanced accuracy >= 0.80" in adr
    assert "Never create, require, or claim evidence from ID-33..ID-36" in adr
