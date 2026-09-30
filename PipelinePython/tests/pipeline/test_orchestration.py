from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import PipelinePython.common.orchestration as orchestration
from PipelinePython.common.orchestration import (
    CallablePipelineStage,
    PipelineOrchestrator,
    PipelineRunContext,
    StageResult,
)
from PipelinePython.common.settings import PipelineConfig, load_pipeline_config


def test_git_identity_is_scoped_to_source_not_generated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        output = "abc123\n" if command[1:3] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(orchestration.subprocess, "run", fake_run)

    head, _ = orchestration._git_identity(tmp_path)

    assert head == "abc123"
    status_command = calls[1]
    assert status_command[status_command.index("--") + 1 :] == [
        "PipelinePython",
        "pyproject.toml",
        "requirements.lock",
        "uv.lock",
        "PRD.md",
        "AGENTS.md",
    ]
    assert "data/artifacts" not in status_command


def _context(tmp_path: Path) -> PipelineRunContext:
    renamed = tmp_path / "raw" / "renamed"
    trimmed = tmp_path / "raw" / "trimmed"
    renamed.mkdir(parents=True)
    trimmed.mkdir(parents=True)
    (renamed / "ecg.csv").write_text("timestamp,value\n0,1\n", encoding="utf-8")
    (trimmed / "ppg.csv").write_text("timestamp,value\n0,2\n", encoding="utf-8")
    anchor = tmp_path / "raw" / "anchors.csv"
    anchor.write_text("participant_id,phase,start\nID-01,relax,0\n", encoding="utf-8")
    loaded = load_pipeline_config()
    raw_source = replace(
        loaded.raw_source,
        renamed_dir=renamed,
        trimmed_dir=trimmed,
        anchor_csv=anchor,
    )
    artifacts = {
        **loaded.artifacts,
        "artifacts_root": str(tmp_path / "artifacts"),
        "reports_dir": str(tmp_path / "reports"),
    }
    config = replace(loaded, raw_source=raw_source, artifacts=artifacts)
    config_path = tmp_path / "data.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    return PipelineRunContext(
        project_root=tmp_path,
        config=config,
        config_path=config_path,
        run_id="20260809T120000Z",
    )


def _stage(context: PipelineRunContext, calls: list[str]) -> CallablePipelineStage:
    def execute(run_context: PipelineRunContext) -> StageResult:
        calls.append(run_context.run_id)
        output = run_context.artifacts_root / "stage.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"call={len(calls)}\n", encoding="utf-8")
        return StageResult((output,), {"call_count": len(calls)})

    return CallablePipelineStage("fake", execute)


def test_resume_skips_only_matching_stage_outputs(tmp_path: Path) -> None:
    context = _context(tmp_path)
    calls: list[str] = []
    orchestrator = PipelineOrchestrator(context, [_stage(context, calls)])

    first = orchestrator.run(["fake"])
    resumed = orchestrator.run(["fake"], resume=True)

    assert calls == [context.run_id]
    assert first["raw_corpus_sha256"] == resumed["raw_corpus_sha256"]
    assert resumed["raw_integrity_verified"] is True
    assert (context.artifacts_root / "raw_integrity_pre.json").is_file()
    assert (context.artifacts_root / "raw_integrity_post.json").is_file()

    (context.artifacts_root / "stage.txt").write_text("tampered\n", encoding="utf-8")
    orchestrator.run(["fake"], resume=True)
    assert calls == [context.run_id, context.run_id]


def test_resume_rejects_changed_raw_corpus(tmp_path: Path) -> None:
    context = _context(tmp_path)
    calls: list[str] = []
    orchestrator = PipelineOrchestrator(context, [_stage(context, calls)])
    orchestrator.run(["fake"])
    (context.config.raw_source.trimmed_dir / "ppg.csv").write_text(
        "timestamp,value\n0,999\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="Raw corpus changed"):
        orchestrator.run(["fake"], resume=True)


def test_existing_run_rejects_changed_config_without_resume(tmp_path: Path) -> None:
    context = _context(tmp_path)
    calls: list[str] = []
    PipelineOrchestrator(context, [_stage(context, calls)]).run(["fake"])
    changed_config: PipelineConfig = replace(
        context.config,
        artifacts={**context.config.artifacts, "reports_dir": str(tmp_path / "other")},
    )
    changed_context = replace(context, config=changed_config)

    with pytest.raises(RuntimeError, match="Pipeline config changed"):
        PipelineOrchestrator(changed_context, [_stage(changed_context, calls)]).run(
            ["fake"]
        )


def test_declared_input_change_invalidates_resume(tmp_path: Path) -> None:
    context = _context(tmp_path)
    dependency = tmp_path / "gold.parquet"
    dependency.write_text("gold-v1", encoding="utf-8")
    output = context.artifacts_root / "result.txt"
    calls: list[str] = []

    def action(_: PipelineRunContext) -> StageResult:
        calls.append(dependency.read_text(encoding="utf-8"))
        output.write_text("result", encoding="utf-8")
        return StageResult((output,))

    stage = CallablePipelineStage("train", action, (dependency,))
    orchestrator = PipelineOrchestrator(context, (stage,))
    orchestrator.run(("train",))
    orchestrator.run(("train",), resume=True)
    dependency.write_text("gold-v2", encoding="utf-8")
    orchestrator.run(("train",), resume=True)

    assert calls == ["gold-v1", "gold-v2"]


def test_interrupted_stage_is_recorded_as_failed(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def interrupt(_: PipelineRunContext) -> StageResult:
        raise KeyboardInterrupt("operator stop")

    stage = CallablePipelineStage("train", interrupt)
    with pytest.raises(KeyboardInterrupt, match="operator stop"):
        PipelineOrchestrator(context, (stage,)).run(("train",))

    state = json.loads(context.state_path.read_text(encoding="utf-8"))
    assert state["stages"]["train"]["status"] == "failed"
    assert state["stages"]["train"]["error_type"] == "KeyboardInterrupt"


def test_missing_declared_input_is_recorded_as_failed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    missing = tmp_path / "missing-model-report"
    stage = CallablePipelineStage(
        "export",
        lambda _: StageResult(()),
        (missing,),
    )

    with pytest.raises(FileNotFoundError, match="Pipeline output does not exist"):
        PipelineOrchestrator(context, (stage,)).run(("export",))

    state = json.loads(context.state_path.read_text(encoding="utf-8"))
    assert state["stages"]["export"]["status"] == "failed"
    assert state["stages"]["export"]["error_type"] == "FileNotFoundError"
