"""Shared command boundary for the canonical numbered workflows."""

from __future__ import annotations

import re
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .cli import _with_isolated_run
from .settings import load_pipeline_config
from .workflows.runner import CanonicalWorkflowRunner, WorkflowOptions

WorkflowName = Literal["data", "peaks", "stress-onnx", "all"]
RUN_ID_PATTERN = re.compile(r"\d{8}T\d{6}Z")


def execute_workflow(
    flow: WorkflowName,
    arguments: Namespace,
    *,
    project_root: Path,
) -> dict[str, object]:
    """Execute one canonical flow with one isolated, checksum-bound run root."""

    run_id = arguments.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must use UTC format YYYYMMDDTHHMMSSZ")

    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()
    config = _with_isolated_run(load_pipeline_config(config_path), run_id)

    participant = getattr(arguments, "participant", None)
    sensor = getattr(arguments, "sensor", None)
    export_mode = str(getattr(arguments, "export_mode", "production"))
    if participant and flow != "data":
        raise ValueError("--participant is supported only with the data workflow")
    if sensor and flow != "stress-onnx":
        raise ValueError("--sensor is supported only with the stress-onnx workflow")

    state = CanonicalWorkflowRunner(
        WorkflowOptions(
            project_root=project_root,
            config=config,
            config_path=config_path,
            run_id=run_id,
            participant=participant,
            sensor=sensor,
            export_mode=export_mode,
        )
    ).run(
        flow,
        resume=bool(getattr(arguments, "resume", False)),
        smoke=bool(getattr(arguments, "smoke", False)),
    )

    artifact_root = Path(config.artifacts["artifacts_root"])
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    return {
        "run_id": run_id,
        "flow": flow,
        "state_path": str(artifact_root / "run_state.json"),
        "raw_integrity_verified": state.get("raw_integrity_verified", False),
        "state": state,
    }
