"""Entry point for participant-safe stress training and ONNX promotion."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from PipelinePython.common.workflow_entry import execute_workflow


def run(arguments: Namespace, *, project_root: Path) -> dict[str, object]:
    flow = "all" if getattr(arguments, "command", None) == "all" else "stress-onnx"
    return execute_workflow(flow, arguments, project_root=project_root)
