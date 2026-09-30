"""Entry point for canonical ECG-PPG peak-to-peak validation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from PipelinePython.common.workflow_entry import execute_workflow


def run(arguments: Namespace, *, project_root: Path) -> dict[str, object]:
    return execute_workflow("peaks", arguments, project_root=project_root)
