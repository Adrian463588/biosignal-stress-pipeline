"""Canonical immutable-Silver to training-Gold entry point."""

from __future__ import annotations

import re
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from PipelinePython.common.cli import _with_isolated_run
from PipelinePython.common.orchestration import (
    CallablePipelineStage,
    PipelineOrchestrator,
    PipelineRunContext,
    StageResult,
)
from PipelinePython.common.settings import load_pipeline_config
from PipelinePython.common.training_gold import TrainingGoldBuilder

RUN_ID_PATTERN = re.compile(r"\d{8}T\d{6}Z")


def run(arguments: Namespace, *, project_root: Path) -> dict[str, object]:
    """Build real PPG-only model windows without rerunning signal preprocessing."""

    run_id = arguments.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must use UTC format YYYYMMDDTHHMMSSZ")
    if not RUN_ID_PATTERN.fullmatch(arguments.source_run_id):
        raise ValueError("source_run_id must use UTC format YYYYMMDDTHHMMSSZ")

    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = _with_isolated_run(load_pipeline_config(config_path.resolve()), run_id)
    runs_root = project_root / "data" / "artifacts" / "runs"
    output_run_root = runs_root / run_id
    source_manifest = (
        runs_root
        / arguments.source_run_id
        / "silver"
        / "preprocessing_manifest_v3.parquet"
    )

    def build_training_gold(_: PipelineRunContext) -> StageResult:
        result = TrainingGoldBuilder(
            project_root,
            config,
            runs_root / arguments.source_run_id,
            output_run_root,
        ).build()
        return StageResult(
            (
                output_run_root / "gold" / "gold_features_v3.parquet",
                output_run_root / "gold" / "gold_ibi_sequences_v3.parquet",
                output_run_root / "gold" / "data_readiness.json",
                output_run_root / "gold" / "manifest_v3.json",
            ),
            {
                "source_run_id": arguments.source_run_id,
                "observed_training_groups": result["observed_training_groups"],
            },
        )

    context = PipelineRunContext(
        project_root=project_root.resolve(),
        config=config,
        config_path=config_path.resolve(),
        run_id=run_id,
    )
    state = PipelineOrchestrator(
        context,
        (
            CallablePipelineStage(
                "derive_training_gold",
                build_training_gold,
                (source_manifest,),
            ),
        ),
    ).run(("derive_training_gold",))
    return {
        "run_id": run_id,
        "flow": "training-gold",
        "source_run_id": arguments.source_run_id,
        "raw_or_silver_modified": False,
        "state": state,
    }
