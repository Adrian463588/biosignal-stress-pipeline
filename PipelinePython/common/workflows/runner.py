"""Single composition root for the three canonical biosignal workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..local_pipeline import LocalBiosignalPipeline
from ..orchestration import (
    PipelineOrchestrator,
    PipelineRunContext,
    PipelineStage,
)
from ..settings import PipelineConfig, pipeline_paths_from_config
from .data_preparation import DataPreparationWorkflow
from .peak_to_peak import PeakToPeakWorkflow
from .stress_onnx import StressOnnxWorkflow

WorkflowName = Literal["data", "peaks", "stress-onnx", "all"]


@dataclass(frozen=True)
class WorkflowOptions:
    """Immutable runtime inputs shared by every stage in one workflow run."""

    project_root: Path
    config: PipelineConfig
    config_path: Path
    run_id: str
    participant: str | None = None
    sensor: str | None = None
    export_mode: Literal["production", "research", "internal"] = "production"


class CanonicalWorkflowRunner:
    """Run data, peak-to-peak, or stress-to-ONNX as one resumable graph."""

    _DATA_STAGES = (
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
    _PEAK_STAGES = ("audit_peak_detector", "validate_selection")
    _STRESS_STAGES = ("train", "validate_full_cohort", "export")
    _ALL_STAGES = (
        *_DATA_STAGES,
        "train",
        "validate_full_cohort",
        "export",
    )

    def __init__(self, options: WorkflowOptions) -> None:
        self.options = options
        if options.participant and options.sensor:
            raise ValueError("--participant and --sensor cannot be combined")
        if options.participant and options.run_id == "":
            raise ValueError("run_id is required for participant-scoped data runs")

        project_root = options.project_root.resolve()
        paths = pipeline_paths_from_config(project_root, options.config)
        data = LocalBiosignalPipeline(project_root, options.config)
        reports_root = _configured_path(
            project_root, options.config, "reports_dir", "reports"
        )
        self.context = PipelineRunContext(
            project_root=project_root,
            config=options.config,
            config_path=options.config_path.resolve(),
            run_id=options.run_id,
            participant=options.participant,
        )
        self._data = DataPreparationWorkflow(
            project_root, options.config, paths, data, reports_root
        )
        self._peaks = PeakToPeakWorkflow(project_root, options.config, paths)
        self._stress = StressOnnxWorkflow(
            project_root,
            options.config,
            paths,
            config_path=options.config_path,
            sensor=options.sensor,
            export_mode=options.export_mode,
        )

    @classmethod
    def stage_names(
        cls,
        workflow: WorkflowName,
        *,
        smoke: bool = False,
        export_mode: str = "production",
    ) -> tuple[str, ...]:
        """Return the explicit stage contract without touching project data."""

        if workflow == "data":
            selected = cls._DATA_STAGES
        elif workflow == "peaks":
            selected = cls._PEAK_STAGES
        elif workflow == "stress-onnx":
            selected = cls._STRESS_STAGES
        elif workflow == "all":
            selected = cls._ALL_STAGES
        else:
            raise ValueError(f"Unknown workflow: {workflow}")
        # The internal bundle owns the ID-29..ID-32 access boundary.  Its
        # explicit ``publish-internal`` command is therefore the first stage
        # allowed to read that cohort.  Do not let the physiological full-cohort
        # report silently open it while building frozen development candidates.
        if export_mode == "internal":
            selected = tuple(
                stage for stage in selected if stage != "validate_full_cohort"
            )
        if smoke:
            if workflow != "all":
                raise ValueError("--smoke is supported only with --flow all")
            selected = (*cls._DATA_STAGES,)
        return tuple(selected)

    def run(
        self,
        workflow: WorkflowName,
        *,
        resume: bool = False,
        smoke: bool = False,
    ) -> dict[str, object]:
        if self.options.participant and workflow != "data":
            raise ValueError("--participant is supported only with --flow data")
        if workflow == "stress-onnx" and self.options.participant:
            raise ValueError("Stress training requires the complete participant cohort")

        stage_objects: list[PipelineStage] = [
            *self._data.stages(),
            *self._peaks.stages(),
            *self._stress.stages(),
        ]
        selected = self.stage_names(
            workflow,
            smoke=smoke,
            export_mode=self.options.export_mode,
        )
        return PipelineOrchestrator(self.context, stage_objects).run(
            selected, resume=resume
        )


def _configured_path(
    project_root: Path, config: PipelineConfig, key: str, default: str
) -> Path:
    value = Path(config.artifacts.get(key, default))
    return value if value.is_absolute() else project_root / value
