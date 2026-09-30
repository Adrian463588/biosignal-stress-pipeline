"""Independent PPG peak detection and ECG--PPG peak-to-peak validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..orchestration import CallablePipelineStage, PipelineRunContext, StageResult
from ..peak_detector_audit import PeakDetectorAudit
from ..peak_validation import generate_canonical_peak_validation_reports
from ..restore_comparison import build_restore_comparison
from ..settings import PipelinePaths


class PeakToPeakWorkflow:
    """Compose development-only detector audit and canonical validation."""

    def __init__(self, project_root: Path, config: Any, paths: PipelinePaths) -> None:
        self.project_root = project_root
        self.config = config
        self.paths = paths

    @property
    def sync_dir(self) -> Path:
        configured = Path(self.config.artifacts.get("sync_dir", "data/sync"))
        return (
            configured if configured.is_absolute() else self.project_root / configured
        )

    @property
    def detector_selection_path(self) -> Path:
        return (
            self.paths.validation_dir / "detector_selection" / "detector_selection.json"
        )

    @property
    def validation_dir(self) -> Path:
        return self.paths.validation_dir / "ppg_ecg_240s"

    def stages(self) -> tuple[CallablePipelineStage, ...]:
        return (
            CallablePipelineStage(
                "audit_peak_detector",
                self.audit_detector,
                (
                    self.paths.silver_dir / "preprocessing_manifest_v3.parquet",
                    self.sync_dir / "synchronization_manifest_v3.parquet",
                ),
            ),
            CallablePipelineStage(
                "validate_selection",
                self.validate_selection,
                (
                    self.paths.silver_dir / "preprocessing_manifest_v3.parquet",
                    self.sync_dir / "synchronization_manifest_v3.parquet",
                    self.detector_selection_path,
                ),
            ),
            CallablePipelineStage(
                "validate_full_cohort",
                self.validate_full_cohort,
                (
                    self.paths.silver_dir / "preprocessing_manifest_v3.parquet",
                    self.sync_dir / "synchronization_manifest_v3.parquet",
                    self.detector_selection_path,
                    self.paths.models_dir / "stress_selection_contract.json",
                ),
            ),
        )

    def audit_detector(self, _: PipelineRunContext) -> StageResult:
        payload = PeakDetectorAudit(self.project_root, self.config, self.paths).run()
        return StageResult(
            (self.detector_selection_path,),
            {
                "frozen_config_verified": payload["frozen_config_verified"],
                "development_participant_count": len(
                    payload["development_participants"]
                ),
            },
        )

    def _validate(self, _: PipelineRunContext, *, evaluation_scope: str) -> StageResult:
        outputs = generate_canonical_peak_validation_reports(
            self.project_root,
            self.config,
            evaluation_scope=evaluation_scope,
        )
        hrv_dir = (
            outputs.hrv_decision_path.parent
            if outputs.hrv_decision_path
            else (self.paths.validation_dir / "hrv_agreement" / evaluation_scope)
        )
        artifacts: list[Path] = [outputs.output_dir, hrv_dir]
        if evaluation_scope == "full_cohort":
            run_id = str(self.config.artifacts.get("run_id", ""))
            comparison = build_restore_comparison(
                self.project_root, run_id, outputs.metrics_path
            )
            artifacts.extend(
                (
                    comparison.json_path,
                    comparison.markdown_path,
                    comparison.report_path,
                )
            )
        return StageResult(
            tuple(artifacts),
            {
                "row_count": outputs.row_count,
                "primary_gate": "hrv_prv_agreement",
                "evaluation_scope": evaluation_scope,
            },
        )

    def validate_selection(self, context: PipelineRunContext) -> StageResult:
        return self._validate(context, evaluation_scope="selection")

    def validate_full_cohort(self, context: PipelineRunContext) -> StageResult:
        return self._validate(context, evaluation_scope="full_cohort")

    def validate(self, context: PipelineRunContext) -> StageResult:
        """Compatibility adapter for callers that used the old method name."""

        return self.validate_full_cohort(context)
