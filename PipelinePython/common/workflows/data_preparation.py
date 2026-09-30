"""Raw-to-Gold data preparation stages.

This module composes existing canonical services.  It does not read from or
write to the raw source tree except for the read-only audit performed by the
ingestion service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..local_pipeline import LocalBiosignalPipeline
from ..orchestration import CallablePipelineStage, PipelineRunContext, StageResult
from ..settings import PipelinePaths
from ..wrangling_eda import generate_eda_reports


class DataPreparationWorkflow:
    """Build stages from immutable raw inputs to training-ready Gold."""

    def __init__(
        self,
        project_root: Path,
        config: Any,
        paths: PipelinePaths,
        data: LocalBiosignalPipeline,
        reports_root: Path,
    ) -> None:
        self.project_root = project_root
        self.config = config
        self.paths = paths
        self.data = data
        self.reports_root = reports_root

    def stages(self) -> tuple[CallablePipelineStage, ...]:
        """Return the ordered data stages and their durable dependencies."""

        return (
            CallablePipelineStage("validate_inputs", self.validate_inputs),
            CallablePipelineStage("ingest", self.ingest),
            CallablePipelineStage(
                "eda_raw",
                self._eda("raw"),
                (self.paths.bronze_dir / "ingestion_manifest_v3.parquet",),
            ),
            CallablePipelineStage(
                "sync",
                self.sync,
                (self.paths.bronze_dir / "ingestion_manifest_v3.parquet",),
            ),
            CallablePipelineStage(
                "segment",
                self.segment,
                (
                    self.paths.bronze_dir / "ingestion_manifest_v3.parquet",
                    self.data.sync_dir / "synchronization_manifest_v3.parquet",
                ),
            ),
            CallablePipelineStage(
                "preprocess",
                self.preprocess,
                (self.data.segmented_dir / "segmentation_manifest_v3.parquet",),
            ),
            CallablePipelineStage(
                "eda_silver",
                self._eda("silver"),
                (self.paths.silver_dir / "preprocessing_manifest_v3.parquet",),
            ),
            CallablePipelineStage(
                "extract",
                self.extract,
                (self.paths.silver_dir / "preprocessing_manifest_v3.parquet",),
            ),
            CallablePipelineStage(
                "eda_gold",
                self._eda("gold"),
                (
                    self.paths.gold_dir / "gold_features_v3.parquet",
                    self.paths.gold_dir / "data_readiness.json",
                ),
            ),
        )

    def validate_inputs(self, context: PipelineRunContext) -> StageResult:
        frame = self.data.validate_inputs()
        output = self.reports_root / "data_quality" / "raw_input_index_v3.parquet"
        return StageResult((output,), {"row_count": len(frame)})

    def ingest(self, context: PipelineRunContext) -> StageResult:
        frame = self.data.ingest(context.participant)
        return StageResult(
            (
                self.paths.bronze_dir / "ingestion_manifest_v3.parquet",
                self.paths.bronze_dir / "ingestion_issues_v3.parquet",
            ),
            {"row_count": len(frame)},
        )

    def sync(self, context: PipelineRunContext) -> StageResult:
        frame = self.data.sync_raw(context.participant)
        return StageResult(
            (self.data.sync_dir / "synchronization_manifest_v3.parquet",),
            {"row_count": len(frame)},
        )

    def segment(self, context: PipelineRunContext) -> StageResult:
        frame = self.data.segment(context.participant)
        return StageResult(
            (self.data.segmented_dir / "segmentation_manifest_v3.parquet",),
            {"row_count": len(frame)},
        )

    def preprocess(self, context: PipelineRunContext) -> StageResult:
        frame = self.data.preprocess(context.participant)
        return StageResult(
            (self.paths.silver_dir / "preprocessing_manifest_v3.parquet",),
            {"row_count": len(frame)},
        )

    def extract(self, context: PipelineRunContext) -> StageResult:
        frame = self.data.extract_features(context.participant)
        return StageResult(
            (
                self.paths.gold_dir / "gold_features_v3.parquet",
                self.paths.gold_dir / "gold_ibi_sequences_v3.parquet",
                self.paths.gold_dir / "data_readiness.json",
            ),
            {"window_count": len(frame)},
        )

    def _eda(self, stage: str):
        def action(_: PipelineRunContext) -> StageResult:
            generate_eda_reports(self.project_root, self.config, stage=stage)
            outputs = [
                self.paths.eda_dir / f"eda_report_{stage}.md",
                self.paths.eda_dir / f"ydata_profiling_status_{stage}.json",
                self.paths.eda_dir / f"eda_manifest_{stage}.json",
            ]
            if stage == "gold":
                outputs.extend(
                    [
                        self.paths.eda_dir / "eda_stage_comparison.csv",
                        self.paths.eda_dir / "eda_stage_comparison.md",
                    ]
                )
            return StageResult(
                tuple(outputs),
                {"eda_stage": stage},
            )

        return action
