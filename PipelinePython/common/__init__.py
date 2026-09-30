"""
PipelinePython/common
------------
Public API surface for the biosignal data wrangling and ML pipeline.

Canonical entry point (CLI):
    python -m PipelinePython <data|peaks|stress-onnx|all>

Programmatic entry point:
    from PipelinePython.common.data_foundation import DataFoundationBuilder
    from PipelinePython.common.settings import DEFAULT_PATHS, PipelineSettings

    paths    = DEFAULT_PATHS
    settings = PipelineSettings()
    builder  = DataFoundationBuilder(paths, settings)
    bronze   = builder.run_ingest()
"""

from .contracts import (
    CANONICAL_SIGNAL_COLUMNS,
    CANONICAL_WINDOW_COLUMNS,
    ModelManifest,
    QualityDecision,
    RawAuditIssue,
    RecordingSelection,
    ReferencePeakSet,
)
from .data_foundation import DataFoundationBuilder
from .ingestion import DataLoader, ParticipantSession, get_sampling_rate_hz
from .settings import DEFAULT_PATHS, PipelinePaths, PipelineSettings
from .sqi import SignalQualityGate

__all__ = [
    # Data contracts
    "CANONICAL_SIGNAL_COLUMNS",
    "CANONICAL_WINDOW_COLUMNS",
    "ModelManifest",
    "RawAuditIssue",
    "RecordingSelection",
    "ReferencePeakSet",
    "QualityDecision",
    # Core classes
    "DataFoundationBuilder",
    "DataLoader",
    "ParticipantSession",
    "SignalQualityGate",
    # Settings
    "DEFAULT_PATHS",
    "PipelinePaths",
    "PipelineSettings",
    # Helpers
    "get_sampling_rate_hz",
]
