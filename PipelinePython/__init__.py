"""Canonical local Python pipeline package.

The public operator entry point is ``python -m PipelinePython``. Domain
implementations live under :mod:`PipelinePython.common`; the numbered folders
contain the three operator-facing workflow runners.
"""

from .common import (
    CANONICAL_SIGNAL_COLUMNS,
    CANONICAL_WINDOW_COLUMNS,
    DataFoundationBuilder,
    DataLoader,
    DEFAULT_PATHS,
    ModelManifest,
    ParticipantSession,
    PipelinePaths,
    PipelineSettings,
    SignalQualityGate,
    get_sampling_rate_hz,
)

__all__ = [
    "CANONICAL_SIGNAL_COLUMNS",
    "CANONICAL_WINDOW_COLUMNS",
    "DataFoundationBuilder",
    "DataLoader",
    "DEFAULT_PATHS",
    "ModelManifest",
    "ParticipantSession",
    "PipelinePaths",
    "PipelineSettings",
    "SignalQualityGate",
    "get_sampling_rate_hz",
]
