"""Augmentation package for biosignal time series."""

from .transforms import (
    AmplitudeInversion,
    AmplitudeScaling,
    BaselineWanderInjection,
    GaussianNoiseInjection,
    TemporalJitter,
    TimeCutMix,
    TimeReversal,
)
from .pipeline import AugmentationPipeline

__all__ = [
    "AmplitudeScaling",
    "GaussianNoiseInjection",
    "BaselineWanderInjection",
    "TemporalJitter",
    "TimeCutMix",
    "TimeReversal",
    "AmplitudeInversion",
    "AugmentationPipeline",
]
