"""Pure, versioned ML/DL performance-tier policy.

Accuracy is intentionally excluded from the gate. Model identity and release
policy identity are separate so a governance change cannot reopen a reused
internal test cohort.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

GATED_METRICS = ("balanced_accuracy", "f1", "roc_auc")


class PerformanceTier(StrEnum):
    BELOW_MINIMUM = "below_minimum"
    RESEARCH_USABLE = "research_usable"
    PREFERRED = "preferred"


class DeploymentScope(StrEnum):
    NONE = "none"
    RESEARCH = "research"
    STANDARD = "standard"


@dataclass(frozen=True)
class MetricThresholds:
    balanced_accuracy: float
    f1: float
    roc_auc: float

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} threshold must be finite and within [0, 1]")

    def as_dict(self) -> dict[str, float]:
        return {
            "balanced_accuracy": float(self.balanced_accuracy),
            "f1": float(self.f1),
            "roc_auc": float(self.roc_auc),
        }


@dataclass(frozen=True)
class PerformanceAssessment:
    performance_tier: PerformanceTier
    deployment_scope: DeploymentScope
    failing_minimum_metrics: tuple[str, ...]
    failing_preferred_metrics: tuple[str, ...]

    @property
    def passes_minimum(self) -> bool:
        return not self.failing_minimum_metrics

    @property
    def passes_preferred(self) -> bool:
        return not self.failing_preferred_metrics

    @property
    def requires_dl(self) -> bool:
        return not self.passes_minimum

    @property
    def disclaimer_required(self) -> bool:
        return self.performance_tier is PerformanceTier.RESEARCH_USABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "performance_tier": self.performance_tier.value,
            "deployment_scope": self.deployment_scope.value,
            "passes_minimum": self.passes_minimum,
            "passes_preferred": self.passes_preferred,
            "requires_dl": self.requires_dl,
            "disclaimer_required": self.disclaimer_required,
            "failing_minimum_metrics": list(self.failing_minimum_metrics),
            "failing_preferred_metrics": list(self.failing_preferred_metrics),
        }


@dataclass(frozen=True)
class PerformancePolicy:
    version: str
    minimum_research: MetricThresholds
    preferred: MetricThresholds
    research_disclaimer: str

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("performance policy version is required")
        if not self.research_disclaimer.strip():
            raise ValueError("research deployment disclaimer is required")
        for metric in GATED_METRICS:
            if (
                self.preferred.as_dict()[metric]
                < self.minimum_research.as_dict()[metric]
            ):
                raise ValueError(f"preferred {metric} cannot be below minimum research")

    def assess(self, metrics: Mapping[str, Any]) -> PerformanceAssessment:
        minimum_failures = self._failures(metrics, self.minimum_research)
        preferred_failures = self._failures(metrics, self.preferred)
        if minimum_failures:
            tier = PerformanceTier.BELOW_MINIMUM
            scope = DeploymentScope.NONE
        elif preferred_failures:
            tier = PerformanceTier.RESEARCH_USABLE
            scope = DeploymentScope.RESEARCH
        else:
            tier = PerformanceTier.PREFERRED
            scope = DeploymentScope.STANDARD
        return PerformanceAssessment(
            performance_tier=tier,
            deployment_scope=scope,
            failing_minimum_metrics=minimum_failures,
            failing_preferred_metrics=preferred_failures,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "minimum_research": self.minimum_research.as_dict(),
            "preferred": self.preferred.as_dict(),
            "research_disclaimer": self.research_disclaimer,
            "accuracy_is_gate": False,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _failures(
        metrics: Mapping[str, Any], thresholds: MetricThresholds
    ) -> tuple[str, ...]:
        required = thresholds.as_dict()
        failures: list[str] = []
        for metric in GATED_METRICS:
            try:
                value = float(metrics[metric])
            except (KeyError, TypeError, ValueError):
                failures.append(metric)
                continue
            if not math.isfinite(value) or value < required[metric]:
                failures.append(metric)
        return tuple(failures)


def policy_from_training_config(training: Any) -> PerformancePolicy:
    """Build the canonical policy from the validated training configuration."""

    return PerformancePolicy(
        version=str(training.performance_policy_version),
        minimum_research=MetricThresholds(
            balanced_accuracy=float(training.minimum_research_balanced_accuracy),
            f1=float(training.minimum_research_f1),
            roc_auc=float(training.minimum_research_roc_auc),
        ),
        preferred=MetricThresholds(
            balanced_accuracy=float(training.preferred_balanced_accuracy),
            f1=float(training.preferred_f1),
            roc_auc=float(training.preferred_roc_auc),
        ),
        research_disclaimer=str(training.research_disclaimer),
    )
