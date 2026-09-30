"""Regression tests for conservative development/internal-test release tiers."""

from PipelinePython.common.performance_policy import policy_from_training_config
from PipelinePython.common.settings import TrainingConfig
from PipelinePython.common.training_canonical import _conservative_gate_metrics


def test_preferred_tier_requires_preferred_metrics_on_both_cohorts() -> None:
    policy = policy_from_training_config(TrainingConfig())
    evidence = _conservative_gate_metrics(
        {"balanced_accuracy": 0.732, "f1": 0.737, "roc_auc": 0.831},
        {"balanced_accuracy": 0.875, "f1": 0.889, "roc_auc": 0.938},
    )

    assessment = policy.assess(evidence)

    assert assessment.performance_tier.value == "research_usable"
    assert assessment.deployment_scope.value == "research"


def test_missing_gate_metric_fails_closed() -> None:
    policy = policy_from_training_config(TrainingConfig())
    evidence = _conservative_gate_metrics(
        {"balanced_accuracy": 0.8, "f1": 0.8},
        {"balanced_accuracy": 0.9, "f1": 0.9, "roc_auc": 0.9},
    )

    assessment = policy.assess(evidence)

    assert assessment.performance_tier.value == "below_minimum"
