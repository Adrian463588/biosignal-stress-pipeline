"""Tiered ML/DL performance-policy contracts."""

from __future__ import annotations

from PipelinePython.common.config import (
    MIN_BALANCED_ACCURACY_GATE,
    MIN_F1_GATE,
    MIN_ROC_AUC_GATE,
    get_tiered_performance_gates,
)
from PipelinePython.common.performance_policy import (
    DeploymentScope,
    MetricThresholds,
    PerformancePolicy,
    PerformanceTier,
)


def _policy() -> PerformancePolicy:
    return PerformancePolicy(
        version="ml_dl_tiered_v1",
        minimum_research=MetricThresholds(
            balanced_accuracy=0.70,
            f1=0.70,
            roc_auc=0.70,
        ),
        preferred=MetricThresholds(
            balanced_accuracy=0.80,
            f1=0.78,
            roc_auc=0.83,
        ),
        research_disclaimer=(
            "Model penelitian — validasi internal terbatas. Bukan diagnosis klinis."
        ),
    )


def test_minimum_boundary_is_inclusive_and_below_boundary_fails() -> None:
    policy = _policy()

    passed = policy.assess({"balanced_accuracy": 0.70, "f1": 0.70, "roc_auc": 0.70})
    failed = policy.assess({"balanced_accuracy": 0.6999, "f1": 0.90, "roc_auc": 0.90})

    assert passed.performance_tier is PerformanceTier.RESEARCH_USABLE
    assert passed.deployment_scope is DeploymentScope.RESEARCH
    assert passed.requires_dl is False
    assert failed.performance_tier is PerformanceTier.BELOW_MINIMUM
    assert failed.deployment_scope is DeploymentScope.NONE
    assert failed.requires_dl is True
    assert failed.failing_minimum_metrics == ("balanced_accuracy",)


def test_all_three_metrics_are_gates_but_accuracy_is_report_only() -> None:
    policy = _policy()

    assessment = policy.assess(
        {
            "accuracy": 1.0,
            "balanced_accuracy": 0.80,
            "f1": 0.6999,
            "roc_auc": 0.90,
        }
    )

    assert assessment.performance_tier is PerformanceTier.BELOW_MINIMUM
    assert assessment.failing_minimum_metrics == ("f1",)
    assert assessment.requires_dl is True


def test_preferred_requires_the_preferred_thresholds() -> None:
    policy = _policy()

    preferred = policy.assess({"balanced_accuracy": 0.80, "f1": 0.78, "roc_auc": 0.83})
    research = policy.assess({"balanced_accuracy": 0.80, "f1": 0.78, "roc_auc": 0.8299})

    assert preferred.performance_tier is PerformanceTier.PREFERRED
    assert preferred.deployment_scope is DeploymentScope.STANDARD
    assert research.performance_tier is PerformanceTier.RESEARCH_USABLE
    assert research.deployment_scope is DeploymentScope.RESEARCH
    assert research.disclaimer_required is True


def test_missing_or_non_finite_metric_fails_closed() -> None:
    policy = _policy()

    missing = policy.assess({"balanced_accuracy": 0.90, "f1": 0.90})
    non_finite = policy.assess(
        {"balanced_accuracy": 0.90, "f1": float("nan"), "roc_auc": 0.90}
    )

    assert missing.failing_minimum_metrics == ("roc_auc",)
    assert non_finite.failing_minimum_metrics == ("f1",)


def test_policy_fingerprint_is_canonical_and_policy_specific() -> None:
    policy = _policy()
    same = _policy()
    changed = PerformancePolicy(
        version="ml_dl_tiered_v2",
        minimum_research=policy.minimum_research,
        preferred=policy.preferred,
        research_disclaimer=policy.research_disclaimer,
    )

    assert policy.fingerprint() == same.fingerprint()
    assert policy.fingerprint() != changed.fingerprint()


def test_checked_in_training_config_is_the_tiered_policy_source() -> None:
    gates = get_tiered_performance_gates()

    assert gates["version"] == "ml_dl_tiered_v1"
    assert gates["minimum_research"] == {
        "balanced_accuracy": 0.70,
        "f1": 0.70,
        "roc_auc": 0.70,
    }
    assert gates["preferred"] == {
        "balanced_accuracy": 0.80,
        "f1": 0.78,
        "roc_auc": 0.83,
    }


def test_deprecated_aliases_remain_preferred_and_are_not_canonical_gates() -> None:
    assert MIN_BALANCED_ACCURACY_GATE == 0.80
    assert MIN_F1_GATE == 0.78
    assert MIN_ROC_AUC_GATE == 0.83
    assert (
        _policy()
        .assess({"balanced_accuracy": 0.70, "f1": 0.70, "roc_auc": 0.70})
        .passes_minimum
    )
