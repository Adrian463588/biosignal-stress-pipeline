from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PipelinePython.common.feature_policy import (
    DEFAULT_PREDICTORS,
    EXPERIMENTAL_FEATURES,
    FORBIDDEN_METADATA,
    QUALITY_GATE_FEATURES,
    available_predictors,
    validate_feature_order,
)
from PipelinePython.common.settings import TrainingConfig
from PipelinePython.common.training_canonical import spectral_ablation_is_eligible
from PipelinePython.common.training_protocol import (
    WINDOW_CANDIDATES,
    aggregate_session_probabilities,
    channel_group_readiness,
    outer_participant_folds,
    rank_model_candidates,
    select_quality_cohort,
    select_session_threshold,
    select_window_candidate,
)


def test_three_channels_have_independent_64_group_readiness() -> None:
    groups = {
        (f"ID-{participant:02d}", phase, channel)
        for participant in range(1, 33)
        for phase in ("relax", "task")
        for channel in ("ppg_green", "ppg_ir", "ppg_red")
    }

    readiness = channel_group_readiness(groups)

    assert readiness["expected_total_groups"] == 192
    assert readiness["observed_total_groups"] == 192
    assert readiness["channel_status"] == {
        "ppg_green": "ready",
        "ppg_ir": "ready",
        "ppg_red": "ready",
    }
    assert all(count == 64 for count in readiness["observed_groups"].values())


def test_ninety_percent_group_coverage_admits_partial_gold_without_filling_groups() -> (
    None
):
    groups = {
        (f"ID-{participant:02d}", phase, channel)
        for participant in range(1, 33)
        for phase in ("relax", "task")
        for channel in ("ppg_green", "ppg_ir", "ppg_red")
    }
    missing = {
        ("ID-04", "relax", "ppg_green"),
        ("ID-04", "task", "ppg_green"),
        ("ID-05", "relax", "ppg_green"),
        ("ID-08", "relax", "ppg_green"),
        ("ID-08", "relax", "ppg_ir"),
        ("ID-04", "task", "ppg_red"),
        ("ID-08", "relax", "ppg_red"),
        ("ID-08", "task", "ppg_red"),
        ("ID-11", "relax", "ppg_red"),
        ("ID-12", "relax", "ppg_red"),
        ("ID-12", "task", "ppg_red"),
    }
    readiness = channel_group_readiness(groups - missing)

    assert readiness["observed_total_groups"] == 181
    assert readiness["coverage_status"] == {
        "ppg_green": "ready",
        "ppg_ir": "ready",
        "ppg_red": "blocked",
    }
    assert readiness["channel_status"] == {
        "ppg_green": "blocked",
        "ppg_ir": "blocked",
        "ppg_red": "blocked",
    }
    assert readiness["training_status"] == "ready"
    assert readiness["training_channel_status"] == {
        "ppg_green": "ready",
        "ppg_ir": "ready",
        "ppg_red": "blocked_insufficient_development_coverage",
    }
    assert readiness["participant_coverage_status"] == {
        "ppg_green": "ready",
        "ppg_ir": "ready",
        "ppg_red": "ready",
    }
    assert len(groups - missing) == readiness["observed_total_groups"]


def test_spectral_ablation_requires_gain_coverage_and_stability() -> None:
    default = {"balanced_accuracy": 0.70, "fold_balanced_accuracy_std": 0.05}
    candidate = {"balanced_accuracy": 0.72, "fold_balanced_accuracy_std": 0.05}

    assert spectral_ablation_is_eligible(
        default,
        candidate,
        default_row_count=392,
        candidate_row_count=392,
    )
    assert not spectral_ablation_is_eligible(
        default,
        candidate,
        default_row_count=392,
        candidate_row_count=391,
    )
    unstable = {"balanced_accuracy": 0.75, "fold_balanced_accuracy_std": 0.06}
    assert not spectral_ablation_is_eligible(
        default,
        unstable,
        default_row_count=392,
        candidate_row_count=392,
    )


def test_red_failure_does_not_change_green_or_ir_readiness() -> None:
    groups = {
        (f"ID-{participant:02d}", phase, channel)
        for participant in range(1, 33)
        for phase in ("relax", "task")
        for channel in ("ppg_green", "ppg_ir", "ppg_red")
    }
    for participant in range(1, 7):
        groups.remove((f"ID-{participant:02d}", "relax", "ppg_red"))

    readiness = channel_group_readiness(groups)

    assert readiness["channel_status"]["ppg_green"] == "ready"
    assert readiness["channel_status"]["ppg_ir"] == "ready"
    assert readiness["channel_status"]["ppg_red"] == "blocked"
    assert (
        readiness["training_channel_status"]["ppg_red"]
        == "blocked_insufficient_development_coverage"
    )
    assert readiness["training_channel_status"]["ppg_green"] == "ready"
    assert readiness["training_channel_status"]["ppg_ir"] == "ready"
    assert readiness["training_status"] == "ready"


def test_training_coverage_blocks_below_prespecified_development_gate() -> None:
    groups = {
        (f"ID-{participant:02d}", phase, channel)
        for participant in range(1, 33)
        for phase in ("relax", "task")
        for channel in ("ppg_green", "ppg_ir", "ppg_red")
    }
    for participant in range(1, 7):
        groups.remove((f"ID-{participant:02d}", "relax", "ppg_red"))

    readiness = channel_group_readiness(groups)

    assert readiness["development_coverage"]["ppg_red"]["phase_counts"]["relax"] == 22
    assert (
        readiness["training_channel_status"]["ppg_red"]
        == "blocked_insufficient_development_coverage"
    )


def test_development_group_threshold_is_exactly_fifty_one_of_fifty_six() -> None:
    groups = {
        (f"ID-{participant:02d}", phase, "ppg_green")
        for participant in range(1, 29)
        for phase in ("relax", "task")
    }

    ready_groups = groups - {
        (f"ID-{participant:02d}", "relax", "ppg_green") for participant in range(1, 6)
    }
    blocked_groups = ready_groups - {("ID-06", "relax", "ppg_green")}
    ready = channel_group_readiness(ready_groups)
    blocked = channel_group_readiness(blocked_groups)

    assert ready["training_channel_status"]["ppg_green"] == "ready"
    assert (
        blocked["training_channel_status"]["ppg_green"]
        == "blocked_insufficient_development_coverage"
    )


def test_window_contract_and_outer_folds_are_locked() -> None:
    config = TrainingConfig()

    assert config.training_channels == ("ppg_green", "ppg_ir", "ppg_red")
    assert [
        (item.window_seconds, item.stride_seconds) for item in WINDOW_CANDIDATES
    ] == [
        (60, 30),
        (120, 30),
        (180, 10),
        (180, 30),
        (240, 240),
    ]
    folds = outer_participant_folds(config.development_participant_ids)
    assert len(folds) == 7
    assert all(len(fold) == 4 for fold in folds)
    assert set().union(*map(set, folds)) == set(config.development_participant_ids)
    with pytest.raises(ValueError, match="outer_fold_contract"):
        outer_participant_folds((*config.development_participant_ids[:-1], "ID-29"))


def test_window_replacement_requires_gain_coverage_and_stability() -> None:
    baseline = {
        "window_seconds": 180,
        "stride_seconds": 10,
        "balanced_accuracy": 0.80,
        "coverage": 1.0,
        "participant_std": 0.08,
        "all_gates_passed": True,
    }
    better = {
        "window_seconds": 120,
        "stride_seconds": 30,
        "balanced_accuracy": 0.82,
        "coverage": 1.0,
        "participant_std": 0.08,
        "all_gates_passed": True,
    }
    worse_coverage = {**better, "balanced_accuracy": 0.90, "coverage": 0.99}

    assert select_window_candidate([baseline, better]) == better
    assert select_window_candidate([baseline, worse_coverage]) == baseline
    with pytest.raises(ValueError, match="unknown_window_candidate"):
        select_window_candidate(
            [baseline, {**better, "window_seconds": 90, "stride_seconds": 30}]
        )
    without_stability = {
        key: value for key, value in better.items() if key != "participant_std"
    }
    assert select_window_candidate([baseline, without_stability]) == baseline


def test_feature_policy_rejects_quality_metadata_and_cross_signal_fields() -> None:
    validate_feature_order(DEFAULT_PREDICTORS)
    validate_feature_order(
        (*DEFAULT_PREDICTORS, *EXPERIMENTAL_FEATURES), EXPERIMENTAL_FEATURES
    )

    with pytest.raises(ValueError, match="forbidden_training_feature"):
        validate_feature_order((*DEFAULT_PREDICTORS, EXPERIMENTAL_FEATURES[0]))
    with pytest.raises(ValueError, match="missing_required_training_feature"):
        available_predictors(DEFAULT_PREDICTORS[:-1])

    for forbidden in (
        *QUALITY_GATE_FEATURES,
        *FORBIDDEN_METADATA,
        "ecg_rr_mean",
        "pat_median_ms",
        "raw_mean",
    ):
        with pytest.raises(ValueError, match="forbidden_training_feature"):
            validate_feature_order((*DEFAULT_PREDICTORS, forbidden))


def test_session_aggregation_gives_each_participant_phase_one_vote() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["ID-01"] * 7 + ["ID-02"],
            "phase": ["relax"] * 7 + ["task"],
            "sensor_channel": ["ppg_green"] * 8,
            "label": [0] * 7 + [1],
        }
    )
    probabilities = np.asarray([0.1] * 7 + [0.9])

    aggregated = aggregate_session_probabilities(frame, probabilities)

    assert len(aggregated) == 2
    assert aggregated["participant_phase_weight"].tolist() == [1.0, 1.0]
    assert aggregated["probability"].tolist() == pytest.approx([0.1, 0.9])


def test_session_aggregation_rejects_cross_channel_fusion() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["ID-01", "ID-01"],
            "phase": ["relax", "relax"],
            "sensor_channel": ["ppg_green", "ppg_ir"],
            "label": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="cross_channel"):
        aggregate_session_probabilities(frame, [0.1, 0.2])


def test_threshold_selection_rejects_reused_test_rows() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["ID-29", "ID-29"],
            "phase": ["relax", "task"],
            "sensor_channel": ["ppg_green", "ppg_green"],
            "label": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="development_oof_only"):
        select_session_threshold(frame, [0.1, 0.9])


def test_model_tie_uses_f1_auc_stability_then_simplicity() -> None:
    results = {
        "xgboost": {
            "balanced_accuracy": 0.805,
            "f1": 0.80,
            "roc_auc": 0.84,
            "fold_balanced_accuracy_std": 0.08,
        },
        "logistic_regression": {
            "balanced_accuracy": 0.80,
            "f1": 0.80,
            "roc_auc": 0.84,
            "fold_balanced_accuracy_std": 0.08,
        },
        "extra_trees": {
            "balanced_accuracy": 0.78,
            "f1": 0.90,
            "roc_auc": 0.90,
            "fold_balanced_accuracy_std": 0.01,
        },
    }

    assert rank_model_candidates(results) == (
        "logistic_regression",
        "xgboost",
        "extra_trees",
    )
    results["xgboost"]["balanced_accuracy"] = np.nan
    with pytest.raises(ValueError, match="non_finite_model_selection_metric"):
        rank_model_candidates(results)


def test_quality_cohort_prefers_strict_within_tolerance() -> None:
    assert (
        select_quality_cohort(
            {
                "strict": {"available": True, "balanced_accuracy": 0.80},
                "expanded": {"available": True, "balanced_accuracy": 0.81},
            }
        )
        == "strict"
    )
    assert (
        select_quality_cohort(
            {
                "strict": {"available": True, "balanced_accuracy": 0.80},
                "expanded": {"available": True, "balanced_accuracy": 0.82},
            }
        )
        == "expanded"
    )
