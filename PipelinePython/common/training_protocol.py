"""Pure participant-safe training protocol utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

TRAINING_CHANNELS = ("ppg_green", "ppg_ir", "ppg_red")
MODEL_PHASES = ("relax", "task")
MODEL_SIMPLICITY_ORDER = (
    "logistic_regression",
    "rbf_svm",
    "extra_trees",
    "lightgbm",
    "xgboost",
)


@dataclass(frozen=True)
class WindowCandidate:
    window_seconds: int
    stride_seconds: int

    @property
    def windows_per_core(self) -> int:
        return ((240 - self.window_seconds) // self.stride_seconds) + 1


WINDOW_CANDIDATES = (
    WindowCandidate(60, 30),
    WindowCandidate(120, 30),
    WindowCandidate(180, 10),
    WindowCandidate(180, 30),
    WindowCandidate(240, 240),
)
BASELINE_WINDOW = WindowCandidate(180, 10)


def outer_participant_folds(
    participant_ids: Sequence[str], fold_size: int = 4
) -> tuple[tuple[str, ...], ...]:
    """Return the locked seven outer folds without row-level splitting."""

    participants = tuple(participant_ids)
    expected = tuple(f"ID-{index:02d}" for index in range(1, 29))
    if participants != expected or fold_size != 4:
        raise ValueError("outer_fold_contract_requires_28_unique_participants")
    return tuple(
        participants[index : index + fold_size]
        for index in range(0, len(participants), fold_size)
    )


def aggregate_session_probabilities(
    frame: pd.DataFrame, probabilities: Sequence[float]
) -> pd.DataFrame:
    """Median-aggregate overlapping windows to one participant-phase vote."""

    required = {"participant_id", "phase", "sensor_channel", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"session_aggregation_missing_columns:{missing}")
    values = np.asarray(probabilities, dtype=np.float64)
    if len(frame) != values.size or not np.isfinite(values).all():
        raise ValueError("session_aggregation_probability_mismatch")
    if frame["sensor_channel"].nunique() != 1:
        raise ValueError("cross_channel_session_aggregation_forbidden")
    if "session_id" in frame:
        session_counts = frame.groupby(["participant_id", "phase"])[
            "session_id"
        ].nunique()
        if (session_counts != 1).any():
            raise ValueError("multiple_sessions_per_participant_phase")
    working = frame.loc[
        :, ["participant_id", "phase", "sensor_channel", "label"]
    ].copy()
    working["probability"] = values
    group_columns = ["participant_id", "phase", "sensor_channel"]
    label_counts = working.groupby(group_columns)["label"].nunique()
    if (label_counts != 1).any():
        raise ValueError("participant_phase_has_conflicting_labels")
    result = (
        working.groupby(group_columns, as_index=False)
        .agg(label=("label", "first"), probability=("probability", "median"))
        .sort_values(["participant_id", "phase"], kind="mergesort")
        .reset_index(drop=True)
    )
    result["participant_phase_weight"] = 1.0
    return result


def session_balanced_accuracy(
    frame: pd.DataFrame, probabilities: Sequence[float], threshold: float
) -> float:
    aggregated = aggregate_session_probabilities(frame, probabilities)
    predictions = (aggregated["probability"].to_numpy() >= threshold).astype(int)
    return float(balanced_accuracy_score(aggregated["label"], predictions))


def select_session_threshold(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    candidates: Sequence[float] | None = None,
) -> float:
    """Select a threshold from grouped development OOF predictions only."""

    thresholds = (
        np.linspace(0.20, 0.80, 121, dtype=np.float64)
        if candidates is None
        else np.asarray(candidates, dtype=np.float64)
    )
    if thresholds.size == 0 or not np.isfinite(thresholds).all():
        raise ValueError("invalid_threshold_candidates")
    development = {f"ID-{index:02d}" for index in range(1, 29)}
    observed = set(frame["participant_id"].astype(str))
    if not observed or not observed <= development:
        raise ValueError("threshold_selection_requires_development_oof_only")
    scored = [
        (session_balanced_accuracy(frame, probabilities, float(value)), float(value))
        for value in thresholds
    ]
    return max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]


def channel_group_readiness(
    observed_groups: Iterable[tuple[str, str, str]],
    *,
    minimum_training_development_group_coverage: float = 0.90,
    minimum_development_groups_per_phase: int = 23,
    minimum_development_participants: int = 26,
) -> dict[str, Any]:
    """Evaluate Gold completeness and development-only training admission."""

    if not 0.0 < minimum_training_development_group_coverage <= 1.0:
        raise ValueError("minimum_training_development_group_coverage must be in (0,1]")

    expected_development_groups = 28 * len(MODEL_PHASES)
    required_development_groups = math.ceil(
        minimum_training_development_group_coverage * expected_development_groups
    )

    observed = set(observed_groups)
    expected_by_channel = {
        channel: {
            (f"ID-{participant:02d}", phase, channel)
            for participant in range(1, 33)
            for phase in MODEL_PHASES
        }
        for channel in TRAINING_CHANNELS
    }
    expected_all = set().union(*expected_by_channel.values())
    unexpected = sorted(observed - expected_all)
    if unexpected:
        raise ValueError(f"unexpected_training_groups:{unexpected}")
    observed_counts: dict[str, int] = {}
    missing_groups: dict[str, list[list[str]]] = {}
    statuses: dict[str, str] = {}
    training_statuses: dict[str, str] = {}
    coverage_fractions: dict[str, float] = {}
    coverage_statuses: dict[str, str] = {}
    participant_coverage_fractions: dict[str, float] = {}
    participant_coverage_statuses: dict[str, str] = {}
    development_diagnostics: dict[str, dict[str, Any]] = {}
    for channel, expected in expected_by_channel.items():
        channel_observed = {group for group in observed if group[2] == channel}
        missing = sorted(expected - channel_observed)
        observed_counts[channel] = len(expected & channel_observed)
        all_group_coverage_fraction = observed_counts[channel] / len(expected)
        participants = {group[0] for group in channel_observed}
        participant_coverage_fraction = len(participants) / 32
        participant_coverage_fractions[channel] = participant_coverage_fraction
        missing_groups[channel] = [list(group) for group in missing]
        statuses[channel] = "ready" if not missing else "blocked"
        development = {
            group
            for group in channel_observed
            if 1 <= int(group[0].split("-")[1]) <= 28
        }
        holdout = {
            group
            for group in channel_observed
            if 29 <= int(group[0].split("-")[1]) <= 32
        }
        phase_counts = {
            phase: sum(group[1] == phase for group in development)
            for phase in MODEL_PHASES
        }
        participant_count = len({group[0] for group in development})
        development_coverage_fraction = len(development) / expected_development_groups
        coverage_fractions[channel] = development_coverage_fraction
        coverage_statuses[channel] = (
            "ready" if len(development) >= required_development_groups else "blocked"
        )
        participant_coverage_statuses[channel] = (
            "ready"
            if participant_count >= minimum_development_participants
            else "blocked"
        )
        holdout_complete = len(holdout) == 8
        if len(development) < required_development_groups:
            training_statuses[channel] = "blocked_insufficient_development_coverage"
        elif any(
            count < minimum_development_groups_per_phase
            for count in phase_counts.values()
        ):
            training_statuses[channel] = "blocked_phase_coverage"
        elif participant_count < minimum_development_participants:
            training_statuses[channel] = "blocked_participant_coverage"
        else:
            training_statuses[channel] = "ready"
        development_diagnostics[channel] = {
            "observed_groups": len(development),
            "expected_groups": expected_development_groups,
            "phase_counts": phase_counts,
            "represented_participants": participant_count,
            "minimum_groups": required_development_groups,
            "minimum_groups_per_phase": minimum_development_groups_per_phase,
            "minimum_participants": minimum_development_participants,
            "all_group_coverage_fraction": all_group_coverage_fraction,
            "all_participant_coverage_fraction": participant_coverage_fraction,
            "development_group_coverage_fraction": development_coverage_fraction,
            "development_group_coverage_threshold": (
                minimum_training_development_group_coverage
            ),
            "development_group_coverage_passed": (
                len(development) >= required_development_groups
            ),
            "holdout_observed_groups": len(holdout),
            "holdout_expected_groups": 8,
            "holdout_complete": holdout_complete,
            "holdout_controls_training_admission": False,
        }
    return {
        "expected_groups": {channel: 64 for channel in TRAINING_CHANNELS},
        "observed_groups": observed_counts,
        "missing_groups": missing_groups,
        "channel_status": statuses,
        "gold_completeness_status": statuses,
        "coverage_fraction": coverage_fractions,
        "coverage_status": coverage_statuses,
        "participant_coverage_fraction": participant_coverage_fractions,
        "participant_coverage_status": participant_coverage_statuses,
        "training_coverage_threshold": (minimum_training_development_group_coverage),
        "training_coverage_unit": "participant_phase_channel_group",
        "training_coverage_cohort": "development_ID-01..ID-28",
        "training_channel_status": training_statuses,
        "development_coverage": development_diagnostics,
        "expected_total_groups": 192,
        "observed_total_groups": sum(observed_counts.values()),
        "status": "ready"
        if all(value == "ready" for value in statuses.values())
        else "blocked",
        "training_status": "ready"
        if any(value == "ready" for value in training_statuses.values())
        else "blocked",
    }


def select_window_candidate(
    results: Sequence[Mapping[str, Any]], min_gain: float = 0.02
) -> Mapping[str, Any]:
    """Keep 180/10 unless a candidate clears every replacement guard."""

    baseline = next(
        (
            item
            for item in results
            if int(item["window_seconds"]) == BASELINE_WINDOW.window_seconds
            and int(item["stride_seconds"]) == BASELINE_WINDOW.stride_seconds
        ),
        None,
    )
    if baseline is None:
        raise ValueError("baseline_window_result_missing")
    allowed = {
        (candidate.window_seconds, candidate.stride_seconds)
        for candidate in WINDOW_CANDIDATES
    }
    identities = {
        (int(item["window_seconds"]), int(item["stride_seconds"])) for item in results
    }
    if not identities <= allowed:
        raise ValueError("unknown_window_candidate")
    required_evidence = {"coverage", "participant_std", "all_gates_passed"}
    eligible = [
        item
        for item in results
        if required_evidence <= set(item)
        if bool(item.get("all_gates_passed"))
        and np.isfinite(float(item.get("balanced_accuracy", np.nan)))
        and np.isfinite(float(item["coverage"]))
        and np.isfinite(float(item["participant_std"]))
        and float(item["balanced_accuracy"]) + 1e-12
        >= float(baseline["balanced_accuracy"]) + min_gain
        and float(item["coverage"]) >= float(baseline["coverage"])
        and float(item["participant_std"]) <= float(baseline["participant_std"])
    ]
    return max(
        eligible,
        key=lambda item: (
            float(item["balanced_accuracy"]),
            -float(item.get("participant_std", np.inf)),
        ),
        default=baseline,
    )


def rank_model_candidates(
    results: Mapping[str, Mapping[str, Any]],
    *,
    balanced_accuracy_tolerance: float = 0.01,
) -> tuple[str, ...]:
    """Rank candidates with the PRD tie-break and simplicity policy.

    Balanced accuracy remains primary. Candidates within the declared
    tolerance of the strongest score are ordered by F1, AUC, participant
    stability, and finally export-friendly model simplicity.
    """

    if not results:
        raise ValueError("model_candidate_results_required")

    def metrics(name: str) -> Mapping[str, Any]:
        payload = results[name]
        nested = payload.get("metrics")
        return nested if isinstance(nested, Mapping) else payload

    for name in results:
        values = metrics(name)
        required = (
            "balanced_accuracy",
            "f1",
            "roc_auc",
            "fold_balanced_accuracy_std",
        )
        if not all(
            key in values and np.isfinite(float(values[key])) for key in required
        ):
            raise ValueError(f"non_finite_model_selection_metric:{name}")

    def tie_key(name: str) -> tuple[float, float, float, int]:
        payload = metrics(name)
        try:
            simplicity = MODEL_SIMPLICITY_ORDER.index(name)
        except ValueError:
            simplicity = len(MODEL_SIMPLICITY_ORDER)
        return (
            -float(payload.get("f1", 0.0)),
            -float(payload.get("roc_auc", 0.0)),
            float(payload.get("fold_balanced_accuracy_std", np.inf)),
            simplicity,
        )

    remaining = sorted(
        results,
        key=lambda name: float(metrics(name).get("balanced_accuracy", 0.0)),
        reverse=True,
    )
    ranked: list[str] = []
    while remaining:
        strongest = float(metrics(remaining[0]).get("balanced_accuracy", 0.0))
        tied = [
            name
            for name in remaining
            if strongest - float(metrics(name).get("balanced_accuracy", 0.0))
            <= balanced_accuracy_tolerance + 1e-12
        ]
        ranked.extend(sorted(tied, key=tie_key))
        remaining = [name for name in remaining if name not in tied]
    return tuple(ranked)


def select_quality_cohort(
    results: Mapping[str, Mapping[str, Any]],
    *,
    tie_tolerance: float = 0.01,
) -> str:
    """Choose strict on a BA tie; use expanded only for material improvement."""

    strict = results.get("strict")
    expanded = results.get("expanded")
    if expanded is None or not bool(expanded.get("available", True)):
        if strict is None or not bool(strict.get("available", True)):
            raise ValueError("no_complete_quality_cohort")
        return "strict"
    if strict is None or not bool(strict.get("available", True)):
        return "expanded"
    strict_ba = float(strict.get("balanced_accuracy", 0.0))
    expanded_ba = float(expanded.get("balanced_accuracy", 0.0))
    return "expanded" if expanded_ba > strict_ba + tie_tolerance else "strict"
