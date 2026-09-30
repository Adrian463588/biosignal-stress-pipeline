"""Independent ECG-HRV versus PPG-PRV agreement and primary decision gate."""

from __future__ import annotations

import json
import os
import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import welch
from scipy.stats import linregress, pearsonr, spearmanr

from .contracts import (
    DetectedPeaks,
    GapEvent,
    HRV_AGREEMENT_FEATURES,
    HrvAgreementMetric,
    HrvFeaturePair,
    HrvValidationDecision,
    ObservedIntervalBlock,
)
from .settings import HrvAgreementConfig, TRAINING_CHANNELS

SELECTION_PARTICIPANTS = frozenset(f"ID-{index:02d}" for index in range(1, 29))
MODEL_INTERNAL_TEST_PARTICIPANTS = frozenset(
    f"ID-{index:02d}" for index in range(29, 33)
)
PHYSIOLOGICAL_VALIDATION_PARTICIPANTS = (
    SELECTION_PARTICIPANTS | MODEL_INTERNAL_TEST_PARTICIPANTS
)
# Internal aliases keep the interval/range helpers concise. Public artifacts use
# the explicit cohort-role names above.
DEVELOPMENT_PARTICIPANTS = SELECTION_PARTICIPANTS
HOLDOUT_PARTICIPANTS = MODEL_INTERNAL_TEST_PARTICIPANTS
FEATURES = HRV_AGREEMENT_FEATURES
TIME_DOMAIN_FEATURES = FEATURES[:3]
SPECTRAL_FEATURES = FEATURES[3:]
SCOPES = ("pooled", "relax", "task")


@dataclass(frozen=True)
class _DomainDecision:
    decision: str
    pooled_count: int
    relax_count: int
    task_count: int
    minimum_composite: float | None
    mean_composite: float | None
    reasons: tuple[str, ...]


class HrvAgreementAnalyzer:
    """Extract block-safe HRV features and evaluate the hardened legacy score."""

    def __init__(self, config: HrvAgreementConfig) -> None:
        self.config = config

    @staticmethod
    def development_reference_ranges(feature_pairs: pd.DataFrame) -> dict[str, float]:
        return _ecg_reference_ranges(feature_pairs)

    def ecg_reference_columns(
        self,
        *,
        participant_id: str,
        phase: str,
        sensor_channel: str,
        ecg: DetectedPeaks,
        lineage: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return ECG-only HRV reference fields without requiring usable PPG."""

        features, statuses = self._ecg_reference_features(
            participant_id=participant_id,
            phase=phase,
            sensor_channel=sensor_channel,
            ecg=ecg,
            lineage=lineage,
        )
        result: dict[str, Any] = {}
        for feature in FEATURES:
            result[f"ecg_reference_hrv_{feature}"] = features[feature]
            result[f"ecg_reference_hrv_{feature}_status"] = statuses[feature]
        return result

    def _ecg_reference_features(
        self,
        *,
        participant_id: str,
        phase: str,
        sensor_channel: str,
        ecg: DetectedPeaks,
        lineage: Mapping[str, str] | None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        features = {feature: float("nan") for feature in FEATURES}
        statuses = {feature: "INSUFFICIENT_EVIDENCE" for feature in FEATURES}
        try:
            intervals = _observed_intervals(
                participant_id,
                phase,
                sensor_channel,
                "ecg",
                _peak_timestamps(ecg),
                ecg.gap_events,
                self.config,
                lineage,
            )
            features, _, _ = _hrv_feature_result(intervals, self.config)
            statuses = {
                feature: (
                    "COMPUTED"
                    if np.isfinite(features[feature])
                    else "INSUFFICIENT_EVIDENCE"
                )
                for feature in FEATURES
            }
        except ValueError:
            pass
        return features, statuses

    def feature_pair(
        self,
        *,
        participant_id: str,
        phase: str,
        sensor_channel: str,
        ecg: DetectedPeaks,
        ppg: DetectedPeaks,
        window_start_ms: float,
        window_end_ms: float,
        lineage: Mapping[str, str] | None = None,
    ) -> tuple[HrvFeaturePair, list[ObservedIntervalBlock]]:
        """Compute features from independent peak streams on common observed support."""

        reference_features, reference_statuses = self._ecg_reference_features(
            participant_id=participant_id,
            phase=phase,
            sensor_channel=sensor_channel,
            ecg=ecg,
            lineage=lineage,
        )

        common_gaps = tuple((*ecg.gap_events, *ppg.gap_events))
        common_observed = _common_observed_seconds(
            window_start_ms, window_end_ms, common_gaps
        )
        if common_observed < self.config.common_observed_min_seconds:
            return (
                HrvFeaturePair(
                    participant_id,
                    phase,
                    sensor_channel,
                    "INSUFFICIENT_EVIDENCE",
                    ecg_reference_features=reference_features,
                    ecg_reference_feature_statuses=reference_statuses,
                    common_observed_seconds=common_observed,
                    exclusion_reason="insufficient_common_observed_support",
                ),
                [],
            )

        try:
            ecg_intervals = _observed_intervals(
                participant_id,
                phase,
                sensor_channel,
                "ecg",
                _peak_timestamps(ecg),
                common_gaps,
                self.config,
                lineage,
            )
            ppg_intervals = _observed_intervals(
                participant_id,
                phase,
                sensor_channel,
                "ppg",
                _peak_timestamps(ppg),
                common_gaps,
                self.config,
                lineage,
            )
            ecg_features, ecg_reasons, ecg_diagnostics = _hrv_feature_result(
                ecg_intervals, self.config
            )
            ppg_features, ppg_reasons, ppg_diagnostics = _hrv_feature_result(
                ppg_intervals, self.config
            )
        except ValueError as exc:
            return (
                HrvFeaturePair(
                    participant_id,
                    phase,
                    sensor_channel,
                    "INSUFFICIENT_EVIDENCE",
                    ecg_reference_features=reference_features,
                    ecg_reference_feature_statuses=reference_statuses,
                    common_observed_seconds=common_observed,
                    exclusion_reason=str(exc),
                ),
                [
                    *locals().get("ecg_intervals", []),
                    *locals().get("ppg_intervals", []),
                ],
            )

        feature_statuses: dict[str, str] = {}
        feature_reasons: dict[str, str] = {}
        for feature in FEATURES:
            finite = np.isfinite([ecg_features[feature], ppg_features[feature]]).all()
            feature_statuses[feature] = (
                "COMPUTED" if finite else "INSUFFICIENT_EVIDENCE"
            )
            feature_reasons[feature] = (
                ""
                if finite
                else ";".join(
                    reason
                    for reason in (
                        f"ecg:{ecg_reasons.get(feature, '')}"
                        if ecg_reasons.get(feature)
                        else "",
                        f"ppg:{ppg_reasons.get(feature, '')}"
                        if ppg_reasons.get(feature)
                        else "",
                    )
                    if reason
                )
            )
        computed_count = sum(
            status == "COMPUTED" for status in feature_statuses.values()
        )
        if computed_count == len(FEATURES):
            pair_status = "COMPUTED"
            pair_reason = None
        elif computed_count:
            pair_status = "PARTIAL"
            pair_reason = "partial_feature_support"
        else:
            pair_status = "INSUFFICIENT_EVIDENCE"
            pair_reason = "insufficient_hrv_feature_support"

        return (
            HrvFeaturePair(
                participant_id,
                phase,
                sensor_channel,
                pair_status,  # type: ignore[arg-type]
                ecg_features=ecg_features,
                ppg_features=ppg_features,
                ecg_reference_features=reference_features,
                ecg_reference_feature_statuses=reference_statuses,
                feature_statuses=feature_statuses,
                feature_exclusion_reasons=feature_reasons,
                common_observed_seconds=common_observed,
                exclusion_reason=pair_reason,
                ecg_spectral_corrected_interval_count=int(
                    ecg_diagnostics["spectral_corrected_interval_count"]
                ),
                ppg_spectral_corrected_interval_count=int(
                    ppg_diagnostics["spectral_corrected_interval_count"]
                ),
                ecg_spectral_correction_fraction=ecg_diagnostics[
                    "spectral_correction_fraction"
                ],
                ppg_spectral_correction_fraction=ppg_diagnostics[
                    "spectral_correction_fraction"
                ],
                ecg_spectral_support_seconds=ecg_diagnostics[
                    "spectral_support_seconds"
                ],
                ppg_spectral_support_seconds=ppg_diagnostics[
                    "spectral_support_seconds"
                ],
                tachogram_interpolation_method=(
                    "bounded_pchip"
                    if np.isfinite(
                        [
                            ecg_features["lf_power_ms2"],
                            ppg_features["lf_power_ms2"],
                        ]
                    ).all()
                    else "none"
                ),
            ),
            [*ecg_intervals, *ppg_intervals],
        )

    def pair_columns(self, pair: HrvFeaturePair) -> dict[str, Any]:
        result: dict[str, Any] = {
            "hrv_feature_status": pair.status,
            "hrv_feature_exclusion_reason": pair.exclusion_reason or "",
            "hrv_common_observed_seconds": pair.common_observed_seconds,
            "hrv_method_version": pair.method_version,
            "hrv_analysis_duration_label": pair.analysis_duration_label,
            "hrv_pnn50_unit": pair.pnn50_unit,
            "ecg_spectral_corrected_interval_count": (
                pair.ecg_spectral_corrected_interval_count
            ),
            "ppg_spectral_corrected_interval_count": (
                pair.ppg_spectral_corrected_interval_count
            ),
            "ecg_spectral_correction_fraction": pair.ecg_spectral_correction_fraction,
            "ppg_spectral_correction_fraction": pair.ppg_spectral_correction_fraction,
            "ecg_spectral_support_seconds": pair.ecg_spectral_support_seconds,
            "ppg_spectral_support_seconds": pair.ppg_spectral_support_seconds,
            "hrv_tachogram_interpolation_method": pair.tachogram_interpolation_method,
        }
        for feature in FEATURES:
            result[f"ecg_hrv_{feature}"] = pair.ecg_features.get(feature, np.nan)
            result[f"ppg_hrv_{feature}"] = pair.ppg_features.get(feature, np.nan)
            result[f"ecg_reference_hrv_{feature}"] = pair.ecg_reference_features.get(
                feature, np.nan
            )
            result[f"ecg_reference_hrv_{feature}_status"] = (
                pair.ecg_reference_feature_statuses.get(
                    feature, "INSUFFICIENT_EVIDENCE"
                )
            )
            result[f"hrv_{feature}_status"] = pair.feature_statuses.get(
                feature, "INSUFFICIENT_EVIDENCE"
            )
            result[f"hrv_{feature}_exclusion_reason"] = (
                pair.feature_exclusion_reasons.get(feature, pair.exclusion_reason or "")
            )
        return result

    def evaluate(
        self,
        feature_pairs: pd.DataFrame,
        *,
        evaluation_scope: str = "selection",
        full_cohort_access_authorized: bool = False,
        bootstrap_iterations: int = 0,
        bootstrap_seed: int = 42,
        development_ranges: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Return selection or frozen full-cohort agreement evidence."""

        if evaluation_scope not in {"selection", "full_cohort"}:
            raise ValueError(f"invalid_hrv_evaluation_scope:{evaluation_scope}")
        observed_participants = set(feature_pairs["participant_id"].astype(str))
        if not observed_participants <= PHYSIOLOGICAL_VALIDATION_PARTICIPANTS:
            raise ValueError("unexpected_physiological_validation_participant")
        if evaluation_scope == "selection":
            if not observed_participants <= SELECTION_PARTICIPANTS:
                raise ValueError("selection_scope_accessed_internal_test_participant")
            required_pooled = math.ceil(
                self.config.minimum_full_cohort_coverage_fraction
                * len(SELECTION_PARTICIPANTS)
                * 2
            )
            required_phase = math.ceil(
                self.config.minimum_full_cohort_coverage_fraction
                * len(SELECTION_PARTICIPANTS)
            )
            evaluation_frame = feature_pairs
        else:
            if not full_cohort_access_authorized:
                raise RuntimeError("physiological_full_cohort_access_not_authorized")
            required_pooled = math.ceil(
                self.config.minimum_full_cohort_coverage_fraction
                * self.config.validation_participant_count
                * 2
            )
            required_phase = math.ceil(
                self.config.minimum_full_cohort_coverage_fraction
                * self.config.validation_participant_count
            )
            evaluation_frame = feature_pairs

        decisions: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        frozen_ranges = (
            dict(development_ranges)
            if development_ranges is not None
            else _ecg_reference_ranges(feature_pairs)
        )
        range_payload = {
            feature: value if np.isfinite(value) else None
            for feature, value in frozen_ranges.items()
        }
        range_fingerprint = hashlib.sha256(
            json.dumps(range_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        ranges_by_channel: dict[str, dict[str, float]] = {}
        channels = sorted(feature_pairs["sensor_channel"].dropna().unique())
        for channel in channels:
            channel_frame = evaluation_frame[
                evaluation_frame["sensor_channel"].eq(channel)
            ]
            ranges = dict(frozen_ranges)
            ranges_by_channel[channel] = ranges
            channel_metrics = _agreement_metrics(
                channel_frame,
                channel,
                ranges,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
            metric_rows.extend(channel_metrics)
            decision = _channel_decision(
                channel,
                channel_metrics,
                self.config,
                evaluation_scope=evaluation_scope,
                required_pooled=required_pooled,
                required_phase=required_phase,
            )
            decisions.append(asdict(decision))

        eligible = [row for row in decisions if row["decision"] == "GO"]
        eligible.sort(
            key=lambda row: (
                -float(row["minimum_composite"]),
                -float(row["mean_composite"]),
                str(row["sensor_channel"]),
            )
        )
        final_values = {row["decision"] for row in decisions}
        overall = (
            "GO"
            if eligible
            else (
                "INSUFFICIENT_EVIDENCE"
                if final_values and final_values == {"INSUFFICIENT_EVIDENCE"}
                else "NO_GO"
            )
        )
        payload = {
            "schema_version": "hrv-prv-agreement-v4",
            "primary_domain": self.config.primary_domain,
            "primary_gate": "hardened_legacy_composite_time_domain",
            "beat_matching_controls_primary_gate": False,
            "feature_denominator_policy": "per_feature_by_domain",
            "analysis_duration_label": "short_term_240s",
            "pnn50_unit": "fraction",
            "features": list(FEATURES),
            "primary_features": list(TIME_DOMAIN_FEATURES),
            "time_domain_features": list(TIME_DOMAIN_FEATURES),
            "diagnostic_spectral_features": list(SPECTRAL_FEATURES),
            "spectral_validation_role": "separate_nonblocking_diagnostic",
            "formula": (
                "mean(pearson_r,spearman_rho,icc_a1,"
                "1-clip(mae/ecg_development_range,0,1))"
            ),
            "thresholds": {
                "minimum_full_cohort_coverage_fraction": (
                    self.config.minimum_full_cohort_coverage_fraction
                ),
                "pooled_required": required_pooled,
                "phase_required": required_phase,
                "pooled_composite": self.config.pooled_composite_threshold,
                "phase_composite": self.config.phase_composite_threshold,
                "spectral_max_artifact_fraction": (
                    self.config.spectral_max_artifact_fraction
                ),
                "spectral_max_consecutive_artifacts": (
                    self.config.spectral_max_consecutive_artifacts
                ),
                "spectral_max_interpolation_span_seconds": (
                    self.config.spectral_max_interpolation_span_seconds
                ),
            },
            "evaluation_scope": evaluation_scope,
            "selection_participants": sorted(SELECTION_PARTICIPANTS),
            "physiological_validation_participants": sorted(
                PHYSIOLOGICAL_VALIDATION_PARTICIPANTS
            ),
            "model_internal_test_participants": sorted(
                MODEL_INTERNAL_TEST_PARTICIPANTS
            ),
            "validation_participants": sorted(observed_participants),
            "full_cohort_access_authorized": full_cohort_access_authorized,
            "coverage_policy": {
                "fraction": self.config.minimum_full_cohort_coverage_fraction,
                "pooled_required": required_pooled,
                "phase_required": required_phase,
            },
            "development_ranges": ranges_by_channel,
            "ecg_reference_development_ranges": frozen_ranges,
            "ecg_reference_range_fingerprint": range_fingerprint,
            "channels": {row["sensor_channel"]: row for row in decisions},
            "eligible_channels": [row["sensor_channel"] for row in eligible],
            "recommended_channel": (
                eligible[0]["sensor_channel"] if eligible else None
            ),
            "hrv_validation_decision": overall,
            "global_policy": "any_channel_go",
        }
        return payload, pd.DataFrame(metric_rows)


def build_hrv_training_admission(
    payload: Mapping[str, Any],
    config: HrvAgreementConfig,
    *,
    run_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate model training from development-only HRV--PRV evidence."""

    if payload.get("evaluation_scope") != "selection":
        raise RuntimeError("hrv_training_admission_requires_selection_scope")
    if payload.get("primary_features") != list(TIME_DOMAIN_FEATURES):
        raise RuntimeError("hrv_training_admission_requires_time_domain_features")

    required_pooled = math.ceil(
        config.minimum_full_cohort_coverage_fraction * len(SELECTION_PARTICIPANTS) * 2
    )
    required_phase = math.ceil(
        config.minimum_full_cohort_coverage_fraction * len(SELECTION_PARTICIPANTS)
    )
    channels: dict[str, dict[str, Any]] = {}
    for channel in TRAINING_CHANNELS:
        row = payload.get("channels", {}).get(channel)
        if not isinstance(row, Mapping):
            channels[channel] = {
                "decision": "INSUFFICIENT_EVIDENCE",
                "status": "blocked_hrv_prv_insufficient_evidence",
                "eligible": False,
                "reason": "channel_evidence_missing",
            }
            continue
        decision = str(row.get("time_domain_decision", row.get("decision", "")))
        pooled = int(row.get("valid_time_domain_pooled_rows", 0))
        relax = int(row.get("valid_time_domain_relax_rows", 0))
        task = int(row.get("valid_time_domain_task_rows", 0))
        quorum = (
            pooled >= required_pooled
            and relax >= required_phase
            and task >= required_phase
        )
        eligible = decision == "GO" and quorum
        if eligible:
            status = "ready"
            reason = None
        elif decision == "NO_GO":
            status = "blocked_hrv_prv_no_go"
            reason = "time_domain_agreement_below_threshold"
        elif not quorum:
            status = "blocked_hrv_prv_insufficient_evidence"
            reason = "time_domain_coverage_below_quorum"
        else:
            status = "blocked_hrv_prv_insufficient_evidence"
            reason = "time_domain_decision_unavailable"
        channels[channel] = {
            "decision": decision or "INSUFFICIENT_EVIDENCE",
            "status": status,
            "eligible": eligible,
            "reason": reason,
            "valid_pooled": pooled,
            "valid_relax": relax,
            "valid_task": task,
            "required_pooled": required_pooled,
            "required_phase": required_phase,
        }

    eligible_channels = [
        channel for channel, row in channels.items() if row["eligible"]
    ]
    result = {
        "schema_version": "hrv-training-admission-v1",
        "evaluation_scope": "selection",
        "selection_participants": sorted(SELECTION_PARTICIPANTS),
        "primary_domain": "time_domain",
        "primary_features": list(TIME_DOMAIN_FEATURES),
        "coverage_policy": {
            "fraction": config.minimum_full_cohort_coverage_fraction,
            "pooled_required": required_pooled,
            "phase_required": required_phase,
        },
        "channels": channels,
        "eligible_channels": eligible_channels,
        "status": "ready" if eligible_channels else "blocked",
        "run_id": payload.get("run_id"),
    }
    if run_identity:
        result.update(
            {
                key: run_identity.get(key)
                for key in (
                    "run_id",
                    "source_tree_sha256",
                    "run_contract_sha256",
                    "config_fingerprint",
                )
                if run_identity.get(key) is not None
            }
        )
    return result


def _peak_timestamps(detected: DetectedPeaks) -> np.ndarray:
    timestamps = np.asarray(detected.timestamps_ms, dtype=float)
    if timestamps.size < 2 or not np.isfinite(timestamps).all():
        raise ValueError("insufficient_finite_peak_timestamps")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("non_monotonic_peak_timestamps")
    return timestamps


def _common_observed_seconds(
    window_start_ms: float,
    window_end_ms: float,
    gaps: Sequence[GapEvent],
) -> float:
    if not np.isfinite([window_start_ms, window_end_ms]).all():
        return 0.0
    if window_end_ms <= window_start_ms:
        return 0.0
    clipped = sorted(
        (
            max(window_start_ms, float(gap.start_ms)),
            min(window_end_ms, float(gap.end_ms)),
        )
        for gap in gaps
        if gap.affects_peak_region
        and gap.end_ms > window_start_ms
        and gap.start_ms < window_end_ms
    )
    merged: list[list[float]] = []
    for start, end in clipped:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gap_ms = sum(end - start for start, end in merged)
    return max(0.0, (window_end_ms - window_start_ms - gap_ms) / 1000.0)


def _crosses_gap(start_ms: float, end_ms: float, gaps: Sequence[GapEvent]) -> bool:
    return any(
        gap.affects_peak_region and gap.start_ms < end_ms and gap.end_ms > start_ms
        for gap in gaps
    )


def _true_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return [run for run in np.split(indices, boundaries) if run.size]


def _observed_intervals(
    participant_id: str,
    phase: str,
    sensor_channel: str,
    signal_role: str,
    peak_timestamps_ms: np.ndarray,
    gaps: Sequence[GapEvent],
    config: HrvAgreementConfig,
    lineage: Mapping[str, str] | None = None,
) -> list[ObservedIntervalBlock]:
    intervals = np.diff(peak_timestamps_ms)
    physiological = (intervals >= config.interval_min_ms) & (
        intervals <= config.interval_max_ms
    )
    gap_crossed = np.asarray(
        [
            _crosses_gap(float(start), float(end), gaps)
            for start, end in zip(
                peak_timestamps_ms[:-1], peak_timestamps_ms[1:], strict=True
            )
        ],
        dtype=bool,
    )
    initial = physiological & ~gap_crossed
    moving_valid = np.zeros(intervals.size, dtype=bool)
    half = config.moving_median_window // 2
    for run in _true_runs(initial):
        for offset, index in enumerate(run):
            low = max(0, offset - half)
            high = min(run.size, offset + half + 1)
            median = float(np.median(intervals[run[low:high]]))
            moving_valid[index] = bool(
                median > 0.0
                and abs(float(intervals[index]) - median) / median
                <= config.moving_median_relative_deviation
            )
    stage_two = initial & moving_valid
    mad_valid = np.zeros(intervals.size, dtype=bool)
    stage_values = intervals[stage_two]
    if stage_values.size:
        median = float(np.median(stage_values))
        robust_mad = float(np.median(np.abs(stage_values - median)) * 1.4826)
        if robust_mad <= np.finfo(float).eps:
            mad_valid[stage_two] = True
        else:
            mad_valid[stage_two] = (
                np.abs(stage_values - median) <= config.mad_multiplier * robust_mad
            )
    eligible = stage_two & mad_valid
    block_by_index: dict[int, int] = {}
    for block_id, run in enumerate(_true_runs(eligible)):
        block_by_index.update({int(index): block_id for index in run})
    support_by_index: dict[int, int] = {}
    for support_id, run in enumerate(_true_runs(~gap_crossed)):
        support_by_index.update({int(index): support_id for index in run})

    rows: list[ObservedIntervalBlock] = []
    metadata = dict(lineage or {})
    for index, interval in enumerate(intervals):
        reason = None
        if not physiological[index]:
            reason = "nonphysiological_interval"
        elif gap_crossed[index]:
            reason = "interval_crosses_gap"
        elif not moving_valid[index]:
            reason = "moving_median_outlier"
        elif not mad_valid[index]:
            reason = "global_mad_outlier"
        rows.append(
            ObservedIntervalBlock(
                participant_id=participant_id,
                phase=phase,
                sensor_channel=sensor_channel,
                signal_role=signal_role,  # type: ignore[arg-type]
                interval_index=index,
                block_id=block_by_index.get(index),
                start_peak_timestamp_ms=float(peak_timestamps_ms[index]),
                end_peak_timestamp_ms=float(peak_timestamps_ms[index + 1]),
                interval_ms=float(interval),
                physiological=bool(physiological[index]),
                gap_crossed=bool(gap_crossed[index]),
                moving_median_valid=bool(moving_valid[index]),
                mad_valid=bool(mad_valid[index]),
                eligible=bool(eligible[index]),
                exclusion_reason=reason,
                spectral_support_id=support_by_index.get(index),
                run_id=str(metadata.get("run_id", "")),
                recording_id=str(metadata.get("recording_id", "")),
                source_file=str(metadata.get(f"{signal_role}_source_file", "")),
                source_checksum=str(metadata.get(f"{signal_role}_source_checksum", "")),
                detector_method=str(
                    metadata.get(f"{signal_role}_detection_method", "")
                ),
                detector_version=str(
                    metadata.get(f"{signal_role}_detector_version", "")
                ),
                config_fingerprint=str(metadata.get("config_fingerprint", "")),
                run_contract_sha256=str(metadata.get("run_contract_sha256", "")),
            )
        )
    return rows


def _hrv_features(
    intervals: Sequence[ObservedIntervalBlock], config: HrvAgreementConfig
) -> dict[str, float]:
    """Return all six features or fail for callers that require complete cases."""

    features, reasons, _ = _hrv_feature_result(intervals, config)
    missing = [feature for feature in FEATURES if not np.isfinite(features[feature])]
    if missing:
        first = missing[0]
        raise ValueError(reasons.get(first, f"insufficient_feature:{first}"))
    return features


def _hrv_feature_result(
    intervals: Sequence[ObservedIntervalBlock], config: HrvAgreementConfig
) -> tuple[dict[str, float], dict[str, str], dict[str, float | int | None]]:
    """Compute each domain independently so spectral failure keeps time evidence."""

    features = {feature: float("nan") for feature in FEATURES}
    reasons = {feature: "" for feature in FEATURES}
    diagnostics: dict[str, float | int | None] = {
        "spectral_corrected_interval_count": 0,
        "spectral_correction_fraction": None,
        "spectral_support_seconds": None,
    }
    eligible = [row for row in intervals if row.eligible]
    if len(eligible) < config.minimum_intervals:
        for feature in FEATURES:
            reasons[feature] = "insufficient_hrv_intervals"
        return features, reasons, diagnostics

    blocks: dict[int, list[ObservedIntervalBlock]] = {}
    for row in eligible:
        assert row.block_id is not None
        blocks.setdefault(row.block_id, []).append(row)
    values = np.asarray([row.interval_ms for row in eligible], dtype=float)
    difference_values = _successive_differences(eligible)
    if difference_values.size < config.minimum_successive_pairs:
        for feature in TIME_DOMAIN_FEATURES:
            reasons[feature] = "insufficient_successive_hrv_pairs"
    else:
        time_features = {
            "rmssd_ms": float(np.sqrt(np.mean(np.square(difference_values)))),
            "sdnn_ms": float(np.std(values, ddof=1)),
            "pnn50": float(np.mean(np.abs(difference_values) > 50.0)),
        }
        if np.isfinite(list(time_features.values())).all():
            features.update(time_features)
        else:
            for feature in TIME_DOMAIN_FEATURES:
                reasons[feature] = "nonfinite_time_domain_feature"

    try:
        lf, hf, diagnostics = _spectral_features_from_intervals(intervals, config)
        if hf <= 0.0 or not np.isfinite([lf, hf]).all():
            raise ValueError("invalid_hrv_spectral_power")
        features.update(
            {
                "lf_power_ms2": float(lf),
                "hf_power_ms2": float(hf),
                "lf_hf_ratio": float(lf / hf),
            }
        )
    except ValueError as exc:
        for feature in SPECTRAL_FEATURES:
            reasons[feature] = str(exc)
    return features, reasons, diagnostics


def _successive_differences(
    intervals: Sequence[ObservedIntervalBlock],
) -> np.ndarray:
    """Difference only originally adjacent intervals in the same quality block."""

    differences = [
        float(current.interval_ms - previous.interval_ms)
        for previous, current in zip(intervals[:-1], intervals[1:], strict=True)
        if previous.block_id is not None
        and previous.block_id == current.block_id
        and current.interval_index == previous.interval_index + 1
    ]
    return np.asarray(differences, dtype=float)


def _spectral_features(
    blocks: Mapping[int, Sequence[ObservedIntervalBlock]],
    config: HrvAgreementConfig,
) -> tuple[float, float]:
    if not blocks:
        raise ValueError("missing_hrv_interval_blocks")
    longest = max(
        blocks.values(),
        key=lambda rows: (
            rows[-1].end_peak_timestamp_ms - rows[0].start_peak_timestamp_ms
        ),
    )
    duration_seconds = (
        longest[-1].end_peak_timestamp_ms - longest[0].start_peak_timestamp_ms
    ) / 1000.0
    if duration_seconds < config.spectral_min_block_seconds:
        raise ValueError("insufficient_contiguous_spectral_support")
    values = np.asarray([row.interval_ms for row in longest], dtype=float)
    times = np.asarray(
        [
            (row.start_peak_timestamp_ms + row.end_peak_timestamp_ms) / 2000.0
            for row in longest
        ],
        dtype=float,
    )
    return _tachogram_band_power(values, times, config)


def _spectral_features_from_intervals(
    intervals: Sequence[ObservedIntervalBlock],
    config: HrvAgreementConfig,
) -> tuple[float, float, dict[str, float | int | None]]:
    """Compute spectrum on structural support with bounded artifact correction."""

    supports: dict[int, list[ObservedIntervalBlock]] = {}
    for row in intervals:
        if row.spectral_support_id is not None:
            supports.setdefault(row.spectral_support_id, []).append(row)
    if not supports:
        raise ValueError("missing_hrv_structural_support")

    failure = "insufficient_contiguous_spectral_support"
    ordered = sorted(
        supports.values(),
        key=lambda rows: (
            rows[-1].end_peak_timestamp_ms - rows[0].start_peak_timestamp_ms
        ),
        reverse=True,
    )
    for rows in ordered:
        eligible_positions = np.flatnonzero([row.eligible for row in rows])
        if eligible_positions.size < 4:
            failure = "insufficient_spectral_knots"
            continue
        interior = rows[int(eligible_positions[0]) : int(eligible_positions[-1]) + 1]
        support_seconds = (
            interior[-1].end_peak_timestamp_ms - interior[0].start_peak_timestamp_ms
        ) / 1000.0
        if support_seconds < config.spectral_min_block_seconds:
            failure = "insufficient_contiguous_spectral_support"
            continue

        artifact_mask = np.asarray([not row.eligible for row in interior], dtype=bool)
        corrected_count = int(artifact_mask.sum())
        correction_fraction = corrected_count / len(interior)
        if correction_fraction > config.spectral_max_artifact_fraction:
            failure = "spectral_artifact_fraction_above_limit"
            continue
        artifact_runs = _true_runs(artifact_mask)
        if any(
            len(run) > config.spectral_max_consecutive_artifacts
            for run in artifact_runs
        ):
            failure = "spectral_consecutive_artifacts_above_limit"
            continue
        if any(
            sum(interior[int(index)].interval_ms for index in run) / 1000.0
            > config.spectral_max_interpolation_span_seconds
            for run in artifact_runs
        ):
            failure = "spectral_interpolation_span_above_limit"
            continue

        eligible = [row for row in interior if row.eligible]
        values = np.asarray([row.interval_ms for row in eligible], dtype=float)
        times = np.asarray(
            [
                (row.start_peak_timestamp_ms + row.end_peak_timestamp_ms) / 2000.0
                for row in eligible
            ],
            dtype=float,
        )
        lf, hf = _tachogram_band_power(values, times, config)
        return (
            lf,
            hf,
            {
                "spectral_corrected_interval_count": corrected_count,
                "spectral_correction_fraction": float(correction_fraction),
                "spectral_support_seconds": float(support_seconds),
            },
        )
    raise ValueError(failure)


def _tachogram_band_power(
    values: np.ndarray,
    times: np.ndarray,
    config: HrvAgreementConfig,
) -> tuple[float, float]:
    times = np.asarray(times, dtype=float) - float(times[0])
    if values.size < 4 or np.any(np.diff(times) <= 0.0):
        raise ValueError("invalid_spectral_tachogram")
    grid = np.arange(times[0], times[-1], 1.0 / config.tachogram_rate_hz)
    if grid.size < 256:
        raise ValueError("insufficient_spectral_grid")
    tachogram = PchipInterpolator(times, values, extrapolate=False)(grid)
    if not np.isfinite(tachogram).all():
        raise ValueError("nonfinite_spectral_tachogram")
    frequency, power = welch(
        tachogram - np.mean(tachogram),
        fs=config.tachogram_rate_hz,
        window="hann",
        nperseg=min(256, tachogram.size),
        detrend="constant",
    )

    def band_power(low: float, high: float) -> float:
        mask = (frequency >= low) & (frequency < high)
        if np.count_nonzero(mask) < 2:
            raise ValueError("insufficient_frequency_bins")
        return float(np.trapezoid(power[mask], frequency[mask]))

    return (
        band_power(config.lf_low_hz, config.lf_high_hz),
        band_power(config.lf_high_hz, config.hf_high_hz),
    )


def _valid_feature_rows(
    frame: pd.DataFrame, feature: str | None = None
) -> pd.DataFrame:
    selected_features = (feature,) if feature is not None else FEATURES
    required = [
        *(f"ecg_hrv_{name}" for name in selected_features),
        *(f"ppg_hrv_{name}" for name in selected_features),
    ]
    if frame.empty or any(column not in frame for column in required):
        return frame.iloc[0:0].copy()
    if feature is not None and f"hrv_{feature}_status" in frame:
        status = frame[f"hrv_{feature}_status"].eq("COMPUTED")
    else:
        status = frame["hrv_feature_status"].eq("COMPUTED")
    valid = frame[status].copy()
    for column in required:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    return valid[np.isfinite(valid[required]).all(axis=1)].copy()


def _finite_range(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    return float(np.ptp(array)) if array.size else float("nan")


def _ecg_reference_ranges(frame: pd.DataFrame) -> dict[str, float]:
    """Freeze one ECG-only development range shared by all PPG channels."""

    development = frame[frame["participant_id"].isin(DEVELOPMENT_PARTICIPANTS)]
    ranges: dict[str, float] = {}
    for feature in FEATURES:
        value_column = f"ecg_reference_hrv_{feature}"
        status_column = f"ecg_reference_hrv_{feature}_status"
        if value_column not in development or status_column not in development:
            raise RuntimeError(f"missing_ecg_reference_feature:{feature}")
        selected = development[development[status_column].eq("COMPUTED")][
            ["participant_id", "phase", value_column]
        ].copy()
        selected[value_column] = pd.to_numeric(selected[value_column], errors="coerce")
        selected = selected[np.isfinite(selected[value_column])]
        grouped = selected.groupby(["participant_id", "phase"])[value_column]
        inconsistent = (grouped.max() - grouped.min()).abs() > 1e-9
        if inconsistent.any():
            raise RuntimeError(f"inconsistent_ecg_reference_feature:{feature}")
        ranges[feature] = _finite_range(grouped.first())
    return ranges


def _scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope in {"relax", "task"}:
        return frame[frame["phase"].eq(scope)].copy()
    # The locked pooled coverage gate is expressed in sessions (46 of 56),
    # not complete participant pairs.  Phase-specific gates still prevent a
    # high pooled correlation from hiding missing relax or task evidence, and
    # bootstrap resampling remains clustered by participant.
    return frame.copy()


def _icc_a1(left: np.ndarray, right: np.ndarray) -> float:
    ratings = np.column_stack((left, right)).astype(float)
    n, k = ratings.shape
    if n < 3 or k != 2 or not np.isfinite(ratings).all():
        return float("nan")
    grand = float(np.mean(ratings))
    row_means = np.mean(ratings, axis=1)
    column_means = np.mean(ratings, axis=0)
    ms_rows = k * float(np.sum(np.square(row_means - grand))) / (n - 1)
    ms_columns = n * float(np.sum(np.square(column_means - grand))) / (k - 1)
    residual = ratings - row_means[:, None] - column_means[None, :] + grand
    ms_error = float(np.sum(np.square(residual))) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    scale = max(abs(ms_rows), abs(ms_columns), abs(ms_error))
    if scale == 0.0 or abs(denominator) <= 64 * np.finfo(float).eps * scale:
        return float("nan")
    return float((ms_rows - ms_error) / denominator)


def _ccc(left: np.ndarray, right: np.ndarray) -> float:
    covariance = float(np.mean((left - np.mean(left)) * (right - np.mean(right))))
    denominator = float(
        np.var(left) + np.var(right) + np.square(np.mean(left) - np.mean(right))
    )
    return float(2.0 * covariance / denominator) if denominator > 0.0 else float("nan")


def _bland_altman(
    frame: pd.DataFrame, left: np.ndarray, right: np.ndarray, scope: str
) -> dict[str, Any]:
    difference = right - left
    if scope == "pooled":
        repeated = frame.assign(_difference=difference).pivot(
            index="participant_id", columns="phase", values="_difference"
        )
        required_phases = {"relax", "task"}
        if not required_phases.issubset(repeated.columns):
            return _unavailable_bland_altman(int(repeated.index.nunique()))
        repeated = repeated.dropna(subset=["relax", "task"])
        if len(repeated) < 3:
            return _unavailable_bland_altman(len(repeated))
        matrix = repeated[["relax", "task"]].to_numpy(dtype=float)
        participant_bias = matrix.mean(axis=1)
        bias = float(participant_bias.mean())
        participant_count, phase_count = matrix.shape
        ms_between = float(
            phase_count
            * np.square(participant_bias - bias).sum()
            / (participant_count - 1)
        )
        ms_within = float(
            np.square(matrix - participant_bias[:, None]).sum()
            / (participant_count * (phase_count - 1))
        )
        between_variance = max((ms_between - ms_within) / phase_count, 0.0)
        standard_deviation = float(np.sqrt(between_variance + ms_within))
        method_mean = (
            frame.assign(_mean=(left + right) / 2.0)
            .pivot(index="participant_id", columns="phase", values="_mean")
            .loc[repeated.index, ["relax", "task"]]
            .mean(axis=1)
            .to_numpy(dtype=float)
        )
        regression_values = participant_bias
    else:
        bias = float(np.mean(difference))
        standard_deviation = float(np.std(difference, ddof=1))
        method_mean = (left + right) / 2.0
        regression_values = difference
        participant_count = int(frame["participant_id"].nunique())
    if np.ptp(method_mean) > 0.0:
        regression = linregress(method_mean, regression_values)
        slope = float(regression.slope)
        p_value = float(regression.pvalue)
    else:
        slope = float("nan")
        p_value = float("nan")
    return {
        "bland_altman_status": "COMPUTED",
        "bias": bias,
        "loa_low": float(bias - 1.96 * standard_deviation),
        "loa_high": float(bias + 1.96 * standard_deviation),
        "proportional_bias_slope": slope,
        "proportional_bias_p": p_value,
        "repeated_measure_participant_count": participant_count,
    }


def _unavailable_bland_altman(participant_count: int) -> dict[str, Any]:
    return {
        "bland_altman_status": "INSUFFICIENT_EVIDENCE",
        "bias": float("nan"),
        "loa_low": float("nan"),
        "loa_high": float("nan"),
        "proportional_bias_slope": float("nan"),
        "proportional_bias_p": float("nan"),
        "repeated_measure_participant_count": participant_count,
    }


def _metric_components(
    frame: pd.DataFrame, feature: str, ecg_range: float, scope: str
) -> dict[str, float]:
    left = frame[f"ecg_hrv_{feature}"].to_numpy(dtype=float)
    right = frame[f"ppg_hrv_{feature}"].to_numpy(dtype=float)
    if (
        left.size < 3
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.ptp(left) <= 0.0
        or np.ptp(right) <= 0.0
        or not np.isfinite(ecg_range)
        or ecg_range <= 0.0
    ):
        raise ValueError("insufficient_agreement_variation")
    pearson = float(pearsonr(left, right).statistic)
    spearman = float(spearmanr(left, right).statistic)
    icc = _icc_a1(left, right)
    difference = right - left
    mae = float(np.mean(np.abs(difference)))
    accuracy = float(1.0 - np.clip(mae / ecg_range, 0.0, 1.0))
    components = np.asarray([pearson, spearman, icc, accuracy], dtype=float)
    if not np.isfinite(components).all():
        raise ValueError("nonfinite_legacy_composite_component")
    bland_altman = _bland_altman(frame, left, right, scope)
    return {
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "icc_a1": icc,
        "mae": mae,
        "ecg_development_range": ecg_range,
        "accuracy_component": accuracy,
        "composite_score": float(np.mean(components)),
        "ccc": _ccc(left, right),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        **bland_altman,
    }


def _bootstrap_composite(
    frame: pd.DataFrame,
    feature: str,
    ecg_range: float,
    *,
    scope: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, int]:
    participants = sorted(frame["participant_id"].unique())
    if iterations <= 0 or len(participants) < 3:
        return float("nan"), float("nan"), 0
    grouped = {
        participant: frame[frame["participant_id"].eq(participant)]
        for participant in participants
    }
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    failed_iterations = 0
    for _ in range(iterations):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        sampled_groups: list[pd.DataFrame] = []
        for draw_index, participant in enumerate(sampled):
            group = grouped[str(participant)].copy()
            # Duplicate participant draws are independent bootstrap clusters,
            # while both phases from one draw remain paired.
            group["participant_id"] = f"BOOT-{draw_index:04d}"
            sampled_groups.append(group)
        sample = pd.concat(sampled_groups, ignore_index=True)
        try:
            scores.append(
                _metric_components(sample, feature, ecg_range, scope)["composite_score"]
            )
        except ValueError:
            failed_iterations += 1
    if len(scores) < max(10, iterations // 2):
        return float("nan"), float("nan"), failed_iterations
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high), failed_iterations


def _agreement_metrics(
    frame: pd.DataFrame,
    channel: str,
    ranges: Mapping[str, float],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        for feature in FEATURES:
            selected = _scope_frame(_valid_feature_rows(frame, feature), scope)
            try:
                components = _metric_components(
                    selected, feature, ranges[feature], scope
                )
                metric = HrvAgreementMetric(
                    sensor_channel=channel,
                    feature=feature,
                    scope=scope,  # type: ignore[arg-type]
                    sample_count=len(selected),
                    status="COMPUTED",
                    **{
                        key: components[key]
                        for key in HrvAgreementMetric.__dataclass_fields__
                        if key in components
                    },
                )
                row = asdict(metric)
                row["point_estimate_participant_count"] = int(
                    selected["participant_id"].nunique()
                )
                low, high, failed_iterations = _bootstrap_composite(
                    selected,
                    feature,
                    ranges[feature],
                    scope=scope,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
                row["composite_ci_low"] = low
                row["composite_ci_high"] = high
                row["bootstrap_failed_iterations"] = failed_iterations
            except ValueError as exc:
                row = asdict(
                    HrvAgreementMetric(
                        sensor_channel=channel,
                        feature=feature,
                        scope=scope,  # type: ignore[arg-type]
                        sample_count=len(selected),
                        pearson_r=None,
                        spearman_rho=None,
                        icc_a1=None,
                        mae=None,
                        ecg_development_range=(
                            float(ranges[feature])
                            if np.isfinite(ranges[feature])
                            else None
                        ),
                        accuracy_component=None,
                        composite_score=None,
                        ccc=None,
                        bias=None,
                        rmse=None,
                        loa_low=None,
                        loa_high=None,
                        status="INSUFFICIENT_EVIDENCE",
                        reason=str(exc),
                    )
                )
                row["composite_ci_low"] = np.nan
                row["composite_ci_high"] = np.nan
                row["bootstrap_failed_iterations"] = bootstrap_iterations
                row["point_estimate_participant_count"] = int(
                    selected["participant_id"].nunique()
                )
            rows.append(row)
    return rows


def _channel_decision(
    channel: str,
    metric_rows: Sequence[Mapping[str, Any]],
    config: HrvAgreementConfig,
    *,
    evaluation_scope: str,
    required_pooled: int,
    required_phase: int,
) -> HrvValidationDecision:
    time_domain = _domain_decision(
        metric_rows,
        TIME_DOMAIN_FEATURES,
        config,
        required_pooled=required_pooled,
        required_phase=required_phase,
    )
    spectral = _domain_decision(
        metric_rows,
        SPECTRAL_FEATURES,
        config,
        required_pooled=required_pooled,
        required_phase=required_phase,
    )
    reasons = [
        *(f"time_domain:{reason}" for reason in time_domain.reasons),
        *(f"spectral:{reason}" for reason in spectral.reasons),
    ]
    return HrvValidationDecision(
        sensor_channel=channel,
        evaluation_scope=evaluation_scope,  # type: ignore[arg-type]
        decision=time_domain.decision,  # type: ignore[arg-type]
        valid_pooled_rows=time_domain.pooled_count,
        valid_relax_rows=time_domain.relax_count,
        valid_task_rows=time_domain.task_count,
        minimum_composite=time_domain.minimum_composite,
        mean_composite=time_domain.mean_composite,
        time_domain_decision=time_domain.decision,  # type: ignore[arg-type]
        valid_time_domain_pooled_rows=time_domain.pooled_count,
        valid_time_domain_relax_rows=time_domain.relax_count,
        valid_time_domain_task_rows=time_domain.task_count,
        time_domain_minimum_composite=time_domain.minimum_composite,
        time_domain_mean_composite=time_domain.mean_composite,
        spectral_validation_status=spectral.decision,  # type: ignore[arg-type]
        valid_spectral_pooled_rows=spectral.pooled_count,
        valid_spectral_relax_rows=spectral.relax_count,
        valid_spectral_task_rows=spectral.task_count,
        spectral_minimum_composite=spectral.minimum_composite,
        spectral_mean_composite=spectral.mean_composite,
        reasons=tuple(dict.fromkeys(reasons)),
        time_domain_reasons=time_domain.reasons,
        spectral_reasons=spectral.reasons,
    )


def _domain_decision(
    metric_rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    config: HrvAgreementConfig,
    *,
    required_pooled: int,
    required_phase: int,
) -> _DomainDecision:
    metric_lookup = {
        (str(row["feature"]), str(row["scope"])): row for row in metric_rows
    }
    pooled_counts = {
        feature: int(metric_lookup.get((feature, "pooled"), {}).get("sample_count", 0))
        for feature in features
    }
    relax_counts = {
        feature: int(metric_lookup.get((feature, "relax"), {}).get("sample_count", 0))
        for feature in features
    }
    task_counts = {
        feature: int(metric_lookup.get((feature, "task"), {}).get("sample_count", 0))
        for feature in features
    }
    pooled_count = min(pooled_counts.values(), default=0)
    relax_count = min(relax_counts.values(), default=0)
    task_count = min(task_counts.values(), default=0)
    reasons: list[str] = []
    for feature in features:
        if pooled_counts[feature] < required_pooled:
            reasons.append(f"pooled_coverage_below_{required_pooled}:{feature}")
        if relax_counts[feature] < required_phase:
            reasons.append(f"relax_coverage_below_{required_phase}:{feature}")
        if task_counts[feature] < required_phase:
            reasons.append(f"task_coverage_below_{required_phase}:{feature}")
    computed = [
        row
        for row in metric_rows
        if row.get("status") == "COMPUTED" and row.get("feature") in features
    ]
    if len(computed) != len(features) * len(SCOPES):
        reasons.append("agreement_metric_unavailable")
    composites = [float(row["composite_score"]) for row in computed]
    minimum = min(composites) if composites else None
    mean = float(np.mean(composites)) if composites else None

    if reasons:
        decision = "INSUFFICIENT_EVIDENCE"
    else:
        failed = [
            row
            for row in computed
            if float(row["composite_score"])
            < (
                config.pooled_composite_threshold
                if row["scope"] == "pooled"
                else config.phase_composite_threshold
            )
        ]
        if failed:
            decision = "NO_GO"
            reasons.extend(
                f"composite_below_threshold:{row['scope']}:{row['feature']}"
                for row in failed
            )
        else:
            decision = "GO"

    return _DomainDecision(
        decision=decision,
        pooled_count=pooled_count,
        relax_count=relax_count,
        task_count=task_count,
        minimum_composite=minimum,
        mean_composite=mean,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def write_hrv_agreement_artifacts(
    *,
    project_root: Path,
    run_id: str,
    validation_metrics: pd.DataFrame,
    interval_rows: Sequence[ObservedIntervalBlock],
    output_dir: Path,
    analyzer: HrvAgreementAnalyzer,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    run_identity: Mapping[str, Any],
    evaluation_scope: str,
    full_cohort_access_authorized: bool,
    detector_selection_path: Path | None = None,
) -> dict[str, Any]:
    """Write run-scoped HRV artifacts without mutating prior runs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    hrv_columns = [
        "participant_id",
        "phase",
        "sensor_channel",
        "recording_id",
        "source_file",
        "source_checksum",
        "ecg_source_file",
        "ecg_source_checksum",
        "ppg_source_file",
        "ppg_source_checksum",
        "hrv_feature_status",
        "hrv_feature_exclusion_reason",
        "hrv_common_observed_seconds",
        "hrv_method_version",
        "hrv_analysis_duration_label",
        "hrv_pnn50_unit",
        "ecg_spectral_corrected_interval_count",
        "ppg_spectral_corrected_interval_count",
        "ecg_spectral_correction_fraction",
        "ppg_spectral_correction_fraction",
        "ecg_spectral_support_seconds",
        "ppg_spectral_support_seconds",
        "hrv_tachogram_interpolation_method",
        *(f"ecg_hrv_{feature}" for feature in FEATURES),
        *(f"ppg_hrv_{feature}" for feature in FEATURES),
        *(f"ecg_reference_hrv_{feature}" for feature in FEATURES),
        *(f"ecg_reference_hrv_{feature}_status" for feature in FEATURES),
        *(f"hrv_{feature}_status" for feature in FEATURES),
        *(f"hrv_{feature}_exclusion_reason" for feature in FEATURES),
        "run_id",
        "source_tree_sha256",
        "run_contract_sha256",
        "config_fingerprint",
    ]
    missing = sorted(set(hrv_columns).difference(validation_metrics.columns))
    if missing:
        raise RuntimeError("hrv_feature_pair_missing_columns:" + ",".join(missing))
    pairs = validation_metrics[hrv_columns].copy()
    if len(pairs) != 192:
        raise RuntimeError(f"hrv_feature_pair_row_count:{len(pairs)}")
    key_columns = ["participant_id", "phase", "sensor_channel"]
    if pairs.duplicated(key_columns).any():
        raise RuntimeError("hrv_feature_pair_duplicate_keys")
    expected_keys = {
        (participant, phase, channel)
        for participant in sorted(DEVELOPMENT_PARTICIPANTS | HOLDOUT_PARTICIPANTS)
        for phase in ("relax", "task")
        for channel in ("ppg_green", "ppg_ir", "ppg_red")
    }
    actual_keys = set(pairs[key_columns].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        raise RuntimeError("hrv_feature_pair_cartesian_mismatch")
    pairs.to_csv(output_dir / "hrv_feature_pairs_240s.csv", index=False)
    pairs.to_parquet(output_dir / "hrv_feature_pairs_240s.parquet", index=False)

    interval_frame = pd.DataFrame(
        [asdict(row) for row in interval_rows],
        columns=[field.name for field in fields(ObservedIntervalBlock)],
    )
    interval_frame.to_parquet(
        output_dir / "observed_interval_blocks_240s.parquet", index=False
    )
    evaluation_pairs = (
        pairs
        if evaluation_scope == "full_cohort"
        else pairs[pairs["participant_id"].isin(SELECTION_PARTICIPANTS)]
    )
    payload, agreement_metrics = analyzer.evaluate(
        evaluation_pairs,
        evaluation_scope=evaluation_scope,
        full_cohort_access_authorized=full_cohort_access_authorized,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    agreement_metrics.to_csv(output_dir / "hrv_agreement_metrics.csv", index=False)
    payload.update({"run_id": run_id, **dict(run_identity)})
    _atomic_json(output_dir / "hrv_channel_recommendation.json", payload)
    _atomic_json(output_dir / "physiological_validation.json", payload)
    if evaluation_scope == "selection":
        _atomic_json(
            output_dir / "hrv_training_admission.json",
            build_hrv_training_admission(
                payload,
                analyzer.config,
                run_identity=run_identity,
            ),
        )
    (output_dir / "hrv_agreement_report.md").write_text(
        _markdown_report(payload, agreement_metrics), encoding="utf-8"
    )
    comparison = _run_comparison(
        project_root,
        run_id,
        validation_metrics,
        current_hrv_payload=payload,
        current_agreement_metrics=agreement_metrics,
    )
    _atomic_json(output_dir / "run_comparison.json", comparison)
    checklist = _implementation_checklist(
        run_id=run_id,
        pairs=pairs,
        validation_metrics=validation_metrics,
        payload=payload,
        detector_selection_path=(
            detector_selection_path
            if detector_selection_path is not None
            else output_dir.parent / "detector_selection" / "detector_selection.json"
        ),
    )
    _atomic_json(output_dir / "implementation_checklist.json", checklist)
    (output_dir / "implementation_checklist.md").write_text(
        _checklist_markdown(checklist), encoding="utf-8"
    )
    return payload


def _implementation_checklist(
    *,
    run_id: str,
    pairs: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    payload: Mapping[str, Any],
    detector_selection_path: Path,
) -> dict[str, Any]:
    detector_payload = (
        json.loads(detector_selection_path.read_text(encoding="utf-8"))
        if detector_selection_path.is_file()
        else None
    )

    def check(name: str, passed: bool | None, evidence: str) -> dict[str, str]:
        status = (
            "PASS" if passed is True else "FAIL" if passed is False else "UNAVAILABLE"
        )
        return {"check": name, "status": status, "evidence": evidence}

    ecg_interpolation = pd.to_numeric(
        validation_metrics.get(
            "ecg_interpolated_sample_count",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )
    source_files = pairs["source_file"].fillna("").astype(str).str.strip()
    source_checksums = pairs["source_checksum"].fillna("").astype(str).str.strip()
    rows_with_source = source_files.ne("")
    traced_source_rows = int(rows_with_source.sum())
    source_lineage_valid = bool(
        traced_source_rows > 0
        and source_checksums[rows_with_source].str.fullmatch(r"[0-9a-f]{64}").all()
    )
    checks = [
        check("exact_192_cartesian_rows", len(pairs) == 192, f"rows={len(pairs)}"),
        check(
            "complete_source_lineage",
            source_lineage_valid,
            (
                f"traced_rows={traced_source_rows}/{len(pairs)}; "
                "every present source requires a SHA-256 checksum; missing "
                "recordings remain explicit exclusions"
            ),
        ),
        check(
            "ecg_waveform_not_interpolated",
            None
            if ecg_interpolation.empty
            else bool((ecg_interpolation.fillna(0.0) == 0.0).all()),
            (
                "column unavailable"
                if ecg_interpolation.empty
                else f"sum={float(ecg_interpolation.fillna(0.0).sum()):g}"
            ),
        ),
        check(
            "time_domain_primary_gate",
            payload.get("primary_features") == list(TIME_DOMAIN_FEATURES),
            ",".join(payload.get("primary_features", [])),
        ),
        check(
            "spectral_evidence_explicit",
            payload.get("diagnostic_spectral_features") == list(SPECTRAL_FEATURES),
            str(payload.get("spectral_validation_role", "missing")),
        ),
        check(
            "beat_matching_diagnostic_only",
            payload.get("beat_matching_controls_primary_gate") is False,
            "beat F1 does not control HRV-PRV GO/NO_GO",
        ),
        check(
            "bounded_tachogram_interpolation_only",
            set(pairs["hrv_tachogram_interpolation_method"].astype(str)).issubset(
                {"none", "bounded_pchip"}
            ),
            ",".join(
                sorted(set(pairs["hrv_tachogram_interpolation_method"].astype(str)))
            ),
        ),
        check(
            "detector_policy_frozen",
            None
            if detector_payload is None
            else bool(
                detector_payload.get("run_id") == run_id
                and detector_payload.get("selection_contract_frozen") is True
            ),
            (
                "detector selection artifact unavailable"
                if detector_payload is None
                else (
                    f"run={detector_payload.get('run_id')};"
                    "frozen="
                    f"{detector_payload.get('selection_contract_frozen')}"
                )
            ),
        ),
    ]
    return {
        "schema_version": "hrv-prv-implementation-checklist-v2",
        "run_id": run_id,
        "checks": checks,
        "overall_status": (
            "PASS" if all(row["status"] == "PASS" for row in checks) else "PARTIAL"
        ),
    }


def _checklist_markdown(checklist: Mapping[str, Any]) -> str:
    lines = [
        "# HRV-PRV implementation checklist",
        "",
        f"Run: `{checklist['run_id']}`",
        "",
        f"Overall: `{checklist['overall_status']}`",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| {row['check']} | {row['status']} | {row['evidence']} |"
        for row in checklist["checks"]
    )
    return "\n".join(lines) + "\n"


def _markdown_report(payload: Mapping[str, Any], metrics: pd.DataFrame) -> str:
    lines = [
        "# HRV-PRV agreement",
        "",
        f"Primary decision: `{payload['hrv_validation_decision']}`",
        "",
        "Beat matching is diagnostic and does not control this decision.",
        "",
        "RMSSD, SDNN, and pNN50 control the canonical gate.",
        "LF, HF, and LF/HF are reported as nonblocking 240-second diagnostics.",
        "",
        "| Channel | Primary | Time N | Relax N | Task N | Spectral | Spectral N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["channels"].values():
        lines.append(
            f"| {row['sensor_channel']} | {row['decision']} | "
            f"{row['valid_time_domain_pooled_rows']}/64 | "
            f"{row['valid_time_domain_relax_rows']}/32 | "
            f"{row['valid_time_domain_task_rows']}/32 | "
            f"{row['spectral_validation_status']} | "
            f"{row['valid_spectral_pooled_rows']}/64 |"
        )
    lines.extend(
        [
            "",
            "## Feature metrics",
            "",
            "| Channel | Scope | Feature | N | Composite | Pearson | Spearman | ICC(A,1) | CCC |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics.to_dict("records"):

        def fmt(value: Any) -> str:
            return (
                f"{float(value):.4f}"
                if value is not None and np.isfinite(value)
                else "NA"
            )

        lines.append(
            f"| {row['sensor_channel']} | {row['scope']} | {row['feature']} | "
            f"{row['sample_count']} | {fmt(row['composite_score'])} | "
            f"{fmt(row['pearson_r'])} | {fmt(row['spearman_rho'])} | "
            f"{fmt(row['icc_a1'])} | {fmt(row['ccc'])} |"
        )
    lines.extend(
        [
            "",
            "`LF/HF` is an agreement feature only; it is not interpreted as direct sympathovagal balance.",
            "Clinical validation status remains `external_validation_required`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_comparison(
    project_root: Path,
    run_id: str,
    current_metrics: pd.DataFrame,
    *,
    current_hrv_payload: Mapping[str, Any],
    current_agreement_metrics: pd.DataFrame,
) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for candidate in (
        run_id,
        "20260828T170722Z",
        "20260828T074345Z",
        "20260809T193919Z",
    ):
        run_root = project_root / "data" / "artifacts" / "runs" / candidate
        if candidate == run_id:
            frame = current_metrics
            source = "current_run"
            hrv_payload: Mapping[str, Any] | None = current_hrv_payload
            hrv_metrics = current_agreement_metrics
        else:
            path = run_root / "validation" / "ppg_ecg_240s" / "ppg_ecg_metrics_240s.csv"
            if not path.is_file():
                runs[candidate] = {"status": "UNAVAILABLE", "path": str(path)}
                continue
            frame = pd.read_csv(path)
            source = str(path)
            hrv_root = run_root / "validation" / "hrv_agreement"
            hrv_payload_path = hrv_root / "physiological_validation.json"
            hrv_metrics_path = hrv_root / "hrv_agreement_metrics.csv"
            hrv_payload = (
                json.loads(hrv_payload_path.read_text(encoding="utf-8"))
                if hrv_payload_path.is_file()
                else None
            )
            hrv_metrics = (
                pd.read_csv(hrv_metrics_path)
                if hrv_metrics_path.is_file()
                else pd.DataFrame()
            )
        channels: dict[str, Any] = {}
        for channel, group in frame.groupby("sensor_channel"):
            valid = group[_metric_computable_mask(group)]
            hrv_computed = group.get(
                "hrv_feature_status", pd.Series("", index=group.index)
            ).eq("COMPUTED")
            development = group["participant_id"].isin(DEVELOPMENT_PARTICIPANTS)
            channels[str(channel)] = {
                "rows": len(group),
                "metric_computable": len(valid),
                "hrv_development_coverage": int((hrv_computed & development).sum()),
                "hrv_holdout_coverage": int((hrv_computed & ~development).sum()),
                "median_beat_f1": _median(valid, "beat_f1"),
                "median_ibi_mae_ms": _median(valid, "ibi_mae_ms"),
                "median_hr_mae_bpm": _median(valid, "hr_mae_bpm"),
            }
        composite_rows = (
            hrv_metrics[
                ["sensor_channel", "scope", "feature", "composite_score", "status"]
            ].to_dict("records")
            if not hrv_metrics.empty
            else []
        )
        runs[candidate] = {
            "status": "AVAILABLE",
            "source": source,
            "channels": channels,
            "hrv_agreement": (
                {
                    "status": "AVAILABLE",
                    "decision": hrv_payload.get("hrv_validation_decision"),
                    "channel_decisions": hrv_payload.get("channels", []),
                    "composites": composite_rows,
                }
                if hrv_payload is not None
                else {"status": "UNAVAILABLE"}
            ),
            "model_metrics": _model_comparison(run_root),
        }
    legacy_report = (
        project_root
        / "DATA PPG-20260609T013317Z-3-001"
        / "DATA PPG"
        / "PipelineSignal"
        / "HRV Validation"
        / "results"
        / "interpretasi_hasil.txt"
    )
    legacy_code = legacy_report.parents[1] / "hrv_validation.py"
    legacy_text = (
        legacy_code.read_text(encoding="utf-8", errors="replace")
        if legacy_code.is_file()
        else ""
    )
    expected_tables = (
        "tabel_hrv_ecg_vs_ppg.csv",
        "tabel_korelasi.csv",
        "tabel_icc.csv",
        "tabel_mae_rmse.csv",
    )
    return {
        "schema_version": "hrv-run-comparison-v2",
        "runs": runs,
        "legacy_pipeline_signal": {
            "status": "historical_unverifiable",
            "narrative_report": str(legacy_report) if legacy_report.is_file() else None,
            "narrative_report_sha256": (
                _sha256_file(legacy_report) if legacy_report.is_file() else None
            ),
            "implementation_file": str(legacy_code) if legacy_code.is_file() else None,
            "implementation_sha256": (
                _sha256_file(legacy_code) if legacy_code.is_file() else None
            ),
            "source_metric_tables": {
                name: (
                    "AVAILABLE"
                    if (legacy_report.parent / name).is_file()
                    else "UNAVAILABLE"
                )
                for name in expected_tables
            },
            "comparability": "NOT_COMPARABLE_WITH_CANONICAL_GATE",
            "verified_implementation_properties": {
                "good_or_moderate_segment_selection": (
                    'quality_label"] in ["Good", "Moderate"]' in legacy_text
                ),
                "compressed_interval_adjacency": "diff = np.diff(rr)" in legacy_text,
                "cubic_tachogram_interpolation": 'kind="cubic"' in legacy_text,
                "tachogram_extrapolation_enabled": 'fill_value="extrapolate"'
                in legacy_text,
                "minimum_frequency_intervals_10": "MIN_RR_FOR_FREQ = 10" in legacy_text,
            },
            "metric_semantics": "HRV-PRV feature agreement; not stress-model accuracy",
        },
    }


def _model_comparison(run_root: Path) -> dict[str, Any]:
    """Read unambiguous per-channel candidate manifests for report comparison."""
    model_root = run_root / "models" / "stress"
    if not model_root.is_dir():
        return {"status": "UNAVAILABLE"}

    channels: dict[str, Any] = {}
    for channel in ("ppg_green", "ppg_ir", "ppg_red"):
        manifests = sorted((model_root / channel).glob("*/candidate_manifest.json"))
        if not manifests:
            channels[channel] = {"status": "UNAVAILABLE"}
            continue
        if len(manifests) != 1:
            channels[channel] = {
                "status": "AMBIGUOUS",
                "manifest_paths": [str(path) for path in manifests],
            }
            continue
        path = manifests[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        feature_order = list(payload.get("feature_order", []))
        channels[channel] = {
            "status": "AVAILABLE",
            "manifest_path": str(path),
            "model_name": payload.get("model_name"),
            "model_version": payload.get("model_version"),
            "performance_tier": payload.get("performance_tier"),
            "release_status": payload.get("release_status"),
            "development_metrics": payload.get("development_metrics"),
            "holdout_metrics": payload.get("holdout_metrics"),
            "spectral_features_promoted": any(
                feature in {"log_lf_power", "log_hf_power", "lf_hf_ratio"}
                for feature in feature_order
            ),
            "converter_preflight": payload.get("converter_preflight"),
        }
    status = (
        "AVAILABLE"
        if any(item["status"] == "AVAILABLE" for item in channels.values())
        else "UNAVAILABLE"
    )
    return {"status": status, "channels": channels}


def _metric_computable_mask(frame: pd.DataFrame) -> pd.Series:
    """Return a schema-aware mask without reclassifying excluded rows."""
    if "metric_status" in frame.columns:
        return frame["metric_status"].isin({"computed", "reported"})
    if "validation_status" in frame.columns:
        return frame["validation_status"].eq("valid")
    if "status" in frame.columns:
        return frame["status"].isin({"ok", "valid"})
    return pd.Series(False, index=frame.index, dtype=bool)


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if not values.empty else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
