"""Canonical ECG--PPG peak-to-peak validation.

The implementation deliberately keeps ECG clock correction and PPG timestamps
separate.  ECG is the physiological reference; PPG peaks are matched after
estimating pulse-arrival time (PAT), never after shifting the PPG timeline.

References:
* NeuroKit2 ECG: https://neuropsychology.github.io/NeuroKit/functions/ecg.html
* NeuroKit2 PPG: https://neuropsychology.github.io/NeuroKit/functions/ppg.html
* SciPy ``find_peaks``:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
* Wearable PPG review: https://pmc.ncbi.nlm.nih.gov/articles/PMC7612541/
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Literal, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr

from .anchors import AnchorMapper, AnchorWindowSlicer
from .config import load_sqa_config
from .contracts import (
    DetectedPeaks,
    GapEvent,
    MatchResult,
    QualityDecision,
    VALID_PARTICIPANTS,
)
from .hrv_agreement import (
    FEATURES as HRV_FEATURES,
    HrvAgreementAnalyzer,
    write_hrv_agreement_artifacts,
)
from .peak_detection import PpgPeakDetector, PpgPeakDetectorPort
from .settings import (
    HrvAgreementConfig,
    PipelineConfig,
    PipelinePaths,
    SynchronizationConfig,
    pipeline_paths_from_config,
)
from .sqi import PostPeakSQA, PrePeakSQA, RawSignalSQA

LOGGER = logging.getLogger(__name__)
PPG_CHANNELS = ("ppg_ir", "ppg_red", "ppg_green")
PHASES = ("relax", "task")
DEVELOPMENT_PARTICIPANTS = frozenset(
    f"ID-{participant:02d}" for participant in range(1, 29)
)
DEFAULT_PAT_TOLERANCE_MS = 150.0
MIN_SIGNAL_SECONDS = 10.0
DEFAULT_VALIDATION_WINDOW_SECONDS = 240
MIN_PHYSIOLOGICAL_INTERVAL_MS = 300.0
MAX_PHYSIOLOGICAL_INTERVAL_MS = 1500.0
MORPHOLOGY_POLICIES = frozenset({"hard", "diagnostic"})
MORPHOLOGY_DIAGNOSTIC_REASONS = {
    "morphology_low_valid_ratio": "ppg_morphology_invalid",
    "morphology_amplitude_inconsistent": "ppg_morphology_amplitude_invalid",
    "morphology_template_unavailable": "ppg_morphology_template_unavailable",
    "morphology_template_inconsistent": "ppg_morphology_template_inconsistent",
}


def assert_canonical_validation_keyset(metrics: pd.DataFrame) -> None:
    """Reject duplicate or missing participant-phase-channel obligations."""

    keys = ("participant_id", "phase", "sensor_channel")
    if any(column not in metrics for column in keys):
        raise RuntimeError("canonical_validation_key_columns_missing")
    if metrics.duplicated(list(keys)).any():
        raise RuntimeError("canonical_validation_duplicate_keys")
    expected = {
        (participant, phase, channel)
        for participant in VALID_PARTICIPANTS
        for phase in PHASES
        for channel in PPG_CHANNELS
    }
    actual = set(metrics[list(keys)].itertuples(index=False, name=None))
    if actual != expected:
        raise RuntimeError("canonical_validation_cartesian_mismatch")


def _package_version(package: str) -> str:
    try:
        return f"{package}=={package_version(package)}"
    except PackageNotFoundError:
        return f"{package}==UNAVAILABLE"


@dataclass(frozen=True)
class PeakValidationOutputs:
    """Files produced by the canonical validator."""

    output_dir: Path
    metrics_path: Path
    row_count: int
    hrv_decision_path: Path | None = None


@dataclass(frozen=True)
class PreparedSignal:
    """Canonical signal view retaining observed samples and gap provenance."""

    timestamps_ms: np.ndarray
    values: np.ndarray
    fs_hz: float
    gap_mask: np.ndarray
    interpolated_mask: np.ndarray
    gap_events: tuple[GapEvent, ...]
    long_gap_detected: bool
    short_gap_interpolated: bool
    raw_sample_indices: np.ndarray | None = None
    timestamp_jitter_count: int = 0
    raw_index_gap_count: int = 0


@dataclass(frozen=True)
class MorphologyAssessment:
    """Observed pulse-shape metrics used by the post-peak quality gate."""

    valid_ratio: float
    score: float
    valid_beat_count: int
    median_amplitude: float | None
    amplitude_cv: float | None
    median_width_half_height_ms: float | None
    median_rise_time_ms: float | None
    median_decay_time_ms: float | None
    template_consistency: float | None
    flags: tuple[str, ...] = ()


def _quality_flag_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {flag for flag in value.split("|") if flag}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(flag) for flag in value if str(flag)}
    return set()


def apply_morphology_status_policy(
    results: pd.DataFrame,
    policy: str,
) -> pd.DataFrame:
    """Return a status-only policy view without altering peaks or metrics."""

    if policy not in MORPHOLOGY_POLICIES:
        raise ValueError(f"invalid_morphology_policy:{policy}")
    required = {
        "participant_id",
        "phase",
        "sensor_channel",
        "status",
        "validation_status",
        "quality_flags",
        "exclusion_reason",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("morphology_policy_missing_columns:" + ",".join(missing))

    output = results.copy(deep=True)
    if "quality_status" not in output:
        output["quality_status"] = "unavailable"
    if "strict_quality_eligible" not in output:
        output["strict_quality_eligible"] = False
    output["morphology_policy"] = policy

    morphology_reasons = set(MORPHOLOGY_DIAGNOSTIC_REASONS.values())
    for index, row in output.iterrows():
        flags = _quality_flag_set(row.get("quality_flags"))
        diagnostic_flags = flags.intersection(MORPHOLOGY_DIAGNOSTIC_REASONS)
        if "morphology_insufficient_peaks" in flags:
            output.at[index, "quality_status"] = "fail"
            output.at[index, "strict_quality_eligible"] = False
            continue

        validation_valid = str(row.get("validation_status", "")) == "valid"
        exclusion_reason = str(row.get("exclusion_reason", "") or "")
        morphology_only_exclusion = exclusion_reason in morphology_reasons
        if not diagnostic_flags:
            if validation_valid:
                output.at[index, "quality_status"] = "pass"
                output.at[index, "strict_quality_eligible"] = True
            continue

        output.at[index, "strict_quality_eligible"] = False
        if policy == "diagnostic":
            if str(row.get("quality_status", "")) == "fail":
                continue
            output.at[index, "quality_status"] = "warning"
            continue

        output.at[index, "quality_status"] = "fail"
        if validation_valid or morphology_only_exclusion:
            reason = next(
                MORPHOLOGY_DIAGNOSTIC_REASONS[flag]
                for flag in MORPHOLOGY_DIAGNOSTIC_REASONS
                if flag in diagnostic_flags
            )
            output.at[index, "status"] = "excluded"
            output.at[index, "validation_status"] = "excluded"
            output.at[index, "exclusion_reason"] = reason
    return output


def evaluate_morphology_policy_ablation(
    hard_results: pd.DataFrame,
    diagnostic_results: pd.DataFrame,
    *,
    maximum_f1_drop: float = 0.01,
    maximum_ibi_mae_increase_ms: float = 5.0,
    maximum_hr_mae_increase_bpm: float = 0.5,
) -> dict[str, Any]:
    """Evaluate morphology policy using only the complete development ledger."""

    key_columns = ["participant_id", "phase", "sensor_channel"]
    for name, frame in (("hard", hard_results), ("diagnostic", diagnostic_results)):
        missing = sorted(set(key_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"{name}_ablation_missing_columns:" + ",".join(missing))
        participants = set(frame["participant_id"].astype(str).unique())
        forbidden = sorted(participants.difference(DEVELOPMENT_PARTICIPANTS))
        if forbidden:
            raise ValueError("ablation_forbidden_participants:" + ",".join(forbidden))

    hard = hard_results.sort_values(key_columns).reset_index(drop=True)
    diagnostic = diagnostic_results.sort_values(key_columns).reset_index(drop=True)
    hard_keys = list(hard[key_columns].itertuples(index=False, name=None))
    diagnostic_keys = list(diagnostic[key_columns].itertuples(index=False, name=None))
    expected_keys = {
        (participant, phase, channel)
        for participant in DEVELOPMENT_PARTICIPANTS
        for phase in PHASES
        for channel in PPG_CHANNELS
    }
    complete_ledger = (
        len(hard_keys) == len(expected_keys)
        and len(set(hard_keys)) == len(hard_keys)
        and set(hard_keys) == expected_keys
        and hard_keys == diagnostic_keys
    )

    parity_columns = (
        "ecg_peak_count",
        "ppg_peak_count",
        "matched_beat_count",
        "beat_f1",
        "ibi_mae_ms",
        "hr_mae_bpm",
    )
    peak_stream_unchanged = hard_keys == diagnostic_keys and all(
        column in hard
        and column in diagnostic
        and hard[column].equals(diagnostic[column])
        for column in parity_columns
    )

    channels: list[dict[str, Any]] = []
    for channel in PPG_CHANNELS:
        hard_channel = hard[hard["sensor_channel"] == channel]
        diagnostic_channel = diagnostic[diagnostic["sensor_channel"] == channel]
        hard_valid = hard_channel[hard_channel["validation_status"] == "valid"]
        diagnostic_valid = diagnostic_channel[
            diagnostic_channel["validation_status"] == "valid"
        ]

        def median(frame: pd.DataFrame, column: str) -> float:
            if column not in frame:
                return float("nan")
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            return float(values.median()) if not values.empty else float("nan")

        hard_f1 = median(hard_valid, "beat_f1")
        diagnostic_f1 = median(diagnostic_valid, "beat_f1")
        hard_ibi = median(hard_valid, "ibi_mae_ms")
        diagnostic_ibi = median(diagnostic_valid, "ibi_mae_ms")
        hard_hr = median(hard_valid, "hr_mae_bpm")
        diagnostic_hr = median(diagnostic_valid, "hr_mae_bpm")
        f1_drop = hard_f1 - diagnostic_f1
        ibi_increase = diagnostic_ibi - hard_ibi
        hr_increase = diagnostic_hr - hard_hr
        finite = all(
            np.isfinite(value) for value in (f1_drop, ibi_increase, hr_increase)
        )
        noninferior = bool(
            finite
            and f1_drop <= maximum_f1_drop
            and ibi_increase <= maximum_ibi_mae_increase_ms
            and hr_increase <= maximum_hr_mae_increase_bpm
        )
        channels.append(
            {
                "sensor_channel": channel,
                "hard_valid": int(len(hard_valid)),
                "diagnostic_valid": int(len(diagnostic_valid)),
                "coverage_delta": int(len(diagnostic_valid) - len(hard_valid)),
                "f1_drop": float(f1_drop) if np.isfinite(f1_drop) else None,
                "ibi_mae_increase_ms": (
                    float(ibi_increase) if np.isfinite(ibi_increase) else None
                ),
                "hr_mae_increase_bpm": (
                    float(hr_increase) if np.isfinite(hr_increase) else None
                ),
                "noninferior": noninferior,
            }
        )

    hard_valid_total = int((hard["validation_status"] == "valid").sum())
    diagnostic_valid_total = int((diagnostic["validation_status"] == "valid").sum())
    coverage_increased = diagnostic_valid_total > hard_valid_total
    for row in channels:
        row["promotion_eligible"] = bool(
            complete_ledger
            and peak_stream_unchanged
            and row["coverage_delta"] > 0
            and row["noninferior"]
        )
    promotion_eligible = any(row["promotion_eligible"] for row in channels)
    return {
        "schema_version": "morphology-policy-ablation-v1",
        "development_participants": sorted(DEVELOPMENT_PARTICIPANTS),
        "holdout_participants_accessed": [],
        "complete_development_ledger": complete_ledger,
        "peak_stream_unchanged": peak_stream_unchanged,
        "hard_valid_total": hard_valid_total,
        "diagnostic_valid_total": diagnostic_valid_total,
        "coverage_increased": coverage_increased,
        "thresholds": {
            "maximum_f1_drop": maximum_f1_drop,
            "maximum_ibi_mae_increase_ms": maximum_ibi_mae_increase_ms,
            "maximum_hr_mae_increase_bpm": maximum_hr_mae_increase_bpm,
        },
        "channels": channels,
        "promotion_eligible": promotion_eligible,
    }


class PpgEcgValidator:
    """Validate all participant/phase/PPG-channel combinations.

    Silver is preferred because it contains post-SQA lineage.  The canonical
    synchronised frames are used as a local fallback when Silver has not yet
    been generated.  The fallback is still read from ``data/sync`` and never
    from the legacy ``DataSyncedECGPartisipan`` tree.
    """

    def __init__(
        self,
        paths: PipelinePaths,
        config: PipelineConfig | None = None,
        *,
        ppg_peak_detector: PpgPeakDetectorPort | None = None,
        write_detail_artifacts: bool = True,
        compute_confidence_intervals: bool = True,
        ecg_cache: dict[Path, tuple[pd.DataFrame, DetectedPeaks]] | None = None,
        ppg_cache: dict[Path, tuple[pd.DataFrame, PreparedSignal]] | None = None,
        ppg_detection_cache: dict[tuple[Path, str, str], DetectedPeaks] | None = None,
        allowed_participants: frozenset[str] | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self._allowed_participants = allowed_participants
        self.silver_dir = paths.silver_dir
        configured_sync = Path(
            config.artifacts.get("sync_dir", "data/sync") if config else "data/sync"
        )
        self.sync_dir = (
            configured_sync
            if configured_sync.is_absolute()
            else paths.project_root / configured_sync
        )
        self.output_dir = paths.validation_dir / "ppg_ecg_240s"
        self.audit_dir = self.output_dir / "plots"
        window = config.window if config is not None else None
        self._window_slicer = AnchorWindowSlicer(
            source_phase_duration_seconds=(
                window.source_phase_duration_seconds if window else 300
            ),
            edge_trim_start_seconds=window.edge_trim_start_seconds if window else 30,
            edge_trim_end_seconds=window.edge_trim_end_seconds if window else 30,
            analysis_phase_duration_seconds=(
                window.analysis_phase_duration_seconds if window else 240
            ),
        )
        self._ppg_peak_detector = ppg_peak_detector or (
            PpgPeakDetector(config.peak_detection) if config is not None else None
        )
        sqa_config = load_sqa_config()
        self._signal_sqa = RawSignalSQA(
            diagnostic_neurokit=bool(
                sqa_config.get("neurokit_quality_diagnostic", False)
            )
        )
        self._pre_peak_sqa = PrePeakSQA(self._signal_sqa)
        self._post_peak_sqa = PostPeakSQA(
            self._signal_sqa,
            minimum_peak_count=(
                config.peak_detection.minimum_peak_count if config else 15
            ),
            minimum_physiological_ibi_ratio=(
                config.peak_detection.minimum_physiological_ibi_ratio
                if config
                else 0.90
            ),
            physiological_ibi_min_ms=(
                config.peak_detection.physiological_ibi_min_ms
                if config
                else MIN_PHYSIOLOGICAL_INTERVAL_MS
            ),
            physiological_ibi_max_ms=(
                config.peak_detection.physiological_ibi_max_ms
                if config
                else MAX_PHYSIOLOGICAL_INTERVAL_MS
            ),
        )
        self._sqa_config = sqa_config
        self._hrv_analyzer = HrvAgreementAnalyzer(
            config.hrv_agreement if config is not None else HrvAgreementConfig()
        )
        self._hrv_interval_rows = []
        self._active_morphology_policy = "hard"
        self._active_morphology_policy_by_channel = {
            channel: "hard" for channel in PPG_CHANNELS
        }
        self._write_detail_artifacts = write_detail_artifacts
        self._compute_confidence_intervals = compute_confidence_intervals
        self._evaluation_ecg_cache = ecg_cache if ecg_cache is not None else {}
        self._evaluation_ppg_cache = ppg_cache if ppg_cache is not None else {}
        self._evaluation_ppg_detection_cache = (
            ppg_detection_cache if ppg_detection_cache is not None else {}
        )
        self.sync_manifest_path = self.sync_dir / "synchronization_manifest_v3.parquet"
        self._sync_manifest = (
            pd.read_parquet(self.sync_manifest_path)
            if self.sync_manifest_path.is_file()
            else pd.DataFrame()
        )
        if (
            self._allowed_participants is not None
            and "participant_id" in self._sync_manifest
        ):
            self._sync_manifest = self._sync_manifest[
                self._sync_manifest["participant_id"].isin(self._allowed_participants)
            ].copy()
        self._sync_ambiguities: set[tuple[str, str]] = set()
        self._session_ambiguities: set[tuple[str, str]] = set()
        self._sync_by_participant_phase = self._build_sync_index()
        self._anchors_by_participant_phase = self._load_anchors()
        self._run_identity = self._load_run_identity()

    def run(
        self,
        *,
        evaluation_scope: Literal["selection", "full_cohort"] | None = None,
    ) -> PeakValidationOutputs:
        """Run one explicit physiology scope without opening holdout early."""

        if evaluation_scope not in {None, "selection", "full_cohort"}:
            raise ValueError(f"invalid_evaluation_scope:{evaluation_scope}")
        if (
            evaluation_scope == "full_cohort"
            and not self._full_cohort_contracts_verified()
        ):
            raise RuntimeError("physiological_full_cohort_access_not_authorized")
        holdout_authorized = (
            (
                not bool(
                    self.config is not None and self.config.artifacts.get("run_id")
                )
                or self._full_cohort_contracts_verified()
            )
            if evaluation_scope is None
            else evaluation_scope == "full_cohort"
        )
        strict_run_scope = bool(
            self.config is not None and self.config.artifacts.get("run_id")
        )
        scope_name = "full_cohort" if holdout_authorized else "selection"
        base_output_dir = self.paths.validation_dir / "ppg_ecg_240s"
        self.output_dir = (
            base_output_dir
            if scope_name == "full_cohort"
            else base_output_dir / "selection"
        )
        self.audit_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._hrv_interval_rows.clear()
        ecg_cache: dict[Path, tuple[pd.DataFrame, DetectedPeaks]] = {}

        # Candidate selection is complete before the reused internal cohort is opened.
        self._active_morphology_policy = "diagnostic"
        self._active_morphology_policy_by_channel = {
            channel: "diagnostic" for channel in PPG_CHANNELS
        }
        development_rows: list[dict[str, Any]] = []
        for participant_id in sorted(DEVELOPMENT_PARTICIPANTS):
            for phase in PHASES:
                session_dir = self._find_session(participant_id, phase)
                for channel in PPG_CHANNELS:
                    development_rows.append(
                        self._validate_combination(
                            participant_id, phase, channel, session_dir, ecg_cache
                        )
                    )

        diagnostic_development = apply_morphology_status_policy(
            pd.DataFrame(development_rows), "diagnostic"
        )
        hard_development = apply_morphology_status_policy(
            diagnostic_development, "hard"
        )
        ablation = evaluate_morphology_policy_ablation(
            hard_development,
            diagnostic_development,
            maximum_f1_drop=float(
                self._sqa_config.get("morphology_ablation_max_f1_drop", 0.01)
            ),
            maximum_ibi_mae_increase_ms=float(
                self._sqa_config.get("morphology_ablation_max_ibi_mae_increase_ms", 5.0)
            ),
            maximum_hr_mae_increase_bpm=float(
                self._sqa_config.get("morphology_ablation_max_hr_mae_increase_bpm", 0.5)
            ),
        )
        requested_policy = self._configured_morphology_policy()
        promotion_by_channel = {
            row["sensor_channel"]: bool(row["promotion_eligible"])
            for row in ablation["channels"]
        }
        effective_policy_by_channel = {
            channel: (
                "diagnostic"
                if requested_policy == "diagnostic"
                and promotion_by_channel.get(channel, False)
                else "hard"
            )
            for channel in PPG_CHANNELS
        }
        distinct_policies = set(effective_policy_by_channel.values())
        effective_policy = (
            next(iter(distinct_policies)) if len(distinct_policies) == 1 else "mixed"
        )
        ablation.update(
            {
                "requested_policy": requested_policy,
                "effective_policy": effective_policy,
                "effective_policy_by_channel": effective_policy_by_channel,
                "promotion_applied": any(
                    policy == "diagnostic"
                    for policy in effective_policy_by_channel.values()
                ),
            }
        )
        self._write_json(self.output_dir / "morphology_policy_ablation.json", ablation)

        selected_development = pd.concat(
            [
                (
                    diagnostic_development
                    if effective_policy_by_channel[channel] == "diagnostic"
                    else hard_development
                ).loc[lambda frame: frame["sensor_channel"].eq(channel)]
                for channel in PPG_CHANNELS
            ],
            ignore_index=True,
        )
        self._active_morphology_policy = (
            effective_policy if effective_policy in MORPHOLOGY_POLICIES else "hard"
        )
        self._active_morphology_policy_by_channel = effective_policy_by_channel
        holdout_rows: list[dict[str, Any]] = []
        holdout_participants = sorted(VALID_PARTICIPANTS - DEVELOPMENT_PARTICIPANTS)
        holdout_ledger = (
            self.paths.validation_dir
            / "hrv_agreement"
            / "physiological_full_cohort_access.json"
        )
        governance_ledger: Path | None = None
        access_fingerprint: str | None = None
        if holdout_authorized and strict_run_scope:
            if holdout_ledger.is_file():
                raise RuntimeError("physiological_full_cohort_already_accessed")
            access_fingerprint = self._physiological_access_fingerprint()
            governance_ledger = self._claim_physiological_access(access_fingerprint)
            self._write_json(
                holdout_ledger,
                {
                    **self._run_identity,
                    "status": "started",
                    "participants": holdout_participants,
                    "selection_participants": sorted(DEVELOPMENT_PARTICIPANTS),
                    "primary_domain": "time_domain",
                    "detector_selection_sha256": self._file_sha256(
                        self.paths.validation_dir
                        / "detector_selection"
                        / "detector_selection.json"
                    ),
                    "stress_selection_contract_sha256": self._file_sha256(
                        self.paths.models_dir / "stress_selection_contract.json"
                    ),
                    "access_fingerprint": access_fingerprint,
                },
            )
            for participant_id in holdout_participants:
                for phase in PHASES:
                    session_dir = self._find_session(participant_id, phase)
                    for channel in PPG_CHANNELS:
                        holdout_rows.append(
                            self._validate_combination(
                                participant_id, phase, channel, session_dir, ecg_cache
                            )
                        )
        else:
            for participant_id in holdout_participants:
                for phase in PHASES:
                    for channel in PPG_CHANNELS:
                        row = self._validate_combination(
                            participant_id, phase, channel, None, ecg_cache
                        )
                        row.update(
                            {
                                "status": "excluded",
                                "structural_status": "unavailable",
                                "metric_status": "unavailable",
                                "quality_status": "unavailable",
                                "validation_status": "excluded",
                                "exclusion_reason": (
                                    "full_cohort_not_evaluated_unfrozen_contract"
                                ),
                                "hrv_feature_status": "INSUFFICIENT_EVIDENCE",
                                "hrv_feature_exclusion_reason": (
                                    "full_cohort_not_evaluated_unfrozen_contract"
                                ),
                                **{
                                    f"hrv_{feature}_status": "INSUFFICIENT_EVIDENCE"
                                    for feature in HRV_FEATURES
                                },
                                **{
                                    f"hrv_{feature}_exclusion_reason": (
                                        "full_cohort_not_evaluated_unfrozen_contract"
                                    )
                                    for feature in HRV_FEATURES
                                },
                            }
                        )
                        holdout_rows.append(row)

        metrics = pd.concat(
            [selected_development, pd.DataFrame(holdout_rows)], ignore_index=True
        )
        expected_rows = len(VALID_PARTICIPANTS) * len(PHASES) * len(PPG_CHANNELS)
        if len(metrics) != expected_rows:
            raise RuntimeError(f"canonical_validation_row_count:{len(metrics)}")
        assert_canonical_validation_keyset(metrics)
        metrics_path = self.output_dir / "ppg_ecg_metrics_240s.csv"
        metrics.to_csv(metrics_path, index=False)
        # Keep the legacy filename as a compatibility alias for existing reports.
        metrics.to_csv(self.output_dir / "ppg_ecg_metrics.csv", index=False)
        self._write_json(
            self.output_dir / "ppg_ecg_metrics.json", metrics.to_dict("records")
        )
        # A validation row is an audit obligation even when its ECG or PPG
        # source is excluded.  Emit an explicit, labelled artifact for every
        # canonical combination rather than leaving a missing plot that could
        # be mistaken for a successful-but-unreported comparison.
        self._ensure_plot_artifacts(metrics)
        self._write_sync_parameters()
        self._write_validation_artifact_tables(metrics)
        beat_recommendation = self._write_channel_recommendation(metrics)
        sync = self.config.synchronization if self.config else None
        hrv_output_dir = self.paths.validation_dir / "hrv_agreement"
        if not holdout_authorized:
            hrv_output_dir = hrv_output_dir / "selection"
        hrv_recommendation = write_hrv_agreement_artifacts(
            project_root=self.paths.project_root,
            run_id=str(self._run_identity["run_id"]),
            validation_metrics=metrics,
            interval_rows=self._hrv_interval_rows,
            output_dir=hrv_output_dir,
            analyzer=self._hrv_analyzer,
            bootstrap_iterations=(int(sync.bootstrap_iterations) if sync else 2000),
            bootstrap_seed=(int(sync.bootstrap_seed) if sync else 42),
            run_identity=self._run_identity,
            evaluation_scope=("full_cohort" if holdout_authorized else "selection"),
            full_cohort_access_authorized=holdout_authorized,
            detector_selection_path=(
                self.paths.validation_dir
                / "detector_selection"
                / "detector_selection.json"
            ),
        )
        if holdout_authorized and strict_run_scope:
            self._write_json(
                holdout_ledger,
                {
                    **self._run_identity,
                    "status": "completed",
                    "participants": holdout_participants,
                    "evaluated_rows": len(holdout_rows),
                    "selection_participants": sorted(DEVELOPMENT_PARTICIPANTS),
                    "primary_domain": "time_domain",
                    "detector_selection_sha256": self._file_sha256(
                        self.paths.validation_dir
                        / "detector_selection"
                        / "detector_selection.json"
                    ),
                    "stress_selection_contract_sha256": self._file_sha256(
                        self.paths.models_dir / "stress_selection_contract.json"
                    ),
                    "access_fingerprint": access_fingerprint,
                },
            )
            assert governance_ledger is not None
            assert access_fingerprint is not None
            self._complete_physiological_access(
                governance_ledger,
                access_fingerprint=access_fingerprint,
                evaluated_rows=len(holdout_rows),
                evidence_path=hrv_output_dir / "physiological_validation.json",
            )
        summary = self._summary(metrics, beat_recommendation)
        summary["beat_diagnostic_status"] = "reported"
        summary["beat_diagnostic_recommended_channel"] = beat_recommendation.get(
            "recommended_channel"
        )
        summary["hrv_validation_decision"] = hrv_recommendation[
            "hrv_validation_decision"
        ]
        summary["recommended_channel"] = hrv_recommendation.get("recommended_channel")
        channel_evidence = hrv_recommendation.get("channels", {})
        summary["hrv_channel_decisions"] = {
            channel: row["decision"] for channel, row in channel_evidence.items()
        }
        summary["spectral_validation_channel_statuses"] = {
            channel: row["spectral_validation_status"]
            for channel, row in channel_evidence.items()
        }
        summary.update(self._run_identity)
        self._write_json(self.output_dir / "validation_summary_240s.json", summary)
        self._write_json(self.output_dir / "validation_summary.json", summary)
        (self.output_dir / "validation_summary_240s.md").write_text(
            self._summary_markdown(summary), encoding="utf-8"
        )
        (self.output_dir / "validation_summary.md").write_text(
            self._summary_markdown(summary), encoding="utf-8"
        )
        self._update_readiness(summary)
        return PeakValidationOutputs(
            self.output_dir,
            metrics_path,
            len(metrics),
            hrv_output_dir / "physiological_validation.json",
        )

    def validate_combination(
        self, participant_id: str, phase: str, channel: str
    ) -> dict[str, Any]:
        """Evaluate one canonical combination without widening cohort scope."""

        return self._validate_combination(
            participant_id,
            phase,
            channel,
            self._find_session(participant_id, phase),
            self._evaluation_ecg_cache,
        )

    def _validate_combination(
        self,
        participant_id: str,
        phase: str,
        channel: str,
        session_dir: Path | None,
        ecg_cache: dict[Path, tuple[pd.DataFrame, DetectedPeaks]],
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "participant_id": participant_id,
            "session_id": participant_id,
            "recording_id": "",
            "phase": phase,
            "sensor_channel": channel,
            "status": "excluded",
            "structural_status": "unavailable",
            "metric_status": "unavailable",
            "quality_status": "unavailable",
            "strict_quality_eligible": False,
            "validation_status": "excluded",
            "morphology_policy": self._morphology_policy_for_channel(channel),
            "agreement_status": "insufficient_evidence",
            "exclusion_reason": "",
            "source_root": "",
            "source_file": "",
            "source_checksum": "",
            "ecg_source_file": "",
            "ppg_source_file": "",
            "ecg_source_checksum": "",
            "ppg_source_checksum": "",
            "config_fingerprint": self._config_fingerprint(),
            "ecg_peak_count": 0,
            "ppg_peak_count": 0,
            "matched_beat_count": 0,
            "matched_block_count": 0,
            "unmatched_ecg_count": 0,
            "unmatched_ppg_count": 0,
            "unmatched_ecg_indices": "",
            "unmatched_ppg_indices": "",
            "ecg_gap_count": 0,
            "ecg_gap_seconds": 0.0,
            "ecg_gap_mask_count": 0,
            "ecg_interpolated_sample_count": 0,
            "ecg_peak_in_gap_count": 0,
            "ppg_gap_count": 0,
            "ppg_gap_seconds": 0.0,
            "ppg_gap_mask_count": 0,
            "ppg_interpolated_sample_count": 0,
            "ppg_observed_sample_count": 0,
            "ppg_gap_provenance_sample_count": 0,
            "ppg_unresolved_sample_count": 0,
            "ppg_interpolation_method": "none",
            "ppg_interpolation_mask_sha256": "",
            "ppg_interpolation_mask_status": "unavailable",
            "ppg_peak_in_gap_count": 0,
            "matched_beat_ratio": np.nan,
            "pat_median_ms": np.nan,
            "pat_iqr_ms": np.nan,
            "pat_mad_ms": np.nan,
            "pat_slope_ms_per_minute": 0.0,
            "matcher_method": "",
            "beat_precision": np.nan,
            "beat_recall": np.nan,
            "beat_f1": np.nan,
            "beat_f1_ci_low": np.nan,
            "beat_f1_ci_high": np.nan,
            "ibi_pairs": 0,
            "ibi_mae_ms": np.nan,
            "ibi_mae_ms_ci_low": np.nan,
            "ibi_mae_ms_ci_high": np.nan,
            "ibi_median_absolute_error_ms": np.nan,
            "ibi_rmse_ms": np.nan,
            "ibi_rmse_ms_ci_low": np.nan,
            "ibi_rmse_ms_ci_high": np.nan,
            "ibi_pearson_r": np.nan,
            "ibi_spearman_r": np.nan,
            "hr_mae_bpm": np.nan,
            "hr_mae_bpm_ci_low": np.nan,
            "hr_mae_bpm_ci_high": np.nan,
            "bland_altman_bias_ms": np.nan,
            "bland_altman_bias_ms_ci_low": np.nan,
            "bland_altman_bias_ms_ci_high": np.nan,
            "bland_altman_loa_low_ms": np.nan,
            "bland_altman_loa_high_ms": np.nan,
            "ci_status": "insufficient_pairs",
            "ci_iterations": 0,
            "long_gap_detected": False,
            "ecg_excluded": False,
            "sync_method": "",
            "sync_alpha": np.nan,
            "sync_beta_seconds": np.nan,
            "ecg_detection_method": "",
            "ecg_detector_version": "",
            "ecg_reference_status": "not_evaluated",
            "ppg_detection_method": "",
            "ppg_detector_version": "",
            "clock_model": "",
            "detection_fallback_reason": "",
            "pre_peak_quality_tier": "",
            "post_peak_quality_tier": "",
            "quality_flags": "",
            "morphology_valid_ratio": np.nan,
            "morphology_score": np.nan,
            "morphology_valid_beat_count": 0,
            "morphology_median_amplitude": np.nan,
            "morphology_amplitude_cv": np.nan,
            "morphology_width_half_height_median_ms": np.nan,
            "morphology_rise_time_median_ms": np.nan,
            "morphology_decay_time_median_ms": np.nan,
            "morphology_template_consistency": np.nan,
            "ppg_physiological_ibi_ratio": np.nan,
            "motion_score": np.nan,
            "motion_quality_status": "not_evaluated",
            "interpolated_short_gap": False,
            "rr_interpolation_used": False,
            "rr_interpolation_method": (
                self.config.synchronization.rr_interpolation_method
                if self.config
                else "pchip"
            ),
            "validation_window_seconds": self._validation_window_seconds(),
            "window_start_utc": "",
            "window_end_utc": "",
            "run_id": self._run_identity["run_id"],
            "source_tree_sha256": self._run_identity["source_tree_sha256"],
            "run_contract_sha256": self._run_identity["run_contract_sha256"],
            "run_contract_status": self._run_identity["status"],
        }
        base.update(
            {
                "hrv_feature_status": "INSUFFICIENT_EVIDENCE",
                "hrv_feature_exclusion_reason": "validation_not_reached",
                "hrv_common_observed_seconds": 0.0,
                "hrv_method_version": "block_safe_hrv_prv_v3",
                "hrv_analysis_duration_label": "short_term_240s",
                "hrv_pnn50_unit": "fraction",
                "ecg_spectral_corrected_interval_count": 0,
                "ppg_spectral_corrected_interval_count": 0,
                "ecg_spectral_correction_fraction": np.nan,
                "ppg_spectral_correction_fraction": np.nan,
                "ecg_spectral_support_seconds": np.nan,
                "ppg_spectral_support_seconds": np.nan,
                "hrv_tachogram_interpolation_method": "none",
                **{
                    f"{role}_hrv_{feature}": np.nan
                    for role in ("ecg", "ppg", "ecg_reference")
                    for feature in HRV_FEATURES
                },
                **{
                    f"ecg_reference_hrv_{feature}_status": "INSUFFICIENT_EVIDENCE"
                    for feature in HRV_FEATURES
                },
                **{
                    f"hrv_{feature}_status": "INSUFFICIENT_EVIDENCE"
                    for feature in HRV_FEATURES
                },
                **{
                    f"hrv_{feature}_exclusion_reason": "validation_not_reached"
                    for feature in HRV_FEATURES
                },
            }
        )
        if (
            self._run_identity["strict_scope"]
            and self._run_identity["status"] != "ready"
        ):
            base["exclusion_reason"] = "run_contract_" + str(
                self._run_identity["status"]
            )
            return base
        if not self.sync_manifest_path.is_file():
            base["exclusion_reason"] = "missing_canonical_sync_manifest"
            return base
        if session_dir is None:
            base["exclusion_reason"] = (
                "LINEAGE_AMBIGUOUS"
                if (participant_id, phase) in self._session_ambiguities
                else "missing_canonical_session"
            )
            return base

        ecg_path = session_dir / "ecg.parquet"
        ppg_path = self._preferred_ppg_path(participant_id, phase, channel)
        base["source_root"] = self._relative(session_dir)
        if not ecg_path.is_file():
            base["exclusion_reason"] = "missing_ecg"
            return base
        if ppg_path is None:
            base["exclusion_reason"] = "missing_canonical_silver_ppg"
            return base

        cached_ppg = self._evaluation_ppg_cache.get(ppg_path)
        ppg = cached_ppg[0] if cached_ppg is not None else pd.read_parquet(ppg_path)
        cached_ecg = ecg_cache.get(session_dir)
        ecg = cached_ecg[0] if cached_ecg is not None else pd.read_parquet(ecg_path)
        if "clean_value" not in ppg.columns:
            base["exclusion_reason"] = "silver_missing_clean_value"
            return base
        base["ecg_source_file"] = self._source_file(ecg, ecg_path)
        base["ppg_source_file"] = self._source_file(ppg, ppg_path)
        base["ecg_source_checksum"] = self._source_checksum(ecg, ecg_path)
        base["ppg_source_checksum"] = self._source_checksum(ppg, ppg_path)
        base["source_file"] = base["ppg_source_file"]
        base["source_checksum"] = base["ppg_source_checksum"]
        base["recording_id"] = self._recording_id(
            ecg, participant_id, phase, session_dir
        )
        if not self._add_sync_metadata(base, participant_id, phase):
            base["exclusion_reason"] = (
                "LINEAGE_AMBIGUOUS"
                if (participant_id, phase) in self._sync_ambiguities
                else "missing_canonical_sync_metadata"
            )
            return base

        try:
            window = self._validation_window(participant_id, phase)
            if window is None:
                base["exclusion_reason"] = "missing_phase_anchor"
                return base
            start, end = window
            base["window_start_utc"] = start.isoformat()
            base["window_end_utc"] = end.isoformat()
            if cached_ecg is None:
                ecg, ecg_coverage = self._crop_to_window(ecg, start, end)
            else:
                ecg_coverage = self._frame_coverage(ecg)
            if cached_ppg is None:
                ppg, ppg_coverage = self._crop_to_window(ppg, start, end)
            else:
                ppg_coverage = self._frame_coverage(ppg)
            required = self._min_overlap_seconds()
            if ecg_coverage < required or ppg_coverage < required:
                base["exclusion_reason"] = (
                    "insufficient_240s_coverage:"
                    f"ecg={ecg_coverage:.3f}s;ppg={ppg_coverage:.3f}s;"
                    f"required>={required:.3f}s"
                )
                return base
            ecg_excluded, ecg_reason = self._ecg_exclusion(ecg)
            base["ecg_excluded"] = ecg_excluded
            if ecg_excluded:
                base["structural_status"] = "fail"
                base["quality_status"] = "fail"
                base["exclusion_reason"] = f"ecg_excluded:{ecg_reason}"
                return base
            if cached_ecg is None:
                ecg_prepared = self._prepare_frame(ecg, "ecg")
                ecg_det = self._detect_ecg(
                    ecg_prepared,
                    config_fingerprint=self._config_fingerprint(),
                )
                ecg_cache[session_dir] = (ecg, ecg_det)
            else:
                ecg, ecg_det = cached_ecg
                ecg_prepared = self._prepare_frame(ecg, "ecg")
            base["ecg_peak_count"] = int(len(ecg_det.timestamps_ms))
            base["ecg_gap_count"] = len(ecg_det.gap_events)
            base["ecg_gap_seconds"] = self._gap_seconds(ecg_det.gap_events)
            base["ecg_gap_mask_count"] = self._mask_count(ecg_det.gap_mask)
            base["ecg_timestamp_jitter_count"] = ecg_det.timestamp_jitter_count
            base["ecg_raw_index_gap_count"] = ecg_det.raw_index_gap_count
            base["ecg_interpolated_sample_count"] = ecg_det.interpolated_sample_count
            base["ecg_peak_in_gap_count"] = self._peak_gap_count(ecg_det)
            base["ecg_detection_method"] = ecg_det.method
            base["ecg_detector_version"] = ecg_det.library_version
            base["ecg_reference_status"] = ecg_det.reference_status
            if ecg_det.reference_status != "CONSENSUS":
                base["structural_status"] = "fail"
                base["exclusion_reason"] = "REFERENCE_UNCERTAIN:" + (
                    ecg_det.fallback_reason or "ecg_detector_consensus_failed"
                )
                return base
            ecg_reference_lineage = {
                key: str(base.get(key, ""))
                for key in (
                    "run_id",
                    "recording_id",
                    "ecg_source_file",
                    "ecg_source_checksum",
                    "ecg_detection_method",
                    "ecg_detector_version",
                    "config_fingerprint",
                    "run_contract_sha256",
                )
            }
            base.update(
                self._hrv_analyzer.ecg_reference_columns(
                    participant_id=participant_id,
                    phase=phase,
                    sensor_channel=channel,
                    ecg=ecg_det,
                    lineage=ecg_reference_lineage,
                )
            )
            if cached_ppg is None:
                ppg_prepared = self._prepare_frame(ppg, channel)
                self._evaluation_ppg_cache[ppg_path] = (ppg, ppg_prepared)
            else:
                ppg, ppg_prepared = cached_ppg
            interpolation_provenance = self._interpolation_mask_provenance(ppg)
            base.update(interpolation_provenance)
            if base["ppg_interpolation_mask_status"] == "invalid_partition":
                base["structural_status"] = "fail"
                base["exclusion_reason"] = "ppg_interpolation_mask_partition_invalid"
                return base
            config_fingerprint = self._config_fingerprint()
            detector_method = str(
                getattr(
                    self._ppg_peak_detector,
                    "method",
                    self.config.peak_detection.primary_method
                    if self.config is not None
                    else "unknown",
                )
            )
            detection_key = (ppg_path, detector_method, config_fingerprint)
            ppg_det = self._evaluation_ppg_detection_cache.get(detection_key)
            if ppg_det is None:
                ppg_det = self._detect_ppg(
                    ppg_prepared,
                    channel=channel,
                    config_fingerprint=config_fingerprint,
                )
                self._evaluation_ppg_detection_cache[detection_key] = ppg_det
            base["ppg_peak_count"] = int(len(ppg_det.timestamps_ms))
            base["long_gap_detected"] = (
                ecg_det.long_gap_detected or ppg_det.long_gap_detected
            )
            base["interpolated_short_gap"] = ppg_det.interpolated_sample_count > 0
            base["ppg_gap_count"] = len(ppg_det.gap_events)
            base["ppg_gap_seconds"] = self._gap_seconds(ppg_det.gap_events)
            base["ppg_gap_mask_count"] = self._mask_count(ppg_det.gap_mask)
            base["ppg_timestamp_jitter_count"] = ppg_det.timestamp_jitter_count
            base["ppg_raw_index_gap_count"] = ppg_det.raw_index_gap_count
            base["ppg_interpolated_sample_count"] = ppg_det.interpolated_sample_count
            base["ppg_peak_in_gap_count"] = self._peak_gap_count(ppg_det)
            base["ppg_detection_method"] = ppg_det.method
            base["ppg_detector_version"] = ppg_det.library_version
            if ppg_det.reference_status == "UNAVAILABLE" or (
                not ppg_det.method and ppg_det.timestamps_ms.size == 0
            ):
                base["structural_status"] = "fail"
                base["exclusion_reason"] = "ppg_detector_unavailable"
                return base
            base["detection_fallback_reason"] = ";".join(
                reason
                for reason in (ecg_det.fallback_reason, ppg_det.fallback_reason)
                if reason
            )
            if min(len(ecg_det.timestamps_ms), len(ppg_det.timestamps_ms)) < 3:
                raise ValueError("too_few_detected_peaks")

            base["structural_status"] = "pass"
            match_result = self._match_result(
                ecg_det.timestamps_ms, ppg_det.timestamps_ms
            )
            matches = list(match_result.matches)
            base.update(
                self._metrics(
                    matches,
                    ecg_det.timestamps_ms,
                    ppg_det.timestamps_ms,
                    ecg_gap_events=ecg_det.gap_events,
                    ppg_gap_events=ppg_det.gap_events,
                )
            )
            base["metric_status"] = "computed"
            base["pat_median_ms"] = (
                match_result.pat_median_ms
                if match_result.pat_median_ms is not None
                else np.nan
            )
            base["pat_iqr_ms"] = (
                match_result.pat_iqr_ms
                if match_result.pat_iqr_ms is not None
                else np.nan
            )
            base["pat_mad_ms"] = (
                match_result.pat_mad_ms
                if match_result.pat_mad_ms is not None
                else np.nan
            )
            base["pat_slope_ms_per_minute"] = match_result.pat_slope_ms_per_minute
            base["matcher_method"] = match_result.method
            base["unmatched_ecg_count"] = match_result.unmatched_ecg_count
            base["unmatched_ppg_count"] = match_result.unmatched_ppg_count
            base["matched_block_count"] = match_result.matched_block_count
            base["unmatched_ecg_indices"] = ",".join(
                str(index) for index in match_result.unmatched_ecg_indices
            )
            base["unmatched_ppg_indices"] = ",".join(
                str(index) for index in match_result.unmatched_ppg_indices
            )

            ecg_quality = self._quality_assessment(
                ecg_prepared,
                ecg_det,
                channel="ecg",
            )
            ppg_quality = self._quality_assessment(
                ppg_prepared,
                ppg_det,
                channel=channel,
            )
            base["pre_peak_quality_tier"] = ppg_quality["pre_tier"]
            base["post_peak_quality_tier"] = ppg_quality["post_tier"]
            base["quality_flags"] = "|".join(
                dict.fromkeys(
                    [
                        *ecg_quality["flags"],
                        *ppg_quality["flags"],
                    ]
                )
            )
            base["morphology_valid_ratio"] = ppg_quality["morphology_valid_ratio"]
            base["morphology_score"] = ppg_quality["morphology_score"]
            base["morphology_valid_beat_count"] = ppg_quality[
                "morphology_valid_beat_count"
            ]
            base["morphology_median_amplitude"] = ppg_quality[
                "morphology_median_amplitude"
            ]
            base["morphology_amplitude_cv"] = ppg_quality["morphology_amplitude_cv"]
            base["morphology_width_half_height_median_ms"] = ppg_quality[
                "morphology_width_half_height_median_ms"
            ]
            base["morphology_rise_time_median_ms"] = ppg_quality[
                "morphology_rise_time_median_ms"
            ]
            base["morphology_decay_time_median_ms"] = ppg_quality[
                "morphology_decay_time_median_ms"
            ]
            base["morphology_template_consistency"] = ppg_quality[
                "morphology_template_consistency"
            ]
            base["ppg_physiological_ibi_ratio"] = ppg_quality["physiological_ibi_ratio"]
            base["motion_score"] = self._motion_score(ppg)
            base["motion_quality_status"] = self._motion_quality(ppg)
            quality_decision = ppg_quality["decision"]
            base["quality_status"] = quality_decision.quality_status
            base["strict_quality_eligible"] = quality_decision.strict_quality_eligible
            hrv_lineage = {
                key: str(base.get(key, ""))
                for key in (
                    "run_id",
                    "recording_id",
                    "ecg_source_file",
                    "ecg_source_checksum",
                    "ppg_source_file",
                    "ppg_source_checksum",
                    "ecg_detection_method",
                    "ecg_detector_version",
                    "ppg_detection_method",
                    "ppg_detector_version",
                    "config_fingerprint",
                    "run_contract_sha256",
                )
            }
            hrv_quality_reason = self._hrv_quality_exclusion_reason(
                ecg_quality,
                ppg_quality,
                base["motion_quality_status"],
            )
            if hrv_quality_reason is None:
                hrv_pair, interval_rows = self._hrv_analyzer.feature_pair(
                    participant_id=participant_id,
                    phase=phase,
                    sensor_channel=channel,
                    ecg=ecg_det,
                    ppg=ppg_det,
                    window_start_ms=float(start.timestamp() * 1000.0),
                    window_end_ms=float(end.timestamp() * 1000.0),
                    lineage=hrv_lineage,
                )
                base.update(self._hrv_analyzer.pair_columns(hrv_pair))
                if self._write_detail_artifacts:
                    self._hrv_interval_rows.extend(interval_rows)
            else:
                base["hrv_feature_exclusion_reason"] = hrv_quality_reason
            quality_reason = self._quality_exclusion_reason(
                ecg_quality,
                ppg_quality,
                base["motion_quality_status"],
            )
            if quality_reason:
                base["quality_status"] = "fail"
                base["strict_quality_eligible"] = False
                base["exclusion_reason"] = quality_reason
                return base
            if base["long_gap_detected"]:
                base["structural_status"] = "fail"
                base["exclusion_reason"] = "long_dropout_detected"
                return base
            min_peak_count = self._min_peak_count()
            min_match_count = self._min_match_count()
            if base["ecg_peak_count"] < min_peak_count:
                base["structural_status"] = "fail"
                base["exclusion_reason"] = (
                    f"insufficient_ecg_peak_count:{base['ecg_peak_count']}<"
                    f"{min_peak_count}"
                )
                return base
            if base["ppg_peak_count"] < min_peak_count:
                base["structural_status"] = "fail"
                base["exclusion_reason"] = (
                    f"insufficient_ppg_peak_count:{base['ppg_peak_count']}<"
                    f"{min_peak_count}"
                )
                return base
            if base["matched_beat_count"] < min_match_count:
                base["structural_status"] = "fail"
                base["exclusion_reason"] = (
                    f"insufficient_matched_beats:{base['matched_beat_count']}<"
                    f"{min_match_count}"
                )
                return base
            if base["ibi_pairs"] < 2:
                base["agreement_status"] = "insufficient_evidence"
            else:
                base["agreement_status"] = "reported"
            base["status"] = "ok"
            base["validation_status"] = "valid"
            if self.config is not None:
                base["preprocessing_version"] = self.config.preprocessing_version
            if self._write_detail_artifacts:
                try:
                    self._plot_audit(
                        participant_id, phase, channel, ecg_det, ppg_det, matches
                    )
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    # Rendering must not turn a valid scientific comparison into
                    # an exclusion. Keep the failure explicit for audit/retry.
                    base["audit_plot_error"] = f"{type(exc).__name__}:{exc}"
                self._write_peak_table(
                    participant_id, phase, channel, ecg_det, ppg_det, matches
                )
        except (ValueError, KeyError, TypeError) as exc:
            base["structural_status"] = "fail"
            base["exclusion_reason"] = str(exc)
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive boundary for one bad recording
            LOGGER.exception(
                "Peak validation failed for %s/%s/%s", participant_id, phase, channel
            )
            base["structural_status"] = "fail"
            base["exclusion_reason"] = f"processing_error:{type(exc).__name__}:{exc}"
        return base

    def _quality_assessment(
        self,
        prepared: PreparedSignal,
        detected: DetectedPeaks,
        *,
        channel: str,
    ) -> dict[str, Any]:
        """Run pre/post peak quality checks without changing signal values."""

        timestamps = prepared.timestamps_ms
        pre = self._pre_peak_sqa.assess(
            prepared.values,
            timestamps_utc=pd.to_datetime(timestamps, unit="ms", utc=True),
            channel=channel,
            sampling_rate_hz=prepared.fs_hz,
            interpolation_mask=prepared.interpolated_mask,
        )
        post = self._post_peak_sqa.assess(
            prepared.values,
            detected.sample_indices,
            sampling_rate_hz=prepared.fs_hz,
            timestamps_utc=pd.to_datetime(timestamps, unit="ms", utc=True),
            interpolation_mask=prepared.interpolated_mask,
            pre_quality=pre,
        )
        morphology = (
            self._morphology_quality(prepared, detected)
            if channel != "ecg"
            else MorphologyAssessment(
                np.nan,
                np.nan,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        )
        flags = tuple(
            dict.fromkeys(
                [
                    *pre.flags,
                    *post.flags,
                    *morphology.flags,
                    *(("long_dropout",) if prepared.long_gap_detected else ()),
                ]
            )
        )
        morphology_failed = self._morphology_exclusion_reason(
            {
                "morphology_valid_ratio": morphology.valid_ratio,
                "flags": morphology.flags,
            },
            policy=self._morphology_policy_for_channel(channel),
        )
        pre_or_post_failed = pre.tier in {"poor", "invalid"} or post.tier in {
            "poor",
            "invalid",
        }
        excluded = pre_or_post_failed or morphology_failed is not None
        morphology_warning = bool(
            set(morphology.flags).intersection(MORPHOLOGY_DIAGNOSTIC_REASONS)
        )
        quality_status = (
            "fail" if excluded else "warning" if morphology_warning else "pass"
        )
        quality_scores = [pre.score, post.score]
        if np.isfinite(morphology.score):
            quality_scores.append(morphology.score)
        decision = QualityDecision(
            status="excluded" if excluded else "valid",
            tier=post.tier,
            score=float(min(quality_scores)),
            flags=flags,
            exclusion_reason=(
                "pre_or_post_peak_or_morphology_quality_failed" if excluded else None
            ),
            morphology_valid_ratio=(
                morphology.valid_ratio if np.isfinite(morphology.valid_ratio) else None
            ),
            structural_status="pass",
            metric_status="computed",
            quality_status=quality_status,
            strict_quality_eligible=quality_status == "pass",
            validation_status="excluded" if excluded else "valid",
        )
        return {
            "pre_tier": pre.tier,
            "post_tier": post.tier,
            "flags": flags,
            "morphology_valid_ratio": morphology.valid_ratio,
            "morphology_score": morphology.score,
            "morphology_valid_beat_count": morphology.valid_beat_count,
            "morphology_median_amplitude": morphology.median_amplitude,
            "morphology_amplitude_cv": morphology.amplitude_cv,
            "morphology_width_half_height_median_ms": morphology.median_width_half_height_ms,
            "morphology_rise_time_median_ms": morphology.median_rise_time_ms,
            "morphology_decay_time_median_ms": morphology.median_decay_time_ms,
            "morphology_template_consistency": morphology.template_consistency,
            "physiological_ibi_ratio": post.physiological_ibi_ratio,
            "decision": decision,
        }

    def _morphology_quality(
        self,
        prepared: PreparedSignal,
        detected: DetectedPeaks,
    ) -> MorphologyAssessment:
        """Measure observed pulse shape without reconstructing waveform data."""

        peaks = np.asarray(detected.sample_indices, dtype=int)
        values = np.asarray(prepared.values, dtype=float)
        timestamps = np.asarray(prepared.timestamps_ms, dtype=float)
        empty_flags = ("morphology_insufficient_peaks",)
        if peaks.size < 3:
            return MorphologyAssessment(
                np.nan,
                np.nan,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                empty_flags,
            )

        valid_mask = np.isfinite(values)
        if prepared.gap_mask is not None:
            valid_mask &= ~prepared.gap_mask
        search_samples = max(1, int(round(prepared.fs_hz * 1.5)))
        amplitudes: list[float] = []
        widths_ms: list[float] = []
        rise_ms: list[float] = []
        decay_ms: list[float] = []
        templates: list[np.ndarray] = []
        for peak_position, peak in enumerate(peaks):
            if (
                peak < 0
                or peak >= values.size
                or not valid_mask[peak]
                or not np.isfinite(timestamps[peak])
            ):
                continue
            previous_peak = peaks[peak_position - 1] if peak_position else 0
            next_peak = (
                peaks[peak_position + 1]
                if peak_position + 1 < peaks.size
                else values.size - 1
            )
            left_start = max(0, int(peak) - search_samples, int(previous_peak) + 1)
            right_end = min(
                values.size - 1,
                int(peak) + search_samples,
                int(next_peak) - 1,
            )
            if left_start >= peak or right_end <= peak:
                continue
            left_indices = np.arange(left_start, int(peak), dtype=int)
            right_indices = np.arange(int(peak) + 1, right_end + 1, dtype=int)
            left_indices = left_indices[valid_mask[left_indices]]
            right_indices = right_indices[valid_mask[right_indices]]
            if left_indices.size == 0 or right_indices.size == 0:
                continue
            trough = int(left_indices[np.argmin(values[left_indices])])
            recovery = int(right_indices[np.argmin(values[right_indices])])
            amplitude = float(values[peak] - values[trough])
            rise = float(timestamps[peak] - timestamps[trough])
            decay = float(timestamps[recovery] - timestamps[peak])
            if (
                not np.isfinite(amplitude)
                or amplitude <= 0.0
                or not np.isfinite(rise)
                or rise <= 0.0
                or not np.isfinite(decay)
                or decay <= 0.0
            ):
                continue
            amplitudes.append(amplitude)
            rise_ms.append(rise)
            decay_ms.append(decay)

            half_level = float(values[trough] + amplitude / 2.0)
            rising = left_indices[values[left_indices] <= half_level]
            falling = right_indices[values[right_indices] <= half_level]
            if rising.size and falling.size:
                width = float(timestamps[int(falling[0])] - timestamps[int(rising[-1])])
                if np.isfinite(width) and width > 0.0:
                    widths_ms.append(width)

            landmark_indices = np.rint(
                np.linspace(trough, recovery, 5, dtype=float)
            ).astype(int)
            if np.all(valid_mask[landmark_indices]):
                templates.append(
                    (values[landmark_indices] - values[trough]) / amplitude
                )

        valid_ratio = float(len(amplitudes) / peaks.size)
        min_ratio = float(self._sqa_config.get("morphology_min_valid_ratio", 0.75))
        if len(amplitudes) < 3:
            flags = list(empty_flags)
            if valid_ratio < min_ratio:
                flags.append("morphology_low_valid_ratio")
            return MorphologyAssessment(
                valid_ratio,
                0.0,
                len(amplitudes),
                float(np.median(amplitudes)) if amplitudes else None,
                None,
                float(np.median(widths_ms)) if widths_ms else None,
                float(np.median(rise_ms)) if rise_ms else None,
                float(np.median(decay_ms)) if decay_ms else None,
                None,
                tuple(dict.fromkeys(flags)),
            )

        median_amplitude = float(np.median(amplitudes))
        amplitude_cv = (
            float(np.std(amplitudes, ddof=1) / median_amplitude)
            if median_amplitude > 0.0 and len(amplitudes) > 1
            else np.inf
        )
        max_cv = float(self._sqa_config.get("morphology_max_amplitude_cv", 1.5))

        template_consistency: float | None = None
        if len(templates) >= 3:
            template_matrix = np.asarray(templates, dtype=float)
            template = np.median(template_matrix, axis=0)
            correlations = [
                float(np.corrcoef(row, template)[0, 1])
                for row in template_matrix
                if np.std(row) > 0.0 and np.std(template) > 0.0
            ]
            if correlations:
                template_consistency = float(np.median(correlations))

        flags: list[str] = []
        if valid_ratio < min_ratio:
            flags.append("morphology_low_valid_ratio")
        if amplitude_cv > max_cv:
            flags.append("morphology_amplitude_inconsistent")
        template_min = float(self._sqa_config.get("morphology_template_corr_min", 0.60))
        if template_consistency is None:
            flags.append("morphology_template_unavailable")
        elif template_consistency < template_min:
            flags.append("morphology_template_inconsistent")

        amplitude_score = max(0.0, 1.0 - amplitude_cv / max(max_cv, 1.0e-12))
        template_score = (
            0.0
            if template_consistency is None
            else np.clip(
                (template_consistency - template_min)
                / max(1.0 - template_min, 1.0e-12),
                0.0,
                1.0,
            )
        )
        score = valid_ratio * amplitude_score * float(template_score)
        return MorphologyAssessment(
            valid_ratio,
            float(np.clip(score, 0.0, 1.0)),
            len(amplitudes),
            median_amplitude,
            amplitude_cv,
            float(np.median(widths_ms)) if widths_ms else None,
            float(np.median(rise_ms)) if rise_ms else None,
            float(np.median(decay_ms)) if decay_ms else None,
            template_consistency,
            tuple(dict.fromkeys(flags)),
        )

    @staticmethod
    def _motion_score(frame: pd.DataFrame) -> float:
        if "motion_score" not in frame.columns:
            return float("nan")
        values = pd.to_numeric(frame["motion_score"], errors="coerce").dropna()
        return float(values.median()) if not values.empty else float("nan")

    @staticmethod
    def _motion_quality(frame: pd.DataFrame) -> str:
        if "motion_heavy_mask" in frame.columns:
            heavy_mask = PpgEcgValidator._boolean_column(
                frame,
                "motion_heavy_mask",
                len(frame),
            )
            if heavy_mask.any():
                return "masked"
        text_columns = ("motion_annotation", "madc_action")
        text = " ".join(
            str(value).lower()
            for column in text_columns
            if column in frame.columns
            for value in frame[column].dropna().unique()
        )
        if "heavy" in text:
            return "heavy"
        if "unavailable" in text or "no_accelerometer" in text:
            return "unavailable"
        if text:
            return "available"
        return "unavailable"

    def _quality_exclusion_reason(
        self,
        ecg_quality: Mapping[str, Any],
        ppg_quality: Mapping[str, Any],
        motion_status: str,
    ) -> str | None:
        failed_tiers = {"poor", "invalid"}
        if (
            ecg_quality["pre_tier"] in failed_tiers
            or ecg_quality["post_tier"] in failed_tiers
        ):
            return (
                "ecg_sqa_failed:"
                f"pre={ecg_quality['pre_tier']};post={ecg_quality['post_tier']}"
            )
        if (
            ppg_quality["pre_tier"] in failed_tiers
            or ppg_quality["post_tier"] in failed_tiers
        ):
            return (
                "ppg_sqa_failed:"
                f"pre={ppg_quality['pre_tier']};post={ppg_quality['post_tier']}"
            )
        if motion_status == "heavy" and bool(
            self._sqa_config.get("motion_heavy_exclusion", True)
        ):
            return "motion_heavy_excluded"
        return self._morphology_exclusion_reason(ppg_quality)

    def _hrv_quality_exclusion_reason(
        self,
        ecg_quality: Mapping[str, Any],
        ppg_quality: Mapping[str, Any],
        motion_status: str,
    ) -> str | None:
        """Fail HRV only on SQA, verified motion, or unusable morphology."""

        failed_tiers = {"poor", "invalid"}
        if (
            ecg_quality["pre_tier"] in failed_tiers
            or ecg_quality["post_tier"] in failed_tiers
        ):
            return "ecg_sqa_failed"
        if (
            ppg_quality["pre_tier"] in failed_tiers
            or ppg_quality["post_tier"] in failed_tiers
        ):
            return "ppg_sqa_failed"
        if motion_status == "heavy" and bool(
            self._sqa_config.get("motion_heavy_exclusion", True)
        ):
            return "motion_heavy_excluded"
        if "morphology_insufficient_peaks" in set(ppg_quality.get("flags", ())):
            return "ppg_morphology_insufficient_peaks"
        return None

    def _morphology_exclusion_reason(
        self,
        quality: Mapping[str, Any],
        *,
        policy: str | None = None,
    ) -> str | None:
        """Apply configured morphology components; keep composite score diagnostic."""

        selected_policy = policy or self._active_morphology_policy
        if selected_policy not in MORPHOLOGY_POLICIES:
            raise ValueError(f"invalid_morphology_policy:{selected_policy}")
        flags = set(quality.get("flags", ()))
        if "morphology_insufficient_peaks" in flags:
            return "ppg_morphology_insufficient_peaks"
        if selected_policy == "diagnostic":
            return None

        valid_ratio = float(quality.get("morphology_valid_ratio", np.nan))
        if np.isfinite(valid_ratio) and valid_ratio < float(
            self._sqa_config.get("morphology_min_valid_ratio", 0.75)
        ):
            return "ppg_morphology_invalid"
        reasons = (
            ("morphology_amplitude_inconsistent", "ppg_morphology_amplitude_invalid"),
            ("morphology_template_unavailable", "ppg_morphology_template_unavailable"),
            (
                "morphology_template_inconsistent",
                "ppg_morphology_template_inconsistent",
            ),
        )
        for flag, reason in reasons:
            if flag in flags:
                return reason
        return None

    def _morphology_policy_for_channel(self, channel: str) -> str:
        return self._active_morphology_policy_by_channel.get(
            channel, self._active_morphology_policy
        )

    def _configured_morphology_policy(self) -> str:
        policy = str(self._sqa_config.get("morphology_policy", "hard")).lower()
        if policy not in MORPHOLOGY_POLICIES:
            raise ValueError(f"invalid_morphology_policy:{policy}")
        return policy

    def _frozen_detector_policy_verified(self) -> bool:
        selection_path = (
            self.paths.validation_dir / "detector_selection" / "detector_selection.json"
        )
        if not selection_path.is_file():
            return False
        try:
            payload = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if self.config is None:
            return False
        selected = payload.get("selected", {})
        configured_match = all(
            isinstance(selected.get(channel), Mapping)
            and selected[channel].get("candidate_method")
            == self.config.peak_detection.selected_method_for(channel)
            and (
                selected[channel].get("candidate_method") != "scipy_find_peaks_adaptive"
                or np.isclose(
                    float(selected[channel].get("prominence_mad_multiplier", np.nan)),
                    self.config.peak_detection.for_channel(
                        channel
                    ).prominence_mad_multiplier,
                )
            )
            for channel in PPG_CHANNELS
        )
        return bool(
            payload.get("selection_contract_frozen") is True
            and configured_match
            and str(payload.get("run_id", "")) == str(self._run_identity["run_id"])
            and str(payload.get("run_contract_sha256", ""))
            == str(self._run_identity["run_contract_sha256"])
            and str(payload.get("source_tree_sha256", ""))
            == str(self._run_identity["source_tree_sha256"])
            and str(payload.get("config_fingerprint", ""))
            == str(self._run_identity["config_fingerprint"])
        )

    def _full_cohort_contracts_verified(self) -> bool:
        if not self._frozen_detector_policy_verified():
            return False
        contract_path = self.paths.models_dir / "stress_selection_contract.json"
        if not contract_path.is_file():
            return False
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        identity_matches = bool(
            payload.get("selection_contract_frozen") is True
            and payload.get("full_cohort_physiology_may_open") is True
            and str(payload.get("run_id", "")) == str(self._run_identity["run_id"])
            and str(payload.get("run_contract_sha256", ""))
            == str(self._run_identity["run_contract_sha256"])
            and str(payload.get("source_tree_sha256", ""))
            == str(self._run_identity["source_tree_sha256"])
            and str(payload.get("config_fingerprint", ""))
            == str(self._run_identity["config_fingerprint"])
        )
        if not identity_matches or self.config is None:
            return False
        expected_selection = list(self.config.training.development_participant_ids)
        if payload.get("selection_participants") != expected_selection:
            return False
        channels = payload.get("channels", {})
        if not isinstance(channels, Mapping) or set(channels) != set(PPG_CHANNELS):
            return False
        fingerprint = str(payload.get("selection_fingerprint", ""))
        fingerprint_payload = dict(payload)
        fingerprint_payload.pop("selection_fingerprint", None)
        expected_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if fingerprint != expected_fingerprint:
            return False
        terminal_statuses = {
            "COVERAGE_BLOCKED",
            "FROZEN_CLASSICAL",
            "TERMINAL_NO_GO",
        }
        reports_root = Path(self.config.artifacts.get("reports_dir", "reports"))
        if not reports_root.is_absolute():
            reports_root = self.paths.project_root / reports_root
        expected_report_root = reports_root / "model_eval" / self.config.model_version
        for channel, evidence in channels.items():
            if not isinstance(evidence, Mapping):
                return False
            terminal_status = str(evidence.get("terminal_status", ""))
            if terminal_status not in terminal_statuses or evidence.get("requires_dl"):
                return False
            report_path = Path(str(evidence.get("report_path", "")))
            expected_report_path = expected_report_root / f"{channel}_full.json"
            if report_path.resolve() != expected_report_path.resolve():
                return False
            if not report_path.is_file():
                return False
            if self._file_sha256(report_path) != str(evidence.get("report_sha256", "")):
                return False
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if (
                str(report.get("run_id", "")) != str(self._run_identity["run_id"])
                or str(report.get("source_tree_sha256", ""))
                != str(self._run_identity["source_tree_sha256"])
                or str(report.get("config_fingerprint", ""))
                != str(self._run_identity["config_fingerprint"])
            ):
                return False
            if (
                terminal_status == "FROZEN_CLASSICAL"
                and re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(evidence.get("model_contract_fingerprint", "")),
                )
                is None
            ):
                return False
            if terminal_status == "COVERAGE_BLOCKED" and (
                not str(evidence.get("training_admission_status", "")).startswith(
                    "blocked_"
                )
                or str(evidence.get("release_status", ""))
                not in {
                    "blocked_insufficient_ppg_coverage",
                    "blocked_hrv_prv_admission",
                }
            ):
                return False
        return True

    def _physiological_governance_ledger_path(self) -> Path:
        """Return the append-only ledger for versioned full-cohort access."""

        return (
            self.paths.project_root
            / "data"
            / "artifacts"
            / "governance"
            / "physiological_full_cohort_access_v2.json"
        )

    def _claim_physiological_access(self, access_fingerprint: str) -> Path:
        """Claim one frozen full-cohort contract without touching legacy ledger."""

        ledger_path = self._physiological_governance_ledger_path()
        if ledger_path.is_file():
            try:
                payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "physiological_full_cohort_governance_ledger_invalid"
                ) from exc
            accesses = payload.get("accesses")
            if not isinstance(accesses, list):
                raise RuntimeError(
                    "physiological_full_cohort_governance_ledger_invalid"
                )
        else:
            payload = {
                "schema_version": "physiological-full-cohort-access-v2",
                "accesses": [],
            }
            accesses = payload["accesses"]

        if any(
            str(item.get("access_fingerprint", "")) == access_fingerprint
            for item in accesses
            if isinstance(item, Mapping)
        ):
            raise RuntimeError("physiological_full_cohort_contract_already_accessed")

        accesses.append(
            {
                **self._run_identity,
                "status": "started",
                "access_fingerprint": access_fingerprint,
                "selection_participants": sorted(DEVELOPMENT_PARTICIPANTS),
                "physiological_validation_participants": sorted(VALID_PARTICIPANTS),
                "primary_domain": "time_domain",
            }
        )
        self._write_json(ledger_path, payload)
        return ledger_path

    def _complete_physiological_access(
        self,
        ledger_path: Path,
        *,
        access_fingerprint: str,
        evaluated_rows: int,
        evidence_path: Path,
    ) -> None:
        """Complete the claimed ledger record after evidence is durable."""

        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        accesses = payload.get("accesses")
        if not isinstance(accesses, list):
            raise RuntimeError("physiological_full_cohort_governance_ledger_invalid")
        for item in accesses:
            if (
                isinstance(item, dict)
                and item.get("access_fingerprint") == access_fingerprint
            ):
                item.update(
                    {
                        "status": "completed",
                        "evaluated_rows": int(evaluated_rows),
                        "physiological_validation_sha256": self._file_sha256(
                            evidence_path
                        ),
                    }
                )
                self._write_json(ledger_path, payload)
                return
        raise RuntimeError("physiological_full_cohort_governance_claim_missing")

    def _physiological_access_fingerprint(self) -> str:
        detector_path = (
            self.paths.validation_dir / "detector_selection" / "detector_selection.json"
        )
        stress_path = self.paths.models_dir / "stress_selection_contract.json"
        detector = json.loads(detector_path.read_text(encoding="utf-8"))
        stress = json.loads(stress_path.read_text(encoding="utf-8"))
        selected_ranges = {
            channel: detector.get("selected", {})
            .get(channel, {})
            .get("ecg_reference_range_fingerprint")
            for channel in PPG_CHANNELS
        }
        payload = {
            "source_tree_sha256": self._run_identity["source_tree_sha256"],
            "config_fingerprint": self._run_identity["config_fingerprint"],
            "detector_selection_sha256": self._file_sha256(detector_path),
            "stress_selection_fingerprint": stress.get("selection_fingerprint"),
            "ecg_reference_range_fingerprints": selected_ranges,
            "hrv_policy": repr(self.config.hrv_agreement) if self.config else None,
            "selection_participants": sorted(DEVELOPMENT_PARTICIPANTS),
            "validation_participants": sorted(VALID_PARTICIPANTS),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _config_fingerprint(self) -> str:
        """Hash only the active peak-validation policy and SQA configuration."""

        payload = {
            "peak_detection": repr(self.config.peak_detection)
            if self.config is not None
            else "default",
            "synchronization": repr(self.config.synchronization)
            if self.config is not None
            else "default",
            "hrv_agreement": repr(self.config.hrv_agreement)
            if self.config is not None
            else "default",
            "sqa": self._sqa_config,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_run_identity(self) -> dict[str, Any]:
        """Load the current run identity without accepting stale artifacts."""

        expected_run_id = str(
            self.config.artifacts.get("run_id", "") if self.config is not None else ""
        )
        contract_path = self.paths.artifacts_root / "run_contract_fingerprint.json"
        identity: dict[str, Any] = {
            "run_id": expected_run_id or "UNSCOPED",
            "source_tree_sha256": "UNAVAILABLE",
            "run_contract_sha256": "UNAVAILABLE",
            "config_fingerprint": "UNAVAILABLE",
            "status": "not_scoped" if not expected_run_id else "UNAVAILABLE",
            "strict_scope": bool(expected_run_id),
        }
        if not contract_path.is_file():
            return identity
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            identity["status"] = "INVALID"
            return identity
        contract_run_id = str(payload.get("run_id", ""))
        if expected_run_id and contract_run_id != expected_run_id:
            identity["status"] = "MISMATCH"
            return identity
        identity.update(
            {
                "run_id": contract_run_id or expected_run_id or "UNSCOPED",
                "source_tree_sha256": str(
                    payload.get("source_tree_sha256", "UNAVAILABLE")
                ),
                "config_fingerprint": str(
                    payload.get("config_fingerprint", "UNAVAILABLE")
                ),
                "run_contract_sha256": self._file_sha256(contract_path),
                "status": "ready",
            }
        )
        current_config_fingerprint = hashlib.sha256(
            json.dumps(
                asdict(self.config),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if identity["config_fingerprint"] != current_config_fingerprint:
            identity["status"] = "CONFIG_MISMATCH"
        return identity

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _find_session(self, participant_id: str, phase: str) -> Path | None:
        """Resolve only the current run's canonical Silver recording."""

        candidate = self.silver_dir / participant_id / phase / "ecg.parquet"
        if candidate.is_file():
            return candidate.parent
        # Do not scan other phases, runs, or legacy trees.  A missing exact
        # path is an explicit exclusion, not permission to guess a source.
        return None

    def _preferred_ppg_path(
        self, participant_id: str, phase: str, channel: str
    ) -> Path | None:
        """Return only the canonical Silver PPG artifact.

        The ECG/PPG validator must agree with Gold.  Falling back to a raw or
        synchronized PPG artifact would silently create a second preprocessing
        path and invalidate the comparison.
        """

        silver_candidate = (
            self.silver_dir / participant_id / phase / f"{channel}.parquet"
        )
        return silver_candidate if silver_candidate.is_file() else None

    @staticmethod
    def _recording_id(
        ecg: pd.DataFrame, participant_id: str, phase: str, session_dir: Path
    ) -> str:
        if "recording_id" in ecg.columns and not ecg["recording_id"].dropna().empty:
            return str(ecg["recording_id"].dropna().iloc[0])
        if session_dir.name in {"relax", "task"}:
            return f"{participant_id}_{phase}_raw"
        return session_dir.name

    @staticmethod
    def _phase_from_path(path: Path) -> str | None:
        text = str(path).lower().replace("\\", "/")
        if re.search(r"(?:^|/|_)relax(?:_|/|$)", text) or "/relaks/" in text:
            return "relax"
        if re.search(r"(?:^|/|_)task(?:_|/|$)", text) or "/stress/" in text:
            return "task"
        return None

    @staticmethod
    def _column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
        for name in names:
            if name in frame.columns:
                return name
        raise KeyError(f"missing signal column; expected one of {names}")

    def _prepare_frame(self, frame: pd.DataFrame, channel: str) -> PreparedSignal:
        timestamp_column = self._column(
            frame,
            (
                "timestamp_corrected_utc",
                "timestamp_utc",
                "timestamp_raw_utc",
                "unix_ms",
            ),
        )
        if channel != "ecg" and "clean_value" not in frame.columns:
            raise KeyError("silver_missing_clean_value")
        value_names = (
            ("clean_value", "raw_value", "ecg_mv")
            if channel == "ecg"
            else ("clean_value",)
        )
        value_column = self._column(frame, value_names)
        timestamps = self._timestamps_ms(frame[timestamp_column])
        values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(
            dtype=float
        )
        has_raw_indices = "raw_sample_index" in frame.columns
        raw_indices = (
            pd.to_numeric(frame["raw_sample_index"], errors="coerce").to_numpy(
                dtype=float
            )
            if has_raw_indices
            else np.arange(len(frame), dtype=float)
        )
        interpolated = self._boolean_column(frame, "interpolated_mask", len(frame))
        explicit_gap = self._boolean_column(frame, "gap_mask", len(frame))
        unresolved = self._boolean_column(frame, "unresolved_mask", len(frame))
        motion_heavy = (
            self._boolean_column(frame, "motion_heavy_mask", len(frame))
            if channel != "ecg"
            else np.zeros(len(frame), dtype=bool)
        )
        keep = np.isfinite(timestamps) & np.isfinite(raw_indices)
        (
            timestamps,
            values,
            raw_indices,
            interpolated,
            explicit_gap,
            unresolved,
            motion_heavy,
        ) = (
            timestamps[keep],
            values[keep],
            raw_indices[keep],
            interpolated[keep],
            explicit_gap[keep],
            unresolved[keep],
            motion_heavy[keep],
        )
        order = np.argsort(
            raw_indices if has_raw_indices else timestamps, kind="stable"
        )
        (
            timestamps,
            values,
            raw_indices,
            interpolated,
            explicit_gap,
            unresolved,
            motion_heavy,
        ) = (
            timestamps[order],
            values[order],
            raw_indices[order],
            interpolated[order],
            explicit_gap[order],
            unresolved[order],
            motion_heavy[order],
        )
        if has_raw_indices and np.any(np.diff(raw_indices) <= 0.0):
            raise ValueError("raw_sample_index_must_be_strictly_increasing")
        if not has_raw_indices:
            unique = np.concatenate(([True], np.diff(timestamps) > 0.0))
            (
                timestamps,
                values,
                raw_indices,
                interpolated,
                explicit_gap,
                unresolved,
                motion_heavy,
            ) = (
                timestamps[unique],
                values[unique],
                raw_indices[unique],
                interpolated[unique],
                explicit_gap[unique],
                unresolved[unique],
                motion_heavy[unique],
            )
        if len(timestamps) < 3:
            raise ValueError("signal_too_short")
        fs = self._sampling_rate(frame, timestamps)
        if not np.isfinite(values).any():
            raise ValueError("signal_all_nan")
        expected_interval_ms = 1000.0 / fs
        gap_mask, gap_events, timestamp_jitter_count, raw_index_gap_count = (
            self._audit_gaps(
                timestamps,
                values,
                expected_interval_ms,
                self._max_unfilled_gap_ms(),
                channel,
                raw_sample_indices=raw_indices if has_raw_indices else None,
                explicit_gap_mask=explicit_gap | unresolved,
            )
        )
        if motion_heavy.any():
            gap_mask |= motion_heavy
            motion_events: list[GapEvent] = []
            starts, ends = self._finite_runs(motion_heavy)
            for start, end in zip(starts, ends, strict=True):
                event_end_ms = float(timestamps[end - 1] + expected_interval_ms)
                motion_events.append(
                    GapEvent(
                        float(timestamps[start]),
                        event_end_ms,
                        float(event_end_ms - timestamps[start]),
                        int(end - start),
                        "motion_heavy",
                        "mask_and_exclude_affected_beats",
                        True,
                    )
                )
            gap_events = gap_events + tuple(motion_events)
        # An ECG interpolation flag from upstream is never trusted as a
        # primary reference.  PPG interpolation is canonical and remains
        # available as provenance for downstream sensitivity analysis.
        if channel == "ecg" and interpolated.any():
            gap_mask |= interpolated
            gap_events = gap_events + tuple(
                GapEvent(
                    float(timestamps[index]),
                    float(timestamps[index]),
                    0.0,
                    1,
                    "upstream_interpolated_sample",
                    "mask_and_exclude_affected_beats",
                    True,
                )
                for index in np.flatnonzero(interpolated)
            )
        long_gap = any(
            event.gap_type in {"long_dropout", "edge_gap"} for event in gap_events
        )
        short_gap = channel != "ecg" and bool(interpolated.any())
        return PreparedSignal(
            timestamps,
            values,
            fs,
            gap_mask,
            interpolated,
            gap_events,
            long_gap,
            short_gap,
            raw_indices.astype(np.int64, copy=False) if has_raw_indices else None,
            timestamp_jitter_count,
            raw_index_gap_count,
        )

    @staticmethod
    def _boolean_column(frame: pd.DataFrame, name: str, length: int) -> np.ndarray:
        if name not in frame.columns:
            return np.zeros(length, dtype=bool)
        values = frame[name]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).to_numpy(dtype=bool)
        numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
        return numeric.to_numpy(dtype=float) != 0.0

    @classmethod
    def _interpolation_mask_provenance(cls, frame: pd.DataFrame) -> dict[str, Any]:
        """Summarize Silver interpolation masks without reconstructing samples."""

        length = len(frame)
        interpolated = cls._boolean_column(frame, "interpolated_mask", length)
        gap = cls._boolean_column(frame, "gap_mask", length)
        unresolved = cls._boolean_column(frame, "unresolved_mask", length)
        explicit = all(
            column in frame.columns
            for column in (
                "observed_mask",
                "interpolated_mask",
                "gap_mask",
                "unresolved_mask",
            )
        )
        if explicit:
            observed = cls._boolean_column(frame, "observed_mask", length)
        else:
            value_column = "clean_value" if "clean_value" in frame else "raw_value"
            finite = (
                np.isfinite(
                    pd.to_numeric(frame[value_column], errors="coerce").to_numpy(
                        dtype=float
                    )
                )
                if value_column in frame
                else np.zeros(length, dtype=bool)
            )
            unresolved |= ~finite
            observed = finite & ~interpolated & ~gap & ~unresolved

        partition_count = (
            observed.astype(np.uint8)
            + interpolated.astype(np.uint8)
            + gap.astype(np.uint8)
            + unresolved.astype(np.uint8)
        )
        valid_partition = bool(np.all(partition_count == 1))
        mask_bytes = np.stack((observed, interpolated, gap, unresolved), axis=0).astype(
            np.uint8, copy=False
        )

        methods: list[str] = []
        if "interpolation_provenance" in frame:
            methods = sorted(
                {
                    str(value).strip()
                    for value in frame["interpolation_provenance"].dropna().unique()
                    if str(value).strip()
                }
            )
        method = "none" if not methods else methods[0] if len(methods) == 1 else "mixed"
        return {
            "ppg_observed_sample_count": int(observed.sum()),
            "ppg_gap_provenance_sample_count": int(gap.sum()),
            "ppg_unresolved_sample_count": int(unresolved.sum()),
            "ppg_interpolation_method": method,
            "ppg_interpolation_mask_sha256": hashlib.sha256(
                mask_bytes.tobytes()
            ).hexdigest(),
            "ppg_interpolation_mask_status": (
                "explicit_complete"
                if explicit and valid_partition
                else "inferred_legacy"
                if valid_partition
                else "invalid_partition"
            ),
        }

    @staticmethod
    def _audit_gaps(
        timestamps_ms: np.ndarray,
        values: np.ndarray,
        expected_interval_ms: float,
        max_gap_ms: float,
        channel: str,
        *,
        raw_sample_indices: np.ndarray | None = None,
        explicit_gap_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[GapEvent, ...], int, int]:
        """Audit observed gaps without filling any ECG or PPG sample."""

        gap_mask = ~np.isfinite(values)
        if explicit_gap_mask is not None:
            explicit = np.asarray(explicit_gap_mask, dtype=bool)
            if explicit.shape != gap_mask.shape:
                raise ValueError("explicit_gap_mask_shape_mismatch")
            gap_mask |= explicit
        events: list[GapEvent] = []
        timestamp_jitter_count = 0
        raw_index_gap_count = 0
        action = (
            "mask_and_exclude_affected_beats"
            if channel == "ecg"
            else "preserve_canonical_mask"
        )
        missing = gap_mask.copy()
        missing_starts, missing_ends = PpgEcgValidator._finite_runs(missing)
        for start, end in zip(missing_starts, missing_ends, strict=True):
            if start == 0 or end >= len(values):
                gap_mask[max(0, start - 1) : min(len(values), end + 1)] = True
                events.append(
                    GapEvent(
                        float(timestamps_ms[start]),
                        float(timestamps_ms[end - 1]),
                        float("inf"),
                        int(end - start),
                        "edge_gap",
                        action,
                        True,
                    )
                )
                continue
            gap_duration = max(
                0.0,
                float(timestamps_ms[end] - timestamps_ms[start - 1])
                - expected_interval_ms,
            )
            gap_mask[start - 1 : end + 1] = True
            events.append(
                GapEvent(
                    float(timestamps_ms[start - 1]),
                    float(timestamps_ms[end]),
                    gap_duration,
                    int(end - start),
                    "long_dropout" if gap_duration > max_gap_ms else "short_gap",
                    action,
                    True,
                )
            )

        for index, interval in enumerate(np.diff(timestamps_ms)):
            gap_duration = max(0.0, float(interval - expected_interval_ms))
            if gap_duration <= max(expected_interval_ms * 0.5, 1.0):
                continue
            if raw_sample_indices is not None:
                raw_delta = float(
                    raw_sample_indices[index + 1] - raw_sample_indices[index]
                )
                if np.isclose(raw_delta, 1.0):
                    timestamp_jitter_count += 1
                    continue
                if raw_delta <= 0.0 or not np.isclose(raw_delta, round(raw_delta)):
                    raise ValueError("invalid_raw_sample_index_delta")
                raw_index_gap_count += int(round(raw_delta)) - 1
            gap_mask[index : index + 2] = True
            events.append(
                GapEvent(
                    float(timestamps_ms[index]),
                    float(timestamps_ms[index + 1]),
                    gap_duration,
                    max(
                        0,
                        int(
                            round(
                                raw_sample_indices[index + 1]
                                - raw_sample_indices[index]
                            )
                        )
                        - 1,
                    )
                    if raw_sample_indices is not None
                    else 0,
                    "long_dropout" if gap_duration > max_gap_ms else "short_gap",
                    action,
                    True,
                )
            )
        return gap_mask, tuple(events), timestamp_jitter_count, raw_index_gap_count

    @staticmethod
    def _finite_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        padded = np.concatenate(([False], mask, [False])).astype(np.int8)
        changes = np.diff(padded)
        return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)

    @staticmethod
    def _timestamps_ms(values: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(values):
            return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
        result = np.full(len(parsed), np.nan, dtype=float)
        valid = parsed.notna().to_numpy()
        result[valid] = (
            parsed[valid].astype("int64").to_numpy(dtype=float) / 1_000_000.0
        )
        return result

    @staticmethod
    def _sampling_rate(frame: pd.DataFrame, timestamps_ms: np.ndarray) -> float:
        intervals = np.diff(timestamps_ms)
        valid = intervals[intervals > 0.0]
        if valid.size:
            effective = float(1000.0 / np.median(valid))
            if np.isfinite(effective) and effective > 0.0:
                return effective
        if "sampling_rate_hz" in frame.columns:
            declared = pd.to_numeric(
                frame["sampling_rate_hz"], errors="coerce"
            ).dropna()
            if not declared.empty and float(declared.median()) > 0.0:
                return float(declared.median())
        raise ValueError("sampling_rate_unavailable")

    @staticmethod
    def _coerce_prepared(
        prepared_or_timestamps: PreparedSignal | np.ndarray,
        values: np.ndarray | None = None,
        fs_hz: float | None = None,
        long_gap_detected: bool = False,
        short_gap_interpolated: bool = False,
    ) -> PreparedSignal:
        """Keep the former internal detector call shape source-compatible."""

        if isinstance(prepared_or_timestamps, PreparedSignal):
            return prepared_or_timestamps
        if values is None or fs_hz is None:
            raise TypeError("values and fs_hz are required for array detector input")
        timestamps = np.asarray(prepared_or_timestamps, dtype=float)
        signal = np.asarray(values, dtype=float)
        if timestamps.shape != signal.shape:
            raise ValueError("timestamps_and_values_shape_mismatch")
        return PreparedSignal(
            timestamps,
            signal,
            float(fs_hz),
            np.zeros(signal.shape, dtype=bool),
            np.zeros(signal.shape, dtype=bool),
            (),
            long_gap_detected,
            short_gap_interpolated,
        )

    @staticmethod
    def _detection_segments(prepared: PreparedSignal) -> list[tuple[int, int]]:
        valid = np.isfinite(prepared.values) & ~prepared.gap_mask
        # ``gap_mask`` marks both sides of timestamp discontinuities and all
        # NaN runs, so each finite run is safe to process independently.
        starts, ends = PpgEcgValidator._finite_runs(valid)
        return [
            (start, end)
            for start, end in zip(starts, ends, strict=True)
            if end - start >= 3
        ]

    def _max_unfilled_gap_ms(self) -> float:
        if self.config is None:
            return 500.0
        return float(self.config.synchronization.ecg_max_unfilled_gap_seconds) * 1000.0

    @staticmethod
    def _gap_seconds(events: tuple[GapEvent, ...]) -> float:
        finite = [
            event.duration_ms for event in events if np.isfinite(event.duration_ms)
        ]
        return float(np.sum(finite) / 1000.0) if finite else 0.0

    @staticmethod
    def _mask_count(mask: np.ndarray | None) -> int:
        return int(np.count_nonzero(mask)) if mask is not None else 0

    @staticmethod
    def _peak_gap_count(peaks: DetectedPeaks) -> int:
        if peaks.gap_mask is None or len(peaks.sample_indices) == 0:
            return 0
        valid_indices = peaks.sample_indices[
            (peaks.sample_indices >= 0) & (peaks.sample_indices < len(peaks.gap_mask))
        ]
        return int(np.count_nonzero(peaks.gap_mask[valid_indices]))

    def _min_peak_count(self) -> int:
        return int(self.config.synchronization.min_peak_count) if self.config else 30

    def _min_match_count(self) -> int:
        return int(self.config.synchronization.min_match_count) if self.config else 30

    def _pat_tolerance_ms(self) -> float:
        return (
            float(self.config.synchronization.pat_tolerance_ms)
            if self.config
            else DEFAULT_PAT_TOLERANCE_MS
        )

    @staticmethod
    def _detect_ecg(
        prepared_or_timestamps: PreparedSignal | np.ndarray,
        values: np.ndarray | None = None,
        fs_hz: float | None = None,
        long_gap_detected: bool = False,
        short_gap_interpolated: bool = False,
        *,
        config_fingerprint: str = "",
    ) -> DetectedPeaks:
        prepared = PpgEcgValidator._coerce_prepared(
            prepared_or_timestamps,
            values,
            fs_hz,
            long_gap_detected,
            short_gap_interpolated,
        )
        clean_signal = np.full(prepared.values.shape, np.nan, dtype=float)
        peak_indices: list[int] = []
        methods: list[str] = []
        library_versions: list[str] = []
        fallback_reasons: list[str] = []
        segment_reference_statuses: list[str] = []
        for start, end in PpgEcgValidator._detection_segments(prepared):
            clean_input = prepared.values[start:end]
            method = "neurokit2_ecg_peaks+pantompkins1985_consensus"
            neurokit_indices = np.asarray([], dtype=int)
            pantompkins_indices = np.asarray([], dtype=int)
            try:
                # Silver already contains the single canonical ECG filter.
                # Running ecg_clean/ecg_process here would filter it twice and
                # could alter QRS timing.  Peak detection therefore consumes
                # the canonical clean segment unchanged.
                clean = np.asarray(clean_input, dtype=float)
                _, info = nk.ecg_peaks(
                    clean,
                    sampling_rate=max(1, int(round(prepared.fs_hz))),
                    method="neurokit",
                )
                neurokit_indices = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            except Exception as exc:
                clean = np.asarray(clean_input, dtype=float)
                fallback_reasons.append(f"ecg_neurokit:{type(exc).__name__}:{exc}")
            try:
                _, info = nk.ecg_peaks(
                    clean,
                    sampling_rate=max(1, int(round(prepared.fs_hz))),
                    method="pantompkins1985",
                )
                pantompkins_indices = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            except Exception as exc:
                fallback_reasons.append(f"ecg_pantompkins:{type(exc).__name__}:{exc}")
            tolerance_samples = max(1, int(round(prepared.fs_hz * 0.150)))
            if neurokit_indices.size and pantompkins_indices.size:
                local_indices = PpgEcgValidator._consensus_indices(
                    neurokit_indices,
                    pantompkins_indices,
                    tolerance_samples,
                )
                agreement_fraction = local_indices.size / max(
                    1, min(neurokit_indices.size, pantompkins_indices.size)
                )
                if local_indices.size < 3:
                    local_indices = neurokit_indices
                    segment_reference_statuses.append("REFERENCE_UNCERTAIN")
                    fallback_reasons.append("ecg_detector_disagreement")
                elif agreement_fraction < 0.80:
                    segment_reference_statuses.append("REFERENCE_UNCERTAIN")
                    fallback_reasons.append(
                        f"ecg_detector_consensus_fraction={agreement_fraction:.3f}"
                    )
                else:
                    segment_reference_statuses.append("CONSENSUS")
            elif neurokit_indices.size:
                local_indices = neurokit_indices
                segment_reference_statuses.append("REFERENCE_UNCERTAIN")
                fallback_reasons.append("ecg_pantompkins_unavailable")
            elif pantompkins_indices.size:
                local_indices = pantompkins_indices
                segment_reference_statuses.append("REFERENCE_UNCERTAIN")
                fallback_reasons.append("ecg_neurokit_unavailable")
            else:
                # Fallback is detector-only. It must not clean or interpolate
                # the ECG waveform a second time.
                method = "scipy_find_peaks_fallback"
                segment_reference_statuses.append("REFERENCE_UNCERTAIN")
                minimum_distance = max(1, int(round(prepared.fs_hz * 0.30)))
                local_indices, _ = find_peaks(
                    clean,
                    distance=minimum_distance,
                    prominence=max(
                        np.finfo(float).eps,
                        float(np.nanstd(clean)) * 0.25,
                    ),
                )
                fallback_reasons.append("ecg_dual_detector_unavailable")
            length = min(end - start, len(clean))
            clean_signal[start : start + length] = clean[:length]
            local_indices = local_indices[
                (local_indices >= 0) & (local_indices < end - start)
            ]
            global_indices = start + local_indices
            global_indices = global_indices[~prepared.gap_mask[global_indices]]
            peak_indices.extend(global_indices.tolist())
            methods.append(method)
            library_versions.append(
                _package_version("neurokit2")
                if method != "scipy_find_peaks_fallback"
                else _package_version("scipy")
            )
        indices = np.asarray(sorted(set(peak_indices)), dtype=int)
        if not segment_reference_statuses:
            reference_status = "UNAVAILABLE"
        elif all(status == "CONSENSUS" for status in segment_reference_statuses):
            reference_status = "CONSENSUS"
        else:
            reference_status = "REFERENCE_UNCERTAIN"
        return DetectedPeaks(
            prepared.timestamps_ms[indices],
            clean_signal,
            indices,
            prepared.fs_hz,
            prepared.long_gap_detected,
            prepared.short_gap_interpolated,
            "+".join(dict.fromkeys(methods)),
            ";".join(fallback_reasons),
            prepared.gap_mask,
            prepared.gap_events,
            int(np.count_nonzero(prepared.interpolated_mask)),
            None,
            reference_status,
            library_version=";".join(dict.fromkeys(library_versions)) or "unknown",
            config_fingerprint=config_fingerprint,
            status=(
                "UNAVAILABLE" if reference_status == "UNAVAILABLE" else "AVAILABLE"
            ),
            timestamp_jitter_count=prepared.timestamp_jitter_count,
            raw_index_gap_count=prepared.raw_index_gap_count,
        )

    @staticmethod
    def _consensus_indices(
        first: np.ndarray,
        second: np.ndarray,
        tolerance_samples: int,
    ) -> np.ndarray:
        """Return first-detector indices with a monotonic one-to-one consensus."""

        left = np.asarray(first, dtype=int)
        right = np.asarray(second, dtype=int)
        if left.size == 0 or right.size == 0:
            return np.asarray([], dtype=int)
        left = np.unique(left)
        right = np.unique(right)
        matches: list[int] = []
        right_index = 0
        for index in left:
            while (
                right_index < right.size
                and right[right_index] < index - tolerance_samples
            ):
                right_index += 1
            if right_index >= right.size:
                break
            if abs(int(right[right_index]) - int(index)) <= tolerance_samples:
                matches.append(int(index))
                right_index += 1
        return np.asarray(matches, dtype=int)

    def _detect_ppg(
        self,
        prepared_or_timestamps: PreparedSignal | np.ndarray,
        values: np.ndarray | None = None,
        fs_hz: float | None = None,
        long_gap_detected: bool = False,
        short_gap_interpolated: bool = False,
        channel: str = "ppg_green",
        *,
        config_fingerprint: str = "",
    ) -> DetectedPeaks:
        prepared = PpgEcgValidator._coerce_prepared(
            prepared_or_timestamps,
            values,
            fs_hz,
            long_gap_detected,
            short_gap_interpolated,
        )
        peak_indices: list[int] = []
        trough_indices: list[int] = []
        methods: list[str] = []
        library_versions: list[str] = []
        fallback_reasons: list[str] = []
        detector_unavailable = False
        for start, end in PpgEcgValidator._detection_segments(prepared):
            clean = prepared.values[start:end]
            if self._ppg_peak_detector is None:
                method = "neurokit2_ppg_elgendi"
                try:
                    info = nk.ppg_findpeaks(
                        clean,
                        sampling_rate=max(1, int(round(prepared.fs_hz))),
                        method="elgendi",
                    )
                    local_indices = np.asarray(info.get("PPG_Peaks", []), dtype=int)
                    if len(local_indices) < 3:
                        raise ValueError("neurokit_ppg_too_few_peaks")
                    local_troughs, _ = find_peaks(
                        -np.asarray(clean, dtype=float),
                        distance=max(1, int(round(0.30 * prepared.fs_hz))),
                    )
                    library_versions.append(_package_version("neurokit2"))
                except Exception as exc:
                    method = "scipy_find_peaks_fallback"
                    library_versions.append(_package_version("scipy"))
                    fallback_reasons.append(f"ppg_elgendi:{type(exc).__name__}:{exc}")
                    distance = max(1, int(round(0.30 * prepared.fs_hz)))
                    prominence = max(
                        float(np.nanstd(clean) * 0.10), np.finfo(float).eps
                    )
                    local_indices, _ = find_peaks(
                        clean, distance=distance, prominence=prominence
                    )
                    local_troughs, _ = find_peaks(
                        -np.asarray(clean, dtype=float),
                        distance=distance,
                        prominence=prominence,
                    )
            else:
                minimum_samples = self._ppg_peak_detector.minimum_sample_count(
                    prepared.fs_hz
                )
                if clean.size < minimum_samples:
                    fallback_reasons.append(
                        "segment_skipped_too_short:"
                        f"samples={clean.size};minimum={minimum_samples}"
                    )
                    continue
                detection = self._ppg_peak_detector.detect(
                    clean,
                    prepared.fs_hz,
                    channel,
                )
                method = detection.method
                local_indices = detection.peaks
                local_troughs = detection.troughs
                library_versions.append(detection.library_version)
                if detection.fallback_reason:
                    fallback_reasons.append(detection.fallback_reason)
                if detection.status == "UNAVAILABLE":
                    detector_unavailable = True
                    fallback_reasons.append(
                        f"{detection.method}:UNAVAILABLE:{detection.status_reason}"
                    )
            local_indices = local_indices[
                (local_indices >= 0) & (local_indices < end - start)
            ]
            global_indices = start + local_indices
            global_indices = global_indices[~prepared.gap_mask[global_indices]]
            peak_indices.extend(global_indices.tolist())
            global_troughs = start + np.asarray(local_troughs, dtype=int)
            global_troughs = global_troughs[
                (global_troughs >= start)
                & (global_troughs < end)
                & ~prepared.gap_mask[global_troughs]
            ]
            trough_indices.extend(global_troughs.tolist())
            methods.append(method)
        indices = np.asarray(sorted(set(peak_indices)), dtype=int)
        return DetectedPeaks(
            prepared.timestamps_ms[indices],
            prepared.values.copy(),
            indices,
            prepared.fs_hz,
            prepared.long_gap_detected,
            prepared.short_gap_interpolated,
            "+".join(dict.fromkeys(methods)),
            ";".join(fallback_reasons),
            prepared.gap_mask,
            prepared.gap_events,
            int(np.count_nonzero(prepared.interpolated_mask)),
            np.asarray(sorted(set(trough_indices)), dtype=int),
            reference_status=(
                "UNAVAILABLE"
                if detector_unavailable or not methods or not peak_indices
                else "AVAILABLE"
            ),
            library_version=";".join(dict.fromkeys(library_versions)) or "unknown",
            config_fingerprint=config_fingerprint,
            status=(
                "UNAVAILABLE"
                if detector_unavailable and not peak_indices
                else "AVAILABLE"
            ),
            timestamp_jitter_count=prepared.timestamp_jitter_count,
            raw_index_gap_count=prepared.raw_index_gap_count,
        )

    def _match_with_pat(
        self, ecg_ms: np.ndarray, ppg_ms: np.ndarray
    ) -> tuple[list[tuple[int, int, float]], float]:
        result = self._match_result(ecg_ms, ppg_ms)
        if result.pat_median_ms is None:
            pat_min = self.config.synchronization.pat_min_ms if self.config else 100.0
            pat_max = self.config.synchronization.pat_max_ms if self.config else 400.0
            candidate_delays = self._initial_pat_delays(
                np.asarray(ecg_ms, dtype=float),
                np.asarray(ppg_ms, dtype=float),
                pat_min,
                pat_max,
            )
            legacy_pat = (
                float(np.median(candidate_delays)) if candidate_delays else float("nan")
            )
        else:
            legacy_pat = result.pat_median_ms
        return list(result.matches), (
            legacy_pat if legacy_pat is not None else float("nan")
        )

    def _match_result(self, ecg_ms: np.ndarray, ppg_ms: np.ndarray) -> MatchResult:
        """Match observed peaks monotonically without waveform time warping."""

        ecg = np.asarray(ecg_ms, dtype=float)
        ppg = np.asarray(ppg_ms, dtype=float)
        if (
            ecg.ndim != 1
            or ppg.ndim != 1
            or not np.isfinite(ecg).all()
            or not np.isfinite(ppg).all()
            or np.any(np.diff(ecg) <= 0.0)
            or np.any(np.diff(ppg) <= 0.0)
        ):
            return MatchResult(
                (),
                None,
                None,
                None,
                0.0,
                int(ecg.size),
                int(ppg.size),
                "monotonic_pat",
                "peak_timestamps_must_be_finite_and_sorted",
            )
        if not ecg.size or not ppg.size:
            return MatchResult(
                (),
                None,
                None,
                None,
                0.0,
                int(ecg.size),
                int(ppg.size),
                "monotonic_pat",
                "empty_peak_sequence",
            )

        pat_min = self.config.synchronization.pat_min_ms if self.config else 100.0
        pat_max = self.config.synchronization.pat_max_ms if self.config else 400.0
        tolerance = self._pat_tolerance_ms()
        initial_delays = self._initial_pat_delays(ecg, ppg, pat_min, pat_max)
        if not initial_delays:
            return MatchResult(
                (),
                None,
                None,
                None,
                0.0,
                int(ecg.size),
                int(ppg.size),
                "monotonic_pat",
                "no_physiological_pat_candidates",
            )
        initial_pat = float(np.median(initial_delays))
        constant_model = (initial_pat, 0.0, float(ecg[0]))
        constant = self._greedy_pat_matches(
            ecg, ppg, constant_model, pat_min, pat_max, tolerance
        )
        # A residual drift model needs enough paired observations to be
        # identifiable.  With fewer than four constant-model matches, keep
        # the baseline result and do not promote an accidental single-beat
        # affine fit.
        if len(constant) >= 4:
            model = self._fit_pat_model(ecg, ppg, constant, tolerance, pat_min, pat_max)
            affine = self._greedy_pat_matches(
                ecg, ppg, model, pat_min, pat_max, tolerance
            )
        else:
            model = constant_model
            affine = []
        chosen = affine
        if len(constant) > len(affine):
            chosen = constant
            model = constant_model
        elif len(constant) == len(affine):
            constant_cost = self._match_cost(constant, constant_model, ecg)
            affine_cost = self._match_cost(affine, model, ecg)
            if constant_cost <= affine_cost:
                chosen = constant
                model = constant_model

        delays = np.asarray([delay for _, _, delay in chosen], dtype=float)
        pat_median = float(np.median(delays)) if delays.size else None
        pat_iqr = (
            float(np.percentile(delays, 75) - np.percentile(delays, 25))
            if delays.size
            else None
        )
        pat_mad = (
            float(np.median(np.abs(delays - pat_median)))
            if delays.size and pat_median is not None
            else None
        )
        matched_ecg_indices = {ecg_index for ecg_index, _, _ in chosen}
        matched_ppg_indices = {ppg_index for _, ppg_index, _ in chosen}
        return MatchResult(
            tuple(chosen),
            pat_median,
            pat_iqr,
            pat_mad,
            float(model[1]),
            int(ecg.size - len(chosen)),
            int(ppg.size - len({ppg_index for _, ppg_index, _ in chosen})),
            "monotonic_pat_affine" if abs(model[1]) > 0.0 else "monotonic_pat_constant",
            None if chosen else "no_matches_within_pat_tolerance",
            tuple(
                index for index in range(ecg.size) if index not in matched_ecg_indices
            ),
            tuple(
                index for index in range(ppg.size) if index not in matched_ppg_indices
            ),
            PpgEcgValidator._matched_block_count(chosen),
        )

    @staticmethod
    def _matched_block_count(matches: list[tuple[int, int, float]]) -> int:
        """Count contiguous ECG/PPG match blocks without bridging skips."""

        if not matches:
            return 0
        blocks = 1
        for previous, current in zip(matches, matches[1:], strict=False):
            if current[0] != previous[0] + 1 or current[1] != previous[1] + 1:
                blocks += 1
        return blocks

    @staticmethod
    def _initial_pat_delays(
        ecg: np.ndarray,
        ppg: np.ndarray,
        pat_min: float,
        pat_max: float,
    ) -> list[float]:
        delays: list[float] = []
        last_ppg = -1
        for ecg_time in ecg:
            start = max(
                last_ppg + 1,
                int(np.searchsorted(ppg, ecg_time + pat_min, side="left")),
            )
            end = int(np.searchsorted(ppg, ecg_time + pat_max, side="right"))
            if start >= end:
                continue
            ppg_index = start
            delays.append(float(ppg[ppg_index] - ecg_time))
            last_ppg = ppg_index
        return delays

    @staticmethod
    def _fit_pat_model(
        ecg: np.ndarray,
        ppg: np.ndarray,
        matches: list[tuple[int, int, float]],
        tolerance: float,
        pat_min: float,
        pat_max: float,
    ) -> tuple[float, float, float]:
        if len(matches) < 4:
            return (
                float(np.median([delay for _, _, delay in matches]))
                if matches
                else (pat_min + pat_max) / 2.0,
                0.0,
                float(ecg[0]),
            )
        delays = np.asarray([delay for _, _, delay in matches], dtype=float)
        center = float(np.median(delays))
        mad = float(np.median(np.abs(delays - center)))
        robust_limit = max(tolerance, 3.0 * 1.4826 * mad)
        keep = np.abs(delays - center) <= robust_limit
        x = (ecg[np.asarray([i for i, _, _ in matches])] - ecg[0]) / 60_000.0
        y = delays
        if int(keep.sum()) < 4 or np.ptp(x[keep]) <= 0.0:
            return center, 0.0, float(ecg[0])
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        predicted = intercept + slope * x
        if (
            not np.isfinite(slope)
            or not np.isfinite(intercept)
            or float(np.min(predicted)) < pat_min
            or float(np.max(predicted)) > pat_max
        ):
            return center, 0.0, float(ecg[0])
        return float(intercept), float(slope), float(ecg[0])

    @staticmethod
    def _greedy_pat_matches(
        ecg: np.ndarray,
        ppg: np.ndarray,
        model: tuple[float, float, float],
        pat_min: float,
        pat_max: float,
        tolerance: float,
    ) -> list[tuple[int, int, float]]:
        intercept, slope, reference_time = model
        matches: list[tuple[int, int, float]] = []
        last_ppg = -1
        for ecg_index, ecg_time in enumerate(ecg):
            start = max(
                last_ppg + 1,
                int(np.searchsorted(ppg, ecg_time + pat_min, side="left")),
            )
            end = int(np.searchsorted(ppg, ecg_time + pat_max, side="right"))
            if start >= end:
                continue
            predicted_pat = intercept + slope * (ecg_time - reference_time) / 60_000.0
            candidates = np.arange(start, end, dtype=int)
            delays = ppg[candidates] - ecg_time
            keep = np.abs(delays - predicted_pat) <= tolerance
            if not keep.any():
                continue
            selected = candidates[
                np.flatnonzero(keep)[np.argmin(np.abs(delays[keep] - predicted_pat))]
            ]
            delay = float(ppg[selected] - ecg_time)
            matches.append((ecg_index, int(selected), delay))
            last_ppg = int(selected)
        return matches

    @staticmethod
    def _match_cost(
        matches: list[tuple[int, int, float]],
        model: tuple[float, float, float],
        ecg: np.ndarray,
    ) -> float:
        intercept, slope, reference_time = model
        return float(
            sum(
                abs(
                    delay
                    - (intercept + slope * (ecg[ecg_index] - reference_time) / 60_000.0)
                )
                for ecg_index, _, delay in matches
            )
        )

    def _metrics(
        self,
        matches: list[tuple[int, int, float]],
        ecg_ms: np.ndarray,
        ppg_ms: np.ndarray,
        *,
        ecg_gap_events: tuple[GapEvent, ...] = (),
        ppg_gap_events: tuple[GapEvent, ...] = (),
    ) -> dict[str, Any]:
        n_ecg, n_ppg = len(ecg_ms), len(ppg_ms)
        matched = len(matches)
        precision = matched / n_ppg if n_ppg else np.nan
        recall = matched / n_ecg if n_ecg else np.nan
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else np.nan
        )
        ratio = matched / min(n_ecg, n_ppg) if min(n_ecg, n_ppg) else np.nan
        ecg_ibi, ppg_ibi = self._adjacent_observed_intervals(
            matches,
            ecg_ms,
            ppg_ms,
            ecg_gap_events=ecg_gap_events,
            ppg_gap_events=ppg_gap_events,
        )
        diff = ppg_ibi - ecg_ibi
        valid = np.isfinite(diff) & (ecg_ibi > 0.0) & (ppg_ibi > 0.0)
        diff = diff[valid]
        ecg_ibi = ecg_ibi[valid]
        ppg_ibi = ppg_ibi[valid]
        if len(diff):
            bias = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
            ibi_metrics: dict[str, Any] = {
                "ibi_pairs": int(len(diff)),
                "ibi_mae_ms": float(np.mean(np.abs(diff))),
                "ibi_median_absolute_error_ms": float(np.median(np.abs(diff))),
                "ibi_rmse_ms": float(np.sqrt(np.mean(np.square(diff)))),
                "ibi_pearson_r": PpgEcgValidator._correlation(
                    pearsonr, ecg_ibi, ppg_ibi
                ),
                "ibi_spearman_r": PpgEcgValidator._correlation(
                    spearmanr, ecg_ibi, ppg_ibi
                ),
                "hr_mae_bpm": float(
                    np.mean(np.abs(60000.0 / ppg_ibi - 60000.0 / ecg_ibi))
                ),
                "bland_altman_bias_ms": bias,
                "bland_altman_loa_low_ms": bias - 1.96 * sd,
                "bland_altman_loa_high_ms": bias + 1.96 * sd,
            }
        else:
            ibi_metrics = {
                "ibi_pairs": 0,
                "ibi_mae_ms": np.nan,
                "ibi_median_absolute_error_ms": np.nan,
                "ibi_rmse_ms": np.nan,
                "ibi_pearson_r": np.nan,
                "ibi_spearman_r": np.nan,
                "hr_mae_bpm": np.nan,
                "bland_altman_bias_ms": np.nan,
                "bland_altman_loa_low_ms": np.nan,
                "bland_altman_loa_high_ms": np.nan,
            }
        result = {
            "matched_beat_count": matched,
            "matched_beat_ratio": float(ratio),
            "beat_precision": float(precision),
            "beat_recall": float(recall),
            "beat_f1": float(f1),
            **ibi_metrics,
        }
        if self._compute_confidence_intervals:
            result.update(
                self._bootstrap_confidence_intervals(
                    n_ecg, n_ppg, matched, ecg_ibi, ppg_ibi
                )
            )
        else:
            result.update(
                {
                    "beat_f1_ci_low": np.nan,
                    "beat_f1_ci_high": np.nan,
                    "ibi_mae_ms_ci_low": np.nan,
                    "ibi_mae_ms_ci_high": np.nan,
                    "ibi_rmse_ms_ci_low": np.nan,
                    "ibi_rmse_ms_ci_high": np.nan,
                    "hr_mae_bpm_ci_low": np.nan,
                    "hr_mae_bpm_ci_high": np.nan,
                    "bland_altman_bias_ms_ci_low": np.nan,
                    "bland_altman_bias_ms_ci_high": np.nan,
                    "ci_status": "not_requested_for_detector_audit",
                    "ci_iterations": 0,
                }
            )
        return result

    @staticmethod
    def _adjacent_observed_intervals(
        matches: list[tuple[int, int, float]],
        ecg_ms: np.ndarray,
        ppg_ms: np.ndarray,
        *,
        ecg_gap_events: tuple[GapEvent, ...] = (),
        ppg_gap_events: tuple[GapEvent, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return physiological RR/PPI pairs inside adjacent match blocks.

        A peak-only table compresses missing samples.  Therefore index
        adjacency alone is insufficient: an interval that crosses an ECG/PPG
        gap is not observed evidence and must not be used for primary metrics.
        """

        ecg_intervals: list[float] = []
        ppg_intervals: list[float] = []
        for current, following in zip(matches, matches[1:], strict=False):
            ecg_index, ppg_index, _ = current
            next_ecg_index, next_ppg_index, _ = following
            if next_ecg_index != ecg_index + 1 or next_ppg_index != ppg_index + 1:
                continue
            ecg_interval = float(ecg_ms[next_ecg_index] - ecg_ms[ecg_index])
            ppg_interval = float(ppg_ms[next_ppg_index] - ppg_ms[ppg_index])
            if (
                np.isfinite(ecg_interval)
                and np.isfinite(ppg_interval)
                and MIN_PHYSIOLOGICAL_INTERVAL_MS
                <= ecg_interval
                <= MAX_PHYSIOLOGICAL_INTERVAL_MS
                and MIN_PHYSIOLOGICAL_INTERVAL_MS
                <= ppg_interval
                <= MAX_PHYSIOLOGICAL_INTERVAL_MS
                and not PpgEcgValidator._interval_crosses_gap(
                    float(ecg_ms[ecg_index]),
                    float(ecg_ms[next_ecg_index]),
                    ecg_gap_events,
                )
                and not PpgEcgValidator._interval_crosses_gap(
                    float(ppg_ms[ppg_index]),
                    float(ppg_ms[next_ppg_index]),
                    ppg_gap_events,
                )
            ):
                ecg_intervals.append(ecg_interval)
                ppg_intervals.append(ppg_interval)
        return (
            np.asarray(ecg_intervals, dtype=float),
            np.asarray(ppg_intervals, dtype=float),
        )

    @staticmethod
    def _interval_crosses_gap(
        start_ms: float,
        end_ms: float,
        gap_events: tuple[GapEvent, ...],
    ) -> bool:
        low, high = sorted((start_ms, end_ms))
        return any(
            event.affects_peak_region and event.start_ms < high and event.end_ms > low
            for event in gap_events
        )

    def _bootstrap_confidence_intervals(
        self,
        n_ecg: int,
        n_ppg: int,
        matched: int,
        ecg_ibi: np.ndarray,
        ppg_ibi: np.ndarray,
    ) -> dict[str, Any]:
        sync = self.config.synchronization if self.config else None
        iterations = int(sync.bootstrap_iterations) if sync else 2000
        seed = int(sync.bootstrap_seed) if sync else 42
        minimum_pairs = int(sync.bootstrap_min_pairs) if sync else 10
        empty = {
            "beat_f1_ci_low": np.nan,
            "beat_f1_ci_high": np.nan,
            "ibi_mae_ms_ci_low": np.nan,
            "ibi_mae_ms_ci_high": np.nan,
            "ibi_rmse_ms_ci_low": np.nan,
            "ibi_rmse_ms_ci_high": np.nan,
            "hr_mae_bpm_ci_low": np.nan,
            "hr_mae_bpm_ci_high": np.nan,
            "bland_altman_bias_ms_ci_low": np.nan,
            "bland_altman_bias_ms_ci_high": np.nan,
            "ci_status": "insufficient_pairs",
            "ci_iterations": 0,
        }
        if len(ecg_ibi) < minimum_pairs or len(ppg_ibi) != len(ecg_ibi):
            return empty

        rng = np.random.default_rng(seed)
        precision_events = np.r_[
            np.ones(matched, dtype=float),
            np.zeros(max(0, n_ppg - matched), dtype=float),
        ]
        recall_events = np.r_[
            np.ones(matched, dtype=float),
            np.zeros(max(0, n_ecg - matched), dtype=float),
        ]
        beat_f1_values = np.empty(iterations, dtype=float)
        ibi_mae_values = np.empty(iterations, dtype=float)
        ibi_rmse_values = np.empty(iterations, dtype=float)
        hr_mae_values = np.empty(iterations, dtype=float)
        bias_values = np.empty(iterations, dtype=float)
        for index in range(iterations):
            precision_sample = rng.choice(
                precision_events, size=len(precision_events), replace=True
            )
            recall_sample = rng.choice(
                recall_events, size=len(recall_events), replace=True
            )
            bootstrap_precision = float(np.mean(precision_sample))
            bootstrap_recall = float(np.mean(recall_sample))
            denominator = bootstrap_precision + bootstrap_recall
            beat_f1_values[index] = (
                2.0 * bootstrap_precision * bootstrap_recall / denominator
                if denominator
                else 0.0
            )
            sample_indices = rng.integers(0, len(ecg_ibi), size=len(ecg_ibi))
            ecg_sample = ecg_ibi[sample_indices]
            ppg_sample = ppg_ibi[sample_indices]
            difference = ppg_sample - ecg_sample
            ibi_mae_values[index] = float(np.mean(np.abs(difference)))
            ibi_rmse_values[index] = float(np.sqrt(np.mean(np.square(difference))))
            hr_mae_values[index] = float(
                np.mean(np.abs(60000.0 / ppg_sample - 60000.0 / ecg_sample))
            )
            bias_values[index] = float(np.mean(difference))

        def interval(values: np.ndarray) -> tuple[float, float]:
            low, high = np.percentile(values, [2.5, 97.5])
            return float(low), float(high)

        beat_low, beat_high = interval(beat_f1_values)
        mae_low, mae_high = interval(ibi_mae_values)
        rmse_low, rmse_high = interval(ibi_rmse_values)
        hr_low, hr_high = interval(hr_mae_values)
        bias_low, bias_high = interval(bias_values)
        return {
            "beat_f1_ci_low": beat_low,
            "beat_f1_ci_high": beat_high,
            "ibi_mae_ms_ci_low": mae_low,
            "ibi_mae_ms_ci_high": mae_high,
            "ibi_rmse_ms_ci_low": rmse_low,
            "ibi_rmse_ms_ci_high": rmse_high,
            "hr_mae_bpm_ci_low": hr_low,
            "hr_mae_bpm_ci_high": hr_high,
            "bland_altman_bias_ms_ci_low": bias_low,
            "bland_altman_bias_ms_ci_high": bias_high,
            "ci_status": "ok",
            "ci_iterations": iterations,
        }

    @staticmethod
    def _correlation(function: Any, left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
            return float("nan")
        return float(function(left, right).statistic)

    def _plot_audit(
        self,
        participant_id: str,
        phase: str,
        channel: str,
        ecg: DetectedPeaks,
        ppg: DetectedPeaks,
        matches: list[tuple[int, int, float]],
    ) -> None:
        path = self._plot_path(participant_id, phase, channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(4, 1, figsize=(16, 17), constrained_layout=True)
        ecg_z = self._zscore(ecg.signal)
        ppg_z = self._zscore(ppg.signal)
        # Some historical fixtures store the full sample clock in
        # ``timestamps_ms``; production DetectedPeaks stores peak timestamps.
        # Normalize both representations at this I/O boundary.
        ecg_peak_ms = (
            ecg.timestamps_ms[ecg.sample_indices]
            if len(ecg.timestamps_ms) == len(ecg.signal)
            else ecg.timestamps_ms
        )
        ppg_peak_ms = (
            ppg.timestamps_ms[ppg.sample_indices]
            if len(ppg.timestamps_ms) == len(ppg.signal)
            else ppg.timestamps_ms
        )
        origin_ms = min(float(ecg_peak_ms[0]), float(ppg_peak_ms[0]))
        ecg_t = (ecg_peak_ms - origin_ms) / 1000.0
        ppg_t = (ppg_peak_ms - origin_ms) / 1000.0
        # ``timestamps_ms`` contains peak timestamps only, while ``signal``
        # contains the complete cleaned waveform. Reconstruct the uniform
        # sample clock from the first detected peak instead of pairing arrays
        # with different lengths. The CSV visualizer remains the authoritative
        # full-resolution audit artifact.
        ecg_start_ms = float(ecg_peak_ms[0]) - (
            float(ecg.sample_indices[0]) * 1000.0 / ecg.fs_hz
        )
        ppg_start_ms = float(ppg_peak_ms[0]) - (
            float(ppg.sample_indices[0]) * 1000.0 / ppg.fs_hz
        )
        ecg_signal_t = (
            np.arange(len(ecg.signal), dtype=float) / ecg.fs_hz
            + (ecg_start_ms - origin_ms) / 1000.0
        )
        ppg_signal_t = (
            np.arange(len(ppg.signal), dtype=float) / ppg.fs_hz
            + (ppg_start_ms - origin_ms) / 1000.0
        )
        stride_ecg = max(1, len(ecg_signal_t) // 10000)
        stride_ppg = max(1, len(ppg_signal_t) // 10000)
        axes[0].plot(
            ecg_signal_t[::stride_ecg],
            ecg_z[::stride_ecg],
            label="ECG clean",
            linewidth=0.5,
        )
        axes[0].plot(
            ppg_signal_t[::stride_ppg],
            ppg_z[::stride_ppg],
            label=f"{channel} clean",
            linewidth=0.5,
        )
        axes[0].scatter(
            ecg_t,
            ecg_z[np.clip(ecg.sample_indices, 0, len(ecg_z) - 1)],
            s=8,
            label="ECG R",
        )
        axes[0].scatter(
            ppg_t,
            ppg_z[np.clip(ppg.sample_indices, 0, len(ppg_z) - 1)],
            s=8,
            label="PPG peaks",
        )
        axes[0].set_title("Core 240-second signal with independently detected peaks")
        axes[0].set_ylabel("z-score")
        axes[0].legend(loc="upper right")

        ecg_ibi = np.diff(ecg_peak_ms)
        ppg_ibi = np.diff(ppg_peak_ms)
        ecg_ibi_t = (ecg_peak_ms[1:] - origin_ms) / 1000.0
        ppg_ibi_t = (ppg_peak_ms[1:] - origin_ms) / 1000.0
        axes[1].scatter(ecg_ibi_t, ecg_ibi, s=9, label="ECG RR", alpha=0.75)
        axes[1].scatter(ppg_ibi_t, ppg_ibi, s=9, label="PPG PPI", alpha=0.75)
        axes[1].set_title("RR/PPI intervals over time")
        axes[1].set_xlabel("time from validation-window start (s)")
        axes[1].set_ylabel("interval (ms)")
        axes[1].legend(loc="upper right")

        interval_sets = ((ecg_ibi, "ECG RR"), (ppg_ibi, "PPG PPI"))
        for values, label in interval_sets:
            if len(values):
                axes[2].hist(
                    values,
                    bins=min(30, max(8, len(values) // 3)),
                    alpha=0.5,
                    label=label,
                )
        axes[2].set_title("Distribution of RR/PPI intervals")
        axes[2].set_xlabel("interval (ms)")
        axes[2].set_ylabel("count")
        axes[2].legend(loc="upper right")

        poincare_values: list[np.ndarray] = []
        for values, label, color in (
            (ecg_ibi, "ECG RR", "tab:blue"),
            (ppg_ibi, "PPG PPI", "tab:orange"),
        ):
            if len(values) < 2:
                continue
            axes[3].scatter(
                values[:-1], values[1:], s=12, alpha=0.65, label=label, color=color
            )
            poincare_values.append(values)
        if poincare_values:
            limits = np.concatenate(poincare_values)
            low, high = float(np.nanmin(limits)), float(np.nanmax(limits))
            padding = max((high - low) * 0.05, 1.0)
            axes[3].plot(
                [low - padding, high + padding],
                [low - padding, high + padding],
                "k--",
                linewidth=0.8,
            )
            axes[3].set_xlim(low - padding, high + padding)
            axes[3].set_ylim(low - padding, high + padding)
        axes[3].set_title("Poincare plot")
        axes[3].set_xlabel("RR/PPI(n) (ms)")
        axes[3].set_ylabel("RR/PPI(n+1) (ms)")
        axes[3].legend(loc="upper left")

        pat_values = [match[2] for match in matches]
        mean_hr = 60000.0 / np.mean(ppg_ibi) if len(ppg_ibi) else np.nan
        fig.suptitle(
            f"Peak Detection — {participant_id} ({phase}, {channel}) | "
            f"ECG peaks: {len(ecg.timestamps_ms)} | PPG peaks: {len(ppg.timestamps_ms)} | "
            f"Mean HR: {mean_hr:.1f} BPM | PAT median: "
            f"{np.median(pat_values) if pat_values else np.nan:.1f} ms"
        )
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def _ensure_plot_artifacts(self, metrics: pd.DataFrame) -> None:
        """Guarantee one visual audit artifact for every canonical row.

        A signal-bearing row receives the four-panel inspection plot above.  If
        a session cannot be processed (for example, missing sync metadata), a
        clearly labelled placeholder is emitted instead of silently omitting
        the participant/phase/channel from the audit folder.
        """

        for row in metrics.itertuples(index=False):
            path = self._plot_path(row.participant_id, row.phase, row.sensor_channel)
            if path.is_file():
                continue
            self._plot_exclusion(
                row.participant_id,
                row.phase,
                row.sensor_channel,
                str(row.exclusion_reason or "validation_not_processed"),
            )

    def _plot_exclusion(
        self, participant_id: str, phase: str, channel: str, reason: str
    ) -> None:
        path = self._plot_path(participant_id, phase, channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.60,
            f"Peak Detection — {participant_id} ({phase}, {channel})",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        axis.text(
            0.5,
            0.38,
            "No 240-second physiological comparison was produced for this combination.",
            ha="center",
            va="center",
            fontsize=10,
        )
        axis.text(
            0.5,
            0.18,
            f"Exclusion reason: {reason}",
            ha="center",
            va="center",
            fontsize=9,
            wrap=True,
        )
        figure.savefig(path, dpi=140)
        plt.close(figure)

    def _plot_path(self, participant_id: str, phase: str, channel: str) -> Path:
        return (
            self.audit_dir
            / participant_id
            / phase
            / channel
            / "peak_detection_240s.png"
        )

    @staticmethod
    def _zscore(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        scale = float(np.nanstd(values))
        return (values - float(np.nanmean(values))) / (scale if scale > 0.0 else 1.0)

    def _validation_window_seconds(self) -> int:
        if self.config is None:
            return DEFAULT_VALIDATION_WINDOW_SECONDS
        return int(self.config.synchronization.validation_window_seconds)

    def _min_overlap_seconds(self) -> float:
        if self.config is None:
            return float(self._validation_window_seconds() - 1.0)
        return float(self.config.synchronization.min_overlap_seconds)

    def _load_anchors(self) -> dict[tuple[str, str], pd.Series]:
        anchor_path = (
            self.config.raw_source.anchor_csv
            if self.config is not None
            else self.paths.anchor_csv_path
        )
        if not anchor_path.is_file():
            return {}
        mapping_path = self.paths.raw_data_root / "mapping.csv"
        try:
            mapper = AnchorMapper(
                anchor_path,
                self.config.timezone_local if self.config is not None else "UTC",
                mapping_csv_path=mapping_path,
            )
            anchors = mapper.load()
        except (OSError, KeyError, ValueError) as exc:
            LOGGER.warning(
                "Unable to load validation anchors from %s: %s", anchor_path, exc
            )
            return {}
        anchors["phase"] = anchors["phase"].map(self._canonical_phase)
        anchors = anchors[anchors["phase"].isin(PHASES)].copy()
        anchors["participant_code"] = anchors["participant_code"].astype(str)
        if self._allowed_participants is not None:
            anchors = anchors[
                anchors["participant_code"].isin(self._allowed_participants)
            ].copy()
        indexed: dict[tuple[str, str], pd.Series] = {}
        for key, group in anchors.groupby(["participant_code", "phase"], sort=True):
            if len(group) == 1:
                indexed[(str(key[0]), str(key[1]))] = group.iloc[0]
            else:
                LOGGER.warning("Duplicate validation anchors for %s/%s", key[0], key[1])
        return indexed

    @staticmethod
    def _canonical_phase(value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized in {"relax", "relaks", "relaksasi"}:
            return "relax"
        if normalized in {"task", "stress", "stres"}:
            return "task"
        return normalized

    def _validation_window(
        self, participant_id: str, phase: str
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        anchor = self._anchors_by_participant_phase.get((participant_id, phase))
        if anchor is None:
            return None
        return self._window_slicer.bounds(
            pd.Timestamp(anchor["start_timestamp"]), mode="validation"
        )

    @staticmethod
    def _crop_to_window(
        frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.DataFrame, float]:
        timestamp_column = PpgEcgValidator._column(
            frame,
            (
                "timestamp_corrected_utc",
                "timestamp_utc",
                "timestamp_raw_utc",
                "unix_ms",
            ),
        )
        timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        selected = frame.loc[(timestamps >= start) & (timestamps < end)].copy()
        if selected.empty:
            return selected, 0.0
        selected_timestamps = pd.to_datetime(
            selected[timestamp_column], utc=True, errors="coerce"
        ).dropna()
        if selected_timestamps.empty:
            return selected, 0.0
        coverage = float(
            (selected_timestamps.iloc[-1] - selected_timestamps.iloc[0]).total_seconds()
        )
        return selected, coverage

    @staticmethod
    def _frame_coverage(frame: pd.DataFrame) -> float:
        """Return observed duration for an already-cropped frame."""

        timestamp_column = PpgEcgValidator._column(
            frame,
            (
                "timestamp_corrected_utc",
                "timestamp_utc",
                "timestamp_raw_utc",
                "unix_ms",
            ),
        )
        timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        timestamps = timestamps.dropna()
        if timestamps.empty:
            return 0.0
        return float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds())

    def _build_sync_index(self) -> dict[tuple[str, str], pd.Series]:
        if self._sync_manifest.empty or "participant_id" not in self._sync_manifest:
            return {}
        manifest = self._sync_manifest.copy()
        phases = (
            manifest["phase"].map(self._canonical_phase)
            if "phase" in manifest
            else manifest.get("recording_id", pd.Series(dtype=str))
            .astype(str)
            .map(self._phase_from_path)
        )
        manifest["_canonical_phase"] = phases
        indexed: dict[tuple[str, str], pd.Series] = {}
        for key, group in manifest.dropna(subset=["_canonical_phase"]).groupby(
            ["participant_id", "_canonical_phase"], sort=True
        ):
            if len(group) == 1:
                indexed[(str(key[0]), str(key[1]))] = group.iloc[0]
            else:
                self._sync_ambiguities.add((str(key[0]), str(key[1])))
        return indexed

    def _add_sync_metadata(
        self, row: dict[str, Any], participant_id: str, phase: str
    ) -> bool:
        sync = self._sync_by_participant_phase.get((participant_id, phase))
        if sync is None:
            return False
        row["sync_method"] = str(sync.get("method", ""))
        row["clock_model"] = str(sync.get("timebase_source", sync.get("method", "")))
        row["sync_alpha"] = float(sync.get("alpha", np.nan))
        row["sync_beta_seconds"] = float(sync.get("beta_seconds", np.nan))
        return True

    @staticmethod
    def _ecg_exclusion(frame: pd.DataFrame) -> tuple[bool, str]:
        if "ecg_excluded" in frame.columns and bool(frame["ecg_excluded"].iloc[0]):
            reason = str(frame.get("exclusion_reason", pd.Series(["unknown"])).iloc[0])
            return True, reason or "canonical_sync_exclusion"
        return False, ""

    @staticmethod
    def _source_file(frame: pd.DataFrame, fallback: Path) -> str:
        if "source_file" in frame.columns and not frame["source_file"].dropna().empty:
            return str(frame["source_file"].dropna().iloc[0])
        return str(fallback)

    def _source_checksum(self, frame: pd.DataFrame, fallback: Path) -> str:
        for column in ("source_checksum", "source_sha256"):
            if column in frame.columns and not frame[column].dropna().empty:
                value = str(frame[column].dropna().iloc[0])
                if value and value.lower() != "nan":
                    return value
        source_file = self._source_file(frame, fallback)
        candidate = Path(source_file)
        if not candidate.is_absolute():
            candidate = self.paths.project_root / candidate
        if candidate.is_file():
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        return ""

    def _write_sync_parameters(self) -> None:
        output = self.output_dir / "sync_parameters_240s.csv"
        if not self._sync_manifest.empty:
            manifest = self._sync_manifest.copy()
            manifest["initial_offset_ms"] = (
                pd.to_numeric(manifest.get("beta_seconds"), errors="coerce") * 1000.0
            )
            manifest["drift_slope"] = (
                pd.to_numeric(manifest.get("alpha"), errors="coerce") - 1.0
            )
            manifest["correction_method"] = manifest.get("method", "")
            manifest["pre_alignment_score"] = pd.to_numeric(
                manifest.get("pre_alignment_score"), errors="coerce"
            )
            manifest["post_alignment_score"] = pd.to_numeric(
                manifest.get("post_alignment_score", manifest.get("confidence")),
                errors="coerce",
            )
            manifest["validation_window_seconds"] = self._validation_window_seconds()
            manifest.to_csv(output, index=False)
            manifest.to_csv(self.output_dir / "sync_parameters.csv", index=False)
            self._write_json(output.with_suffix(".json"), manifest.to_dict("records"))
            self._write_json(
                (self.output_dir / "sync_parameters.json"), manifest.to_dict("records")
            )
        else:
            empty = pd.DataFrame(
                columns=[
                    "participant_id",
                    "recording_id",
                    "initial_offset_ms",
                    "drift_slope",
                    "correction_method",
                    "pre_alignment_score",
                    "post_alignment_score",
                    "validation_window_seconds",
                ]
            )
            empty.to_csv(output, index=False)
            empty.to_csv(self.output_dir / "sync_parameters.csv", index=False)

    def _write_validation_artifact_tables(self, metrics: pd.DataFrame) -> None:
        self._participant_phase_summary(metrics).to_csv(
            self.output_dir / "participant_phase_summary_240s.csv", index=False
        )
        self._timebase_candidate_audit().to_csv(
            self.output_dir / "timebase_candidate_audit_240s.csv", index=False
        )
        self._exclusion_breakdown(metrics).to_csv(
            self.output_dir / "exclusion_breakdown_240s.csv", index=False
        )
        self._baseline_comparison(metrics).to_csv(
            self.output_dir / "baseline_vs_rebuild_240s.csv", index=False
        )

    def _participant_phase_summary(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for participant_id in sorted(VALID_PARTICIPANTS):
            for phase in PHASES:
                selected = metrics[
                    (metrics["participant_id"] == participant_id)
                    & (metrics["phase"] == phase)
                ].copy()
                if "validation_status" in selected:
                    valid_mask = selected["validation_status"] == "valid"
                else:
                    valid_mask = selected.get("status", pd.Series(dtype=str)) == "ok"
                observed = int(len(selected))
                valid = int(valid_mask.sum())
                excluded = observed - valid
                if observed != len(PPG_CHANNELS):
                    session_status = "incomplete"
                elif valid == len(PPG_CHANNELS):
                    session_status = "valid"
                elif valid:
                    session_status = "partial"
                else:
                    session_status = "excluded"
                reasons = sorted(
                    {
                        str(value).strip()
                        for value in selected.get(
                            "exclusion_reason", pd.Series(dtype=str)
                        ).dropna()
                        if str(value).strip()
                    }
                )
                rows.append(
                    {
                        "participant_id": participant_id,
                        "phase": phase,
                        "session_validation_status": session_status,
                        "expected_channel_count": len(PPG_CHANNELS),
                        "observed_channel_count": observed,
                        "valid_channel_count": valid,
                        "excluded_channel_count": excluded,
                        "agreement_reported_count": int(
                            (
                                selected.get("agreement_status", pd.Series(dtype=str))
                                == "reported"
                            ).sum()
                        ),
                        "ecg_peak_count": self._numeric_max(selected, "ecg_peak_count"),
                        "ppg_peak_count_total": self._numeric_sum(
                            selected, "ppg_peak_count"
                        ),
                        "matched_beat_count_total": self._numeric_sum(
                            selected, "matched_beat_count"
                        ),
                        "median_beat_f1": self._numeric_median(selected, "beat_f1"),
                        "median_ibi_mae_ms": self._numeric_median(
                            selected, "ibi_mae_ms"
                        ),
                        "median_ibi_rmse_ms": self._numeric_median(
                            selected, "ibi_rmse_ms"
                        ),
                        "median_hr_mae_bpm": self._numeric_median(
                            selected, "hr_mae_bpm"
                        ),
                        "sync_method": self._first_text(selected, "sync_method"),
                        "sync_alpha": self._numeric_median(selected, "sync_alpha"),
                        "sync_beta_seconds": self._numeric_median(
                            selected, "sync_beta_seconds"
                        ),
                        "exclusion_reasons": "|".join(reasons),
                    }
                )
        return pd.DataFrame(rows)

    def _timebase_candidate_audit(self) -> pd.DataFrame:
        source_fields = (
            "recording_id",
            "method",
            "timebase_source",
            "clock_selection_basis",
            "observed_sample_count",
            "declared_sample_count",
            "declared_duration_seconds",
            "metadata_stop_unix_ms",
            "nominal_sample_rate_hz",
            "effective_sample_rate_hz",
            "effective_rate_relative_error",
            "candidate_start_unix_ms",
            "candidate_end_unix_ms",
            "reference_signal_role",
            "ppg_diagnostic_reason",
            "clock_excluded",
            "clock_exclusion_reason",
            "diagnostic_reference_channel",
            "alpha",
            "beta_seconds",
            "overlap_seconds",
            "pre_alignment_score",
            "post_alignment_score",
            "excluded",
            "exclusion_reason",
        )
        rows: list[dict[str, Any]] = []
        for participant_id in sorted(VALID_PARTICIPANTS):
            for phase in PHASES:
                sync = self._sync_by_participant_phase.get((participant_id, phase))
                row: dict[str, Any] = {
                    "participant_id": participant_id,
                    "phase": phase,
                    "candidate_status": "missing",
                    "audit_reason": "missing_canonical_sync_metadata",
                    "validation_window_seconds": self._validation_window_seconds(),
                }
                row.update({field: None for field in source_fields})
                if sync is not None:
                    row.update({field: sync.get(field) for field in source_fields})
                    excluded = self._truthy(sync.get("excluded", False))
                    row["candidate_status"] = "excluded" if excluded else "accepted"
                    row["audit_reason"] = (
                        self._scalar_text(sync.get("exclusion_reason"))
                        if excluded
                        else ""
                    )
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _exclusion_breakdown(metrics: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "sensor_channel",
            "exclusion_reason",
            "combination_count",
            "participant_count",
            "phase_count",
            "fraction_of_channel",
        ]
        if "validation_status" in metrics:
            excluded = metrics[metrics["validation_status"] != "valid"].copy()
        else:
            excluded = metrics[
                metrics.get("status", pd.Series(dtype=str)) != "ok"
            ].copy()
        if excluded.empty:
            return pd.DataFrame(columns=columns)
        excluded["exclusion_reason"] = (
            excluded["exclusion_reason"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "unspecified_exclusion")
        )
        channel_totals = metrics.groupby("sensor_channel", sort=True).size().to_dict()
        rows = []
        for (channel, reason), group in excluded.groupby(
            ["sensor_channel", "exclusion_reason"], sort=True
        ):
            count = int(len(group))
            rows.append(
                {
                    "sensor_channel": channel,
                    "exclusion_reason": reason,
                    "combination_count": count,
                    "participant_count": int(group["participant_id"].nunique()),
                    "phase_count": int(group["phase"].nunique()),
                    "fraction_of_channel": count
                    / max(1, int(channel_totals.get(channel, 0))),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def _baseline_comparison(self, metrics: pd.DataFrame) -> pd.DataFrame:
        sync_config = (
            self.config.synchronization if self.config else SynchronizationConfig()
        )
        baseline_run_id = sync_config.baseline_validation_run_id
        configured_path = Path(sync_config.baseline_validation_metrics_path)
        baseline_path = (
            configured_path
            if configured_path.is_absolute()
            else self.paths.project_root / configured_path
        )
        baseline, read_status = self._read_baseline_metrics(baseline_path)
        keys = ("participant_id", "phase", "sensor_channel")
        comparison_metrics = (
            "ecg_peak_count",
            "ppg_peak_count",
            "matched_beat_count",
            "beat_f1",
            "ibi_pairs",
            "ibi_mae_ms",
            "ibi_rmse_ms",
            "ibi_pearson_r",
            "ibi_spearman_r",
            "hr_mae_bpm",
            "pat_median_ms",
        )
        baseline_index: dict[tuple[str, str, str], dict[str, Any]] = {}
        duplicate_keys: set[tuple[str, str, str]] = set()
        if baseline is not None and set(keys).issubset(baseline.columns):
            baseline = baseline.copy()
            baseline["phase"] = baseline["phase"].map(self._canonical_phase)
            for record in baseline.to_dict("records"):
                key = tuple(str(record[field]) for field in keys)
                if key in baseline_index:
                    duplicate_keys.add(key)
                else:
                    baseline_index[key] = record
        elif baseline is not None:
            read_status = "baseline_schema_invalid"

        rows: list[dict[str, Any]] = []
        for rebuild in metrics.to_dict("records"):
            key = tuple(str(rebuild[field]) for field in keys)
            baseline_row = baseline_index.get(key)
            if read_status != "available":
                comparison_status = read_status
                baseline_row = None
            elif key in duplicate_keys:
                comparison_status = "baseline_duplicate_key"
                baseline_row = None
            elif baseline_row is None:
                comparison_status = "baseline_row_missing"
            else:
                comparison_status = "compared"
            row: dict[str, Any] = {
                **{field: rebuild[field] for field in keys},
                "baseline_run_id": baseline_run_id,
                "baseline_metrics_path": self._relative(baseline_path),
                "baseline_available": baseline_row is not None,
                "comparison_status": comparison_status,
                "baseline_status": self._record_text(baseline_row, "status"),
                "rebuild_status": self._record_text(rebuild, "status"),
                "baseline_validation_status": self._record_text(
                    baseline_row, "validation_status"
                ),
                "rebuild_validation_status": self._record_text(
                    rebuild, "validation_status"
                ),
                "baseline_exclusion_reason": self._record_text(
                    baseline_row, "exclusion_reason"
                ),
                "rebuild_exclusion_reason": self._record_text(
                    rebuild, "exclusion_reason"
                ),
            }
            for metric in comparison_metrics:
                baseline_value = self._record_number(baseline_row, metric)
                rebuild_value = self._record_number(rebuild, metric)
                row[f"baseline_{metric}"] = baseline_value
                row[f"rebuild_{metric}"] = rebuild_value
                row[f"delta_{metric}"] = (
                    rebuild_value - baseline_value
                    if np.isfinite(baseline_value) and np.isfinite(rebuild_value)
                    else np.nan
                )
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _read_baseline_metrics(path: Path) -> tuple[pd.DataFrame | None, str]:
        if not path.is_file():
            return None, "baseline_file_missing"
        try:
            return pd.read_csv(path), "available"
        except (OSError, pd.errors.ParserError, UnicodeError):
            return None, "baseline_read_error"

    @staticmethod
    def _numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").dropna()

    @classmethod
    def _numeric_median(cls, frame: pd.DataFrame, column: str) -> float:
        values = cls._numeric_values(frame, column)
        return float(values.median()) if not values.empty else np.nan

    @classmethod
    def _numeric_max(cls, frame: pd.DataFrame, column: str) -> float:
        values = cls._numeric_values(frame, column)
        return float(values.max()) if not values.empty else np.nan

    @classmethod
    def _numeric_sum(cls, frame: pd.DataFrame, column: str) -> float:
        values = cls._numeric_values(frame, column)
        return float(values.sum()) if not values.empty else 0.0

    @staticmethod
    def _first_text(frame: pd.DataFrame, column: str) -> str:
        if column not in frame:
            return ""
        values = frame[column].dropna().astype(str)
        values = values[values.str.strip() != ""]
        return values.iloc[0] if not values.empty else ""

    @staticmethod
    def _truthy(value: Any) -> bool:
        if value is None or (
            not isinstance(value, (list, tuple, dict)) and pd.isna(value)
        ):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)

    @staticmethod
    def _scalar_text(value: Any) -> str:
        if value is None or (
            not isinstance(value, (list, tuple, dict)) and pd.isna(value)
        ):
            return ""
        return str(value)

    @classmethod
    def _record_text(cls, record: Mapping[str, Any] | None, field: str) -> str:
        return "" if record is None else cls._scalar_text(record.get(field))

    @staticmethod
    def _record_number(record: Mapping[str, Any] | None, field: str) -> float:
        if record is None:
            return np.nan
        try:
            value = float(record.get(field, np.nan))
        except (TypeError, ValueError):
            return np.nan
        return value if np.isfinite(value) else np.nan

    def _write_peak_table(
        self,
        participant_id: str,
        phase: str,
        channel: str,
        ecg: DetectedPeaks,
        ppg: DetectedPeaks,
        matches: list[tuple[int, int, float]],
    ) -> None:
        matched = {
            ecg_index: (ppg_index, pat_ms) for ecg_index, ppg_index, pat_ms in matches
        }
        rows = []
        for ecg_index, ecg_timestamp in enumerate(ecg.timestamps_ms):
            ppg_match = matched.get(ecg_index)
            rows.append(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "sensor_channel": channel,
                    "ecg_peak_index": int(ecg.sample_indices[ecg_index]),
                    "ecg_peak_timestamp_ms": float(ecg_timestamp),
                    "ppg_peak_index": (
                        int(ppg.sample_indices[ppg_match[0]])
                        if ppg_match is not None
                        else None
                    ),
                    "ppg_peak_timestamp_ms": (
                        float(ppg.timestamps_ms[ppg_match[0]])
                        if ppg_match is not None
                        else None
                    ),
                    "pat_ms": ppg_match[1] if ppg_match is not None else None,
                    "matched": ppg_match is not None,
                    "peak_role": "matched"
                    if ppg_match is not None
                    else "ecg_unmatched",
                    "validation_window_seconds": self._validation_window_seconds(),
                }
            )
        matched_ppg_indices = {ppg_index for _, ppg_index, _ in matches}
        for ppg_index, ppg_timestamp in enumerate(ppg.timestamps_ms):
            if ppg_index in matched_ppg_indices:
                continue
            rows.append(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "sensor_channel": channel,
                    "ecg_peak_index": None,
                    "ecg_peak_timestamp_ms": None,
                    "ppg_peak_index": int(ppg.sample_indices[ppg_index]),
                    "ppg_peak_timestamp_ms": float(ppg_timestamp),
                    "pat_ms": None,
                    "matched": False,
                    "peak_role": "ppg_unmatched",
                    "validation_window_seconds": self._validation_window_seconds(),
                }
            )
        output = self.output_dir / "peak_tables" / participant_id / phase
        output.mkdir(parents=True, exist_ok=True)
        peak_table = pd.DataFrame(rows)
        peak_table.to_parquet(output / f"{channel}.parquet", index=False)
        peak_table.to_csv(output / f"{channel}.csv", index=False)

    def _update_readiness(self, summary: dict[str, Any]) -> None:
        readiness_path = self.paths.gold_dir / "data_readiness.json"
        if not readiness_path.is_file():
            return
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        expected_run_id = (
            str(self.config.artifacts.get("run_id", "")) if self.config else ""
        )
        readiness_run_id = str(readiness.get("run_id", ""))
        if expected_run_id and readiness_run_id != expected_run_id:
            LOGGER.warning(
                "Not updating stale readiness artifact: expected=%s actual=%s",
                expected_run_id,
                readiness_run_id,
            )
            return
        total = int(summary.get("total_combinations", 0))
        readiness["ecg_validation_status"] = (
            "ready" if summary.get("hrv_validation_decision") == "GO" else "blocked"
        )
        readiness["ecg_validation_reason"] = (
            "hrv_prv_primary_gate_passed"
            if readiness["ecg_validation_status"] == "ready"
            else "hrv_prv_primary_gate_not_passed"
        )
        readiness["ecg_validation_combinations"] = total
        readiness.update(
            {
                "run_id": self._run_identity["run_id"],
                "source_tree_sha256": self._run_identity["source_tree_sha256"],
                "run_contract_sha256": self._run_identity["run_contract_sha256"],
            }
        )
        readiness["beat_diagnostic_status"] = summary.get(
            "beat_diagnostic_status", "not_evaluated"
        )
        readiness["beat_diagnostic_channel_statuses"] = summary.get(
            "channel_decisions", {}
        )
        readiness["hrv_validation_decision"] = summary.get(
            "hrv_validation_decision", "INSUFFICIENT_EVIDENCE"
        )
        readiness["validation_channel_decision"] = summary.get("recommended_channel")
        readiness["hrv_validation_channel_statuses"] = summary.get(
            "hrv_channel_decisions", {}
        )
        readiness["spectral_validation_channel_statuses"] = summary.get(
            "spectral_validation_channel_statuses", {}
        )
        readiness["ecg_validation_channel_status"] = {
            channel: (
                "ready"
                if decision == "GO"
                else (
                    "insufficient_evidence"
                    if decision == "INSUFFICIENT_EVIDENCE"
                    else "blocked"
                )
            )
            for channel, decision in readiness[
                "hrv_validation_channel_statuses"
            ].items()
        }
        readiness.setdefault(
            "model_release_status_by_channel",
            dict(readiness.get("baseline_ml_reports", {})),
        )
        readiness["status"] = (
            "ready"
            if readiness.get("ppg_gold_status") == "ready"
            and readiness.get("training_status") == "ready"
            and readiness["ecg_validation_status"] == "ready"
            else "blocked"
        )
        self._write_json(readiness_path, readiness)

    def _write_channel_recommendation(self, metrics: pd.DataFrame) -> dict[str, Any]:
        sync = self.config.synchronization if self.config else None
        thresholds = {
            "minimum_valid_fraction": (
                float(sync.recommendation_min_valid_fraction) if sync else 0.80
            ),
            "minimum_median_beat_f1": (
                float(sync.recommendation_min_median_beat_f1) if sync else 0.80
            ),
            "maximum_median_ibi_mae_ms": (
                float(sync.recommendation_max_median_ibi_mae_ms) if sync else 100.0
            ),
            "maximum_median_hr_mae_bpm": (
                float(sync.recommendation_max_median_hr_mae_bpm) if sync else 5.0
            ),
        }
        rows = [
            self._channel_recommendation_row(metrics, channel, thresholds)
            for channel in PPG_CHANNELS
        ]
        eligible = [row for row in rows if row["decision"] == "go"]
        eligible.sort(
            key=lambda row: (
                -float(row["median_beat_f1"]),
                float(row["median_ibi_mae_ms"]),
                str(row["sensor_channel"]),
            )
        )
        payload = {
            "schema_version": "beat-diagnostic-channel-recommendation-v2",
            "decision_role": "beat_diagnostic_only",
            "controls_primary_go_no_go": False,
            "validation_window_seconds": self._validation_window_seconds(),
            "bootstrap_iterations": (int(sync.bootstrap_iterations) if sync else 2000),
            "bootstrap_seed": int(sync.bootstrap_seed) if sync else 42,
            "thresholds": thresholds,
            "recommended_channel": (
                eligible[0]["sensor_channel"] if eligible else None
            ),
            "training_gate_independent": True,
            "channels": rows,
        }
        recommendation_frame = pd.DataFrame(rows)
        recommendation_frame.to_csv(
            self.output_dir / "channel_recommendation.csv", index=False
        )
        self._write_json(self.output_dir / "channel_recommendation.json", payload)
        markdown = self._channel_recommendation_markdown(payload)
        (self.output_dir / "channel_recommendation.md").write_text(
            markdown, encoding="utf-8"
        )
        # Keep one compatibility pointer for existing manifest readers.
        legacy = payload["recommended_channel"] or "NO_GO"
        (self.paths.validation_dir / "channel_recommendation.txt").write_text(
            str(legacy), encoding="utf-8"
        )
        return payload

    def _channel_recommendation_row(
        self,
        metrics: pd.DataFrame,
        channel: str,
        thresholds: Mapping[str, float],
    ) -> dict[str, Any]:
        selected = metrics[metrics["sensor_channel"] == channel]
        valid = selected[selected["status"] == "ok"].copy()
        required_metrics = ["beat_f1", "ibi_mae_ms", "hr_mae_bpm"]
        valid = valid.dropna(subset=required_metrics)
        valid_fraction = float(len(valid) / max(1, len(selected)))

        def median(column: str) -> float:
            return float(valid[column].median()) if not valid.empty else float("nan")

        row: dict[str, Any] = {
            "sensor_channel": channel,
            "expected_combinations": int(len(selected)),
            "valid_combinations": int(len(valid)),
            "valid_fraction": valid_fraction,
            "median_beat_f1": median("beat_f1"),
            "median_ibi_mae_ms": median("ibi_mae_ms"),
            "median_ibi_rmse_ms": median("ibi_rmse_ms"),
            "median_hr_mae_bpm": median("hr_mae_bpm"),
            "median_bland_altman_bias_ms": median("bland_altman_bias_ms"),
        }
        row.update(self._cluster_bootstrap_intervals(valid))
        minimum_evidence = (
            int(self.config.synchronization.bootstrap_min_pairs) if self.config else 10
        )
        reasons: list[str] = []
        if len(valid) < minimum_evidence:
            decision = "insufficient_evidence"
            reasons.append(f"valid_combinations<{minimum_evidence}")
        else:
            checks = (
                (
                    valid_fraction >= thresholds["minimum_valid_fraction"],
                    "valid_fraction_below_threshold",
                ),
                (
                    row["median_beat_f1"] >= thresholds["minimum_median_beat_f1"],
                    "median_beat_f1_below_threshold",
                ),
                (
                    row["median_ibi_mae_ms"] <= thresholds["maximum_median_ibi_mae_ms"],
                    "median_ibi_mae_above_threshold",
                ),
                (
                    row["median_hr_mae_bpm"] <= thresholds["maximum_median_hr_mae_bpm"],
                    "median_hr_mae_above_threshold",
                ),
            )
            reasons.extend(reason for passed, reason in checks if not passed)
            decision = "go" if not reasons else "no_go"
        row["decision"] = decision
        row["decision_reasons"] = "|".join(reasons) if reasons else "all_gates_passed"
        return row

    def _cluster_bootstrap_intervals(
        self, valid: pd.DataFrame
    ) -> dict[str, float | str | int]:
        columns = {
            "beat_f1": "median_beat_f1",
            "ibi_mae_ms": "median_ibi_mae_ms",
            "ibi_rmse_ms": "median_ibi_rmse_ms",
            "hr_mae_bpm": "median_hr_mae_bpm",
            "bland_altman_bias_ms": "median_bland_altman_bias_ms",
        }
        empty: dict[str, float | str | int] = {
            f"{output}_ci_{bound}": np.nan
            for output in columns.values()
            for bound in ("low", "high")
        }
        minimum = (
            int(self.config.synchronization.bootstrap_min_pairs) if self.config else 10
        )
        participants = sorted(valid["participant_id"].dropna().unique().tolist())
        if len(valid) < minimum or len(participants) < 2:
            empty["cluster_ci_status"] = "insufficient_participants"
            empty["cluster_ci_iterations"] = 0
            return empty
        iterations = (
            int(self.config.synchronization.bootstrap_iterations)
            if self.config
            else 2000
        )
        seed = int(self.config.synchronization.bootstrap_seed) if self.config else 42
        rng = np.random.default_rng(seed)
        samples = {column: np.empty(iterations, dtype=float) for column in columns}
        grouped = {
            participant: valid[valid["participant_id"] == participant]
            for participant in participants
        }
        for index in range(iterations):
            chosen = rng.choice(participants, size=len(participants), replace=True)
            sample = pd.concat([grouped[str(participant)] for participant in chosen])
            for column in columns:
                samples[column][index] = float(sample[column].median())
        result: dict[str, float | str | int] = {}
        for column, output in columns.items():
            low, high = np.percentile(samples[column], [2.5, 97.5])
            result[f"{output}_ci_low"] = float(low)
            result[f"{output}_ci_high"] = float(high)
        result["cluster_ci_status"] = "ok"
        result["cluster_ci_iterations"] = iterations
        return result

    @staticmethod
    def _channel_recommendation_markdown(payload: Mapping[str, Any]) -> str:
        lines = [
            "# ECG-PPG channel recommendation",
            "",
            "This scientific agreement decision is independent from PPG Gold and training readiness.",
            "",
            f"Recommended channel: `{payload.get('recommended_channel') or 'none'}`",
            "",
            "| Channel | Decision | Valid | Beat F1 | IBI MAE (ms) | HR MAE (bpm) | Reasons |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in payload["channels"]:
            lines.append(
                f"| {row['sensor_channel']} | {row['decision']} | "
                f"{row['valid_combinations']}/{row['expected_combinations']} | "
                f"{row['median_beat_f1']:.4f} | {row['median_ibi_mae_ms']:.3f} | "
                f"{row['median_hr_mae_bpm']:.3f} | {row['decision_reasons']} |"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _summary(
        metrics: pd.DataFrame, recommendation: Mapping[str, Any]
    ) -> dict[str, Any]:
        status = metrics.get("status", pd.Series(dtype=str))
        metric_status = metrics.get("metric_status", pd.Series(dtype=str))
        strict_quality = metrics.get(
            "strict_quality_eligible", pd.Series(False, index=metrics.index)
        ).fillna(False)
        quality_status = metrics.get("quality_status", pd.Series(dtype=str))
        return {
            "validation_window_seconds": DEFAULT_VALIDATION_WINDOW_SECONDS,
            "total_combinations": int(len(metrics)),
            "expected_combinations": 32 * 2 * 3,
            "ok_combinations": int((status == "ok").sum()),
            "excluded_combinations": int((status != "ok").sum()),
            "metric_computable_combinations": int(
                metric_status.isin({"computed", "reported"}).sum()
            ),
            "strict_quality_combinations": int(strict_quality.astype(bool).sum()),
            "quality_warning_combinations": int((quality_status == "warning").sum()),
            "participants": sorted(
                metrics.get("participant_id", pd.Series(dtype=str))
                .dropna()
                .unique()
                .tolist()
            ),
            "phases": sorted(
                metrics.get("phase", pd.Series(dtype=str)).dropna().unique().tolist()
            ),
            "channels": list(PPG_CHANNELS),
            "pat_preserved": True,
            "ppg_timeline_shifted": False,
            "holdout_used_for_tuning": False,
            "recommended_channel": recommendation.get("recommended_channel"),
            "channel_decisions": {
                row["sensor_channel"]: row["decision"]
                for row in recommendation.get("channels", [])
            },
        }

    @staticmethod
    def _summary_markdown(summary: dict[str, Any]) -> str:
        return (
            "# Canonical PPG-vs-ECG validation\n\n"
            f"Validation window: `{summary['validation_window_seconds']}` seconds.\n\n"
            f"Processed `{summary['total_combinations']}` of `{summary['expected_combinations']}` "
            "participant-phase-channel combinations.\n\n"
            f"- Valid: `{summary['ok_combinations']}`\n"
            f"- Excluded: `{summary['excluded_combinations']}`\n"
            f"- Metric-computable: `{summary['metric_computable_combinations']}`\n"
            f"- Strict-quality eligible: `{summary['strict_quality_combinations']}`\n"
            f"- Quality warnings: `{summary['quality_warning_combinations']}`\n"
            f"- Recommended channel: `{summary.get('recommended_channel') or 'none'}`\n"
            "- ECG clock correction is consumed before matching; PPG timestamps are unchanged.\n"
            "- PAT is preserved and matching is one-to-one.\n"
        )

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.paths.project_root)).replace("\\", "/")
        except ValueError:
            return str(path)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, default=PpgEcgValidator._json_default),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"unsupported JSON value: {type(value)!r}")


def paths_for_project(project_root: Path, config: PipelineConfig) -> PipelinePaths:
    """Compatibility boundary for the shared canonical path resolver."""

    return pipeline_paths_from_config(project_root, config)


def generate_canonical_peak_validation_reports(
    project_root: Path,
    config: PipelineConfig,
    *,
    evaluation_scope: Literal["selection", "full_cohort"] | None = None,
) -> PeakValidationOutputs:
    """CLI boundary for canonical PPG-vs-ECG validation."""

    return PpgEcgValidator(paths_for_project(project_root, config), config).run(
        evaluation_scope=evaluation_scope
    )
