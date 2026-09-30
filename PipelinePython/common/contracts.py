from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


VALID_PHASES = frozenset({"relax", "task", "unknown"})
MODEL_PHASES = frozenset({"relax", "task"})
VALID_SENSORS = frozenset({"ecg", "ppg_ir", "ppg_red", "ppg_green"})
VALID_PARTICIPANTS = frozenset(f"ID-{index:02d}" for index in range(1, 33))
VALID_QUALITY_TIERS = frozenset({"excellent", "good", "acceptable", "poor", "invalid"})
TRAINING_QUALITY_TIERS = frozenset({"excellent", "good"})
HRV_AGREEMENT_FEATURES = (
    "rmssd_ms",
    "sdnn_ms",
    "pnn50",
    "lf_power_ms2",
    "hf_power_ms2",
    "lf_hf_ratio",
)


@dataclass(frozen=True)
class RawAuditIssue:
    """One lossless raw-ingestion issue with file and sample provenance."""

    participant_id: str
    session_id: str
    phase: str
    sensor_channel: str
    source_file: str
    source_checksum: str
    issue_type: str = ""
    row_index: int | None = None
    raw_sample_index: int | None = None
    column: str = ""
    reason: str = ""
    run_id: str = ""
    phase_source: str | None = None
    severity: Literal["info", "warning", "error"] = "warning"
    issue_code: str | None = None

    def __post_init__(self) -> None:
        if not self.issue_type and self.issue_code:
            object.__setattr__(self, "issue_type", self.issue_code)
        if not self.issue_code and self.issue_type:
            object.__setattr__(self, "issue_code", self.issue_type)
        if self.issue_code and self.issue_type != self.issue_code:
            raise ValueError("issue_type_and_issue_code_mismatch")
        assert self.participant_id in VALID_PARTICIPANTS
        assert self.phase in VALID_PHASES
        assert self.sensor_channel in VALID_SENSORS or self.sensor_channel == "unknown"
        assert self.source_file
        assert self.source_checksum
        assert self.issue_type
        assert self.severity in {"info", "warning", "error"}


@dataclass(frozen=True)
class RecordingSelection:
    """Explicit source selection for one participant/phase recording."""

    participant_id: str
    phase: str
    recording_id: str
    source_files: Mapping[str, str] = field(default_factory=dict)
    status: Literal[
        "selected", "SELECTED", "LINEAGE_AMBIGUOUS", "missing", "MISSING", "INVALID"
    ] = "selected"
    exclusion_reason: str | None = None
    source_checksums: Mapping[str, str] = field(default_factory=dict)
    file_order: tuple[str, ...] = ()
    overlap: bool = False

    def __post_init__(self) -> None:
        assert self.participant_id in VALID_PARTICIPANTS
        assert self.phase in VALID_PHASES
        assert self.recording_id
        if self.status not in {"selected", "SELECTED"}:
            assert self.exclusion_reason


@dataclass(frozen=True)
class ClockCorrection:
    """Immutable ECG clock correction and primary-validation contract."""

    corrected_unix_ms: NDArray[np.float64]
    alpha: float
    beta_seconds: float
    excluded: bool
    exclusion_reason: str | None
    timebase_source: str = "raw_timestamp"
    observed_sample_count: int | None = None
    declared_sample_count: int | None = None
    declared_duration_seconds: float | None = None
    metadata_stop_unix_ms: float | None = None
    nominal_sample_rate_hz: float | None = None
    effective_sample_rate_hz: float | None = None
    effective_rate_relative_error: float | None = None
    candidate_start_unix_ms: float | None = None
    candidate_end_unix_ms: float | None = None
    clock_selection_basis: str = "metadata_only"
    pat_slope_ms_per_minute: float = 0.0
    residual_correction_status: str = "not_evaluated"
    raw_unix_ms: NDArray[np.float64] | None = None
    acceptance_reason: str | None = None
    ecg_interpolated_sample_count: int = 0
    reference_status: str = "not_evaluated"
    source_checksum: str | None = None

    def __post_init__(self) -> None:
        assert 0.98 <= self.alpha <= 1.02
        assert self.ecg_interpolated_sample_count == 0
        if self.excluded:
            assert self.exclusion_reason
        if self.raw_unix_ms is not None:
            assert self.raw_unix_ms.ndim == 1
            assert self.raw_unix_ms.size == self.corrected_unix_ms.size


@dataclass(frozen=True)
class GapEvent:
    """Observed signal gap; no missing waveform samples are synthesized."""

    start_ms: float
    end_ms: float
    duration_ms: float
    sample_count: int
    gap_type: str
    action: str
    affects_peak_region: bool


@dataclass(frozen=True)
class DetectedPeaks:
    """Detector output with observed-sample and gap provenance."""

    timestamps_ms: NDArray[np.float64]
    signal: NDArray[np.float64]
    sample_indices: NDArray[np.int64]
    fs_hz: float
    long_gap_detected: bool = False
    short_gap_interpolated: bool = False
    method: str = ""
    fallback_reason: str = ""
    gap_mask: NDArray[np.bool_] | None = None
    gap_events: tuple[GapEvent, ...] = ()
    interpolated_sample_count: int = 0
    trough_indices: NDArray[np.int64] | None = None
    reference_status: str = "not_evaluated"
    quality_flags: tuple[str, ...] = ()
    library_version: str = "unknown"
    config_fingerprint: str = ""
    status: Literal["AVAILABLE", "UNAVAILABLE"] = "AVAILABLE"
    timestamp_jitter_count: int = 0
    raw_index_gap_count: int = 0


@dataclass(frozen=True)
class MatchResult:
    """Monotonic one-to-one ECG--PPG assignment with PAT provenance."""

    matches: tuple[tuple[int, int, float], ...]
    pat_median_ms: float | None
    pat_iqr_ms: float | None
    pat_mad_ms: float | None
    pat_slope_ms_per_minute: float
    unmatched_ecg_count: int
    unmatched_ppg_count: int
    method: str
    rejection_reason: str | None = None
    unmatched_ecg_indices: tuple[int, ...] = ()
    unmatched_ppg_indices: tuple[int, ...] = ()
    matched_block_count: int = 0


@dataclass(frozen=True)
class RunContractFingerprint:
    """Complete identity of one raw-to-model run."""

    run_id: str
    source_checksums: Mapping[str, str]
    source_tree_sha256: str
    git_head: str
    git_state: str
    config_fingerprint: str
    preprocessing_version: str
    feature_version: str
    split_fingerprint: str
    package_versions: Mapping[str, str]
    artifact_checksums: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReferencePeakSet:
    """Audited reference peak stream retained independently of PPG peaks."""

    participant_id: str
    phase: str
    recording_id: str
    sensor_channel: Literal["ecg", "ppg_ir", "ppg_red", "ppg_green"]
    sample_indices: np.ndarray
    timestamps_ms: np.ndarray
    method: str
    library_version: str
    status: Literal["valid", "CONSENSUS", "REFERENCE_UNCERTAIN", "UNAVAILABLE"] = (
        "valid"
    )
    quality_flags: tuple[str, ...] = ()
    config_fingerprint: str = ""
    source_file: str = ""
    source_checksum: str = ""

    def __post_init__(self) -> None:
        assert self.participant_id in VALID_PARTICIPANTS
        assert self.phase in MODEL_PHASES
        assert self.sample_indices.ndim == self.timestamps_ms.ndim == 1
        assert self.sample_indices.size == self.timestamps_ms.size
        assert self.method and self.library_version


@dataclass(frozen=True)
class QualityDecision:
    """Pre/post peak quality decision with explicit abstention semantics."""

    status: Literal["valid", "excluded", "UNAVAILABLE"]
    tier: str
    score: float
    flags: tuple[str, ...] = ()
    exclusion_reason: str | None = None
    morphology_valid_ratio: float | None = None
    motion_status: str = "unavailable"
    structural_status: Literal["pass", "fail", "unavailable"] = "unavailable"
    metric_status: Literal["computed", "unavailable"] = "unavailable"
    quality_status: Literal["pass", "warning", "fail", "unavailable"] = "unavailable"
    strict_quality_eligible: bool = False
    validation_status: Literal["valid", "excluded"] = "excluded"

    def __post_init__(self) -> None:
        assert self.tier in VALID_QUALITY_TIERS
        assert 0.0 <= self.score <= 1.0
        if self.status != "valid":
            assert self.exclusion_reason
        if self.quality_status != "pass":
            assert not self.strict_quality_eligible
        if self.validation_status == "valid":
            assert self.status == "valid"


@dataclass(frozen=True)
class ObservedIntervalBlock:
    """One adjacent observed RR/PPI interval with block provenance."""

    participant_id: str
    phase: str
    sensor_channel: str
    signal_role: Literal["ecg", "ppg"]
    interval_index: int
    block_id: int | None
    start_peak_timestamp_ms: float
    end_peak_timestamp_ms: float
    interval_ms: float
    physiological: bool
    gap_crossed: bool
    moving_median_valid: bool
    mad_valid: bool
    eligible: bool
    exclusion_reason: str | None = None
    spectral_support_id: int | None = None
    run_id: str = ""
    recording_id: str = ""
    source_file: str = ""
    source_checksum: str = ""
    detector_method: str = ""
    detector_version: str = ""
    config_fingerprint: str = ""
    run_contract_sha256: str = ""

    def __post_init__(self) -> None:
        assert self.participant_id in VALID_PARTICIPANTS
        assert self.phase in MODEL_PHASES
        assert self.sensor_channel in VALID_SENSORS - {"ecg"}
        assert self.end_peak_timestamp_ms > self.start_peak_timestamp_ms
        assert self.interval_ms > 0.0
        if self.eligible:
            assert self.block_id is not None
            assert self.exclusion_reason is None
        else:
            assert self.block_id is None
            assert self.exclusion_reason
        if self.gap_crossed:
            assert self.spectral_support_id is None


@dataclass(frozen=True)
class HrvFeaturePair:
    """Independent ECG-HRV and PPG-PRV features for one canonical group."""

    participant_id: str
    phase: str
    sensor_channel: str
    status: Literal["COMPUTED", "PARTIAL", "EXCLUDED", "INSUFFICIENT_EVIDENCE"]
    ecg_features: Mapping[str, float] = field(default_factory=dict)
    ppg_features: Mapping[str, float] = field(default_factory=dict)
    ecg_reference_features: Mapping[str, float] = field(default_factory=dict)
    ecg_reference_feature_statuses: Mapping[str, str] = field(default_factory=dict)
    feature_statuses: Mapping[str, str] = field(default_factory=dict)
    feature_exclusion_reasons: Mapping[str, str] = field(default_factory=dict)
    common_observed_seconds: float = 0.0
    exclusion_reason: str | None = None
    method_version: str = "block_safe_hrv_prv_v3"
    analysis_duration_label: str = "short_term_240s"
    pnn50_unit: Literal["fraction"] = "fraction"
    ecg_spectral_corrected_interval_count: int = 0
    ppg_spectral_corrected_interval_count: int = 0
    ecg_spectral_correction_fraction: float | None = None
    ppg_spectral_correction_fraction: float | None = None
    ecg_spectral_support_seconds: float | None = None
    ppg_spectral_support_seconds: float | None = None
    tachogram_interpolation_method: Literal["bounded_pchip", "none"] = "none"

    def __post_init__(self) -> None:
        assert self.participant_id in VALID_PARTICIPANTS
        assert self.phase in MODEL_PHASES
        assert self.sensor_channel in VALID_SENSORS - {"ecg"}
        if self.ecg_features or self.ppg_features:
            assert set(self.ecg_features) == set(HRV_AGREEMENT_FEATURES)
            assert set(self.ppg_features) == set(HRV_AGREEMENT_FEATURES)
            assert set(self.feature_statuses) == set(HRV_AGREEMENT_FEATURES)
            assert set(self.feature_exclusion_reasons) == set(HRV_AGREEMENT_FEATURES)
        if self.ecg_reference_features:
            assert set(self.ecg_reference_features) == set(HRV_AGREEMENT_FEATURES)
            assert set(self.ecg_reference_feature_statuses) == set(
                HRV_AGREEMENT_FEATURES
            )
        if self.status == "COMPUTED":
            assert all(np.isfinite(value) for value in self.ecg_features.values())
            assert all(np.isfinite(value) for value in self.ppg_features.values())
            assert set(self.feature_statuses.values()) == {"COMPUTED"}
            assert self.exclusion_reason is None
        elif self.status == "PARTIAL":
            computed = {
                feature
                for feature, status in self.feature_statuses.items()
                if status == "COMPUTED"
            }
            assert computed and computed != set(HRV_AGREEMENT_FEATURES)
            assert self.exclusion_reason
        else:
            assert self.exclusion_reason


@dataclass(frozen=True)
class HrvAgreementMetric:
    """Hardened legacy-composite result for one feature and scope."""

    sensor_channel: str
    feature: str
    scope: Literal["pooled", "relax", "task"]
    sample_count: int
    pearson_r: float | None
    spearman_rho: float | None
    icc_a1: float | None
    mae: float | None
    ecg_development_range: float | None
    accuracy_component: float | None
    composite_score: float | None
    ccc: float | None
    bias: float | None
    rmse: float | None
    loa_low: float | None
    loa_high: float | None
    status: Literal["COMPUTED", "INSUFFICIENT_EVIDENCE"]
    bland_altman_status: Literal["COMPUTED", "INSUFFICIENT_EVIDENCE"] = (
        "INSUFFICIENT_EVIDENCE"
    )
    proportional_bias_slope: float | None = None
    proportional_bias_p: float | None = None
    repeated_measure_participant_count: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        assert self.sensor_channel in VALID_SENSORS - {"ecg"}
        if self.status == "COMPUTED":
            required = (
                self.pearson_r,
                self.spearman_rho,
                self.icc_a1,
                self.mae,
                self.ecg_development_range,
                self.accuracy_component,
                self.composite_score,
            )
            assert all(value is not None and np.isfinite(value) for value in required)
            assert self.reason is None
        else:
            assert self.reason


@dataclass(frozen=True)
class HrvValidationDecision:
    """Per-channel HRV-PRV decision with spectral diagnostics."""

    sensor_channel: str
    evaluation_scope: Literal["selection", "full_cohort"]
    decision: Literal["GO", "NO_GO", "INSUFFICIENT_EVIDENCE"]
    valid_pooled_rows: int
    valid_relax_rows: int
    valid_task_rows: int
    minimum_composite: float | None
    mean_composite: float | None
    time_domain_decision: Literal["GO", "NO_GO", "INSUFFICIENT_EVIDENCE"]
    valid_time_domain_pooled_rows: int
    valid_time_domain_relax_rows: int
    valid_time_domain_task_rows: int
    time_domain_minimum_composite: float | None
    time_domain_mean_composite: float | None
    spectral_validation_status: Literal["GO", "NO_GO", "INSUFFICIENT_EVIDENCE"]
    valid_spectral_pooled_rows: int
    valid_spectral_relax_rows: int
    valid_spectral_task_rows: int
    spectral_minimum_composite: float | None
    spectral_mean_composite: float | None
    reasons: tuple[str, ...] = ()
    time_domain_reasons: tuple[str, ...] = ()
    spectral_reasons: tuple[str, ...] = ()


SIGNAL_REQUIRED_COLUMNS = (
    "participant_id",
    "session_id",
    "recording_id",
    "phase",
    "sensor_channel",
    "timestamp_raw_utc",
    "timestamp_corrected_utc",
    "timestamp_local",
    "elapsed_seconds",
    "sampling_rate_hz",
    "raw_value",
    "clean_value",
    "quality_score",
    "quality_tier",
    "quality_flags",
    "source_file",
    "source_checksum",
    "exclusion_reason",
    "preprocessing_version",
)

CANONICAL_LINEAGE_COLUMNS = (
    "participant_id",
    "session_id",
    "phase",
    "sensor_channel",
    "source_file",
    "source_checksum",
    "sampling_rate_hz",
    "start_timestamp_utc",
    "end_timestamp_utc",
    "window_id",
    "quality_score",
    "quality_tier",
    "quality_flags",
    "exclusion_reason",
    "preprocessing_version",
)


@dataclass(frozen=True)
class SignalFrameContract:
    """Executable schema and timestamp contract for one lineage group."""

    required_columns: tuple[str, ...] = SIGNAL_REQUIRED_COLUMNS

    def assert_valid(self, frame: pd.DataFrame) -> None:
        missing = sorted(set(self.required_columns) - set(frame.columns))
        assert not missing, f"signal frame missing columns: {missing}"
        assert not frame.empty, "signal frame must not be empty"
        assert set(frame["participant_id"].dropna().astype(str)) <= VALID_PARTICIPANTS
        assert set(frame["phase"].dropna().astype(str)) <= VALID_PHASES
        assert "routine" not in set(frame["phase"].dropna().astype(str).str.lower())
        assert set(frame["sensor_channel"].dropna().astype(str)) <= VALID_SENSORS
        corrected = pd.to_datetime(frame["timestamp_corrected_utc"], utc=True)
        assert corrected.notna().all(), "corrected timestamps must be parseable"
        assert corrected.is_monotonic_increasing, (
            "corrected timestamps must be monotonic"
        )
        values = pd.to_numeric(frame["raw_value"], errors="coerce").to_numpy(
            dtype=float
        )
        assert np.isfinite(values).any(), (
            "signal requires at least one finite raw value"
        )


@dataclass(frozen=True)
class ManifestRecord:
    """Immutable raw-file audit record used by Bronze and readiness reports."""

    participant_id: str
    session_id: str
    phase: str
    sensor_channel: str
    source_file: str
    source_checksum: str
    sampling_rate_hz: float
    start_timestamp_utc: str | None
    end_timestamp_utc: str | None
    row_count: int
    status: str
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        assert self.participant_id in VALID_PARTICIPANTS
        # Raw/audit manifests preserve an unknown phase for traceability. The
        # Gold/model contracts below still restrict training inputs to the
        # canonical relax/task phases.
        assert self.phase in VALID_PHASES
        assert self.sensor_channel in VALID_SENSORS
        assert self.source_checksum and len(self.source_checksum) == 64
        assert self.sampling_rate_hz >= 0.0
        assert self.row_count >= 0
        if self.status != "ok":
            assert self.exclusion_reason


@dataclass(frozen=True)
class GoldWindowContract:
    """Contract for a feature/waveform window that may enter ML preparation."""

    participant_id: str
    session_id: str
    phase: str
    sensor_channel: str
    window_id: str
    quality_score: float
    quality_tier: str
    source_file: str
    source_checksum: str
    preprocessing_version: str
    feature_version: str
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        assert self.participant_id in VALID_PARTICIPANTS
        assert self.phase in MODEL_PHASES
        assert self.sensor_channel in VALID_SENSORS - {"ecg"}
        assert self.window_id
        assert 0.0 <= self.quality_score <= 1.0
        assert self.quality_tier in VALID_QUALITY_TIERS
        assert self.source_checksum and len(self.source_checksum) == 64
        if self.quality_tier in {"poor", "invalid"}:
            assert self.exclusion_reason


@dataclass(frozen=True)
class SynchronizationResult:
    participant_id: str
    recording_id: str
    alpha: float
    beta_seconds: float
    method: str
    confidence: float
    overlap_seconds: float
    ecg_peak_count: int
    ppg_peak_count: int
    matched_peak_count: int
    median_pat_ms: float | None
    excluded: bool
    exclusion_reason: str | None = None
    pre_alignment_score: float | None = None
    post_alignment_score: float | None = None
    clock_selection_basis: str = "unavailable"
    observed_sample_count: int | None = None
    declared_sample_count: int | None = None
    declared_duration_seconds: float | None = None
    metadata_stop_unix_ms: float | None = None
    nominal_sample_rate_hz: float | None = None
    effective_sample_rate_hz: float | None = None
    effective_rate_relative_error: float | None = None
    candidate_start_unix_ms: float | None = None
    candidate_end_unix_ms: float | None = None
    reference_signal_role: str = "diagnostic_only"
    ppg_diagnostic_reason: str | None = None
    clock_excluded: bool | None = None
    clock_exclusion_reason: str | None = None
    diagnostic_reference_channel: str | None = None
    pat_iqr_ms: float | None = None
    pat_mad_ms: float | None = None
    pat_slope_ms_per_minute: float = 0.0
    unmatched_ecg_count: int = 0
    unmatched_ppg_count: int = 0
    matcher_method: str = "monotonic_pat_constant"
    residual_correction_status: str = "not_evaluated"
    reference_status: str = "not_evaluated"
    timebase_source: str = "unavailable"

    def __post_init__(self) -> None:
        assert 0.98 <= self.alpha <= 1.02
        assert 0.0 <= self.confidence <= 1.0
        assert self.overlap_seconds >= 0.0
        assert (
            min(self.ecg_peak_count, self.ppg_peak_count, self.matched_peak_count) >= 0
        )
        assert self.unmatched_ecg_count >= 0
        assert self.unmatched_ppg_count >= 0
        assert np.isfinite(self.pat_slope_ms_per_minute)
        if self.excluded:
            assert self.exclusion_reason
        elif self.median_pat_ms is not None:
            assert 100.0 <= self.median_pat_ms <= 400.0
        if self.clock_excluded is not None:
            assert self.clock_excluded == self.excluded
            assert self.clock_exclusion_reason == self.exclusion_reason


@dataclass(frozen=True)
class FeatureWindow:
    participant_id: str
    session_id: str
    recording_id: str
    phase: str
    sensor_channel: str
    window_id: str
    window_start_utc: str
    window_end_utc: str
    sampling_rate_hz: float
    label: int
    quality_score: float
    quality_tier: str
    feature_version: str
    source_file: str
    features: Mapping[str, float]

    def __post_init__(self) -> None:
        assert self.phase in MODEL_PHASES
        assert self.sensor_channel in VALID_SENSORS - {"ecg"}
        assert self.label == (0 if self.phase == "relax" else 1)
        assert self.quality_tier in VALID_QUALITY_TIERS
        assert 0.0 <= self.quality_score <= 1.0
        assert self.features and all(
            np.isfinite(float(v)) for v in self.features.values()
        )


@dataclass(frozen=True)
class FoldResult:
    participant_id: str
    model_name: str
    sensor_channel: str
    balanced_accuracy: float
    f1: float
    roc_auc: float
    threshold: float
    sample_count: int

    def __post_init__(self) -> None:
        for value in (self.balanced_accuracy, self.f1, self.roc_auc, self.threshold):
            assert 0.0 <= value <= 1.0
        assert self.sample_count > 0


@dataclass(frozen=True)
class CandidateBundleManifest:
    model_version: str
    preprocessing_version: str
    feature_version: str
    sensor_channel: str
    feature_order: tuple[str, ...]
    label_order: tuple[str, str]
    probability_threshold: float
    model_sha256: str
    development_participants: tuple[str, ...]
    holdout_participants: tuple[str, ...]
    native_model_path: str
    onnx_model_path: str | None = None

    def __post_init__(self) -> None:
        assert re.fullmatch(
            r"stress_v\d+\.\d+\.\d+(?:[-+][A-Za-z0-9._-]+)?", self.model_version
        )
        assert self.sensor_channel in VALID_SENSORS - {"ecg"}
        assert self.feature_order and len(set(self.feature_order)) == len(
            self.feature_order
        )
        assert self.label_order == ("relax", "task")
        assert 0.0 <= self.probability_threshold <= 1.0
        assert len(self.model_sha256) == 64
        assert len(self.development_participants) == 28
        assert len(self.holdout_participants) == 4
        assert set(self.development_participants).isdisjoint(self.holdout_participants)
        assert not Path(self.native_model_path).is_absolute()
        if self.onnx_model_path is not None:
            assert not Path(self.onnx_model_path).is_absolute()


@dataclass(frozen=True)
class OnnxParityResult:
    checked_samples: int
    max_abs_diff: float
    mean_abs_diff: float
    model_size_bytes: int
    batch1_latency_ms: float
    passed: bool

    def __post_init__(self) -> None:
        assert self.checked_samples >= 0
        assert self.max_abs_diff >= 0.0 and self.mean_abs_diff >= 0.0
        assert self.model_size_bytes > 0 and self.batch1_latency_ms >= 0.0
        assert self.passed == (
            self.checked_samples >= 100 and self.max_abs_diff < 1.0e-5
        )


def assert_feature_tensor(
    tensor: NDArray[np.floating],
    *,
    n_features: int,
) -> NDArray[np.float32]:
    """Validate the public ONNX input contract and return contiguous float32."""

    array = np.asarray(tensor, dtype=np.float32)
    assert array.ndim == 2, f"float_input must be rank 2, got {array.shape}"
    assert array.shape[0] > 0 and array.shape[1] == n_features
    assert np.isfinite(array).all(), "float_input contains NaN/Inf"
    return np.ascontiguousarray(array)


def assert_stress_probability(
    probability: Sequence[float] | NDArray[np.floating],
    *,
    batch_size: int,
) -> NDArray[np.float32]:
    """Validate the public ONNX output contract."""

    array = np.asarray(probability, dtype=np.float32).reshape(-1)
    assert array.shape == (batch_size,)
    assert np.isfinite(array).all()
    assert ((0.0 <= array) & (array <= 1.0)).all()
    return array


# Bronze schema — immutable raw ingestion columns (no SQA yet)
CANONICAL_BRONZE_COLUMNS = [
    "participant_code",
    "session_id",
    "signal_type",
    "timestamp_utc",
    "timestamp_local",
    "sample_index",
    "raw_value",
    "sampling_rate_hz",
    "phase",
    "anchor_start",
    "anchor_end",
    "quality_flag",
    "preprocessing_version",
]

# Silver schema — cleaned + SQA-annotated (superset of bronze)
CANONICAL_SIGNAL_COLUMNS = [
    "participant_code",
    "session_id",
    "signal_type",
    "timestamp_utc",
    "timestamp_local",
    "sample_index",
    "raw_value",
    "sampling_rate_hz",
    "phase",
    "anchor_start",
    "anchor_end",
    "quality_flag",  # string flags e.g. "FLATLINE|TIME_GAP" or "PASS"
    "quality_score",  # float 0.0–1.0, fraction of SQA checks passed
    "quality_tier",  # "excellent" | "good" | "acceptable" | "poor" | "invalid"
    "preprocessing_version",
]

CANONICAL_WINDOW_COLUMNS = [
    "participant_code",
    "session_id",
    "phase",
    "window_id",
    "window_start_utc",
    "window_end_utc",
    "label",
    "quality_flag",
    "feature_version",
]

VALIDATION_COLUMNS = [
    "participant_code",
    "phase",
    "channel",
    "precision",
    "recall",
    "f1",
    "peak_mae_ms",
    "ibi_mae_ms",
    "ibi_rmse_ms",
    "pearson_r",
    "spearman_r",
    "bland_altman_bias_ms",
    "bland_altman_loa_low_ms",
    "bland_altman_loa_high_ms",
    "n_pairs",
]


@dataclass(frozen=True)
class ModelManifest:
    preprocessing_version: str
    feature_version: str
    validation_channel_decision: str
    model_version: str
    export_version: str
    acceptance_metrics: dict


class Augmentation(ABC):
    """Physiologically-constrained signal augmentation interface."""

    @abstractmethod
    def apply(self, signal: np.ndarray, label: int) -> np.ndarray:
        """Apply transform to 1D raw/filtered biosignal waveform."""
        pass


class InterpolationEngine(ABC):
    """Cubic spline / resample interpolation engine interface."""

    @abstractmethod
    def interpolate_ibi(
        self, ibi_series: np.ndarray, missing_indices: np.ndarray
    ) -> np.ndarray:
        """Interpolate missing IBIs using cubic spline."""
        pass


class ONNXExporterContract(ABC):
    """Interface for model export ensuring probability output."""

    @abstractmethod
    def export(
        self, model: Any, output_path: str, include_probability: bool = True
    ) -> dict:
        pass

    @abstractmethod
    def verify_outputs(self, onnx_path: str) -> list[str]:
        """Return list of graph output names."""
        pass
