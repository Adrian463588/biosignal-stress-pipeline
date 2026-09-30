from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

PREPROCESSING_VERSION = "v3.1.0"
FEATURE_VERSION = "v4.0.0_robust_channel_independent"
MODEL_VERSION = "stress_v4.0.0-core240s"
TRAINING_CHANNELS = ("ppg_green", "ppg_ir", "ppg_red")


@dataclass(frozen=True)
class RawSourceConfig:
    """Immutable raw-source policy."""

    renamed_dir: Path
    trimmed_dir: Path
    anchor_csv: Path
    ecg_source: str = "renamed_complete"
    ppg_source: str = "trimmed_phase"
    trimmed_ecg_role: str = "audit_only"
    participant_count: int = 32

    def __post_init__(self) -> None:
        assert self.participant_count == 32, "v3 requires the locked 32 participants"
        assert self.renamed_dir != self.trimmed_dir
        assert self.ecg_source == "renamed_complete"
        assert self.ppg_source == "trimmed_phase"
        assert self.trimmed_ecg_role == "audit_only"


@dataclass(frozen=True)
class SynchronizationConfig:
    """Whole-record ECG clock and physiological validation contract."""

    alpha_min: float = 0.98
    alpha_max: float = 1.02
    metadata_effective_rate_tolerance_fraction: float = 0.10
    pat_min_ms: float = 100.0
    pat_max_ms: float = 400.0
    pat_tolerance_ms: float = 150.0
    pat_tolerance_sensitivity_ms: tuple[float, ...] = (100.0, 150.0, 200.0)
    matcher_method: str = "monotonic_pat_affine"
    validation_window_seconds: int = 240
    min_overlap_seconds: float = 239.0
    min_peak_count: int = 30
    min_match_count: int = 30
    ecg_waveform_interpolation: str = "disabled"
    ecg_gap_action: str = "mask_and_exclude_affected_beats"
    ecg_max_unfilled_gap_seconds: float = 0.5
    rr_interpolation_enabled: bool = False
    rr_interpolation_method: str = "pchip"
    max_lag_seconds: float = 5.0
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 42
    bootstrap_min_pairs: int = 10
    recommendation_min_valid_fraction: float = 0.80
    recommendation_min_median_beat_f1: float = 0.80
    recommendation_max_median_ibi_mae_ms: float = 100.0
    recommendation_max_median_hr_mae_bpm: float = 5.0
    pat_residual_correction_enabled: bool = True
    pat_residual_min_matches: int = 4
    pat_residual_require_stability_improvement: bool = True
    baseline_validation_run_id: str = "20260809T044902Z"
    baseline_validation_metrics_path: str = (
        "data/artifacts/runs/20260809T044902Z/validation/ppg_ecg_240s/"
        "ppg_ecg_metrics_240s.csv"
    )

    def __post_init__(self) -> None:
        assert 0.0 < self.alpha_min <= 1.0 <= self.alpha_max
        if not 0.0 < self.metadata_effective_rate_tolerance_fraction <= 0.10:
            raise ValueError(
                "ECG metadata effective-rate tolerance must be in (0, 0.10]"
            )
        assert 0.0 < self.pat_min_ms < self.pat_max_ms
        if not 0.0 < self.pat_tolerance_ms <= self.pat_max_ms - self.pat_min_ms:
            raise ValueError("PAT tolerance must fit inside the physiological range")
        if not self.pat_tolerance_sensitivity_ms or any(
            value <= 0.0 or value > self.pat_max_ms - self.pat_min_ms
            for value in self.pat_tolerance_sensitivity_ms
        ):
            raise ValueError("PAT sensitivity tolerances are invalid")
        if 150.0 not in self.pat_tolerance_sensitivity_ms:
            raise ValueError("PAT sensitivity analysis must include 150 ms")
        assert self.validation_window_seconds == 240
        assert self.min_overlap_seconds > 0.0
        assert self.min_peak_count > 1 and self.min_match_count > 1
        if self.ecg_waveform_interpolation != "disabled":
            raise ValueError("primary ECG waveform interpolation must be disabled")
        if self.ecg_gap_action != "mask_and_exclude_affected_beats":
            raise ValueError("unsupported ECG gap action")
        if self.ecg_max_unfilled_gap_seconds <= 0.0:
            raise ValueError("ECG gap limit must be positive")
        if self.rr_interpolation_method != "pchip":
            raise ValueError("RR interpolation method must remain pchip")
        if self.bootstrap_iterations < 100 or self.bootstrap_seed != 42:
            raise ValueError(
                "Validation bootstrap must use at least 100 iterations and seed 42"
            )
        if self.bootstrap_min_pairs < 2:
            raise ValueError("Validation bootstrap requires at least two pairs")
        if self.pat_residual_min_matches < 4:
            raise ValueError("PAT residual correction requires at least four matches")
        if not 0.0 < self.recommendation_min_valid_fraction <= 1.0:
            raise ValueError("Recommendation coverage threshold must be in (0, 1]")
        if not self.baseline_validation_run_id.strip():
            raise ValueError("Validation baseline run id is required")
        if not self.baseline_validation_metrics_path.strip():
            raise ValueError("Validation baseline metrics path is required")


@dataclass(frozen=True)
class HrvAgreementConfig:
    """Locked HRV-PRV feature agreement and primary decision policy."""

    interval_min_ms: float = 333.0
    interval_max_ms: float = 1500.0
    moving_median_window: int = 5
    moving_median_relative_deviation: float = 0.20
    mad_multiplier: float = 3.0
    minimum_intervals: int = 120
    minimum_successive_pairs: int = 100
    common_observed_min_seconds: float = 192.0
    spectral_min_block_seconds: float = 180.0
    spectral_max_artifact_fraction: float = 0.05
    spectral_max_consecutive_artifacts: int = 2
    spectral_max_interpolation_span_seconds: float = 3.0
    tachogram_rate_hz: float = 4.0
    lf_low_hz: float = 0.04
    lf_high_hz: float = 0.15
    hf_high_hz: float = 0.40
    validation_participant_count: int = 32
    minimum_full_cohort_coverage_fraction: float = 0.80
    primary_domain: str = "time_domain"
    pooled_composite_threshold: float = 0.70
    phase_composite_threshold: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 < self.interval_min_ms < self.interval_max_ms:
            raise ValueError("HRV interval bounds are invalid")
        if self.moving_median_window < 3 or self.moving_median_window % 2 == 0:
            raise ValueError("HRV moving-median window must be odd and >=3")
        if not 0.0 < self.moving_median_relative_deviation < 1.0:
            raise ValueError("HRV moving-median deviation must be in (0,1)")
        if self.mad_multiplier <= 0.0:
            raise ValueError("HRV MAD multiplier must be positive")
        if self.minimum_intervals < 3 or self.minimum_successive_pairs < 2:
            raise ValueError("HRV interval support is too small")
        if not 0.0 < self.common_observed_min_seconds <= 240.0:
            raise ValueError("HRV common observed support must fit the 240s core")
        if not 0.0 < self.spectral_min_block_seconds <= 240.0:
            raise ValueError("HRV spectral support must fit the 240s core")
        if self.spectral_min_block_seconds > self.common_observed_min_seconds:
            raise ValueError("HRV spectral support cannot exceed common support")
        if not 0.0 <= self.spectral_max_artifact_fraction <= 0.05:
            raise ValueError("HRV spectral artifact fraction must remain in [0,0.05]")
        if self.spectral_max_consecutive_artifacts < 0:
            raise ValueError("HRV consecutive spectral artifacts cannot be negative")
        if self.spectral_max_interpolation_span_seconds <= 0.0:
            raise ValueError("HRV spectral interpolation span must be positive")
        if not 0.0 < self.lf_low_hz < self.lf_high_hz < self.hf_high_hz:
            raise ValueError("HRV frequency bands are invalid")
        if self.tachogram_rate_hz <= 2.0 * self.hf_high_hz:
            raise ValueError("HRV tachogram rate violates the Nyquist criterion")
        if self.validation_participant_count != 32:
            raise ValueError("HRV physiological validation requires 32 participants")
        if not 0.0 < self.minimum_full_cohort_coverage_fraction <= 1.0:
            raise ValueError("HRV full-cohort coverage fraction must be in (0,1]")
        if self.primary_domain != "time_domain":
            raise ValueError("HRV primary domain must remain time_domain")
        for threshold in (
            self.pooled_composite_threshold,
            self.phase_composite_threshold,
        ):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("HRV decision thresholds must be in [0,1]")


@dataclass(frozen=True)
class ChannelPeakConfig:
    selected_method: str = "neurokit2_ppg_elgendi"
    min_distance_seconds: float = 0.30
    prominence_mad_multiplier: float = 0.80
    min_width_seconds: float = 0.0
    max_width_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.selected_method.strip():
            raise ValueError("Selected peak detector method is required")
        if self.min_distance_seconds <= 0.0:
            raise ValueError("Peak distance must be positive")
        if self.prominence_mad_multiplier <= 0.0:
            raise ValueError("Peak prominence multiplier must be positive")
        if self.min_width_seconds < 0.0:
            raise ValueError("Peak minimum width cannot be negative")
        if self.max_width_seconds is not None and self.max_width_seconds <= 0.0:
            raise ValueError("Peak maximum width must be positive")


@dataclass(frozen=True)
class LegacyNativePeakConfig:
    """Development-only observed-sample ablation derived from old Green code."""

    provenance: str = "legacy_31_participant_green_reference"
    prominence_std_multiplier: float = 0.25
    height_std_multiplier: float = 0.20
    min_distance_seconds: float = 1.0 / 3.0
    edge_margin_seconds: float = 0.50
    segment_seconds: float = 30.0
    retry_prominence_scale: float = 0.30

    def __post_init__(self) -> None:
        if not self.provenance.strip():
            raise ValueError("Legacy peak provenance is required")
        if self.prominence_std_multiplier <= 0.0:
            raise ValueError("Legacy peak prominence multiplier must be positive")
        if self.height_std_multiplier <= 0.0:
            raise ValueError("Legacy peak height multiplier must be positive")
        if self.min_distance_seconds <= 0.0:
            raise ValueError("Legacy peak distance must be positive")
        if self.edge_margin_seconds < 0.0:
            raise ValueError("Legacy peak edge margin cannot be negative")
        if self.segment_seconds <= 0.0:
            raise ValueError("Legacy peak segment length must be positive")
        if not 0.0 < self.retry_prominence_scale < 1.0:
            raise ValueError("Legacy retry prominence scale must be in (0,1)")


@dataclass(frozen=True)
class PeakDetectionConfig:
    """PPG detector policy shared by Gold extraction and ECG validation."""

    primary_method: str = "neurokit2_ppg_elgendi"
    fallback_method: str = "scipy_find_peaks"
    allow_runtime_fallback: bool = False
    candidate_methods: tuple[str, ...] = (
        "neurokit2_ppg_elgendi",
        "neurokit2_ppg_bishop",
        "neurokit2_ppg_charlton_msptdfast",
        "scipy_find_peaks_adaptive",
        "scipy_legacy_green_native",
    )
    minimum_peak_count: int = 15
    physiological_ibi_min_ms: float = 300.0
    physiological_ibi_max_ms: float = 1500.0
    minimum_physiological_ibi_ratio: float = 0.90
    audit_prominence_mad_multipliers: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0)
    legacy_native: LegacyNativePeakConfig = field(
        default_factory=LegacyNativePeakConfig
    )
    channels: Mapping[str, ChannelPeakConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primary_method != "neurokit2_ppg_elgendi":
            raise ValueError("NeuroKit2 Elgendi must remain the primary PPG detector")
        if self.fallback_method != "scipy_find_peaks":
            raise ValueError("SciPy find_peaks must remain the PPG fallback")
        if not self.candidate_methods or len(set(self.candidate_methods)) != len(
            self.candidate_methods
        ):
            raise ValueError("PPG detector registry must contain unique candidates")
        if self.minimum_peak_count < 3:
            raise ValueError("PPG detector minimum peak count is too small")
        if not 0.0 < self.minimum_physiological_ibi_ratio <= 1.0:
            raise ValueError("Physiological IBI ratio must be in (0, 1]")
        if self.physiological_ibi_min_ms >= self.physiological_ibi_max_ms:
            raise ValueError("Physiological IBI bounds are invalid")
        if len(self.audit_prominence_mad_multipliers) < 2 or any(
            value <= 0.0 for value in self.audit_prominence_mad_multipliers
        ):
            raise ValueError("Peak audit requires at least two positive candidates")
        if set(self.channels) != {"ppg_green", "ppg_ir", "ppg_red"}:
            raise ValueError("Peak configuration must define Green, IR, and Red")
        unregistered = {
            value.selected_method
            for value in self.channels.values()
            if value.selected_method not in self.candidate_methods
        }
        if unregistered:
            raise ValueError(
                "Selected PPG detector is not registered: "
                + ",".join(sorted(unregistered))
            )

    def for_channel(self, channel: str) -> ChannelPeakConfig:
        try:
            return self.channels[channel]
        except KeyError as exc:
            raise ValueError(f"Unsupported PPG channel: {channel}") from exc

    def selected_method_for(self, channel: str) -> str:
        method = self.for_channel(channel).selected_method
        if method not in self.candidate_methods:
            raise ValueError(f"Selected PPG detector is not registered: {method}")
        return method


@dataclass(frozen=True)
class WindowConfig:
    source_phase_duration_seconds: int = 300
    edge_trim_start_seconds: int = 30
    edge_trim_end_seconds: int = 30
    analysis_phase_duration_seconds: int = 240
    validation_window_seconds: int = 240
    gold_window_seconds: int = 180
    gold_stride_seconds: int = 10
    size_seconds: int = 180
    step_seconds: int = 10
    validation_buffer_seconds: int = 0
    gold_buffer_seconds: int = 0
    # Deprecated compatibility field. Canonical slicing intentionally ignores
    # it so historical configs cannot reintroduce the old 10-second trim.
    anchor_buffer_seconds: int = 0
    ibi_sequence_length: int = 720

    def __post_init__(self) -> None:
        assert self.source_phase_duration_seconds == 300
        assert self.edge_trim_start_seconds == 30
        assert self.edge_trim_end_seconds == 30
        assert self.analysis_phase_duration_seconds == 240
        assert self.validation_window_seconds == 240
        assert self.gold_window_seconds == 180
        assert self.gold_stride_seconds == 10
        assert (
            self.edge_trim_start_seconds
            + self.edge_trim_end_seconds
            + self.analysis_phase_duration_seconds
            == self.source_phase_duration_seconds
        )
        assert self.size_seconds == self.gold_window_seconds
        assert self.step_seconds == self.gold_stride_seconds
        assert self.size_seconds == 180
        assert 0 < self.step_seconds <= self.size_seconds
        assert self.validation_buffer_seconds >= 0
        assert self.gold_buffer_seconds >= 0
        assert self.anchor_buffer_seconds >= 0
        assert self.ibi_sequence_length == 720

    def buffer_seconds(self, mode: Literal["validation", "gold"]) -> int:
        """Return the explicit boundary buffer for a canonical stage.

        ``anchor_buffer_seconds`` is retained only for deserializing older
        configs. It is deliberately not consulted by the canonical path.
        """

        if mode == "validation":
            return self.validation_buffer_seconds
        if mode == "gold":
            return self.gold_buffer_seconds
        raise ValueError(f"unsupported anchor slicing mode: {mode!r}")


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 42
    ml_execution: str = "local"
    training_channels: tuple[str, ...] = TRAINING_CHANNELS
    candidate_order: tuple[str, ...] = (
        "logistic_regression",
        "rbf_svm",
        "extra_trees",
        "lightgbm",
        "xgboost",
    )
    gate_metric: str = "balanced_accuracy"
    development_participants: int = 28
    parity_pool_participants: int = 4
    development_participant_ids: tuple[str, ...] = tuple(
        f"ID-{index:02d}" for index in range(1, 29)
    )
    parity_pool_participant_ids: tuple[str, ...] = tuple(
        f"ID-{index:02d}" for index in range(29, 33)
    )
    inner_group_splits: int = 5
    outer_group_splits: int = 7
    outer_fold_size: int = 4
    minimum_training_development_group_coverage: float = 0.90
    minimum_development_groups_per_phase: int = 23
    minimum_development_participants: int = 26
    hpo_top_candidates: int = 2
    optuna_trials: int = 50
    optuna_sampler: str = "tpe"
    optuna_storage_uri: str = "sqlite:///data/artifacts/optuna/training.db"
    probability_calibration: str = "sigmoid"
    correlation_pruning_threshold: float = 0.95
    window_baseline_seconds: int = 180
    window_baseline_stride_seconds: int = 10
    window_minimum_ba_gain: float = 0.02
    window_candidates: tuple[tuple[int, int], ...] = (
        (60, 30),
        (120, 30),
        (180, 10),
        (180, 30),
        (240, 240),
    )
    strict_quality_tiers: tuple[str, ...] = ("excellent", "good")
    expanded_quality_tiers: tuple[str, ...] = (
        "excellent",
        "good",
        "acceptable",
    )
    quality_cohort_tie_tolerance: float = 0.01
    reused_test_cohort_status: str = "reused_internal_test_cohort"
    clinical_validation_status: str = "external_validation_required"
    performance_policy_version: str = "ml_dl_tiered_v1"
    minimum_research_balanced_accuracy: float = 0.70
    minimum_research_f1: float = 0.70
    minimum_research_roc_auc: float = 0.70
    preferred_balanced_accuracy: float = 0.80
    preferred_f1: float = 0.78
    preferred_roc_auc: float = 0.83
    research_disclaimer: str = (
        "Model penelitian — validasi internal terbatas. Bukan diagnosis klinis."
    )
    decision_threshold: float = 0.5
    mlflow_tracking_uri: str = "sqlite:///data/artifacts/mlflow/mlflow.db"
    allowed_quality_tiers: tuple[str, ...] = ("excellent", "good", "acceptable")

    def __post_init__(self) -> None:
        if self.development_participants != 28 or self.parity_pool_participants != 4:
            raise ValueError("Approved local protocol requires a fixed 28+4 split")
        expected = tuple(f"ID-{index:02d}" for index in range(1, 33))
        actual = self.development_participant_ids + self.parity_pool_participant_ids
        if actual != expected or set(self.development_participant_ids) & set(
            self.parity_pool_participant_ids
        ):
            raise ValueError("Participant split must be ID-01..ID-28 plus ID-29..ID-32")
        assert self.random_seed == 42
        assert self.inner_group_splits == 5 and self.optuna_trials >= 50
        if self.outer_group_splits != 7 or self.outer_fold_size != 4:
            raise ValueError(
                "Outer development evaluation must use seven folds of four"
            )
        if not 0.0 < self.minimum_training_development_group_coverage <= 1.0:
            raise ValueError("Development training coverage must be in (0,1]")
        if (
            self.minimum_development_groups_per_phase,
            self.minimum_development_participants,
        ) != (23, 26):
            raise ValueError(
                "Training coverage gate must retain 23/28 per phase "
                "and at least 26/28 represented participants"
            )
        if self.hpo_top_candidates != 2:
            raise ValueError("Only the two strongest fixed baselines may enter HPO")
        if self.optuna_sampler != "tpe" or not self.optuna_storage_uri.startswith(
            "sqlite:///"
        ):
            raise ValueError("Optuna must use seeded TPE with persistent SQLite")
        if self.probability_calibration != "sigmoid":
            raise ValueError("Canonical probability calibration must be sigmoid")
        if self.correlation_pruning_threshold != 0.95:
            raise ValueError("Correlation pruning threshold must remain 0.95")
        if (
            self.window_baseline_seconds,
            self.window_baseline_stride_seconds,
        ) != (180, 10):
            raise ValueError("Canonical baseline window must remain 180/10")
        if self.window_minimum_ba_gain != 0.02:
            raise ValueError("Window replacement requires at least 0.02 BA gain")
        if self.window_candidates != (
            (60, 30),
            (120, 30),
            (180, 10),
            (180, 30),
            (240, 240),
        ):
            raise ValueError("Window ablation candidates do not match the PRD")
        if self.ml_execution != "local":
            raise ValueError("Classical ML execution must be local")
        if self.training_channels != TRAINING_CHANNELS:
            raise ValueError(
                "Training requires independent Green, IR, and Red channels"
            )
        if self.candidate_order != (
            "logistic_regression",
            "rbf_svm",
            "extra_trees",
            "lightgbm",
            "xgboost",
        ):
            raise ValueError("Classical baseline order does not match the PRD")
        if self.gate_metric != "balanced_accuracy":
            raise ValueError("Classical selection gate must be balanced_accuracy")
        if (
            self.minimum_research_balanced_accuracy,
            self.minimum_research_f1,
            self.minimum_research_roc_auc,
        ) != (0.70, 0.70, 0.70):
            raise ValueError("Minimum research gates must remain 0.70/0.70/0.70")
        if (
            self.preferred_balanced_accuracy,
            self.preferred_f1,
            self.preferred_roc_auc,
        ) != (0.80, 0.78, 0.83):
            raise ValueError("Preferred gates must remain 0.80/0.78/0.83")
        if not self.performance_policy_version.strip():
            raise ValueError("Performance policy version is required")
        if not self.research_disclaimer.strip():
            raise ValueError("Research-tier disclaimer is required")
        assert self.decision_threshold == 0.5
        assert self.mlflow_tracking_uri.startswith("sqlite:///")
        if self.reused_test_cohort_status != "reused_internal_test_cohort":
            raise ValueError("ID-29..ID-32 must be disclosed as reused internal test")
        if self.clinical_validation_status != "external_validation_required":
            raise ValueError("External clinical validation must remain required")

    @property
    def min_balanced_accuracy(self) -> float:
        """Deprecated preferred-tier alias; never use for minimum gating."""

        return self.preferred_balanced_accuracy

    @property
    def min_f1(self) -> float:
        """Deprecated preferred-tier alias; never use for minimum gating."""

        return self.preferred_f1

    @property
    def min_roc_auc(self) -> float:
        """Deprecated preferred-tier alias; never use for minimum gating."""

        return self.preferred_roc_auc


@dataclass(frozen=True)
class FeatureConfig:
    predictor_default: tuple[str, ...]
    experimental_only: tuple[str, ...]
    quality_gate_only: tuple[str, ...]
    forbidden_metadata: tuple[str, ...]
    promoted_experimental: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from .feature_policy import (
            DEFAULT_PREDICTORS,
            EXPERIMENTAL_FEATURES,
            FORBIDDEN_METADATA,
            QUALITY_GATE_FEATURES,
        )

        if self.predictor_default != DEFAULT_PREDICTORS:
            raise ValueError("Feature default predictor config drift")
        if self.experimental_only != EXPERIMENTAL_FEATURES:
            raise ValueError("Experimental feature config drift")
        if self.quality_gate_only != QUALITY_GATE_FEATURES:
            raise ValueError("Quality-gate feature config drift")
        if self.forbidden_metadata != FORBIDDEN_METADATA:
            raise ValueError("Forbidden metadata config drift")
        if not set(self.promoted_experimental) <= set(self.experimental_only):
            raise ValueError("Only declared experimental features can be promoted")


@dataclass(frozen=True)
class OnnxConfig:
    export_version: str = "onnx-v1.0.0"
    opset: int = 17
    static_batch_size: int = 1
    input_name: str = "float_input"
    output_name: str = "stress_probability"
    min_parity_samples: int = 100
    max_abs_diff: float = 1.0e-5
    max_model_size_bytes: int = 2 * 1024 * 1024
    max_batch1_latency_ms: float = 1.0

    def __post_init__(self) -> None:
        assert self.opset == 17
        assert self.static_batch_size == 1
        assert self.min_parity_samples >= 100 and self.max_abs_diff <= 1.0e-5


@dataclass(frozen=True)
class AndroidConfig:
    valid_vector_count: int = 94
    edge_vector_count: int = 6
    invalid_quality_tiers: tuple[str, ...] = ("poor", "invalid")
    deployable_sensors: tuple[str, ...] = TRAINING_CHANNELS
    channel_priority: tuple[str, ...] = TRAINING_CHANNELS
    require_physical_device: bool = True

    def __post_init__(self) -> None:
        assert self.valid_vector_count + self.edge_vector_count == 100
        if self.deployable_sensors != TRAINING_CHANNELS:
            raise ValueError("Android packaging may accept only independent PPG models")
        if self.channel_priority != TRAINING_CHANNELS:
            raise ValueError(
                "Android independent fallback order must be Green, IR, Red"
            )


@dataclass(frozen=True)
class PipelineConfig:
    raw_source: RawSourceConfig
    synchronization: SynchronizationConfig
    hrv_agreement: HrvAgreementConfig
    peak_detection: PeakDetectionConfig
    window: WindowConfig
    training: TrainingConfig
    features: FeatureConfig
    onnx: OnnxConfig
    android: AndroidConfig
    preprocessing_version: str = PREPROCESSING_VERSION
    feature_version: str = FEATURE_VERSION
    model_version: str = MODEL_VERSION
    timezone_local: str = "Asia/Jakarta"
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_preprocessing = {PREPROCESSING_VERSION, "v3.0.0"}
        allowed_features = {FEATURE_VERSION}
        if self.preprocessing_version not in allowed_preprocessing:
            raise ValueError(
                f"Unknown preprocessing_version: {self.preprocessing_version!r}"
            )
        if self.feature_version not in allowed_features:
            raise ValueError(f"Unknown feature_version: {self.feature_version!r}")
        # Accept historical versions and isolated run-suffixed versions.
        allowed_legacy = {
            MODEL_VERSION,
            "stress_v4.0.0-ml-baseline",
            "stress_v3.0.0",
        }
        if self.model_version not in allowed_legacy and not re.fullmatch(
            r"stress_v\d+\.\d+\.\d+(?:-[A-Za-z0-9_.-]+)+", self.model_version
        ):
            raise ValueError(f"Unknown model_version: {self.model_version!r}")


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    raw_data_root: Path
    anchor_csv_path: Path
    artifacts_root: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    gold_features_path: Path
    eda_dir: Path
    validation_dir: Path
    models_dir: Path
    export_dir: Path


@dataclass(frozen=True)
class PipelineSettings:
    preprocessing_version: str = PREPROCESSING_VERSION
    feature_version: str = FEATURE_VERSION
    model_version: str = MODEL_VERSION
    timezone_local: str = "Asia/Jakarta"
    # Legacy data-foundation adapter setting. Canonical local_pipeline
    # slicing uses WindowConfig's explicit zero-buffer fields instead.
    anchor_buffer_seconds: int = 0
    # Canonical analysis uses the 240-second core. This field is retained for
    # legacy adapters that still construct PipelineSettings directly.
    analysis_window_seconds: int = 240
    source_phase_duration_seconds: int = 300
    edge_trim_start_seconds: int = 30
    edge_trim_end_seconds: int = 30
    validation_window_seconds: int = 240
    gold_window_seconds: int = 180
    gold_stride_seconds: int = 10
    model_window_seconds: int = 180
    model_window_step_seconds: int = 10
    beat_match_tolerance_ms: int = 150
    ibi_match_tolerance_ms: int = 500
    random_seed: int = 42
    test_size_participants: float = 0.2
    bootstrap_iterations: int = 200


# Resolve project root once to avoid repeating the same expression nine times.
_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_PATHS = PipelinePaths(
    project_root=_ROOT,
    # Canonical raw-data location — see README.md "Data Layout" section.
    # Run `python PipelinePython/01_data_preprocess/tools/batch_rename.py` first if your raw folder still
    # uses the original participant names.
    raw_data_root=_ROOT / "data" / "raw" / "DataRenamedPartisipan",
    anchor_csv_path=_ROOT / "TimeStampAnchor" / "participants_2026-05-03_22-47_WIB.csv",
    artifacts_root=_ROOT / "data" / "artifacts",
    bronze_dir=_ROOT / "data" / "bronze",
    silver_dir=_ROOT / "data" / "silver",
    gold_dir=_ROOT / "data" / "gold",
    gold_features_path=_ROOT / "data" / "artifacts" / "gold" / "gold_features.csv",
    eda_dir=_ROOT / "data" / "artifacts" / "eda",
    validation_dir=_ROOT / "data" / "artifacts" / "validation",
    models_dir=_ROOT / "data" / "artifacts" / "models",
    export_dir=_ROOT / "data" / "artifacts" / "export",
)


def ensure_dirs(paths: PipelinePaths) -> None:
    for p in [
        paths.artifacts_root,
        paths.bronze_dir,
        paths.silver_dir,
        paths.gold_dir,
        paths.gold_features_path.parent,
        paths.eda_dir,
        paths.validation_dir,
        paths.models_dir,
        paths.export_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def load_settings(path: str | None) -> PipelineSettings:
    if not path:
        return PipelineSettings()
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return PipelineSettings(**cfg)


def settings_to_dict(settings: PipelineSettings) -> dict:
    return asdict(settings)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required local config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def load_pipeline_config(config_path: str | Path | None = None) -> PipelineConfig:
    """Load the locked contract with an optional data-artifact overlay.

    A run-specific YAML can only override the data/artifact locations. The
    synchronization, training, ONNX, and Android contracts remain the checked-
    in canonical files unless a colocated file is explicitly supplied. This
    permits immutable, isolated reruns without copying or weakening gates.
    """

    canonical_dir = _ROOT / "PipelinePython" / "configs"
    selected = (
        Path(config_path).resolve() if config_path else canonical_dir / "data.yaml"
    )

    def component(name: str) -> Mapping[str, Any]:
        candidate = selected.parent / name
        return _read_yaml(candidate if candidate.exists() else canonical_dir / name)

    data = _read_yaml(selected)
    sync = component("synchronization.yaml")
    hrv = component("hrv_agreement.yaml")
    peak = component("peak_detection.yaml")
    training = component("training.yaml")
    features = component("features.yaml")
    onnx = component("onnx.yaml")
    android = component("android.yaml")

    raw = data["raw_source"]
    window = data["window"]
    return PipelineConfig(
        raw_source=RawSourceConfig(
            renamed_dir=(_ROOT / str(raw["renamed_dir"])).resolve(),
            trimmed_dir=(_ROOT / str(raw["trimmed_dir"])).resolve(),
            anchor_csv=(_ROOT / str(raw["anchor_csv"])).resolve(),
            ecg_source=str(raw.get("ecg_source", "renamed_complete")),
            ppg_source=str(raw.get("ppg_source", "trimmed_phase")),
            trimmed_ecg_role=str(raw.get("trimmed_ecg_role", "audit_only")),
            participant_count=int(raw.get("participant_count", 32)),
        ),
        synchronization=SynchronizationConfig(**sync),
        hrv_agreement=HrvAgreementConfig(**hrv),
        peak_detection=PeakDetectionConfig(
            primary_method=str(peak.get("primary_method", "neurokit2_ppg_elgendi")),
            fallback_method=str(peak.get("fallback_method", "scipy_find_peaks")),
            allow_runtime_fallback=bool(peak.get("allow_runtime_fallback", False)),
            candidate_methods=tuple(
                str(value)
                for value in peak.get(
                    "candidate_methods",
                    (
                        "neurokit2_ppg_elgendi",
                        "neurokit2_ppg_bishop",
                        "neurokit2_ppg_charlton_msptdfast",
                        "scipy_find_peaks_adaptive",
                        "scipy_legacy_green_native",
                    ),
                )
            ),
            minimum_peak_count=int(peak.get("minimum_peak_count", 15)),
            physiological_ibi_min_ms=float(peak.get("physiological_ibi_min_ms", 300.0)),
            physiological_ibi_max_ms=float(
                peak.get("physiological_ibi_max_ms", 1500.0)
            ),
            minimum_physiological_ibi_ratio=float(
                peak.get("minimum_physiological_ibi_ratio", 0.90)
            ),
            audit_prominence_mad_multipliers=tuple(
                float(value)
                for value in peak.get(
                    "audit_prominence_mad_multipliers", (0.4, 0.6, 0.8, 1.0)
                )
            ),
            legacy_native=LegacyNativePeakConfig(**dict(peak.get("legacy_native", {}))),
            channels={
                str(channel): ChannelPeakConfig(**values)
                for channel, values in dict(peak.get("channels", {})).items()
            },
        ),
        window=WindowConfig(**window),
        training=TrainingConfig(
            random_seed=int(training.get("random_seed", 42)),
            ml_execution=str(training.get("ml_execution", "local")),
            training_channels=tuple(
                training.get("training_channels", TRAINING_CHANNELS)
            ),
            candidate_order=tuple(
                training.get("candidate_order", TrainingConfig().candidate_order)
            ),
            gate_metric=str(training.get("gate_metric", "balanced_accuracy")),
            development_participants=int(training.get("development_participants", 28)),
            parity_pool_participants=int(training.get("parity_pool_participants", 4)),
            development_participant_ids=tuple(
                training.get("development_participant_ids", ())
            ),
            parity_pool_participant_ids=tuple(
                training.get("parity_pool_participant_ids", ())
            ),
            inner_group_splits=int(training.get("inner_group_splits", 5)),
            outer_group_splits=int(training.get("outer_group_splits", 7)),
            outer_fold_size=int(training.get("outer_fold_size", 4)),
            minimum_training_development_group_coverage=float(
                training.get(
                    "minimum_training_development_group_coverage",
                    0.90,
                )
            ),
            minimum_development_groups_per_phase=int(
                training.get("minimum_development_groups_per_phase", 23)
            ),
            minimum_development_participants=int(
                training.get("minimum_development_participants", 26)
            ),
            hpo_top_candidates=int(training.get("hpo_top_candidates", 2)),
            optuna_trials=int(training.get("optuna_trials", 50)),
            optuna_sampler=str(training.get("optuna_sampler", "tpe")),
            optuna_storage_uri=str(
                training.get(
                    "optuna_storage_uri",
                    "sqlite:///data/artifacts/optuna/training.db",
                )
            ),
            probability_calibration=str(
                training.get("probability_calibration", "sigmoid")
            ),
            correlation_pruning_threshold=float(
                training.get("correlation_pruning_threshold", 0.95)
            ),
            window_baseline_seconds=int(training.get("window_baseline_seconds", 180)),
            window_baseline_stride_seconds=int(
                training.get("window_baseline_stride_seconds", 10)
            ),
            window_minimum_ba_gain=float(training.get("window_minimum_ba_gain", 0.02)),
            window_candidates=tuple(
                (int(item["window_seconds"]), int(item["stride_seconds"]))
                for item in training.get("window_candidates", ())
            ),
            strict_quality_tiers=tuple(
                training.get("strict_quality_tiers", ("excellent", "good"))
            ),
            expanded_quality_tiers=tuple(
                training.get(
                    "expanded_quality_tiers",
                    ("excellent", "good", "acceptable"),
                )
            ),
            quality_cohort_tie_tolerance=float(
                training.get("quality_cohort_tie_tolerance", 0.01)
            ),
            reused_test_cohort_status=str(
                training.get("reused_test_cohort_status", "reused_internal_test_cohort")
            ),
            clinical_validation_status=str(
                training.get(
                    "clinical_validation_status", "external_validation_required"
                )
            ),
            performance_policy_version=str(
                training.get("performance_policy_version", "ml_dl_tiered_v1")
            ),
            minimum_research_balanced_accuracy=float(
                training.get("minimum_research_balanced_accuracy", 0.70)
            ),
            minimum_research_f1=float(training.get("minimum_research_f1", 0.70)),
            minimum_research_roc_auc=float(
                training.get("minimum_research_roc_auc", 0.70)
            ),
            preferred_balanced_accuracy=float(
                training.get("preferred_balanced_accuracy", 0.80)
            ),
            preferred_f1=float(training.get("preferred_f1", 0.78)),
            preferred_roc_auc=float(training.get("preferred_roc_auc", 0.83)),
            research_disclaimer=str(
                training.get(
                    "research_disclaimer",
                    "Model penelitian — validasi internal terbatas. Bukan diagnosis klinis.",
                )
            ),
            decision_threshold=float(training.get("decision_threshold", 0.5)),
            mlflow_tracking_uri=str(
                training.get(
                    "mlflow_tracking_uri",
                    "sqlite:///data/artifacts/mlflow/mlflow.db",
                )
            ),
            allowed_quality_tiers=tuple(
                training.get(
                    "allowed_quality_tiers", ("excellent", "good", "acceptable")
                )
            ),
        ),
        features=FeatureConfig(
            predictor_default=tuple(features.get("predictor_default", ())),
            experimental_only=tuple(features.get("experimental_only", ())),
            quality_gate_only=tuple(features.get("quality_gate_only", ())),
            forbidden_metadata=tuple(features.get("forbidden_metadata", ())),
            promoted_experimental=tuple(features.get("promoted_experimental", ())),
        ),
        onnx=OnnxConfig(
            opset=int(onnx.get("opset", 17)),
            input_name=str(onnx.get("input_name", "float_input")),
            output_name=str(onnx.get("output_name", "stress_probability")),
            min_parity_samples=int(onnx.get("min_parity_samples", 100)),
            max_abs_diff=float(onnx.get("max_abs_diff", 1.0e-5)),
            max_model_size_bytes=int(onnx.get("max_model_size_bytes", 2 * 1024 * 1024)),
            max_batch1_latency_ms=float(onnx.get("max_batch1_latency_ms", 1.0)),
        ),
        android=AndroidConfig(
            valid_vector_count=int(android.get("valid_vector_count", 94)),
            edge_vector_count=int(android.get("edge_vector_count", 6)),
            invalid_quality_tiers=tuple(
                android.get("invalid_quality_tiers", ("poor", "invalid"))
            ),
            deployable_sensors=tuple(
                android.get("deployable_sensors", TRAINING_CHANNELS)
            ),
            channel_priority=tuple(android.get("channel_priority", TRAINING_CHANNELS)),
            require_physical_device=bool(android.get("require_physical_device", True)),
        ),
        preprocessing_version=str(data["versions"]["preprocessing"]),
        feature_version=str(data["versions"]["features"]),
        model_version=str(data["versions"]["model"]),
        timezone_local=str(data.get("timezone_local", "Asia/Jakarta")),
        artifacts={
            str(key): str(value) for key, value in data.get("artifacts", {}).items()
        },
    )


def pipeline_paths_from_config(
    project_root: Path, config: PipelineConfig
) -> PipelinePaths:
    """Resolve every artifact boundary from one immutable pipeline config."""

    root = project_root.resolve()

    def configured_path(key: str, default: str) -> Path:
        candidate = Path(config.artifacts.get(key, default))
        return candidate if candidate.is_absolute() else root / candidate

    artifacts_root = configured_path("artifacts_root", "data/artifacts")
    return PipelinePaths(
        project_root=root,
        raw_data_root=config.raw_source.renamed_dir,
        anchor_csv_path=config.raw_source.anchor_csv,
        artifacts_root=artifacts_root,
        bronze_dir=configured_path("bronze_dir", "data/bronze"),
        silver_dir=configured_path("silver_dir", "data/silver"),
        gold_dir=configured_path("gold_dir", "data/gold"),
        gold_features_path=configured_path(
            "gold_features", "data/gold/gold_features_v3.parquet"
        ),
        eda_dir=configured_path("eda_dir", "data/artifacts/eda"),
        validation_dir=configured_path("validation_dir", "data/artifacts/validation"),
        models_dir=configured_path("models_dir", "data/artifacts/models"),
        export_dir=configured_path("export_dir", "data/artifacts/export"),
    )
