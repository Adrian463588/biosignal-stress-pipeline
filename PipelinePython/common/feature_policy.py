"""Canonical PPG predictor policy shared by training and export gates."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DEFAULT_PREDICTORS = (
    "median_hr_bpm",
    "ppi_median_ms",
    "ppi_iqr_ms",
    "ppi_cv",
    "rmssd_ms",
    "sdnn_ms",
    "pnn50",
    "pulse_amplitude_cv",
    "normalized_pulse_amplitude_median",
    "pulse_width_half_height_median_ms",
    "rise_time_median_ms",
    "decay_time_median_ms",
    "normalized_pulse_area_median",
    "normalized_rise_slope_median",
    "normalized_decay_slope_median",
)

EXPERIMENTAL_FEATURES = (
    "sd1_ms",
    "sd2_ms",
    "sample_entropy",
    "waveform_skewness",
    "waveform_kurtosis",
    "log_lf_power",
    "log_hf_power",
    "lf_hf_ratio",
)

QUALITY_GATE_FEATURES = (
    "peak_count",
    "physiological_ibi_ratio",
    "valid_peak_ratio",
    "signal_quality_index",
    "missing_ratio",
    "interpolated_fraction",
    "artifact_ratio",
    "clipping_ratio",
    "flatline_ratio",
    "motion_score",
    "sampling_jitter",
    "quality_tier",
    "quality_flags",
)

FORBIDDEN_METADATA = (
    "participant_id",
    "participant_code",
    "session_id",
    "recording_id",
    "phase",
    "label",
    "label_name",
    "source_file",
    "source_path",
    "source_checksum",
    "timestamp",
    "window_id",
    "window_index",
    "window_start_utc",
    "window_end_utc",
    "source_rate_hz",
    "model_rate_hz",
    "sampling_rate_hz",
    "sensor_channel",
    "split",
)

_FORBIDDEN_PREFIXES = (
    "ecg_",
    "pat_",
    "motion_",
    "raw_",
    "source_",
    "timestamp_",
    "window_",
)


def validate_feature_order(
    feature_order: Iterable[str],
    promoted_experimental: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return a validated independent-channel feature order.

    Quality and technical fields remain available for eligibility auditing but
    are never accepted as predictors. Experimental features must be explicitly
    present in the caller-provided order; the default policy promotes none.
    """

    ordered = tuple(str(name) for name in feature_order)
    if not ordered:
        raise ValueError("empty_training_feature_order")
    if len(set(ordered)) != len(ordered):
        raise ValueError("duplicate_training_feature")
    promoted = tuple(str(name) for name in promoted_experimental)
    unknown_promotions = sorted(set(promoted) - set(EXPERIMENTAL_FEATURES))
    if unknown_promotions:
        raise ValueError(f"unknown_experimental_feature:{','.join(unknown_promotions)}")
    allowed = set(DEFAULT_PREDICTORS) | set(promoted)
    gate_only = set(QUALITY_GATE_FEATURES)
    metadata = set(FORBIDDEN_METADATA)
    rejected = [
        name
        for name in ordered
        if name not in allowed
        or name in gate_only
        or name in metadata
        or name.lower().startswith(_FORBIDDEN_PREFIXES)
    ]
    if rejected:
        raise ValueError(f"forbidden_training_feature:{','.join(sorted(rejected))}")
    return ordered


def available_predictors(
    columns: Iterable[str], promoted_experimental: Iterable[str] = ()
) -> tuple[str, ...]:
    """Select approved predictors in physiological preference order."""

    present = set(columns)
    promoted = tuple(promoted_experimental)
    unknown = sorted(set(promoted) - set(EXPERIMENTAL_FEATURES))
    if unknown:
        raise ValueError(f"unknown_experimental_feature:{','.join(unknown)}")
    missing_default = sorted(set(DEFAULT_PREDICTORS) - present)
    if missing_default:
        raise ValueError(
            f"missing_required_training_feature:{','.join(missing_default)}"
        )
    selected = (*DEFAULT_PREDICTORS, *promoted)
    return validate_feature_order(selected, promoted)


def correlation_pruned_features(
    training_frame: pd.DataFrame,
    feature_order: Iterable[str],
    threshold: float = 0.95,
    promoted_experimental: Iterable[str] = (),
) -> tuple[str, ...]:
    """Prune correlated fields using only one training fold.

    Earlier fields win, so callers pass the physiological preference order.
    """

    ordered = validate_feature_order(feature_order, promoted_experimental)
    if not 0.0 < threshold < 1.0:
        raise ValueError("invalid_correlation_threshold")
    correlation = training_frame.loc[:, ordered].corr().abs()
    retained: list[str] = []
    for feature in ordered:
        if all(
            not pd.notna(correlation.loc[feature, existing])
            or float(correlation.loc[feature, existing]) <= threshold
            for existing in retained
        ):
            retained.append(feature)
    if not retained:
        raise ValueError("correlation_pruning_removed_all_features")
    return tuple(retained)
