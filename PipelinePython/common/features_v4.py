"""Robust independent-channel PPG features for the v4 training contract."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from scipy.stats import iqr, kurtosis, skew

from .feature_policy import (
    DEFAULT_PREDICTORS,
    EXPERIMENTAL_FEATURES,
    available_predictors,
)
from .peak_detection import PpgPeakDetection, PpgPeakDetector


def _finite(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("non_finite_feature")
    return float(value)


def _sample_entropy(values: NDArray[np.float64]) -> float:
    if values.size < 12:
        raise ValueError("insufficient_sample_entropy_support")
    deviation = float(np.std(values, ddof=1))
    if deviation <= np.finfo(np.float64).eps:
        raise ValueError("sample_entropy_undefined")
    standardized = (values - np.mean(values)) / deviation

    def count_matches(length: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(standardized, length)
        matches = 0
        for index in range(len(templates) - 1):
            distances = np.max(
                np.abs(templates[index + 1 :] - templates[index]), axis=1
            )
            matches += int(np.sum(distances <= 0.2))
        return matches

    matches_two = count_matches(2)
    matches_three = count_matches(3)
    if matches_two == 0 or matches_three == 0:
        raise ValueError("sample_entropy_undefined")
    return _finite(float(-np.log(matches_three / matches_two)))


def _frequency_features(ibi_ms: NDArray[np.float64]) -> dict[str, float]:
    if ibi_ms.size < 8:
        raise ValueError("insufficient_frequency_feature_support")
    beat_time = np.cumsum(ibi_ms) / 1000.0
    beat_time -= beat_time[0]
    if beat_time[-1] <= 2.0:
        raise ValueError("insufficient_frequency_feature_support")
    grid = np.arange(0.0, beat_time[-1], 0.25)
    tachogram = np.interp(grid, beat_time, ibi_ms)
    frequency, power = welch(
        tachogram - np.mean(tachogram),
        fs=4.0,
        nperseg=min(256, tachogram.size),
    )

    def band_power(low: float, high: float) -> float:
        mask = (frequency >= low) & (frequency < high)
        return float(np.trapezoid(power[mask], frequency[mask]))

    low_frequency = band_power(0.04, 0.15)
    high_frequency = band_power(0.15, 0.40)
    if low_frequency <= 0.0 or high_frequency <= 0.0:
        raise ValueError("non_positive_frequency_power")
    return {
        "log_lf_power": _finite(float(np.log(low_frequency))),
        "log_hf_power": _finite(float(np.log(high_frequency))),
        "lf_hf_ratio": _finite(low_frequency / high_frequency),
    }


def _contiguous_true_runs(mask: NDArray[np.bool_]) -> list[slice]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [slice(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def _longest_valid_run(
    values: NDArray[np.float64], mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    runs = _contiguous_true_runs(mask)
    if not runs:
        return np.asarray([], dtype=np.float64)
    selected = max(runs, key=lambda item: item.stop - item.start)
    return values[selected]


def _morphology_features(
    values: NDArray[np.float64],
    peaks: NDArray[np.int64],
    troughs: NDArray[np.int64],
    sampling_rate_hz: float,
) -> dict[str, float]:
    robust_scale = float(iqr(values))
    if robust_scale <= np.finfo(np.float64).eps:
        raise ValueError("insufficient_waveform_robust_scale")
    amplitudes: list[float] = []
    widths_ms: list[float] = []
    rise_ms: list[float] = []
    decay_ms: list[float] = []
    normalized_areas: list[float] = []
    normalized_rise_slopes: list[float] = []
    normalized_decay_slopes: list[float] = []

    for peak_value in peaks:
        peak = int(peak_value)
        before = troughs[troughs < peak]
        after = troughs[troughs > peak]
        if before.size == 0 or after.size == 0:
            continue
        left, right = int(before[-1]), int(after[0])
        amplitude = float(values[peak] - values[left])
        if right <= peak or peak <= left or amplitude <= np.finfo(float).eps:
            continue
        duration_seconds = (right - left) / sampling_rate_hz
        amplitudes.append(amplitude)
        rise_ms.append((peak - left) * 1000.0 / sampling_rate_hz)
        decay_ms.append((right - peak) * 1000.0 / sampling_rate_hz)

        half_height = values[left] + 0.5 * amplitude
        left_crossings = np.flatnonzero(values[left : peak + 1] <= half_height)
        right_crossings = np.flatnonzero(values[peak : right + 1] <= half_height)
        if left_crossings.size and right_crossings.size:
            crossing_left = left + int(left_crossings[-1])
            crossing_right = peak + int(right_crossings[0])
            if crossing_right > crossing_left:
                widths_ms.append(
                    (crossing_right - crossing_left) * 1000.0 / sampling_rate_hz
                )

        pulse = values[left : right + 1]
        baseline = np.linspace(values[left], values[right], pulse.size)
        positive_area = float(np.trapezoid(np.maximum(pulse - baseline, 0.0)))
        normalized_areas.append(
            positive_area / (sampling_rate_hz * amplitude * duration_seconds)
        )
        rise_gradient = np.diff(values[left : peak + 1]) * sampling_rate_hz
        decay_gradient = np.diff(values[peak : right + 1]) * sampling_rate_hz
        if rise_gradient.size:
            normalized_rise_slopes.append(float(np.max(rise_gradient) / amplitude))
        if decay_gradient.size:
            normalized_decay_slopes.append(
                float(abs(np.min(decay_gradient)) / amplitude)
            )

    collections = (
        amplitudes,
        widths_ms,
        rise_ms,
        decay_ms,
        normalized_areas,
        normalized_rise_slopes,
        normalized_decay_slopes,
    )
    if any(len(collection) < 3 for collection in collections):
        raise ValueError("insufficient_pulse_morphology_support")
    amplitude_values = np.asarray(amplitudes, dtype=np.float64)
    amplitude_median = float(np.median(amplitude_values))
    if amplitude_median <= np.finfo(float).eps:
        raise ValueError("insufficient_pulse_amplitude")
    return {
        "pulse_amplitude_cv": _finite(
            float(np.std(amplitude_values, ddof=1) / amplitude_median)
        ),
        "normalized_pulse_amplitude_median": _finite(amplitude_median / robust_scale),
        "pulse_width_half_height_median_ms": _finite(float(np.median(widths_ms))),
        "rise_time_median_ms": _finite(float(np.median(rise_ms))),
        "decay_time_median_ms": _finite(float(np.median(decay_ms))),
        "normalized_pulse_area_median": _finite(float(np.median(normalized_areas))),
        "normalized_rise_slope_median": _finite(
            float(np.median(normalized_rise_slopes))
        ),
        "normalized_decay_slope_median": _finite(
            float(np.median(normalized_decay_slopes))
        ),
    }


def extract_ppg_features(
    signal: NDArray[np.floating],
    sampling_rate_hz: float,
    *,
    channel: str = "ppg_green",
    detector: PpgPeakDetector | None = None,
    detection: PpgPeakDetection | None = None,
    window_seconds: int = 180,
    experimental_features: tuple[str, ...] = (),
) -> tuple[dict[str, float], NDArray[np.float32], NDArray[np.uint8]]:
    """Return robust predictors, diagnostic features, and a masked IBI series."""

    values = np.asarray(signal, dtype=np.float64)
    unknown_experimental = sorted(
        set(experimental_features) - set(EXPERIMENTAL_FEATURES)
    )
    if unknown_experimental:
        raise ValueError(
            f"unknown_experimental_feature:{','.join(unknown_experimental)}"
        )
    minimum_samples = int(np.floor((window_seconds - 0.5) * sampling_rate_hz))
    if window_seconds <= 0 or values.ndim != 1 or values.size < minimum_samples:
        raise ValueError("insufficient_ppg_window")
    if not np.isfinite(values).all():
        raise ValueError("non_finite_ppg_window")
    if detection is None:
        if detector is None:
            from .settings import load_pipeline_config

            detector = PpgPeakDetector(load_pipeline_config().peak_detection)
        detection = detector.detect(values, sampling_rate_hz, channel)
    peaks, troughs = detection.peaks, detection.troughs
    if peaks.size < 15:
        raise ValueError(f"insufficient_peaks:{peaks.size}")

    all_ppi_ms = np.diff(peaks).astype(np.float64) * 1000.0 / sampling_rate_hz
    physiological_mask = (all_ppi_ms >= 300.0) & (all_ppi_ms <= 1500.0)
    ppi_ms = all_ppi_ms[physiological_mask]
    if ppi_ms.size < 12:
        raise ValueError("insufficient_physiological_ppi")
    adjacent_mask = physiological_mask[:-1] & physiological_mask[1:]
    ppi_difference = np.diff(all_ppi_ms)[adjacent_mask]
    if ppi_difference.size < 3:
        raise ValueError("insufficient_adjacent_physiological_ppi")
    ppi_median = float(np.median(ppi_ms))
    sdnn = float(np.std(ppi_ms, ddof=1))
    difference_std = float(np.std(ppi_difference, ddof=1))
    gradient = np.diff(values)
    robust_sigma = float(iqr(values) / 1.349)
    if robust_sigma <= np.finfo(float).eps:
        raise ValueError("insufficient_waveform_robust_scale")
    flatline_ratio = float(np.mean(np.abs(gradient) < 1e-6))
    artifact_ratio = float(
        np.mean(np.abs(values - np.median(values)) > 6.0 * robust_sigma)
    )

    features: dict[str, float] = {
        "median_hr_bpm": _finite(float(np.median(60_000.0 / ppi_ms))),
        "ppi_median_ms": _finite(ppi_median),
        "ppi_iqr_ms": _finite(float(iqr(ppi_ms))),
        "ppi_cv": _finite(sdnn / ppi_median),
        "rmssd_ms": _finite(float(np.sqrt(np.mean(np.square(ppi_difference))))),
        "sdnn_ms": _finite(sdnn),
        "pnn50": _finite(float(np.mean(np.abs(ppi_difference) > 50.0))),
        "peak_count": float(peaks.size),
        "physiological_ibi_ratio": _finite(float(np.mean(physiological_mask))),
        "valid_peak_ratio": _finite(ppi_ms.size / max(peaks.size - 1, 1)),
        "signal_quality_index": _finite(
            float(np.clip(1.0 - flatline_ratio - artifact_ratio, 0.0, 1.0))
        ),
        "flatline_ratio": flatline_ratio,
        "artifact_ratio": artifact_ratio,
    }
    features.update(_morphology_features(values, peaks, troughs, sampling_rate_hz))
    if "sd1_ms" in experimental_features:
        features["sd1_ms"] = _finite(float(np.sqrt(0.5) * difference_std))
    if "sd2_ms" in experimental_features:
        features["sd2_ms"] = _finite(
            float(np.sqrt(max(2.0 * sdnn**2 - 0.5 * difference_std**2, 0.0)))
        )
    if "sample_entropy" in experimental_features:
        features["sample_entropy"] = _sample_entropy(
            _longest_valid_run(all_ppi_ms, physiological_mask)
        )
    if "waveform_skewness" in experimental_features:
        features["waveform_skewness"] = _finite(float(skew(values, bias=False)))
    if "waveform_kurtosis" in experimental_features:
        features["waveform_kurtosis"] = _finite(
            float(kurtosis(values, fisher=True, bias=False))
        )
    if set(experimental_features) & {"log_lf_power", "log_hf_power", "lf_hf_ratio"}:
        frequency = _frequency_features(
            _longest_valid_run(all_ppi_ms, physiological_mask)
        )
        features.update(
            {
                name: frequency[name]
                for name in experimental_features
                if name in frequency
            }
        )
    missing_default = sorted(set(DEFAULT_PREDICTORS) - set(features))
    if missing_default or not all(np.isfinite(value) for value in features.values()):
        raise ValueError(f"invalid_feature_contract:{missing_default}")

    sequence_length = int(window_seconds * 4)
    sequence = np.zeros(sequence_length, dtype=np.float32)
    mask = np.zeros(sequence_length, dtype=np.uint8)
    interval_end_time = np.cumsum(all_ppi_ms) / 1000.0
    grid = np.arange(sequence_length, dtype=np.float64) / 4.0
    for run in _contiguous_true_runs(physiological_mask):
        run_time = interval_end_time[run]
        run_ppi = all_ppi_ms[run]
        if run_ppi.size < 2 or run_time[-1] <= run_time[0]:
            continue
        within = (grid >= run_time[0]) & (grid <= run_time[-1])
        sequence[within] = np.interp(grid[within], run_time, run_ppi).astype(np.float32)
        mask[within] = 1
    return features, sequence, mask


def feature_order(
    rows: list[Mapping[str, float]], promoted_experimental: tuple[str, ...] = ()
) -> tuple[str, ...]:
    if not rows:
        raise ValueError("feature_rows_required")
    canonical = tuple(sorted(rows[0]))
    if any(tuple(sorted(row)) != canonical for row in rows):
        raise ValueError("inconsistent_feature_rows")
    return available_predictors(canonical, promoted_experimental)
