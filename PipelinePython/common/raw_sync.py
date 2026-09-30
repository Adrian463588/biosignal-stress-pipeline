"""Whole-recording ECG clock correction and physiological PPG validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, find_peaks

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover - exercised by dependency audit
    nk = None

from .contracts import ClockCorrection, SynchronizationResult
from .settings import SynchronizationConfig

DEFAULT_MAX_UNFILLED_GAP_MS = 500.0
ECG_REFERENCE_CONSENSUS_TOLERANCE_MS = 150.0


@dataclass(frozen=True)
class _MetadataClockCandidate:
    timeline_unix_ms: NDArray[np.float64] | None
    observed_sample_count: int
    declared_sample_count: int | None
    declared_duration_seconds: float | None
    metadata_stop_unix_ms: float | None
    nominal_sample_rate_hz: float | None
    effective_sample_rate_hz: float | None
    effective_rate_relative_error: float | None
    alpha: float | None
    exclusion_reason: str | None


def _finite_sorted(
    timestamps_ms: NDArray[np.floating],
    values: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    assert timestamps.ndim == signal.ndim == 1 and timestamps.size == signal.size
    keep = np.isfinite(timestamps) & np.isfinite(signal)
    timestamps, signal = timestamps[keep], signal[keep]
    assert timestamps.size >= 3, "signal requires at least three finite samples"
    order = np.argsort(timestamps, kind="stable")
    timestamps, signal = timestamps[order], signal[order]
    unique = np.concatenate(([True], np.diff(timestamps) > 0))
    return timestamps[unique], signal[unique]


def _finite_sample_order(
    timestamps_ms: NDArray[np.floating],
    values: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return finite samples in acquisition order without timestamp sorting."""

    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    assert timestamps.ndim == signal.ndim == 1 and timestamps.size == signal.size
    keep = np.isfinite(timestamps) & np.isfinite(signal)
    timestamps, signal = timestamps[keep], signal[keep]
    assert timestamps.size >= 3, "signal requires at least three finite samples"
    return timestamps, signal


def _excluded_input_result(
    *,
    participant_id: str,
    recording_id: str,
    ecg_timestamps_ms: NDArray[np.floating],
    ppg_timestamps_ms: NDArray[np.floating],
    reason: str,
    diagnostic_reference_channel: str | None,
) -> tuple[ClockCorrection, SynchronizationResult]:
    """Return a traceable exclusion instead of aborting the sync batch."""

    ecg_timestamps = np.asarray(ecg_timestamps_ms, dtype=np.float64).reshape(-1)
    ppg_timestamps = np.asarray(ppg_timestamps_ms, dtype=np.float64).reshape(-1)
    correction = ClockCorrection(
        corrected_unix_ms=ecg_timestamps.copy(),
        alpha=1.0,
        beta_seconds=0.0,
        excluded=True,
        exclusion_reason=reason,
        timebase_source="raw_timestamp_invalid_input",
        observed_sample_count=int(ecg_timestamps.size),
        clock_selection_basis="metadata_only",
        raw_unix_ms=ecg_timestamps.copy(),
        reference_status="UNAVAILABLE",
        acceptance_reason=reason,
    )
    result = SynchronizationResult(
        participant_id=participant_id,
        recording_id=recording_id,
        alpha=1.0,
        beta_seconds=0.0,
        method="raw_timestamp_invalid_input",
        confidence=0.0,
        overlap_seconds=0.0,
        ecg_peak_count=0,
        ppg_peak_count=0,
        matched_peak_count=0,
        median_pat_ms=None,
        excluded=True,
        exclusion_reason=reason,
        clock_selection_basis="metadata_only",
        observed_sample_count=int(ecg_timestamps.size),
        reference_signal_role="diagnostic_only",
        ppg_diagnostic_reason=(f"invalid_ppg_input:sample_count={ppg_timestamps.size}"),
        clock_excluded=True,
        clock_exclusion_reason=reason,
        diagnostic_reference_channel=diagnostic_reference_channel,
        timebase_source="raw_timestamp_invalid_input",
    )
    return correction, result


def _contains_long_dropout(
    timestamps_ms: NDArray[np.floating],
    values: NDArray[np.floating],
    *,
    max_unfilled_gap_ms: float = DEFAULT_MAX_UNFILLED_GAP_MS,
) -> bool:
    """Return true when a gap exceeds the configured short-gap policy.

    This helper is used by the legacy preliminary clock-estimation adapter.
    It never writes an interpolated waveform.  The canonical primary
    validator uses ``PpgEcgValidator`` and masks gaps instead.
    """

    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    valid_timestamps = np.isfinite(timestamps)
    timestamps, signal = timestamps[valid_timestamps], signal[valid_timestamps]
    if timestamps.size < 3:
        return True
    order = np.argsort(timestamps, kind="stable")
    timestamps, signal = timestamps[order], signal[order]
    intervals = np.diff(timestamps)
    positive = intervals[intervals > 0.0]
    if positive.size == 0:
        return True
    expected_interval_ms = float(np.median(positive))
    if np.any(intervals - expected_interval_ms > max_unfilled_gap_ms):
        return True
    missing = ~np.isfinite(signal)
    if not missing.any():
        return False
    starts = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(np.int8)) == -1)
    for start, end in zip(starts, ends, strict=True):
        if start == 0 or end >= len(signal):
            return True
        span_ms = timestamps[end] - timestamps[start - 1] - expected_interval_ms
        if span_ms > max_unfilled_gap_ms:
            return True
    return False


def _metadata_duration_seconds(metadata: Mapping[str, str]) -> float | None:
    raw = (
        str(metadata.get("duration", ""))
        .lower()
        .replace("seconds", "")
        .replace("sec", "")
    )
    try:
        duration = float(raw.strip())
    except ValueError:
        return None
    return duration if duration > 0.0 else None


def _metadata_positive_float(metadata: Mapping[str, str], key: str) -> float | None:
    try:
        value = float(str(metadata.get(key, "")).strip())
    except ValueError:
        return None
    return value if np.isfinite(value) and value > 0.0 else None


def _metadata_sample_count(metadata: Mapping[str, str]) -> int | None:
    value = _metadata_positive_float(metadata, "samplescount")
    if value is None or not value.is_integer() or value < 3:
        return None
    return int(value)


def _metadata_stop_ms(metadata: Mapping[str, str]) -> float | None:
    raw = metadata.get("stoptime")
    if not raw:
        return None
    # Java Date.toString emits ``GMT+07:00``. Pandas/dateutil treats the GMT
    # prefix with POSIX sign semantics, so remove only that literal prefix and
    # retain the explicit ISO-style offset.
    normalized = str(raw).replace("GMT+", "+").replace("GMT-", "-")
    parsed = pd.to_datetime(normalized, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed.timestamp() * 1000.0)


def _metadata_clock_candidate(
    timestamps_ms: NDArray[np.float64],
    metadata: Mapping[str, str],
    config: SynchronizationConfig,
) -> _MetadataClockCandidate:
    """Build one metadata-only ECG clock candidate from sample order."""

    observed_count = len(timestamps_ms)
    declared_count = _metadata_sample_count(metadata)
    duration_seconds = _metadata_duration_seconds(metadata)
    stop_ms = _metadata_stop_ms(metadata)
    nominal_rate_hz = _metadata_positive_float(metadata, "samplingrate")
    effective_rate_hz = (
        (observed_count - 1) / duration_seconds
        if duration_seconds is not None
        else None
    )
    relative_error = (
        abs(effective_rate_hz - nominal_rate_hz) / nominal_rate_hz
        if effective_rate_hz is not None and nominal_rate_hz is not None
        else None
    )

    reason: str | None = None
    if declared_count is None:
        reason = "metadata_samplescount_missing_or_invalid"
    elif declared_count != observed_count:
        reason = (
            "metadata_samplescount_mismatch:"
            f"declared={declared_count},observed={observed_count}"
        )
    elif duration_seconds is None:
        reason = "metadata_duration_missing_or_invalid"
    elif stop_ms is None:
        reason = "metadata_stoptime_missing_or_invalid"
    elif nominal_rate_hz is None:
        reason = "metadata_samplingrate_missing_or_invalid"
    elif relative_error is None or (
        relative_error > config.metadata_effective_rate_tolerance_fraction
    ):
        reason = (
            "metadata_effective_rate_out_of_tolerance:"
            f"effective_hz={effective_rate_hz:.6f},"
            f"nominal_hz={nominal_rate_hz:.6f},"
            f"relative_error={relative_error:.6f},"
            f"limit={config.metadata_effective_rate_tolerance_fraction:.6f}"
        )

    # The metadata-duration timeline is already expressed in the corrected
    # clock.  Its effective rate is validation evidence, not an additional
    # affine scale to apply.  Acquisition padding is removed by anchor crop,
    # so alpha remains the identity instead of compressing ECG to PPG bounds.
    alpha = 1.0 if reason is None else None

    timeline: NDArray[np.float64] | None = None
    if reason is None:
        assert duration_seconds is not None and stop_ms is not None
        timeline = np.linspace(
            stop_ms - duration_seconds * 1000.0,
            stop_ms,
            observed_count,
            dtype=np.float64,
        )

    return _MetadataClockCandidate(
        timeline_unix_ms=timeline,
        observed_sample_count=observed_count,
        declared_sample_count=declared_count,
        declared_duration_seconds=duration_seconds,
        metadata_stop_unix_ms=stop_ms,
        nominal_sample_rate_hz=nominal_rate_hz,
        effective_sample_rate_hz=effective_rate_hz,
        effective_rate_relative_error=relative_error,
        alpha=alpha,
        exclusion_reason=reason,
    )


def affine_clock_correction(
    ecg_timestamps_ms: NDArray[np.floating],
    metadata: Mapping[str, str],
    config: SynchronizationConfig,
    *,
    reference_timestamps_ms: NDArray[np.floating] | None = None,
    phase_start_ms: float | None = None,
    phase_end_ms: float | None = None,
) -> ClockCorrection:
    """Build an ECG timeline from metadata without scaling it to PPG bounds.

    PPG and phase endpoints are compatibility inputs only. The selected ECG
    clock candidate depends exclusively on ECG sample order and metadata.
    """

    timestamps = np.asarray(ecg_timestamps_ms, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 3 or not np.isfinite(timestamps).all():
        return ClockCorrection(
            timestamps.copy(),
            1.0,
            0.0,
            True,
            "invalid_ecg_raw_timestamp_for_clock_candidate",
            "raw_timestamp_unmodified",
            observed_sample_count=int(timestamps.size),
            clock_selection_basis="metadata_only",
            residual_correction_status="not_evaluated",
            raw_unix_ms=timestamps.copy(),
            acceptance_reason="invalid_raw_timestamp",
        )
    # Kept in the signature for compatibility only. PPG/phase endpoints and
    # beat agreement never select or scale the ECG clock candidate.
    _ = (reference_timestamps_ms, phase_start_ms, phase_end_ms)
    candidate = _metadata_clock_candidate(timestamps, metadata, config)
    evidence = {
        "observed_sample_count": candidate.observed_sample_count,
        "declared_sample_count": candidate.declared_sample_count,
        "declared_duration_seconds": candidate.declared_duration_seconds,
        "metadata_stop_unix_ms": candidate.metadata_stop_unix_ms,
        "nominal_sample_rate_hz": candidate.nominal_sample_rate_hz,
        "effective_sample_rate_hz": candidate.effective_sample_rate_hz,
        "effective_rate_relative_error": candidate.effective_rate_relative_error,
        "clock_selection_basis": "metadata_only",
    }
    if candidate.timeline_unix_ms is None or candidate.exclusion_reason is not None:
        return ClockCorrection(
            timestamps.copy(),
            1.0,
            0.0,
            True,
            candidate.exclusion_reason or "metadata_timebase_unavailable",
            "raw_timestamp_unmodified",
            **evidence,
            raw_unix_ms=timestamps.copy(),
            acceptance_reason=candidate.exclusion_reason
            or "metadata_timebase_unavailable",
        )

    corrected = candidate.timeline_unix_ms
    beta_ms = corrected[0] - timestamps[0]
    assert np.all(np.diff(corrected) > 0.0), "affine correction must preserve order"
    return ClockCorrection(
        corrected.astype(np.float64),
        float(candidate.alpha),
        float(beta_ms / 1000.0),
        False,
        None,
        "sample_index_metadata_duration_stoptime",
        candidate_start_unix_ms=float(corrected[0]),
        candidate_end_unix_ms=float(corrected[-1]),
        raw_unix_ms=timestamps.copy(),
        acceptance_reason="metadata_sample_order_duration_stoptime_accepted",
        **evidence,
    )


def _clean_for_peaks(
    timestamps_ms: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    is_ecg: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Filter finite original samples without creating waveform samples.

    Peak timestamps remain tied to the original sample positions. The primary
    ECG–PPG validator may use a median interval only for filter design; it must
    never interpolate the ECG waveform or invent a sample at a missing time.
    """
    dt = np.diff(timestamps_ms) / 1000.0
    positive_dt = dt[dt > 0]
    if positive_dt.size == 0:
        raise ValueError("signal timestamps are not strictly increasing")
    median_dt = float(np.median(positive_dt))
    fs = 1.0 / median_dt
    cleaned = values.astype(np.float64, copy=True)
    cleaned -= np.median(cleaned)
    lowcut, highcut = (
        (0.5, min(35.0, fs * 0.45)) if is_ecg else (0.5, min(4.0, fs * 0.45))
    )
    if highcut <= lowcut:
        return timestamps_ms, cleaned, fs
    b, a = butter(3, [lowcut / (fs / 2.0), highcut / (fs / 2.0)], btype="band")
    if cleaned.size > 3 * max(len(a), len(b)):
        cleaned = filtfilt(b, a, cleaned)
    return timestamps_ms, cleaned, fs


def _package_version(package: str) -> str:
    try:
        return f"{package}=={package_version(package)}"
    except PackageNotFoundError:
        return f"{package}==UNAVAILABLE"


def _sanitize_peak_indices(indices: object, sample_count: int) -> NDArray[np.int64]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    values = values[(values >= 0) & (values < sample_count)]
    return np.unique(values).astype(np.int64, copy=False)


def _finite_ecg_segments(
    timestamps_ms: NDArray[np.floating], values: NDArray[np.floating]
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Split observed ECG samples before detector calls.

    Missing samples are removed only from the detector view.  No waveform
    sample is created and the caller retains the original timestamp arrays for
    gap/dropout accounting.
    """

    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(timestamps) & np.isfinite(signal)
    timestamps, signal = timestamps[finite], signal[finite]
    if timestamps.size < 3:
        return []
    if np.any(np.diff(timestamps) <= 0.0):
        return []
    intervals = np.diff(timestamps)
    positive = intervals[intervals > 0.0]
    if positive.size == 0:
        return []
    expected = float(np.median(positive))
    gap_break = np.flatnonzero(intervals > max(1.5 * expected, 500.0)) + 1
    starts = np.r_[0, gap_break]
    ends = np.r_[gap_break, timestamps.size]
    return [
        (timestamps[start:end], signal[start:end])
        for start, end in zip(starts, ends, strict=True)
        if end - start >= 3
    ]


def _detect_neurokit_ecg_segment(
    timestamps_ms: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    method: str,
) -> NDArray[np.float64]:
    if nk is None:
        return np.asarray([], dtype=np.float64)
    _, clean, fs_hz = _clean_for_peaks(timestamps_ms, values, is_ecg=True)
    _, info = nk.ecg_peaks(
        clean,
        sampling_rate=max(1, int(round(fs_hz))),
        method=method,
    )
    indices = _sanitize_peak_indices(info.get("ECG_R_Peaks", []), clean.size)
    return timestamps_ms[indices].astype(np.float64, copy=False)


def _consensus_peak_times(
    first_ms: NDArray[np.float64],
    second_ms: NDArray[np.float64],
    *,
    tolerance_ms: float,
) -> NDArray[np.float64]:
    """Greedy monotonic one-to-one consensus for two ECG detectors."""

    first = np.asarray(first_ms, dtype=np.float64)
    second = np.asarray(second_ms, dtype=np.float64)
    if first.size == 0 or second.size == 0:
        return np.asarray([], dtype=np.float64)
    matches: list[float] = []
    second_index = 0
    for timestamp in first:
        while (
            second_index < second.size
            and second[second_index] < timestamp - tolerance_ms
        ):
            second_index += 1
        if second_index >= second.size:
            break
        if abs(float(second[second_index] - timestamp)) <= tolerance_ms:
            matches.append(float(timestamp))
            second_index += 1
    return np.asarray(matches, dtype=np.float64)


def _detect_reference_ecg_peaks(
    timestamps_ms: NDArray[np.floating],
    values: NDArray[np.floating],
    *,
    tolerance_ms: float = ECG_REFERENCE_CONSENSUS_TOLERANCE_MS,
) -> tuple[NDArray[np.float64], str, str, str]:
    """Detect ECG reference peaks with independent NeuroKit/Pan--Tompkins.

    The returned status is evidence about detector agreement, not a clinical
    ground-truth claim.  A disagreement never changes the ECG waveform or
    silently selects a clock from PPG metrics.
    """

    segments = _finite_ecg_segments(timestamps_ms, values)
    if not segments:
        return (
            np.asarray([], dtype=np.float64),
            "",
            "UNAVAILABLE",
            "no_finite_ecg_segment",
        )

    peak_streams: list[NDArray[np.float64]] = []
    methods: list[str] = []
    reasons: list[str] = []
    statuses: list[str] = []
    for segment_timestamps, segment_values in segments:
        neurokit_peaks = np.asarray([], dtype=np.float64)
        pantompkins_peaks = np.asarray([], dtype=np.float64)
        try:
            neurokit_peaks = _detect_neurokit_ecg_segment(
                segment_timestamps,
                segment_values,
                method="neurokit",
            )
        except Exception as exc:
            reasons.append(f"ecg_neurokit:{type(exc).__name__}:{exc}")
        try:
            pantompkins_peaks = _detect_neurokit_ecg_segment(
                segment_timestamps,
                segment_values,
                method="pantompkins1985",
            )
        except Exception as exc:
            reasons.append(f"ecg_pantompkins:{type(exc).__name__}:{exc}")

        if neurokit_peaks.size and pantompkins_peaks.size:
            consensus = _consensus_peak_times(
                neurokit_peaks,
                pantompkins_peaks,
                tolerance_ms=tolerance_ms,
            )
            agreement = consensus.size / max(
                1, min(neurokit_peaks.size, pantompkins_peaks.size)
            )
            if consensus.size >= 3 and agreement >= 0.80:
                peak_streams.append(consensus)
                statuses.append("CONSENSUS")
            else:
                peak_streams.append(neurokit_peaks)
                statuses.append("REFERENCE_UNCERTAIN")
                reasons.append(f"ecg_detector_consensus_fraction={agreement:.3f}")
            methods.append("neurokit2+pan_tompkins")
        elif neurokit_peaks.size:
            peak_streams.append(neurokit_peaks)
            statuses.append("REFERENCE_UNCERTAIN")
            methods.append("neurokit2")
            reasons.append("ecg_pantompkins_unavailable")
        elif pantompkins_peaks.size:
            peak_streams.append(pantompkins_peaks)
            statuses.append("REFERENCE_UNCERTAIN")
            methods.append("pan_tompkins")
            reasons.append("ecg_neurokit_unavailable")
        else:
            # This is diagnostic-only fallback.  It keeps clock validity and
            # failure status separate while preserving the observed samples.
            try:
                peak_streams.append(
                    detect_peak_times_ms(
                        segment_timestamps,
                        segment_values,
                        is_ecg=True,
                    )
                )
                methods.append("scipy_diagnostic_fallback")
                statuses.append("REFERENCE_UNCERTAIN")
                reasons.append("ecg_dual_detector_unavailable")
            except (AssertionError, ValueError, IndexError) as exc:
                reasons.append(f"ecg_scipy_fallback:{type(exc).__name__}:{exc}")

    peaks = (
        np.unique(np.concatenate(peak_streams)).astype(np.float64)
        if peak_streams
        else np.asarray([], dtype=np.float64)
    )
    if not peaks.size:
        status = "UNAVAILABLE"
    elif all(status == "CONSENSUS" for status in statuses):
        status = "CONSENSUS"
    else:
        status = "REFERENCE_UNCERTAIN"
    return peaks, "+".join(dict.fromkeys(methods)), status, ";".join(reasons)


def detect_peak_times_ms(
    timestamps_ms: NDArray[np.floating],
    values: NDArray[np.floating],
    *,
    is_ecg: bool,
) -> NDArray[np.float64]:
    timestamps, signal = _finite_sorted(timestamps_ms, values)
    grid, cleaned, fs = _clean_for_peaks(timestamps, signal, is_ecg=is_ecg)
    median = float(np.median(cleaned))
    mad = float(np.median(np.abs(cleaned - median)))
    prominence = max(1.0e-9, 1.5 * mad if is_ecg else 0.8 * mad)
    distance = max(1, int(round(0.30 * fs)))
    peaks, _ = find_peaks(cleaned, distance=distance, prominence=prominence)
    if is_ecg:
        inverted, _ = find_peaks(-cleaned, distance=distance, prominence=prominence)
        if inverted.size > peaks.size:
            peaks = inverted
    return timestamps[peaks].astype(np.float64)


def _match_physiological_pat(
    ecg_peaks_ms: NDArray[np.float64],
    ppg_peaks_ms: NDArray[np.float64],
    config: SynchronizationConfig,
) -> NDArray[np.float64]:
    return np.asarray(
        [
            delay
            for _, _, delay in _match_pat_pairs(
                ecg_peaks_ms,
                ppg_peaks_ms,
                config,
            )
        ],
        dtype=np.float64,
    )


def _match_pat_pairs(
    ecg_peaks_ms: NDArray[np.float64],
    ppg_peaks_ms: NDArray[np.float64],
    config: SynchronizationConfig,
    *,
    pat_intercept_ms: float | None = None,
    pat_slope_ms_per_minute: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Assign ECG and PPG peaks monotonically and at most once each."""

    ecg = np.asarray(ecg_peaks_ms, dtype=np.float64)
    ppg = np.asarray(ppg_peaks_ms, dtype=np.float64)
    if ecg.size == 0 or ppg.size == 0:
        return []
    if (
        not np.isfinite(ecg).all()
        or not np.isfinite(ppg).all()
        or np.any(np.diff(ecg) <= 0.0)
        or np.any(np.diff(ppg) <= 0.0)
    ):
        return []

    tolerance = float(config.pat_tolerance_ms)
    reference_time = float(ecg[0])
    last_ppg = -1
    matches: list[tuple[int, int, float]] = []
    for ecg_index, ecg_time in enumerate(ecg):
        start = max(
            last_ppg + 1,
            int(np.searchsorted(ppg, ecg_time + config.pat_min_ms, side="left")),
        )
        end = int(np.searchsorted(ppg, ecg_time + config.pat_max_ms, side="right"))
        if start >= end:
            continue
        candidates = np.arange(start, end, dtype=int)
        delays = ppg[candidates] - ecg_time
        if pat_intercept_ms is None:
            selected = int(candidates[0])
        else:
            target = (
                pat_intercept_ms
                + pat_slope_ms_per_minute * (ecg_time - reference_time) / 60_000.0
            )
            keep = np.abs(delays - target) <= tolerance
            if not keep.any():
                continue
            selected = int(
                candidates[
                    np.flatnonzero(keep)[np.argmin(np.abs(delays[keep] - target))]
                ]
            )
        delay = float(ppg[selected] - ecg_time)
        if config.pat_min_ms <= delay <= config.pat_max_ms:
            matches.append((ecg_index, selected, delay))
            last_ppg = selected
    return matches


def _pat_statistics(
    matches: list[tuple[int, int, float]],
) -> tuple[float | None, float | None, float | None]:
    delays = np.asarray([delay for _, _, delay in matches], dtype=np.float64)
    if not delays.size:
        return None, None, None
    median = float(np.median(delays))
    iqr = float(np.percentile(delays, 75) - np.percentile(delays, 25))
    mad = float(np.median(np.abs(delays - median)))
    return median, iqr, mad


def _estimate_pat_residual_slope(
    ecg_peaks_ms: NDArray[np.float64],
    matches: list[tuple[int, int, float]],
    config: SynchronizationConfig,
) -> tuple[float, float, str]:
    """Estimate one bounded PAT drift line from observed matched beats."""

    minimum = int(config.pat_residual_min_matches)
    if len(matches) < minimum:
        return 0.0, 0.0, "insufficient_matches"
    ecg = np.asarray(ecg_peaks_ms, dtype=np.float64)
    indices = np.asarray([ecg_index for ecg_index, _, _ in matches], dtype=int)
    delays = np.asarray([delay for _, _, delay in matches], dtype=np.float64)
    x = (ecg[indices] - ecg[0]) / 60_000.0
    center = float(np.median(delays))
    mad = float(np.median(np.abs(delays - center)))
    robust_limit = max(float(config.pat_tolerance_ms), 3.0 * 1.4826 * mad)
    keep = np.abs(delays - center) <= robust_limit
    if int(keep.sum()) < minimum or np.ptp(x[keep]) <= 0.0:
        return 0.0, 0.0, "insufficient_robust_matches"
    slope, intercept = np.polyfit(x[keep], delays[keep], 1)
    full_x = (ecg - ecg[0]) / 60_000.0
    predicted = intercept + slope * full_x
    alpha = 1.0 + slope / 60_000.0
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return 0.0, 0.0, "nonfinite_residual_model"
    if not config.alpha_min <= alpha <= config.alpha_max:
        return 0.0, 0.0, "residual_alpha_out_of_bounds"
    if (
        float(np.min(predicted)) < config.pat_min_ms
        or float(np.max(predicted)) > config.pat_max_ms
    ):
        return 0.0, 0.0, "residual_pat_out_of_bounds"
    return float(intercept), float(slope), "estimated"


def _residual_correction(
    correction: ClockCorrection,
    ecg_peaks_ms: NDArray[np.float64],
    ppg_peaks_ms: NDArray[np.float64],
    baseline_matches: list[tuple[int, int, float]],
    config: SynchronizationConfig,
) -> tuple[ClockCorrection, list[tuple[int, int, float]]]:
    """Try one bounded PAT-drift correction after metadata acceptance."""

    if not config.pat_residual_correction_enabled:
        return (
            ClockCorrection(
                **{
                    **correction.__dict__,
                    "residual_correction_status": "disabled",
                }
            ),
            baseline_matches,
        )
    if len(baseline_matches) < config.pat_residual_min_matches:
        return (
            ClockCorrection(
                **{
                    **correction.__dict__,
                    "residual_correction_status": "insufficient_matches",
                }
            ),
            baseline_matches,
        )
    intercept, slope, status = _estimate_pat_residual_slope(
        ecg_peaks_ms, baseline_matches, config
    )
    if status != "estimated" or abs(slope) <= np.finfo(float).eps:
        return (
            ClockCorrection(
                **{
                    **correction.__dict__,
                    "pat_slope_ms_per_minute": float(slope),
                    "residual_correction_status": (
                        "no_change" if status == "estimated" else f"rejected_{status}"
                    ),
                }
            ),
            baseline_matches,
        )

    candidate_start = float(correction.corrected_unix_ms[0])
    shift = slope * (correction.corrected_unix_ms - candidate_start) / 60_000.0
    candidate_timestamps = correction.corrected_unix_ms + shift
    if not np.all(np.diff(candidate_timestamps) > 0.0):
        status = "corrected_timestamps_not_monotonic"
        return (
            ClockCorrection(
                **{
                    **correction.__dict__,
                    "pat_slope_ms_per_minute": float(slope),
                    "residual_correction_status": f"rejected_{status}",
                }
            ),
            baseline_matches,
        )

    candidate_ecg_peaks = (
        ecg_peaks_ms + slope * (ecg_peaks_ms - ecg_peaks_ms[0]) / 60_000.0
    )
    candidate_matches = _match_pat_pairs(
        candidate_ecg_peaks,
        ppg_peaks_ms,
        config,
        pat_intercept_ms=intercept,
        pat_slope_ms_per_minute=0.0,
    )
    _, _, baseline_mad = _pat_statistics(baseline_matches)
    _, _, candidate_mad = _pat_statistics(candidate_matches)
    baseline_coverage = len(baseline_matches) / max(
        1, min(len(ecg_peaks_ms), len(ppg_peaks_ms))
    )
    candidate_coverage = len(candidate_matches) / max(
        1, min(len(ecg_peaks_ms), len(ppg_peaks_ms))
    )
    improves = candidate_coverage > baseline_coverage
    if (
        config.pat_residual_require_stability_improvement
        and candidate_coverage == baseline_coverage
    ):
        improves = (
            candidate_mad is not None
            and baseline_mad is not None
            and candidate_mad < baseline_mad
        )
    if not improves:
        return (
            ClockCorrection(
                **{
                    **correction.__dict__,
                    "pat_slope_ms_per_minute": float(slope),
                    "residual_correction_status": "rejected_no_improvement",
                }
            ),
            baseline_matches,
        )

    alpha = correction.alpha * (1.0 + slope / 60_000.0)
    if not config.alpha_min <= alpha <= config.alpha_max:
        return (
            ClockCorrection(
                **{
                    **correction.__dict__,
                    "pat_slope_ms_per_minute": float(slope),
                    "residual_correction_status": "rejected_alpha_out_of_bounds",
                }
            ),
            baseline_matches,
        )
    corrected = ClockCorrection(
        **{
            **correction.__dict__,
            "corrected_unix_ms": candidate_timestamps.astype(np.float64),
            "alpha": float(alpha),
            "beta_seconds": float(
                (candidate_timestamps[0] - correction.corrected_unix_ms[0]) / 1000.0
                + correction.beta_seconds
            ),
            "timebase_source": (
                f"{correction.timebase_source}+bounded_pat_residual_affine"
            ),
            "clock_selection_basis": "metadata_plus_bounded_pat_residual",
            "pat_slope_ms_per_minute": float(slope),
            "residual_correction_status": "accepted",
            "candidate_start_unix_ms": float(candidate_timestamps[0]),
            "candidate_end_unix_ms": float(candidate_timestamps[-1]),
        }
    )
    return corrected, candidate_matches


def _crop_observed_timebase(
    timestamps_ms: NDArray[np.floating],
    values: NDArray[np.floating],
    *,
    start_ms: float | None,
    end_ms: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Crop observed samples to the analysis core without padding."""

    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    if timestamps.shape != signal.shape:
        raise ValueError("timestamps_and_values_shape_mismatch")
    keep = np.isfinite(timestamps)
    if start_ms is not None:
        keep &= timestamps >= float(start_ms)
    if end_ms is not None:
        keep &= timestamps < float(end_ms)
    return timestamps[keep], signal[keep]


def validate_ecg_against_ppg(
    *,
    participant_id: str,
    recording_id: str,
    ecg_timestamps_ms: NDArray[np.floating],
    ecg_values: NDArray[np.floating],
    ppg_timestamps_ms: NDArray[np.floating],
    ppg_values: NDArray[np.floating],
    metadata: Mapping[str, str],
    config: SynchronizationConfig,
    phase_start_ms: float | None = None,
    phase_end_ms: float | None = None,
    diagnostic_reference_channel: str | None = None,
) -> tuple[ClockCorrection, SynchronizationResult]:
    """Correct ECG time and validate without shifting the PPG reference."""

    raw_ecg_timestamps = np.asarray(ecg_timestamps_ms, dtype=np.float64)
    raw_ecg_values = np.asarray(ecg_values, dtype=np.float64)
    raw_ppg_timestamps = np.asarray(ppg_timestamps_ms, dtype=np.float64)
    if (
        raw_ecg_timestamps.ndim != 1
        or raw_ecg_values.ndim != 1
        or raw_ecg_timestamps.shape != raw_ecg_values.shape
        or raw_ecg_timestamps.size < 3
        or not np.isfinite(raw_ecg_timestamps).all()
    ):
        return _excluded_input_result(
            participant_id=participant_id,
            recording_id=recording_id,
            ecg_timestamps_ms=raw_ecg_timestamps,
            ppg_timestamps_ms=raw_ppg_timestamps,
            reason="invalid_ecg_timestamp_input:sample_order_timeline_unavailable",
            diagnostic_reference_channel=diagnostic_reference_channel,
        )
    try:
        # The ECG sample order is the only valid pairing for metadata-derived
        # timebase construction.  This view is used only for the optional
        # pre-correction diagnostic and is never used to choose a clock.
        if (
            raw_ecg_values.size < 3
            or not np.isfinite(raw_ecg_values).sum() >= 3
            or np.any(np.diff(raw_ecg_timestamps) <= 0.0)
        ):
            pre_ecg_ts = pre_ecg_signal = None
        else:
            pre_ecg_ts, pre_ecg_signal = _finite_sample_order(
                raw_ecg_timestamps, raw_ecg_values
            )
    except (AssertionError, TypeError, ValueError) as exc:
        return _excluded_input_result(
            participant_id=participant_id,
            recording_id=recording_id,
            ecg_timestamps_ms=raw_ecg_timestamps,
            ppg_timestamps_ms=raw_ppg_timestamps,
            reason=f"invalid_ecg_input:{type(exc).__name__}:{exc}",
            diagnostic_reference_channel=diagnostic_reference_channel,
        )
    try:
        ppg_ts, ppg_signal = _finite_sorted(ppg_timestamps_ms, ppg_values)
    except (AssertionError, TypeError, ValueError) as exc:
        return _excluded_input_result(
            participant_id=participant_id,
            recording_id=recording_id,
            ecg_timestamps_ms=raw_ecg_timestamps,
            ppg_timestamps_ms=raw_ppg_timestamps,
            reason=f"invalid_ppg_input:{type(exc).__name__}:{exc}",
            diagnostic_reference_channel=diagnostic_reference_channel,
        )
    # Diagnostic only: estimate the physiological PAT match ratio before
    # applying the ECG correction. This view is never used to select the ECG
    # clock candidate or as primary ECG ground truth.  A non-monotonic raw
    # ECG clock has no valid pre-correction sample-order timestamp view.
    pre_alignment_score: float | None = None
    try:
        if pre_ecg_ts is None or pre_ecg_signal is None:
            raise ValueError("raw_ecg_timestamp_not_monotonic")
        pre_ecg_peaks, _, _, _ = _detect_reference_ecg_peaks(
            pre_ecg_ts,
            pre_ecg_signal,
            tolerance_ms=float(config.pat_tolerance_ms),
        )
        pre_ppg_peaks = detect_peak_times_ms(ppg_ts, ppg_signal, is_ecg=False)
        pre_pats = _match_physiological_pat(pre_ecg_peaks, pre_ppg_peaks, config)
        pre_denominator = max(1, min(pre_ecg_peaks.size, pre_ppg_peaks.size))
        pre_alignment_score = float(min(1.0, pre_pats.size / pre_denominator))
    except (AssertionError, ValueError, IndexError):
        pre_alignment_score = None
    correction = affine_clock_correction(
        raw_ecg_timestamps,
        metadata,
        config,
        reference_timestamps_ms=ppg_ts,
        phase_start_ms=phase_start_ms,
        phase_end_ms=phase_end_ms,
    )
    corrected = correction.corrected_unix_ms
    ecg_dropout_timebase = correction.corrected_unix_ms
    if phase_start_ms is not None and phase_end_ms is not None:
        overlap = max(
            0.0,
            (
                min(corrected[-1], float(phase_end_ms))
                - max(corrected[0], float(phase_start_ms))
            )
            / 1000.0,
        )
    else:
        # Compatibility fallback for unit callers without canonical anchors.
        overlap = max(
            0.0,
            (min(corrected[-1], ppg_ts[-1]) - max(corrected[0], ppg_ts[0])) / 1000.0,
        )
    clock_reasons: list[str] = []
    ppg_diagnostics: list[str] = []
    max_gap_ms = float(config.ecg_max_unfilled_gap_seconds) * 1000.0
    core_start = phase_start_ms
    core_end = phase_end_ms
    ecg_core_ts, ecg_core_values = _crop_observed_timebase(
        ecg_dropout_timebase,
        raw_ecg_values,
        start_ms=core_start,
        end_ms=core_end,
    )
    ppg_core_ts, ppg_core_values = _crop_observed_timebase(
        ppg_ts,
        ppg_signal,
        start_ms=core_start,
        end_ms=core_end,
    )
    ecg_long_dropout = (not correction.excluded) and _contains_long_dropout(
        ecg_core_ts,
        ecg_core_values,
        max_unfilled_gap_ms=max_gap_ms,
    )
    ppg_long_dropout = _contains_long_dropout(
        ppg_core_ts,
        ppg_core_values,
        max_unfilled_gap_ms=max_gap_ms,
    )
    if ecg_long_dropout:
        clock_reasons.append("ecg_long_dropout_detected")
    if ppg_long_dropout:
        ppg_diagnostics.append("ppg_long_dropout_detected")
    if correction.excluded and correction.exclusion_reason:
        clock_reasons.append(correction.exclusion_reason)
    if overlap < config.min_overlap_seconds:
        clock_reasons.append(f"insufficient_core_coverage:{overlap:.3f}s")

    reference_status = "UNAVAILABLE"
    reference_method = ""
    reference_reason = ""
    if correction.excluded or ecg_long_dropout:
        ecg_peaks = np.asarray([], dtype=np.float64)
        reference_reason = (
            "ecg_clock_excluded" if correction.excluded else "ecg_long_dropout"
        )
    else:
        (
            ecg_peaks,
            reference_method,
            reference_status,
            reference_reason,
        ) = _detect_reference_ecg_peaks(
            ecg_core_ts,
            ecg_core_values,
            tolerance_ms=float(config.pat_tolerance_ms),
        )
    if ppg_long_dropout:
        ppg_peaks = np.asarray([], dtype=np.float64)
    else:
        ppg_peaks = detect_peak_times_ms(ppg_core_ts, ppg_core_values, is_ecg=False)
    baseline_matches = (
        _match_pat_pairs(ecg_peaks, ppg_peaks, config)
        if ecg_peaks.size and ppg_peaks.size
        else []
    )
    correction, matches = _residual_correction(
        correction,
        ecg_peaks,
        ppg_peaks,
        baseline_matches,
        config,
    )
    correction = replace(
        correction,
        reference_status=reference_status,
        acceptance_reason=(
            correction.acceptance_reason
            if not reference_reason
            else ";".join(
                part
                for part in (correction.acceptance_reason, reference_reason)
                if part
            )
        ),
    )
    pats = np.asarray([delay for _, _, delay in matches], dtype=np.float64)
    if ecg_peaks.size < config.min_peak_count:
        clock_reasons.append(f"insufficient_ecg_peaks:{ecg_peaks.size}")
    if ppg_peaks.size < config.min_peak_count:
        ppg_diagnostics.append(f"insufficient_ppg_peaks:{ppg_peaks.size}")
    if pats.size < config.min_match_count:
        ppg_diagnostics.append(f"insufficient_pat_matches:{pats.size}")

    median_pat = float(np.median(pats)) if pats.size else None
    pat_iqr = (
        float(np.percentile(pats, 75) - np.percentile(pats, 25)) if pats.size else None
    )
    pat_mad = (
        float(np.median(np.abs(pats - median_pat)))
        if pats.size and median_pat is not None
        else None
    )
    denominator = max(1, min(ecg_peaks.size, ppg_peaks.size))
    confidence = float(min(1.0, pats.size / denominator))
    excluded = bool(clock_reasons)
    result = SynchronizationResult(
        participant_id=participant_id,
        recording_id=recording_id,
        alpha=correction.alpha,
        beta_seconds=correction.beta_seconds,
        method=(
            f"{correction.timebase_source}+"
            f"{reference_method or 'ecg_reference_unavailable'}"
            "+ppg_diagnostic_only"
        ),
        confidence=confidence,
        overlap_seconds=overlap,
        ecg_peak_count=int(ecg_peaks.size),
        ppg_peak_count=int(ppg_peaks.size),
        matched_peak_count=int(pats.size),
        median_pat_ms=median_pat,
        excluded=excluded,
        exclusion_reason=";".join(clock_reasons) if clock_reasons else None,
        pre_alignment_score=pre_alignment_score,
        post_alignment_score=confidence,
        clock_selection_basis=correction.clock_selection_basis,
        observed_sample_count=correction.observed_sample_count,
        declared_sample_count=correction.declared_sample_count,
        declared_duration_seconds=correction.declared_duration_seconds,
        metadata_stop_unix_ms=correction.metadata_stop_unix_ms,
        nominal_sample_rate_hz=correction.nominal_sample_rate_hz,
        effective_sample_rate_hz=correction.effective_sample_rate_hz,
        effective_rate_relative_error=correction.effective_rate_relative_error,
        candidate_start_unix_ms=correction.candidate_start_unix_ms,
        candidate_end_unix_ms=correction.candidate_end_unix_ms,
        reference_signal_role="diagnostic_only",
        ppg_diagnostic_reason=(";".join(ppg_diagnostics) if ppg_diagnostics else None),
        clock_excluded=excluded,
        clock_exclusion_reason=(";".join(clock_reasons) if clock_reasons else None),
        diagnostic_reference_channel=diagnostic_reference_channel,
        pat_iqr_ms=pat_iqr,
        pat_mad_ms=pat_mad,
        pat_slope_ms_per_minute=correction.pat_slope_ms_per_minute,
        unmatched_ecg_count=int(ecg_peaks.size - len(matches)),
        unmatched_ppg_count=int(
            ppg_peaks.size - len({index for _, index, _ in matches})
        ),
        matcher_method=(
            "monotonic_pat_affine"
            if correction.residual_correction_status == "accepted"
            else "monotonic_pat_constant"
        ),
        residual_correction_status=correction.residual_correction_status,
        reference_status=reference_status,
        timebase_source=correction.timebase_source,
    )
    return correction, result
