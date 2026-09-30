"""Shared, configurable PPG peak detection without duplicate filtering."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Literal, Protocol

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover - dependency-minimal installations
    nk = None
import numpy as np
from numpy.typing import NDArray
from scipy.signal import detrend, find_peaks

from .settings import ChannelPeakConfig, PeakDetectionConfig


@dataclass(frozen=True)
class PpgPeakDetection:
    peaks: NDArray[np.int64]
    troughs: NDArray[np.int64]
    method: str
    fallback_reason: str
    physiological_ibi_ratio: float
    status: Literal["AVAILABLE", "UNAVAILABLE"] = "AVAILABLE"
    status_reason: str = ""
    library_version: str = "unknown"
    config_fingerprint: str = ""


def _library_version(package: str) -> str:
    try:
        return f"{package}=={package_version(package)}"
    except PackageNotFoundError:
        return f"{package}==UNAVAILABLE"


class PpgPeakDetectorPort(Protocol):
    def minimum_sample_count(self, sampling_rate_hz: float) -> int:
        """Return the smallest segment that can contain the configured peaks."""

    def detect(
        self,
        signal: NDArray[np.floating],
        sampling_rate_hz: float,
        channel: str,
    ) -> PpgPeakDetection:
        """Detect peaks without filtering the canonical Silver waveform."""


def msptd_fast_boundary_safe(
    signal: NDArray[np.floating],
    sampling_rate_hz: float,
    *,
    window_seconds: float = 6.0,
    overlap_fraction: float = 0.20,
    minimum_scale_seconds: float = 0.50,
    tolerance_seconds: float = 0.15,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Run the MSPTDfast-v2 semantics used by the Android Green detector.

    NeuroKit2 0.2.13 can pass an empty edge slice to ``argmin``/``argmax``
    when a candidate lies within the refinement tolerance of a window edge.
    The Android contract clamps that refinement slice to the observed window.
    This small port keeps the same observed samples, window overlap, scalogram,
    and refinement rules while making that boundary behavior explicit.
    """

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("invalid_msptd_input")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("invalid_msptd_sampling_rate")
    window_samples = max(3, int(round(window_seconds * sampling_rate_hz)))
    windows = _msptd_windows(values.size, window_samples, overlap_fraction)
    peaks: list[int] = []
    troughs: list[int] = []
    for start, end in windows:
        segment = values[start:end]
        local_peaks, local_troughs = _msptd_window(
            segment,
            sampling_rate_hz,
            minimum_scale_seconds,
        )
        tolerance = max(1, int(np.ceil(sampling_rate_hz * tolerance_seconds)))
        peaks.extend(
            start + _refine_msptd_index(segment, index, tolerance, find_maximum=True)
            for index in local_peaks
        )
        troughs.extend(
            start + _refine_msptd_index(segment, index, tolerance, find_maximum=False)
            for index in local_troughs
        )
    return (
        np.unique(np.asarray(peaks, dtype=np.int64)),
        np.unique(np.asarray(troughs, dtype=np.int64)),
    )


def _msptd_windows(
    sample_count: int, window_samples: int, overlap_fraction: float
) -> tuple[tuple[int, int], ...]:
    if sample_count <= window_samples:
        return ((0, max(0, sample_count - 1)),)
    offset = max(1, int(round(window_samples * (1.0 - overlap_fraction))))
    starts = list(range(0, sample_count - window_samples + 1, offset))
    ends = [start + 1 + window_samples for start in starts]
    if ends[-1] < sample_count:
        starts.append(sample_count - 1 - window_samples)
        ends.append(sample_count)
    return tuple(zip(starts, ends, strict=True))


def _msptd_window(
    values: NDArray[np.float64],
    sampling_rate_hz: float,
    minimum_scale_seconds: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if values.size < 5:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    max_scale = int(np.ceil(values.size / 2.0)) - 1
    duration_seconds = values.size / sampling_rate_hz
    scales = tuple(
        scale
        for scale in range(1, max_scale + 1)
        if (max_scale / scale) / duration_seconds >= minimum_scale_seconds
    )
    if not scales:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    detrended = detrend(values, type="linear")
    maxima = np.zeros((len(scales), values.size), dtype=bool)
    minima = np.zeros((len(scales), values.size), dtype=bool)
    for row, scale in enumerate(scales):
        for index in range(scale, values.size - scale):
            maxima[row, index] = (
                detrended[index] > detrended[index - scale]
                and detrended[index] > detrended[index + scale]
            )
            minima[row, index] = (
                detrended[index] < detrended[index - scale]
                and detrended[index] < detrended[index + scale]
            )
    peak_lambda = int(np.argmax(np.sum(maxima, axis=1)))
    trough_lambda = int(np.argmax(np.sum(minima, axis=1)))
    return (
        _msptd_consensus(maxima, peak_lambda),
        _msptd_consensus(minima, trough_lambda),
    )


def _msptd_consensus(
    scalogram: NDArray[np.bool_], lambda_index: int
) -> NDArray[np.int64]:
    if lambda_index < 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.all(scalogram[: lambda_index + 1], axis=0)).astype(
        np.int64,
        copy=False,
    )


def _refine_msptd_index(
    values: NDArray[np.float64],
    index: int,
    tolerance: int,
    *,
    find_maximum: bool,
) -> int:
    start = max(0, index - tolerance)
    stop = min(values.size, index + tolerance + 1)
    if start >= stop:
        raise ValueError("invalid_msptd_refinement_window")
    segment = values[start:stop]
    relative = int(np.argmax(segment) if find_maximum else np.argmin(segment))
    return start + relative


class PpgPeakDetector(PpgPeakDetectorPort):
    """Detect PPG systolic peaks through a small, explicit candidate registry."""

    # NeuroKit2's Bishop implementation allocates an ``O(N²)`` scalogram and
    # its own documentation describes it for short signals (for example,
    # roughly six seconds at 100 Hz).  Applying it to a full 240-second
    # validation window would make the development benchmark impractical and
    # can exhaust memory.  Bounded overlapping views keep the method genuine:
    # every value is observed data, and no padding or interpolation is added.
    _BISHOP_CHUNK_SECONDS = 6.0
    _BISHOP_OVERLAP_FRACTION = 0.20
    _INTEGER_RATE_RELATIVE_TOLERANCE = 0.01

    _CANDIDATE_ALIASES = {
        "neurokit2_ppg_elgendi": "neurokit2_ppg_elgendi",
        "elgendi": "neurokit2_ppg_elgendi",
        "neurokit2_ppg_bishop": "neurokit2_ppg_bishop",
        "bishop": "neurokit2_ppg_bishop",
        "neurokit2_ppg_charlton_msptdfast": "neurokit2_ppg_charlton_msptdfast",
        "neurokit2_ppg_charlton": "neurokit2_ppg_charlton_msptdfast",
        "charlton": "neurokit2_ppg_charlton_msptdfast",
        "msptdfast": "neurokit2_ppg_charlton_msptdfast",
        "scipy_find_peaks": "scipy_find_peaks_adaptive",
        "scipy_find_peaks_adaptive": "scipy_find_peaks_adaptive",
        "scipy_adaptive": "scipy_find_peaks_adaptive",
        "scipy_legacy_green_native": "scipy_legacy_green_native",
        "legacy_green_native": "scipy_legacy_green_native",
    }
    _NEUROKIT_METHODS = {
        "neurokit2_ppg_elgendi": "elgendi",
        "neurokit2_ppg_bishop": "bishop",
        "neurokit2_ppg_charlton_msptdfast": "charlton",
    }

    def __init__(self, config: PeakDetectionConfig) -> None:
        self.config = config

    @classmethod
    def candidate_methods(cls) -> tuple[str, ...]:
        """Return canonical candidates supported by this registry."""

        return (
            "neurokit2_ppg_elgendi",
            "neurokit2_ppg_bishop",
            "neurokit2_ppg_charlton_msptdfast",
            "scipy_find_peaks_adaptive",
            "scipy_legacy_green_native",
        )

    def detect(
        self,
        signal: NDArray[np.floating],
        sampling_rate_hz: float,
        channel: str,
    ) -> PpgPeakDetection:
        values = self._validated_values(signal, sampling_rate_hz, channel)
        selected_method = self.config.selected_method_for(channel)
        primary = self.evaluate_candidate(
            values, sampling_rate_hz, channel, selected_method
        )
        if primary.status == "AVAILABLE" and self._is_usable(
            primary.peaks, primary.physiological_ibi_ratio
        ):
            return primary

        if primary.status == "UNAVAILABLE":
            fallback_reason = f"{selected_method}:{primary.status_reason}"
        else:
            fallback_reason = (
                f"{selected_method}:ValueError:primary_quality_failed:"
                f"peaks={len(primary.peaks)};"
                f"ibi_ratio={primary.physiological_ibi_ratio:.6f}"
            )

        if not self.config.allow_runtime_fallback:
            return self._unavailable_detection(selected_method, fallback_reason)

        fallback = self.evaluate_candidate(
            values, sampling_rate_hz, channel, self.config.fallback_method
        )
        return PpgPeakDetection(
            fallback.peaks,
            fallback.troughs,
            f"{self.config.fallback_method}_fallback",
            fallback_reason,
            fallback.physiological_ibi_ratio,
            fallback.status,
            fallback.status_reason,
            fallback.library_version,
            fallback.config_fingerprint,
        )

    def evaluate_candidate(
        self,
        signal: NDArray[np.floating],
        sampling_rate_hz: float,
        channel: str,
        method: str,
    ) -> PpgPeakDetection:
        """Evaluate one registry candidate without silently substituting another.

        The method validates the canonical, already-cleaned signal before calling
        any detector. Unsupported or unavailable NeuroKit2 methods return an
        explicit ``UNAVAILABLE`` result with no fabricated peaks.
        """

        values = self._validated_values(signal, sampling_rate_hz, channel)
        canonical_method = self._CANDIDATE_ALIASES.get(method.lower())
        if canonical_method is None:
            return self._unavailable_detection(
                method, f"unsupported_candidate_method:{method}"
            )
        if canonical_method == "scipy_legacy_green_native" and channel != "ppg_green":
            return self._unavailable_detection(
                canonical_method,
                "legacy_green_candidate_is_not_valid_for_ir_or_red",
            )

        try:
            if canonical_method == "scipy_find_peaks_adaptive":
                peaks, troughs = self._scipy_detect(values, sampling_rate_hz, channel)
            elif canonical_method == "scipy_legacy_green_native":
                peaks, troughs = self._legacy_native_detect(values, sampling_rate_hz)
            elif canonical_method == "neurokit2_ppg_bishop":
                peaks, troughs = self._bishop_detect(values, sampling_rate_hz)
            else:
                if nk is None:
                    raise ImportError("neurokit2 is not installed")
                neurokit_sampling_rate = sampling_rate_hz
                if canonical_method == "neurokit2_ppg_charlton_msptdfast":
                    rounded_rate = round(sampling_rate_hz)
                    if not np.isclose(
                        sampling_rate_hz,
                        rounded_rate,
                        rtol=self._INTEGER_RATE_RELATIVE_TOLERANCE,
                        atol=0.0,
                    ):
                        raise ValueError(
                            "MSPTDfast requires an integer sampling rate near the "
                            "observed rate; "
                            f"actual_rate_hz={sampling_rate_hz:g}"
                        )
                    # MSPTDfast's API requires an integer rate. Rounding this
                    # metadata parameter does not resample the waveform: the
                    # observed samples, timestamps, and actual rate remain the
                    # source for interval and physiological-ratio calculations.
                    neurokit_sampling_rate = rounded_rate
                try:
                    info = nk.ppg_findpeaks(
                        values,
                        sampling_rate=neurokit_sampling_rate,
                        method=self._NEUROKIT_METHODS[canonical_method],
                    )
                    peaks = self._peak_indices(info.get("PPG_Peaks", []), len(values))
                except ValueError as error:
                    if canonical_method != "neurokit2_ppg_charlton_msptdfast":
                        raise
                    # NeuroKit2 0.2.13 has an empty refinement slice at a
                    # window boundary for some valid short-rate segments. The
                    # run-scoped Android contract clamps that slice; use the
                    # same observed-sample port instead of silently changing
                    # detector method or inventing a boundary sample.
                    if "empty sequence" not in str(error):
                        raise
                    peaks, _ = msptd_fast_boundary_safe(
                        values,
                        float(neurokit_sampling_rate),
                    )
                _, troughs = self._scipy_detect(values, sampling_rate_hz, channel)

            return PpgPeakDetection(
                peaks,
                troughs,
                canonical_method,
                "",
                self._physiological_ratio(peaks, sampling_rate_hz),
                library_version=(
                    _library_version("scipy")
                    if canonical_method.startswith("scipy_")
                    else _library_version("neurokit2")
                ),
                config_fingerprint=self._config_fingerprint(channel, canonical_method),
            )
        except Exception as exc:
            return self._unavailable_detection(
                canonical_method, f"{type(exc).__name__}:{exc}"
            )

    def _bishop_detect(
        self,
        values: NDArray[np.float64],
        sampling_rate_hz: float,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Run Bishop on bounded observed chunks and stitch local indices.

        The upstream Bishop algorithm is intentionally not called on a long
        recording because its scalogram scales quadratically with the input
        length.  Overlap gives beats near chunk boundaries an observed context
        without changing timestamps or fabricating samples.  Duplicate local
        detections from overlapping views are removed deterministically.
        """

        chunk_size = max(3, int(round(self._BISHOP_CHUNK_SECONDS * sampling_rate_hz)))
        if values.size <= chunk_size:
            starts = (0,)
        else:
            step = max(
                1,
                int(round(chunk_size * (1.0 - self._BISHOP_OVERLAP_FRACTION))),
            )
            starts_list = list(range(0, values.size - chunk_size + 1, step))
            final_start = values.size - chunk_size
            if starts_list[-1] != final_start:
                starts_list.append(final_start)
            starts = tuple(starts_list)

        peaks: list[int] = []
        troughs: list[int] = []
        for start in starts:
            stop = min(values.size, start + chunk_size)
            chunk = values[start:stop]
            if chunk.size < 3:
                continue
            if nk is None:
                raise ImportError("neurokit2 is not installed")
            info = nk.ppg_findpeaks(
                chunk,
                sampling_rate=sampling_rate_hz,
                method="bishop",
            )
            # A third-party detector may return an index outside the bounded
            # view.  Ignore that invalid reference rather than failing the
            # whole candidate; no observed sample is created by this filter.
            local_peaks = self._bounded_peak_indices(
                info.get("PPG_Peaks", []), chunk.size
            )
            local_troughs = self._bounded_peak_indices(
                info.get("PPG_Onsets", []), chunk.size
            )
            peaks.extend((local_peaks + start).tolist())
            troughs.extend((local_troughs + start).tolist())

        return (
            np.unique(np.asarray(peaks, dtype=np.int64)),
            np.unique(np.asarray(troughs, dtype=np.int64)),
        )

    @staticmethod
    def _bounded_peak_indices(
        raw_indices: object, sample_count: int
    ) -> NDArray[np.int64]:
        """Keep only integer peak references inside an observed segment."""

        try:
            indices = np.asarray(raw_indices, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return np.empty(0, dtype=np.int64)
        finite = indices[np.isfinite(indices)]
        integer = finite[np.equal(finite, np.floor(finite))]
        bounded = integer[(integer >= 0.0) & (integer < float(sample_count))]
        return np.unique(bounded.astype(np.int64, copy=False))

    def minimum_sample_count(self, sampling_rate_hz: float) -> int:
        """Derive segment length from peak count and the fastest allowed IBI."""

        minimum_duration_seconds = (
            self.config.minimum_peak_count
            * self.config.physiological_ibi_min_ms
            / 1000.0
        )
        return max(3, int(np.ceil(minimum_duration_seconds * sampling_rate_hz)))

    def detect_fallback(
        self,
        signal: NDArray[np.floating],
        sampling_rate_hz: float,
        channel: str,
    ) -> PpgPeakDetection:
        """Evaluate the configured SciPy fallback directly for development audit."""

        result = self.evaluate_candidate(
            signal, sampling_rate_hz, channel, self.config.fallback_method
        )
        return PpgPeakDetection(
            result.peaks,
            result.troughs,
            f"{self.config.fallback_method}_development_audit",
            "",
            result.physiological_ibi_ratio,
            result.status,
            result.status_reason,
            result.library_version,
            result.config_fingerprint,
        )

    def _config_fingerprint(self, channel: str, method: str) -> str:
        """Return a stable detector/config identity for persisted evidence."""

        import hashlib
        import json

        channel_config = self.config.for_channel(channel)
        payload = {
            "method": method,
            "channel": channel,
            "channel_config": repr(channel_config),
            "minimum_peak_count": self.config.minimum_peak_count,
            "physiological_ibi_min_ms": self.config.physiological_ibi_min_ms,
            "physiological_ibi_max_ms": self.config.physiological_ibi_max_ms,
            "legacy_native": repr(self.config.legacy_native),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _validated_values(
        self,
        signal: NDArray[np.floating],
        sampling_rate_hz: float,
        channel: str,
    ) -> NDArray[np.float64]:
        values = np.asarray(signal, dtype=np.float64)
        if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
            raise ValueError("PPG sampling rate must be finite and positive")
        if values.ndim != 1:
            raise ValueError("PPG signal must be one-dimensional")
        if values.size < self.minimum_sample_count(sampling_rate_hz):
            raise ValueError("PPG signal is too short for peak detection")
        if not np.isfinite(values).all():
            raise ValueError("PPG signal contains non-finite values")
        self.config.for_channel(channel)
        return values

    @staticmethod
    def _peak_indices(peaks: object, signal_length: int) -> NDArray[np.int64]:
        values = np.asarray(peaks)
        if values.ndim != 1:
            raise ValueError("detector returned non-vector peak indices")
        if values.size == 0:
            return np.empty(0, dtype=np.int64)
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("detector returned non-finite peak indices")
        integer_values = np.rint(values).astype(np.int64)
        if not np.array_equal(values, integer_values):
            raise ValueError("detector returned non-integer peak indices")
        if np.any(integer_values < 0) or np.any(integer_values >= signal_length):
            raise ValueError("detector returned out-of-range peak indices")
        return np.unique(integer_values)

    @staticmethod
    def _unavailable_detection(method: str, reason: str) -> PpgPeakDetection:
        empty = np.empty(0, dtype=np.int64)
        return PpgPeakDetection(
            empty,
            empty,
            method,
            "",
            0.0,
            "UNAVAILABLE",
            reason,
            _library_version(
                "scipy" if method.lower().startswith("scipy") else "neurokit2"
            ),
        )

    def _is_usable(self, peaks: NDArray[np.int64], ibi_ratio: float) -> bool:
        return (
            len(peaks) >= self.config.minimum_peak_count
            and ibi_ratio >= self.config.minimum_physiological_ibi_ratio
        )

    def _physiological_ratio(
        self, peaks: NDArray[np.int64], sampling_rate_hz: float
    ) -> float:
        if len(peaks) < 2:
            return 0.0
        ibi_ms = np.diff(peaks).astype(float) * 1000.0 / sampling_rate_hz
        physiological = (ibi_ms >= self.config.physiological_ibi_min_ms) & (
            ibi_ms <= self.config.physiological_ibi_max_ms
        )
        return float(np.mean(physiological))

    def _scipy_detect(
        self,
        values: NDArray[np.float64],
        sampling_rate_hz: float,
        channel: str,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        channel_config = self.config.for_channel(channel)
        centered = values - np.median(values)
        mad = float(np.median(np.abs(centered)))
        prominence = max(
            np.finfo(float).eps,
            channel_config.prominence_mad_multiplier * mad,
        )
        distance = max(
            1,
            int(round(channel_config.min_distance_seconds * sampling_rate_hz)),
        )
        width = self._width_samples(channel_config, sampling_rate_hz)
        peaks, _ = find_peaks(
            centered,
            distance=distance,
            prominence=prominence,
            width=width,
        )
        troughs, _ = find_peaks(
            -centered,
            distance=distance,
            prominence=prominence,
            width=width,
        )
        return peaks.astype(np.int64), troughs.astype(np.int64)

    def _legacy_native_detect(
        self,
        values: NDArray[np.float64],
        sampling_rate_hz: float,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Evaluate the old segment-local Green threshold on observed samples.

        The historical cubic upsampling remains excluded because it adds no
        observations. Thresholds and the no-peak retry are recomputed within
        each 30-second view, matching the useful adaptive part of the legacy
        method without waveform reconstruction.
        """

        candidate = self.config.legacy_native
        segment_samples = max(
            3, int(round(candidate.segment_seconds * sampling_rate_hz))
        )
        distance = max(
            1,
            int(round(candidate.min_distance_seconds * sampling_rate_hz)),
        )
        margin = int(np.ceil(candidate.edge_margin_seconds * sampling_rate_hz))
        peak_indices: list[int] = []
        trough_indices: list[int] = []
        for start in range(0, len(values), segment_samples):
            stop = min(len(values), start + segment_samples)
            segment = values[start:stop]
            if len(segment) < max(3, 2 * margin + 1):
                continue
            centered = segment - np.median(segment)
            scale = float(np.std(centered))
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                continue
            prominence = candidate.prominence_std_multiplier * scale
            peaks, _ = find_peaks(
                centered,
                distance=distance,
                prominence=prominence,
                height=candidate.height_std_multiplier * scale,
            )
            if peaks.size == 0:
                peaks, _ = find_peaks(
                    centered,
                    distance=distance,
                    prominence=prominence * candidate.retry_prominence_scale,
                )
            troughs, _ = find_peaks(
                -centered,
                distance=distance,
                prominence=prominence,
            )
            if margin:
                peaks = peaks[(peaks >= margin) & (peaks < len(segment) - margin)]
                troughs = troughs[
                    (troughs >= margin) & (troughs < len(segment) - margin)
                ]
            peak_indices.extend((peaks + start).tolist())
            trough_indices.extend((troughs + start).tolist())
        return (
            np.asarray(peak_indices, dtype=np.int64),
            np.asarray(trough_indices, dtype=np.int64),
        )

    def _troughs(
        self,
        values: NDArray[np.float64],
        sampling_rate_hz: float,
        channel: str,
    ) -> NDArray[np.int64]:
        _, troughs = self._scipy_detect(values, sampling_rate_hz, channel)
        return troughs

    @staticmethod
    def _width_samples(
        channel_config: ChannelPeakConfig, sampling_rate_hz: float
    ) -> tuple[float | None, float | None] | None:
        minimum = channel_config.min_width_seconds * sampling_rate_hz
        maximum = (
            channel_config.max_width_seconds * sampling_rate_hz
            if channel_config.max_width_seconds is not None
            else None
        )
        if minimum <= 0.0 and maximum is None:
            return None
        return (minimum if minimum > 0.0 else None, maximum)
