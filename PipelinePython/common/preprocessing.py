"""Signal filtering, resampling, and compatibility imports.

Single Responsibility:
  - SignalFilter  → removes noise from a 1-D signal array
  - Resampler     → changes the sampling rate of a DataFrame

These classes operate on numpy arrays / DataFrames and are stateless;
they can be composed freely without side effects (Open/Closed).

References:
  SciPy Butterworth filtering: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
  SciPy zero-phase filtering: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
  SciPy polyphase resampling: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import gcd
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from .config import (
    BUTTER_HIGHCUT,
    BUTTER_LOWCUT,
    BUTTER_ORDER,
    ECG_FS,
    PPG_FS,
    POWERLINE_FREQ,
    TARGET_FS,
    filter_config_for_channel,
    load_cleaning_config,
)

logger = logging.getLogger(__name__)

from .madc import MotionArtifactDetector  # noqa: E402

# Compatibility alias; MotionArtifactDetector is the only MADC implementation.
MADC = MotionArtifactDetector


# ---------------------------------------------------------------------------
# SignalFilter
# ---------------------------------------------------------------------------


class SignalFilter:
    """
    Stateless namespace for bandpass and notch filtering.

    All methods are @staticmethod so they can also be called without
    instantiating the class (e.g. SignalFilter.bandpass(...)).
    """

    @staticmethod
    def bandpass(
        signal: np.ndarray,
        fs: float,
        lowcut: float = BUTTER_LOWCUT,
        highcut: float = BUTTER_HIGHCUT,
        order: int = BUTTER_ORDER,
    ) -> np.ndarray:
        """
        Apply a zero-phase Butterworth bandpass filter.

        Removes sub-Hz baseline wander and HF motion artefacts while
        preserving the physiologically relevant 0.5–40 Hz band.

        Args:
            signal:  1-D raw signal array.
            fs:      Sampling frequency in Hz.
            lowcut:  Lower cutoff frequency.
            highcut: Upper cutoff frequency.
            order:   Filter order (default 4 → 8-pole zero-phase).

        Returns:
            Filtered signal as a numpy array of the same length.
        """
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1 or signal.size == 0:
            raise ValueError("signal must be a non-empty one-dimensional array")
        if not np.isfinite(signal).all():
            raise ValueError("signal must contain only finite values")
        if fs <= 0 or lowcut <= 0 or highcut <= 0:
            raise ValueError("fs and cutoff frequencies must be positive")
        nyquist = 0.5 * fs
        if highcut >= nyquist or lowcut >= highcut:
            raise ValueError(
                f"bandpass: require 0 < low ({lowcut}) < high ({highcut}) < Nyquist ({nyquist}) "
                f"(fs={fs}, lowcut={lowcut}, highcut={highcut})"
            )
        low = lowcut / nyquist
        high = highcut / nyquist

        sos = butter(order, [low, high], btype="band", output="sos")
        pad_length = 3 * (2 * len(sos) + 1)
        if signal.size <= pad_length:
            raise ValueError(
                f"signal is too short for SOS zero-phase filtering: {signal.size} <= {pad_length}"
            )
        return sosfiltfilt(sos, signal)

    @staticmethod
    def notch(
        signal: np.ndarray,
        fs: float,
        freq: float = POWERLINE_FREQ,
        quality_factor: float = 30.0,
    ) -> np.ndarray:
        """
        Apply an IIR notch filter to remove powerline interference (50 Hz).

        Args:
            signal:         1-D raw signal array.
            fs:             Sampling frequency in Hz.
            freq:           Frequency to notch out (50 Hz for Indonesia).
            quality_factor: Higher Q → narrower notch (less distortion).

        Returns:
            Filtered signal array.
        """
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1 or signal.size == 0:
            raise ValueError("signal must be a non-empty one-dimensional array")
        if not np.isfinite(signal).all():
            raise ValueError("signal must contain only finite values")
        if fs <= 0 or freq <= 0:
            raise ValueError("fs and notch frequency must be positive")
        if freq >= fs / 2:
            # Can't filter a frequency above Nyquist; skip gracefully
            logger.debug(
                "Notch freq %.1f Hz >= Nyquist %.1f Hz; skipping.", freq, fs / 2
            )
            return signal
        b, a = iirnotch(freq, quality_factor, fs)
        pad_length = 3 * max(len(a), len(b))
        if signal.size <= pad_length:
            raise ValueError(
                f"signal is too short for zero-phase filtering: {signal.size} <= {pad_length}"
            )
        return filtfilt(b, a, signal)

    @classmethod
    def apply_all(
        cls,
        signal: np.ndarray,
        fs: float,
        lowcut: float = BUTTER_LOWCUT,
        highcut: float = BUTTER_HIGHCUT,
    ) -> np.ndarray:
        """
        Convenience method: notch filter then bandpass filter in one call.

        Order matters: powerline removal first preserves the bandpass
        transition band integrity.
        """
        filtered = cls.notch(signal, fs)
        # Respect the actual Nyquist frequency for low-rate PPG recordings;
        # the configured 40 Hz ECG upper band is not valid at 25/50 Hz PPG.
        safe_highcut = min(highcut, 0.45 * float(fs))
        filtered = cls.bandpass(filtered, fs, lowcut, safe_highcut)
        return filtered


@dataclass(frozen=True)
class InterpolationResult:
    """Result of bounded waveform interpolation with explicit provenance."""

    values: np.ndarray
    observed_mask: np.ndarray
    interpolated_mask: np.ndarray
    gap_mask: np.ndarray
    unresolved_mask: np.ndarray
    runs: tuple[tuple[int, int], ...]
    intervals: tuple[tuple[int, int, str], ...]
    long_dropout: bool
    unbounded_dropout: bool

    @property
    def mask(self) -> np.ndarray:
        """Compatibility alias for the historical interpolation mask."""
        return self.interpolated_mask


@dataclass(frozen=True)
class CleanedPpgFrame:
    """Pure preprocessing result consumed by Silver and downstream windows."""

    frame: pd.DataFrame
    clean_value: np.ndarray
    observed_mask: np.ndarray
    interpolated_mask: np.ndarray
    gap_mask: np.ndarray
    unresolved_mask: np.ndarray
    interpolation_provenance: Mapping[str, Any]
    filter_spec: Mapping[str, Any]
    filter_applied: bool
    exclusion_reason: str | None


def interpolate_short_gaps(
    values: np.ndarray,
    timestamps: pd.Series | np.ndarray,
    sampling_rate_hz: float,
    max_gap_seconds: float = 0.5,
    *,
    method: str = "linear",
    phase_labels: pd.Series | np.ndarray | None = None,
    pchip_min_neighbors_each_side: int = 2,
) -> InterpolationResult:
    """Interpolate bounded internal NaN runs at existing timestamps only.

    No timestamp or sample row is created. ``phase_labels`` must be supplied
    when one frame can contain multiple phases; a run crossing a phase boundary
    remains unresolved. PCHIP is opt-in and requires at least two contiguous,
    finite source neighbors on each side.
    """
    signal = np.asarray(values, dtype=float).copy()
    if signal.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if not np.isfinite(max_gap_seconds) or not 0 < max_gap_seconds <= 0.5:
        raise ValueError("max_gap_seconds must be finite and in (0, 0.5]")
    interpolation_method = str(method).strip().lower()
    if interpolation_method not in {"linear", "pchip"}:
        raise ValueError("method must be 'linear' or 'pchip'")
    if pchip_min_neighbors_each_side < 2:
        raise ValueError("pchip_min_neighbors_each_side must be at least 2")

    timestamps_utc = pd.to_datetime(timestamps, utc=True, errors="coerce")
    if len(timestamps_utc) != signal.size:
        raise ValueError("timestamps length must match values length")
    if pd.isna(timestamps_utc).any():
        raise ValueError("timestamps must be finite")
    timestamp_ms = (
        np.asarray(timestamps_utc.astype("int64"), dtype=np.float64) / 1_000_000.0
    )
    expected_ms = 1000.0 / float(sampling_rate_hz)
    timestamp_gaps = np.diff(timestamp_ms)
    if np.any(timestamp_gaps <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    max_gap_ms = max_gap_seconds * 1000.0
    long_dropout = bool(np.any(timestamp_gaps - expected_ms > max_gap_ms))

    phases: np.ndarray | None = None
    if phase_labels is not None:
        phases = np.asarray(phase_labels, dtype=object)
        if phases.ndim != 1 or phases.size != signal.size:
            raise ValueError("phase_labels length must match values length")

    missing = np.isnan(signal)
    padded = np.concatenate(([False], missing, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    observed_mask = np.isfinite(signal)
    interpolated_mask = np.zeros(signal.size, dtype=bool)
    gap_mask = np.zeros(signal.size, dtype=bool)
    unresolved_mask = ~np.isfinite(signal) & ~missing
    runs: list[tuple[int, int]] = []
    intervals: list[tuple[int, int, str]] = []
    unbounded_dropout = False
    max_samples = max(1, int(np.floor(max_gap_seconds * sampling_rate_hz)))

    for start, end in zip(starts, ends):
        if start == 0 or end == signal.size - 1:
            unresolved_mask[start : end + 1] = True
            intervals.append((int(start), int(end), "unresolved_edge"))
            unbounded_dropout = True
            continue

        phase_slice = phases[start - 1 : end + 2] if phases is not None else None
        if phase_slice is not None and (
            pd.isna(phase_slice).any() or not np.all(phase_slice == phase_slice[0])
        ):
            unresolved_mask[start : end + 1] = True
            intervals.append((int(start), int(end), "unresolved_phase_boundary"))
            continue

        gap_duration_ms = timestamp_ms[end + 1] - timestamp_ms[start - 1] - expected_ms
        if end - start + 1 > max_samples or gap_duration_ms > max_gap_ms:
            gap_mask[start : end + 1] = True
            intervals.append((int(start), int(end), "gap_too_long"))
            long_dropout = True
            continue
        if not np.isfinite(signal[start - 1]) or not np.isfinite(signal[end + 1]):
            unresolved_mask[start : end + 1] = True
            intervals.append((int(start), int(end), "unresolved_neighbors"))
            unbounded_dropout = True
            continue

        indices = np.arange(start, end + 1)
        if interpolation_method == "linear":
            signal[indices] = np.interp(
                timestamp_ms[indices],
                timestamp_ms[[start - 1, end + 1]],
                signal[[start - 1, end + 1]],
            )
        else:
            left = np.arange(start - pchip_min_neighbors_each_side, start)
            right = np.arange(end + 1, end + 1 + pchip_min_neighbors_each_side)
            source_indices = np.concatenate((left, right))
            source_is_valid = (
                source_indices[0] >= 0
                and source_indices[-1] < signal.size
                and np.isfinite(signal[source_indices]).all()
                and not missing[source_indices].any()
            )
            if source_is_valid and phases is not None:
                source_phases = phases[source_indices]
                source_is_valid = bool(
                    not pd.isna(source_phases).any()
                    and np.all(source_phases == phase_slice[0])
                )
            if not source_is_valid:
                unresolved_mask[indices] = True
                intervals.append((int(start), int(end), "unresolved_pchip_neighbors"))
                continue
            signal[indices] = PchipInterpolator(
                timestamp_ms[source_indices],
                signal[source_indices],
                extrapolate=False,
            )(timestamp_ms[indices])
            if not np.isfinite(signal[indices]).all():
                signal[indices] = np.nan
                unresolved_mask[indices] = True
                intervals.append((int(start), int(end), "unresolved_pchip_result"))
                continue

        interpolated_mask[indices] = True
        runs.append((int(start), int(end)))
        intervals.append((int(start), int(end), f"interpolated_{interpolation_method}"))

    category_count = (
        observed_mask.astype(np.uint8)
        + interpolated_mask.astype(np.uint8)
        + gap_mask.astype(np.uint8)
        + unresolved_mask.astype(np.uint8)
    )
    if not np.all(category_count == 1):
        raise RuntimeError("interpolation provenance masks must form one partition")

    return InterpolationResult(
        values=signal,
        observed_mask=observed_mask,
        interpolated_mask=interpolated_mask,
        gap_mask=gap_mask,
        unresolved_mask=unresolved_mask,
        runs=tuple(runs),
        intervals=tuple(intervals),
        long_dropout=long_dropout,
        unbounded_dropout=unbounded_dropout,
    )


class PpgPreprocessor:
    """Canonical PPG cleaning boundary; no I/O and no motion correction."""

    _CHANNELS = frozenset({"ppg_ir", "ppg_red", "ppg_green"})

    def __init__(self, cleaning_config: dict | None = None) -> None:
        self.cleaning_config = cleaning_config or load_cleaning_config()

    @staticmethod
    def _value_column(frame: pd.DataFrame) -> str:
        for name in ("raw_value", "ppg_raw", "signal", "clean_value"):
            if name in frame.columns:
                return name
        raise KeyError("PPG frame requires raw_value, ppg_raw, signal, or clean_value")

    def preprocess(
        self,
        frame: pd.DataFrame,
        channel: str,
        sampling_rate_hz: float,
    ) -> CleanedPpgFrame:
        """Return a sorted, deduplicated, interpolated and filtered PPG frame."""
        if channel not in self._CHANNELS:
            raise ValueError(f"Unsupported PPG channel: {channel}")
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")

        timestamp_column = next(
            (
                name
                for name in (
                    "timestamp_corrected_utc",
                    "timestamp_raw_utc",
                    "timestamp",
                )
                if name in frame.columns
            ),
            None,
        )
        if timestamp_column is None:
            raise KeyError("PPG frame requires a timestamp column")

        working = frame.copy()
        working["__timestamp"] = pd.to_datetime(
            working[timestamp_column], utc=True, errors="coerce"
        )
        working = working.dropna(subset=["__timestamp"])
        # Deduplicate exact timestamp repeats deterministically.  The first
        # source row remains the lineage representative; raw files are never
        # overwritten.
        working = working.drop_duplicates(subset=["__timestamp"], keep="first")
        working = working.sort_values("__timestamp", kind="stable").reset_index(
            drop=True
        )
        values = pd.to_numeric(working[self._value_column(working)], errors="coerce")
        timestamps = working["__timestamp"]
        max_gap_seconds = float(
            self.cleaning_config.get("max_interpolation_seconds", 0.5)
        )
        interpolation_method = str(
            self.cleaning_config.get("interpolation_method", "linear")
        )
        interpolation = interpolate_short_gaps(
            values.to_numpy(dtype=float),
            timestamps,
            sampling_rate_hz,
            max_gap_seconds=max_gap_seconds,
            method=interpolation_method,
            phase_labels=working["phase"] if "phase" in working else None,
            pchip_min_neighbors_each_side=int(
                self.cleaning_config.get("pchip_min_neighbors_each_side", 2)
            ),
        )
        spec = filter_config_for_channel(channel, self.cleaning_config)
        clean = interpolation.values.copy()
        filter_applied = False
        exclusion_reason: str | None = None
        lowcut = float(spec["lowcut_hz"])
        highcut = float(spec["highcut_hz"])
        order = int(spec.get("order", 4))
        if interpolation.long_dropout:
            exclusion_reason = "long_dropout_rejected"
        elif interpolation.unbounded_dropout:
            exclusion_reason = "unbounded_dropout_rejected"
        elif np.isfinite(clean).all():
            clean = SignalFilter.bandpass(
                clean,
                fs=float(sampling_rate_hz),
                lowcut=lowcut,
                highcut=min(highcut, 0.45 * float(sampling_rate_hz)),
                order=order,
            )
            filter_applied = True
        else:
            exclusion_reason = "nonfinite_signal_rejected"

        working["clean_value"] = clean
        working["observed_mask"] = interpolation.observed_mask
        working["interpolated_mask"] = interpolation.interpolated_mask
        working["gap_mask"] = interpolation.gap_mask
        working["unresolved_mask"] = interpolation.unresolved_mask
        working["interpolated_short_gap"] = bool(interpolation.interpolated_mask.any())
        working["interpolated_sample_count"] = int(
            interpolation.interpolated_mask.sum()
        )
        working["interpolation_provenance"] = (
            f"bounded_{interpolation_method}_waveform"
            if interpolation.interpolated_mask.any()
            else ""
        )
        working = working.drop(columns=["__timestamp"])
        provenance = {
            "method": f"bounded_{interpolation_method}_waveform",
            "max_gap_seconds": max_gap_seconds,
            "runs": [list(run) for run in interpolation.runs],
            "intervals": [list(interval) for interval in interpolation.intervals],
            "observed_sample_count": int(interpolation.observed_mask.sum()),
            "interpolated_sample_count": int(interpolation.interpolated_mask.sum()),
            "gap_sample_count": int(interpolation.gap_mask.sum()),
            "unresolved_sample_count": int(interpolation.unresolved_mask.sum()),
            "timestamp_grid_synthesized": False,
            "extrapolation_enabled": False,
            "long_dropout_reconstructed": False,
            "filter_application": "once" if filter_applied else "not_applied",
        }
        return CleanedPpgFrame(
            frame=working,
            clean_value=clean,
            observed_mask=interpolation.observed_mask,
            interpolated_mask=interpolation.interpolated_mask,
            gap_mask=interpolation.gap_mask,
            unresolved_mask=interpolation.unresolved_mask,
            interpolation_provenance=provenance,
            filter_spec=spec,
            filter_applied=filter_applied,
            exclusion_reason=exclusion_reason,
        )


# ---------------------------------------------------------------------------
# Resampler
# ---------------------------------------------------------------------------


class Resampler:
    """
    Resamples a time-series DataFrame to a target sampling rate using
    polyphase filtering (scipy.signal.resample_poly).

    Polyphase resampling is preferred over FFT-based resample() for
    physiological signals because it avoids Gibbs artefacts at boundaries.

    Usage:
        resampler = Resampler(source_fs=130, target_fs=100)
        df_resampled = resampler.resample(df, signal_col="ecg_mv")
    """

    def __init__(
        self,
        source_fs: float,
        target_fs: float = TARGET_FS,
    ) -> None:
        if source_fs <= 0 or target_fs <= 0:
            raise ValueError("source_fs and target_fs must be positive")
        self._source_fs = float(source_fs)
        self._target_fs = float(target_fs)

        # Compute up/down integers for polyphase filter (integer ratios only)
        src = int(round(source_fs))
        tgt = int(round(target_fs))
        common = gcd(src, tgt)
        self._up = tgt // common
        self._down = src // common

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resample_signal(self, signal: np.ndarray) -> np.ndarray:
        """Resample a 1-D signal array from source_fs to target_fs."""
        values = np.asarray(signal, dtype=float)
        if values.ndim != 1:
            raise ValueError("signal must be a one-dimensional array")
        if values.size == 0:
            return values.copy()
        if not np.isfinite(values).all():
            raise ValueError("signal must contain only finite values")
        return resample_poly(values, self._up, self._down)

    def resample_dataframe(
        self,
        df: pd.DataFrame,
        signal_col: str,
        unix_col: str = "unix_ms",
    ) -> pd.DataFrame:
        """
        Resample a DataFrame's signal column and re-compute timestamps.

        Args:
            df:         Input DataFrame with at least unix_col and signal_col.
            signal_col: Name of the column containing signal values.
            unix_col:   Name of the Unix timestamp column (milliseconds).

        Returns:
            New DataFrame with resampled signal and linearly spaced timestamps.
        """
        if df.empty:
            return df
        if unix_col not in df or signal_col not in df:
            raise KeyError(f"DataFrame must contain {unix_col!r} and {signal_col!r}")

        resampled_signal = self.resample_signal(df[signal_col].to_numpy())

        # Reconstruct a linearly-spaced timestamp array
        t_start = df[unix_col].iloc[0]
        t_end = df[unix_col].iloc[-1]
        new_timestamps = np.linspace(t_start, t_end, num=len(resampled_signal))

        return pd.DataFrame(
            {
                unix_col: new_timestamps.astype(np.int64),
                signal_col: resampled_signal,
            }
        )


def resample_timestamped_dataframe(
    df: pd.DataFrame,
    signal_col: str,
    target_fs: float,
    unix_col: str = "unix_ms",
    max_gap_ms: float = 500.0,
) -> pd.DataFrame:
    """Resample finite timestamped samples without fabricating across gaps.

    This is the deployment contract for Samsung ``PPG_CONTINUOUS``. Unlike
    ``Resampler.resample_dataframe`` it does not assume the first/last rows
    form a uniform grid and it preserves a physical-time origin. Duplicate
    timestamps keep the first sample, matching the Android boundary.
    """
    if df.empty:
        return df.loc[:, [unix_col, signal_col]].copy()
    if target_fs <= 0 or max_gap_ms <= 0:
        raise ValueError("target_fs and max_gap_ms must be positive")
    if unix_col not in df or signal_col not in df:
        raise KeyError(f"DataFrame must contain {unix_col!r} and {signal_col!r}")
    timestamps = pd.to_numeric(df[unix_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(df[signal_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(timestamps) & np.isfinite(values)
    timestamps = timestamps[finite]
    values = values[finite]
    if timestamps.size < 2:
        raise ValueError("timestamped signal requires at least two finite samples")
    if np.any(np.diff(timestamps) < 0.0):
        raise ValueError("timestamp_order_regression")
    unique_mask = np.concatenate(([True], np.diff(timestamps) > 0.0))
    timestamps = timestamps[unique_mask]
    values = values[unique_mask]
    if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing after deduplication")
    if np.any(np.diff(timestamps) > max_gap_ms):
        raise ValueError("timestamp_gap_exceeds_policy")
    step = 1_000.0 / float(target_fs)
    target = np.arange(timestamps[0], timestamps[-1] + 0.5 * step, step)
    target = target[target <= timestamps[-1] + 1.0e-6]
    if target.size < 2:
        raise ValueError("timestamp span is shorter than one target interval")
    return pd.DataFrame(
        {
            unix_col: np.rint(target).astype(np.int64),
            signal_col: np.interp(target, timestamps, values),
        }
    )


# ---------------------------------------------------------------------------
# Convenience factory functions (DRY)
# ---------------------------------------------------------------------------


def make_ecg_resampler(target_fs: float = TARGET_FS) -> Resampler:
    """Return a Resampler configured for ECG (130 Hz → target)."""
    return Resampler(source_fs=ECG_FS, target_fs=target_fs)


def make_ppg_resampler(target_fs: float = TARGET_FS) -> Resampler:
    """Return a Resampler configured for PPG (25 Hz → target)."""
    return Resampler(source_fs=PPG_FS, target_fs=target_fs)


if __name__ == "__main__":
    from PipelinePython.common.cli import main

    main(["sqa"])
