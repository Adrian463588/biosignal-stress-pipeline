"""Physiologically constrained training-only waveform augmentations."""

from __future__ import annotations

import numpy as np

from ..contracts import Augmentation

_MIN_SCALE = 0.8
_MAX_SCALE = 1.2
_MIN_SNR_DB = 20.0
_MAX_SHIFT_SAMPLES = 10


def _floating_1d(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal)
    if values.ndim != 1:
        raise ValueError("Augmentation transforms require a 1D waveform")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Augmentation transforms require a floating-point waveform")
    return values


def _floating_finite_1d(signal: np.ndarray) -> np.ndarray:
    values = _floating_1d(signal)
    if not np.isfinite(values).all():
        raise ValueError("Augmentation transforms require finite waveform values")
    return values


class AmplitudeScaling(Augmentation):
    """Scale amplitude within the canonical conservative range."""

    def __init__(
        self,
        scale_min: float = _MIN_SCALE,
        scale_max: float = _MAX_SCALE,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not (_MIN_SCALE <= scale_min <= scale_max <= _MAX_SCALE):
            raise ValueError("Canonical amplitude scaling must stay within 0.8-1.2")
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = rng or np.random.default_rng()

    def apply(self, signal: np.ndarray, label: int = 0) -> np.ndarray:
        augmented, _ = self.apply_with_metadata(signal, label)
        return augmented

    def apply_with_metadata(
        self, signal: np.ndarray, label: int = 0
    ) -> tuple[np.ndarray, dict[str, float]]:
        values = _floating_finite_1d(signal)
        scale = float(self.rng.uniform(self.scale_min, self.scale_max))
        augmented = (values * scale).astype(values.dtype, copy=False)
        return augmented, {"scale": scale}


class GaussianNoiseInjection(Augmentation):
    """Add low-amplitude Gaussian noise at no less than 20 dB SNR."""

    def __init__(
        self,
        min_snr_db: float = _MIN_SNR_DB,
        max_snr_db: float = 30.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if min_snr_db < _MIN_SNR_DB or max_snr_db < min_snr_db:
            raise ValueError("Canonical Gaussian noise requires SNR >= 20 dB")
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.rng = rng or np.random.default_rng()

    def apply(self, signal: np.ndarray, label: int = 0) -> np.ndarray:
        augmented, _ = self.apply_with_metadata(signal, label)
        return augmented

    def apply_with_metadata(
        self, signal: np.ndarray, label: int = 0
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        values = _floating_finite_1d(signal)
        sig_power = float(np.mean(values.astype(np.float64) ** 2))
        if sig_power < 1e-12:
            return np.copy(values), {"applied": False, "snr_db": self.min_snr_db}
        snr_db = float(self.rng.uniform(self.min_snr_db, self.max_snr_db))
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise = self.rng.normal(0.0, np.sqrt(noise_power), size=values.shape)
        augmented = (values + noise).astype(values.dtype, copy=False)
        return augmented, {"applied": True, "snr_db": snr_db}


class BaselineWanderInjection(Augmentation):
    """Inject sinusoidal baseline wander (0.05-0.4 Hz) simulating respiration / motion."""

    def __init__(
        self,
        fs: float = 100.0,
        min_freq: float = 0.05,
        max_freq: float = 0.4,
        max_amp_ratio: float = 0.15,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.fs = fs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.max_amp_ratio = max_amp_ratio
        self.rng = rng or np.random.default_rng()

    def apply(self, signal: np.ndarray, label: int = 0) -> np.ndarray:
        values = _floating_finite_1d(signal)
        n_samples = len(values)
        t = np.arange(n_samples) / self.fs
        freq = self.rng.uniform(self.min_freq, self.max_freq)
        phase = self.rng.uniform(0, 2 * np.pi)
        std_val = np.std(values)
        amp = self.rng.uniform(0.05, self.max_amp_ratio) * (
            std_val if std_val > 0 else 1.0
        )
        wander = amp * np.sin(2 * np.pi * freq * t + phase)
        return (values + wander).astype(values.dtype, copy=False)


class EdgeSafeCropShift(Augmentation):
    """Select a shifted contiguous window from explicit observed phase context."""

    def __init__(
        self,
        max_shift: int = _MAX_SHIFT_SAMPLES,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 1 <= max_shift <= _MAX_SHIFT_SAMPLES:
            raise ValueError("Canonical crop/shift must be between 1 and 10 samples")
        self.max_shift = max_shift
        self.rng = rng or np.random.default_rng()

    def apply(self, signal: np.ndarray, label: int = 0) -> np.ndarray:
        raise ValueError(
            "EdgeSafeCropShift requires explicit phase context; generic apply is unavailable"
        )

    def apply_with_metadata(
        self, signal: np.ndarray, label: int = 0
    ) -> tuple[np.ndarray, dict[str, int | bool]]:
        raise ValueError(
            "EdgeSafeCropShift requires explicit phase context; generic apply is unavailable"
        )

    def apply_with_phase_context(
        self,
        phase_signal: np.ndarray,
        *,
        observed_mask: np.ndarray,
        window_start: int,
        window_length: int,
        phase_start: int = 0,
        phase_end: int | None = None,
        shift_samples: int | None = None,
    ) -> tuple[np.ndarray, dict[str, int | bool]]:
        """Return one equal-length observed slice; never pad, wrap, or extrapolate."""
        values = _floating_1d(phase_signal)
        observed = np.asarray(observed_mask)
        if observed.dtype != np.bool_ or observed.shape != values.shape:
            raise ValueError(
                "observed_mask must be a boolean mask matching phase_signal"
            )

        end = len(values) if phase_end is None else phase_end
        bounds = (phase_start, end, window_start, window_length)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in bounds
        ):
            raise ValueError("Phase and window bounds must be integer sample indices")
        if not 0 <= phase_start < end <= len(values):
            raise ValueError("Invalid observed phase bounds")
        if window_length <= 0 or not phase_start <= window_start < end:
            raise ValueError("Invalid source window bounds")

        source_end = window_start + window_length
        if source_end > end:
            raise ValueError("Source window is outside observed phase bounds")
        if not self._slice_is_observed(values, observed, window_start, source_end):
            raise ValueError("Source window contains unobserved samples")

        if shift_samples is None:
            feasible = [
                shift
                for shift in range(-self.max_shift, self.max_shift + 1)
                if shift
                and self._slice_is_valid(
                    values,
                    observed,
                    window_start + shift,
                    source_end + shift,
                    phase_start,
                    end,
                )
            ]
            if not feasible:
                raise ValueError(
                    "No observed in-phase crop/shift candidate is available"
                )
            shift = int(self.rng.choice(feasible))
        else:
            if (
                isinstance(shift_samples, bool)
                or not isinstance(shift_samples, (int, np.integer))
                or shift_samples == 0
                or abs(shift_samples) > self.max_shift
            ):
                raise ValueError(
                    "shift_samples must be a non-zero integer within max_shift"
                )
            shift = int(shift_samples)

        target_start = window_start + shift
        target_end = source_end + shift
        if not phase_start <= target_start < target_end <= end:
            raise ValueError("Shifted window is outside observed phase bounds")
        if not self._slice_is_observed(values, observed, target_start, target_end):
            raise ValueError("Shifted window contains unobserved samples")

        return np.copy(values[target_start:target_end]), {
            "shift_samples": shift,
            "source_start": window_start,
            "target_start": target_start,
            "window_length": window_length,
            "phase_start": phase_start,
            "phase_end": end,
            "observed_only": True,
        }

    @staticmethod
    def _slice_is_observed(
        values: np.ndarray, observed: np.ndarray, start: int, end: int
    ) -> bool:
        return bool(observed[start:end].all() and np.isfinite(values[start:end]).all())

    @classmethod
    def _slice_is_valid(
        cls,
        values: np.ndarray,
        observed: np.ndarray,
        start: int,
        end: int,
        phase_start: int,
        phase_end: int,
    ) -> bool:
        return phase_start <= start < end <= phase_end and cls._slice_is_observed(
            values, observed, start, end
        )


class TemporalJitter(EdgeSafeCropShift):
    """Compatibility name; also requires explicit phase context."""


class TimeCutMix(Augmentation):
    """CutMix transform replacing a 10-25% slice with a segment from same-class candidate."""

    def __init__(
        self,
        min_cut_ratio: float = 0.10,
        max_cut_ratio: float = 0.25,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.min_cut_ratio = min_cut_ratio
        self.max_cut_ratio = max_cut_ratio
        self.rng = rng or np.random.default_rng()

    def apply(self, signal: np.ndarray, label: int = 0) -> np.ndarray:
        values = _floating_finite_1d(signal)
        n_samples = len(values)
        cut_len = int(
            n_samples * self.rng.uniform(self.min_cut_ratio, self.max_cut_ratio)
        )
        if cut_len <= 0 or cut_len >= n_samples // 2:
            return np.copy(values)
        src_start = self.rng.integers(0, n_samples - cut_len)
        dst_start = self.rng.integers(0, n_samples - cut_len)
        out = np.copy(values)
        out[dst_start : dst_start + cut_len] = values[src_start : src_start + cut_len]
        return out

    def apply_pair(self, signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
        """Replace segment in signal_a with corresponding segment from signal_b of same class."""
        values_a = _floating_finite_1d(signal_a)
        values_b = _floating_finite_1d(signal_b)
        if values_a.shape != values_b.shape:
            raise ValueError("TimeCutMix inputs must have the same shape")
        n_samples = len(values_a)
        cut_len = int(
            n_samples * self.rng.uniform(self.min_cut_ratio, self.max_cut_ratio)
        )
        if cut_len <= 0 or cut_len >= n_samples:
            return np.copy(values_a)
        start_idx = self.rng.integers(0, n_samples - cut_len)
        out = np.copy(values_a)
        out[start_idx : start_idx + cut_len] = values_b[start_idx : start_idx + cut_len]
        return out


class TimeReversal(Augmentation):
    """FORBIDDEN TRANSFORM: Raises ValueError per physiological rules."""

    def apply(self, signal: np.ndarray, label: int = 0) -> np.ndarray:
        raise ValueError(
            "TimeReversal is STRICTLY FORBIDDEN: PPG waveforms have asymmetric pulse morphology "
            "(rapid systolic upstroke, slow diastolic decay). Time reversal violates physiological validity."
        )


class AmplitudeInversion(Augmentation):
    """FORBIDDEN TRANSFORM: Raises ValueError per physiological rules."""

    def apply(self, signal: np.ndarray, label: int = 0) -> np.ndarray:
        raise ValueError(
            "AmplitudeInversion is STRICTLY FORBIDDEN: Inverting PPG pulse amplitude creates "
            "physiologically impossible signals."
        )
