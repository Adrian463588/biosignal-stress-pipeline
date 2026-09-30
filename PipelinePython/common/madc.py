"""Motion annotation for PPG/ECG preprocessing.

MADC is intentionally annotation-only in the canonical pipeline.  It never
rewrites a biosignal, interpolates motion, applies an adaptive filter, or
decides the Gold quality tier.  This keeps motion provenance separate from
signal reconstruction and avoids comparing uncalibrated accelerometer units
to a fixed physical threshold.

References:
    Wearable PPG motion-artifact review:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7612541/
    SciPy rolling/robust numerical building blocks:
    https://docs.scipy.org/doc/scipy/reference/
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MotionArtifactDetector:
    """Detect accelerometer-related motion without modifying the signal."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.window_seconds = max(0.1, float(self.config.get("window_seconds", 2.0)))
        self.robust_z_threshold = float(self.config.get("robust_z_threshold", 6.0))
        self.heavy_motion_threshold = float(
            self.config.get("heavy_motion_threshold", 0.8)
        )
        self.reject_ratio_threshold = float(
            self.config.get("reject_ratio_threshold", 0.3)
        )
        self.light_motion_threshold = float(
            self.config.get("light_motion_threshold", 0.2)
        )
        self.sampling_rate_hz = float(
            self.config.get("sampling_rate_hz", self.config.get("fs", 100.0))
        )
        self.accelerometer_scale_status = str(
            self.config.get("accelerometer_scale_status", "uncalibrated")
        )

    @staticmethod
    def _as_vector(name: str, values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        if vector.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        return vector

    @classmethod
    def _validate_lengths(cls, **arrays: np.ndarray) -> dict[str, np.ndarray]:
        vectors = {name: cls._as_vector(name, value) for name, value in arrays.items()}
        lengths = {len(value) for value in vectors.values()}
        if len(lengths) > 1:
            raise ValueError("signal and accelerometer arrays must have equal length")
        return vectors

    @staticmethod
    def compute_accel_magnitude(
        ax: np.ndarray, ay: np.ndarray, az: np.ndarray
    ) -> np.ndarray:
        """Return the Euclidean accelerometer magnitude."""
        vectors = MotionArtifactDetector._validate_lengths(ax=ax, ay=ay, az=az)
        return np.sqrt(vectors["ax"] ** 2 + vectors["ay"] ** 2 + vectors["az"] ** 2)

    def _window_samples(self, sampling_rate_hz: float | None = None) -> int:
        """Resolve the physical MADC window at the observed sampling rate."""

        rate = (
            self.sampling_rate_hz
            if sampling_rate_hz is None
            else float(sampling_rate_hz)
        )
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("sampling_rate_hz must be finite and positive")
        return max(3, int(round(self.window_seconds * rate)))

    def compute_rolling_energy(
        self,
        magnitude: np.ndarray,
        sampling_rate_hz: float | None = None,
    ) -> np.ndarray:
        """Return local magnitude variance over a sampling-rate-aware window."""
        values = self._as_vector("magnitude", magnitude)
        if values.size == 0:
            return values.copy()
        window_samples = self._window_samples(sampling_rate_hz)
        return (
            pd.Series(values)
            .rolling(
                window=min(window_samples, values.size),
                center=True,
                min_periods=1,
            )
            .var()
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    def _robust_motion_score(self, energy: np.ndarray) -> np.ndarray:
        """Map per-recording energy to [0, 1] using median/MAD, not units."""
        values = self._as_vector("energy", energy)
        if values.size == 0:
            return values.copy()
        finite = np.isfinite(values)
        if not finite.any():
            return np.ones(values.size, dtype=float)
        center = float(np.median(values[finite]))
        mad = float(np.median(np.abs(values[finite] - center)))
        scale = max(1.4826 * mad, np.finfo(float).eps)
        robust_z = np.abs(values - center) / scale
        score = robust_z / max(self.robust_z_threshold, np.finfo(float).eps)
        return np.clip(np.nan_to_num(score, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)

    def _detect_with_provenance(
        self,
        ppg_signal: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
        sampling_rate_hz: float | None = None,
    ) -> dict[str, Any]:
        vectors = self._validate_lengths(signal=ppg_signal, ax=ax, ay=ay, az=az)
        signal = vectors["signal"]
        if not all(np.isfinite(vectors[name]).all() for name in ("ax", "ay", "az")):
            return self._unavailable_result(
                signal,
                reason="accelerometer_nonfinite",
            )
        magnitude = self.compute_accel_magnitude(
            vectors["ax"], vectors["ay"], vectors["az"]
        )
        window_samples = self._window_samples(sampling_rate_hz)
        energy = self.compute_rolling_energy(magnitude, sampling_rate_hz)
        motion_scores = self._robust_motion_score(energy)
        artifact_mask = (
            np.isfinite(motion_scores) & (motion_scores >= self.light_motion_threshold)
        ) | ~np.isfinite(signal)
        heavy_mask = np.isfinite(motion_scores) & (
            motion_scores >= self.heavy_motion_threshold
        )
        heavy_ratio = float(np.mean(heavy_mask) if motion_scores.size else 0.0)
        if not signal.size:
            action = "PASS_EMPTY"
        elif heavy_ratio > self.reject_ratio_threshold:
            action = "MOTION_HEAVY_ANNOTATED"
        elif np.any(artifact_mask):
            action = "MOTION_ANNOTATED"
        else:
            action = "PASS_ANNOTATION_ONLY"

        # Critical invariant: MADC never changes signal values.
        unchanged_signal = signal.copy()
        return {
            "cleaned_signal": unchanged_signal,
            "motion_score_array": motion_scores,
            "motion_score": float(np.mean(motion_scores))
            if motion_scores.size
            else 0.0,
            "motion_available": True,
            "artifact_mask": artifact_mask,
            "motion_heavy_mask": heavy_mask,
            "heavy_motion_ratio": heavy_ratio,
            "madc_window_seconds": self.window_seconds,
            "madc_window_samples": window_samples,
            "interpolated_mask": np.zeros(signal.size, dtype=bool),
            "madc_action": action,
            "motion_annotation": action.lower(),
            "accelerometer_scale_status": self.accelerometer_scale_status,
            "motion_method": "time_windowed_rolling_variance_robust_mad",
            "provenance": {
                "interpolation_method": None,
                "interpolated_sample_count": 0,
                "interpolated_runs": [],
                "long_gap_reconstructed": False,
                "adaptive_filter_applied": False,
                "signal_modified": False,
                "accelerometer_scale_status": self.accelerometer_scale_status,
            },
        }

    @staticmethod
    def _unavailable_result(signal: np.ndarray, *, reason: str) -> dict[str, Any]:
        """Return an explicit unknown-motion result without zero-filling axes."""

        values = np.asarray(signal, dtype=float)
        return {
            "cleaned_signal": values.copy(),
            "motion_score_array": np.full(values.size, np.nan, dtype=float),
            "motion_score": np.nan,
            "motion_available": False,
            "artifact_mask": ~np.isfinite(values),
            "motion_heavy_mask": np.zeros(values.size, dtype=bool),
            "heavy_motion_ratio": np.nan,
            "madc_window_seconds": np.nan,
            "madc_window_samples": np.nan,
            "interpolated_mask": np.zeros(values.size, dtype=bool),
            "madc_action": "MOTION_UNAVAILABLE",
            "motion_annotation": reason,
            "accelerometer_scale_status": "unavailable",
            "motion_method": "unavailable",
            "provenance": {
                "interpolation_method": None,
                "interpolated_sample_count": 0,
                "interpolated_runs": [],
                "long_gap_reconstructed": False,
                "adaptive_filter_applied": False,
                "signal_modified": False,
                "accelerometer_scale_status": "unavailable",
            },
        }

    def detect_artifacts(
        self,
        ppg_signal: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Compatibility API returning an unchanged signal and annotation."""
        result = self._detect_with_provenance(ppg_signal, ax, ay, az)
        return (
            result["cleaned_signal"],
            result["motion_score_array"],
            result["madc_action"],
        )

    def detect_and_correct(
        self,
        signal: np.ndarray,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        fs: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility API; correction is intentionally a no-op."""
        result = self._detect_with_provenance(
            signal,
            acc_x,
            acc_y,
            acc_z,
            sampling_rate_hz=fs,
        )
        return result["cleaned_signal"], result["artifact_mask"]

    def process_window(
        self,
        window_df: pd.DataFrame,
        sampling_rate_hz: float | None = None,
    ) -> dict[str, Any]:
        """Annotate a window while accepting legacy accelerometer aliases."""
        signal_column = next(
            (
                name
                for name in ("raw_value", "clean_value", "ppg_raw", "ppg", "signal")
                if name in window_df
            ),
            None,
        )
        signal = (
            window_df[signal_column].to_numpy(dtype=float)
            if signal_column is not None
            else np.array([], dtype=float)
        )
        provenance_values = (
            window_df["accelerometer_provenance_status"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            if "accelerometer_provenance_status" in window_df
            else np.array([], dtype=str)
        )
        if len(provenance_values) != 1 or provenance_values[0] != "VERIFIED":
            return self._unavailable_result(
                signal,
                reason="accelerometer_provenance_unverified",
            )

        def axis(*names: str) -> np.ndarray | None:
            for name in names:
                if name in window_df:
                    return window_df[name].to_numpy(dtype=float)
            return None

        axes = (
            axis("acc_x", "accel_x"),
            axis("acc_y", "accel_y"),
            axis("acc_z", "accel_z"),
        )
        if signal.size == 0 or any(value is None for value in axes):
            return self._unavailable_result(signal, reason="no_accelerometer")
        if not all(np.isfinite(value).all() for value in axes):
            return self._unavailable_result(
                signal,
                reason="accelerometer_nonfinite",
            )
        return self._detect_with_provenance(
            signal,
            axes[0],
            axes[1],
            axes[2],
            sampling_rate_hz=sampling_rate_hz,
        )
