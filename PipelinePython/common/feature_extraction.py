"""
feature_extraction.py
---------------------
Extracts physiological features from windowed segments of aligned signals
using NeuroKit2 as the primary HRV computation engine.

Two classes with distinct responsibilities (SRP):
  - WindowProcessor   → slices a continuous DataFrame into fixed-length windows
  - FeatureExtractor  → computes HRV and PPG features per window via NeuroKit2

DRY principle:
  - All feature helpers are private methods returning plain dicts.
  - _extract_window() composes them without repetition.
  - NeuroKit2 handles peak detection, cleaning, and HRV analysis internally,
    replacing the manual scipy-based implementation.

NeuroKit2 reference (2026 best practice):
  Makowski, D. et al. (2021). NeuroKit2: A Python Toolbox for Neurophysiological
  Signal Processing. Behavior Research Methods, 53(4), 1689–1696.
  https://doi.org/10.3758/s13428-020-01516-y
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Generator, Optional, Tuple

import neurokit2 as nk
import numpy as np
import pandas as pd

from .config import (
    MIN_PEAKS_PER_WINDOW,
    TARGET_FS,
    WINDOW_SECONDS,
    WINDOW_STEP_SECONDS,
)

logger = logging.getLogger(__name__)

# Suppress NeuroKit2 verbose warnings in production; set to DEBUG for research
warnings.filterwarnings("ignore", category=RuntimeWarning, module="neurokit2")


# ---------------------------------------------------------------------------
# WindowProcessor
# ---------------------------------------------------------------------------


class WindowProcessor:
    """
    Slices an aligned, resampled DataFrame into overlapping time windows.

    Args:
        window_sec: Window length in seconds.
        step_sec:   Step size between window starts (controls overlap).
        target_fs:  Sampling frequency of the input DataFrame (Hz).

    Yields:
        (window_index, window_DataFrame)
    """

    def __init__(
        self,
        window_sec: int = WINDOW_SECONDS,
        step_sec: int = WINDOW_STEP_SECONDS,
        target_fs: float = TARGET_FS,
    ) -> None:
        self._win_samples = int(window_sec * target_fs)
        self._step_samples = int(step_sec * target_fs)

    def generate_windows(
        self, df: pd.DataFrame
    ) -> Generator[tuple[int, pd.DataFrame], None, None]:
        """
        Yield successive windows from a DataFrame.

        Args:
            df: Aligned DataFrame (rows = samples, cols include signal channels).

        Yields:
            Tuple of (window_index, window_sub_dataframe).
        """
        n = len(df)
        idx = 0
        win_num = 0
        while idx + self._win_samples <= n:
            yield win_num, df.iloc[idx : idx + self._win_samples].copy()
            idx += self._step_samples
            win_num += 1


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """
    Computes physiological features from one data window using NeuroKit2.

    Feature groups extracted:
      Time-domain HRV (SDNN, RMSSD, pNN50, mean HR, …):
        Computed by nk.hrv_time() on the RR interval series.

      Frequency-domain HRV (LF, HF, LF/HF ratio, …):
        Computed by nk.hrv_frequency() using Welch's PSD method.

      Non-linear HRV (SD1, SD2, SampEn, …):
        Computed by nk.hrv_nonlinear() (Poincaré + entropy).

      PPG amplitude features (mean, std, peak amplitude statistics):
        Derived directly from the PPG Green channel signal array.

    Args:
        target_fs:  Sampling frequency (Hz).
        min_peaks:  Minimum peaks required to compute HRV features.
    """

    def __init__(
        self,
        target_fs: float = TARGET_FS,
        min_peaks: int = MIN_PEAKS_PER_WINDOW,
    ) -> None:
        self._fs = target_fs
        self._min_peaks = min_peaks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        window: pd.DataFrame,
        participant: str,
        condition: str,
        label: int,
        window_idx: int,
    ) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
        """
        Extract all features from a single window.

        Forces PPG Green as the primary signal for ML features.
        Calculates amplitude features for Green, IR, and Red.
        If ECG is present, calculates its HRV strictly as 'ecg_groundtruth_...'

        Returns:
            A tuple of (features_dict, ibi_dataframe).
            features_dict is a Dictionary of feature names → values, or None if the window
            has insufficient peaks to compute reliable PPG HRV.
            ibi_dataframe contains the continuous peak and IBI series.
        """
        # Determine which PPG channel is present in this window
        # Per-sensor training: each window contains exactly one channel
        ppg_channel = None
        for ch in ("ppg_green", "ppg_ir", "ppg_red"):
            if ch in window.columns and window[ch].notna().any():
                ppg_channel = ch
                break
        if ppg_channel is None:
            return None, None  # No PPG channel found

        primary_signal = window[ppg_channel].to_numpy()
        signal_source = "ppg"

        # --- HRV features via NeuroKit2 (from PPG) ---
        hrv_features = self._compute_hrv_features(primary_signal, signal_source)
        if hrv_features is None:
            logger.debug(
                "Window %d [%s/%s]: insufficient PPG peaks or NeuroKit2 error; skipping.",
                window_idx,
                participant,
                condition,
            )
            return None, None

        # --- Assemble base metadata ---
        features: Dict = {
            "participant": participant,
            "condition": condition,
            "label": label,
            "window_idx": window_idx,
            "signal_source": signal_source,
            "window_start_ms": int(window["unix_ms"].iloc[0]),
            "window_end_ms": int(window["unix_ms"].iloc[-1]),
        }
        features.update(hrv_features)

        # --- PPG amplitude features for all 3 channels ---
        for ch in ["ppg_green", "ppg_ir", "ppg_red"]:
            if ch in window.columns and window[ch].notna().any():
                channel_sig = window[ch].to_numpy()
                amp_feats = self._compute_ppg_amplitude_features(
                    channel_sig, prefix=f"{ch}_"
                )
                features.update(amp_feats)

        # --- ECG Ground Truth Validation ---
        has_ecg = "ecg_mv" in window.columns and window["ecg_mv"].notna().any()
        if has_ecg:
            ecg_signal = window["ecg_mv"].fillna(0).to_numpy()
            ecg_features = self._compute_hrv_features(ecg_signal, "ecg")
            if ecg_features is not None:
                # Prefix with ecg_groundtruth_ to separate from primary features
                ecg_gt = {f"ecg_groundtruth_{k}": v for k, v in ecg_features.items()}
                features.update(ecg_gt)

        # --- Continuous IBI Series ---
        ibi_df = self._extract_ibi_series(window, window_idx)

        return features, ibi_df

    def _extract_ibi_series(
        self, window: pd.DataFrame, window_idx: int
    ) -> Optional[pd.DataFrame]:
        ppg_channel = None
        for ch in ("ppg_green", "ppg_ir", "ppg_red"):
            if ch in window.columns and window[ch].notna().any():
                ppg_channel = ch
                break
        if ppg_channel is None:
            return None

        ppg_signal = window[ppg_channel].to_numpy()
        try:
            ppg_peaks = self._detect_peaks_nk(ppg_signal, "ppg")
        except Exception:
            return None

        if ppg_peaks is None or len(ppg_peaks) < 2:
            return None

        # Get timestamps for PPG peaks
        unix_ms = window["unix_ms"].to_numpy()
        ppg_peak_times = unix_ms[ppg_peaks]

        # Calculate IBI
        ppg_ibi = np.diff(ppg_peak_times)

        # The timestamp for an IBI is usually the time of the second peak
        ibi_times = ppg_peak_times[1:]

        df_ibi = pd.DataFrame(
            {
                "window_idx": window_idx,
                "peak_timestamp_ms": ibi_times,
                "ppg_ibi_ms": ppg_ibi,
            }
        )

        # Handle ECG Ground Truth
        has_ecg = "ecg_mv" in window.columns and window["ecg_mv"].notna().any()
        if has_ecg:
            ecg_signal = window["ecg_mv"].fillna(0).to_numpy()
            try:
                ecg_peaks = self._detect_peaks_nk(ecg_signal, "ecg")
            except Exception:
                ecg_peaks = None

            if ecg_peaks is not None and len(ecg_peaks) >= 2:
                ecg_peak_times = unix_ms[ecg_peaks]
                ecg_ibi = np.diff(ecg_peak_times)
                ecg_ibi_times = ecg_peak_times[1:]

                # Match ECG IBI to PPG IBI using nearest neighbor
                matched_ecg_ibi = np.full(len(ibi_times), np.nan)
                ibi_error = np.full(len(ibi_times), np.nan)

                for i, p_time in enumerate(ibi_times):
                    # Find closest ECG peak time
                    diffs = np.abs(ecg_ibi_times - p_time)
                    min_idx = np.argmin(diffs)

                    # If the closest ECG peak is within 500ms (PTT + tolerance)
                    if diffs[min_idx] < 500:
                        matched_ecg_ibi[i] = ecg_ibi[min_idx]
                        ibi_error[i] = np.abs(ppg_ibi[i] - ecg_ibi[min_idx])

                df_ibi["ecg_ibi_ms"] = matched_ecg_ibi
                df_ibi["ibi_error_ms"] = ibi_error
                # valid if diff < 50ms, meaning PPG tracked heart rate correctly compared to ECG
                df_ibi["is_valid"] = df_ibi["ibi_error_ms"] < 50.0
            else:
                df_ibi["ecg_ibi_ms"] = np.nan
                df_ibi["ibi_error_ms"] = np.nan
                df_ibi["is_valid"] = False
        else:
            df_ibi["ecg_ibi_ms"] = np.nan
            df_ibi["ibi_error_ms"] = np.nan
            df_ibi["is_valid"] = False

        return df_ibi

    # ------------------------------------------------------------------
    # Private: HRV computation via NeuroKit2
    # ------------------------------------------------------------------

    def _compute_hrv_features(
        self, signal: np.ndarray, signal_type: str
    ) -> Optional[Dict]:
        """
        Detect peaks → compute RR intervals → extract full HRV suite via NK2.

        Strategy:
          - ECG:  nk.ecg_process()  handles R-peak detection + cleaning.
          - PPG:  nk.ppg_process()  handles systolic peak detection + cleaning.
          - HRV features are computed from the cleaned RR interval series.

        Returns None when NeuroKit2 fails (noisy signal, too few peaks, etc.).
        """
        try:
            rpeaks = self._detect_peaks_nk(signal, signal_type)
        except Exception as exc:
            logger.debug("Peak detection failed (%s): %s", signal_type, exc)
            return None

        if rpeaks is None or len(rpeaks) < self._min_peaks:
            return None

        # Compute RR intervals in milliseconds
        rr_ms = np.diff(rpeaks) / self._fs * 1000.0
        if len(rr_ms) < 2:
            return None

        # --- Build a synthetic RR-peak list compatible with nk.hrv() ---
        # nk.hrv() accepts a dict with key "ECG_R_Peaks" (sample indices)
        peaks_info = {"ECG_R_Peaks": rpeaks}

        features: Dict = {
            "num_peaks": len(rpeaks),
        }

        # 1. Time-domain HRV
        try:
            hrv_time = nk.hrv_time(peaks_info, sampling_rate=self._fs, show=False)
            features.update(self._flatten_nk_df(hrv_time, prefix="hrv_"))
        except Exception as exc:
            logger.debug("hrv_time failed: %s", exc)
            # Fall back to manual time-domain computation
            features.update(self._manual_time_domain(rr_ms))

        # 2. Frequency-domain HRV (requires at least 2 full breath cycles → ~30s)
        try:
            hrv_freq = nk.hrv_frequency(
                peaks_info,
                sampling_rate=self._fs,
                show=False,
                psd_method="welch",
            )
            features.update(self._flatten_nk_df(hrv_freq, prefix="hrv_"))
        except Exception as exc:
            logger.debug("hrv_frequency failed: %s", exc)

        # 3. Non-linear HRV (Poincaré SD1/SD2, SampEn, DFA)
        try:
            hrv_nl = nk.hrv_nonlinear(peaks_info, sampling_rate=self._fs, show=False)
            features.update(self._flatten_nk_df(hrv_nl, prefix="hrv_"))
        except Exception as exc:
            logger.debug("hrv_nonlinear failed: %s", exc)

        # --- Convenience aliases for downstream pipelines ---
        # Ensure canonical names exist even when NeuroKit2 uses different keys
        features.setdefault("mean_rr_ms", float(np.mean(rr_ms)))
        features.setdefault("std_rr_ms", float(np.std(rr_ms, ddof=1)))
        features.setdefault(
            "mean_hr_bpm", float(60_000 / np.mean(rr_ms)) if np.mean(rr_ms) > 0 else 0.0
        )
        features.setdefault("rmssd_ms", self._compute_rmssd(rr_ms))

        return features

    def _detect_peaks_nk(
        self, signal: np.ndarray, signal_type: str
    ) -> Optional[np.ndarray]:
        """
        Use NeuroKit2's pipeline to detect physiological peaks.

        ECG → nk.ecg_peaks()  (Pan-Tompkins QRS detector, default 2021)
        PPG → nk.ppg_findpeaks()  (Elgendi et al. 2013, optimized for wrist PPG)

        Returns:
            1-D integer array of peak sample indices, or None on failure.
        """
        if signal_type == "ecg":
            _, info = nk.ecg_peaks(
                signal,
                sampling_rate=self._fs,
                method="pantompkins1985",
                correct_artifacts=True,
            )
            return np.array(info["ECG_R_Peaks"], dtype=int)

        # PPG — clean first with NK2's dedicated filter then detect peaks
        ppg_cleaned = nk.ppg_clean(
            signal,
            sampling_rate=self._fs,
            method="elgendi",
        )
        info = nk.ppg_findpeaks(
            ppg_cleaned,
            sampling_rate=self._fs,
            method="elgendi",
        )
        return np.array(info.get("PPG_Peaks", []), dtype=int)

    # ------------------------------------------------------------------
    # Private: PPG amplitude features (independent of HRV)
    # ------------------------------------------------------------------

    def _compute_ppg_amplitude_features(
        self, ppg_signal: np.ndarray, prefix: str = "ppg_"
    ) -> Dict:
        """
        Amplitude statistics from the PPG channel.

        These complement HRV features by capturing perfusion-related changes
        (pulse amplitude decreases under stress / vasoconstriction).
        """
        try:
            ppg_cleaned = nk.ppg_clean(
                ppg_signal,
                sampling_rate=self._fs,
                method="elgendi",
            )
            info = nk.ppg_findpeaks(
                ppg_cleaned,
                sampling_rate=self._fs,
                method="elgendi",
            )
            ppg_peaks = np.array(info.get("PPG_Peaks", []), dtype=int)
            peak_amps = (
                ppg_cleaned[ppg_peaks] if len(ppg_peaks) > 0 else np.array([np.nan])
            )
        except Exception:
            ppg_cleaned = ppg_signal
            peak_amps = np.array([np.nan])

        return {
            f"{prefix}mean": float(np.nanmean(ppg_cleaned)),
            f"{prefix}std": float(
                np.nanstd(ppg_cleaned, ddof=1) if len(ppg_cleaned) > 1 else 0.0
            ),
            f"{prefix}peak_amp_mean": float(np.nanmean(peak_amps)),
            f"{prefix}peak_amp_std": float(
                np.nanstd(peak_amps, ddof=1) if len(peak_amps) > 1 else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Private: fallback manual time-domain (when NK2 unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def _manual_time_domain(rr_ms: np.ndarray) -> Dict:
        """Compute basic time-domain HRV manually (DRY fallback for NK2)."""
        successive_diffs = np.diff(rr_ms)
        nn50 = int(np.sum(np.abs(successive_diffs) > 50))
        pnn50 = (
            (nn50 / len(successive_diffs)) * 100.0 if len(successive_diffs) > 0 else 0.0
        )
        mean_rr = float(np.mean(rr_ms))
        return {
            "mean_rr_ms": mean_rr,
            "std_rr_ms": float(np.std(rr_ms, ddof=1)),
            "rmssd_ms": float(np.sqrt(np.mean(successive_diffs**2))),
            "mean_hr_bpm": float(60_000 / mean_rr) if mean_rr > 0 else 0.0,
            "hrv_HRV_NN50": float(nn50),
            "hrv_HRV_pNN50": float(pnn50),
            "hrv_HRV_CVNN": float(np.std(rr_ms, ddof=1) / mean_rr)
            if mean_rr > 0
            else 0.0,
        }

    @staticmethod
    def _compute_rmssd(rr_ms: np.ndarray) -> float:
        """Root mean square of successive RR differences."""
        if len(rr_ms) < 2:
            return 0.0
        return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))

    # ------------------------------------------------------------------
    # Private: flatten NeuroKit2 result DataFrame → flat dict (DRY)
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_nk_df(df: pd.DataFrame, prefix: str = "") -> Dict:
        """
        Convert a single-row NeuroKit2 HRV result DataFrame to a flat dict.

        NeuroKit2's hrv_* functions return a 1-row DataFrame with one column
        per metric. This helper flattens it for storage in a feature dict.

        NaN values are kept so downstream SQI can detect computation failures.
        """
        if df is None or df.empty:
            return {}
        row = df.iloc[0].to_dict()
        return {
            f"{prefix}{k}": (None if pd.isna(v) else float(v)) for k, v in row.items()
        }


if __name__ == "__main__":
    from PipelinePython.common.cli import main

    main(["features"])
