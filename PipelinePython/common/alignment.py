"""
alignment.py
------------
Aligns ECG and PPG signals to a common timeline using their Unix timestamps.
Also performs ECG timestamp correction (offset and linear drift) based on preliminary peak analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal

from .config import TARGET_FS

logger = logging.getLogger(__name__)


class EcgClockCorrector:
    """
    Corrects ECG timestamps relative to PPG using coarse alignment and cross-correlation
    of physiological peaks/envelopes.
    """

    def __init__(
        self,
        participant_id: str,
        session_id: str = "default",
        output_dir: str | Path = "reports/alignment",
    ) -> None:
        self.participant_id = participant_id
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _detect_preliminary_peaks(
        self, ts_ms: np.ndarray, sig: np.ndarray, fs: float
    ) -> np.ndarray:
        distance = int(0.5 * fs)  # HR < 120 bpm
        peaks, _ = signal.find_peaks(
            sig, distance=distance, prominence=np.std(sig) * 0.5
        )
        return ts_ms[peaks]

    def _compute_alignment_score(self, ecg_ts: np.ndarray, ppg_ts: np.ndarray) -> float:
        if len(ecg_ts) == 0 or len(ppg_ts) == 0:
            return 0.0

        matched = 0
        tolerance_ms = 300

        j = 0
        for e_t in ecg_ts:
            while j < len(ppg_ts) and ppg_ts[j] < e_t - tolerance_ms:
                j += 1
            if j < len(ppg_ts) and abs(ppg_ts[j] - e_t) <= tolerance_ms:
                matched += 1
                j += 1

        return matched / max(len(ecg_ts), 1)

    def _estimate_drift_and_offset(
        self, ecg_df: pd.DataFrame, ppg_df: pd.DataFrame
    ) -> tuple[float, float, float, float]:
        ecg_start = ecg_df["unix_ms"].iloc[0]
        ppg_start = ppg_df["unix_ms"].iloc[0]

        coarse_offset = (
            ppg_start - ecg_start
            if (ecg_start < ppg_start - 10000 or ecg_start > ppg_start + 10000)
            else 0
        )
        temp_ecg_ts = ecg_df["unix_ms"].values + coarse_offset

        ecg_sig = ecg_df["ecg_mv"].values
        ppg_sig = ppg_df["ppg_raw"].values

        fs_ecg = 1000.0 / np.median(np.diff(temp_ecg_ts))
        fs_ppg = 1000.0 / np.median(np.diff(ppg_df["unix_ms"].values))

        ecg_peaks = self._detect_preliminary_peaks(temp_ecg_ts, ecg_sig, fs_ecg)
        ppg_peaks = self._detect_preliminary_peaks(
            ppg_df["unix_ms"].values, ppg_sig, fs_ppg
        )

        pre_score = self._compute_alignment_score(ecg_peaks, ppg_peaks)

        min_ts = min(temp_ecg_ts[0], ppg_df["unix_ms"].iloc[0])
        max_ts = max(temp_ecg_ts[-1], ppg_df["unix_ms"].iloc[-1])

        grid_step = 10
        grid_len = int((max_ts - min_ts) / grid_step) + 1

        ecg_train = np.zeros(grid_len)
        ppg_train = np.zeros(grid_len)

        for p in ecg_peaks:
            idx = int((p - min_ts) / grid_step)
            if 0 <= idx < grid_len:
                ecg_train[idx] = 1.0

        for p in ppg_peaks:
            idx = int((p - min_ts) / grid_step)
            if 0 <= idx < grid_len:
                ppg_train[idx] = 1.0

        window = signal.windows.gaussian(51, std=10)
        ecg_train = signal.convolve(ecg_train, window, mode="same")
        ppg_train = signal.convolve(ppg_train, window, mode="same")

        xcorr = signal.correlate(ppg_train, ecg_train, mode="full")
        lags = signal.correlation_lags(len(ppg_train), len(ecg_train), mode="full")

        if len(xcorr) > 0:
            best_lag = lags[np.argmax(xcorr)]
            fine_offset_ms = best_lag * grid_step
        else:
            fine_offset_ms = 0

        if abs(fine_offset_ms) > 5000:
            fine_offset_ms = 0

        total_offset_ms = coarse_offset + fine_offset_ms
        drift_slope = 0.0

        post_ecg_peaks = ecg_peaks + fine_offset_ms
        post_score = self._compute_alignment_score(post_ecg_peaks, ppg_peaks)

        if post_score < pre_score and post_score < 0.1:
            total_offset_ms = coarse_offset
            post_score = pre_score

        return (
            float(total_offset_ms),
            float(drift_slope),
            float(pre_score),
            float(post_score),
        )

    def correct(
        self, ecg_df: pd.DataFrame, ppg_frames: list[pd.DataFrame]
    ) -> pd.DataFrame:
        if not ppg_frames or ecg_df.empty:
            return ecg_df

        ref_ppg = ppg_frames[0]
        offset_ms, drift_slope, pre_score, post_score = self._estimate_drift_and_offset(
            ecg_df, ref_ppg
        )

        corrected_ecg = ecg_df.copy()
        ts_start = corrected_ecg["unix_ms"].iloc[0]
        corrected_ecg["unix_ms"] = (
            corrected_ecg["unix_ms"]
            + offset_ms
            + drift_slope * (corrected_ecg["unix_ms"] - ts_start)
        )

        report = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "initial_offset_ms": offset_ms,
            "drift_slope": drift_slope,
            "correction_method": "cross_correlation_peaks",
            "pre_alignment_score": pre_score,
            "post_alignment_score": post_score,
        }

        report_path = self.output_dir / f"ecg_correction_{self.participant_id}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            "ECG correction %s: offset=%.1fms, drift=%.6f, score=%.2f->%.2f",
            self.participant_id,
            offset_ms,
            drift_slope,
            pre_score,
            post_score,
        )
        return corrected_ecg


class DataAligner:
    """
    Merges ECG and multiple PPG DataFrames into one unified DataFrame
    on a common, regularly-spaced timestamp grid.
    """

    def __init__(
        self,
        target_fs: float = TARGET_FS,
        participant_id: str = "unknown",
        session_id: str = "default",
    ) -> None:
        self._step_ms = int(round(1000 / target_fs))
        self.corrector = EcgClockCorrector(participant_id, session_id)

    def align(
        self,
        ecg: pd.DataFrame,
        ppg_green: pd.DataFrame,
        ppg_ir: pd.DataFrame | None = None,
        ppg_red: pd.DataFrame | None = None,
    ) -> pd.DataFrame:

        ppg_frames = [
            f for f in (ppg_green, ppg_ir, ppg_red) if f is not None and not f.empty
        ]
        if ppg_frames and ecg is not None and not ecg.empty:
            ecg = self.corrector.correct(ecg, ppg_frames)

        signals: dict[str, tuple[pd.DataFrame, str]] = {
            "ppg_green": (ppg_green, "ppg_raw"),
        }
        if ecg is not None and not ecg.empty:
            signals["ecg_mv"] = (ecg, "ecg_mv")
        if ppg_ir is not None:
            signals["ppg_ir"] = (ppg_ir, "ppg_raw")
        if ppg_red is not None:
            signals["ppg_red"] = (ppg_red, "ppg_raw")

        t_start, t_end = self._overlap_range(signals)
        if t_start >= t_end:
            logger.warning(
                "No temporal overlap found between signals; returning empty DataFrame."
            )
            return pd.DataFrame()

        grid = np.arange(t_start, t_end, self._step_ms, dtype=np.int64)
        merged = pd.DataFrame({"unix_ms": grid})

        for col_name, (df, raw_col) in signals.items():
            series = self._interpolate_to_grid(
                df["unix_ms"].to_numpy(), df[raw_col].to_numpy(), grid
            )
            merged[col_name] = series

        signal_cols = [c for c in merged.columns if c != "unix_ms"]
        merged = merged.dropna(subset=signal_cols, how="all").reset_index(drop=True)

        n_rows = len(merged)
        has_ecg = "ecg_mv" in merged.columns and merged["ecg_mv"].notna().any()
        logger.debug(
            "Aligned DataFrame: %d rows, %.1f seconds | ECG present: %s",
            n_rows,
            n_rows / (1000 / self._step_ms),
            has_ecg,
        )
        return merged

    @staticmethod
    def _overlap_range(signals: dict[str, tuple[pd.DataFrame, str]]) -> tuple[int, int]:
        ppg_keys = [k for k in signals if k != "ecg_mv"]
        ecg_entry = signals.get("ecg_mv")

        ppg_starts, ppg_ends = [], []
        for key in ppg_keys:
            df, _ = signals[key]
            if df is not None and not df.empty:
                ppg_starts.append(int(df["unix_ms"].iloc[0]))
                ppg_ends.append(int(df["unix_ms"].iloc[-1]))

        if not ppg_starts:
            return 0, 0

        session_start = max(ppg_starts)
        session_end = min(ppg_ends)

        if session_start >= session_end:
            return 0, 0

        if ecg_entry is not None:
            ecg_df, _ = ecg_entry
            if ecg_df is not None and not ecg_df.empty:
                ecg_start = int(ecg_df["unix_ms"].iloc[0])
                ecg_end = int(ecg_df["unix_ms"].iloc[-1])
                if ecg_end > session_start and ecg_start < session_end:
                    session_start = max(session_start, ecg_start)
                    session_end = min(session_end, ecg_end)

        return session_start, session_end

    @staticmethod
    def _interpolate_to_grid(
        timestamps: np.ndarray, values: np.ndarray, grid: np.ndarray
    ) -> np.ndarray:
        interpolated = np.interp(grid, timestamps, values)
        out_of_range = (grid < timestamps[0]) | (grid > timestamps[-1])
        interpolated[out_of_range] = np.nan
        return interpolated
