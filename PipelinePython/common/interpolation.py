"""interpolation.py
----------------
Cubic spline interpolation engine for inter-beat interval (IBI) series and signal resampling.
Follows SOLID design principles and implements InterpolationEngine contract.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from .contracts import InterpolationEngine

logger = logging.getLogger(__name__)


class CubicSplineInterpolationEngine(InterpolationEngine):
    """Cubic spline interpolation engine for missing/corrupted IBIs and irregular time series."""

    def interpolate_ibi(
        self, ibi_series: np.ndarray, missing_indices: np.ndarray
    ) -> np.ndarray:
        """Interpolate missing or ectopic IBI values in a 1D sequence using cubic spline.

        Args:
            ibi_series: 1D numpy array of IBI values in milliseconds.
            missing_indices: Boolean array or integer array of indices to replace.

        Returns:
            Cleaned 1D numpy array with missing IBIs interpolated.
        """
        ibi_clean = np.copy(ibi_series).astype(np.float64)
        if len(ibi_clean) < 4:
            # Need at least 4 points for reliable cubic spline
            return ibi_clean

        if missing_indices.dtype == bool:
            mask_missing = missing_indices
        else:
            mask_missing = np.zeros(len(ibi_series), dtype=bool)
            mask_missing[missing_indices] = True

        valid_idx = np.where(~mask_missing)[0]
        missing_idx = np.where(mask_missing)[0]

        if len(missing_idx) == 0:
            return ibi_clean

        if len(valid_idx) < 2:
            return ibi_clean

        first_valid = valid_idx[0]
        last_valid = valid_idx[-1]

        gaps = []
        current_gap = []
        for idx in missing_idx:
            # never interpolate across invalid boundaries (extrapolation)
            if idx < first_valid or idx > last_valid:
                continue
            if not current_gap:
                current_gap.append(idx)
            elif idx == current_gap[-1] + 1:
                current_gap.append(idx)
            else:
                gaps.append(current_gap)
                current_gap = [idx]
        if current_gap:
            gaps.append(current_gap)

        for gap in gaps:
            if len(gap) <= 2:
                # Linear for short gaps
                left_idx = gap[0] - 1
                right_idx = gap[-1] + 1
                while mask_missing[left_idx] and left_idx > first_valid:
                    left_idx -= 1
                while mask_missing[right_idx] and right_idx < last_valid:
                    right_idx += 1
                x = [left_idx, right_idx]
                y = [ibi_clean[left_idx], ibi_clean[right_idx]]
                f = interp1d(x, y, kind="linear")
                ibi_clean[gap] = f(gap)
            else:
                # Cubic spline for long gaps
                if len(valid_idx) >= 4:
                    cs = CubicSpline(
                        valid_idx,
                        ibi_clean[valid_idx],
                        bc_type="not-a-knot",
                        extrapolate=False,
                    )
                    ibi_clean[gap] = cs(gap)
                else:
                    # Fallback to linear if not enough points for cubic spline
                    left_idx = gap[0] - 1
                    right_idx = gap[-1] + 1
                    while mask_missing[left_idx] and left_idx > first_valid:
                        left_idx -= 1
                    while mask_missing[right_idx] and right_idx < last_valid:
                        right_idx += 1
                    x = [left_idx, right_idx]
                    y = [ibi_clean[left_idx], ibi_clean[right_idx]]
                    f = interp1d(x, y, kind="linear")
                    ibi_clean[gap] = f(gap)

        # Clamp interpolated values to physiologically realistic IBI range (300 ms - 2000 ms -> 30-200 bpm)
        ibi_clean[missing_idx] = np.clip(ibi_clean[missing_idx], 300.0, 2000.0)
        return ibi_clean

    def resample_regular(
        self, timestamps: np.ndarray, values: np.ndarray, target_fs: float = 4.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample non-uniform time series (e.g. IBIs) to a uniform grid (e.g. 4 Hz) for spectral analysis.

        Args:
            timestamps: Monotonic time stamps in seconds.
            values: Values (e.g., IBIs) corresponding to timestamps.
            target_fs: Uniform target sampling rate in Hz (default 4 Hz for HRV).

        Returns:
            (new_timestamps, resampled_values)
        """
        if len(timestamps) < 4:
            return timestamps, values

        t_start = timestamps[0]
        t_end = timestamps[-1]
        step = 1.0 / target_fs
        new_t = np.arange(t_start, t_end, step)

        cs = CubicSpline(timestamps, values, bc_type="not-a-knot")
        resampled_vals = cs(new_t)
        return new_t, resampled_vals
