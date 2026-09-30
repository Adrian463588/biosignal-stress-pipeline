"""Deterministic, sensor-local PPG features for the v3 stress model."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import welch
from scipy.stats import iqr, kurtosis, skew
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from .peak_detection import PpgPeakDetection, PpgPeakDetector


def _safe_stat(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("non_finite_feature")
    return float(value)


def _frequency_features(ibi_ms: NDArray[np.float64]) -> dict[str, float]:
    if ibi_ms.size < 8:
        return {"lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0}
    beat_time = np.cumsum(ibi_ms) / 1000.0
    beat_time -= beat_time[0]
    if beat_time[-1] <= 2.0:
        return {"lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0}
    grid = np.arange(0.0, beat_time[-1], 0.25)
    interpolated = np.interp(grid, beat_time, ibi_ms)
    frequency, power = welch(
        interpolated - np.mean(interpolated),
        fs=4.0,
        nperseg=min(256, interpolated.size),
    )
    lf = float(
        np.trapezoid(
            power[(frequency >= 0.04) & (frequency < 0.15)],
            frequency[(frequency >= 0.04) & (frequency < 0.15)],
        )
    )
    hf = float(
        np.trapezoid(
            power[(frequency >= 0.15) & (frequency <= 0.40)],
            frequency[(frequency >= 0.15) & (frequency <= 0.40)],
        )
    )
    return {
        "lf_power": _safe_stat(lf),
        "hf_power": _safe_stat(hf),
        "lf_hf_ratio": _safe_stat(lf / hf) if hf > 0.0 else 0.0,
    }


def extract_ppg_features(
    signal: NDArray[np.floating],
    sampling_rate_hz: float,
    *,
    channel: str = "ppg_green",
    detector: PpgPeakDetector | None = None,
    detection: PpgPeakDetection | None = None,
) -> tuple[dict[str, float], NDArray[np.float32], NDArray[np.uint8]]:
    """Return specific features per PRD and a 720-value masked IBI sequence."""

    values = np.asarray(signal, dtype=np.float64)
    assert values.ndim == 1 and values.size >= int(170 * sampling_rate_hz)
    assert np.isfinite(values).all()
    if detection is None:
        if detector is None:
            from .settings import load_pipeline_config

            detector = PpgPeakDetector(load_pipeline_config().peak_detection)
        detection = detector.detect(values, sampling_rate_hz, channel)
    peaks, troughs = detection.peaks, detection.troughs
    assert peaks.size >= 15, f"insufficient peaks: {peaks.size}"

    # 1. Timing (HRV)
    ibi_ms = np.diff(peaks).astype(np.float64) * 1000.0 / sampling_rate_hz
    physiological = ibi_ms[(ibi_ms >= 300.0) & (ibi_ms <= 1500.0)]
    assert physiological.size >= 12, "insufficient physiological IBI values"
    ibi_ms = physiological
    ibi_diff = np.diff(ibi_ms)
    heart_rate_arr = 60_000.0 / ibi_ms

    # 2. Morphology
    peak_amps = values[peaks]
    pulse_widths = []
    rise_times = []
    decay_times = []
    areas = []
    slopes = []
    gradient = np.diff(values)

    for i in range(len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i + 1]
        # find trough between p1 and p2
        tr = troughs[(troughs > p1) & (troughs < p2)]
        if len(tr) > 0:
            t1 = tr[0]
            decay_times.append((t1 - p1) / sampling_rate_hz)
            rise_times.append((p2 - t1) / sampling_rate_hz)
            pulse_widths.append((p2 - p1) / sampling_rate_hz)
            areas.append(np.sum(values[p1:p2]))
            if t1 > p1:
                slopes.append(np.min(gradient[p1:t1]))  # max decay slope (negative)
            if p2 > t1:
                slopes.append(np.max(gradient[t1:p2]))  # max rise slope (positive)

    # 3. Quality
    flatline = np.mean(np.abs(gradient) < 1e-6)
    outliers = np.mean(np.abs(values - np.mean(values)) > 3 * np.std(values))
    sqi = 1.0 - flatline - outliers

    features: dict[str, float] = {
        # Timing (HRV)
        "heart_rate": float(np.mean(heart_rate_arr)),
        "ibi_mean": float(np.mean(ibi_ms)),
        "ibi_std": float(np.std(ibi_ms, ddof=1)),
        "ibi_min": float(np.min(ibi_ms)),
        "ibi_max": float(np.max(ibi_ms)),
        "rmssd": float(np.sqrt(np.mean(np.square(ibi_diff)))),
        "sdnn": float(np.std(ibi_ms, ddof=1)),
        "pnn50": float(np.mean(np.abs(ibi_diff) > 50.0)),
        # Morphology
        "pulse_amplitude_mean": float(np.mean(peak_amps)),
        "pulse_amplitude_std": float(np.std(peak_amps, ddof=1)),
        "pulse_width": float(np.mean(pulse_widths)) if pulse_widths else 0.0,
        "rise_time": float(np.mean(rise_times)) if rise_times else 0.0,
        "decay_time": float(np.mean(decay_times)) if decay_times else 0.0,
        "area_under_pulse": float(np.mean(areas)) if areas else 0.0,
        "slope_features": float(np.mean(np.abs(slopes))) if slopes else 0.0,
        # Quality
        "signal_quality_index": float(sqi),
        "clipping_ratio": float(np.mean(values >= np.percentile(values, 99.9))),
        "flatline_ratio": float(flatline),
        "missing_ratio": float(np.mean(np.isnan(values))),
        "sampling_jitter": 0.0,
        "artifact_ratio": float(outliers),
        "valid_peak_ratio": float(len(physiological) / len(peaks))
        if len(peaks) > 0
        else 0.0,
        # Statistics
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "iqr": float(iqr(values)),
        "skewness": _safe_stat(float(skew(values, bias=False))),
        "kurtosis": _safe_stat(float(kurtosis(values, fisher=True, bias=False))),
        "rolling_min": float(
            np.mean(
                pd.Series(values).rolling(window=int(sampling_rate_hz)).min().dropna()
            )
        ),
        "rolling_max": float(
            np.mean(
                pd.Series(values).rolling(window=int(sampling_rate_hz)).max().dropna()
            )
        ),
    }

    features.update(_frequency_features(ibi_ms))

    if not all(np.isfinite(v) for v in features.values()):
        raise ValueError("non_finite_feature")

    sequence = np.zeros(720, dtype=np.float32)
    mask = np.zeros(720, dtype=np.uint8)
    beat_time = np.cumsum(ibi_ms) / 1000.0
    if beat_time.size >= 2 and beat_time[-1] > beat_time[0]:
        grid = np.arange(720, dtype=np.float64) / 4.0
        within = (grid >= beat_time[0]) & (grid <= beat_time[-1])
        sequence[within] = np.interp(grid[within], beat_time, ibi_ms).astype(np.float32)
        mask[within] = 1
    return features, sequence, mask


def feature_order(rows: list[Mapping[str, float]]) -> tuple[str, ...]:
    assert rows, "feature rows are required"
    canonical = tuple(sorted(rows[0]))
    assert all(tuple(sorted(row)) == canonical for row in rows)
    return canonical


# =============================================================================
# DATASET PROCESSING & LEAKAGE-SAFE UTILITIES
# =============================================================================


def process_and_save_features(
    silver_data: pd.DataFrame,
    output_path: str | Path = "data/gold/gold_features_v3.parquet",
    sampling_rate_hz: float = 64.0,
) -> pd.DataFrame:
    """
    Iterate over windowed signals in a Silver DataFrame, extract timing/HRV
    and morphology features, and save to a Gold Parquet file.

    Args:
        silver_data: A DataFrame containing windowed PPG signals.
                     Expected to have a 'signal' column with NumPy arrays.
        output_path: Path where the resulting Gold feature DataFrame is saved.
        sampling_rate_hz: Sampling rate of the PPG signal.
    """
    from .settings import load_pipeline_config

    feature_rows = []
    detector = PpgPeakDetector(load_pipeline_config().peak_detection)

    for idx, row in silver_data.iterrows():
        # Example assumes row['signal'] holds the raw window signal
        signal = row["signal"]

        try:
            features, _, _ = extract_ppg_features(
                signal,
                sampling_rate_hz,
                channel=str(row.get("sensor_channel", "ppg_green")),
                detector=detector,
            )
            # Add metadata from the silver dataset (e.g. participant, phase, label)
            # We explicitly copy existing metadata that is not the signal array itself.
            meta = {k: v for k, v in row.items() if k != "signal"}
            meta.update(features)
            feature_rows.append(meta)
        except AssertionError:
            # e.g., insufficient peaks
            continue

    gold_df = pd.DataFrame(feature_rows)

    # Ensure directory exists and save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gold_df.to_parquet(out_path, index=False)

    return gold_df


class FoldSafeTransformer:
    """
    Utility class to achieve leakage-free imputation and scaling inside GroupKFold splits.

    Fits Imputer and Scaler ONLY on the training folds (identified by train_idx),
    preventing data leakage from the holdout/validation sets. It then transforms
    the entire dataset or a specified validation set.
    """

    def __init__(self, feature_cols: list[str], strategy: str = "median"):
        self.feature_cols = feature_cols
        self.imputer = SimpleImputer(strategy=strategy)
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
        """
        Fits on the training subset only, and transforms the ENTIRE DataFrame.
        Useful when you want to return a safe version of the whole dataframe
        for evaluation while strictly fitting on training data.
        """
        df_out = df.copy()

        # 1. Fit on train subset
        train_data = df.iloc[train_idx][self.feature_cols]

        # Fit imputer and scaler
        self.imputer.fit(train_data)
        train_imputed = self.imputer.transform(train_data)
        self.scaler.fit(train_imputed)

        # 2. Transform the full DataFrame
        full_data = df_out[self.feature_cols]
        full_imputed = self.imputer.transform(full_data)
        full_scaled = self.scaler.transform(full_imputed)

        df_out[self.feature_cols] = full_scaled
        return df_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a new DataFrame (e.g., Holdout set) using the already fitted
        Imputer and Scaler.
        """
        df_out = df.copy()
        data = df_out[self.feature_cols]
        imputed = self.imputer.transform(data)
        scaled = self.scaler.transform(imputed)
        df_out[self.feature_cols] = scaled
        return df_out
