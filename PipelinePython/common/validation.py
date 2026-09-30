from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .alignment import DataAligner
from .anchors import AnchorMapper
from .config import ECG_FS, PPG_FS, PPG_IR_FS, PPG_RED_FS
from .ingestion import DataLoader, get_sampling_rate_hz
from .preprocessing import Resampler, SignalFilter
from .settings import PipelinePaths, PipelineSettings

logger = logging.getLogger(__name__)


@dataclass
class ChannelValidationResult:
    channel: str
    participant_code: str
    phase: str
    precision: float
    recall: float
    f1: float
    peak_mae_ms: float
    ibi_mae_ms: float
    ibi_rmse_ms: float
    pearson_r: float
    spearman_r: float
    bland_altman_bias_ms: float
    bland_altman_loa_low_ms: float
    bland_altman_loa_high_ms: float
    n_pairs: int


class SignalValidator:
    def __init__(self, paths: PipelinePaths, settings: PipelineSettings) -> None:
        self.paths = paths
        self.settings = settings
        self.loader = DataLoader(paths.raw_data_root)
        self.anchor_mapper = AnchorMapper(
            paths.anchor_csv_path,
            settings.timezone_local,
            mapping_csv_path=paths.raw_data_root / "mapping.csv",
        )
        self.filter = SignalFilter()
        self.aligner = DataAligner()

    def run(self, participant: Optional[str] = None) -> pd.DataFrame:
        anchors = self.anchor_mapper.load()
        sessions = (
            self.loader.load_participant(participant)
            if participant
            else self.loader.load_all()
        )

        out: List[dict] = []
        for s in sessions:
            pcode = self.anchor_mapper.infer_code_from_session_participant(
                s.participant
            )
            if s.ecg is None:
                continue
            panchors = anchors[
                (anchors["participant_code"] == pcode)
                & (anchors["phase"].isin(["Relax", "Task"]))
            ]
            if panchors.empty:
                continue

            if len(s.ecg) < 30:
                logger.warning(
                    "Session %s ECG too short for filtering (<30 samples), skipping validation.",
                    s.participant,
                )
                continue

            ecg = s.ecg.copy()
            ecg_fs = get_sampling_rate_hz(ecg, default=ECG_FS)
            ecg["ecg_mv"] = self.filter.apply_all(ecg["ecg_mv"].to_numpy(), fs=ecg_fs)
            ecg = Resampler(source_fs=ecg_fs, target_fs=100).resample_dataframe(
                ecg, "ecg_mv"
            )

            channels = {
                "PPG_GREEN": (s.ppg_green, PPG_FS),
                "PPG_IR": (s.ppg_ir, PPG_IR_FS),
                "PPG_RED": (s.ppg_red, PPG_RED_FS),
            }
            for ch_name, (ch_df, ch_default_fs) in channels.items():
                if ch_df is None or len(ch_df) < 30:
                    continue
                ppg = ch_df.copy()
                ppg_fs = get_sampling_rate_hz(ppg, default=ch_default_fs)
                ppg["ppg_raw"] = self.filter.apply_all(
                    ppg["ppg_raw"].to_numpy(), fs=ppg_fs
                )
                ppg = Resampler(source_fs=ppg_fs, target_fs=100).resample_dataframe(
                    ppg, "ppg_raw"
                )
                # Always alias the current channel's PPG data to ppg_green when
                # calling align() — the aligner joins on that key — then rename
                # the merged column to the channel-agnostic sentinel "ppg_signal"
                # so that _validate_segment reads the correct channel's data.
                aligned = self.aligner.align(ecg=ecg, ppg_green=ppg)
                if aligned.empty:
                    continue
                if "ppg_green" in aligned.columns:
                    aligned = aligned.rename(columns={"ppg_green": "ppg_signal"})
                elif "ppg_raw" in aligned.columns:
                    aligned = aligned.rename(columns={"ppg_raw": "ppg_signal"})
                aligned["timestamp_utc"] = pd.to_datetime(
                    aligned["unix_ms"], unit="ms", utc=True
                )

                for _, a in panchors.iterrows():
                    phase = a["phase"]
                    start = a["start_timestamp"] + pd.Timedelta(
                        seconds=self.settings.anchor_buffer_seconds
                    )
                    end = a["end_timestamp"] - pd.Timedelta(
                        seconds=self.settings.anchor_buffer_seconds
                    )
                    seg = aligned[
                        (aligned["timestamp_utc"] >= start)
                        & (aligned["timestamp_utc"] <= end)
                    ].copy()
                    if len(seg) < 500:
                        continue
                    row = self._validate_segment(seg, pcode, phase, ch_name)
                    if row is not None:
                        out.append(row.__dict__)

        df = pd.DataFrame(out)
        out_path = self.paths.validation_dir / "channel_validation_metrics.csv"
        df.to_csv(out_path, index=False)

        recommendation = self._recommend_channel(df)
        (self.paths.validation_dir / "channel_recommendation.txt").write_text(
            recommendation, encoding="utf-8"
        )
        logger.info("Validation saved: %s (%d rows)", out_path, len(df))
        return df

    def _detect_peaks(self, signal: np.ndarray, signal_type: str) -> np.ndarray:
        if signal_type == "ecg":
            _, info = nk.ecg_peaks(
                signal,
                sampling_rate=100,
                method="pantompkins1985",
                correct_artifacts=True,
            )
            return np.array(info.get("ECG_R_Peaks", []), dtype=int)
        clean = nk.ppg_clean(signal, sampling_rate=100, method="elgendi")
        info = nk.ppg_findpeaks(clean, sampling_rate=100, method="elgendi")
        return np.array(info.get("PPG_Peaks", []), dtype=int)

    def _match_peaks(
        self, ecg_ts: np.ndarray, ppg_ts: np.ndarray
    ) -> Tuple[int, int, int, List[float]]:
        tol = self.settings.beat_match_tolerance_ms
        matched = 0
        used = set()
        errors: List[float] = []
        for t in ppg_ts:
            diffs = np.abs(ecg_ts - t)
            idx = int(np.argmin(diffs)) if len(diffs) else -1
            if idx >= 0 and diffs[idx] <= tol and idx not in used:
                matched += 1
                used.add(idx)
                errors.append(float(diffs[idx]))
        fp = max(0, len(ppg_ts) - matched)
        fn = max(0, len(ecg_ts) - matched)
        return matched, fp, fn, errors

    def _validate_segment(
        self, seg: pd.DataFrame, participant_code: str, phase: str, channel: str
    ) -> Optional[ChannelValidationResult]:
        ecg = seg["ecg_mv"].to_numpy()
        # "ppg_signal" is the channel-agnostic column set by run() after renaming
        # the aligner output — ensures IR and Red channels are not silently
        # evaluated against the Green column.
        ppg = seg["ppg_signal"].to_numpy()
        ecg_peaks = self._detect_peaks(ecg, "ecg")
        ppg_peaks = self._detect_peaks(ppg, "ppg")
        if len(ecg_peaks) < 3 or len(ppg_peaks) < 3:
            return None

        unix_ms = seg["unix_ms"].to_numpy()
        ecg_ts = unix_ms[ecg_peaks]
        ppg_ts = unix_ms[ppg_peaks]

        tp, fp, fn, peak_errors = self._match_peaks(ecg_ts, ppg_ts)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        ecg_ibi = np.diff(ecg_ts)
        ppg_ibi = np.diff(ppg_ts)
        n = min(len(ecg_ibi), len(ppg_ibi))
        if n < 2:
            return None
        ecg_ibi = ecg_ibi[:n]
        ppg_ibi = ppg_ibi[:n]
        diff = ppg_ibi - ecg_ibi

        try:
            pear = float(pearsonr(ecg_ibi, ppg_ibi).statistic)
        except Exception:
            pear = float("nan")
        try:
            spear = float(spearmanr(ecg_ibi, ppg_ibi).correlation)
        except Exception:
            spear = float("nan")

        bias = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0

        return ChannelValidationResult(
            channel=channel,
            participant_code=participant_code,
            phase=phase,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            peak_mae_ms=float(np.mean(peak_errors) if peak_errors else np.nan),
            ibi_mae_ms=float(np.mean(np.abs(diff))),
            ibi_rmse_ms=float(np.sqrt(np.mean(diff**2))),
            pearson_r=pear,
            spearman_r=spear,
            bland_altman_bias_ms=bias,
            bland_altman_loa_low_ms=float(bias - 1.96 * sd),
            bland_altman_loa_high_ms=float(bias + 1.96 * sd),
            n_pairs=int(n),
        )

    def _recommend_channel(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "NO_GO: no validation rows"
        summary = df.groupby("channel")[["f1", "pearson_r", "ibi_mae_ms"]].mean(
            numeric_only=True
        )
        summary = summary.sort_values(
            by=["f1", "pearson_r", "ibi_mae_ms"], ascending=[False, False, True]
        )
        best = summary.index[0]
        row = summary.iloc[0]
        decision = (
            "GO"
            if row["f1"] >= 0.9 and row["pearson_r"] >= 0.9 and row["ibi_mae_ms"] <= 50
            else "CONDITIONAL"
        )
        return f"{decision}: {best} | mean_f1={row['f1']:.3f}, mean_pearson={row['pearson_r']:.3f}, mean_ibi_mae_ms={row['ibi_mae_ms']:.2f}"
