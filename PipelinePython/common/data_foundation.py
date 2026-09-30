"""Legacy data-foundation compatibility adapter.

This module is retained for older Bronze/Silver/Gold consumers only.  It is
not the canonical ECG--PPG validation path: primary validation is implemented
by :class:`PipelinePython.common.peak_validation.PpgEcgValidator`, which preserves ECG
gaps and consumes canonical Silver PPG values without a second interpolation
or filtering pass.  Keep this adapter import-compatible, but do not use it as
the source of ECG--PPG validation evidence.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .anchors import AnchorMapper
from .config import ECG_FS, PPG_FS, PPG_IR_FS, PPG_RED_FS, TARGET_FS
from .contracts import CANONICAL_SIGNAL_COLUMNS, CANONICAL_BRONZE_COLUMNS
from .feature_extraction import FeatureExtractor, WindowProcessor
from .ingestion import DataLoader, get_sampling_rate_hz
from .preprocessing import Resampler, SignalFilter, resample_timestamped_dataframe
from .alignment import DataAligner
from .settings import PipelinePaths, PipelineSettings
from .sqi import RawSignalSQA

logger = logging.getLogger(__name__)

LEGACY_ADAPTER_ONLY = True
CANONICAL_ECG_PPG_VALIDATOR = "PipelinePython.common.peak_validation.PpgEcgValidator"


class DataFoundationBuilder:
    """Compatibility adapter for the pre-canonical data-foundation flow.

    The public API remains available for legacy consumers.  New validation
    runs must use ``PpgEcgValidator`` and the canonical Silver preprocessing
    contract instead of this class's historical filtering/SQA/windowing path.
    """

    legacy_adapter_only = True
    canonical_ecg_ppg_validator = CANONICAL_ECG_PPG_VALIDATOR

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
        self.extractor = FeatureExtractor()
        self.windower = WindowProcessor(
            window_sec=settings.model_window_seconds,
            step_sec=settings.model_window_step_seconds,
            target_fs=100,
        )
        self._raw_sqa = RawSignalSQA()

    def run_ingest(self, participant: Optional[str] = None) -> pd.DataFrame:
        sessions = (
            self.loader.load_participant(participant)
            if participant
            else self.loader.load_all()
        )
        rows = []
        for s in sessions:
            session_id = f"{s.participant}_{s.condition}"
            participant_code = self.anchor_mapper.infer_code_from_session_participant(
                s.participant
            )
            for signal_type, df, fs in [
                ("ECG", s.ecg, ECG_FS),
                ("PPG_GREEN", s.ppg_green, PPG_FS),
                ("PPG_IR", s.ppg_ir, PPG_IR_FS),
                ("PPG_RED", s.ppg_red, PPG_RED_FS),
            ]:
                if df is None or df.empty:
                    continue
                source_col = "ecg_mv" if signal_type == "ECG" else "ppg_raw"
                sampling_rate = get_sampling_rate_hz(df, default=fs)
                ts_utc = pd.to_datetime(df["unix_ms"], unit="ms", utc=True)
                ts_local = ts_utc.dt.tz_convert(self.settings.timezone_local)
                local_df = pd.DataFrame(
                    {
                        "participant_code": participant_code,
                        "session_id": session_id,
                        "signal_type": signal_type,
                        "timestamp_utc": ts_utc,
                        "timestamp_local": ts_local,
                        "sample_index": np.arange(len(df), dtype=int),
                        "raw_value": pd.to_numeric(df[source_col], errors="coerce"),
                        "sampling_rate_hz": sampling_rate,
                        "phase": pd.NA,
                        "anchor_start": pd.NaT,
                        "anchor_end": pd.NaT,
                        "quality_flag": "UNKNOWN",
                        "preprocessing_version": self.settings.preprocessing_version,
                    }
                )
                rows.append(local_df)

        bronze = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(columns=CANONICAL_BRONZE_COLUMNS)
        )
        bronze = bronze[CANONICAL_BRONZE_COLUMNS]

        if not bronze.empty:
            for (pcode, sig_type), df_group in bronze.groupby(
                ["participant_code", "signal_type"]
            ):
                ch_label = sig_type.lower()
                out_dir = self.paths.bronze_dir / pcode / ch_label
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "raw.csv"
                df_group.to_csv(out_path, index=False)

        logger.info(
            "Bronze saved into separated files in %s (%d rows)",
            self.paths.bronze_dir,
            len(bronze),
        )
        return bronze

    def run_segment(self, bronze: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if bronze is None:
            import glob

            # We search recursively since bronze is now split by participant/sensor
            bronze_files = glob.glob(str(self.paths.bronze_dir / "*" / "*" / "raw.csv"))
            if not bronze_files:
                logger.warning("No bronze data found in %s", self.paths.bronze_dir)
                return pd.DataFrame(columns=CANONICAL_BRONZE_COLUMNS)

            dfs = [pd.read_csv(f) for f in bronze_files]
            bronze = pd.concat(dfs, ignore_index=True)
            bronze["timestamp_utc"] = pd.to_datetime(
                bronze["timestamp_utc"], utc=True, errors="coerce", format="mixed"
            )
            bronze["timestamp_local"] = pd.to_datetime(
                bronze["timestamp_local"], utc=True, errors="coerce", format="mixed"
            )
            bronze = bronze.dropna(subset=["timestamp_utc"])

        anchors = self.anchor_mapper.load()
        buffer_delta = pd.Timedelta(seconds=self.settings.anchor_buffer_seconds)

        silver_parts = []
        for _, a in anchors.iterrows():
            p = a["participant_code"]
            phase = a["phase"]
            start = a["start_timestamp"] + buffer_delta
            end = a["end_timestamp"] - buffer_delta
            if end <= start:
                continue

            part = bronze[
                (bronze["participant_code"] == p)
                & (bronze["timestamp_utc"] >= start)
                & (bronze["timestamp_utc"] <= end)
            ].copy()
            if part.empty:
                continue
            part["phase"] = phase
            part["anchor_start"] = a["start_timestamp"]
            part["anchor_end"] = a["end_timestamp"]
            # ── SQA per sensor per segment (AGENTS §10, PRD §9) ──────────
            for (sig_id, sig_type), idx in part.groupby(
                ["session_id", "signal_type"]
            ).groups.items():
                seg = part.loc[idx]
                q_score, q_tier, q_flags = self._raw_sqa.score(
                    seg["raw_value"],
                    timestamps_utc=seg["timestamp_utc"],
                )
                legacy_flag = "|".join(q_flags) if q_flags else "PASS"
                part.loc[idx, "quality_flag"] = legacy_flag
                part.loc[idx, "quality_score"] = q_score
                part.loc[idx, "quality_tier"] = q_tier
                if q_tier in {"poor", "invalid"}:
                    logger.warning(
                        "SQA excluded: participant=%s phase=%s signal=%s tier=%s flags=%s",
                        p,
                        phase,
                        sig_type,
                        q_tier,
                        q_flags,
                    )
            silver_parts.append(part)

        silver = (
            pd.concat(silver_parts, ignore_index=True)
            if silver_parts
            else pd.DataFrame(columns=CANONICAL_SIGNAL_COLUMNS)
        )
        # Ensure new SQA columns exist even if silver is empty
        for col, default in [("quality_score", 0.0), ("quality_tier", "invalid")]:
            if col not in silver.columns:
                silver[col] = default
        silver = silver[CANONICAL_SIGNAL_COLUMNS]

        if not silver.empty:
            for (pcode, phase, sig_type), df_group in silver.groupby(
                ["participant_code", "phase", "signal_type"]
            ):
                ch_label = sig_type.lower()
                out_dir = self.paths.silver_dir / pcode / phase
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{ch_label}.parquet"
                df_group.to_parquet(out_path, index=False)

        logger.info(
            "Silver saved into separated files in %s (%d rows)",
            self.paths.silver_dir,
            len(silver),
        )
        return silver

    def run_gold_features(self, participant: Optional[str] = None) -> pd.DataFrame:
        sessions = (
            self.loader.load_participant(participant)
            if participant
            else self.loader.load_all()
        )
        anchors = self.anchor_mapper.load()
        phases_keep = {"Relax": 0, "Task": 1}
        continuous_rate_contract = (
            self.settings.preprocessing_version == "v3.1.0_continuous25_ir100"
        )

        # Sensor channel definitions: (attr_name, sensor_channel_label, default_fs)
        PPG_CHANNELS = [
            ("ppg_ir", "ppg_ir", PPG_IR_FS),
            ("ppg_red", "ppg_red", PPG_RED_FS),
            ("ppg_green", "ppg_green", PPG_FS),
        ]

        rows = []
        for s in sessions:
            pcode = self.anchor_mapper.infer_code_from_session_participant(
                s.participant
            )
            panchors = anchors[
                (anchors["participant_code"] == pcode)
                & (anchors["phase"].isin(phases_keep.keys()))
            ]
            if panchors.empty:
                continue

            for attr, channel_label, default_fs in PPG_CHANNELS:
                ppg_raw = getattr(s, attr, None)
                if ppg_raw is None or len(ppg_raw) < 30:
                    logger.warning(
                        "Session %s channel=%s too short or missing, skipping.",
                        s.participant,
                        channel_label,
                    )
                    continue

                ppg = ppg_raw.copy()
                fs = get_sampling_rate_hz(ppg, default=default_fs)
                model_fs = 25.0 if channel_label == "ppg_green" else 100.0
                if continuous_rate_contract:
                    # Emulate the supported Samsung source: both channels are
                    # physically 25 Hz. Historical IR recordings may be 100 Hz,
                    # so they are first reduced to 25 Hz, then canonicalized to
                    # the IR model's 100 Hz input rate using physical timestamps.
                    source_25 = resample_timestamped_dataframe(
                        ppg,
                        "ppg_raw",
                        target_fs=25.0,
                    )
                    canonical = (
                        source_25
                        if model_fs == 25.0
                        else resample_timestamped_dataframe(
                            source_25,
                            "ppg_raw",
                            target_fs=model_fs,
                        )
                    )
                    canonical["ppg_raw"] = self.filter.apply_all(
                        canonical["ppg_raw"].to_numpy(), fs=model_fs
                    )
                    ppg_r = canonical
                else:
                    ppg["ppg_raw"] = self.filter.apply_all(
                        ppg["ppg_raw"].to_numpy(), fs=fs
                    )
                    ppg_r = Resampler(
                        source_fs=fs,
                        target_fs=TARGET_FS,
                    ).resample_dataframe(ppg, "ppg_raw")
                channel_extractor = FeatureExtractor(target_fs=model_fs)
                channel_windower = WindowProcessor(
                    window_sec=self.settings.model_window_seconds,
                    step_sec=self.settings.model_window_step_seconds,
                    target_fs=model_fs,
                )

                ts_utc = pd.to_datetime(ppg_r["unix_ms"], unit="ms", utc=True)
                ppg_r = ppg_r.assign(timestamp_utc=ts_utc)

                for _, a in panchors.iterrows():
                    phase = a["phase"]
                    start = a["start_timestamp"] + pd.Timedelta(
                        seconds=self.settings.anchor_buffer_seconds
                    )
                    end = a["end_timestamp"] - pd.Timedelta(
                        seconds=self.settings.anchor_buffer_seconds
                    )
                    phase_df = ppg_r[
                        (ppg_r["timestamp_utc"] >= start)
                        & (ppg_r["timestamp_utc"] <= end)
                    ].copy()
                    if phase_df.empty:
                        continue

                    # SQA gate before windowing (AGENTS §10, PRD §9)
                    q_score, q_tier, q_flags = self._raw_sqa.score(
                        phase_df["ppg_raw"],
                        timestamps_utc=phase_df["timestamp_utc"],
                    )
                    if q_tier in {"poor", "invalid"}:
                        logger.warning(
                            "Gold skipped: participant=%s phase=%s channel=%s tier=%s",
                            pcode,
                            phase,
                            channel_label,
                            q_tier,
                        )
                        continue

                    windows_list = []
                    for window_idx, window in channel_windower.generate_windows(
                        phase_df
                    ):
                        # Rename 'ppg_raw' to the canonical channel column name
                        # so FeatureExtractor can identify the signal type
                        w = window.rename(columns={"ppg_raw": channel_label})

                        w_copy = w.copy()
                        w_copy["window_id"] = window_idx
                        w_copy["participant_code"] = pcode
                        w_copy["phase"] = phase
                        w_copy["sensor_channel"] = channel_label
                        windows_list.append(w_copy)

                        features, _ = channel_extractor.extract(
                            window=w,
                            participant=pcode,
                            condition=phase,
                            label=phases_keep[phase],
                            window_idx=window_idx,
                        )
                        if not features:
                            continue
                        features["participant_code"] = pcode
                        features["session_id"] = f"{s.participant}_{s.condition}"
                        features["phase"] = phase
                        features["sensor_channel"] = channel_label
                        features["quality_score"] = q_score
                        features["quality_tier"] = q_tier
                        features["feature_version"] = self.settings.feature_version
                        rows.append(features)

                    if windows_list:
                        all_windows = pd.concat(windows_list, ignore_index=True)
                        win_out_dir = (
                            self.paths.gold_dir / "windows" / "stress" / pcode / phase
                        )
                        win_out_dir.mkdir(parents=True, exist_ok=True)
                        all_windows.to_parquet(
                            win_out_dir / f"{channel_label}.parquet", index=False
                        )

        gold = pd.DataFrame(rows)
        if not gold.empty:
            for c in ["condition", "participant"]:
                if c in gold.columns:
                    gold = gold.drop(columns=[c])

            for (pcode, phase, ch_label), df_group in gold.groupby(
                ["participant_code", "phase", "sensor_channel"]
            ):
                feat_out_dir = (
                    self.paths.gold_dir / "features" / "stress" / pcode / phase
                )
                feat_out_dir.mkdir(parents=True, exist_ok=True)
                feat_out_path = feat_out_dir / f"{ch_label}.parquet"
                df_group.to_parquet(feat_out_path, index=False)

        flat_gold_path = getattr(self.paths, "gold_features_path", None)
        if flat_gold_path is not None:
            flat_gold_path.parent.mkdir(parents=True, exist_ok=True)
            gold.to_csv(flat_gold_path, index=False)

        logger.info(
            "Gold saved into separated files in %s and flat file %s (%d features extracted, channels=%s)",
            self.paths.gold_dir,
            flat_gold_path,
            len(gold),
            sorted(gold["sensor_channel"].unique().tolist()) if not gold.empty else [],
        )
        return gold

    @staticmethod
    def _quality_flag_for_segment(df: pd.DataFrame) -> str:
        if df.empty:
            return "EMPTY"
        vals = pd.to_numeric(df["raw_value"], errors="coerce")
        vals = vals.dropna()
        if vals.empty:
            return "MISSING"

        ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        ts = ts.dropna()
        flags = []

        if len(vals) > 5 and float(vals.std()) < 1e-9:
            flags.append("FLATLINE")

        if len(ts) > 5:
            diffs_ms = ts.sort_values().diff().dropna().dt.total_seconds() * 1000.0
            median = float(diffs_ms.median()) if not diffs_ms.empty else 0.0
            if median > 0:
                gap_ratio = float((diffs_ms > (1.5 * median)).mean())
                if gap_ratio > 0.02:
                    flags.append("TIME_GAP")

        # Saturation heuristic: too many samples at exact min/max.
        if len(vals) > 20:
            vmin, vmax = float(vals.min()), float(vals.max())
            sat_ratio = float(((vals == vmin) | (vals == vmax)).mean())
            if sat_ratio > 0.2:
                flags.append("SATURATED")

        return "PASS" if not flags else "|".join(flags)
