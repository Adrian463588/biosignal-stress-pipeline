"""CSV-first, human-readable ECG--PPG peak validation artifacts.

The validator remains responsible for signal processing and metric computation.
This module is an I/O and visualization boundary: it consumes the canonical
Silver CSV, validation metrics, and peak tables, then writes one auditable CSV
bundle and plot for every participant/phase/channel combination.

The normalized signal is for visualization only. Peak timing always comes from
the canonical ``clean_value`` signal. ECG is corrected upstream; PPG timestamps
are never shifted.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import VALID_PARTICIPANTS

PPG_CHANNELS = ("ppg_ir", "ppg_red", "ppg_green")
PHASES = ("relax", "task")
EXPECTED_COLUMNS = (
    "signal_source",
    "elapsed_seconds",
    "timestamp_raw",
    "timestamp_corrected",
    "raw_value",
    "clean_value",
    "normalized_clean_value",
    "interpolated_mask",
    "peak_flag",
    "matched_peak_id",
    "pat_ms",
)

GAP_COLUMNS = (
    "signal_source",
    "gap_start",
    "gap_end",
    "gap_duration_ms",
    "sample_count",
    "gap_type",
    "action",
    "affects_peak_region",
)


@dataclass(frozen=True)
class CsvVisualOutputs:
    """Paths and counts produced by :class:`CsvPeakToPeakVisualizer`."""

    output_dir: Path
    manifest_path: Path
    row_count: int
    valid_count: int


class CsvPeakToPeakVisualizer:
    """Build complete per-session CSV and plot artifacts from a run."""

    def __init__(self, run_root: Path, project_root: Path | None = None) -> None:
        self.run_root = Path(run_root)
        self.project_root = Path(project_root or self.run_root.parents[3])
        self.validation_root = self.run_root / "validation" / "ppg_ecg_240s"
        if not self.validation_root.exists():
            # Historical fixtures/runs remain readable; new runs always write
            # to the canonical 240-second output tree.
            self.validation_root = self.run_root / "validation" / "ppg_ecg"
        self.output_dir = self.run_root / "validation" / "ppg_ecg_csv"
        self.csv_dir = self.output_dir / "csv"
        self.plot_dir = self.output_dir / "plots"
        self.metrics_path = next(
            (
                self.validation_root / filename
                for filename in (
                    "ppg_ecg_metrics_240s.csv",
                    "ppg_ecg_metrics_5min.csv",
                    "ppg_ecg_metrics.csv",
                )
                if (self.validation_root / filename).is_file()
            ),
            self.validation_root / "ppg_ecg_metrics_240s.csv",
        )
        self._signal_frame_cache: dict[
            tuple[Path, Path], tuple[pd.DataFrame, pd.DataFrame]
        ] = {}
        self.metric_columns = [
            "participant_id",
            "phase",
            "sensor_channel",
            "status",
            "validation_status",
            "agreement_status",
            "exclusion_reason",
            "validation_window_seconds",
            "window_start_utc",
            "window_end_utc",
            "ecg_detection_method",
            "ppg_detection_method",
            "detection_fallback_reason",
            "ecg_peak_count",
            "ppg_peak_count",
            "matched_beat_count",
            "matched_beat_ratio",
            "beat_precision",
            "beat_recall",
            "beat_f1",
            "pat_median_ms",
            "ibi_mae_ms",
            "ibi_median_absolute_error_ms",
            "ibi_rmse_ms",
            "ibi_pearson_r",
            "ibi_spearman_r",
            "ibi_pearson",
            "ibi_spearman",
            "hr_mae_bpm",
            "bland_altman_bias_ms",
            "bland_altman_loa_low_ms",
            "bland_altman_loa_high_ms",
            "ecg_gap_count",
            "ecg_gap_seconds",
            "ecg_gap_mask_count",
            "ecg_interpolated_sample_count",
            "ecg_peak_in_gap_count",
            "ppg_gap_count",
            "ppg_gap_seconds",
            "ppg_gap_mask_count",
            "ppg_interpolated_sample_count",
            "ppg_peak_in_gap_count",
            "rr_interpolation_used",
            "rr_interpolation_method",
        ]

    def run(self) -> CsvVisualOutputs:
        """Write 192 status rows, CSV bundles, plots, and a report."""

        metrics = self._load_metrics()
        self._reset_output_directory()
        rows: list[dict[str, Any]] = []
        for participant_id in VALID_PARTICIPANTS:
            for phase in PHASES:
                for channel in PPG_CHANNELS:
                    key = (participant_id, phase, channel)
                    metric = self._metric_for(metrics, key)
                    rows.append(self._write_session(metric, *key))

        manifest = pd.DataFrame(rows)
        self._copy_canonical_tables()
        manifest_path = self.output_dir / "csv_visual_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        self._write_summary(manifest)
        self._write_report(manifest)
        self._mirror_into_canonical_validation_tree()
        return CsvVisualOutputs(
            self.output_dir,
            manifest_path,
            len(manifest),
            int((manifest["validation_status"] == "valid").sum()),
        )

    def _mirror_into_canonical_validation_tree(self) -> None:
        """Expose CSV-first audits under the canonical 240-second tree.

        ``ppg_ecg_csv`` remains a compatibility boundary for existing readers.
        The canonical run contract also requires ``ppg_ecg_240s/csv`` and the
        corresponding human-readable plots.  Copying generated artifacts is
        an I/O concern only; no signal is recalculated or normalized again.
        """

        if self.validation_root == self.output_dir:
            return
        for directory in ("csv", "plots"):
            source = self.output_dir / directory
            destination = self.validation_root / directory
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
        for filename in ("csv_visual_manifest.csv",):
            source = self.output_dir / filename
            if source.is_file():
                shutil.copyfile(source, self.validation_root / filename)

    def _copy_canonical_tables(self) -> None:
        """Co-locate the canonical CSV metrics with the visual CSV bundle."""
        metrics_source = self.metrics_path
        if not metrics_source.is_file():
            raise FileNotFoundError(f"missing canonical CSV: {metrics_source}")
        shutil.copyfile(metrics_source, self.output_dir / "ppg_ecg_metrics_240s.csv")

        # Synchronization is required by the real pipeline, but lightweight
        # fixture runs may intentionally provide metrics only.
        sync_source = self.validation_root / "sync_parameters_240s.csv"
        if sync_source.is_file():
            shutil.copyfile(sync_source, self.output_dir / "sync_parameters_240s.csv")

    def _reset_output_directory(self) -> None:
        """Create only the new run-local output tree; never remove old runs."""

        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def _load_metrics(self) -> pd.DataFrame:
        if not self.metrics_path.is_file():
            raise FileNotFoundError(f"missing canonical metrics: {self.metrics_path}")
        metrics = pd.read_csv(self.metrics_path)
        required = {"participant_id", "phase", "sensor_channel", "status"}
        missing = required - set(metrics.columns)
        if missing:
            raise ValueError(f"metrics missing columns: {sorted(missing)}")
        return metrics

    @staticmethod
    def _metric_for(metrics: pd.DataFrame, key: tuple[str, str, str]) -> dict[str, Any]:
        participant_id, phase, channel = key
        selected = metrics.loc[
            (metrics["participant_id"] == participant_id)
            & (metrics["phase"] == phase)
            & (metrics["sensor_channel"] == channel)
        ]
        if selected.empty:
            return {
                "participant_id": participant_id,
                "phase": phase,
                "sensor_channel": channel,
                "status": "excluded",
                "exclusion_reason": "missing_validation_row",
            }
        if len(selected) != 1:
            raise ValueError(
                "duplicate_validation_rows:"
                f"{participant_id}/{phase}/{channel}:{len(selected)}"
            )
        return selected.iloc[0].to_dict()

    def _write_session(
        self, metric: dict[str, Any], participant_id: str, phase: str, channel: str
    ) -> dict[str, Any]:
        session_csv = self.csv_dir / participant_id / phase / channel
        session_plot = self.plot_dir / participant_id / phase / channel
        session_csv.mkdir(parents=True, exist_ok=True)
        session_plot.mkdir(parents=True, exist_ok=True)

        validation_status = self._validation_status(metric)
        signal_frames = self._load_signal_frames(participant_id, phase, channel)
        plot_path = session_plot / "peak_to_peak_240s.png"
        if validation_status == "valid" and signal_frames is not None:
            ecg, ppg = signal_frames
            window_start = self._timestamp(metric.get("window_start_utc"))
            window_end = self._timestamp(metric.get("window_end_utc"))
            ecg = self._crop(ecg, window_start, window_end, "ecg")
            ppg = self._crop(ppg, window_start, window_end, channel)
            peaks = self._load_peak_table(participant_id, phase, channel)
            sample_frame, matches, intervals = self._build_csv_frames(
                ecg, ppg, peaks, window_start, participant_id, phase, channel
            )
            gap_audit = self._gap_audit(ecg, ppg, peaks)
            self._plot_valid(
                plot_path,
                ecg,
                ppg,
                matches,
                intervals,
                metric,
                participant_id,
                phase,
                channel,
                window_start,
            )
            output_status = "ok"
        else:
            reason = str(metric.get("exclusion_reason") or "validation_excluded")
            if signal_frames is None:
                sample_frame = self._empty_samples()
                gap_audit = self._empty_gap_audit()
            else:
                ecg, ppg = signal_frames
                start_value = metric.get("window_start_utc")
                end_value = metric.get("window_end_utc")
                if pd.notna(start_value) and pd.notna(end_value):
                    start = self._timestamp(start_value)
                    end = self._timestamp(end_value)
                    ecg = self._crop(ecg, start, end, "ecg")
                    ppg = self._crop(ppg, start, end, channel)
                sample_frame = self._observed_samples(ecg, ppg, channel)
                gap_audit = self._gap_audit(ecg, ppg, pd.DataFrame())
            matches = self._empty_matches()
            intervals = self._empty_intervals()
            self._plot_excluded(plot_path, participant_id, phase, channel, reason)
            output_status = "excluded"

        sample_frame.to_csv(session_csv / "signal_samples.csv", index=False)
        matches.to_csv(session_csv / "peak_matches.csv", index=False)
        intervals.to_csv(session_csv / "intervals.csv", index=False)
        gap_audit.to_csv(session_csv / "gap_audit.csv", index=False)

        session_metric = {
            column: metric.get(column, np.nan) for column in self.metric_columns
        }
        session_metric.update(
            {
                "participant_id": participant_id,
                "phase": phase,
                "sensor_channel": channel,
                "status": output_status,
                "validation_status": validation_status,
                "agreement_status": str(
                    metric.get(
                        "agreement_status",
                        "reported"
                        if validation_status == "valid"
                        else "insufficient_evidence",
                    )
                ),
                "csv_directory": self._relative(session_csv),
                "plot_path": self._relative(plot_path),
                "signal_csv_path": self._relative(session_csv / "signal_samples.csv"),
                "peak_matches_csv": self._relative(session_csv / "peak_matches.csv"),
                "intervals_csv": self._relative(session_csv / "intervals.csv"),
                "gap_audit_csv": self._relative(session_csv / "gap_audit.csv"),
            }
        )
        pd.DataFrame([session_metric]).to_csv(
            session_csv / "session_metrics.csv", index=False
        )
        return session_metric

    @staticmethod
    def _validation_status(metric: dict[str, Any]) -> str:
        """Resolve the new status while accepting legacy ``status=ok`` rows."""

        value = metric.get("validation_status")
        if value is not None and str(value).strip().lower() not in {"", "nan"}:
            return (
                "valid" if str(value).strip().lower() in {"valid", "ok"} else "excluded"
            )
        return (
            "valid"
            if str(metric.get("status", "excluded")).strip().lower() == "ok"
            else "excluded"
        )

    def _load_signal_frames(
        self, participant_id: str, phase: str, channel: str
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        ecg_path = self._signal_csv(participant_id, phase, "ecg")
        ppg_path = self._signal_csv(participant_id, phase, channel)
        if ecg_path is None or ppg_path is None:
            return None
        key = (ecg_path, ppg_path)
        cached = self._signal_frame_cache.get(key)
        if cached is None:
            cached = (self._read_signal(ecg_path), self._read_signal(ppg_path))
            self._signal_frame_cache[key] = cached
        return cached

    @staticmethod
    def _read_signal(path: Path) -> pd.DataFrame:
        """Read a canonical signal source without changing the CSV output API."""

        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _signal_csv(self, participant_id: str, phase: str, channel: str) -> Path | None:
        candidates = (
            self.run_root / "silver" / participant_id / phase / f"{channel}.parquet",
            self.run_root / "silver" / participant_id / phase / f"{channel}.csv",
            self.run_root
            / "csv_export"
            / "silver"
            / participant_id
            / phase
            / f"{channel}.parquet",
            self.run_root
            / "csv_export"
            / "silver"
            / participant_id
            / phase
            / f"{channel}.csv",
        )
        return next((path for path in candidates if path.is_file()), None)

    @staticmethod
    def _timestamp(value: Any) -> pd.Timestamp:
        parsed = pd.Timestamp(value)
        return (
            parsed.tz_localize("UTC")
            if parsed.tzinfo is None
            else parsed.tz_convert("UTC")
        )

    def _crop(
        self, frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, source: str
    ) -> pd.DataFrame:
        timestamp_column = (
            "timestamp_corrected_utc"
            if "timestamp_corrected_utc" in frame
            else "timestamp_utc"
        )
        timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        cropped = frame.loc[(timestamps >= start) & (timestamps < end)].copy()
        cropped["_timestamp"] = timestamps.loc[cropped.index]
        cropped["_source"] = source
        return cropped.sort_values("_timestamp").reset_index(drop=True)

    def _load_peak_table(
        self, participant_id: str, phase: str, channel: str
    ) -> pd.DataFrame:
        path = (
            self.validation_root
            / "peak_tables"
            / participant_id
            / phase
            / f"{channel}.csv"
        )
        if not path.is_file():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _build_csv_frames(
        self,
        ecg: pd.DataFrame,
        ppg: pd.DataFrame,
        peak_table: pd.DataFrame,
        start: pd.Timestamp,
        participant_id: str,
        phase: str,
        channel: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        matches = self._normalize_peak_table(
            peak_table, participant_id, phase, channel, start
        )
        ecg_map = self._event_map(matches, "ecg_peak_timestamp_ms", "ecg")
        ppg_map = self._event_map(matches, "ppg_peak_timestamp_ms", "ppg")
        ecg_out = self._sample_rows(ecg, "ecg", start, ecg_map)
        ppg_out = self._sample_rows(ppg, channel, start, ppg_map)
        samples = pd.concat([ecg_out, ppg_out], ignore_index=True)
        samples = samples[list(EXPECTED_COLUMNS)]
        intervals = self._intervals(matches)
        return samples, matches, intervals

    def _observed_samples(
        self, ecg: pd.DataFrame, ppg: pd.DataFrame, channel: str
    ) -> pd.DataFrame:
        """Export observed Silver samples without attaching any peak labels.

        This path is used for excluded sessions.  It is intentionally limited
        to values already present in the Silver CSV; the visualizer never
        reconstructs a missing ECG/PPG waveform.
        """

        frames = [
            self._sample_rows(
                ecg,
                "ecg",
                self._frame_start(ecg),
                {},
            ),
            self._sample_rows(
                ppg,
                channel,
                self._frame_start(ppg),
                {},
            ),
        ]
        return pd.concat(frames, ignore_index=True)[list(EXPECTED_COLUMNS)]

    @staticmethod
    def _frame_start(frame: pd.DataFrame) -> pd.Timestamp:
        column = (
            "timestamp_corrected_utc"
            if "timestamp_corrected_utc" in frame.columns
            else "timestamp_utc"
        )
        timestamps = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if timestamps.notna().any():
            return timestamps.dropna().min()
        return pd.Timestamp("1970-01-01T00:00:00Z")

    def _gap_audit(
        self,
        ecg: pd.DataFrame,
        ppg: pd.DataFrame,
        peak_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Describe observed timestamp/value gaps without filling them."""

        records: list[dict[str, Any]] = []
        for source, frame in (("ecg", ecg), ("ppg", ppg)):
            if frame.empty:
                continue
            timestamp_column = (
                "timestamp_corrected_utc"
                if "timestamp_corrected_utc" in frame.columns
                else "timestamp_utc"
            )
            timestamps = pd.to_datetime(
                frame[timestamp_column], utc=True, errors="coerce"
            )
            values = pd.to_numeric(
                frame.get("clean_value", frame.get("raw_value")), errors="coerce"
            )
            valid_timestamps = timestamps.dropna().sort_values()
            if valid_timestamps.empty:
                continue
            cadence_ms = float(
                valid_timestamps.diff().dt.total_seconds().median() * 1000.0
            )
            if not np.isfinite(cadence_ms) or cadence_ms <= 0.0:
                cadence_ms = 0.0
            interpolation = (
                frame.get("interpolated_mask", pd.Series(False, index=frame.index))
                .fillna(False)
                .astype(bool)
            )
            peak_times = self._peak_times_for_source(peak_table, source)

            def append_record(
                start: pd.Timestamp,
                end: pd.Timestamp,
                duration_ms: float,
                sample_count: int,
                gap_type: str,
                action: str,
            ) -> None:
                start_ms = float(start.value / 1_000_000)
                end_ms = float(end.value / 1_000_000)
                affects = bool(
                    any(
                        abs(float(peak) - start_ms) <= max(duration_ms, cadence_ms)
                        or abs(float(peak) - end_ms) <= max(duration_ms, cadence_ms)
                        for peak in peak_times
                    )
                )
                records.append(
                    {
                        "signal_source": source,
                        "gap_start": start.isoformat(),
                        "gap_end": end.isoformat(),
                        "gap_duration_ms": duration_ms,
                        "sample_count": sample_count,
                        "gap_type": gap_type,
                        "action": action,
                        "affects_peak_region": affects,
                    }
                )

            if cadence_ms > 0.0:
                ordered = valid_timestamps.index.to_numpy()
                for left, right in zip(ordered[:-1], ordered[1:], strict=True):
                    gap_ms = float(
                        (timestamps.loc[right] - timestamps.loc[left]).total_seconds()
                        * 1000.0
                        - cadence_ms
                    )
                    if gap_ms > max(cadence_ms * 0.5, 1.0):
                        append_record(
                            timestamps.loc[left],
                            timestamps.loc[right],
                            gap_ms,
                            0,
                            "long_dropout" if gap_ms > 500.0 else "short_gap",
                            "mask_and_exclude_affected_beats"
                            if source == "ecg"
                            else "preserve_canonical_mask",
                        )

            missing = ~np.isfinite(values.to_numpy(dtype=float))
            starts, ends = self._finite_runs(missing)
            for start_index, end_index in zip(starts, ends, strict=True):
                if len(timestamps) == 0:
                    continue
                start_index = int(start_index)
                end_index = int(end_index)
                if start_index >= len(timestamps) or end_index <= 0:
                    continue
                left = max(0, start_index - 1)
                right = min(len(timestamps) - 1, end_index)
                start_time = timestamps.iloc[left]
                end_time = timestamps.iloc[right]
                duration = max(
                    0.0,
                    (end_time - start_time).total_seconds() * 1000.0 - cadence_ms,
                )
                append_record(
                    start_time,
                    end_time,
                    duration,
                    end_index - start_index,
                    "long_dropout" if duration > 500.0 else "short_gap",
                    "mask_and_exclude_affected_beats"
                    if source == "ecg"
                    else "preserve_canonical_mask",
                )

            for index in np.flatnonzero(interpolation.to_numpy(dtype=bool)):
                if index >= len(timestamps):
                    continue
                append_record(
                    timestamps.iloc[index],
                    timestamps.iloc[index],
                    0.0,
                    1,
                    "canonical_interpolated_sample",
                    "mask_and_exclude_affected_beats"
                    if source == "ecg"
                    else "preserve_canonical_mask",
                )

        return pd.DataFrame(records, columns=list(GAP_COLUMNS))

    @staticmethod
    def _peak_times_for_source(peak_table: pd.DataFrame, source: str) -> list[float]:
        column = f"{source}_peak_timestamp_ms"
        if peak_table.empty or column not in peak_table.columns:
            return []
        values = pd.to_numeric(peak_table[column], errors="coerce").dropna()
        return values.to_numpy(dtype=float).tolist()

    @staticmethod
    def _finite_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
        changes = np.diff(padded.astype(np.int8))
        return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)

    def _normalize_peak_table(
        self,
        peak_table: pd.DataFrame,
        participant_id: str,
        phase: str,
        channel: str,
        start: pd.Timestamp,
    ) -> pd.DataFrame:
        columns = [
            "participant_id",
            "phase",
            "sensor_channel",
            "match_id",
            "ecg_peak_index",
            "ppg_peak_index",
            "ecg_peak_timestamp_ms",
            "ppg_peak_timestamp_ms",
            "pat_ms",
            "matched",
            "peak_role",
            "ecg_elapsed_seconds",
            "ppg_elapsed_seconds",
        ]
        if peak_table.empty:
            return pd.DataFrame(columns=columns)
        table = peak_table.copy()
        if "matched" not in table:
            table["matched"] = False
        if "peak_role" not in table:
            table["peak_role"] = np.where(
                table["matched"].astype(bool), "matched", "ecg_unmatched"
            )
        table["participant_id"] = participant_id
        table["phase"] = phase
        table["sensor_channel"] = channel
        table["match_id"] = np.arange(len(table), dtype=int)
        for index_column in ("ecg_peak_index", "ppg_peak_index"):
            if index_column not in table:
                table[index_column] = np.nan
            table[index_column] = pd.to_numeric(table[index_column], errors="coerce")
        for timestamp_column in (
            "ecg_peak_timestamp_ms",
            "ppg_peak_timestamp_ms",
            "pat_ms",
        ):
            if timestamp_column not in table:
                table[timestamp_column] = np.nan
            table[timestamp_column] = pd.to_numeric(
                table[timestamp_column], errors="coerce"
            )
        table["matched"] = table["matched"].fillna(False).astype(bool)
        for signal in ("ecg", "ppg"):
            timestamp = self._peak_timestamp_series(
                table[f"{signal}_peak_timestamp_ms"]
            )
            table[f"{signal}_elapsed_seconds"] = (timestamp - start).dt.total_seconds()
        return table[columns]

    @staticmethod
    def _peak_timestamp_series(values: pd.Series) -> pd.Series:
        """Convert finite epoch-millisecond peaks without NaN overflow."""
        numeric = pd.to_numeric(values, errors="coerce")
        timestamps = pd.Series(pd.NaT, index=numeric.index, dtype="datetime64[ns, UTC]")
        finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        if finite.any():
            finite_values = numeric.to_numpy(dtype=float, na_value=np.nan)[finite]
            with np.errstate(over="ignore", invalid="ignore"):
                timestamps.iloc[np.flatnonzero(finite)] = pd.to_datetime(
                    np.rint(finite_values).astype("int64"),
                    unit="ms",
                    utc=True,
                    errors="coerce",
                )
        return timestamps

    @staticmethod
    def _event_map(
        table: pd.DataFrame, column: str, source: str
    ) -> dict[float, dict[str, Any]]:
        result: dict[float, dict[str, Any]] = {}
        for row in table.itertuples(index=False):
            timestamp = getattr(row, column)
            if pd.isna(timestamp):
                continue
            result[float(timestamp)] = {
                "match_id": getattr(row, "match_id"),
                "matched": bool(getattr(row, "matched", False)),
                "pat_ms": getattr(row, "pat_ms") if source == "ppg" else np.nan,
            }
        return result

    def _sample_rows(
        self,
        frame: pd.DataFrame,
        source: str,
        start: pd.Timestamp,
        event_map: dict[float, dict[str, Any]],
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=EXPECTED_COLUMNS)
        timestamp_column = (
            "timestamp_corrected_utc"
            if "timestamp_corrected_utc" in frame
            else "timestamp_utc"
        )
        corrected = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        raw_timestamp = frame.get("timestamp_raw_utc", frame[timestamp_column])
        clean = pd.to_numeric(
            frame.get("clean_value", frame.get("raw_value")), errors="coerce"
        )
        raw = pd.to_numeric(frame.get("raw_value", clean), errors="coerce")
        normalized = self._robust_z(clean.to_numpy(dtype=float))
        output = pd.DataFrame(
            {
                "signal_source": source,
                "elapsed_seconds": (corrected - start).dt.total_seconds(),
                "timestamp_raw": raw_timestamp.astype(str),
                "timestamp_corrected": corrected.astype(str),
                "raw_value": raw,
                "clean_value": clean,
                "normalized_clean_value": normalized,
                "interpolated_mask": frame.get(
                    "interpolated_mask", pd.Series(False, index=frame.index)
                )
                .fillna(False)
                .astype(bool),
                "peak_flag": False,
                "matched_peak_id": np.nan,
                "pat_ms": np.nan,
            }
        )
        timestamps_ms = corrected.astype("int64") / 1_000_000.0
        if event_map:
            event_timestamps = np.asarray(sorted(event_map), dtype=float)
            sample_timestamps = timestamps_ms.to_numpy(dtype=float)
            insertion = np.searchsorted(event_timestamps, sample_timestamps)
            right = np.clip(insertion, 0, len(event_timestamps) - 1)
            left = np.clip(insertion - 1, 0, len(event_timestamps) - 1)
            right_error = np.abs(event_timestamps[right] - sample_timestamps)
            left_error = np.abs(event_timestamps[left] - sample_timestamps)
            nearest = np.where(left_error <= right_error, left, right)
            matched = np.minimum(left_error, right_error) <= 0.5
            for row_position in np.flatnonzero(matched):
                event = event_map[float(event_timestamps[nearest[row_position]])]
                output.iloc[row_position, output.columns.get_loc("peak_flag")] = True
                if event["matched"]:
                    output.iloc[
                        row_position, output.columns.get_loc("matched_peak_id")
                    ] = event["match_id"]
                    output.iloc[row_position, output.columns.get_loc("pat_ms")] = event[
                        "pat_ms"
                    ]
        return output

    @staticmethod
    def _robust_z(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        center = float(np.nanmedian(values))
        scale = float(np.nanmedian(np.abs(values - center)) * 1.4826)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.nanstd(values))
        return (values - center) / (
            scale if np.isfinite(scale) and scale > 0.0 else 1.0
        )

    @staticmethod
    def _intervals(matches: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "sequence_index",
            "interval_source",
            "ecg_rr_ms",
            "ppg_ppi_ms",
            "pat_ms",
            "ecg_rr_valid",
            "ppg_ppi_valid",
        ]
        if matches.empty:
            return pd.DataFrame(columns=columns)
        matched = matches.loc[matches["matched"]].copy()
        required_index_columns = {"ecg_peak_index", "ppg_peak_index"}
        if not required_index_columns.issubset(matched.columns):
            return pd.DataFrame(columns=columns)
        matched = matched.dropna(subset=list(required_index_columns)).sort_values(
            "ecg_peak_index"
        )
        if len(matched) < 2:
            return pd.DataFrame(columns=columns)
        ecg_indices = matched["ecg_peak_index"].to_numpy(dtype=int)
        ppg_indices = matched["ppg_peak_index"].to_numpy(dtype=int)
        adjacent = (np.diff(ecg_indices) == 1) & (np.diff(ppg_indices) == 1)
        if not adjacent.any():
            return pd.DataFrame(columns=columns)
        ecg = matched["ecg_peak_timestamp_ms"].to_numpy(dtype=float)
        ppg = matched["ppg_peak_timestamp_ms"].to_numpy(dtype=float)
        ecg_rr = np.diff(ecg)
        ppg_ppi = np.diff(ppg)
        output = pd.DataFrame(
            {
                "sequence_index": np.arange(len(ecg_rr), dtype=int),
                "interval_source": "observed",
                "ecg_rr_ms": ecg_rr,
                "ppg_ppi_ms": ppg_ppi,
                "pat_ms": matched["pat_ms"].to_numpy(dtype=float)[1:],
            }
        )
        output = output.loc[adjacent].reset_index(drop=True)
        output["sequence_index"] = np.arange(len(output), dtype=int)
        output["ecg_rr_valid"] = output["ecg_rr_ms"].between(300.0, 1500.0)
        output["ppg_ppi_valid"] = output["ppg_ppi_ms"].between(300.0, 1500.0)
        return output[columns]

    @staticmethod
    def _empty_matches() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "phase",
                "sensor_channel",
                "match_id",
                "ecg_peak_timestamp_ms",
                "ppg_peak_timestamp_ms",
                "pat_ms",
                "matched",
                "peak_role",
                "ecg_elapsed_seconds",
                "ppg_elapsed_seconds",
            ]
        )

    @staticmethod
    def _empty_intervals() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "sequence_index",
                "interval_source",
                "ecg_rr_ms",
                "ppg_ppi_ms",
                "pat_ms",
                "ecg_rr_valid",
                "ppg_ppi_valid",
            ]
        )

    @staticmethod
    def _empty_samples() -> pd.DataFrame:
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS))

    @staticmethod
    def _empty_gap_audit() -> pd.DataFrame:
        return pd.DataFrame(columns=list(GAP_COLUMNS))

    def _plot_valid(
        self,
        path: Path,
        ecg: pd.DataFrame,
        ppg: pd.DataFrame,
        matches: pd.DataFrame,
        intervals: pd.DataFrame,
        metric: dict[str, Any],
        participant_id: str,
        phase: str,
        channel: str,
        start: pd.Timestamp,
    ) -> None:
        figure, axes = plt.subplots(6, 1, figsize=(18, 22), constrained_layout=True)
        ecg_t, ecg_values = self._plot_signal(
            axes[0], ecg, start, "ECG clean", "tab:blue"
        )
        ppg_t, ppg_values = self._plot_signal(
            axes[1], ppg, start, f"{channel} clean", "tab:orange"
        )
        axes[0].set_title("ECG corrected timeline with detected R-peaks")
        axes[1].set_title("PPG clean timeline with detected systolic peaks")
        self._plot_peaks(axes[0], matches, "ecg", "tab:red", start, ecg_t, ecg_values)
        self._plot_peaks(axes[1], matches, "ppg", "tab:green", start, ppg_t, ppg_values)
        axes[0].set_ylabel("ECG clean")
        axes[1].set_ylabel("PPG clean")
        self._plot_overlay(axes[2], ecg, ppg, matches, start, channel)
        axes[2].set_title(
            "Normalized ECG–PPG overlay; lines show one-to-one matches and PAT"
        )
        self._plot_intervals(axes[3], intervals)
        axes[3].set_title("Matched RR versus PPI")
        self._plot_distributions(axes[4], intervals)
        axes[4].set_title("Physiological interval distributions")
        self._plot_poincare(axes[5], intervals)
        axes[5].set_title("Poincaré plot of matched intervals")
        for axis in axes[:4]:
            axis.set_xlim(0.0, 240.0)
            axis.grid(alpha=0.25)
        figure.suptitle(
            f"PPG–ECG Peak-to-Peak — {participant_id} ({phase}, {channel}) | "
            f"ECG peaks={metric.get('ecg_peak_count', 'n/a')} | "
            f"PPG peaks={metric.get('ppg_peak_count', 'n/a')} | "
            f"F1={self._number(metric.get('beat_f1'))} | "
            f"PAT median={self._number(metric.get('pat_median_ms'))} ms",
            fontsize=16,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=140)
        plt.close(figure)

    @staticmethod
    def _plot_signal(
        axis: Any, frame: pd.DataFrame, start: pd.Timestamp, label: str, color: str
    ) -> tuple[np.ndarray, np.ndarray]:
        timestamps = pd.to_datetime(frame["_timestamp"], utc=True)
        time = (timestamps - start).dt.total_seconds().to_numpy(dtype=float)
        values = pd.to_numeric(frame.get("clean_value"), errors="coerce").to_numpy(
            dtype=float
        )
        stride = max(1, len(time) // 12000)
        axis.plot(
            time[::stride], values[::stride], color=color, linewidth=0.55, label=label
        )
        axis.legend(loc="upper right")
        return time, values

    @staticmethod
    def _plot_peaks(
        axis: Any,
        matches: pd.DataFrame,
        signal: str,
        color: str,
        start: pd.Timestamp,
        time: np.ndarray,
        values: np.ndarray,
    ) -> None:
        column = f"{signal}_peak_timestamp_ms"
        peak_timestamps = pd.to_numeric(
            matches.get(column, pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        if peak_timestamps.empty:
            return
        timestamps = pd.to_datetime(peak_timestamps, unit="ms", utc=True)
        peak_time = np.asarray(
            [(timestamp - start).total_seconds() for timestamp in timestamps],
            dtype=float,
        )
        peak_values = np.interp(peak_time, time, values)
        axis.scatter(
            peak_time,
            peak_values,
            color=color,
            s=12,
            zorder=3,
            label=f"{signal.upper()} peaks",
        )

    def _plot_overlay(
        self,
        axis: Any,
        ecg: pd.DataFrame,
        ppg: pd.DataFrame,
        matches: pd.DataFrame,
        start: pd.Timestamp,
        channel: str,
    ) -> None:
        ecg_t = (
            (pd.to_datetime(ecg["_timestamp"], utc=True) - start)
            .dt.total_seconds()
            .to_numpy(dtype=float)
        )
        ppg_t = (
            (pd.to_datetime(ppg["_timestamp"], utc=True) - start)
            .dt.total_seconds()
            .to_numpy(dtype=float)
        )
        ecg_z = self._robust_z(
            pd.to_numeric(ecg["clean_value"], errors="coerce").to_numpy(dtype=float)
        )
        ppg_z = self._robust_z(
            pd.to_numeric(ppg["clean_value"], errors="coerce").to_numpy(dtype=float)
        )
        axis.plot(
            ecg_t[:: max(1, len(ecg_t) // 12000)],
            ecg_z[:: max(1, len(ecg_z) // 12000)],
            label="ECG normalized",
            color="tab:blue",
            linewidth=0.5,
        )
        axis.plot(
            ppg_t[:: max(1, len(ppg_t) // 12000)],
            ppg_z[:: max(1, len(ppg_z) // 12000)],
            label=f"{channel} normalized",
            color="tab:orange",
            linewidth=0.5,
        )
        matched = matches.loc[matches["matched"]]
        for row in matched.itertuples(index=False):
            ecg_time = float(row.ecg_elapsed_seconds)
            ppg_time = float(row.ppg_elapsed_seconds)
            ecg_value = float(np.interp(ecg_time, ecg_t, ecg_z))
            ppg_value = float(np.interp(ppg_time, ppg_t, ppg_z))
            axis.plot(
                [ecg_time, ppg_time],
                [ecg_value, ppg_value],
                color="tab:green",
                alpha=0.22,
                linewidth=0.65,
            )
        axis.legend(loc="upper right")
        axis.set_ylabel("robust z-score")

    @staticmethod
    def _plot_intervals(axis: Any, intervals: pd.DataFrame) -> None:
        if intervals.empty:
            return
        axis.plot(
            intervals["sequence_index"],
            intervals["ecg_rr_ms"],
            label="ECG RR",
            color="tab:blue",
            linewidth=0.8,
        )
        axis.plot(
            intervals["sequence_index"],
            intervals["ppg_ppi_ms"],
            label="PPG PPI",
            color="tab:orange",
            linewidth=0.8,
        )
        axis.axhspan(
            300, 1500, color="tab:green", alpha=0.06, label="physiological range"
        )
        axis.set_ylabel("ms")
        axis.set_xlabel("matched beat sequence")
        axis.legend(loc="upper right")

    @staticmethod
    def _plot_distributions(axis: Any, intervals: pd.DataFrame) -> None:
        if intervals.empty:
            return
        for column, label, color in (
            ("ecg_rr_ms", "ECG RR", "tab:blue"),
            ("ppg_ppi_ms", "PPG PPI", "tab:orange"),
        ):
            values = intervals.loc[intervals[column].between(300, 1500), column]
            if not values.empty:
                axis.hist(
                    values,
                    bins=min(30, max(8, len(values) // 3)),
                    alpha=0.55,
                    label=label,
                    color=color,
                )
        axis.set_xlabel("interval (ms)")
        axis.set_ylabel("count")
        axis.legend(loc="upper right")

    @staticmethod
    def _plot_poincare(axis: Any, intervals: pd.DataFrame) -> None:
        if intervals.empty:
            return
        for column, label, color in (
            ("ecg_rr_ms", "ECG RR", "tab:blue"),
            ("ppg_ppi_ms", "PPG PPI", "tab:orange"),
        ):
            values = intervals.loc[
                intervals[column].between(300, 1500), column
            ].to_numpy(dtype=float)
            if len(values) > 1:
                axis.scatter(
                    values[:-1], values[1:], s=10, alpha=0.65, label=label, color=color
                )
        axis.set_xlabel("interval n (ms)")
        axis.set_ylabel("interval n+1 (ms)")
        axis.legend(loc="upper left")

    @staticmethod
    def _plot_excluded(
        path: Path, participant_id: str, phase: str, channel: str, reason: str
    ) -> None:
        figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.65,
            f"PPG–ECG Peak-to-Peak — {participant_id} ({phase}, {channel})",
            ha="center",
            fontsize=15,
            fontweight="bold",
        )
        axis.text(
            0.5,
            0.45,
            "NO_VALID_240S_COMPARISON",
            ha="center",
            fontsize=13,
            color="tab:red",
        )
        axis.text(
            0.5,
            0.25,
            f"Exclusion reason: {reason}",
            ha="center",
            fontsize=10,
            wrap=True,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=140)
        plt.close(figure)

    def _write_summary(self, manifest: pd.DataFrame) -> None:
        summary = {
            "validation_window_seconds": 240,
            "expected_combinations": 192,
            "observed_combinations": int(len(manifest)),
            "valid_combinations": int((manifest["status"] == "ok").sum()),
            "excluded_combinations": int((manifest["status"] != "ok").sum()),
            "channels": list(PPG_CHANNELS),
            "phases": list(PHASES),
            "participants": list(VALID_PARTICIPANTS),
            "normalization": "robust_zscore_for_visualization_only",
            "ppg_timestamp_shifted": False,
            "synthetic_signal_created": False,
        }
        (self.output_dir / "validation_summary_240s.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def _write_report(self, manifest: pd.DataFrame) -> None:
        report_path = (
            self.project_root
            / "reports"
            / "runs"
            / self.run_root.name
            / "ppg_ecg_peak_to_peak_report.md"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        valid = int((manifest["status"] == "ok").sum())
        excluded = int((manifest["status"] != "ok").sum())

        channel_summary = (
            manifest.assign(
                beat_f1=pd.to_numeric(manifest["beat_f1"], errors="coerce"),
                ibi_mae_ms=pd.to_numeric(manifest["ibi_mae_ms"], errors="coerce"),
                hr_mae_bpm=pd.to_numeric(manifest["hr_mae_bpm"], errors="coerce"),
            )
            .groupby("sensor_channel", as_index=False)
            .agg(
                combinations=("sensor_channel", "size"),
                valid=(
                    "validation_status",
                    lambda values: int((values == "valid").sum()),
                ),
                excluded=(
                    "validation_status",
                    lambda values: int((values != "valid").sum()),
                ),
                mean_f1=("beat_f1", "mean"),
                median_f1=("beat_f1", "median"),
                median_ibi_mae_ms=("ibi_mae_ms", "median"),
                median_hr_mae_bpm=("hr_mae_bpm", "median"),
            )
        )

        session_summary_path = (
            self.validation_root / "participant_phase_summary_240s.csv"
        )
        if session_summary_path.is_file():
            session_summary = pd.read_csv(session_summary_path).loc[
                :,
                [
                    "participant_id",
                    "phase",
                    "session_validation_status",
                    "valid_channel_count",
                    "excluded_channel_count",
                    "median_beat_f1",
                    "median_ibi_mae_ms",
                    "median_hr_mae_bpm",
                    "sync_alpha",
                    "sync_beta_seconds",
                    "exclusion_reasons",
                ],
            ]
        else:
            session_summary = pd.DataFrame()

        exclusion_path = self.validation_root / "exclusion_breakdown_240s.csv"
        exclusion_summary = (
            pd.read_csv(exclusion_path) if exclusion_path.is_file() else pd.DataFrame()
        )

        baseline_path = self.validation_root / "baseline_vs_rebuild_240s.csv"
        baseline = (
            pd.read_csv(baseline_path) if baseline_path.is_file() else pd.DataFrame()
        )
        baseline_overview = pd.DataFrame()
        if not baseline.empty:
            baseline_overview = pd.DataFrame(
                [
                    {
                        "comparison_rows": len(baseline),
                        "baseline_available_rows": int(
                            baseline["baseline_available"].fillna(False).sum()
                        ),
                        "rebuild_valid_rows": int(
                            (baseline["rebuild_validation_status"] == "valid").sum()
                        ),
                        "f1_delta_mean": pd.to_numeric(
                            baseline["delta_beat_f1"], errors="coerce"
                        ).mean(),
                        "ibi_mae_delta_median_ms": pd.to_numeric(
                            baseline["delta_ibi_mae_ms"], errors="coerce"
                        ).median(),
                    }
                ]
            )

        sync_path = self.validation_root / "sync_parameters_240s.csv"
        sync = pd.read_csv(sync_path) if sync_path.is_file() else pd.DataFrame()
        sync_overview = pd.DataFrame()
        if not sync.empty:
            clock_excluded = (
                sync.get("clock_excluded", pd.Series(False, index=sync.index))
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
            )
            sync_overview = pd.DataFrame(
                [
                    {
                        "sync_rows": len(sync),
                        "clock_excluded_rows": int(clock_excluded.sum()),
                        "ecg_interpolated_samples": int(
                            pd.to_numeric(
                                manifest["ecg_interpolated_sample_count"],
                                errors="coerce",
                            )
                            .fillna(0)
                            .sum()
                        ),
                        "ppg_timestamp_shift": "none",
                        "rr_interpolation": "disabled",
                        "pat_min_ms": pd.to_numeric(
                            manifest["pat_median_ms"], errors="coerce"
                        ).min(),
                        "pat_max_ms": pd.to_numeric(
                            manifest["pat_median_ms"], errors="coerce"
                        ).max(),
                    }
                ]
            )

        lines = [
            f"# PPG–ECG Peak-to-Peak Report — {self.run_root.name}",
            "",
            "The legacy reference is PPG-only; this report adds ECG R-peaks, corrected ECG timestamps, and PAT-preserving matching.",
            "",
            "- Validation duration: **240 seconds** per session after 30-second edge trimming.",
            "- Expected combinations: **192**.",
            f"- Valid combinations: **{valid}**.",
            f"- Excluded combinations: **{excluded}**.",
            "- ECG timestamp correction is consumed before matching; PPG timestamps are unchanged.",
            "- Normalized values are visualization-only; peak detection uses canonical clean values.",
            "- No padding, synthetic waveform, or fake peak was created.",
            "",
            "## Validation status and evidence",
            "",
            "`validation_status` reports coverage/peak/match sufficiency. `agreement_status` reports whether ECG–PPG agreement metrics are reportable; a low F1 is retained as evidence and is not repaired by interpolation or time warping.",
            "",
            *self._markdown_table(channel_summary),
            "",
            "### Synchronization and gap policy",
            "",
            *self._markdown_table(sync_overview),
            "",
            "ECG waveform interpolation is disabled for primary validation. ECG gaps are masked, affected peaks are excluded, and PPG timestamps/values are not shifted.",
            "",
            "### Participant–phase summary (64 rows)",
            "",
            *self._markdown_table(session_summary),
            "",
            "### Exclusion breakdown",
            "",
            *self._markdown_table(exclusion_summary),
            "",
            "### Baseline versus rebuild",
            "",
            *self._markdown_table(baseline_overview),
            "",
            "The full baseline comparison remains in the CSV artifact; it is descriptive and does not use reused internal test data for detector selection.",
            "",
            "## Output folders",
            "",
            f"- CSV bundles: `{self._relative(self.csv_dir)}`",
            f"- Plots: `{self._relative(self.plot_dir)}`",
            f"- Canonical metrics CSV: `{self._relative(self.validation_root / 'ppg_ecg_metrics_240s.csv')}`",
            f"- CSV visual metrics: `{self._relative(self.output_dir / 'ppg_ecg_metrics_240s.csv')}`",
            f"- Canonical sync CSV: `{self._relative(self.validation_root / 'sync_parameters_240s.csv')}`",
            f"- Manifest: `{self._relative(self.output_dir / 'csv_visual_manifest.csv')}`",
            "",
            "ECG remains a reference/cross-validation signal and is not a predictor for PPG stress training. PPG Gold/training readiness is evaluated independently from ECG validation readiness.",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _markdown_table(frame: pd.DataFrame) -> list[str]:
        """Render a bounded DataFrame as dependency-free Markdown."""

        if frame.empty:
            return ["_No rows available._"]
        values = frame.copy().fillna("")
        headers = [str(column) for column in values.columns]

        def format_value(value: Any) -> str:
            if value == "":
                return "—"
            if isinstance(value, (float, np.floating)):
                return "—" if not np.isfinite(value) else f"{value:.3f}"
            return str(value).replace("|", "\\|").replace("\n", " ")

        rows = [
            [format_value(value) for value in row]
            for row in values.itertuples(index=False, name=None)
        ]
        divider = ["---"] * len(headers)
        return [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(divider) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]

    def _relative(self, path: Path) -> str:
        candidate = path if path.is_absolute() else self.project_root / path
        try:
            return str(
                candidate.resolve().relative_to(self.project_root.resolve())
            ).replace("\\", "/")
        except ValueError:
            return str(candidate).replace("\\", "/")

    @staticmethod
    def _number(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not np.isfinite(number) else f"{number:.3f}"


def build_csv_peak_visual_report(
    run_root: Path, project_root: Path | None = None
) -> CsvVisualOutputs:
    """Public boundary for the CSV-first visualization stage."""

    return CsvPeakToPeakVisualizer(run_root, project_root).run()
