from __future__ import annotations

import json
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import CANONICAL_SIGNAL_COLUMNS
from .settings import PipelinePaths, ensure_dirs, pipeline_paths_from_config

logger = logging.getLogger(__name__)

_SILVER_EDA_SAMPLE_ROWS = 1_000
_EDA_SIGNAL_COLUMNS = (
    "participant_id",
    "session_id",
    "recording_id",
    "phase",
    "sensor_channel",
    "timestamp_raw_utc",
    "timestamp_corrected_utc",
    "timestamp_local",
    "sampling_rate_hz",
    "raw_value",
    "clean_value",
    "quality_score",
    "quality_tier",
    "quality_flags",
    "exclusion_reason",
    "interpolated_mask",
    "motion_artifact_mask",
    "motion_heavy_mask",
    "motion_score",
    "madc_action",
)


class WranglingEDA:
    def __init__(self, paths: PipelinePaths, config: Any | None = None) -> None:
        self.paths = paths
        self._input_sampling_note = ""
        window = getattr(config, "window", None)
        self._source_phase_seconds = int(
            getattr(window, "source_phase_duration_seconds", 300)
        )
        self._edge_trim_start_seconds = int(
            getattr(window, "edge_trim_start_seconds", 30)
        )
        self._edge_trim_end_seconds = int(getattr(window, "edge_trim_end_seconds", 30))
        self._analysis_phase_seconds = int(
            getattr(window, "analysis_phase_duration_seconds", 240)
        )
        features = getattr(config, "features", None)
        self._predictors = tuple(getattr(features, "predictor_default", ()))

    def run(self, stage: str = "silver") -> dict:
        """Generate deterministic EDA for one canonical pipeline stage.

        ``stage`` is one of ``raw``, ``silver``, or ``gold``.  Raw EDA reads
        immutable Bronze rows before cleaning; Silver and Gold are analyzed
        after their respective gates.  The time-series profile follows the
        YData Profiling time-series contract:
        https://docs.profiling.ydata.ai/latest/features/time_series_datasets/
        """

        if stage not in {"raw", "silver", "gold"}:
            raise ValueError(f"unsupported EDA stage: {stage}")
        self._input_sampling_note = ""
        bronze = self._read_bronze() if stage == "raw" else pd.DataFrame()
        silver = self._read_silver() if stage == "silver" else pd.DataFrame()
        gold = self._read_gold() if stage == "gold" else pd.DataFrame()
        source = bronze if stage == "raw" else silver if stage == "silver" else gold
        structural = (
            self._read_structural_summary(stage)
            if stage in {"raw", "silver"}
            else pd.DataFrame()
        )

        for df in [source]:
            if not df.empty:
                df.rename(
                    columns={
                        "participant_id": "participant_code",
                        "sensor_channel": "signal_type",
                        "timestamp_corrected_utc": "timestamp_utc",
                    },
                    inplace=True,
                )

        for col in ["timestamp_utc", "timestamp_local", "anchor_start", "anchor_end"]:
            if col in bronze.columns:
                bronze[col] = pd.to_datetime(bronze[col], errors="coerce", utc=True)
            if col in silver.columns:
                silver[col] = pd.to_datetime(silver[col], errors="coerce", utc=True)

        value_column = "clean_value" if stage == "silver" else "raw_value"
        counts = self._structural_table(structural, "counts")
        duration = self._structural_table(structural, "duration")
        interval = self._structural_table(structural, "interval")
        missing = self._structural_table(structural, "missing")
        amplitude = self._structural_table(structural, "amplitude")
        coverage = self._structural_coverage(structural)
        if structural.empty:
            counts = self._sample_counts(source)
            duration = self._duration_consistency(source, value_column)
            interval = self._sampling_intervals(source)
            missing = self._missingness_discontinuity(source, value_column)
            amplitude = self._amplitude_outliers(source, value_column)
            coverage = self._phase_coverage(source)
        label_balance = self._label_balance(gold)
        correlation = self._cross_channel_correlation(source, value_column)
        edge_trim_impact = self._edge_trim_impact(duration, stage)

        prefix = f"{stage}_"
        counts.to_csv(self.paths.eda_dir / f"{prefix}sample_counts.csv", index=False)
        duration.to_csv(
            self.paths.eda_dir / f"{prefix}duration_consistency.csv", index=False
        )
        interval.to_csv(
            self.paths.eda_dir / f"{prefix}sampling_interval_summary.csv", index=False
        )
        missing.to_csv(
            self.paths.eda_dir / f"{prefix}missingness_discontinuity.csv", index=False
        )
        amplitude.to_csv(
            self.paths.eda_dir / f"{prefix}amplitude_outliers.csv", index=False
        )
        coverage.to_csv(self.paths.eda_dir / f"{prefix}phase_coverage.csv", index=False)
        label_balance.to_csv(
            self.paths.eda_dir / f"{prefix}label_balance.csv", index=False
        )
        correlation.to_csv(
            self.paths.eda_dir / f"{prefix}cross_channel_correlation.csv", index=False
        )
        edge_trim_impact.to_csv(
            self.paths.eda_dir / f"{prefix}edge_trim_impact.csv", index=False
        )
        quality_policy = (
            self._quality_policy_summary(gold) if stage == "gold" else pd.DataFrame()
        )
        if stage == "gold":
            quality_policy.to_csv(
                self.paths.eda_dir / "gold_quality_policy_summary.csv", index=False
            )

        self._plot_label_balance(label_balance, prefix)
        self._plot_sampling_interval(interval, prefix)
        if stage in {"raw", "silver"}:
            self._plot_overlay_example(source, prefix)
            self._plot_raw_vs_filtered_with_peaks(source, prefix)
        self._generate_ydata_profiling(source, stage)

        report = self._build_report(
            counts=counts,
            duration=duration,
            interval=interval,
            missing=missing,
            amplitude=amplitude,
            coverage=coverage,
            label_balance=label_balance,
            correlation=correlation,
            quality_policy=quality_policy,
            stage=stage,
        )
        report_path = self.paths.eda_dir / (
            "eda_report.md" if stage == "silver" else f"eda_report_{stage}.md"
        )
        report_path.write_text(report, encoding="utf-8")
        if stage == "silver":
            # ``eda_report.md`` predates the explicit Raw/Silver/Gold naming
            # convention. Keep it for consumers while giving Silver a
            # stage-qualified report alongside Raw and Gold.
            (self.paths.eda_dir / "eda_report_silver.md").write_text(
                report, encoding="utf-8"
            )
        self._write_data_dictionary()
        if stage == "gold":
            self._write_stage_comparison(gold)
        if stage == "silver":
            self._write_silver_compatibility_aliases()
        manifest_path = self._write_stage_manifest(stage)

        logger.info("EDA report generated at %s", self.paths.eda_dir)
        return {
            "counts_rows": len(counts),
            "duration_rows": len(duration),
            "coverage_rows": len(coverage),
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
        }

    @staticmethod
    def _read_legacy_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def _read_bronze(self) -> pd.DataFrame:
        parts = [
            self._read_sampled_signal_parquet(p)
            for p in sorted(self.paths.bronze_dir.rglob("*.parquet"))
            if any(part.startswith("ID-") for part in p.parts)
        ]
        if parts:
            self._input_sampling_note = (
                "- Raw waveform EDA is a deterministic head/tail sample of at most "
                f"{_SILVER_EDA_SAMPLE_ROWS} rows per participant-phase-channel file; "
                "full raw counts and issue totals remain in the immutable audit manifests."
            )
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _read_silver(self) -> pd.DataFrame:
        parts = [
            self._read_sampled_signal_parquet(p)
            for p in sorted(self.paths.silver_dir.rglob("*.parquet"))
            if any(part.startswith("ID-") for part in p.parts)
        ]
        if parts:
            self._input_sampling_note = (
                "- Waveform EDA is a deterministic head/tail sample of at most "
                f"{_SILVER_EDA_SAMPLE_ROWS} rows per participant-phase-channel file; "
                "full quality and lineage totals remain in Silver manifests."
            )
            return pd.concat(parts, ignore_index=True)
        legacy = self.paths.silver_dir / "canonical_segmented_signals.csv"
        return (
            pd.read_csv(legacy)
            if legacy.exists()
            else pd.DataFrame(columns=CANONICAL_SIGNAL_COLUMNS)
        )

    @staticmethod
    def _read_sampled_signal_parquet(path: Path) -> pd.DataFrame:
        """Read only EDA columns and a bounded waveform sample.

        This bounded inspection prevents an EDA visualization from exhausting
        local memory on multi-million-row Silver data. It is not preprocessing
        and never writes, modifies, or fabricates signal samples.
        """

        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema.names)
        columns = [column for column in _EDA_SIGNAL_COLUMNS if column in available]
        row_count = parquet_file.metadata.num_rows
        if row_count <= _SILVER_EDA_SAMPLE_ROWS:
            return pd.read_parquet(path, columns=columns)

        # Reading the complete Silver file and sampling afterwards defeats the
        # memory bound. Read only bounded head/tail portions from Arrow row
        # groups. These are real observed rows; no waveform is reconstructed.
        head_count = _SILVER_EDA_SAMPLE_ROWS // 2
        head_batches = parquet_file.iter_batches(batch_size=head_count, columns=columns)
        head = next(head_batches).to_pandas()
        last_group = parquet_file.num_row_groups - 1
        tail_group = parquet_file.read_row_group(
            last_group, columns=columns
        ).to_pandas()
        tail = tail_group.tail(_SILVER_EDA_SAMPLE_ROWS - len(head))
        return pd.concat([head, tail], ignore_index=True)

    def _signal_paths(self, stage: str) -> list[Path]:
        root = self.paths.bronze_dir if stage == "raw" else self.paths.silver_dir
        return [
            path
            for path in sorted(root.rglob("*.parquet"))
            if any(part.startswith("ID-") for part in path.parts)
        ]

    def _read_structural_summary(self, stage: str) -> pd.DataFrame:
        rows = [
            self._summarize_signal_parquet(path, stage)
            for path in self._signal_paths(stage)
        ]
        return pd.DataFrame(row for row in rows if row)

    @staticmethod
    def _summarize_signal_parquet(path: Path, stage: str) -> dict[str, Any]:
        """Summarize every observed row while keeping waveform previews bounded."""

        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema.names)
        value_column = "clean_value" if stage == "silver" else "raw_value"
        timestamp_column = (
            "timestamp_corrected_utc"
            if "timestamp_corrected_utc" in available
            else "timestamp_raw_utc"
        )
        required = {
            "participant_id",
            "session_id",
            "phase",
            "sensor_channel",
            "sampling_rate_hz",
            timestamp_column,
            value_column,
        }
        if not required.issubset(available):
            return {}
        optional_masks = [
            column
            for column in (
                "interpolated_mask",
                "motion_artifact_mask",
                "motion_heavy_mask",
            )
            if column in available
        ]
        columns = [*sorted(required), *optional_masks]
        frame = pd.read_parquet(path, columns=columns)
        if frame.empty:
            return {}
        timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
        timestamps = timestamps.dropna().sort_values()
        intervals_ms = timestamps.diff().dt.total_seconds().mul(1000.0).dropna()
        positive_intervals = intervals_ms[intervals_ms > 0.0]
        median_interval = (
            float(positive_intervals.median())
            if not positive_intervals.empty
            else np.nan
        )
        values = pd.to_numeric(frame[value_column], errors="coerce")
        finite = values[np.isfinite(values)]
        sampling_rate = float(
            pd.to_numeric(frame["sampling_rate_hz"], errors="coerce").median()
        )
        q1, q3 = (
            np.percentile(finite, [25.0, 75.0]) if len(finite) else (np.nan, np.nan)
        )
        iqr = q3 - q1
        outlier_ratio = (
            float(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).mean())
            if len(finite) and np.isfinite(iqr)
            else np.nan
        )
        row: dict[str, Any] = {
            "participant_code": str(frame["participant_id"].iloc[0]),
            "session_id": str(frame["session_id"].iloc[0]),
            "phase": str(frame["phase"].iloc[0]),
            "signal_type": str(frame["sensor_channel"].iloc[0]),
            "sample_count": int(parquet_file.metadata.num_rows),
            "n": int(parquet_file.metadata.num_rows),
            "fs": sampling_rate,
            "t0": timestamps.iloc[0] if len(timestamps) else pd.NaT,
            "t1": timestamps.iloc[-1] if len(timestamps) else pd.NaT,
            "interval_ms_mean": float(positive_intervals.mean())
            if len(positive_intervals)
            else np.nan,
            "interval_ms_std": float(positive_intervals.std(ddof=1))
            if len(positive_intervals) > 1
            else 0.0,
            "interval_ms_p95": float(positive_intervals.quantile(0.95))
            if len(positive_intervals)
            else np.nan,
            "missing_ratio": float(values.isna().mean()),
            "nonfinite_ratio": float((~np.isfinite(values)).mean()),
            "gap_count": int((positive_intervals > 1.5 * median_interval).sum())
            if np.isfinite(median_interval)
            else 0,
            "mean": float(finite.mean()) if len(finite) else np.nan,
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "outlier_ratio_iqr": outlier_ratio,
            "value_column": value_column,
            "source_path": str(path),
        }
        row["actual_duration_s"] = (
            float((row["t1"] - row["t0"]).total_seconds())
            if pd.notna(row["t0"]) and pd.notna(row["t1"])
            else np.nan
        )
        row["expected_duration_s"] = (
            (row["n"] - 1) / sampling_rate
            if row["n"] > 1 and np.isfinite(sampling_rate) and sampling_rate > 0.0
            else np.nan
        )
        row["duration_error_s"] = row["actual_duration_s"] - row["expected_duration_s"]
        for mask_column in optional_masks:
            mask = frame[mask_column].fillna(False).astype(bool)
            row[f"{mask_column}_count"] = int(mask.sum())
            row[f"{mask_column}_ratio"] = float(mask.mean())
        return row

    @staticmethod
    def _structural_table(summary: pd.DataFrame, table: str) -> pd.DataFrame:
        if summary.empty:
            return pd.DataFrame()
        identifiers = ["participant_code", "session_id", "phase", "signal_type"]
        fields = {
            "counts": ["sample_count"],
            "duration": [
                "n",
                "fs",
                "t0",
                "t1",
                "actual_duration_s",
                "expected_duration_s",
                "duration_error_s",
            ],
            "interval": ["interval_ms_mean", "interval_ms_std", "interval_ms_p95"],
            "missing": [
                column
                for column in summary.columns
                if column in {"missing_ratio", "nonfinite_ratio", "gap_count"}
                or column.endswith("_mask_count")
                or column.endswith("_mask_ratio")
            ],
            "amplitude": ["mean", "std", "outlier_ratio_iqr", "value_column"],
        }[table]
        return summary[[*identifiers, *fields]].copy()

    @staticmethod
    def _structural_coverage(summary: pd.DataFrame) -> pd.DataFrame:
        if summary.empty:
            return pd.DataFrame()
        return (
            summary.groupby(["participant_code", "phase"], dropna=False)["sample_count"]
            .sum()
            .reset_index()
        )

    def _read_gold(self) -> pd.DataFrame:
        candidates = [
            getattr(self.paths, "gold_features_path", None),
            self.paths.gold_dir / "gold_features.csv",
        ]
        for path in candidates:
            if path and Path(path).exists():
                return (
                    pd.read_parquet(path)
                    if Path(path).suffix == ".parquet"
                    else pd.read_csv(path)
                )
        parquet = self.paths.gold_dir / "gold_features_v3.parquet"
        if parquet.exists():
            return pd.read_parquet(parquet)
        return pd.DataFrame()

    def _write_data_dictionary(self) -> None:
        mapping = {
            "participant_code": "Canonical participant ID (e.g., ID-03).",
            "session_id": "Session key composed from folder participant and condition.",
            "signal_type": "ECG, PPG_GREEN, PPG_IR, or PPG_RED.",
            "timestamp_utc": "UTC timestamp for each sample.",
            "timestamp_local": "Local timezone timestamp (Asia/Jakarta).",
            "sample_index": "Monotonic index per raw signal file.",
            "raw_value": "Raw sensor reading value from source file.",
            "sampling_rate_hz": "Sampling rate attached from metadata/defaults.",
            "phase": "Anchor-derived phase label (Relax/Task/Routine) after segmentation.",
            "anchor_start": "Original anchor window start timestamp.",
            "anchor_end": "Original anchor window end timestamp.",
            "quality_flag": "Segment-level quality marker (PASS or artifact tags).",
            "preprocessing_version": "Version string of wrangling configuration.",
        }
        lines = ["# Data Dictionary", "", "## Canonical Signal Schema", ""]
        for col in CANONICAL_SIGNAL_COLUMNS:
            lines.append(
                f"- `{col}`: {mapping.get(col, 'Defined in canonical schema.')}"
            )
        (self.paths.eda_dir / "data_dictionary.md").write_text(
            "\\n".join(lines), encoding="utf-8"
        )

    def _write_silver_compatibility_aliases(self) -> None:
        """Keep legacy Silver filenames without letting stages overwrite each other."""

        for source in self.paths.eda_dir.glob("silver_*"):
            if not source.is_file():
                continue
            target = self.paths.eda_dir / source.name.removeprefix("silver_")
            shutil.copyfile(source, target)

    def _write_stage_comparison(self, gold: pd.DataFrame) -> None:
        """Write a compact Raw/Silver/Gold comparison from stage-qualified evidence."""

        rows: list[dict[str, Any]] = []
        for stage in ("raw", "silver"):
            counts = _read_csv_if_present(
                self.paths.eda_dir / f"{stage}_sample_counts.csv"
            )
            duration = _read_csv_if_present(
                self.paths.eda_dir / f"{stage}_duration_consistency.csv"
            )
            missing = _read_csv_if_present(
                self.paths.eda_dir / f"{stage}_missingness_discontinuity.csv"
            )
            rows.append(
                {
                    "stage": stage,
                    "group_count": int(len(counts)),
                    "row_or_window_count": int(
                        pd.to_numeric(
                            counts.get("sample_count", pd.Series(dtype=float)),
                            errors="coerce",
                        ).sum()
                    ),
                    "median_duration_s": _finite_median(duration, "actual_duration_s"),
                    "median_missing_ratio": _finite_median(missing, "missing_ratio"),
                    "gap_count": int(
                        pd.to_numeric(
                            missing.get("gap_count", pd.Series(dtype=float)),
                            errors="coerce",
                        ).sum()
                    ),
                    "finite_predictor_rows": np.nan,
                }
            )

        group_columns = [
            "participant_id"
            if "participant_id" in gold.columns
            else "participant_code",
            "phase",
            "sensor_channel" if "sensor_channel" in gold.columns else "signal_type",
        ]
        predictor_columns = [
            column for column in self._predictors if column in gold.columns
        ]
        finite_predictor_rows = 0
        if predictor_columns:
            predictor_values = gold[predictor_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            finite_predictor_rows = int(
                np.isfinite(predictor_values.to_numpy(dtype=float)).all(axis=1).sum()
            )
        rows.append(
            {
                "stage": "gold",
                "group_count": int(
                    gold[group_columns].drop_duplicates().shape[0]
                    if set(group_columns).issubset(gold.columns)
                    else 0
                ),
                "row_or_window_count": int(len(gold)),
                "median_duration_s": np.nan,
                "median_missing_ratio": np.nan,
                "gap_count": np.nan,
                "finite_predictor_rows": finite_predictor_rows,
            }
        )
        comparison = pd.DataFrame(rows)
        comparison.to_csv(self.paths.eda_dir / "eda_stage_comparison.csv", index=False)
        markdown = [
            "# Raw, Silver, and Gold EDA Comparison",
            "",
            "All counts come from observed run-scoped artifacts. Waveform previews are bounded only for plotting and profiling.",
            "",
            "| Stage | Groups | Rows/windows | Median duration (s) | Median missing ratio | Gaps | Finite predictor rows |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            markdown.append(
                "| {stage} | {groups} | {count} | {duration} | {missing} | {gaps} | {finite} |".format(
                    stage=row["stage"],
                    groups=_format_number(row["group_count"]),
                    count=_format_number(row["row_or_window_count"]),
                    duration=_format_number(row["median_duration_s"]),
                    missing=_format_number(row["median_missing_ratio"]),
                    gaps=_format_number(row["gap_count"]),
                    finite=_format_number(row["finite_predictor_rows"]),
                )
            )
        (self.paths.eda_dir / "eda_stage_comparison.md").write_text(
            "\n".join(markdown), encoding="utf-8"
        )

    def _write_stage_manifest(self, stage: str) -> Path:
        """Fingerprint all EDA evidence emitted for one stage."""

        artifacts: list[dict[str, Any]] = []
        for path in sorted(self.paths.eda_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.paths.eda_dir)
            is_stage_artifact = (
                path.name.startswith(f"{stage}_")
                or path.name.startswith(f"eda_report_{stage}")
                or path.name == f"ydata_profiling_status_{stage}.json"
                or (
                    stage == "silver"
                    and path.name in {"eda_report.md", "eda_report.html"}
                )
                or (stage == "gold" and path.name.startswith("eda_stage_comparison"))
                or (
                    len(relative.parts) >= 2
                    and relative.parts[0] == "plots"
                    and relative.parts[1] == stage
                )
            )
            if not is_stage_artifact or path.name == f"eda_manifest_{stage}.json":
                continue
            artifacts.append(
                {
                    "path": str(path.relative_to(self.paths.artifacts_root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        manifest_path = self.paths.eda_dir / f"eda_manifest_{stage}.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "artifacts": artifacts,
                    "artifact_count": len(artifacts),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return manifest_path

    @staticmethod
    def _quality_policy_summary(gold: pd.DataFrame) -> pd.DataFrame:
        """Summarize strict and expanded Gold policy without changing rows."""
        if gold.empty or "quality_tier" not in gold.columns:
            return pd.DataFrame(
                columns=[
                    "quality_policy",
                    "quality_tier",
                    "window_count",
                    "group_count",
                ]
            )
        group_columns = ["participant_id", "phase", "sensor_channel"]
        rows = []
        for policy, tiers in {
            "strict_excellent_good": {"excellent", "good"},
            "expanded_excellent_good_acceptable": {"excellent", "good", "acceptable"},
        }.items():
            selected = gold[gold["quality_tier"].isin(tiers)]
            groups = (
                selected[group_columns].drop_duplicates().shape[0]
                if set(group_columns).issubset(selected.columns)
                else 0
            )
            rows.append(
                {
                    "quality_policy": policy,
                    "quality_tier": ",".join(sorted(tiers)),
                    "window_count": int(len(selected)),
                    "group_count": int(groups),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _sample_counts(silver: pd.DataFrame) -> pd.DataFrame:
        required = {"participant_code", "session_id", "phase", "signal_type"}
        if not required.issubset(silver.columns):
            return pd.DataFrame()
        return (
            silver.groupby(
                ["participant_code", "session_id", "phase", "signal_type"], dropna=False
            )
            .size()
            .reset_index(name="sample_count")
        )

    @staticmethod
    def _duration_consistency(
        silver: pd.DataFrame, value_column: str = "raw_value"
    ) -> pd.DataFrame:
        required = {
            "participant_code",
            "session_id",
            "phase",
            "signal_type",
            value_column,
            "sampling_rate_hz",
            "timestamp_utc",
        }
        if not required.issubset(silver.columns):
            return pd.DataFrame()
        g = (
            silver.groupby(
                ["participant_code", "session_id", "phase", "signal_type"], dropna=False
            )
            .agg(
                n=(value_column, "size"),
                fs=("sampling_rate_hz", "median"),
                t0=("timestamp_utc", "min"),
                t1=("timestamp_utc", "max"),
            )
            .reset_index()
        )
        g["actual_duration_s"] = (g["t1"] - g["t0"]).dt.total_seconds()
        g["expected_duration_s"] = (g["n"] - 1) / g["fs"].replace(0, np.nan)
        g["duration_error_s"] = g["actual_duration_s"] - g["expected_duration_s"]
        return g

    @staticmethod
    def _sampling_intervals(silver: pd.DataFrame) -> pd.DataFrame:
        required = {
            "participant_code",
            "session_id",
            "phase",
            "signal_type",
            "timestamp_utc",
        }
        if not required.issubset(silver.columns):
            return pd.DataFrame()
        parts = []
        for (p, sid, phase, signal_type), grp in silver.groupby(
            ["participant_code", "session_id", "phase", "signal_type"], dropna=False
        ):
            s = grp.sort_values("timestamp_utc")
            dt_ms = s["timestamp_utc"].diff().dt.total_seconds().mul(1000).dropna()
            if dt_ms.empty:
                continue
            parts.append(
                {
                    "participant_code": p,
                    "session_id": sid,
                    "phase": phase,
                    "signal_type": signal_type,
                    "interval_ms_mean": float(dt_ms.mean()),
                    "interval_ms_std": float(dt_ms.std(ddof=1))
                    if len(dt_ms) > 1
                    else 0.0,
                    "interval_ms_p95": float(dt_ms.quantile(0.95)),
                }
            )
        return pd.DataFrame(parts)

    @staticmethod
    def _missingness_discontinuity(
        silver: pd.DataFrame, value_column: str = "raw_value"
    ) -> pd.DataFrame:
        required = {
            "participant_code",
            "session_id",
            "phase",
            "signal_type",
            value_column,
            "timestamp_utc",
        }
        if not required.issubset(silver.columns):
            return pd.DataFrame()
        rows = []
        for (p, sid, phase, signal_type), grp in silver.groupby(
            ["participant_code", "session_id", "phase", "signal_type"], dropna=False
        ):
            s = grp.sort_values("timestamp_utc")
            values = pd.to_numeric(s[value_column], errors="coerce")
            miss_ratio = float(values.isna().mean())
            dt_ms = s["timestamp_utc"].diff().dt.total_seconds().mul(1000).dropna()
            median_dt = float(dt_ms.median()) if not dt_ms.empty else np.nan
            gap_count = (
                int((dt_ms > 1.5 * median_dt).sum())
                if median_dt and not np.isnan(median_dt)
                else 0
            )
            rows.append(
                {
                    "participant_code": p,
                    "session_id": sid,
                    "phase": phase,
                    "signal_type": signal_type,
                    "missing_ratio": miss_ratio,
                    "nonfinite_ratio": float((~np.isfinite(values)).mean()),
                    "gap_count": gap_count,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _amplitude_outliers(
        silver: pd.DataFrame, value_column: str = "raw_value"
    ) -> pd.DataFrame:
        if not {"signal_type", value_column}.issubset(silver.columns):
            return pd.DataFrame()
        rows = []
        for (signal_type,), grp in silver.groupby(["signal_type"], dropna=False):
            vals = pd.to_numeric(grp[value_column], errors="coerce").dropna()
            if vals.empty:
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            out_ratio = float(((vals < lo) | (vals > hi)).mean())
            rows.append(
                {
                    "signal_type": signal_type,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "outlier_ratio_iqr": out_ratio,
                    "value_column": value_column,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _phase_coverage(silver: pd.DataFrame) -> pd.DataFrame:
        if not {"participant_code", "phase"}.issubset(silver.columns):
            return pd.DataFrame()
        return (
            silver.groupby(["participant_code", "phase"], dropna=False)
            .size()
            .reset_index(name="sample_count")
        )

    @staticmethod
    def _label_balance(gold: pd.DataFrame) -> pd.DataFrame:
        if gold.empty or "label" not in gold.columns:
            return pd.DataFrame(columns=["label", "count", "ratio"])
        vc = gold["label"].value_counts(dropna=False).sort_index()
        out = vc.rename_axis("label").reset_index(name="count")
        out["ratio"] = out["count"] / out["count"].sum()
        return out

    @staticmethod
    def _cross_channel_correlation(
        silver: pd.DataFrame, value_column: str = "raw_value"
    ) -> pd.DataFrame:
        if not {
            "signal_type",
            "participant_code",
            "session_id",
            "phase",
            "timestamp_utc",
            value_column,
        }.issubset(silver.columns):
            return pd.DataFrame()
        rows = []
        normalized = silver["signal_type"].astype(str).str.lower()
        ppg = silver[normalized.isin(["ppg_green", "ppg_ir", "ppg_red"])].copy()
        ppg["signal_type"] = ppg["signal_type"].astype(str).str.upper()
        if ppg.empty:
            return pd.DataFrame(
                columns=[
                    "participant_code",
                    "session_id",
                    "phase",
                    "corr_green_ir",
                    "corr_green_red",
                    "corr_ir_red",
                ]
            )

        for (p, sid, phase), grp in ppg.groupby(
            ["participant_code", "session_id", "phase"], dropna=False
        ):
            pivot = grp.pivot_table(
                index="timestamp_utc",
                columns="signal_type",
                values=value_column,
                aggfunc="mean",
            )
            corr = pivot.corr()
            rows.append(
                {
                    "participant_code": p,
                    "session_id": sid,
                    "phase": phase,
                    "corr_green_ir": float(corr.loc["PPG_GREEN", "PPG_IR"])
                    if {"PPG_GREEN", "PPG_IR"}.issubset(corr.columns)
                    else np.nan,
                    "corr_green_red": float(corr.loc["PPG_GREEN", "PPG_RED"])
                    if {"PPG_GREEN", "PPG_RED"}.issubset(corr.columns)
                    else np.nan,
                    "corr_ir_red": float(corr.loc["PPG_IR", "PPG_RED"])
                    if {"PPG_IR", "PPG_RED"}.issubset(corr.columns)
                    else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def _plot_label_balance(self, lb: pd.DataFrame, prefix: str = "") -> None:
        if lb.empty:
            return
        plt.figure(figsize=(5, 3))
        plt.bar(lb["label"].astype(str), lb["count"].to_numpy())
        plt.title("Label Balance")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.paths.eda_dir / f"{prefix}label_balance.png", dpi=150)
        plt.close()

    def _plot_sampling_interval(self, interval: pd.DataFrame, prefix: str = "") -> None:
        if interval.empty:
            return
        plt.figure(figsize=(7, 4))
        for stype, grp in interval.groupby("signal_type"):
            plt.hist(
                grp["interval_ms_mean"].dropna().to_numpy(),
                bins=30,
                alpha=0.4,
                label=stype,
            )
        plt.title("Sampling Interval Mean Histogram")
        plt.xlabel("Interval (ms)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.paths.eda_dir / f"{prefix}sampling_interval_hist.png", dpi=150)
        plt.close()

    def _plot_overlay_example(self, silver: pd.DataFrame, prefix: str = "") -> None:
        # Plot one example ECG+PPG overlay for traceability
        normalized = silver["signal_type"].astype(str).str.lower()
        ex = silver[normalized.isin(["ecg", "ppg_green"])].copy()
        ex["signal_type"] = ex["signal_type"].astype(str).str.upper()
        if ex.empty:
            return
        first_key = (
            ex[["participant_code", "session_id", "phase"]].drop_duplicates().iloc[0]
        )
        sub = ex[
            (ex["participant_code"] == first_key["participant_code"])
            & (ex["session_id"] == first_key["session_id"])
            & (ex["phase"] == first_key["phase"])
        ].copy()
        value_column = "clean_value" if "clean_value" in sub.columns else "raw_value"
        pivot = sub.pivot_table(
            index="timestamp_utc",
            columns="signal_type",
            values=value_column,
            aggfunc="mean",
        )
        if "ECG" not in pivot.columns or "PPG_GREEN" not in pivot.columns:
            return
        pivot = pivot.dropna().head(4000)
        if pivot.empty:
            return
        # normalize for visual comparison only
        ecg = (pivot["ECG"] - pivot["ECG"].mean()) / (pivot["ECG"].std() + 1e-9)
        ppg = (pivot["PPG_GREEN"] - pivot["PPG_GREEN"].mean()) / (
            pivot["PPG_GREEN"].std() + 1e-9
        )
        plt.figure(figsize=(9, 4))
        plt.plot(pivot.index, ecg, label="ECG (norm)", linewidth=1)
        plt.plot(pivot.index, ppg, label="PPG_GREEN (norm)", linewidth=1)
        plt.title(
            f"Example Overlay {first_key['participant_code']} / {first_key['phase']}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            self.paths.eda_dir / f"{prefix}example_ecg_ppg_overlay.png", dpi=150
        )
        plt.close()

    def _plot_raw_vs_filtered_with_peaks(
        self, silver: pd.DataFrame, prefix: str = ""
    ) -> None:
        from scipy.signal import find_peaks

        if silver.empty:
            return

        p_col = (
            "participant_code"
            if "participant_code" in silver.columns
            else "participant_id"
        )
        ch_col = "signal_type" if "signal_type" in silver.columns else "sensor_channel"
        timestamp_col = "timestamp_utc" if "timestamp_utc" in silver.columns else None
        if timestamp_col is None:
            return

        group_columns = [
            column
            for column in (p_col, "session_id", "phase", ch_col)
            if column in silver.columns
        ]
        for key, grp in silver.groupby(group_columns, dropna=False):
            key_values = key if isinstance(key, tuple) else (key,)
            key_map = dict(zip(group_columns, key_values, strict=True))
            p = key_map[p_col]
            ch = key_map[ch_col]
            plot_df = grp.sort_values(timestamp_col).head(1500)
            if (
                plot_df.empty
                or "raw_value" not in plot_df.columns
                or "clean_value" not in plot_df.columns
            ):
                continue

            timestamp = plot_df["timestamp_utc"]
            raw = plot_df["raw_value"].to_numpy()
            clean = plot_df["clean_value"].to_numpy()

            sampling_rate = float(
                pd.to_numeric(plot_df.get("sampling_rate_hz"), errors="coerce").median()
            )
            minimum_distance = max(1, int(round(0.30 * sampling_rate)))
            sys_peaks, _ = find_peaks(clean, distance=minimum_distance)
            dia_peaks, _ = find_peaks(-clean, distance=minimum_distance)

            out_dir = (
                self.paths.eda_dir / "plots" / prefix.rstrip("_") / str(p) / str(ch)
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            figure, (peak_axis, madc_axis) = plt.subplots(2, 1, figsize=(12, 7))
            peak_axis.plot(timestamp, raw, label="Raw", alpha=0.5, color="gray")
            peak_axis.plot(timestamp, clean, label="Filtered", alpha=0.9, color="green")
            if len(sys_peaks) > 0:
                peak_axis.scatter(
                    timestamp.iloc[sys_peaks],
                    clean[sys_peaks],
                    color="red",
                    label="Diagnostic local maxima preview",
                    zorder=5,
                )
            if len(dia_peaks) > 0:
                peak_axis.scatter(
                    timestamp.iloc[dia_peaks],
                    clean[dia_peaks],
                    color="blue",
                    label="Diagnostic local minima preview",
                    zorder=5,
                )
            peak_axis.set_title(f"{p} - {ch}: Raw vs Filtered diagnostic preview")
            peak_axis.legend()
            madc_axis.plot(
                timestamp,
                raw,
                label="Raw (before canonical filter)",
                color="red",
                alpha=0.5,
            )
            madc_axis.plot(
                timestamp,
                clean,
                label="After canonical filter (MADC annotation-only)",
                color="blue",
                alpha=0.8,
            )
            madc_axis.set_title(f"{p} - {ch}: Raw vs canonical filter")
            madc_axis.legend()
            figure.tight_layout()
            figure.savefig(out_dir / "raw_vs_filtered_peaks.png", dpi=100)
            plt.close(figure)

    def _generate_ydata_profiling(
        self, silver: pd.DataFrame, stage: str = "silver"
    ) -> None:
        status_path = self.paths.eda_dir / f"ydata_profiling_status_{stage}.json"
        # YData Profiling performs optional product analytics over HTTPS.  The
        # project contract is local-only, so disable that integration before
        # importing/constructing the profiler.
        os.environ.setdefault("YDATA_PROFILING_NO_ANALYTICS", "1")
        try:
            from ydata_profiling import ProfileReport
        except ImportError:
            logger.warning(
                "ydata-profiling not installed. Skipping time-series profile report."
            )
            status_path.write_text(
                json.dumps(
                    {"status": "blocked", "reason": "ydata_profiling_not_installed"},
                    indent=2,
                ),
                encoding="utf-8",
            )
            return

        if silver.empty:
            status_path.write_text(
                json.dumps({"status": "blocked", "reason": "empty_stage"}, indent=2),
                encoding="utf-8",
            )
            return

        p_col = (
            "participant_code"
            if "participant_code" in silver.columns
            else "participant_id"
        )
        ch_col = "signal_type" if "signal_type" in silver.columns else "sensor_channel"
        s_col = "session_id" if "session_id" in silver.columns else "recording_id"

        group_columns = [p_col, s_col, ch_col]
        if "phase" in silver.columns:
            group_columns.append("phase")
        sort_columns = [*group_columns]
        timestamp_column = (
            "timestamp_utc" if "timestamp_utc" in silver.columns else None
        )
        if timestamp_column:
            sort_columns.append(timestamp_column)
        group_count = max(1, int(silver[group_columns].drop_duplicates().shape[0]))
        rows_per_group = max(1, 5000 // group_count)
        sample_df = (
            silver.sort_values(sort_columns, kind="stable")
            .groupby(group_columns, dropna=False, sort=False)
            .head(rows_per_group)
            .head(5000)
        )

        if sample_df.empty:
            return

        try:
            profile = ProfileReport(
                sample_df,
                title=f"{stage.title()} Cohort EDA Profile",
                minimal=True,
                progress_bar=False,
            )
            out_file = self.paths.eda_dir / f"eda_report_{stage}.html"
            profile.to_file(out_file)
            if stage == "silver":
                # Compatibility alias for existing consumers of the first
                # canonical Silver report; the content is byte-identical.
                shutil.copyfile(out_file, self.paths.eda_dir / "eda_report.html")
            logger.info(f"ydata-profiling report generated at {out_file}")
            status_path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "path": str(out_file),
                        "input_sampling_note": self._input_sampling_note,
                        "sampling_strategy": "deterministic_stratified_real_rows",
                        "sampled_row_count": int(len(sample_df)),
                        "group_count": group_count,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to generate ydata-profiling report: {e}")
            status_path.write_text(
                json.dumps({"status": "failed", "reason": str(e)}, indent=2),
                encoding="utf-8",
            )

    def _build_report(
        self,
        counts: pd.DataFrame,
        duration: pd.DataFrame,
        interval: pd.DataFrame,
        missing: pd.DataFrame,
        amplitude: pd.DataFrame,
        coverage: pd.DataFrame,
        label_balance: pd.DataFrame,
        correlation: pd.DataFrame,
        quality_policy: pd.DataFrame,
        stage: str = "silver",
    ) -> str:
        lines = [
            f"# EDA & Data Quality Report ({stage})",
            "",
            self._input_sampling_note,
            f"- Total segmented samples: {len(counts) if not counts.empty else 0} grouped rows",
            f"- Duration check rows: {len(duration)}",
            f"- Interval summary rows: {len(interval)}",
            f"- Missingness rows: {len(missing)}",
            f"- Phase coverage rows: {len(coverage)}",
            "",
            "## Key Stats",
        ]
        if stage == "gold":
            lines.extend(
                [
                    "- Gold training policy: `excellent`, `good`, dan `acceptable`; `poor`/`invalid` tetap exclusion lineage.",
                    "- MADC bersifat annotation-only; tidak mengubah waveform dan tidak menjadi hard gate.",
                    "- Gold tidak memakai ECG, PAT, atau motion sebagai fitur training; tidak ada synthetic recovery/augmentasi preprocessing.",
                ]
            )
            if not quality_policy.empty:
                lines.append(
                    f"- Strict-vs-expanded quality summary: {quality_policy.to_dict(orient='records')}"
                )
        if not label_balance.empty:
            lines.append(f"- Label balance: {label_balance.to_dict(orient='records')}")
        if not amplitude.empty:
            lines.append(
                f"- Amplitude outlier ratios: {amplitude[['signal_type', 'outlier_ratio_iqr']].to_dict(orient='records')}"
            )
        if not correlation.empty:
            corr_cols = [c for c in correlation.columns if c.startswith("corr_")]
            means = (
                correlation[corr_cols].mean(numeric_only=True).to_dict()
                if corr_cols
                else {}
            )
            lines.append(f"- Mean cross-channel correlations: {means}")

        lines.extend(
            [
                "",
                "## Generated Files",
                "- sample_counts.csv",
                "- duration_consistency.csv",
                "- sampling_interval_summary.csv",
                "- missingness_discontinuity.csv",
                "- amplitude_outliers.csv",
                "- phase_coverage.csv",
                "- label_balance.csv",
                "- cross_channel_correlation.csv",
                "- edge_trim_impact.csv",
                "- label_balance.png",
                "- sampling_interval_hist.png",
                "- example_ecg_ppg_overlay.png",
                "- eda_report.html (ydata-profiling time-series)",
                "- plots/ (Raw vs Filtered & MADC comparisons per participant/channel)",
                "- gold_quality_policy_summary.csv (strict vs expanded policy)",
            ]
        )
        return "\n".join(lines)

    def _edge_trim_impact(self, duration: pd.DataFrame, stage: str) -> pd.DataFrame:
        """Export observed duration and the locked 300-to-240s crop contract.

        The table never fabricates a cropped waveform.  For Raw it records the
        observed source duration; for Silver/Gold it records the observed
        analysis-stage duration.  The configured contract makes the difference
        explicit for participant/channel/phase audit.
        """

        columns = [
            "participant_code",
            "session_id",
            "phase",
            "signal_type",
            "stage",
            "source_phase_duration_seconds",
            "edge_trim_start_seconds",
            "edge_trim_end_seconds",
            "analysis_phase_duration_seconds",
            "observed_duration_seconds",
            "duration_target_seconds",
            "duration_delta_seconds",
            "coverage_status",
        ]
        if duration.empty:
            return pd.DataFrame(columns=columns)
        result = duration[
            [
                "participant_code",
                "session_id",
                "phase",
                "signal_type",
                "actual_duration_s",
            ]
        ].copy()
        result = result.rename(
            columns={"actual_duration_s": "observed_duration_seconds"}
        )
        result["stage"] = stage
        result["source_phase_duration_seconds"] = self._source_phase_seconds
        result["edge_trim_start_seconds"] = self._edge_trim_start_seconds
        result["edge_trim_end_seconds"] = self._edge_trim_end_seconds
        result["analysis_phase_duration_seconds"] = self._analysis_phase_seconds
        result["duration_target_seconds"] = (
            self._source_phase_seconds
            if stage == "raw"
            else self._analysis_phase_seconds
        )
        result["duration_delta_seconds"] = (
            result["observed_duration_seconds"] - result["duration_target_seconds"]
        )
        result["coverage_status"] = np.where(
            result["observed_duration_seconds"]
            >= result["duration_target_seconds"] - 1.0,
            "within_tolerance",
            "insufficient_observed_duration",
        )
        return result[columns]


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _finite_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else np.nan


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    return f"{number:.6g}"


def generate_eda_reports(
    project_root: Path, config: Any, stage: str = "silver"
) -> dict:
    paths = pipeline_paths_from_config(project_root, config)
    ensure_dirs(paths)
    eda = WranglingEDA(paths, config)
    return eda.run(stage=stage)


if __name__ == "__main__":
    from PipelinePython.common.cli import main

    main(["eda"])
