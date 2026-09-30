from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

from .settings import PipelinePaths

logger = logging.getLogger(__name__)

PPG_SENSORS = ("ppg_green", "ppg_ir", "ppg_red")
PHASES = ("Relax", "Task")
SEGMENT_SECONDS = 30.0


@dataclass(frozen=True)
class DiagnosticsReportOutputs:
    output_dir: Path
    summary_path: Path
    participant_sensor_phase_count: int
    figure_count: int


class SQAPeakDiagnosticsReport:
    """Per-participant SQA and peak-to-peak diagnostics from existing pipeline layers."""

    def __init__(self, paths: PipelinePaths) -> None:
        self.paths = paths
        self.out_dir = paths.project_root / "reports" / "diagnostics" / "sqa_peak"
        self.per_subject_dir = self.out_dir / "per_subject"

    def run(self) -> DiagnosticsReportOutputs:
        self.per_subject_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: list[dict[str, Any]] = []
        figure_count = 0
        combo_count = 0

        for silver_path in sorted(self.paths.silver_dir.glob("*/[RT]*/*.parquet")):
            participant = silver_path.parts[-3]
            phase = silver_path.parts[-2]
            sensor = silver_path.stem
            if phase not in PHASES or sensor not in PPG_SENSORS:
                continue

            try:
                result = self._process_one(participant, phase, sensor, silver_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Diagnostics skipped for %s/%s/%s: %s",
                    participant,
                    phase,
                    sensor,
                    exc,
                )
                continue

            if result is None:
                continue
            combo_count += 1
            summary_rows.append(result["summary"])
            figure_count += int(result["figures"])

        summary = pd.DataFrame(summary_rows)
        summary_path = self.out_dir / "summary_sqa_peak.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("SQA/peak diagnostics saved in %s", self.out_dir)
        return DiagnosticsReportOutputs(
            self.out_dir, summary_path, combo_count, figure_count
        )

    def _process_one(
        self, participant: str, phase: str, sensor: str, silver_path: Path
    ) -> dict[str, Any] | None:
        df = pd.read_parquet(silver_path)
        if df.empty:
            return None
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"], utc=True, errors="coerce"
        )

        fs = float(pd.to_numeric(df["sampling_rate_hz"], errors="coerce").median())
        if not np.isfinite(fs) or fs <= 0:
            return None

        signal = pd.to_numeric(df["raw_value"], errors="coerce").to_numpy(dtype=float)
        time_s = (
            (df["timestamp_utc"] - df["timestamp_utc"].iloc[0])
            .dt.total_seconds()
            .to_numpy(dtype=float)
        )
        quality_score = float(
            pd.to_numeric(
                df.get("quality_score", pd.Series([np.nan])), errors="coerce"
            ).median()
        )
        quality_tier = self._mode_or_unknown(df.get("quality_tier"))
        quality_flag = self._mode_or_unknown(df.get("quality_flag"))

        cleaned, peaks_global = self._detect_peaks(signal, fs)
        segment_samples = max(1, int(round(SEGMENT_SECONDS * fs)))

        sqa_rows: list[dict[str, Any]] = []
        accepted_rows: list[dict[str, Any]] = []
        segment_rows: list[dict[str, Any]] = []
        peak_rows: list[dict[str, Any]] = []
        rr_rows: list[dict[str, Any]] = []

        for segment_idx, start in enumerate(range(0, len(signal), segment_samples)):
            end = min(start + segment_samples, len(signal))
            if end - start < max(10, int(0.5 * segment_samples)):
                continue
            row = self._segment_metrics(
                segment_idx=segment_idx,
                start=start,
                end=end,
                fs=fs,
                signal=signal,
                cleaned=cleaned,
                time_s=time_s,
                peaks_global=peaks_global,
                quality_score=quality_score,
                quality_tier=quality_tier,
                quality_flag=quality_flag,
            )
            sqa_rows.append(row["sqa"])
            segment_rows.append(row["segment"])
            peak_rows.extend(row["peaks"])
            rr_rows.extend(row["rr"])
            if row["sqa"]["final_label"] in {"ACCEPTED", "CAUTION"}:
                accepted_rows.append(row["accepted"])

        out = self.per_subject_dir / participant / phase / sensor
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sqa_rows).to_csv(out / "sqa_report.csv", index=False)
        pd.DataFrame(accepted_rows).to_csv(out / "accepted_segments.csv", index=False)
        pd.DataFrame(segment_rows).to_csv(out / "segments_summary.csv", index=False)
        pd.DataFrame(peak_rows).to_csv(out / "peaks_report.csv", index=False)
        pd.DataFrame(rr_rows).to_csv(out / "rr_intervals.csv", index=False)

        self._plot_sqa(
            out / "sqa_plot.png", participant, phase, sensor, time_s, signal, sqa_rows
        )
        self._plot_peaks(
            out / "peak_plot.png",
            participant,
            phase,
            sensor,
            time_s,
            cleaned,
            peaks_global,
            peak_rows,
            rr_rows,
            segment_rows,
        )

        summary = {
            "participant_code": participant,
            "phase": phase,
            "sensor_channel": sensor,
            "fs_hz": fs,
            "n_samples": len(signal),
            "n_segments": len(segment_rows),
            "n_accepted": int(sum(r["final_label"] == "ACCEPTED" for r in sqa_rows)),
            "n_caution": int(sum(r["final_label"] == "CAUTION" for r in sqa_rows)),
            "n_rejected": int(sum(r["final_label"] == "REJECTED" for r in sqa_rows)),
            "n_peaks_total": len(peak_rows),
            "n_rr_total": len(rr_rows),
            "mean_hr_bpm": self._nanmean([r["hrv_mean_hr_bpm"] for r in segment_rows]),
            "mean_rmssd_ms": self._nanmean([r["hrv_rmssd_ms"] for r in segment_rows]),
            "mean_sdnn_ms": self._nanmean([r["hrv_sdnn_ms"] for r in segment_rows]),
            "quality_score": quality_score,
            "quality_tier": quality_tier,
            "quality_flag": quality_flag,
        }
        return {"summary": summary, "figures": 2}

    def _segment_metrics(
        self,
        *,
        segment_idx: int,
        start: int,
        end: int,
        fs: float,
        signal: np.ndarray,
        cleaned: np.ndarray,
        time_s: np.ndarray,
        peaks_global: np.ndarray,
        quality_score: float,
        quality_tier: str,
        quality_flag: str,
    ) -> dict[str, Any]:
        segment = signal[start:end]
        segment_clean = cleaned[start:end]
        peaks = peaks_global[(peaks_global >= start) & (peaks_global < end)]
        peaks_in_segment = peaks - start
        rr_ms = (
            np.diff(peaks) / fs * 1000.0
            if len(peaks) >= 2
            else np.array([], dtype=float)
        )
        rr_valid = (
            (rr_ms >= 333.0) & (rr_ms <= 1500.0)
            if len(rr_ms)
            else np.array([], dtype=bool)
        )

        missing_ratio = float(np.isnan(segment).mean())
        clipping_ratio = self._clipping_ratio(segment)
        flatline = bool(np.nanstd(segment) < 1e-9)
        low_variance = bool(np.nanstd(segment) < 1e-6 and not flatline)
        gap_count = self._gap_count(time_s[start:end])
        mean_rr = float(np.nanmean(rr_ms)) if len(rr_ms) else np.nan
        sdnn = float(np.nanstd(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan
        rmssd = (
            float(np.sqrt(np.nanmean(np.diff(rr_ms) ** 2)))
            if len(rr_ms) > 2
            else np.nan
        )
        mean_hr = (
            float(60000.0 / mean_rr) if np.isfinite(mean_rr) and mean_rr > 0 else np.nan
        )
        min_rr = float(np.nanmin(rr_ms)) if len(rr_ms) else np.nan
        max_rr = float(np.nanmax(rr_ms)) if len(rr_ms) else np.nan
        rr_range = (
            float(max_rr - min_rr)
            if np.isfinite(min_rr) and np.isfinite(max_rr)
            else np.nan
        )

        final_label = self._quality_label(
            quality_score,
            len(peaks),
            missing_ratio,
            clipping_ratio,
            flatline,
            low_variance,
        )
        start_time = float(time_s[start]) if len(time_s) > start else float(start / fs)
        end_time = float(time_s[end - 1]) if len(time_s) >= end else float(end / fs)
        composite_score = self._composite_score(final_label, quality_score)

        peak_rows = [
            {
                "segment_idx": segment_idx,
                "peak_in_segment": i,
                "peak_index_in_segment": int(p_local),
                "peak_time_in_segment_s": float(p_local / fs),
                "peak_time_global_s": float(time_s[int(p_global)]),
                "peak_amplitude": float(segment_clean[int(p_local)]),
            }
            for i, (p_global, p_local) in enumerate(zip(peaks, peaks_in_segment))
        ]
        rr_rows = [
            {
                "segment_idx": segment_idx,
                "rr_index": i,
                "rr_interval_ms": float(rr),
                "rr_valid": bool(valid),
            }
            for i, (rr, valid) in enumerate(zip(rr_ms, rr_valid))
        ]
        sqa = {
            "segment_idx": segment_idx,
            "start_sample": start,
            "end_sample": end,
            "start_time_s": start_time,
            "end_time_s": end_time,
            "quality_score": quality_score,
            "quality_tier": quality_tier,
            "quality_flag": quality_flag,
            "missing_ratio": missing_ratio,
            "clipping_ratio": clipping_ratio,
            "flatline": flatline,
            "low_variance": low_variance,
            "timestamp_gap_count": gap_count,
            "n_peaks": len(peaks),
            "hrv_mean_hr_bpm": mean_hr,
            "hrv_mean_rr_ms": mean_rr,
            "hrv_sdnn_ms": sdnn,
            "hrv_rmssd_ms": rmssd,
            "n_valid_rr": int(rr_valid.sum()),
            "n_total_rr": len(rr_ms),
            "composite_score": composite_score,
            "final_label": final_label,
        }
        segment_summary = {
            "segment_idx": segment_idx,
            "start_sample": start,
            "end_sample": end,
            "start_time_s": start_time,
            "end_time_s": end_time,
            "composite_score": composite_score,
            "n_samples": end - start,
            "n_peaks_detected": len(peaks),
            "n_valid_rr": int(rr_valid.sum()),
            "n_total_rr": len(rr_ms),
            "hrv_mean_hr_bpm": mean_hr,
            "hrv_sdnn_ms": sdnn,
            "hrv_rmssd_ms": rmssd,
            "hrv_mean_rr_ms": mean_rr,
            "hrv_min_rr_ms": min_rr,
            "hrv_max_rr_ms": max_rr,
            "rr_range_ms": rr_range,
        }
        accepted = {
            "segment_idx": segment_idx,
            "start_sample": start,
            "end_sample": end,
            "start_time_s": start_time,
            "end_time_s": end_time,
            "composite_score": composite_score,
            "final_label": final_label,
            "hr_bpm": mean_hr,
            "n_peaks": len(peaks),
            "n_valid_rr": int(rr_valid.sum()),
            "duration_s": end_time - start_time,
        }
        return {
            "sqa": sqa,
            "segment": segment_summary,
            "accepted": accepted,
            "peaks": peak_rows,
            "rr": rr_rows,
        }

    def _detect_peaks(
        self, signal: np.ndarray, fs: float
    ) -> tuple[np.ndarray, np.ndarray]:
        clean_input = (
            pd.Series(signal)
            .interpolate(limit_direction="both")
            .bfill()
            .ffill()
            .to_numpy(dtype=float)
        )
        if len(clean_input) < max(10, int(fs * 2)):
            return clean_input, np.array([], dtype=int)
        cleaned = nk.ppg_clean(clean_input, sampling_rate=fs, method="elgendi")
        info = nk.ppg_findpeaks(cleaned, sampling_rate=fs, method="elgendi")
        peaks = np.array(info.get("PPG_Peaks", []), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(cleaned))]
        return np.asarray(cleaned, dtype=float), peaks

    def _plot_sqa(
        self,
        path: Path,
        participant: str,
        phase: str,
        sensor: str,
        time_s: np.ndarray,
        signal: np.ndarray,
        sqa_rows: list[dict[str, Any]],
    ) -> None:
        fig, axes = plt.subplots(4, 1, figsize=(15, 11), constrained_layout=True)
        fig.suptitle(
            f"SQA Diagnostics | {participant} | {phase} | {sensor}",
            fontsize=13,
            fontweight="bold",
        )

        axes[0].plot(time_s, signal, color="#2f3e46", linewidth=0.45)
        for row in sqa_rows:
            color = {
                "ACCEPTED": "#2ecc71",
                "CAUTION": "#f39c12",
                "REJECTED": "#e74c3c",
            }.get(row["final_label"], "#95a5a6")
            axes[0].axvspan(
                row["start_time_s"],
                row["end_time_s"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        axes[0].set_title("Aligned signal with segment quality overlay")
        axes[0].set_ylabel("Amplitude")

        x = [r["segment_idx"] for r in sqa_rows]
        axes[1].bar(
            x,
            [r["composite_score"] for r in sqa_rows],
            color="#4c78a8",
            alpha=0.85,
            label="Composite",
        )
        axes[1].plot(
            x,
            [r["quality_score"] for r in sqa_rows],
            color="#e45756",
            marker="o",
            label="Silver quality",
        )
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Segment-level quality score")
        axes[1].legend()

        axes[2].plot(
            x, [r["missing_ratio"] for r in sqa_rows], marker="o", label="Missing"
        )
        axes[2].plot(
            x, [r["clipping_ratio"] for r in sqa_rows], marker="o", label="Clipping"
        )
        axes[2].bar(
            x,
            [r["timestamp_gap_count"] for r in sqa_rows],
            alpha=0.25,
            label="Gap count",
        )
        axes[2].set_title("Missingness, clipping, and timestamp gaps")
        axes[2].legend()

        axes[3].plot(
            x, [r["hrv_mean_hr_bpm"] for r in sqa_rows], marker="o", label="Mean HR bpm"
        )
        axes[3].plot(
            x, [r["hrv_mean_rr_ms"] for r in sqa_rows], marker="o", label="Mean RR ms"
        )
        axes[3].set_title("HR/RR summary per segment")
        axes[3].set_xlabel("segment_idx")
        axes[3].legend()

        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_peaks(
        self,
        path: Path,
        participant: str,
        phase: str,
        sensor: str,
        time_s: np.ndarray,
        cleaned: np.ndarray,
        peaks: np.ndarray,
        peak_rows: list[dict[str, Any]],
        rr_rows: list[dict[str, Any]],
        segment_rows: list[dict[str, Any]],
    ) -> None:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), constrained_layout=True)
        mean_hr = self._nanmean([r["hrv_mean_hr_bpm"] for r in segment_rows])
        mean_rmssd = self._nanmean([r["hrv_rmssd_ms"] for r in segment_rows])
        mean_sdnn = self._nanmean([r["hrv_sdnn_ms"] for r in segment_rows])
        fig.suptitle(
            f"Peak-to-Peak Diagnostics | {participant} | {phase} | {sensor} | "
            f"peaks={len(peaks)} | valid_rr={sum(r['rr_valid'] for r in rr_rows)} | "
            f"HR={mean_hr:.1f} | RMSSD={mean_rmssd:.1f} | SDNN={mean_sdnn:.1f}",
            fontsize=12,
            fontweight="bold",
        )

        sampled_idx = (
            np.linspace(0, len(cleaned) - 1, min(len(cleaned), 3500)).astype(int)
            if len(cleaned)
            else np.array([], dtype=int)
        )
        axes[0].plot(
            time_s[sampled_idx], cleaned[sampled_idx], color="#34495e", linewidth=0.5
        )
        shown_peaks = (
            peaks[np.isin(peaks, sampled_idx)]
            if len(sampled_idx) < len(cleaned)
            else peaks
        )
        axes[0].scatter(
            time_s[shown_peaks],
            cleaned[shown_peaks],
            s=12,
            color="#e74c3c",
            label="Detected peaks",
        )
        axes[0].set_title("Full cleaned signal with detected peaks")
        axes[0].legend()

        zoom_start, zoom_end = self._zoom_bounds(segment_rows)
        zoom_mask = (time_s >= zoom_start) & (time_s <= zoom_end)
        zoom_peaks = (
            peaks[(time_s[peaks] >= zoom_start) & (time_s[peaks] <= zoom_end)]
            if len(peaks)
            else np.array([], dtype=int)
        )
        axes[1].plot(
            time_s[zoom_mask], cleaned[zoom_mask], color="#2f3e46", linewidth=0.9
        )
        axes[1].scatter(time_s[zoom_peaks], cleaned[zoom_peaks], s=28, color="#e74c3c")
        for a, b in zip(zoom_peaks[:-1], zoom_peaks[1:]):
            axes[1].annotate(
                "",
                xy=(time_s[b], cleaned[b]),
                xytext=(time_s[a], cleaned[a]),
                arrowprops={"arrowstyle": "<->", "color": "#3498db", "lw": 0.8},
            )
        axes[1].set_title("Zoomed peak-to-peak view")

        rr_df = pd.DataFrame(rr_rows)
        if not rr_df.empty:
            axes[2].plot(
                np.arange(len(rr_df)),
                rr_df["rr_interval_ms"],
                marker="o",
                linewidth=0.9,
            )
            axes[2].axhline(333, color="#e74c3c", linestyle="--", linewidth=0.8)
            axes[2].axhline(1500, color="#e74c3c", linestyle="--", linewidth=0.8)
        axes[2].set_title("RR interval timeline")
        axes[2].set_ylabel("ms")

        peak_df = pd.DataFrame(peak_rows)
        if not peak_df.empty:
            axes[3].plot(
                peak_df["peak_time_global_s"],
                peak_df["peak_amplitude"],
                marker=".",
                linewidth=0.7,
            )
        axes[3].set_title("Peak amplitude timeline")
        axes[3].set_xlabel("time_s")

        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _mode_or_unknown(series: pd.Series | None) -> str:
        if series is None:
            return "unknown"
        clean = series.dropna()
        return str(clean.mode().iloc[0]) if not clean.empty else "unknown"

    @staticmethod
    def _clipping_ratio(segment: np.ndarray) -> float:
        clean = pd.Series(segment).dropna().to_numpy(dtype=float)
        if len(clean) == 0:
            return 1.0
        vmin, vmax = np.nanmin(clean), np.nanmax(clean)
        if vmax <= vmin:
            return 1.0
        return float(((clean == vmin) | (clean == vmax)).mean())

    @staticmethod
    def _gap_count(time_s: np.ndarray) -> int:
        if len(time_s) < 3:
            return 0
        dt = np.diff(time_s)
        median = np.nanmedian(dt)
        if not np.isfinite(median) or median <= 0:
            return 0
        return int(np.sum(dt > 1.5 * median))

    @staticmethod
    def _quality_label(
        quality_score: float,
        n_peaks: int,
        missing_ratio: float,
        clipping_ratio: float,
        flatline: bool,
        low_variance: bool,
    ) -> str:
        if (
            flatline
            or low_variance
            or n_peaks < 3
            or missing_ratio > 0.10
            or clipping_ratio > 0.20
        ):
            return "REJECTED"
        if quality_score >= 0.95:
            return "ACCEPTED"
        if quality_score >= 0.80:
            return "CAUTION"
        return "REJECTED"

    @staticmethod
    def _composite_score(label: str, quality_score: float) -> float:
        if label == "ACCEPTED":
            return float(max(0.60, quality_score))
        if label == "CAUTION":
            return float(min(0.59, max(0.40, quality_score)))
        return float(min(0.39, quality_score if np.isfinite(quality_score) else 0.0))

    @staticmethod
    def _zoom_bounds(segment_rows: list[dict[str, Any]]) -> tuple[float, float]:
        for row in segment_rows:
            if row["n_peaks_detected"] >= 3:
                return float(row["start_time_s"]), float(row["end_time_s"])
        if segment_rows:
            return float(segment_rows[0]["start_time_s"]), float(
                segment_rows[0]["end_time_s"]
            )
        return 0.0, SEGMENT_SECONDS

    @staticmethod
    def _nanmean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if len(arr) else np.nan
