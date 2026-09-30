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
from scipy.stats import pearsonr, spearmanr

from .config import ECG_FS, PPG_FS, PPG_IR_FS, PPG_RED_FS
from .ingestion import (
    _load_ecg,
    _load_ppg_green,
    _load_ppg_ir,
    _load_ppg_red,
    get_sampling_rate_hz,
)
from .preprocessing import SignalFilter
from .settings import PipelinePaths
from .peak_validation import PpgEcgValidator

__all__ = ["PpgEcgValidator", "SyncedPeakValidator", "generate_peak_validation_reports"]

logger = logging.getLogger(__name__)

SYNCED_ROOT_NAME = "DataSyncedECGPartisipan"
# Keep the legacy report boundary stable for callers that still consume this
# adapter.  The canonical validator writes to ``data/artifacts/validation``.
OUT_REL = Path("reports") / "peaks"
PHASE_MAP = {"Relaks": "Relax", "Stress": "Task"}
PTT_MIN_MS = 80.0
PTT_MAX_MS = 450.0
DIRECT_TOLERANCE_MS = 150.0

CHANNELS = {
    "ppg_green": ("PPG Green", _load_ppg_green, PPG_FS),
    "ppg_ir": ("PPG IR", _load_ppg_ir, PPG_IR_FS),
    "ppg_red": ("PPG Red", _load_ppg_red, PPG_RED_FS),
}


@dataclass(frozen=True)
class SyncedPeakReportOutputs:
    output_dir: Path
    summary_path: Path
    channel_count: int
    figure_count: int


@dataclass(frozen=True)
class PeakDetection:
    clean_signal: np.ndarray
    peak_indices: np.ndarray
    peak_times_ms: np.ndarray
    fs_hz: float


class SyncedPeakValidator:
    """PPG-vs-ECG peak-to-peak report from synchronized ECG raw folders."""

    def __init__(self, paths: PipelinePaths) -> None:
        self.paths = paths
        self.synced_root = paths.project_root / "data" / "raw" / SYNCED_ROOT_NAME
        self.out_dir = paths.project_root / OUT_REL
        self.per_subject_dir = self.out_dir / "per_subject"

    def run(self) -> SyncedPeakReportOutputs:
        self.per_subject_dir.mkdir(parents=True, exist_ok=True)
        sync_summary = self._load_sync_summary()
        rows: list[dict[str, Any]] = []
        participant_rows: dict[str, list[dict[str, Any]]] = {}
        figure_count = 0

        for phase_dir in sorted(self.synced_root.glob("ID-*/*")):
            if not phase_dir.is_dir() or phase_dir.name not in PHASE_MAP:
                continue
            participant = phase_dir.parent.name
            phase_original = phase_dir.name
            phase = PHASE_MAP[phase_original]
            ecg_files = sorted(phase_dir.glob("*_ECG_*_synced.csv"))
            if not ecg_files:
                rows.append(
                    self._failed_row(
                        participant, phase_original, phase, "", "missing_synced_ecg"
                    )
                )
                continue
            ecg_path = ecg_files[0]
            session_rows: list[dict[str, Any]] = []

            for channel, (marker, loader, default_fs) in CHANNELS.items():
                ppg_files = sorted(phase_dir.glob(f"*{marker}*.csv"))
                selected = (
                    self._selected_channel(sync_summary, participant, phase_original)
                    == channel
                )
                if not ppg_files:
                    row = self._failed_row(
                        participant,
                        phase_original,
                        phase,
                        channel,
                        "missing_ppg",
                        selected,
                    )
                    rows.append(row)
                    session_rows.append(row)
                    participant_rows.setdefault(participant, []).append(row)
                    continue
                try:
                    row, figures = self._process_channel(
                        participant=participant,
                        phase_original=phase_original,
                        phase=phase,
                        channel=channel,
                        is_selected_reference=selected,
                        ecg_path=ecg_path,
                        ppg_path=ppg_files[0],
                        ppg_loader=loader,
                        ppg_default_fs=default_fs,
                        sync_summary=sync_summary,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Synced peak validation failed for %s/%s/%s: %s",
                        participant,
                        phase_original,
                        channel,
                        exc,
                    )
                    row = self._failed_row(
                        participant, phase_original, phase, channel, str(exc), selected
                    )
                    figures = 0
                rows.append(row)
                session_rows.append(row)
                participant_rows.setdefault(participant, []).append(row)
                figure_count += figures

            self._plot_session_overview(
                participant, phase_original, phase, session_rows
            )
            figure_count += 1

        summary = pd.DataFrame(rows)
        summary_path = self.out_dir / "summary_metrics.csv"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        self._write_cohort_outputs(summary)

        import json

        def _json_clean(obj):
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return None if np.isnan(obj) else float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _json_clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_clean(v) for v in obj]
            return obj

        for participant, p_rows in participant_rows.items():
            json_path = self.out_dir / f"{participant}_validation.json"
            with open(json_path, "w") as f:
                cleaned_rows = [_json_clean(r) for r in p_rows]
                json.dump(cleaned_rows, f, indent=2)

        logger.info("Synced PPG-vs-ECG report saved: %s", self.out_dir)
        return SyncedPeakReportOutputs(
            self.out_dir, summary_path, int(len(summary)), figure_count
        )

    def _process_channel(
        self,
        *,
        participant: str,
        phase_original: str,
        phase: str,
        channel: str,
        is_selected_reference: bool,
        ecg_path: Path,
        ppg_path: Path,
        ppg_loader: Any,
        ppg_default_fs: float,
        sync_summary: pd.DataFrame,
    ) -> tuple[dict[str, Any], int]:
        ecg = _load_ecg(ecg_path)
        ppg = ppg_loader(ppg_path)
        if len(ecg) < 30 or len(ppg) < 30:
            raise ValueError("signal_too_short")

        ecg_det = self._detect_ecg(ecg)
        ppg_det = self._detect_ppg(ppg, ppg_default_fs)
        if len(ecg_det.peak_times_ms) < 3 or len(ppg_det.peak_times_ms) < 3:
            raise ValueError("too_few_peaks")

        direct = self._match_direct(ecg_det.peak_times_ms, ppg_det.peak_times_ms)
        ptt = self._match_ptt(ecg_det.peak_times_ms, ppg_det.peak_times_ms)
        ibi = self._ibi_comparison(ecg_det.peak_times_ms, ppg_det.peak_times_ms)
        quality = self._quality(ptt["f1"], ibi["pearson_r"], ptt["timing_mae_ms"])

        out = self.per_subject_dir / participant / phase_original / channel
        out.mkdir(parents=True, exist_ok=True)
        ibi_metrics = {k: v for k, v in ibi.items() if k != "comparison"}
        metrics_dict = {
            "participant_code": participant,
            "phase_original": phase_original,
            "phase": phase,
            "sensor_channel": channel,
            "is_selected_reference_channel": is_selected_reference,
            "status": "ok",
            "quality_flag": quality,
            **self._sync_meta(sync_summary, participant, phase_original),
            "ecg_fs_hz": ecg_det.fs_hz,
            "ppg_fs_hz": ppg_det.fs_hz,
            "n_ecg_peaks": len(ecg_det.peak_times_ms),
            "n_ppg_peaks": len(ppg_det.peak_times_ms),
            **{f"direct_{k}": v for k, v in direct.items() if k != "matches"},
            **{f"ptt_{k}": v for k, v in ptt.items() if k != "matches"},
            **ibi_metrics,
        }
        pd.DataFrame([metrics_dict]).to_csv(out / "peak_metrics.csv", index=False)
        pd.concat(
            [
                direct["matches"].assign(match_mode="direct_nearest_ms"),
                ptt["matches"].assign(match_mode="ptt_aware_ms"),
            ],
            ignore_index=True,
        ).to_csv(out / "peak_matches.csv", index=False)
        ibi["comparison"].to_csv(out / "ibi_comparison.csv", index=False)

        self._plot_peak_overlay(
            out / "ppg_vs_ecg_peak_overlay.png",
            participant,
            phase_original,
            channel,
            ecg,
            ppg,
            ecg_det,
            ppg_det,
            ptt["matches"],
        )
        self._plot_ibi_bland_altman(
            out / "ibi_bland_altman.png",
            participant,
            phase_original,
            channel,
            ibi["comparison"],
        )

        return metrics_dict, 2

    def _detect_ecg(self, ecg: pd.DataFrame) -> PeakDetection:
        fs = get_sampling_rate_hz(ecg, default=ECG_FS)
        values = (
            pd.to_numeric(ecg["ecg_mv"], errors="coerce")
            .interpolate(limit_direction="both")
            .to_numpy(dtype=float)
        )
        clean = SignalFilter().apply_all(values, fs=fs)
        _, info = nk.ecg_peaks(
            clean,
            sampling_rate=max(1, int(round(fs))),
            method="pantompkins1985",
            correct_artifacts=True,
        )
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(ecg))]
        return PeakDetection(
            clean, peaks, ecg["unix_ms"].to_numpy(dtype=np.int64)[peaks], fs
        )

    def _detect_ppg(self, ppg: pd.DataFrame, default_fs: float) -> PeakDetection:
        fs = get_sampling_rate_hz(ppg, default=default_fs)
        values = (
            pd.to_numeric(ppg["ppg_raw"], errors="coerce")
            .interpolate(limit_direction="both")
            .to_numpy(dtype=float)
        )
        clean = nk.ppg_clean(
            values, sampling_rate=max(1, int(round(fs))), method="elgendi"
        )
        try:
            info = nk.ppg_findpeaks(
                clean, sampling_rate=max(1, int(round(fs))), method="elgendi"
            )
            peaks = np.asarray(info.get("PPG_Peaks", []), dtype=int)
            if len(peaks) < 3:
                raise ValueError("too_few_peaks")
        except Exception:
            from scipy.signal import find_peaks

            distance = int(0.3 * fs)
            peaks, _ = find_peaks(clean, distance=distance)

        peaks = peaks[(peaks >= 0) & (peaks < len(ppg))]
        return PeakDetection(
            clean, peaks, ppg["unix_ms"].to_numpy(dtype=np.int64)[peaks], fs
        )

    def _match_direct(self, ecg_ts: np.ndarray, ppg_ts: np.ndarray) -> dict[str, Any]:
        matches = []
        used: set[int] = set()
        for i, t in enumerate(ppg_ts):
            diffs = np.abs(ecg_ts - t)
            if len(diffs) == 0:
                continue
            j = int(np.argmin(diffs))
            error = float(diffs[j])
            if error <= DIRECT_TOLERANCE_MS and j not in used:
                used.add(j)
                matches.append(
                    {
                        "ecg_peak_idx": j,
                        "ppg_peak_idx": i,
                        "ecg_unix_ms": int(ecg_ts[j]),
                        "ppg_unix_ms": int(t),
                        "delta_ms": float(t - ecg_ts[j]),
                        "abs_error_ms": error,
                    }
                )
        return self._match_metrics(matches, len(ecg_ts), len(ppg_ts))

    def _match_ptt(self, ecg_ts: np.ndarray, ppg_ts: np.ndarray) -> dict[str, Any]:
        deltas = []
        for t in ecg_ts:
            valid_ppg = ppg_ts[(ppg_ts >= t + PTT_MIN_MS) & (ppg_ts <= t + PTT_MAX_MS)]
            if len(valid_ppg) > 0:
                deltas.append(valid_ppg[0] - t)

        if not deltas:
            return self._match_metrics([], len(ecg_ts), len(ppg_ts), ptt=True)

        median_ptt = float(np.median(deltas))

        matches = []
        used_ppg = set()
        for ecg_i, t in enumerate(ecg_ts):
            expected_ppg_t = t + median_ptt
            diffs = np.abs(ppg_ts - expected_ppg_t)
            if len(diffs) == 0:
                continue
            best_ppg_i = int(np.argmin(diffs))
            actual_t = ppg_ts[best_ppg_i]

            if (
                np.abs(actual_t - expected_ppg_t) <= 150.0
                and best_ppg_i not in used_ppg
            ):
                delta = float(actual_t - t)
                if PTT_MIN_MS <= delta <= PTT_MAX_MS:
                    used_ppg.add(best_ppg_i)
                    matches.append(
                        {
                            "ecg_peak_idx": ecg_i,
                            "ppg_peak_idx": best_ppg_i,
                            "ecg_unix_ms": int(t),
                            "ppg_unix_ms": int(actual_t),
                            "delta_ms": delta,
                            "abs_error_ms": abs(delta),
                        }
                    )

        return self._match_metrics(matches, len(ecg_ts), len(ppg_ts), ptt=True)

    def _match_metrics(
        self, matches: list[dict[str, Any]], n_ecg: int, n_ppg: int, ptt: bool = False
    ) -> dict[str, Any]:
        tp = len(matches)
        fp = max(0, n_ppg - tp)
        fn = max(0, n_ecg - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        deltas = np.array([m["delta_ms"] for m in matches], dtype=float)
        abs_errors = np.array([m["abs_error_ms"] for m in matches], dtype=float)
        prefix_error = (
            abs_errors
            if not ptt
            else np.abs(deltas - np.nanmedian(deltas))
            if len(deltas)
            else np.array([])
        )
        return {
            "matched_beats": tp,
            "false_positive_ppg": fp,
            "false_negative_ecg": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "timing_mae_ms": float(np.nanmean(prefix_error))
            if len(prefix_error)
            else np.nan,
            "timing_rmse_ms": float(np.sqrt(np.nanmean(prefix_error**2)))
            if len(prefix_error)
            else np.nan,
            "median_ptt_ms": float(np.nanmedian(deltas)) if len(deltas) else np.nan,
            "ptt_iqr_ms": float(
                np.nanpercentile(deltas, 75) - np.nanpercentile(deltas, 25)
            )
            if len(deltas)
            else np.nan,
            "matches": pd.DataFrame(matches),
        }

    def _ibi_comparison(self, ecg_ts: np.ndarray, ppg_ts: np.ndarray) -> dict[str, Any]:
        ecg_ibi = np.diff(ecg_ts).astype(float)
        ppg_ibi = np.diff(ppg_ts).astype(float)
        n = min(len(ecg_ibi), len(ppg_ibi))
        comparison = pd.DataFrame(
            {
                "beat_pair_idx": np.arange(n, dtype=int),
                "ecg_ibi_ms": ecg_ibi[:n],
                "ppg_ibi_ms": ppg_ibi[:n],
                "ibi_diff_ms": ppg_ibi[:n] - ecg_ibi[:n],
            }
        )
        if n < 2:
            pear = spear = mae = rmse = bias = loa_low = loa_high = hr_mae = np.nan
        else:
            diff = comparison["ibi_diff_ms"].to_numpy(dtype=float)
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            bias = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1))
            loa_low = bias - 1.96 * sd
            loa_high = bias + 1.96 * sd
            pear = self._safe_corr(
                pearsonr, comparison["ecg_ibi_ms"], comparison["ppg_ibi_ms"]
            )
            spear = self._safe_corr(
                spearmanr, comparison["ecg_ibi_ms"], comparison["ppg_ibi_ms"]
            )

            valid = (comparison["ecg_ibi_ms"] > 0) & (comparison["ppg_ibi_ms"] > 0)
            if valid.sum() > 0:
                ecg_hr = 60000.0 / comparison.loc[valid, "ecg_ibi_ms"]
                ppg_hr = 60000.0 / comparison.loc[valid, "ppg_ibi_ms"]
                hr_mae = float(np.mean(np.abs(ppg_hr - ecg_hr)))
            else:
                hr_mae = np.nan

        return {
            "ibi_pairs": int(n),
            "pearson_r": pear,
            "spearman_r": spear,
            "ibi_mae_ms": mae,
            "ibi_rmse_ms": rmse,
            "hr_mae_bpm": hr_mae,
            "bland_altman_bias_ms": bias,
            "bland_altman_loa_low_ms": loa_low,
            "bland_altman_loa_high_ms": loa_high,
            "comparison": comparison,
        }

    def _plot_peak_overlay(
        self,
        path: Path,
        participant: str,
        phase: str,
        channel: str,
        ecg: pd.DataFrame,
        ppg: pd.DataFrame,
        ecg_det: PeakDetection,
        ppg_det: PeakDetection,
        matches: pd.DataFrame,
    ) -> None:
        ecg_t = (
            ecg["unix_ms"].to_numpy(dtype=np.int64) - int(ecg["unix_ms"].iloc[0])
        ) / 1000.0
        ppg_t = (
            ppg["unix_ms"].to_numpy(dtype=np.int64) - int(ecg["unix_ms"].iloc[0])
        ) / 1000.0
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
        fig.suptitle(f"{participant} {phase} {channel}: ECG synced vs PPG peaks")
        axes[0].plot(
            ecg_t, self._z(ecg_det.clean_signal), linewidth=0.6, label="ECG clean (z)"
        )
        axes[0].plot(
            ppg_t,
            self._z(ppg_det.clean_signal),
            linewidth=0.6,
            alpha=0.8,
            label=f"{channel} clean (z)",
        )
        axes[0].scatter(
            ecg_t[ecg_det.peak_indices],
            self._z(ecg_det.clean_signal)[ecg_det.peak_indices],
            s=10,
            color="red",
            label=f"ECG R ({len(ecg_det.peak_indices)})",
        )
        axes[0].scatter(
            ppg_t[ppg_det.peak_indices],
            self._z(ppg_det.clean_signal)[ppg_det.peak_indices],
            s=10,
            color="black",
            label=f"PPG peaks ({len(ppg_det.peak_indices)})",
        )
        for _, m in matches.head(80).iterrows():
            axes[0].axvline(
                (int(m["ecg_unix_ms"]) - int(ecg["unix_ms"].iloc[0])) / 1000.0,
                color="red",
                alpha=0.04,
            )
        axes[0].set_ylabel("z-score")
        axes[0].legend(loc="upper right")

        zoom_end = min(30.0, float(max(ecg_t.max(), ppg_t.max())))
        axes[1].plot(ecg_t, self._z(ecg_det.clean_signal), linewidth=0.8, label="ECG")
        axes[1].plot(ppg_t, self._z(ppg_det.clean_signal), linewidth=0.8, label="PPG")
        axes[1].set_xlim(0, zoom_end)
        axes[1].set_title("30-second zoom")
        axes[1].legend(loc="upper right")

        ecg_ibi = np.diff(ecg_det.peak_times_ms)
        ppg_ibi = np.diff(ppg_det.peak_times_ms)
        axes[2].plot(
            (ecg_det.peak_times_ms[1:] - int(ecg["unix_ms"].iloc[0])) / 1000.0,
            ecg_ibi,
            label="ECG IBI",
            linewidth=0.8,
        )
        axes[2].plot(
            (ppg_det.peak_times_ms[1:] - int(ecg["unix_ms"].iloc[0])) / 1000.0,
            ppg_ibi,
            label="PPG IBI",
            linewidth=0.8,
        )
        axes[2].set_ylabel("IBI ms")
        axes[2].set_xlabel("Time from ECG start (s)")
        axes[2].legend(loc="upper right")
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _plot_ibi_bland_altman(
        self,
        path: Path,
        participant: str,
        phase: str,
        channel: str,
        comparison: pd.DataFrame,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        fig.suptitle(f"{participant} {phase} {channel}: IBI agreement")
        if comparison.empty:
            for ax in axes:
                ax.text(0.5, 0.5, "No IBI pairs", ha="center", va="center")
        else:
            axes[0].scatter(
                comparison["ecg_ibi_ms"], comparison["ppg_ibi_ms"], s=12, alpha=0.7
            )
            axes[0].set_xlabel("ECG IBI ms")
            axes[0].set_ylabel("PPG IBI ms")
            mean_ibi = (comparison["ecg_ibi_ms"] + comparison["ppg_ibi_ms"]) / 2
            diff = comparison["ibi_diff_ms"]
            bias = float(diff.mean())
            sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
            axes[1].scatter(mean_ibi, diff, s=12, alpha=0.7)
            axes[1].axhline(bias, color="red", label="bias")
            axes[1].axhline(bias + 1.96 * sd, color="gray", linestyle="--", label="LoA")
            axes[1].axhline(bias - 1.96 * sd, color="gray", linestyle="--")
            axes[1].set_xlabel("Mean IBI ms")
            axes[1].set_ylabel("PPG - ECG ms")
            axes[1].legend(loc="upper right")
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _plot_session_overview(
        self,
        participant: str,
        phase_original: str,
        phase: str,
        rows: list[dict[str, Any]],
    ) -> None:
        out = self.per_subject_dir / participant / phase_original
        out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.axis("off")
        if df.empty:
            ax.text(0.5, 0.5, "No channel metrics", ha="center", va="center")
        else:
            cols = [
                "sensor_channel",
                "is_selected_reference_channel",
                "status",
                "quality_flag",
                "ptt_f1",
                "pearson_r",
                "ibi_mae_ms",
            ]
            view = df[[c for c in cols if c in df.columns]].copy()
            table = ax.table(
                cellText=view.fillna("").values, colLabels=view.columns, loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
        ax.set_title(
            f"{participant} {phase_original} ({phase}) synced PPG-vs-ECG overview"
        )
        fig.savefig(out / "session_overview.png", dpi=160)
        plt.close(fig)

    def _write_cohort_outputs(self, summary: pd.DataFrame) -> None:
        if summary.empty:
            return
        ok = summary[summary["status"] == "ok"].copy()
        selected = ok[ok["is_selected_reference_channel"] == True].copy()  # noqa: E712

        channel_summary = self._channel_summary(ok)
        participant_summary = self._participant_summary(selected)
        selected_summary = self._selected_reference_summary(selected)

        channel_summary.to_csv(self.out_dir / "channel_summary.csv", index=False)
        participant_summary.to_csv(
            self.out_dir / "participant_summary.csv", index=False
        )
        selected_summary.to_csv(
            self.out_dir / "selected_reference_summary.csv", index=False
        )

        self._plot_cohort_quality(ok, self.out_dir / "cohort_channel_quality.png")
        self._plot_metric_box(
            ok, "ptt_f1", "PTT-aware F1", self.out_dir / "cohort_ptt_f1_by_channel.png"
        )
        self._plot_metric_box(
            ok,
            "ibi_mae_ms",
            "IBI MAE (ms)",
            self.out_dir / "cohort_ibi_mae_by_channel.png",
        )
        self._plot_selected_reference_heatmap(
            selected, self.out_dir / "selected_reference_heatmap.png"
        )
        (self.out_dir / "ppg_ecg_synced_discussion.md").write_text(
            self._discussion_markdown(
                summary, channel_summary, participant_summary, selected_summary
            ),
            encoding="utf-8",
        )

    def _channel_summary(self, ok: pd.DataFrame) -> pd.DataFrame:
        if ok.empty:
            return pd.DataFrame()
        required = ["ptt_f1", "pearson_r", "ibi_mae_ms", "ptt_median_ptt_ms"]
        for col in required:
            if col not in ok.columns:
                ok[col] = np.nan
        grouped = ok.groupby("sensor_channel", dropna=False)
        out = grouped.agg(
            n_sessions=("sensor_channel", "size"),
            n_acceptable=("quality_flag", lambda s: int((s == "acceptable").sum())),
            n_poor=("quality_flag", lambda s: int((s == "poor").sum())),
            mean_ptt_f1=("ptt_f1", "mean"),
            median_ptt_f1=("ptt_f1", "median"),
            mean_pearson_r=("pearson_r", "mean"),
            median_ibi_mae_ms=("ibi_mae_ms", "median"),
            mean_ibi_mae_ms=("ibi_mae_ms", "mean"),
            median_ptt_ms=("ptt_median_ptt_ms", "median"),
        ).reset_index()
        out["acceptable_rate"] = out["n_acceptable"] / out["n_sessions"].replace(
            0, np.nan
        )
        return out.sort_values(
            ["acceptable_rate", "mean_ptt_f1"], ascending=[False, False]
        )

    def _participant_summary(self, selected: pd.DataFrame) -> pd.DataFrame:
        if selected.empty:
            return pd.DataFrame()
        out = selected.copy()
        for col in [
            "ptt_f1",
            "ibi_mae_ms",
            "pearson_r",
            "ptt_median_ptt_ms",
            "sync_quality",
        ]:
            if col not in out.columns:
                out[col] = np.nan
        ibi_fallback = out["ibi_mae_ms"].median()
        if not np.isfinite(ibi_fallback):
            ibi_fallback = 1000.0
        out["difficulty_score"] = (1.0 - out["ptt_f1"].fillna(0.0)) + out[
            "ibi_mae_ms"
        ].fillna(ibi_fallback) / 1000.0
        out["difficulty_rank"] = (
            out["difficulty_score"].rank(method="dense", ascending=False).astype(int)
        )
        cols = [
            "participant_code",
            "phase_original",
            "phase",
            "sensor_channel",
            "quality_flag",
            "sync_quality",
            "ptt_f1",
            "pearson_r",
            "ibi_mae_ms",
            "ptt_median_ptt_ms",
            "difficulty_score",
            "difficulty_rank",
        ]
        return out[[c for c in cols if c in out.columns]].sort_values("difficulty_rank")

    def _selected_reference_summary(self, selected: pd.DataFrame) -> pd.DataFrame:
        if selected.empty:
            return pd.DataFrame()
        out = (
            selected.groupby(["sensor_channel", "quality_flag"], dropna=False)
            .size()
            .reset_index(name="n_selected_sessions")
        )
        totals = (
            selected.groupby("sensor_channel", dropna=False)
            .size()
            .rename("n_selected_total")
            .reset_index()
        )
        out = out.merge(totals, on="sensor_channel", how="left")
        out["quality_rate_within_sensor"] = out["n_selected_sessions"] / out[
            "n_selected_total"
        ].replace(0, np.nan)
        return out.sort_values(
            ["n_selected_total", "sensor_channel"], ascending=[False, True]
        )

    def _plot_cohort_quality(self, ok: pd.DataFrame, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        if ok.empty:
            ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        else:
            table = (
                ok.groupby(["sensor_channel", "quality_flag"])
                .size()
                .unstack(fill_value=0)
            )
            table.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Session-channel count")
            ax.set_title("Quality flag distribution per PPG channel")
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _plot_metric_box(
        self, ok: pd.DataFrame, metric: str, label: str, path: Path
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        if ok.empty or metric not in ok.columns:
            ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        else:
            data = [
                ok.loc[ok["sensor_channel"] == ch, metric].dropna().to_numpy()
                for ch in CHANNELS
            ]
            ax.boxplot(data, tick_labels=list(CHANNELS.keys()), showmeans=True)
            ax.set_ylabel(label)
            ax.set_title(f"{label} by PPG channel")
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _plot_selected_reference_heatmap(
        self, selected: pd.DataFrame, path: Path
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        if selected.empty or "ptt_f1" not in selected.columns:
            ax.text(0.5, 0.5, "No selected reference metrics", ha="center", va="center")
        else:
            pivot = selected.pivot_table(
                index="participant_code",
                columns="phase_original",
                values="ptt_f1",
                aggfunc="mean",
            ).sort_index()
            im = ax.imshow(
                pivot.fillna(0).to_numpy(),
                aspect="auto",
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
            ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index, fontsize=7)
            ax.set_title("Selected reference channel PTT-aware F1")
            fig.colorbar(im, ax=ax, label="PTT-aware F1")
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _discussion_markdown(
        self,
        summary: pd.DataFrame,
        channel_summary: pd.DataFrame,
        participant_summary: pd.DataFrame,
        selected_summary: pd.DataFrame,
    ) -> str:
        total = len(summary)
        ok = int((summary["status"] == "ok").sum()) if "status" in summary else 0
        selected = (
            summary[summary["is_selected_reference_channel"].fillna(False).astype(bool)]
            if "is_selected_reference_channel" in summary
            else pd.DataFrame()
        )
        most_selected = "tidak tersedia"
        if not selected.empty:
            most_selected = str(selected["sensor_channel"].value_counts().idxmax())
        best_sensor = "tidak tersedia"
        if not channel_summary.empty:
            best_sensor = str(channel_summary.iloc[0]["sensor_channel"])
        hardest = (
            participant_summary.head(5)
            if not participant_summary.empty
            else pd.DataFrame()
        )

        hard_lines = (
            "\n".join(
                f"- {r.participant_code} {r.phase_original} ({r.sensor_channel}): F1={r.ptt_f1:.3f}, "
                f"IBI MAE={r.ibi_mae_ms:.2f} ms, quality={r.quality_flag}"
                for r in hardest.itertuples()
            )
            or "- Tidak tersedia."
        )

        channel_table = (
            channel_summary.to_markdown(index=False)
            if not channel_summary.empty
            else "_Tidak tersedia._"
        )
        selected_table = (
            selected_summary.to_markdown(index=False)
            if not selected_summary.empty
            else "_Tidak tersedia._"
        )

        return (
            "# Diskusi PPG-vs-ECG Synced Peak-to-Peak\n\n"
            "## Ringkasan\n"
            f"Analisis ini mengevaluasi {ok} dari {total} kombinasi participant-phase-channel yang berhasil diproses "
            "dari `DataSyncedECGPartisipan`. Metrik utama adalah `ptt_f1`, yaitu pencocokan beat dengan asumsi "
            "PPG peak muncul setelah ECG R-peak akibat pulse transit time.\n\n"
            f"Channel yang paling sering menjadi selected reference saat sinkronisasi adalah `{most_selected}`. "
            f"Berdasarkan agregasi channel, sensor dengan ringkasan terbaik saat ini adalah `{best_sensor}`.\n\n"
            "## Perbandingan Sensor\n"
            f"{channel_table}\n\n"
            "## Selected Reference Channel\n"
            f"{selected_table}\n\n"
            "## Failure Analysis\n"
            "Participant-phase tersulit berdasarkan selected reference channel:\n\n"
            f"{hard_lines}\n\n"
            "Sesi dengan quality `poor` menunjukkan bahwa ECG synced berguna untuk audit visual dan kuantitatif, "
            "tetapi belum otomatis membuktikan bahwa seluruh ECG valid sebagai ground truth. Penyebab yang paling "
            "mungkin adalah residual timestamp drift, motion artifact, kontak sensor wearable yang tidak stabil, "
            "atau peak morphology PPG yang lemah pada sebagian fase.\n\n"
            "## Implikasi\n"
            "Hasil ini adalah lapisan validasi PPG-vs-ECG dan tidak mengubah pipeline stress classification yang "
            "menggunakan fitur PPG-only. Interpretasi untuk tesis sebaiknya memisahkan validitas ECG sebagai referensi "
            "dari performa model stress PPG-only.\n"
        )

    def _load_sync_summary(self) -> pd.DataFrame:
        path = self.synced_root / "sync_summary.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def _selected_channel(
        self, sync_summary: pd.DataFrame, participant: str, phase_original: str
    ) -> str:
        if sync_summary.empty:
            return ""
        rows = sync_summary[
            (sync_summary["participant_code"] == participant)
            & (sync_summary["phase"] == phase_original)
        ]
        if rows.empty:
            return ""
        return str(rows.iloc[0].get("selected_reference_channel", ""))

    def _sync_meta(
        self, sync_summary: pd.DataFrame, participant: str, phase_original: str
    ) -> dict[str, Any]:
        if sync_summary.empty:
            return {}
        rows = sync_summary[
            (sync_summary["participant_code"] == participant)
            & (sync_summary["phase"] == phase_original)
        ]
        if rows.empty:
            return {}
        row = rows.iloc[0]
        return {
            "sync_status": row.get("status", ""),
            "selected_reference_channel": row.get("selected_reference_channel", ""),
            "sync_quality": row.get("sync_quality", ""),
            "sync_pearson_r_after": row.get("pearson_r_after", np.nan),
            "total_shift_ms": row.get("total_shift_ms", np.nan),
        }

    def _failed_row(
        self,
        participant: str,
        phase_original: str,
        phase: str,
        channel: str,
        reason: str,
        selected: bool = False,
    ) -> dict[str, Any]:
        return {
            "participant_code": participant,
            "phase_original": phase_original,
            "phase": phase,
            "sensor_channel": channel,
            "is_selected_reference_channel": selected,
            "status": "failed",
            "quality_flag": "failed",
            "exclusion_reason": reason,
        }

    @staticmethod
    def _quality(f1: float, pearson_r: float, timing_mae: float) -> str:
        if not np.isfinite(f1):
            return "failed"
        if f1 >= 0.90 and np.isfinite(pearson_r) and pearson_r >= 0.85:
            return "excellent"
        if f1 >= 0.85 and np.isfinite(pearson_r) and pearson_r >= 0.80:
            return "good"
        if (
            f1 >= 0.50
            or (np.isfinite(pearson_r) and pearson_r >= 0.50)
            or (np.isfinite(timing_mae) and timing_mae <= 80)
        ):
            return "acceptable"
        return "poor"

    @staticmethod
    def _safe_corr(fn: Any, left: pd.Series, right: pd.Series) -> float:
        try:
            value = fn(left, right)
            return float(
                getattr(value, "statistic", getattr(value, "correlation", value[0]))
            )
        except Exception:
            return float("nan")

    @staticmethod
    def _z(values: np.ndarray) -> np.ndarray:
        std = float(np.nanstd(values))
        if not np.isfinite(std) or std == 0:
            return np.zeros_like(values, dtype=float)
        return (values - float(np.nanmean(values))) / std


def generate_peak_validation_reports(project_root: Path, config: Any) -> None:
    from PipelinePython.common.settings import DEFAULT_PATHS

    validator = SyncedPeakValidator(DEFAULT_PATHS)
    validator.run()
