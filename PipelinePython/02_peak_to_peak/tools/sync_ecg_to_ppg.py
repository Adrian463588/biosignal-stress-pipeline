from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

from PipelinePython.common.config import ECG_FS, PPG_FS, PPG_IR_FS, PPG_RED_FS
from PipelinePython.common.ingestion import (
    _load_ecg,
    _load_ppg_green,
    _load_ppg_ir,
    _load_ppg_red,
    get_sampling_rate_hz,
)
from PipelinePython.common.preprocessing import SignalFilter
from PipelinePython.common.settings import DEFAULT_PATHS

logger = logging.getLogger(__name__)

PROJECT_ROOT = DEFAULT_PATHS.project_root
RAW_RENAMED = PROJECT_ROOT / "data" / "raw" / "DataRenamedPartisipan"
RAW_TRIMMED = PROJECT_ROOT / "data" / "raw" / "DataTrimmedPartisipan"
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "DataSyncedECGPartisipan"

CHANNELS = {
    "ppg_green": ("PPG Green", _load_ppg_green, PPG_FS),
    "ppg_ir": ("PPG IR", _load_ppg_ir, PPG_IR_FS),
    "ppg_red": ("PPG Red", _load_ppg_red, PPG_RED_FS),
}
PHASE_DIRS = ("Relaks", "Stress")


@dataclass(frozen=True)
class PeakSeries:
    peak_times_ms: np.ndarray
    ibi_times_s: np.ndarray
    ibi_ms: np.ndarray
    clean_signal: np.ndarray
    peak_indices: np.ndarray


@dataclass(frozen=True)
class SyncCandidate:
    channel: str
    status: str
    fine_lag_seconds: float
    pearson_r_before: float
    pearson_r_after: float
    crosscorr_value: float
    n_ecg_peaks: int
    n_ppg_peaks: int
    peak_ratio: float
    sync_quality: str
    exclusion_reason: str

    @property
    def score(self) -> float:
        if self.status != "ok":
            return -np.inf
        ratio_penalty = abs(1.0 - self.peak_ratio)
        return self.pearson_r_after - 0.15 * ratio_penalty


def _duration_seconds(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float((df["unix_ms"].iloc[-1] - df["unix_ms"].iloc[0]) / 1000.0)


def regularize_ecg_timestamps(
    ecg: pd.DataFrame, default_fs: float = ECG_FS
) -> pd.DataFrame:
    """Rebuild Polar ECG unix_ms from sample index when exported timestamps drift/gap."""
    if ecg.empty:
        return ecg.copy()
    fs = get_sampling_rate_hz(ecg, default=default_fs)
    start_ms = int(ecg["unix_ms"].iloc[0])
    out = ecg.copy()
    out["unix_ms"] = np.rint(
        start_ms + np.arange(len(out), dtype=float) * (1000.0 / fs)
    ).astype(np.int64)
    return out


def _overlaps(ecg: pd.DataFrame, ppg_frames: Iterable[pd.DataFrame]) -> bool:
    ppg_frames = [df for df in ppg_frames if df is not None and not df.empty]
    if ecg.empty or not ppg_frames:
        return False
    ppg_start = min(int(df["unix_ms"].iloc[0]) for df in ppg_frames)
    ppg_end = max(int(df["unix_ms"].iloc[-1]) for df in ppg_frames)
    return (
        int(ecg["unix_ms"].iloc[0]) <= ppg_end
        and int(ecg["unix_ms"].iloc[-1]) >= ppg_start
    )


def coarse_shift_to_ppg(ecg: pd.DataFrame, ppg_frames: Iterable[pd.DataFrame]) -> int:
    frames = [df for df in ppg_frames if df is not None and not df.empty]
    if not frames or ecg.empty or _overlaps(ecg, frames):
        return 0
    ppg_start = min(int(df["unix_ms"].iloc[0]) for df in frames)
    ecg_start = int(ecg["unix_ms"].iloc[0])
    return ppg_start - ecg_start


def _detect_peak_series(
    df: pd.DataFrame, signal_col: str, signal_type: str, default_fs: float
) -> PeakSeries:
    fs = get_sampling_rate_hz(df, default=default_fs)
    fs_int = max(1, int(round(fs)))
    values = (
        pd.to_numeric(df[signal_col], errors="coerce")
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )
    if len(values) < fs_int * 10:
        raise ValueError("signal_too_short")

    if signal_type == "ecg":
        clean = SignalFilter().apply_all(values, fs=fs)
        _, info = nk.ecg_peaks(
            clean,
            sampling_rate=fs_int,
            method="pantompkins1985",
            correct_artifacts=True,
        )
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    else:
        clean = nk.ppg_clean(values, sampling_rate=fs_int, method="elgendi")
        info = nk.ppg_findpeaks(clean, sampling_rate=fs_int, method="elgendi")
        peaks = np.asarray(info.get("PPG_Peaks", []), dtype=int)

    peaks = peaks[(peaks >= 0) & (peaks < len(df))]
    if len(peaks) < 4:
        raise ValueError("too_few_peaks")

    peak_times = df["unix_ms"].to_numpy(dtype=np.int64)[peaks]
    ibi = np.diff(peak_times).astype(float)
    ibi_times = peak_times[1:].astype(float) / 1000.0
    valid = (ibi >= 300.0) & (ibi <= 2000.0)
    if valid.sum() < 3:
        raise ValueError("too_few_valid_ibi")
    return PeakSeries(
        peak_times_ms=peak_times,
        ibi_times_s=ibi_times[valid],
        ibi_ms=ibi[valid],
        clean_signal=clean,
        peak_indices=peaks,
    )


def _pearson_at_lag(
    ecg_series: PeakSeries, ppg_series: PeakSeries, lag_s: float
) -> float:
    ecg_t = ecg_series.ibi_times_s + lag_s
    ppg_t = ppg_series.ibi_times_s
    start = max(float(ecg_t.min()), float(ppg_t.min()))
    end = min(float(ecg_t.max()), float(ppg_t.max()))
    mask = (ppg_t >= start) & (ppg_t <= end)
    if mask.sum() < 3:
        return float("nan")
    ecg_interp = np.interp(ppg_t[mask], ecg_t, ecg_series.ibi_ms)
    ppg_ibi = ppg_series.ibi_ms[mask]
    if np.std(ecg_interp) == 0 or np.std(ppg_ibi) == 0:
        return float("nan")
    return float(np.corrcoef(ecg_interp, ppg_ibi)[0, 1])


def estimate_fine_lag_seconds(
    ecg_series: PeakSeries,
    ppg_series: PeakSeries,
    max_lag_s: float = 90.0,
    step_s: float = 0.05,
) -> tuple[float, float, float]:
    lags = np.arange(-max_lag_s, max_lag_s + step_s / 2, step_s)
    scores = np.array(
        [_pearson_at_lag(ecg_series, ppg_series, float(lag)) for lag in lags]
    )
    finite = np.isfinite(scores)
    before = _pearson_at_lag(ecg_series, ppg_series, 0.0)
    if not finite.any():
        raise ValueError("no_usable_lag")
    idx = int(np.nanargmax(scores))
    return (
        float(lags[idx]),
        float(scores[idx]),
        float(before) if np.isfinite(before) else float("nan"),
    )


def _quality(corr: float, peak_ratio: float) -> str:
    if not np.isfinite(corr):
        return "failed"
    if corr >= 0.85 and 0.75 <= peak_ratio <= 1.25:
        return "good"
    if corr >= 0.50:
        return "acceptable"
    return "poor"


def evaluate_channel(
    ecg_series: PeakSeries, ppg: pd.DataFrame, channel: str, default_fs: float
) -> SyncCandidate:
    try:
        ppg_series = _detect_peak_series(ppg, "ppg_raw", "ppg", default_fs)
        lag, after, before = estimate_fine_lag_seconds(ecg_series, ppg_series)
        ratio = len(ppg_series.peak_times_ms) / max(1, len(ecg_series.peak_times_ms))
        quality = _quality(after, ratio)
        return SyncCandidate(
            channel=channel,
            status="ok",
            fine_lag_seconds=lag,
            pearson_r_before=before,
            pearson_r_after=after,
            crosscorr_value=after,
            n_ecg_peaks=len(ecg_series.peak_times_ms),
            n_ppg_peaks=len(ppg_series.peak_times_ms),
            peak_ratio=ratio,
            sync_quality=quality,
            exclusion_reason="",
        )
    except Exception as exc:  # noqa: BLE001
        return SyncCandidate(
            channel=channel,
            status="failed",
            fine_lag_seconds=0.0,
            pearson_r_before=float("nan"),
            pearson_r_after=float("nan"),
            crosscorr_value=float("nan"),
            n_ecg_peaks=len(ecg_series.peak_times_ms),
            n_ppg_peaks=0,
            peak_ratio=float("nan"),
            sync_quality="failed",
            exclusion_reason=str(exc),
        )


def choose_best_candidate(candidates: list[SyncCandidate]) -> SyncCandidate:
    ok = [c for c in candidates if c.status == "ok"]
    if not ok:
        return SyncCandidate(
            channel="",
            status="failed",
            fine_lag_seconds=0.0,
            pearson_r_before=float("nan"),
            pearson_r_after=float("nan"),
            crosscorr_value=float("nan"),
            n_ecg_peaks=0,
            n_ppg_peaks=0,
            peak_ratio=float("nan"),
            sync_quality="failed",
            exclusion_reason="no_channel_candidate",
        )
    return max(ok, key=lambda c: c.score)


def _format_datetime_str(unix_ms: pd.Series) -> pd.Series:
    ts = pd.to_datetime(unix_ms.astype("int64"), unit="ms", utc=True).dt.tz_convert(
        "Asia/Jakarta"
    )
    return ts.dt.strftime("%d %b %Y %H:%M:%S:%f %z").str.replace(
        r"(\d{3})\d{3} ", r"\1 ", regex=True
    )


def write_synced_ecg(
    ecg: pd.DataFrame, output_path: Path, total_shift_ms: int
) -> pd.DataFrame:
    out = ecg.copy()
    out["unix_ms"] = out["unix_ms"].astype("int64") + int(total_shift_ms)
    out["datetime_str"] = _format_datetime_str(out["unix_ms"])
    out = out[
        ["ecg_mv", "is_peak", "unix_ms", "datetime_str", "acc_x", "acc_y", "acc_z"]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, header=False)
    return out


def _find_phase_dir(root: Path, participant: str, phase: str) -> Path | None:
    p_root = root / participant
    candidates = [phase]
    if phase == "Relaks":
        candidates.append("Relaksasi")
    for name in candidates:
        path = p_root / name
        if path.exists():
            return path
    return None


def _load_ppg_refs(
    trimmed_phase_dir: Path,
) -> dict[str, tuple[Path, pd.DataFrame, float]]:
    refs: dict[str, tuple[Path, pd.DataFrame, float]] = {}
    for channel, (needle, loader, default_fs) in CHANNELS.items():
        matches = sorted(trimmed_phase_dir.glob(f"*{needle}*.csv"))
        if matches:
            refs[channel] = (matches[0], loader(matches[0]), default_fs)
    return refs


def _plot_validation(
    output_path: Path,
    ecg_synced: pd.DataFrame,
    ppg: pd.DataFrame,
    candidate: SyncCandidate,
    participant: str,
    phase: str,
) -> None:
    try:
        ecg_series = _detect_peak_series(ecg_synced, "ecg_mv", "ecg", ECG_FS)
        ppg_series = _detect_peak_series(ppg, "ppg_raw", "ppg", PPG_FS)
    except Exception:
        return
    t0 = max(ecg_synced["unix_ms"].min(), ppg["unix_ms"].min())
    ecg_t = (ecg_synced["unix_ms"].to_numpy() - t0) / 1000.0
    ppg_t = (ppg["unix_ms"].to_numpy() - t0) / 1000.0
    ecg_norm = (ecg_series.clean_signal - np.nanmean(ecg_series.clean_signal)) / (
        np.nanstd(ecg_series.clean_signal) + 1e-9
    )
    ppg_norm = (ppg_series.clean_signal - np.nanmean(ppg_series.clean_signal)) / (
        np.nanstd(ppg_series.clean_signal) + 1e-9
    )

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(ecg_t, ecg_norm, lw=0.7, label="ECG cleaned normalized")
    ax.plot(
        ppg_t,
        ppg_norm,
        lw=0.7,
        alpha=0.75,
        label=f"{candidate.channel} cleaned normalized",
    )
    ax.scatter(
        ecg_t[ecg_series.peak_indices],
        ecg_norm[ecg_series.peak_indices],
        s=14,
        label="ECG R-peaks",
    )
    ax.scatter(
        ppg_t[ppg_series.peak_indices],
        ppg_norm[ppg_series.peak_indices],
        s=14,
        label="PPG peaks",
    )
    ax.set_xlim(0, min(60, max(ecg_t.max(), ppg_t.max())))
    ax.set_title(
        f"{participant} {phase} ECG sync vs {candidate.channel} | "
        f"lag={candidate.fine_lag_seconds:.2f}s r={candidate.pearson_r_after:.3f} {candidate.sync_quality}"
    )
    ax.set_xlabel("seconds from synced overlap")
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def sync_one(
    participant: str, phase: str, renamed_root: Path, trimmed_root: Path, out_root: Path
) -> dict:
    raw_phase_dir = _find_phase_dir(renamed_root, participant, phase)
    trimmed_phase_dir = _find_phase_dir(trimmed_root, participant, phase)
    if raw_phase_dir is None or trimmed_phase_dir is None:
        return {
            "participant_code": participant,
            "phase": phase,
            "status": "failed",
            "exclusion_reason": "missing_phase_dir",
        }

    ecg_files = sorted(raw_phase_dir.glob("*_ECG_*.csv"))
    if not ecg_files:
        return {
            "participant_code": participant,
            "phase": phase,
            "status": "failed",
            "exclusion_reason": "missing_raw_ecg",
        }
    refs = _load_ppg_refs(trimmed_phase_dir)
    if not refs:
        return {
            "participant_code": participant,
            "phase": phase,
            "status": "failed",
            "exclusion_reason": "missing_trimmed_ppg",
        }

    ecg_path = ecg_files[0]
    ecg_raw = _load_ecg(ecg_path)
    ecg = regularize_ecg_timestamps(ecg_raw)
    coarse_ms = coarse_shift_to_ppg(ecg, [df for _, df, _ in refs.values()])
    ecg_coarse = ecg.copy()
    ecg_coarse["unix_ms"] = ecg_coarse["unix_ms"].astype("int64") + coarse_ms

    try:
        ecg_series = _detect_peak_series(ecg_coarse, "ecg_mv", "ecg", ECG_FS)
    except Exception as exc:  # noqa: BLE001
        return {
            "participant_code": participant,
            "phase": phase,
            "status": "failed",
            "exclusion_reason": f"ecg_{exc}",
        }

    candidates = [
        evaluate_channel(ecg_series, df, ch, default_fs)
        for ch, (_, df, default_fs) in refs.items()
    ]
    for candidate in candidates:
        _append_candidate_detail(out_root, participant, phase, candidate)
    best = choose_best_candidate(candidates)
    if best.status != "ok":
        return {
            "participant_code": participant,
            "phase": phase,
            "status": "failed",
            "exclusion_reason": best.exclusion_reason,
        }

    total_shift_ms = int(round(coarse_ms + best.fine_lag_seconds * 1000.0))
    out_phase_dir = out_root / participant / phase
    out_name = ecg_path.name.replace(".csv", "_synced.csv")
    ecg_out_path = out_phase_dir / out_name
    synced = write_synced_ecg(ecg, ecg_out_path, total_shift_ms)

    for source_path, _, _ in refs.values():
        shutil.copy2(source_path, out_phase_dir / source_path.name)

    plot_path = out_root / "sync_plots" / f"{participant}_{phase}_validation.png"
    selected_ppg = refs[best.channel][1]
    _plot_validation(plot_path, synced, selected_ppg, best, participant, phase)

    return {
        "participant_code": participant,
        "phase": phase,
        "status": "ok",
        "selected_reference_channel": best.channel,
        "coarse_shift_ms": coarse_ms,
        "fine_lag_seconds": best.fine_lag_seconds,
        "total_shift_ms": total_shift_ms,
        "pearson_r_before": best.pearson_r_before,
        "pearson_r_after": best.pearson_r_after,
        "crosscorr_value": best.crosscorr_value,
        "n_ecg_peaks": best.n_ecg_peaks,
        "n_ppg_peaks": best.n_ppg_peaks,
        "ecg_duration_s": _duration_seconds(ecg),
        "ppg_duration_s": _duration_seconds(selected_ppg),
        "sync_quality": best.sync_quality,
        "exclusion_reason": "",
        "ecg_output_path": str(ecg_out_path),
        "plot_path": str(plot_path),
    }


def _append_candidate_detail(
    out_root: Path, participant: str, phase: str, candidate: SyncCandidate
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "sync_candidate_details.csv"
    row = pd.DataFrame(
        [
            {
                "participant_code": participant,
                "phase": phase,
                "candidate_channel": candidate.channel,
                "status": candidate.status,
                "fine_lag_seconds": candidate.fine_lag_seconds,
                "pearson_r_before": candidate.pearson_r_before,
                "pearson_r_after": candidate.pearson_r_after,
                "crosscorr_value": candidate.crosscorr_value,
                "n_ecg_peaks": candidate.n_ecg_peaks,
                "n_ppg_peaks": candidate.n_ppg_peaks,
                "peak_ratio": candidate.peak_ratio,
                "sync_quality": candidate.sync_quality,
                "score": candidate.score,
                "exclusion_reason": candidate.exclusion_reason,
            }
        ]
    )
    row.to_csv(path, mode="a", index=False, header=not path.exists())


def run(
    renamed_root: Path = RAW_RENAMED,
    trimmed_root: Path = RAW_TRIMMED,
    out_root: Path = OUT_ROOT,
    participant: str | None = None,
) -> pd.DataFrame:
    candidate_path = out_root / "sync_candidate_details.csv"
    if candidate_path.exists():
        candidate_path.unlink()
    participants = (
        [participant]
        if participant
        else sorted(p.name for p in trimmed_root.glob("ID-*") if p.is_dir())
    )
    rows: list[dict] = []
    for p in participants:
        for phase in PHASE_DIRS:
            logger.info("Synchronizing ECG: %s %s", p, phase)
            rows.append(sync_one(p, phase, renamed_root, trimmed_root, out_root))

    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "sync_summary.csv", index=False)
    flags = df[
        df["status"].ne("ok")
        | df.get("sync_quality", pd.Series(index=df.index, dtype=str)).isin(
            ["poor", "failed"]
        )
    ].copy()
    flags.to_csv(out_root / "participant_quality_flags.csv", index=False)
    _write_markdown_report(out_root, df)
    logger.info("ECG synchronization summary saved: %s", out_root / "sync_summary.csv")
    return df


def _write_markdown_report(out_root: Path, summary: pd.DataFrame) -> None:
    detail_path = out_root / "sync_candidate_details.csv"
    detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    lines = [
        "# ECG Synchronization Report",
        "",
        "## Summary",
        f"- Participant-phase rows: {len(summary)}",
        f"- Participants: {summary['participant_code'].nunique() if 'participant_code' in summary else 0}",
        f"- Successful rows: {int(summary['status'].eq('ok').sum()) if 'status' in summary else 0}",
        "",
        "## Selected Reference Channels",
        summary.get("selected_reference_channel", pd.Series(dtype=str))
        .value_counts(dropna=False)
        .to_markdown(),
        "",
        "## Sync Quality",
        summary.get("sync_quality", pd.Series(dtype=str))
        .value_counts(dropna=False)
        .to_markdown(),
        "",
    ]
    if not detail.empty:
        lines.extend(
            [
                "## Candidate Channel Quality",
                pd.crosstab(
                    detail["candidate_channel"], detail["sync_quality"]
                ).to_markdown(),
                "",
                "## Candidate Mean Correlation",
                detail.groupby("candidate_channel")[["pearson_r_after", "peak_ratio"]]
                .mean(numeric_only=True)
                .round(3)
                .to_markdown(),
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "- ECG timestamps are rebuilt from Polar sampling rate before synchronization when raw Unix timestamps drift.",
            "- `good` and `acceptable` rows are suitable for stronger ECG-vs-PPG discussion.",
            "- `poor` rows remain traceable and should not be treated as clean ECG ground truth without visual review.",
        ]
    )
    (out_root / "sync_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize raw Polar ECG timestamps to trimmed PPG sessions."
    )
    parser.add_argument("--participant", help="Optional participant code, e.g. ID-02")
    parser.add_argument("--renamed-root", type=Path, default=RAW_RENAMED)
    parser.add_argument("--trimmed-root", type=Path, default=RAW_TRIMMED)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(args.renamed_root, args.trimmed_root, args.out_root, args.participant)


if __name__ == "__main__":
    main()
