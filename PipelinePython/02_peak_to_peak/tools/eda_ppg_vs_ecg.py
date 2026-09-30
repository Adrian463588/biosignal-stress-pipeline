"""
eda_ppg_vs_ecg.py
-----------------
Exploratory Data Analysis: PPG vs ECG peak detection & IBI comparison
over 5-minute segments per participant (Relax & Task phases).

Outputs (saved to data/artifacts/eda/ppg_ecg_comparison/):
  - before_after_signals_<ID>.png  : raw vs filtered signals + detected peaks
  - ibi_comparison_<ID>.png        : IBI time-series, scatter, Bland-Altman
  - summary_table.csv              : per-participant metric summary
  - cohort_overview.png            : cohort-level heatmap + box plots

Usage:
    python PipelinePython/02_peak_to_peak/tools/eda_ppg_vs_ecg.py
    python PipelinePython/02_peak_to_peak/tools/eda_ppg_vs_ecg.py --participant ID-28
    python PipelinePython/02_peak_to_peak/tools/eda_ppg_vs_ecg.py --phase Relaks
"""

from __future__ import annotations

# This retained research plotting script intentionally configures Matplotlib
# before importing pyplot and keeps compact plotting statements.
# ruff: noqa: E402, E701, E702

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PipelinePython.common.ingestion import DataLoader
from PipelinePython.common.anchors import AnchorMapper
from PipelinePython.common.preprocessing import SignalFilter, Resampler
from PipelinePython.common.config import ECG_FS, PPG_FS, PPG_IR_FS, PPG_RED_FS
from PipelinePython.common.ingestion import get_sampling_rate_hz
from PipelinePython.common.settings import DEFAULT_PATHS, PipelineSettings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SETTINGS = PipelineSettings()
TARGET_FS = 100.0
FIVE_MIN_SAMPLES = int(5 * 60 * TARGET_FS)  # 30 000 samples @ 100 Hz
BEAT_TOL_MS = 150  # peak-match tolerance

OUT_DIR = DEFAULT_PATHS.eda_dir / "ppg_ecg_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib style
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
    }
)

CHANNEL_COLORS = {
    "PPG_GREEN": "#2ecc71",
    "PPG_IR": "#e67e22",
    "PPG_RED": "#e74c3c",
    "ECG": "#2980b9",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_ecg_peaks(signal: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    _, info = nk.ecg_peaks(
        signal, sampling_rate=int(fs), method="pantompkins1985", correct_artifacts=True
    )
    return np.array(info.get("ECG_R_Peaks", []), dtype=int)


def _detect_ppg_peaks(signal: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    clean = nk.ppg_clean(signal, sampling_rate=int(fs), method="elgendi")
    info = nk.ppg_findpeaks(clean, sampling_rate=int(fs), method="elgendi")
    return np.array(info.get("PPG_Peaks", []), dtype=int)


def _ibi_ms(peaks: np.ndarray, unix_ms: np.ndarray) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([])
    return np.diff(unix_ms[peaks]).astype(float)


def _match_peaks(ecg_ts: np.ndarray, ppg_ts: np.ndarray, tol: int = BEAT_TOL_MS):
    matched, used, errors = 0, set(), []
    for t in ppg_ts:
        diffs = np.abs(ecg_ts - t)
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tol and idx not in used:
            matched += 1
            used.add(idx)
            errors.append(float(diffs[idx]))
    fp = max(0, len(ppg_ts) - matched)
    fn = max(0, len(ecg_ts) - matched)
    return matched, fp, fn, errors


def _stats(ecg_ibi: np.ndarray, ppg_ibi: np.ndarray) -> dict:
    n = min(len(ecg_ibi), len(ppg_ibi))
    if n < 3:
        return {}
    e, p = ecg_ibi[:n], ppg_ibi[:n]
    diff = p - e
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    try:
        pr = float(pearsonr(e, p).statistic)
    except Exception:
        pr = float("nan")
    try:
        sr = float(spearmanr(e, p).correlation)
    except Exception:
        sr = float("nan")
    return dict(
        n=n,
        bias_ms=bias,
        sd_ms=sd,
        loa_low=bias - 1.96 * sd,
        loa_high=bias + 1.96 * sd,
        mae_ms=float(np.mean(np.abs(diff))),
        rmse_ms=float(np.sqrt(np.mean(diff**2))),
        pearson_r=pr,
        spearman_r=sr,
    )


# ---------------------------------------------------------------------------
# Plot 1: before / after signals + peaks
# ---------------------------------------------------------------------------


def plot_before_after(
    pid: str,
    phase: str,
    raw_ecg: np.ndarray,
    filt_ecg: np.ndarray,
    raw_ppg: np.ndarray,
    filt_ppg: np.ndarray,
    unix_ms: np.ndarray,
    ecg_peaks: np.ndarray,
    ppg_peaks: np.ndarray,
    channel: str = "PPG_IR",
):
    t_sec = (unix_ms - unix_ms[0]) / 1000.0

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{pid} – {phase} – 5 min segment  |  ECG vs {channel}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ---- Column 0: BEFORE -----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_sec, raw_ecg, lw=0.6, color=CHANNEL_COLORS["ECG"], alpha=0.8)
    ax.set_title("ECG Raw", fontsize=9)
    ax.set_ylabel("mV")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_sec, raw_ppg, lw=0.6, color=CHANNEL_COLORS[channel], alpha=0.8)
    ax.set_title(f"{channel} Raw", fontsize=9)
    ax.set_ylabel("ADC")

    # ---- Column 1: AFTER -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_sec, filt_ecg, lw=0.6, color=CHANNEL_COLORS["ECG"])
    if len(ecg_peaks):
        ax.scatter(
            t_sec[ecg_peaks],
            filt_ecg[ecg_peaks],
            s=20,
            color="red",
            zorder=5,
            label=f"R-peaks ({len(ecg_peaks)})",
        )
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title("ECG Filtered + R-peaks", fontsize=9)
    ax.set_ylabel("mV")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_sec, filt_ppg, lw=0.6, color=CHANNEL_COLORS[channel])
    if len(ppg_peaks):
        ax.scatter(
            t_sec[ppg_peaks],
            filt_ppg[ppg_peaks],
            s=20,
            color="darkred",
            zorder=5,
            label=f"PPG peaks ({len(ppg_peaks)})",
        )
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"{channel} Filtered + peaks", fontsize=9)
    ax.set_ylabel("ADC")

    # ---- Row 2: IBI overlay ----------------------------------------------
    ecg_ibi = _ibi_ms(ecg_peaks, unix_ms)
    ppg_ibi = _ibi_ms(ppg_peaks, unix_ms)

    ax = fig.add_subplot(gs[2, :])
    if len(ecg_ibi):
        t_ecg = (unix_ms[ecg_peaks[1:]] - unix_ms[0]) / 1000.0
        ax.plot(
            t_ecg,
            ecg_ibi,
            "o-",
            ms=3,
            lw=1,
            color=CHANNEL_COLORS["ECG"],
            label=f"ECG IBI (n={len(ecg_ibi)})",
        )
    if len(ppg_ibi):
        t_ppg = (unix_ms[ppg_peaks[1:]] - unix_ms[0]) / 1000.0
        ax.plot(
            t_ppg,
            ppg_ibi,
            "s--",
            ms=3,
            lw=1,
            color=CHANNEL_COLORS[channel],
            alpha=0.85,
            label=f"{channel} IBI (n={len(ppg_ibi)})",
        )
    ax.set_title("IBI Time-series Overlay", fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("IBI (ms)")
    ax.legend(fontsize=7)

    # ---- Row 3: Bland-Altman ---------------------------------------------
    stats = _stats(ecg_ibi, ppg_ibi)
    ax = fig.add_subplot(gs[3, 0])
    if stats and stats["n"] >= 3:
        n = stats["n"]
        mean_ibi = (ecg_ibi[:n] + ppg_ibi[:n]) / 2
        diff_ibi = ppg_ibi[:n] - ecg_ibi[:n]
        ax.scatter(mean_ibi, diff_ibi, s=15, alpha=0.6, color=CHANNEL_COLORS[channel])
        ax.axhline(
            stats["bias_ms"], color="red", lw=1.2, label=f"Bias={stats['bias_ms']:.1f}"
        )
        ax.axhline(
            stats["loa_high"],
            color="red",
            lw=0.8,
            ls="--",
            label=f"+1.96SD={stats['loa_high']:.1f}",
        )
        ax.axhline(
            stats["loa_low"],
            color="red",
            lw=0.8,
            ls="--",
            label=f"-1.96SD={stats['loa_low']:.1f}",
        )
        ax.set_title("Bland–Altman  IBI", fontsize=9)
        ax.set_xlabel("Mean IBI (ms)")
        ax.set_ylabel("PPG−ECG (ms)")
        ax.legend(fontsize=6)
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # ---- Scatter ---------------------------------------------------------
    ax = fig.add_subplot(gs[3, 1])
    if stats and stats["n"] >= 3:
        n = stats["n"]
        ax.scatter(
            ecg_ibi[:n], ppg_ibi[:n], s=15, alpha=0.6, color=CHANNEL_COLORS[channel]
        )
        lo = min(ecg_ibi[:n].min(), ppg_ibi[:n].min()) - 20
        hi = max(ecg_ibi[:n].max(), ppg_ibi[:n].max()) + 20
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="Identity")
        ax.set_title(
            f"IBI Scatter  r={stats['pearson_r']:.2f}  MAE={stats['mae_ms']:.1f}ms",
            fontsize=9,
        )
        ax.set_xlabel("ECG IBI (ms)")
        ax.set_ylabel("PPG IBI (ms)")
        ax.legend(fontsize=7)
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fname = OUT_DIR / f"before_after_{pid}_{phase}_{channel}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", fname.name)
    return fname


# ---------------------------------------------------------------------------
# Plot 2: cohort overview
# ---------------------------------------------------------------------------


def plot_cohort_overview(summary: pd.DataFrame):
    if summary.empty:
        log.warning("No summary data for cohort overview.")
        return

    channels = summary["channel"].unique().tolist()
    metrics = ["f1", "pearson_r", "mae_ms"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)))
    fig.suptitle(
        "Cohort-level PPG vs ECG Comparison per Channel", fontsize=13, fontweight="bold"
    )

    for ax, metric in zip(axes, metrics):
        data_by_ch = [
            summary[summary["channel"] == ch][metric].dropna().values for ch in channels
        ]
        bp = ax.boxplot(
            data_by_ch,
            labels=channels,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
        )
        colors = [CHANNEL_COLORS.get(c, "#aaa") for c in channels]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.upper()} distribution per PPG channel vs ECG")

        # annotate medians
        for i, ch in enumerate(channels):
            vals = summary[summary["channel"] == ch][metric].dropna()
            if len(vals):
                ax.text(
                    i + 1,
                    float(vals.median()),
                    f"{vals.median():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="navy",
                )

    plt.tight_layout()
    fname = OUT_DIR / "cohort_overview.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", fname.name)


# ---------------------------------------------------------------------------
# Core per-participant processing
# ---------------------------------------------------------------------------


def process_participant(
    s, anchors: pd.DataFrame, pcode: str, phase_filter: str | None
) -> list[dict]:
    sig_filter = SignalFilter()
    results = []

    if s.ecg is None or len(s.ecg) < 30:
        log.warning("%s: no valid ECG – skipping", pcode)
        return results

    # Prepare ECG
    ecg_fs = get_sampling_rate_hz(s.ecg, default=ECG_FS)
    ecg_raw = s.ecg["ecg_mv"].to_numpy()
    ecg_filt = sig_filter.apply_all(ecg_raw, fs=ecg_fs)
    ecg_re = Resampler(ecg_fs, TARGET_FS).resample_dataframe(
        s.ecg.assign(ecg_filt=ecg_filt), "ecg_filt"
    )
    ecg_re_raw = Resampler(ecg_fs, TARGET_FS).resample_dataframe(s.ecg, "ecg_mv")

    # PPG channels to test
    ppg_channels = {
        "PPG_GREEN": (s.ppg_green, PPG_FS),
        "PPG_IR": (s.ppg_ir, PPG_IR_FS),
        "PPG_RED": (s.ppg_red, PPG_RED_FS),
    }

    panchors = anchors[
        (anchors["participant_code"] == pcode)
        & (anchors["phase"].isin(["Relax", "Task"]))
    ]

    for _, row_a in panchors.iterrows():
        phase = row_a["phase"]
        if phase_filter and phase != phase_filter:
            continue

        buf = pd.Timedelta(seconds=SETTINGS.anchor_buffer_seconds)
        t_start = row_a["start_timestamp"] + buf
        t_end = row_a["end_timestamp"] - buf

        # Slice ECG to phase
        ecg_ts = pd.to_datetime(ecg_re["unix_ms"], unit="ms", utc=True)
        mask_e = (ecg_ts >= t_start) & (ecg_ts <= t_end)
        if mask_e.sum() < 500:
            continue

        # Trim to first 5 min
        e_seg = ecg_re[mask_e].iloc[:FIVE_MIN_SAMPLES]
        e_raw_seg = ecg_re_raw[mask_e].iloc[:FIVE_MIN_SAMPLES]
        unix_e = e_seg["unix_ms"].to_numpy()
        filt_ecg_seg = e_seg["ecg_filt"].to_numpy()
        raw_ecg_seg = e_raw_seg["ecg_mv"].to_numpy()

        ecg_peaks = _detect_ecg_peaks(filt_ecg_seg)
        if len(ecg_peaks) < 3:
            log.warning("%s %s: too few ECG peaks", pcode, phase)
            continue

        ecg_ibi = _ibi_ms(ecg_peaks, unix_e)

        for ch_name, (ch_df, ch_fs_default) in ppg_channels.items():
            if ch_df is None or len(ch_df) < 30:
                continue

            ppg_fs = get_sampling_rate_hz(ch_df, default=ch_fs_default)
            ppg_raw = ch_df["ppg_raw"].to_numpy()
            ppg_filt = sig_filter.apply_all(ppg_raw, fs=ppg_fs)
            ppg_re_filt = Resampler(ppg_fs, TARGET_FS).resample_dataframe(
                ch_df.assign(ppg_filt=ppg_filt), "ppg_filt"
            )
            ppg_re_raw = Resampler(ppg_fs, TARGET_FS).resample_dataframe(
                ch_df, "ppg_raw"
            )

            ppg_ts = pd.to_datetime(ppg_re_filt["unix_ms"], unit="ms", utc=True)
            mask_p = (ppg_ts >= t_start) & (ppg_ts <= t_end)
            if mask_p.sum() < 500:
                continue

            p_seg = ppg_re_filt[mask_p].iloc[:FIVE_MIN_SAMPLES]
            p_raw_seg = ppg_re_raw[mask_p].iloc[:FIVE_MIN_SAMPLES]

            # Align to same unix_ms range as ECG segment
            unix_p = p_seg["unix_ms"].to_numpy()
            filt_ppg = p_seg["ppg_filt"].to_numpy()
            raw_ppg = p_raw_seg["ppg_raw"].to_numpy()

            ppg_peaks = _detect_ppg_peaks(filt_ppg)
            ppg_ibi = _ibi_ms(ppg_peaks, unix_p)

            # Peak matching (use ECG timestamps mapped onto shared unix grid)
            ecg_ts_abs = unix_e[ecg_peaks]
            ppg_ts_abs = unix_p[ppg_peaks] if len(ppg_peaks) else np.array([])
            tp, fp, fn, errors = _match_peaks(ecg_ts_abs, ppg_ts_abs)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

            stats = _stats(ecg_ibi, ppg_ibi)

            row = dict(
                participant_code=pcode,
                channel=ch_name,
                phase=phase,
                duration_sec=int(len(filt_ecg_seg) / TARGET_FS),
                n_ecg_peaks=int(len(ecg_peaks)),
                n_ppg_peaks=int(len(ppg_peaks)),
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4),
                peak_mae_ms=round(float(np.mean(errors)) if errors else np.nan, 2),
                **{k: round(v, 3) for k, v in stats.items() if k != "n"},
                n_ibi_pairs=stats.get("n", 0),
            )
            results.append(row)

            # Generate plots for the best channel per participant (PPG_IR default)
            # or for all channels
            try:
                # Align to same length for plotting
                min_len = min(len(filt_ecg_seg), len(filt_ppg))
                # Use ECG unix_ms for the overlay (common x-axis)
                unix_plot = unix_e[:min_len]
                # Re-detect peaks on trimmed arrays
                ep_plot = _detect_ecg_peaks(filt_ecg_seg[:min_len])
                pp_plot = _detect_ppg_peaks(filt_ppg[:min_len])

                plot_before_after(
                    pid=pcode,
                    phase=phase,
                    raw_ecg=raw_ecg_seg[:min_len],
                    filt_ecg=filt_ecg_seg[:min_len],
                    raw_ppg=raw_ppg[:min_len],
                    filt_ppg=filt_ppg[:min_len],
                    unix_ms=unix_plot,
                    ecg_peaks=ep_plot,
                    ppg_peaks=pp_plot,
                    channel=ch_name,
                )
            except Exception as exc:
                log.warning("Plot failed %s %s %s: %s", pcode, phase, ch_name, exc)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="EDA: PPG vs ECG 5-min segments")
    parser.add_argument(
        "--participant", default=None, help="Limit to one participant, e.g. ID-28"
    )
    parser.add_argument(
        "--phase", default=None, help="Limit to one phase: Relax or Task"
    )
    args = parser.parse_args()

    paths = DEFAULT_PATHS
    loader = DataLoader(paths.raw_data_root)
    anchor_mapper = AnchorMapper(
        paths.anchor_csv_path,
        SETTINGS.timezone_local,
        mapping_csv_path=paths.raw_data_root / "mapping.csv",
    )
    anchors = anchor_mapper.load()

    sessions = (
        loader.load_participant(args.participant)
        if args.participant
        else loader.load_all()
    )

    all_rows: list[dict] = []
    for s in sessions:
        pcode = anchor_mapper.infer_code_from_session_participant(s.participant)
        log.info("Processing %s (session: %s_%s)", pcode, s.participant, s.condition)
        rows = process_participant(s, anchors, pcode, args.phase)
        all_rows.extend(rows)
        log.info("  → %d channel-phase records", len(rows))

    if not all_rows:
        log.error("No results produced. Check ECG availability and anchor timestamps.")
        return

    summary = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "summary_table.csv"
    summary.to_csv(csv_path, index=False)
    log.info("Summary saved → %s  (%d rows)", csv_path, len(summary))

    # Print summary table grouped by channel
    print("\n" + "=" * 80)
    print("SUMMARY: PPG vs ECG  –  5-minute segments per participant")
    print("=" * 80)
    for ch in ["PPG_GREEN", "PPG_IR", "PPG_RED"]:
        sub = summary[summary["channel"] == ch]
        if sub.empty:
            continue
        print(f"\n── {ch} ──")
        cols = [
            "participant_code",
            "phase",
            "f1",
            "pearson_r",
            "mae_ms",
            "bias_ms",
            "loa_low",
            "loa_high",
            "n_ecg_peaks",
            "n_ppg_peaks",
        ]
        cols = [c for c in cols if c in sub.columns]
        print(sub[cols].to_string(index=False))

    # Cohort overview
    plot_cohort_overview(summary)

    # Aggregated stats
    print("\n── Aggregated per channel ──")
    num_cols = ["f1", "pearson_r", "mae_ms", "bias_ms"]
    num_cols = [c for c in num_cols if c in summary.columns]
    print(summary.groupby("channel")[num_cols].agg(["mean", "std", "median"]).round(3))
    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
