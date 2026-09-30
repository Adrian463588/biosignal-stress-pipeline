from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import WINDOW_SECONDS
from .settings import PipelinePaths

logger = logging.getLogger(__name__)

PPG_SENSORS = ("ppg_green", "ppg_ir", "ppg_red")
PHASES = ("Relax", "Task")
SIGNAL_LABELS = {
    "ppg_green": "PPG_GREEN",
    "ppg_ir": "PPG_IR",
    "ppg_red": "PPG_RED",
}


@dataclass(frozen=True)
class ThesisReportOutputs:
    report_path: Path
    eda_summary_path: Path
    classification_metrics_path: Path
    failure_analysis_path: Path
    figure_count: int


class ThesisReportGenerator:
    """Build thesis-grade EDA, cleaning, and failure-analysis artifacts."""

    def __init__(self, paths: PipelinePaths) -> None:
        self.paths = paths
        self.out_dir = paths.project_root / "reports" / "thesis"
        self.fig_dir = self.out_dir / "figures"

    def run(self, make_plots: bool = True) -> ThesisReportOutputs:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        gold = self._read_gold()
        eda = self._read_eda_tables()
        metrics = self._read_model_metrics()

        participant_eda = self._build_participant_eda_summary(gold, eda)
        classification = self._build_classification_metrics(gold, metrics)
        failure = self._build_failure_analysis(participant_eda, classification)

        eda_path = self.out_dir / "participant_eda_summary.csv"
        cls_path = self.out_dir / "participant_classification_metrics.csv"
        fail_path = self.out_dir / "participant_failure_analysis.csv"
        report_path = self.out_dir / "thesis_results_discussion.md"

        participant_eda.to_csv(eda_path, index=False)
        classification.to_csv(cls_path, index=False)
        failure.to_csv(fail_path, index=False)

        figure_count = 0
        if make_plots:
            figure_count += self._plot_cleaning_pipeline(gold)
            figure_count += self._plot_model_metrics(classification)

        report_path.write_text(
            self._build_markdown_report(
                participant_eda, classification, failure, figure_count
            ),
            encoding="utf-8",
        )

        logger.info("Thesis report saved in %s", self.out_dir)
        return ThesisReportOutputs(
            report_path, eda_path, cls_path, fail_path, figure_count
        )

    def _read_gold(self) -> pd.DataFrame:
        if not self.paths.gold_features_path.exists():
            raise FileNotFoundError(
                f"Gold features not found: {self.paths.gold_features_path}"
            )
        return pd.read_csv(self.paths.gold_features_path)

    def _read_eda_tables(self) -> dict[str, pd.DataFrame]:
        names = [
            "sample_counts",
            "duration_consistency",
            "sampling_interval_summary",
            "missingness_discontinuity",
            "phase_coverage",
            "cross_channel_correlation",
        ]
        out: dict[str, pd.DataFrame] = {}
        for name in names:
            path = self.paths.eda_dir / f"{name}.csv"
            out[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
        return out

    def _read_model_metrics(self) -> dict[str, Any]:
        path = (
            self.paths.project_root
            / "reports"
            / "model_eval"
            / "v1.0.0"
            / "stress_metrics.json"
        )
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        metrics: dict[str, Any] = {}
        for sensor in PPG_SENSORS:
            split_path = (
                self.paths.models_dir / "stress" / sensor / "split_and_metrics.json"
            )
            if split_path.exists():
                metrics[sensor] = json.loads(split_path.read_text(encoding="utf-8"))
        return metrics

    def _build_participant_eda_summary(
        self, gold: pd.DataFrame, eda: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        base = (
            gold[gold["phase"].isin(PHASES)]
            .groupby(["participant_code", "phase", "sensor_channel"], dropna=False)
            .agg(
                gold_windows=("window_idx", "count"),
                quality_score_mean=("quality_score", "mean"),
                quality_score_min=("quality_score", "min"),
                excellent_windows=(
                    "quality_tier",
                    lambda s: int((s == "excellent").sum()),
                ),
                good_windows=("quality_tier", lambda s: int((s == "good").sum())),
                mean_hr_bpm=("mean_hr_bpm", "mean"),
                mean_rr_ms=("mean_rr_ms", "mean"),
                rmssd_ms=("rmssd_ms", "mean"),
                std_rr_ms=("std_rr_ms", "mean"),
                num_peaks_mean=("num_peaks", "mean"),
            )
            .reset_index()
        )

        sample = self._normalize_signal_type(eda.get("sample_counts", pd.DataFrame()))
        if not sample.empty:
            sample = sample[sample["phase"].isin(PHASES)]
            base = base.merge(
                sample.rename(columns={"signal_type": "sensor_channel"}),
                on=["participant_code", "phase", "sensor_channel"],
                how="left",
            )

        duration = self._normalize_signal_type(
            eda.get("duration_consistency", pd.DataFrame())
        )
        keep_duration = [
            "participant_code",
            "phase",
            "sensor_channel",
            "actual_duration_s",
            "duration_error_s",
        ]
        if not duration.empty:
            base = base.merge(
                duration[keep_duration],
                on=["participant_code", "phase", "sensor_channel"],
                how="left",
            )

        missing = self._normalize_signal_type(
            eda.get("missingness_discontinuity", pd.DataFrame())
        )
        keep_missing = [
            "participant_code",
            "phase",
            "sensor_channel",
            "missing_ratio",
            "gap_count",
        ]
        if not missing.empty:
            base = base.merge(
                missing[keep_missing],
                on=["participant_code", "phase", "sensor_channel"],
                how="left",
            )

        interval = self._normalize_signal_type(
            eda.get("sampling_interval_summary", pd.DataFrame())
        )
        keep_interval = [
            "participant_code",
            "phase",
            "sensor_channel",
            "interval_ms_mean",
            "interval_ms_std",
            "interval_ms_p95",
        ]
        if not interval.empty:
            base = base.merge(
                interval[keep_interval],
                on=["participant_code", "phase", "sensor_channel"],
                how="left",
            )

        base["quality_summary"] = np.where(
            base["quality_score_min"].fillna(0) >= 0.95,
            "excellent_consistent",
            np.where(
                base["quality_score_min"].fillna(0) >= 0.80,
                "good_with_minor_artifacts",
                "needs_review",
            ),
        )
        return base.sort_values(
            ["participant_code", "phase", "sensor_channel"]
        ).reset_index(drop=True)

    def _build_classification_metrics(
        self, gold: pd.DataFrame, metrics: dict[str, Any]
    ) -> pd.DataFrame:
        participants = sorted(
            gold["participant_code"].dropna().astype(str).unique().tolist()
        )
        rows: list[dict[str, Any]] = []
        for sensor in PPG_SENSORS:
            sensor_metrics = metrics.get(sensor, {})
            test_participants = set(
                map(str, sensor_metrics.get("test_participants", []))
            )
            per_pid = sensor_metrics.get("per_participant_metrics", {})
            model_name = sensor_metrics.get("model_name", "unknown")
            below_threshold = bool(
                sensor_metrics.get("test_metrics", {}).get("below_threshold", True)
            )

            for pid in participants:
                if pid not in test_participants:
                    rows.append(
                        {
                            "participant_code": pid,
                            "sensor_channel": sensor,
                            "model_name": model_name,
                            "status": "not_in_test_holdout",
                            "below_export_threshold": below_threshold,
                        }
                    )
                    continue

                m = per_pid.get(pid, {})
                cm = m.get("confusion_matrix", {}).get("matrix", [[0, 0], [0, 0]])
                tn, fp = cm[0]
                fn, tp = cm[1]
                precision = self._safe_div(tp, tp + fp)
                recall = self._safe_div(tp, tp + fn)
                error_rate = self._safe_div(fp + fn, tn + fp + fn + tp)
                rows.append(
                    {
                        "participant_code": pid,
                        "sensor_channel": sensor,
                        "model_name": model_name,
                        "status": "evaluated",
                        "below_export_threshold": below_threshold,
                        "accuracy": m.get("accuracy"),
                        "precision": precision,
                        "recall_sensitivity_task": recall,
                        "specificity_relax": m.get("specificity_relax"),
                        "f1_score": m.get("f1"),
                        "balanced_accuracy": m.get("balanced_accuracy"),
                        "roc_auc": m.get("roc_auc"),
                        "pr_auc": m.get("pr_auc"),
                        "true_relax": tn,
                        "false_positive_relax_to_task": fp,
                        "false_negative_task_to_relax": fn,
                        "true_task": tp,
                        "error_rate": error_rate,
                    }
                )

        out = pd.DataFrame(rows)
        evaluated = out["status"] == "evaluated"
        out["difficulty_rank"] = np.nan
        if evaluated.any():
            rank_source = out.loc[evaluated].sort_values(
                ["error_rate", "balanced_accuracy", "specificity_relax", "f1_score"],
                ascending=[False, True, True, True],
            )
            out.loc[rank_source.index, "difficulty_rank"] = range(
                1, len(rank_source) + 1
            )
        return out.sort_values(
            ["sensor_channel", "status", "difficulty_rank", "participant_code"]
        ).reset_index(drop=True)

    def _build_failure_analysis(
        self, eda: pd.DataFrame, classification: pd.DataFrame
    ) -> pd.DataFrame:
        evaluated = classification[classification["status"] == "evaluated"].copy()
        if evaluated.empty:
            return pd.DataFrame()

        evidence = (
            eda.groupby(["participant_code", "sensor_channel"], dropna=False)
            .agg(
                mean_quality_score=("quality_score_mean", "mean"),
                min_quality_score=("quality_score_min", "min"),
                total_gap_count=("gap_count", "sum"),
                max_missing_ratio=("missing_ratio", "max"),
                mean_interval_std_ms=("interval_ms_std", "mean"),
                mean_num_peaks=("num_peaks_mean", "mean"),
                mean_std_rr_ms=("std_rr_ms", "mean"),
            )
            .reset_index()
        )
        out = evaluated.merge(
            evidence, on=["participant_code", "sensor_channel"], how="left"
        )
        out["failure_hypothesis"] = out.apply(self._failure_hypothesis, axis=1)
        return out.sort_values(["difficulty_rank", "sensor_channel"]).reset_index(
            drop=True
        )

    def _plot_cleaning_pipeline(self, gold: pd.DataFrame) -> int:
        count = 0
        combos = (
            gold[gold["phase"].isin(PHASES)][
                ["participant_code", "phase", "sensor_channel"]
            ]
            .drop_duplicates()
            .sort_values(["participant_code", "phase", "sensor_channel"])
        )
        for row in combos.itertuples(index=False):
            try:
                if self._plot_one_cleaning_pipeline(
                    str(row.participant_code), str(row.phase), str(row.sensor_channel)
                ):
                    count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping thesis plot for %s/%s/%s: %s",
                    row.participant_code,
                    row.phase,
                    row.sensor_channel,
                    exc,
                )
        return count

    def _plot_one_cleaning_pipeline(
        self, participant: str, phase: str, sensor: str
    ) -> bool:
        silver_path = self.paths.silver_dir / participant / phase / f"{sensor}.parquet"
        gold_path = (
            self.paths.gold_dir
            / "features"
            / "stress"
            / participant
            / phase
            / f"{sensor}.parquet"
        )
        bronze_path = self.paths.bronze_dir / participant / sensor / "raw.csv"
        if not silver_path.exists() or not gold_path.exists():
            return False

        silver = pd.read_parquet(silver_path)
        gold = pd.read_parquet(gold_path)
        if silver.empty or gold.empty:
            return False
        silver["timestamp_utc"] = pd.to_datetime(
            silver["timestamp_utc"], utc=True, errors="coerce"
        )

        raw = pd.DataFrame()
        if bronze_path.exists():
            raw = pd.read_csv(bronze_path, usecols=["timestamp_utc", "raw_value"])
            raw["timestamp_utc"] = pd.to_datetime(
                raw["timestamp_utc"], utc=True, errors="coerce"
            )
            t0, t1 = silver["timestamp_utc"].min(), silver["timestamp_utc"].max()
            raw = raw[(raw["timestamp_utc"] >= t0) & (raw["timestamp_utc"] <= t1)]

        fig, axes = plt.subplots(5, 1, figsize=(12, 12), constrained_layout=True)
        title = f"{participant} | {phase} | {sensor} cleaning pipeline"
        fig.suptitle(title, fontsize=13)

        self._plot_series(axes[0], raw, "Raw trimmed/canonical signal", "raw_value")
        self._plot_series(
            axes[1], silver, "Aligned silver signal after SQA gating", "raw_value"
        )

        dt_ms = silver["timestamp_utc"].diff().dt.total_seconds().mul(1000)
        axes[2].plot(np.arange(len(dt_ms)), dt_ms, linewidth=0.8)
        axes[2].set_title("Sampling interval / timestamp gap diagnostic")
        axes[2].set_ylabel("ms")

        vals = pd.to_numeric(silver["raw_value"], errors="coerce").dropna()
        axes[3].hist(vals, bins=50, color="#4c78a8", alpha=0.85)
        axes[3].set_title("Amplitude distribution after alignment")
        axes[3].set_xlabel("raw amplitude")

        feature_cols = [
            c
            for c in [
                "mean_hr_bpm",
                "mean_rr_ms",
                "rmssd_ms",
                "std_rr_ms",
                f"{sensor}_peak_amp_mean",
            ]
            if c in gold.columns
        ]
        for col in feature_cols:
            axes[4].plot(
                gold["window_idx"],
                pd.to_numeric(gold[col], errors="coerce"),
                marker="o",
                linewidth=1,
                label=col,
            )
        axes[4].set_title(f"Gold feature trend per {WINDOW_SECONDS}s window")
        axes[4].set_xlabel("window_idx")
        axes[4].legend(loc="best", fontsize=8)

        q_tier = (
            silver["quality_tier"].mode().iloc[0]
            if "quality_tier" in silver and not silver["quality_tier"].dropna().empty
            else "unknown"
        )
        q_flag = (
            silver["quality_flag"].mode().iloc[0]
            if "quality_flag" in silver and not silver["quality_flag"].dropna().empty
            else "unknown"
        )
        axes[1].text(
            0.01,
            0.90,
            f"quality={q_tier}; flags={q_flag}",
            transform=axes[1].transAxes,
            fontsize=9,
        )

        out = self.fig_dir / participant / phase
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{sensor}_cleaning_pipeline.png", dpi=140)
        plt.close(fig)
        return True

    def _plot_model_metrics(self, classification: pd.DataFrame) -> int:
        model_dir = self.fig_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        evaluated = classification[classification["status"] == "evaluated"].copy()
        for sensor, grp in evaluated.groupby("sensor_channel"):
            grp = grp.sort_values("participant_code")
            if grp.empty:
                continue
            fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
            x = np.arange(len(grp))
            ax.bar(
                x - 0.2, grp["balanced_accuracy"], width=0.4, label="Balanced accuracy"
            )
            ax.bar(x + 0.2, 1 - grp["error_rate"], width=0.4, label="1 - error rate")
            ax.set_xticks(x)
            ax.set_xticklabels(grp["participant_code"], rotation=45, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Per-participant classification performance: {sensor}")
            ax.legend()
            fig.savefig(model_dir / f"{sensor}_per_participant_metrics.png", dpi=150)
            plt.close(fig)
            count += 1
        return count

    def _build_markdown_report(
        self,
        eda: pd.DataFrame,
        classification: pd.DataFrame,
        failure: pd.DataFrame,
        figure_count: int,
    ) -> str:
        participants = eda["participant_code"].nunique()
        sensors = ", ".join(sorted(eda["sensor_channel"].dropna().unique()))
        evaluated = classification[classification["status"] == "evaluated"]
        worst = failure.head(8)

        lines = [
            "# Laporan Hasil dan Pembahasan: EDA, SQA, dan Klasifikasi Stres PPG",
            "",
            "## Ringkasan Eksekutif",
            (
                f"Analisis ini mencakup {participants} partisipan dengan sensor {sensors}. "
                "Data diproses secara bertingkat dari raw trimmed data, alignment dan SQA pada silver layer, "
                "ekstraksi fitur pada gold layer, hingga evaluasi model klasifikasi stres berbasis PPG."
            ),
            "",
            "## EDA dan Visualisasi Per Tahapan",
            (
                "Setiap partisipan divisualisasikan secara kronologis dalam panel raw signal, sinyal silver setelah "
                "alignment dan SQA, diagnostik interval sampling, distribusi amplitudo, serta tren fitur HRV dan amplitudo "
                f"pada window {WINDOW_SECONDS} detik. Plot disimpan pada `reports/thesis/figures/{{participant}}/{{phase}}/`."
            ),
            f"Jumlah plot yang dibuat: {figure_count}.",
            "",
            "## Kualitas Data dan SQA",
            (
                "Ringkasan EDA per partisipan disimpan dalam `participant_eda_summary.csv`. Kolom quality score, gap count, "
                "missing ratio, interval jitter, jumlah peak, RMSSD, SDNN, dan heart rate digunakan untuk menilai apakah "
                "window layak dipakai sebagai dasar ekstraksi fitur fisiologis."
            ),
            "",
            "## Performa Klasifikasi Per Partisipan",
            (
                "`participant_classification_metrics.csv` memuat akurasi, presisi, recall, F1-score, specificity Relax, "
                "balanced accuracy, false positive Relax→Task, false negative Task→Relax, dan error rate. Partisipan di luar "
                "held-out test set diberi status `not_in_test_holdout` agar tidak terjadi pelaporan metrik palsu."
            ),
            "",
        ]
        if not evaluated.empty:
            summary = (
                evaluated.groupby("sensor_channel")
                .agg(
                    n_participants=("participant_code", "nunique"),
                    accuracy=("accuracy", "mean"),
                    f1_score=("f1_score", "mean"),
                    balanced_accuracy=("balanced_accuracy", "mean"),
                    error_rate=("error_rate", "mean"),
                )
                .reset_index()
            )
            lines.extend(
                ["### Ringkasan Sensor", "", summary.to_markdown(index=False), ""]
            )

        lines.extend(
            [
                "## Failure Analysis",
                (
                    "Partisipan tersulit diidentifikasi dengan prioritas error rate tertinggi, kemudian balanced accuracy, "
                    "specificity Relax, dan F1-score. Interpretasi kegagalan dikaitkan dengan bukti kualitas sinyal seperti "
                    "gap timestamp, missingness, penurunan quality score, instabilitas RR/peak count, serta pola false positive "
                    "Relax→Task yang umum pada data wearable dengan motion artifact atau kontak sensor kurang stabil."
                ),
                "",
            ]
        )
        if not worst.empty:
            cols = [
                "difficulty_rank",
                "participant_code",
                "sensor_channel",
                "error_rate",
                "balanced_accuracy",
                "specificity_relax",
                "f1_score",
                "failure_hypothesis",
            ]
            lines.extend(
                [
                    "### Partisipan Tersulit",
                    "",
                    worst[cols].to_markdown(index=False),
                    "",
                ]
            )

        lines.extend(
            [
                "## Catatan Interpretatif",
                (
                    "Hasil ini harus dibaca sebagai evaluasi window-level dengan split held-out berbasis partisipan. "
                    "Sensor `ppg_green` tetap menjadi kandidat deployment karena sudah lolos gate ONNX/parity, sedangkan "
                    "sensor lain digunakan untuk pembanding analitis dan failure analysis lintas kanal."
                ),
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _normalize_signal_type(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "signal_type" not in df.columns:
            return df
        out = df.copy()
        out["signal_type"] = (
            out["signal_type"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "ppg_green": "ppg_green",
                    "ppg_ir": "ppg_ir",
                    "ppg_red": "ppg_red",
                    "ecg": "ecg",
                }
            )
            .fillna(out["signal_type"].astype(str).str.lower())
        )
        return out.rename(columns={"signal_type": "sensor_channel"})

    @staticmethod
    def _plot_series(ax: plt.Axes, df: pd.DataFrame, title: str, y_col: str) -> None:
        ax.set_title(title)
        if df.empty or y_col not in df.columns:
            ax.text(0.5, 0.5, "not available", ha="center", va="center")
            return
        sampled = (
            df.iloc[np.linspace(0, len(df) - 1, min(len(df), 2500)).astype(int)]
            if len(df) > 2500
            else df
        )
        x = (
            sampled["timestamp_utc"]
            if "timestamp_utc" in sampled
            else np.arange(len(sampled))
        )
        ax.plot(x, pd.to_numeric(sampled[y_col], errors="coerce"), linewidth=0.75)
        ax.set_ylabel("amplitude")

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")

    @staticmethod
    def _failure_hypothesis(row: pd.Series) -> str:
        reasons: list[str] = []
        if row.get("false_positive_relax_to_task", 0) > row.get(
            "false_negative_task_to_relax", 0
        ):
            reasons.append(
                "dominant false positive Relax-to-Task suggests stress-like morphology during rest"
            )
        if row.get("total_gap_count", 0) and row.get("total_gap_count", 0) > 20:
            reasons.append(
                "high timestamp discontinuity may indicate wearable sampling instability"
            )
        if row.get("max_missing_ratio", 0) and row.get("max_missing_ratio", 0) > 0:
            reasons.append("missing samples are present")
        if row.get("min_quality_score", 1) < 0.95:
            reasons.append("quality score drops below excellent")
        if row.get("mean_std_rr_ms", 0) and row.get("mean_std_rr_ms", 0) > 250:
            reasons.append(
                "RR variability is high and may reflect noisy peak detection or physiological overlap"
            )
        if not reasons:
            reasons.append(
                "errors likely reflect participant-specific physiological overlap between Relax and Task"
            )
        return "; ".join(reasons)
