"""Read-only evidence report for one isolated biosignal rerun.

The report deliberately consumes canonical artifacts rather than raw files as
processing inputs. Raw source checksums are recomputed only to prove that the
rerun preserved source truth. Signal methods are implemented upstream with the
documented references: NeuroKit2 ECG/PPG, SciPy ``find_peaks``, and YData
Profiling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .settings import PipelineConfig, pipeline_paths_from_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configured_path(
    project_root: Path, config: PipelineConfig, key: str, default: str
) -> Path:
    candidate = Path(config.artifacts.get(key, default))
    return candidate if candidate.is_absolute() else project_root / candidate


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_Tidak ada data._"
    table = frame.loc[
        :, [column for column in columns if column in frame.columns]
    ].copy()
    headers = [str(column) for column in table.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|")

    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join("---" for _ in headers) + " |")
    rows.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def _checksum_evidence(manifest: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    if manifest.empty or not {"source_file", "source_checksum"}.issubset(
        manifest.columns
    ):
        return pd.DataFrame(columns=["source_file", "status"])

    rows: list[dict[str, str]] = []
    for record in (
        manifest[["source_file", "source_checksum"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        source = project_root / str(record.source_file)
        if not source.exists():
            status = "missing"
        elif _sha256(source) == str(record.source_checksum):
            status = "identical"
        else:
            status = "mismatch"
        rows.append({"source_file": str(record.source_file), "status": status})
    return pd.DataFrame(rows)


def _ppg_sync_integrity(bronze_dir: Path, sync_dir: Path) -> tuple[int, int]:
    """Compare PPG timestamp/value pairs before and after ECG-only sync."""

    checked = 0
    mismatches = 0
    for bronze_path in bronze_dir.rglob("*.parquet"):
        if bronze_path.stem not in {"ppg_ir", "ppg_red", "ppg_green"}:
            continue
        sync_path = sync_dir / bronze_path.relative_to(bronze_dir)
        if not sync_path.exists():
            mismatches += 1
            continue
        bronze = pd.read_parquet(
            bronze_path, columns=["timestamp_raw_utc", "raw_value"]
        )
        synced = pd.read_parquet(sync_path, columns=["timestamp_raw_utc", "raw_value"])
        checked += 1
        if not bronze.equals(synced):
            mismatches += 1
    return checked, mismatches


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics.loc[metrics.get("status", pd.Series(dtype=str)).eq("ok")].copy()
    if valid.empty:
        return pd.DataFrame(columns=["sensor_channel", "valid_combinations"])
    numeric = [
        "beat_f1",
        "ibi_mae_ms",
        "ibi_median_absolute_error_ms",
        "ibi_rmse_ms",
        "ibi_pearson_r",
        "ibi_spearman_r",
        "hr_mae_bpm",
        "bland_altman_bias_ms",
        "pat_median_ms",
    ]
    for column in numeric:
        valid[column] = pd.to_numeric(valid.get(column), errors="coerce")
    return (
        valid.groupby("sensor_channel", dropna=False)
        .agg(
            valid_combinations=("status", "size"),
            beat_f1_mean=("beat_f1", "mean"),
            ibi_mae_ms_mean=("ibi_mae_ms", "mean"),
            ibi_median_ae_ms_mean=("ibi_median_absolute_error_ms", "mean"),
            ibi_rmse_ms_mean=("ibi_rmse_ms", "mean"),
            pearson_mean=("ibi_pearson_r", "mean"),
            spearman_mean=("ibi_spearman_r", "mean"),
            hr_mae_bpm_mean=("hr_mae_bpm", "mean"),
            bland_altman_bias_ms_mean=("bland_altman_bias_ms", "mean"),
            pat_median_ms_mean=("pat_median_ms", "mean"),
        )
        .reset_index()
    )


def _participant_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics.loc[metrics.get("status", pd.Series(dtype=str)).eq("ok")].copy()
    if valid.empty:
        return pd.DataFrame(columns=["participant_id", "valid_combinations"])
    for column in (
        "beat_f1",
        "ibi_mae_ms",
        "ibi_rmse_ms",
        "hr_mae_bpm",
        "pat_median_ms",
    ):
        valid[column] = pd.to_numeric(valid.get(column), errors="coerce")
    return (
        valid.groupby("participant_id", dropna=False)
        .agg(
            valid_combinations=("status", "size"),
            beat_f1_mean=("beat_f1", "mean"),
            ibi_mae_ms_mean=("ibi_mae_ms", "mean"),
            ibi_rmse_ms_mean=("ibi_rmse_ms", "mean"),
            hr_mae_bpm_mean=("hr_mae_bpm", "mean"),
            pat_median_ms_mean=("pat_median_ms", "mean"),
        )
        .reset_index()
    )


def generate_preprocessing_peak_report(
    project_root: Path, config: PipelineConfig
) -> Path:
    """Write a non-destructive Markdown handoff for one configured rerun."""

    root = project_root.resolve()
    paths = pipeline_paths_from_config(root, config)
    report_dir = _configured_path(root, config, "reports_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    sync_dir = _configured_path(root, config, "sync_dir", "data/sync")

    raw_index = _read_parquet(
        report_dir / "data_quality" / "raw_input_index_v3.parquet"
    )
    ingestion = _read_parquet(paths.bronze_dir / "ingestion_manifest_v3.parquet")
    silver = _read_parquet(paths.silver_dir / "preprocessing_manifest_v3.parquet")
    gold = _read_parquet(paths.gold_features_path)
    readiness = _read_json(paths.gold_dir / "data_readiness.json")
    sync = _read_parquet(sync_dir / "synchronization_manifest_v3.parquet")
    validation_roots = (
        paths.validation_dir / "ppg_ecg_240s",
        paths.validation_dir / "ppg_ecg",
    )
    metrics_path = next(
        (
            candidate
            for validation_root in validation_roots
            for candidate in (
                validation_root / "ppg_ecg_metrics_240s.csv",
                validation_root / "ppg_ecg_metrics_5min.csv",
                validation_root / "ppg_ecg_metrics.csv",
            )
            if candidate.exists()
        ),
        validation_roots[0] / "ppg_ecg_metrics_240s.csv",
    )
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    validation = {}
    for validation_root in validation_roots:
        for filename in (
            "validation_summary_240s.json",
            "validation_summary_5min.json",
            "validation_summary.json",
        ):
            validation = _read_json(validation_root / filename)
            if validation:
                break
        if validation:
            break
    checksum = _checksum_evidence(ingestion, root)
    ppg_files_checked, ppg_sync_mismatches = _ppg_sync_integrity(
        paths.bronze_dir, sync_dir
    )

    checksum.to_csv(report_dir / "raw_checksum_verification.csv", index=False)
    metric_summary = _metric_summary(metrics)
    participant_summary = _participant_summary(metrics)
    metric_summary.to_csv(report_dir / "peak_metrics_per_channel.csv", index=False)
    participant_summary.to_csv(
        report_dir / "peak_metrics_per_participant.csv", index=False
    )

    eda_status = {
        stage: _read_json(paths.eda_dir / f"ydata_profiling_status_{stage}.json")
        for stage in ("raw", "silver", "gold")
    }
    quality_tiers = (
        silver["quality_tier"].value_counts(dropna=False).rename_axis("quality_tier")
        if "quality_tier" in silver.columns
        else pd.Series(dtype=int)
    )
    strict_tiers = {"excellent", "good"}
    strict_window_count = int(
        readiness.get(
            "strict_training_window_count",
            gold["quality_tier"].isin(strict_tiers).sum()
            if "quality_tier" in gold.columns
            else 0,
        )
    )
    expanded_window_count = int(readiness.get("expanded_training_window_count", 0))
    strict_group_count = int(readiness.get("strict_observed_group_count", 0))
    sync_ok = sync.loc[~sync.get("excluded", pd.Series(dtype=bool)).astype(bool)]
    exclusions = (
        metrics.loc[
            metrics.get("status", pd.Series(dtype=str)).ne("ok"), "exclusion_reason"
        ]
        .fillna("unspecified")
        .value_counts()
        .head(12)
        if "exclusion_reason" in metrics.columns
        else pd.Series(dtype=int)
    )
    interpolation_count = int(
        metrics.get("interpolated_short_gap", pd.Series(dtype=bool)).eq(True).sum()
    )
    long_gap_count = int(
        metrics.get("long_gap_detected", pd.Series(dtype=bool)).eq(True).sum()
    )
    overall_status = "BLOCKED" if readiness.get("status") != "ready" else "DATA READY"

    lines = [
        "# Laporan Rerun Preprocessing dan Peak-to-Peak",
        "",
        f"- Status data: **{overall_status}**.",
        "- Scope: preprocessing core 240 detik, ECG–PPG validation core 240 detik, PPG-only Gold, participant-safe ML lokal, dan ONNX/Android asset verification. DL conditional tidak dijalankan karena kanal Green dan IR memenuhi gate ML.",
        "- Raw tidak ditimpa. Koreksi waktu hanya diterapkan pada `timestamp_corrected_utc` ECG.",
        "",
        "## Sumber dan Integritas Raw",
        "",
        f"- Raw audit: {len(raw_index)} sensor-file rows; ingestion: {len(ingestion)} rows.",
        f"- Peserta: {ingestion['participant_id'].nunique() if 'participant_id' in ingestion else 0}/32; phase: {sorted(ingestion.get('source_phase_hint', pd.Series(dtype=str)).dropna().unique().tolist())}.",
        f"- Checksum pasca-run: {(checksum['status'] == 'identical').sum() if not checksum.empty else 0}/{len(checksum)} identik; mismatch: {(checksum['status'] == 'mismatch').sum() if not checksum.empty else 0}.",
        "",
        "## EDA Otomatis",
        "",
    ]
    for stage, status in eda_status.items():
        lines.append(f"- YData {stage}: `{status.get('status', 'missing')}`.")
    lines.extend(
        [
            "- Raw/Silver/Gold HTML dan plot audit disimpan di artifact `eda/` run ini.",
            "- Silver waveform EDA menggunakan sample deterministik per file untuk mencegah tekanan memori; data asal tidak dimodifikasi.",
            "",
            "## Cleaning, MADC, dan SQA",
            "",
            f"- Silver manifest: {len(silver)} signal artifacts.",
            f"- Tier SQA: {quality_tiers.to_dict()}.",
            f"- Short-gap berflag pada validation: {interpolation_count}; long-gap terdeteksi: {long_gap_count}. Long dropout tidak direkonstruksi.",
            f"- Gold expanded policy (excellent/good/acceptable): {expanded_window_count or readiness.get('ml_ready_window_count', 0)} windows; strict policy (excellent/good): {strict_window_count} windows; excluded lineage: {readiness.get('excluded_window_count', 0)}; coverage grup expanded: {readiness.get('observed_group_count', 0)}/{readiness.get('expected_group_count', 192)}, strict: {strict_group_count}/192.",
            f"- PPG Gold: `{readiness.get('ppg_gold_status', 'missing')}`; ECG validation: `{readiness.get('ecg_validation_status', 'missing')}`; training: `{readiness.get('training_status', 'missing')}`.",
            f"- Model release: `{readiness.get('model_release_status', 'missing')}`; channels: {readiness.get('model_release_channels', [])}; max ONNX parity diff: {readiness.get('model_release_max_abs_diff', 'n/a')}; reason: `{readiness.get('model_release_reason', 'n/a')}`.",
            f"- Training channels: {readiness.get('training_channels', [])}; observed training groups: {readiness.get('observed_training_groups', 0)}/{readiness.get('expected_training_groups', 128)}; DL handoff: {readiness.get('baseline_ml_dl_handoff_channels', [])}.",
            f"- Quality policy: `{readiness.get('quality_policy_version', 'unknown')}`; MADC annotation-only; acceptable tetap berflag dan dipisahkan untuk sensitivity analysis.",
            f"- Lokasi Gold features: `{paths.gold_features_path}`; IBI: `{paths.gold_dir / 'gold_ibi_sequences_v3.parquet'}`; waveform/mask: `{paths.gold_dir / 'waveforms'}`.",
            f"- Lokasi Gold lineage/readiness: `{paths.gold_dir / 'manifest_v3.json'}` dan `{paths.gold_dir / 'data_readiness.json'}`.",
            "",
            "## Koreksi Clock ECG",
            "",
            f"- Sync session: {len(sync)}; ECG valid: {len(sync_ok)}; ECG excluded: {int(sync.get('excluded', pd.Series(dtype=bool)).sum()) if not sync.empty else 0}.",
            f"- Alpha valid: {sync_ok['alpha'].min() if not sync_ok.empty else float('nan'):.6f}–{sync_ok['alpha'].max() if not sync_ok.empty else float('nan'):.6f}; batas tetap [0.98, 1.02].",
            "- PPG tidak di-shift; PAT fisiologis dipertahankan 100–400 ms.",
            f"- Integritas PPG Bronze→sync: {ppg_files_checked} file diperiksa; mismatch timestamp/value: {ppg_sync_mismatches}.",
            "",
            "## Peak-to-Peak ECG–PPG",
            "",
            f"- Window validasi: {validation.get('validation_window_seconds', 240)} detik; kombinasi: {validation.get('total_combinations', 0)}/{validation.get('expected_combinations', 192)}; valid: {validation.get('ok_combinations', 0)}; excluded: {validation.get('excluded_combinations', 0)}.",
            "",
            "### Ringkasan per Kanal",
            "",
            _markdown_table(
                metric_summary,
                [
                    "sensor_channel",
                    "valid_combinations",
                    "beat_f1_mean",
                    "ibi_mae_ms_mean",
                    "ibi_median_ae_ms_mean",
                    "ibi_rmse_ms_mean",
                    "pearson_mean",
                    "spearman_mean",
                    "hr_mae_bpm_mean",
                    "bland_altman_bias_ms_mean",
                    "pat_median_ms_mean",
                ],
            ),
            "",
            "### Ringkasan per Peserta",
            "",
            _markdown_table(
                participant_summary,
                [
                    "participant_id",
                    "valid_combinations",
                    "beat_f1_mean",
                    "ibi_mae_ms_mean",
                    "ibi_rmse_ms_mean",
                    "hr_mae_bpm_mean",
                    "pat_median_ms_mean",
                ],
            ),
            "",
            "### Eksklusi Utama",
            "",
            *[f"- `{reason}`: {count}" for reason, count in exclusions.items()],
            "",
            "## Evidence dan Risiko",
            "",
            "- Detail 192 kombinasi core 240 detik: `validation/ppg_ecg_240s/ppg_ecg_metrics_240s.csv`; CSV per sesi berada di `validation/ppg_ecg_csv/csv/` dan plot di `validation/ppg_ecg_csv/plots/`.",
            "- Model release canonical: `data/artifacts/export/model_index.json`; generated Android assets: `android/inference/build/generated/inferenceAssets/`.",
            "- ECG failure memblokir `ecg_validation_status`, tetapi tidak otomatis menghapus PPG dari Gold. PPG Gold tetap dibatasi oleh coverage dan quality tier PPG.",
            "- NeuroKit2 PPG quality bersifat diagnostik opsional pada clean signal; rerun lightweight ini menonaktifkan template matching berat (`neurokit_quality_diagnostic=false`). Jika diaktifkan, error/low quality menjadi flag (`ppg_quality_error:<ExceptionType>`/`nk_low_quality`), bukan hard rejection; hard rejection hanya untuk nonfinite/dropout/flatline/clipping/coverage/peak fisiologis yang gagal.",
            "",
            "## Referensi",
            "",
            "- [NeuroKit2 ECG](https://neuropsychology.github.io/NeuroKit/functions/ecg.html)",
            "- [NeuroKit2 PPG](https://neuropsychology.github.io/NeuroKit/functions/ppg.html)",
            "- [SciPy find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)",
            "- [YData Profiling time-series](https://docs.profiling.ydata.ai/latest/features/time_series_datasets/)",
        ]
    )
    output = report_dir / "preprocessing_peak_report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
