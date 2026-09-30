"""Export human-readable CSV views of a completed local pipeline run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _mirror_parquet_tree(
    source_root: Path,
    target_root: Path,
    columns: tuple[str, ...] | None = None,
) -> int:
    """Mirror parquet tables below one run layer as CSV without touching raw."""

    count = 0
    for source in sorted(source_root.rglob("*.parquet")):
        relative = source.relative_to(source_root).with_suffix(".csv")
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(source)
        if columns is not None:
            frame = frame.loc[:, [column for column in columns if column in frame]]
        frame.to_csv(target, index=False)
        count += 1
    return count


def export_run_csv(run_root: Path) -> dict[str, int | str]:
    """Export Silver, Gold, and peak-table CSV views for one isolated run.

    The raw source remains in ``data/raw`` and is never copied or rewritten.
    Parquet remains the typed canonical artifact; these files are explicit
    human-readable/data-science views requested for inspection and wrangling.
    """

    root = run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")
    output_root = root / "csv_export"
    silver_columns = (
        "participant_id",
        "session_id",
        "recording_id",
        "phase",
        "sensor_channel",
        "timestamp_raw_utc",
        "timestamp_corrected_utc",
        "timestamp_local",
        "elapsed_seconds",
        "sampling_rate_hz",
        "raw_value",
        "clean_value",
        "interpolated_mask",
        "interpolated_sample_count",
        "interpolation_provenance",
        "motion_score",
        "motion_annotation",
        "accelerometer_scale_status",
        "motion_method",
        "quality_score",
        "quality_tier",
        "quality_flags",
        "source_file",
        "source_checksum",
        "exclusion_reason",
        "preprocessing_version",
    )
    layer_counts = {
        "silver": _mirror_parquet_tree(
            root / "silver", output_root / "silver", silver_columns
        ),
        "gold": _mirror_parquet_tree(root / "gold", output_root / "gold"),
    }
    validation_root = root / "validation" / "ppg_ecg_240s"
    if not validation_root.exists():
        validation_root = root / "validation" / "ppg_ecg"
    layer_counts["validation"] = _mirror_parquet_tree(
        validation_root / "peak_tables",
        output_root / "validation" / validation_root.name / "peak_tables",
    )
    manifest = pd.DataFrame(
        [
            {
                "layer": layer,
                "csv_file_count": count,
                "source": f"{layer}_parquet",
                "raw_mutated": False,
            }
            for layer, count in layer_counts.items()
        ]
    )
    manifest_path = output_root / "csv_export_manifest.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return {
        "run_root": str(root),
        **layer_counts,
        "manifest": str(manifest_path),
    }
