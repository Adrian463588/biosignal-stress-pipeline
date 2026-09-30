from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from PipelinePython.common.peak_csv_visualization import CsvPeakToPeakVisualizer
from PipelinePython.common.contracts import VALID_PARTICIPANTS


def _write_signal(path: Path, source: str, start: pd.Timestamp) -> None:
    offset = timedelta(milliseconds=200) if "Trimmed" in source else timedelta(0)
    timestamps = [start + offset + timedelta(seconds=index) for index in range(301)]
    frame = pd.DataFrame(
        {
            "timestamp_raw_utc": [value.isoformat() for value in timestamps],
            "timestamp_corrected_utc": [value.isoformat() for value in timestamps],
            "raw_value": range(301),
            "clean_value": range(301),
            "interpolated_mask": False,
            "source_file": f"data/raw/{source}.csv",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _metric_rows() -> list[dict[str, object]]:
    rows = []
    for participant_id in VALID_PARTICIPANTS:
        for phase in ("relax", "task"):
            for channel in ("ppg_ir", "ppg_red", "ppg_green"):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "phase": phase,
                        "sensor_channel": channel,
                        "status": "excluded",
                        "exclusion_reason": "fixture_excluded",
                        "validation_window_seconds": 240,
                    }
                )
    return rows


def test_visualizer_emits_all_192_status_rows_without_signal_fabrication(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "data" / "artifacts" / "runs" / "test-run"
    validation = run_root / "validation" / "ppg_ecg_240s"
    validation.mkdir(parents=True)
    pd.DataFrame(_metric_rows()).to_csv(
        validation / "ppg_ecg_metrics_240s.csv", index=False
    )

    outputs = CsvPeakToPeakVisualizer(run_root, tmp_path).run()

    assert outputs.row_count == 192
    assert outputs.valid_count == 0
    manifest = pd.read_csv(outputs.manifest_path)
    assert len(manifest) == 192
    assert set(manifest["status"]) == {"excluded"}
    assert not any("parquet" in column.lower() for column in manifest.columns)
    assert (
        run_root
        / "validation"
        / "ppg_ecg_csv"
        / "plots"
        / "ID-01"
        / "relax"
        / "ppg_ir"
        / "peak_to_peak_240s.png"
    ).is_file()
    assert (
        validation / "plots" / "ID-01" / "relax" / "ppg_ir" / "peak_to_peak_240s.png"
    ).is_file()
    assert (validation / "csv_visual_manifest.csv").is_file()


def test_visualizer_writes_clean_normalized_samples_and_matches(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    validation = run_root / "validation" / "ppg_ecg_240s"
    silver = run_root / "silver" / "ID-01" / "relax"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    _write_signal(silver / "ecg.csv", "DataRenamedPartisipan/ID-01/ecg", start)
    _write_signal(silver / "ppg_ir.csv", "DataTrimmedPartisipan/ID-01/ppg_ir", start)
    validation.mkdir(parents=True)

    rows = _metric_rows()
    for row in rows:
        if (
            row["participant_id"],
            row["phase"],
            row["sensor_channel"],
        ) == ("ID-01", "relax", "ppg_ir"):
            row.update(
                {
                    "status": "ok",
                    "exclusion_reason": "",
                    "validation_window_seconds": 240,
                    "window_start_utc": start.isoformat(),
                    "window_end_utc": (start + timedelta(seconds=240)).isoformat(),
                    "ecg_peak_count": 3,
                    "ppg_peak_count": 3,
                    "matched_beat_count": 3,
                    "beat_f1": 1.0,
                    "pat_median_ms": 200.0,
                }
            )
            break
    pd.DataFrame(rows).to_csv(validation / "ppg_ecg_metrics_240s.csv", index=False)
    peak_dir = validation / "peak_tables" / "ID-01" / "relax"
    peak_dir.mkdir(parents=True)
    peak_times = [start + timedelta(seconds=value) for value in (10, 11, 12)]
    peak_table = pd.DataFrame(
        {
            "ecg_peak_timestamp_ms": [value.value / 1_000_000 for value in peak_times],
            "ppg_peak_timestamp_ms": [
                (value + timedelta(milliseconds=200)).value / 1_000_000
                for value in peak_times
            ],
            "pat_ms": [200.0, 200.0, 200.0],
            "matched": [True, True, True],
        }
    )
    peak_table.to_csv(peak_dir / "ppg_ir.csv", index=False)

    outputs = CsvPeakToPeakVisualizer(run_root, tmp_path).run()
    session = (
        run_root / "validation" / "ppg_ecg_csv" / "csv" / "ID-01" / "relax" / "ppg_ir"
    )
    samples = pd.read_csv(session / "signal_samples.csv")
    matches = pd.read_csv(session / "peak_matches.csv")
    assert outputs.valid_count == 1
    assert {
        "raw_value",
        "clean_value",
        "normalized_clean_value",
        "peak_flag",
        "matched_peak_id",
    }.issubset(samples.columns)
    assert set(samples["signal_source"]) == {"ecg", "ppg_ir"}
    assert int(samples["peak_flag"].sum()) == 6
    assert int(matches["matched"].sum()) == 3
    assert matches["pat_ms"].dropna().between(100, 400).all()
    assert (session / "intervals.csv").is_file()
    assert (
        validation / "csv" / "ID-01" / "relax" / "ppg_ir" / "signal_samples.csv"
    ).is_file()
    assert (
        run_root
        / "validation"
        / "ppg_ecg_csv"
        / "plots"
        / "ID-01"
        / "relax"
        / "ppg_ir"
        / "peak_to_peak_240s.png"
    ).is_file()
