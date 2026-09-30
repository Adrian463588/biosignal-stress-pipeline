from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from PipelinePython.common.restore_comparison import build_restore_comparison


def _metrics(run_id: str, *, with_quality: bool) -> pd.DataFrame:
    rows = []
    for participant in range(1, 33):
        for phase in ("relax", "task"):
            for channel in ("ppg_green", "ppg_ir", "ppg_red"):
                row = {
                    "participant_id": f"ID-{participant:02d}",
                    "phase": phase,
                    "sensor_channel": channel,
                    "status": "ok",
                    "validation_status": "valid",
                    "exclusion_reason": "",
                    "beat_f1": 0.4,
                    "ibi_mae_ms": 30.0,
                    "hr_mae_bpm": 3.0,
                    "ecg_interpolated_sample_count": 0,
                    "ppg_interpolated_sample_count": 0,
                    "run_id": run_id,
                }
                if with_quality:
                    row.update(
                        {
                            "metric_status": "reported",
                            "quality_status": "pass",
                            "strict_quality_eligible": True,
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _write_run(root: Path, run_id: str, *, with_quality: bool) -> Path:
    run_root = root / "data" / "artifacts" / "runs" / run_id
    metrics_path = run_root / "validation" / "ppg_ecg_240s" / "ppg_ecg_metrics_240s.csv"
    metrics_path.parent.mkdir(parents=True)
    _metrics(run_id, with_quality=with_quality).to_csv(metrics_path, index=False)
    (run_root / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "raw_corpus_sha256": "8d156cc4932826c3e022a6867552b5b720685ef336ce22c049a8036bc50d3561",
                "config_sha256": f"config-{run_id}",
            }
        ),
        encoding="utf-8",
    )
    return metrics_path


def test_restore_comparison_separates_analysis_and_strict_quality(
    tmp_path: Path,
) -> None:
    old = _write_run(tmp_path, "20260809T193919Z", with_quality=False)
    current = _write_run(tmp_path, "20260829T010101Z", with_quality=True)
    assert old.is_file()

    outputs = build_restore_comparison(
        tmp_path,
        "20260829T010101Z",
        current,
        historical_run_ids=("20260809T193919Z",),
    )

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    runs = {item["run_id"]: item for item in payload["runs"]}
    assert runs["20260809T193919Z"]["strict_quality_count"] is None
    assert runs["20260829T010101Z"]["strict_quality_count"] == 192
    assert runs["20260829T010101Z"]["metric_computable_count"] == 192
    assert payload["raw_corpus_match"] is True
    assert outputs.markdown_path.is_file()
    assert outputs.report_path.is_file()


def test_restore_comparison_rejects_noncanonical_row_count(tmp_path: Path) -> None:
    current = _write_run(tmp_path, "20260829T020202Z", with_quality=True)
    pd.read_csv(current).iloc[:-1].to_csv(current, index=False)

    outputs = build_restore_comparison(
        tmp_path,
        "20260829T020202Z",
        current,
        historical_run_ids=(),
    )

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    assert payload["runs"][0]["artifact_status"] == "invalid_row_count"
    assert payload["runs"][0]["row_count"] == 191


def test_run28_validity_is_reported_as_legacy_strict_quality(tmp_path: Path) -> None:
    old = _write_run(tmp_path, "20260828T220000Z", with_quality=False)
    current = _write_run(tmp_path, "20260829T030303Z", with_quality=True)
    frame = pd.read_csv(old)
    frame.loc[:9, ["status", "validation_status"]] = ["excluded", "excluded"]
    frame.to_csv(old, index=False)

    outputs = build_restore_comparison(
        tmp_path,
        "20260829T030303Z",
        current,
        historical_run_ids=("20260828T220000Z",),
    )

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    runs = {item["run_id"]: item for item in payload["runs"]}
    assert runs["20260828T220000Z"]["analysis_valid_count"] == 182
    assert runs["20260828T220000Z"]["strict_quality_count"] == 182
