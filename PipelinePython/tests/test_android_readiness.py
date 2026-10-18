from __future__ import annotations

import json
from pathlib import Path

import pytest

from PipelinePython.common.android_readiness import (
    _aggregate_research_status,
    _production_status,
    _sensor_path_receipt,
    evaluate_android_readiness,
    evaluate_internal_android_readiness,
)
from PipelinePython.common.cli import _parse_args


@pytest.mark.parametrize(
    "schema,cleaned", [("live-ppg-parity-v2", True), ("live-ppg-parity-v3", False)]
)
def test_sensor_receipt_requires_independent_canonical_preprocessing(
    tmp_path: Path, schema: str, cleaned: bool
) -> None:
    run_root = tmp_path / "20260916T132736Z"
    target = (
        run_root
        / "export/internal/device_parity/sensor_path/live_ppg_sensor_parity_receipt.json"
    )
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "run_id": run_root.name,
                "status": "GO",
                "sensor_path_parity_status": "GO",
                "actual_sensor_acquisition": True,
                "raw_lineage_verified": True,
                "preprocessing_verified": cleaned,
            }
        ),
        encoding="utf-8",
    )

    receipt = _sensor_path_receipt(run_root)

    assert receipt["status"] != "GO"
    assert receipt["actual_sensor_acquisition"] is False


def _research_index(
    run_id: str, models: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "model_version": f"stress-{run_id}",
        "release_status": "research_candidate",
        "artifact_status": "research_candidate",
        "evaluation_scope": "development_only",
        "deployment_scope": "research",
        "android_ready": False,
        "models": models or [],
    }


def _run_root(tmp_path: Path, run_id: str) -> Path:
    root = tmp_path / "data" / "artifacts" / "runs" / run_id
    (root / "export" / "research").mkdir(parents=True)
    (root / "export" / "research" / "research_model_index.json").write_text(
        json.dumps(_research_index(run_id)), encoding="utf-8"
    )
    return root


def test_android_readiness_rejects_stale_research_index(tmp_path: Path) -> None:
    run_id = "20260907T222002Z"
    root = _run_root(tmp_path, run_id)
    index_path = root / "export" / "research" / "research_model_index.json"
    index_path.write_text(
        json.dumps(_research_index("20260830T065109Z")), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="run_id_mismatch"):
        evaluate_android_readiness(
            tmp_path,
            run_id,
            apk_path=None,
            device_connection_status="READY",
        )


def test_research_and_production_statuses_are_explicit_and_independent(
    tmp_path: Path,
) -> None:
    run_id = "20260907T222002Z"
    _run_root(tmp_path, run_id)

    report = evaluate_android_readiness(
        tmp_path,
        run_id,
        apk_path=tmp_path / "missing.apk",
        device_connection_status="READY",
    )

    assert report["device_connection_status"] == "READY"
    assert report["research_model_status"] == "NO_GO"
    assert report["global_production_index_used"] is False
    assert report["research_model_status_by_channel"]["ppg_green"]["reason"] == (
        "research_channel_entry_missing"
    )
    assert (
        report["production_model_status_by_channel"]["ppg_green"]["status"] == "NO_GO"
    )
    assert report["production_model_status_by_channel"]["ppg_red"]["status"] == (
        "NO_GO"
    )


def test_device_offline_does_not_become_research_go(tmp_path: Path) -> None:
    run_id = "20260907T222002Z"
    _run_root(tmp_path, run_id)

    report = evaluate_android_readiness(
        tmp_path,
        run_id,
        apk_path=None,
        device_connection_status="OFFLINE",
    )

    assert report["research_model_status"] == "NO_GO"
    assert all(
        channel["device_connection_status"] == "OFFLINE"
        for channel in report["research_model_status_by_channel"].values()
    )


def test_one_verified_research_channel_is_enough_for_global_status() -> None:
    assert _aggregate_research_status(["GO", "NO_GO"]) == "GO"
    assert _aggregate_research_status(["NO_GO", "NO_GO"]) == "NO_GO"


def test_production_red_insufficient_does_not_block_green_no_go() -> None:
    readiness = {
        "model_release_status_by_channel": {
            "ppg_green": "blocked_existing_holdout_contract_mismatch",
            "ppg_ir": "blocked_existing_holdout_contract_mismatch",
            "ppg_red": "blocked_insufficient_ppg_coverage",
        }
    }

    green = _production_status("ppg_green", "20260907T222002Z", readiness, None)
    red = _production_status("ppg_red", "20260907T222002Z", readiness, None)

    assert green == {
        "status": "NO_GO",
        "reason": "blocked_existing_holdout_contract_mismatch",
    }
    assert red == {
        "status": "INSUFFICIENT_EVIDENCE",
        "reason": "blocked_insufficient_ppg_coverage",
    }


def test_android_readiness_is_a_non_pipeline_cli_command() -> None:
    args = _parse_args(
        [
            "android-readiness",
            "--run-id",
            "20260907T222002Z",
            "--apk",
            "app-research-debug.apk",
            "--device-status",
            "READY",
        ]
    )

    assert args.command == "android-readiness"
    assert args.receipt == []


def test_internal_readiness_rejects_legacy_status_fields(tmp_path: Path) -> None:
    run_id = "20260916T000000Z"
    index_path = (
        tmp_path
        / "data"
        / "artifacts"
        / "runs"
        / run_id
        / "export"
        / "internal"
        / "internal_model_index.json"
    )
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "internal-android-v2",
                "run_id": run_id,
                "validation_scope": "internal_32_participants",
                "claims_policy": "decision_support_only",
                "android_ready": False,
                "models": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="legacy_status"):
        evaluate_internal_android_readiness(
            tmp_path,
            run_id,
            apk_path=None,
            device_connection_status="READY",
        )
