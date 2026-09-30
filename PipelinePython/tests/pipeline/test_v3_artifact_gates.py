from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from PipelinePython.common.validation_v3 import V3ReleaseValidator


ROOT = Path(__file__).resolve().parents[3]


def test_regenerated_data_gates() -> None:
    raw_index = pd.read_parquet(
        ROOT / "reports" / "data_quality" / "raw_input_index_v3.parquet"
    )
    ingestion = pd.read_parquet(
        ROOT / "data" / "bronze" / "ingestion_manifest_v3.parquet"
    )
    sync = pd.read_parquet(
        ROOT / "data" / "sync" / "synchronization_manifest_v3.parquet"
    )
    segmented = pd.read_parquet(
        ROOT / "data" / "segmented" / "segmentation_manifest_v3.parquet"
    )
    gold = pd.read_parquet(ROOT / "data" / "gold" / "gold_features_v3.parquet")
    sequences = pd.read_parquet(
        ROOT / "data" / "gold" / "gold_ibi_sequences_v3.parquet"
    )
    manifest = json.loads(
        (ROOT / "data" / "gold" / "manifest_v3.json").read_text(encoding="utf-8")
    )
    readiness = json.loads(
        (ROOT / "data" / "gold" / "data_readiness.json").read_text(encoding="utf-8")
    )

    assert len(raw_index) == 32 * 2 * 4
    assert (raw_index["status"] == "ok").all()
    assert sorted(raw_index["sha256"]) == sorted(ingestion["source_sha256"])
    assert set(raw_index["phase"]) == {"relax", "task"}
    assert len(sync) == 32 * 2
    assert sync.loc[sync["excluded"], "exclusion_reason"].str.len().gt(0).all()
    assert (
        segmented[segmented["sensor_channel"].isin(["ppg_ir", "ppg_red", "ppg_green"])][
            "status"
        ]
        == "ok"
    ).all()
    assert set(gold["participant_id"]).issubset({f"ID-{i:02d}" for i in range(1, 33)})
    assert set(gold["phase"]) == {"relax", "task"}
    assert set(gold["sensor_channel"]) == {"ppg_ir", "ppg_red", "ppg_green"}
    assert readiness["status"] in {"ready", "blocked"}
    if readiness["status"] == "ready":
        assert gold["participant_id"].nunique() == 32
        assert gold.loc[gold["split"] == "train", "participant_id"].nunique() == 28
        assert gold.loc[gold["split"] == "parity_pool", "participant_id"].nunique() == 4
    else:
        assert readiness["missing_groups"]
    assert len(manifest["feature_order"]) >= 30
    assert not any(
        name.startswith(("ecg", "pat")) for name in manifest["feature_order"]
    )
    assert manifest["sequence_length"] == 720
    assert len(sequences) == len(gold)
    assert set(sequences["ibi_values"].map(len)) == {720}
    assert set(sequences["ibi_mask"].map(len)) == {720}


def test_release_validator_keeps_physical_android_gate_pending() -> None:
    result = V3ReleaseValidator(ROOT).run()

    assert result["data_gates"]["raw_index_complete"]
    assert result["data_gates"]["raw_hash_matches_cleanup_baseline"]
    assert result["data_gates"]["ppg_segments_complete"]
    assert result["deployment"]["android_ready"] is False
    assert "physical_android_parity_pending" in result["blockers"]
