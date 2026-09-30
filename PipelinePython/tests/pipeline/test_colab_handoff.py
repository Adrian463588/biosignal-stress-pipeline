"""Tests for the local-only, conditional Google Colab DL handoff."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PipelinePython.common.cli import _prepare_colab_handoff
from PipelinePython.common.colab_handoff import (
    DEVELOPMENT_PARTICIPANTS,
    DL_ARCHITECTURES,
    HandoffContractError,
    build_handoff,
    fold_contract_sha256,
    load_handoff_manifest,
    validate_gold_artifact,
    validate_resume_manifest,
    write_handoff_manifest,
)
from PipelinePython.common.settings import load_pipeline_config

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = (
    ROOT
    / "PipelinePython"
    / "03_stress_onnx"
    / "tools"
    / "build_colab_training_notebook.py"
)


def _gold_ibi_artifact(directory: Path) -> Path:
    artifact = directory / "gold_ibi_sequences_v3.parquet"
    artifact.write_bytes(b"small real-artifact test fingerprint")
    return artifact


def _local_ml_handoff(*, requires_dl: bool = True) -> dict[str, dict[str, object]]:
    return {
        "ppg_ir": {
            "requires_dl": requires_dl,
            "trigger_reason": "local_ml_development_metrics_below_minimum_research_gate"
            if requires_dl
            else "local_ml_gate_passed",
            "predictor_columns": ["mean_ibi_ms", "rmssd_ms"],
        },
        "ppg_green": {
            "requires_dl": False,
            "trigger_reason": "local_ml_gate_passed",
            "predictor_columns": ["mean_ibi_ms"],
        },
    }


def test_handoff_is_local_manifest_with_only_ml_escalation_channels(
    tmp_path: Path,
) -> None:
    handoff = build_handoff(
        run_id="20260801T120000Z",
        gold_ibi_path=_gold_ibi_artifact(tmp_path),
        training_config_sha256="a" * 64,
        local_ml_handoff=_local_ml_handoff(),
    )

    destination = write_handoff_manifest(handoff, tmp_path / "colab_dl_handoff.json")
    loaded = load_handoff_manifest(destination)

    assert loaded.dl_handoff_channels == ("ppg_ir",)
    assert loaded.sequence_length == 720
    assert loaded.value_column == "ibi_values"
    assert loaded.mask_column == "ibi_mask"
    assert loaded.fold_contract.development_participants == DEVELOPMENT_PARTICIPANTS


def test_handoff_rejects_no_ml_escalation_channel(tmp_path: Path) -> None:
    with pytest.raises(HandoffContractError, match="no local-ML channels"):
        build_handoff(
            run_id="20260801T120000Z",
            gold_ibi_path=_gold_ibi_artifact(tmp_path),
            training_config_sha256="a" * 64,
            local_ml_handoff=_local_ml_handoff(requires_dl=False),
        )


def test_handoff_allows_independent_red_escalation(tmp_path: Path) -> None:
    handoff = _local_ml_handoff()
    handoff["ppg_red"] = {
        "requires_dl": True,
        "trigger_reason": "local_ml_development_metrics_below_minimum_research_gate",
        "predictor_columns": ["mean_ibi_ms"],
    }

    manifest = build_handoff(
        run_id="20260801T120000Z",
        gold_ibi_path=_gold_ibi_artifact(tmp_path),
        training_config_sha256="a" * 64,
        local_ml_handoff=handoff,
    )

    assert set(manifest.dl_handoff_channels) == {"ppg_ir", "ppg_red"}


def test_handoff_rejects_ecg_pat_and_motion_predictors(tmp_path: Path) -> None:
    handoff = _local_ml_handoff()
    handoff["ppg_ir"]["predictor_columns"] = ["mean_ibi_ms", "pat_median_ms"]

    with pytest.raises(HandoffContractError, match="forbidden predictors"):
        build_handoff(
            run_id="20260801T120000Z",
            gold_ibi_path=_gold_ibi_artifact(tmp_path),
            training_config_sha256="a" * 64,
            local_ml_handoff=handoff,
        )


def test_gold_checksum_and_resume_contract_fail_closed(tmp_path: Path) -> None:
    artifact = _gold_ibi_artifact(tmp_path)
    handoff = build_handoff(
        run_id="20260801T120000Z",
        gold_ibi_path=artifact,
        training_config_sha256="b" * 64,
        local_ml_handoff=_local_ml_handoff(),
    )
    resume = {
        "schema_version": handoff.schema_version,
        "gold_sha256": handoff.gold_sha256,
        "training_config_sha256": handoff.training_config_sha256,
        "fold_contract_sha256": fold_contract_sha256(handoff.fold_contract),
        "channel": "ppg_ir",
        "architecture": "1d_cnn",
        "completed_folds": ["ID-01", "ID-02"],
    }

    validate_gold_artifact(artifact, handoff)
    validate_resume_manifest(resume, handoff)
    artifact.write_bytes(b"changed")
    with pytest.raises(HandoffContractError, match="checksum"):
        validate_gold_artifact(artifact, handoff)

    resume["completed_folds"] = ["ID-29"]
    with pytest.raises(HandoffContractError, match="non-development"):
        validate_resume_manifest(resume, handoff)


def test_generator_creates_dl_only_t4_notebook(tmp_path: Path) -> None:
    notebook_path = tmp_path / "BiosignalStressTraining_Colab.ipynb"
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(notebook_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"generated_cell_{index}", "exec")

    assert notebook["nbformat"] == 4
    assert "gold_ibi_sequences_v3.parquet" in source
    assert "ibi_values" in source and "ibi_mask" in source
    assert "torch.cuda.is_available" in source and '"T4" in GPU_NAME.upper()' in source
    assert "KeyboardInterrupt" in source and "os.replace" in source
    assert "every epoch, atomically" in source and "fold_index % 5 == 0" in source
    assert tuple(DL_ARCHITECTURES) == ("1d_cnn", "tcn")
    assert "DataRenamedPartisipan" not in source
    assert "DataTrimmedPartisipan" not in source

    split_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def split_outer_fold_indices" in "".join(cell["source"])
    )
    namespace = {"np": np}
    split_helper = split_cell.partition("def train_outer_fold")[0]
    exec(split_helper, namespace)
    participants = np.asarray(["ID-01", "ID-02", "ID-03", "ID-04"])
    train_indices, test_indices = namespace["split_outer_fold_indices"](
        participants, {"ID-03", "ID-04"}
    )
    assert participants[train_indices].tolist() == ["ID-01", "ID-02"]
    assert participants[test_indices].tolist() == ["ID-03", "ID-04"]


def test_cli_handoff_writes_only_escalated_channel_ibi_contract(
    tmp_path: Path,
) -> None:
    required = {
        "participant_id": "ID-01",
        "session_id": "session-1",
        "phase": "relax",
        "window_id": "window-1",
        "label": 0,
        "split": "development",
        "quality_score": 0.95,
        "quality_tier": "good",
        "source_file": "deidentified-source",
        "source_checksum": "a" * 64,
        "preprocessing_version": "v3.0.0",
        "feature_version": "v3.0.0_core240s",
        "ibi_values": [1.0] * 720,
        "ibi_mask": [1] * 720,
    }
    source = tmp_path / "gold_ibi_sequences_v3.parquet"
    pd.DataFrame(
        [
            {**required, "sensor_channel": "ppg_ir", "motion_score": 0.1},
            {**required, "sensor_channel": "ppg_green", "motion_score": 0.2},
            {**required, "sensor_channel": "ppg_red", "motion_score": 0.3},
        ]
    ).to_parquet(source, index=False)
    loaded = load_pipeline_config()
    config = replace(
        loaded,
        artifacts={
            **loaded.artifacts,
            "gold_input_ibi_sequences": str(source),
            "reports_dir": str(tmp_path / "reports"),
            "run_id": "20260809T120000Z",
        },
    )
    reports = {
        "ppg_green": {
            "requires_dl": False,
            "predictor_columns": ["mean_ibi_ms"],
            "channel_data_fingerprint": "b" * 64,
        },
        "ppg_ir": {
            "requires_dl": True,
            "trigger_reason": "local_ml_balanced_accuracy_below_gate",
            "predictor_columns": ["mean_ibi_ms"],
            "channel_data_fingerprint": "c" * 64,
        },
    }

    manifest = _prepare_colab_handoff(
        tmp_path,
        config,
        reports,
        config_path="PipelinePython/configs/data.yaml",
    )

    assert manifest is not None
    filtered = pd.read_parquet(
        manifest.parent / "colab_input" / "gold_ibi_sequences_v3.parquet"
    )
    assert set(filtered["sensor_channel"]) == {"ppg_ir"}
    assert "motion_score" not in filtered.columns
    assert "ecg" not in " ".join(filtered.columns).lower()
    assert "pat" not in " ".join(filtered.columns).lower()


def test_cli_handoff_derives_frozen_split_and_allows_independent_red(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gold_ibi_sequences_v3.parquet"
    rows = []
    for participant_id in ("ID-01", "ID-29"):
        rows.append(
            {
                "participant_id": participant_id,
                "session_id": f"session-{participant_id}",
                "phase": "task",
                "sensor_channel": "ppg_red",
                "window_id": f"window-{participant_id}",
                "label": 1,
                "quality_score": 0.9,
                "quality_tier": "good",
                "source_file": "deidentified-source",
                "source_checksum": "a" * 64,
                "preprocessing_version": "v3.0.0",
                "feature_version": "v3.0.0_core240s",
                "ibi_values": [1.0] * 720,
                "ibi_mask": [1] * 720,
            }
        )
    pd.DataFrame(rows).to_parquet(source, index=False)
    loaded = load_pipeline_config()
    config = replace(
        loaded,
        artifacts={
            **loaded.artifacts,
            "gold_input_ibi_sequences": str(source),
            "reports_dir": str(tmp_path / "reports"),
            "run_id": "20260809T120000Z",
        },
    )

    manifest = _prepare_colab_handoff(
        tmp_path,
        config,
        {
            "ppg_red": {
                "requires_dl": True,
                "trigger_reason": "local_ml_balanced_accuracy_below_gate",
                "predictor_columns": ["rmssd_ms"],
                "channel_data_fingerprint": "c" * 64,
            }
        },
        config_path="PipelinePython/configs/data.yaml",
    )

    assert manifest is not None
    filtered = pd.read_parquet(
        manifest.parent / "colab_input" / "gold_ibi_sequences_v3.parquet"
    )
    assert set(filtered["sensor_channel"]) == {"ppg_red"}
    assert dict(zip(filtered["participant_id"], filtered["split"], strict=True)) == {
        "ID-01": "development",
        "ID-29": "holdout",
    }
