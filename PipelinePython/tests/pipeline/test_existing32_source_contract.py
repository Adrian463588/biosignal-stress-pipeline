from __future__ import annotations

from pathlib import Path

from PipelinePython.common.local_pipeline import LocalBiosignalPipeline
from PipelinePython.common.settings import load_pipeline_config


def test_participant_contract_contains_only_existing_ids() -> None:
    pipeline = LocalBiosignalPipeline(Path.cwd(), load_pipeline_config())

    assert pipeline._participant_ids() == tuple(
        f"ID-{index:02d}" for index in range(1, 33)
    )
    assert not any(
        value in pipeline._participant_ids()
        for value in ("ID-33", "ID-34", "ID-35", "ID-36")
    )


def test_source_policy_routes_ecg_and_ppg_differently() -> None:
    config = load_pipeline_config()

    assert config.raw_source.renamed_dir != config.raw_source.trimmed_dir
    assert config.raw_source.ecg_source == "renamed_complete"
    assert config.raw_source.ppg_source == "trimmed_phase"
    assert config.raw_source.trimmed_ecg_role == "audit_only"
