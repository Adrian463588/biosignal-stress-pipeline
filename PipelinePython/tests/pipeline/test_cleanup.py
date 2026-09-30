from __future__ import annotations

from pathlib import Path

from PipelinePython.common.cleanup import apply_cleanup, build_cleanup_plan


def _write(path: Path, value: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_cleanup_moves_only_allowlisted_generated_outputs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    backup = tmp_path / "backup"
    _write(project / "data/raw/DataRenamedPartisipan/ID-01/Relax/raw.csv")
    _write(project / "TimeStampAnchor/anchors.csv")
    _write(project / "PersonalityAnchor/labels.csv")
    _write(project / "data/artifacts/gold/gold_features.csv", "legacy")
    _write(project / "data/artifacts/gold/gold_features_v3.parquet", "new")
    _write(project / "data/artifacts/models/stress/model.joblib")
    _write(project / "reports/model_eval/metrics.json")
    _write(project / "android/mobile/src/main/assets/stress_ppg_ir_model.onnx")

    plan = build_cleanup_plan(project, backup)
    planned = {entry.relative_path for entry in plan.entries}

    assert "data/raw" not in planned
    assert "data/artifacts/gold/gold_features.csv" not in planned
    assert "data/artifacts/gold/gold_features_v3.parquet" in planned
    assert "data/artifacts/models" in planned
    assert "reports" in planned

    result = apply_cleanup(plan)

    assert result.applied
    assert (project / "data/raw/DataRenamedPartisipan/ID-01/Relax/raw.csv").exists()
    assert (project / "data/artifacts/gold/gold_features.csv").exists()
    assert not (project / "data/artifacts/gold/gold_features_v3.parquet").exists()
    assert (backup / "data/artifacts/gold/gold_features_v3.parquet").exists()
    assert (backup / "cleanup_manifest.json").is_file()
    assert (project / "reports/migration/cleanup_reference_v3.json").is_file()


def test_cleanup_rejects_backup_inside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write(project / "data/raw/source.csv")

    try:
        build_cleanup_plan(project, project / "backup")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("expected unsafe backup root to be rejected")
