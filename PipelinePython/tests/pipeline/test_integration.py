"""
test_integration.py
-------------------
Integration tests for full pipeline execution on synthetic data.

Task 20.5: Smoke test that runs the core pipeline modules end-to-end.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PipelinePython.common.data_foundation import DataFoundationBuilder
from PipelinePython.common.settings import PipelinePaths, PipelineSettings

logger = logging.getLogger(__name__)


@pytest.fixture
def test_paths(tmp_path: Path) -> PipelinePaths:
    """Create a temporary directory structure mimicking the project."""
    raw_dir = tmp_path / "data" / "raw" / "DataTrimmedPartisipan"
    raw_dir.mkdir(parents=True)

    # Create one mock participant
    p_dir = raw_dir / "ID-01"
    p_dir.mkdir()
    (p_dir / "Relaks").mkdir()
    (p_dir / "Stress").mkdir()

    duration_seconds = 240
    n_green = 25 * duration_seconds

    # Mock PPG Green for Relax
    df_relax = pd.DataFrame(
        {
            "ppg_raw": np.random.randn(n_green)
            + 1.0,  # 4 mins at 25Hz for 180s windows
            "unix_ms": np.linspace(
                1777604400000, 1777604400000 + duration_seconds * 1000, n_green
            ),
            "datetime_str": "2026-05-01T10:00:00",
            "acc_x": 0.0,
            "acc_y": 0.0,
            "acc_z": 0.0,
        }
    )
    df_relax.to_csv(p_dir / "Relaks" / "ID-01_PPG Green_Relaks.csv", index=False)

    # Mock PPG Green for Stress
    df_stress = pd.DataFrame(
        {
            "ppg_raw": np.random.randn(n_green)
            + 1.0,  # 4 mins at 25Hz for 180s windows
            "unix_ms": np.linspace(
                1777608000000, 1777608000000 + duration_seconds * 1000, n_green
            ),
            "datetime_str": "2026-05-01T11:00:00",
            "acc_x": 0.0,
            "acc_y": 0.0,
            "acc_z": 0.0,
        }
    )
    df_stress.to_csv(p_dir / "Stress" / "ID-01_PPG Green_Stress.csv", index=False)

    # Create anchor mapping
    anchor_dir = tmp_path / "TimeStampAnchor"
    anchor_dir.mkdir()
    anchor_csv = anchor_dir / "participants_2026-05-03_22-47_WIB.csv"
    anchor_csv.write_text(
        "participant_code,phase,start_timestamp,end_timestamp\n"
        "ID-01,Relax,2026-05-01T10:00:00+07:00,2026-05-01T10:04:00+07:00\n"
        "ID-01,Task,2026-05-01T11:00:00+07:00,2026-05-01T11:04:00+07:00\n",
        encoding="utf-8",
    )

    # Map paths
    return PipelinePaths(
        project_root=tmp_path,
        raw_data_root=raw_dir,
        anchor_csv_path=anchor_csv,
        artifacts_root=tmp_path / "data" / "artifacts",
        bronze_dir=tmp_path / "data" / "artifacts" / "bronze",
        silver_dir=tmp_path / "data" / "artifacts" / "silver",
        gold_dir=tmp_path / "data" / "artifacts" / "gold",
        gold_features_path=tmp_path
        / "data"
        / "artifacts"
        / "gold"
        / "gold_features.csv",
        eda_dir=tmp_path / "data" / "artifacts" / "eda",
        validation_dir=tmp_path / "data" / "artifacts" / "validation",
        models_dir=tmp_path / "data" / "artifacts" / "models",
        export_dir=tmp_path / "data" / "artifacts" / "export",
    )


@pytest.fixture
def test_settings() -> PipelineSettings:
    """Return pipeline settings tailored for testing."""
    return PipelineSettings(
        timezone_local="Asia/Jakarta",
        analysis_window_seconds=180,
        anchor_buffer_seconds=5,
    )


def test_pipeline_data_foundation_end_to_end(
    test_paths: PipelinePaths, test_settings: PipelineSettings
) -> None:
    """Smoke test: execute Bronze -> Silver -> Gold layers on synthetic data."""
    # Create output directories
    for p in [test_paths.bronze_dir, test_paths.silver_dir, test_paths.gold_dir]:
        p.mkdir(parents=True, exist_ok=True)

    builder = DataFoundationBuilder(test_paths, test_settings)

    # 1. Ingest (Bronze)
    bronze = builder.run_ingest(participant="ID-01")
    assert not bronze.empty
    assert "signal_type" in bronze.columns
    assert "participant_code" in bronze.columns

    # 2. Segment (Silver)
    silver = builder.run_segment(bronze)
    assert not silver.empty
    assert "phase" in silver.columns
    silver_files = list(test_paths.silver_dir.glob("ID-01/*/*.parquet"))
    assert silver_files

    # 3. Features (Gold)
    # The synthetic data is purely random noise, so NeuroKit2 HRV extraction will
    # likely fail/return NaNs for this mock participant. We just verify it runs without crashing
    # and produces the gold_features.csv (even if empty or minimal).
    try:
        builder.run_gold_features("ID-01")
        gold_files = list(
            (test_paths.gold_dir / "features" / "stress").glob("ID-01/*/*.parquet")
        )
        window_files = list(
            (test_paths.gold_dir / "windows" / "stress").glob("ID-01/*/*.parquet")
        )
        assert window_files, "Window persistence must succeed before feature extraction"
        assert gold_files, (
            "Feature extraction must produce a traceable feature artifact"
        )
        assert not pd.read_parquet(gold_files[0]).empty
    except Exception as e:
        pytest.fail(f"Gold feature extraction crashed: {e}")
