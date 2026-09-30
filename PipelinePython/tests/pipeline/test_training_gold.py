from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from PipelinePython.common.feature_policy import DEFAULT_PREDICTORS
from PipelinePython.common.training_gold import TrainingGoldBuilder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_uses_silver_clean_value_and_emits_real_window_variants(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "runs" / "20260809T044902Z"
    silver_path = source_run / "silver" / "ID-01" / "relax" / "ppg_green.parquet"
    silver_path.parent.mkdir(parents=True)
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=241, freq="1s")
    clean_values = np.arange(241, dtype=np.float64)
    pd.DataFrame(
        {
            "participant_id": "ID-01",
            "session_id": "ID-01",
            "recording_id": "ID-01_relax",
            "phase": "relax",
            "sensor_channel": "ppg_green",
            "timestamp_corrected_utc": timestamps,
            "core_start_utc": timestamps[0],
            "clean_value": clean_values,
            "raw_value": clean_values + 10_000.0,
            "interpolated_mask": False,
            "quality_score": 0.97,
            "quality_tier": "excellent",
            "quality_flags": "",
            "source_file": "data/raw/immutable.csv",
            "source_checksum": "a" * 64,
            "preprocessing_version": "v3.0.0",
            "sampling_rate_hz": 1.0,
        }
    ).to_parquet(silver_path, index=False)
    manifest_path = source_run / "silver" / "preprocessing_manifest_v3.parquet"
    pd.DataFrame(
        [
            {
                "participant_id": "ID-01",
                "phase": "relax",
                "sensor_channel": "ppg_green",
                "quality_tier": "excellent",
                "source_file": "data/raw/immutable.csv",
                "source_checksum": "a" * 64,
                "preprocessing_version": "v3.0.0",
                "path": str(silver_path),
            }
        ]
    ).to_parquet(manifest_path, index=False)

    observed_windows: list[np.ndarray] = []

    def fake_extract(
        signal: np.ndarray,
        sampling_rate_hz: float,
        *,
        channel: str,
        window_seconds: int,
        experimental_features: tuple[str, ...] = (),
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        del sampling_rate_hz, channel, experimental_features
        observed_windows.append(np.asarray(signal).copy())
        features = {
            feature: float(index + 1)
            for index, feature in enumerate(DEFAULT_PREDICTORS)
        }
        return (
            features,
            np.ones(window_seconds * 4, dtype=np.float32),
            np.ones(window_seconds * 4, dtype=np.uint8),
        )

    config = SimpleNamespace(
        model_version="training-gold-test",
        feature_version="v4.0.0_robust_channel_independent",
        training=SimpleNamespace(
            training_channels=("ppg_green", "ppg_ir", "ppg_red"),
            window_candidates=((60, 30), (120, 30), (180, 10), (180, 30), (240, 240)),
            strict_quality_tiers=("excellent", "good"),
            expanded_quality_tiers=("excellent", "good", "acceptable"),
        ),
        features=SimpleNamespace(experimental_only=()),
    )
    output_run = tmp_path / "runs" / "derived"

    result = TrainingGoldBuilder(
        project_root=tmp_path,
        config=config,
        source_run_root=source_run,
        output_run_root=output_run,
        extractor=fake_extract,
    ).build()

    assert result["source_silver_manifest_sha256"] == _sha256(manifest_path)
    assert result["source_run_id"] == "20260809T044902Z"
    assert len(observed_windows) == 23
    assert all(float(window.max()) < 10_000.0 for window in observed_windows)
    for window_seconds, stride_seconds in config.training.window_candidates:
        variant = (
            output_run
            / "gold"
            / "training_variants"
            / (f"w{window_seconds:03d}_s{stride_seconds:03d}")
        )
        for cohort in ("strict", "expanded"):
            features = pd.read_parquet(variant / cohort / "gold_features.parquet")
            assert (
                tuple(feature for feature in DEFAULT_PREDICTORS if feature in features)
                == DEFAULT_PREDICTORS
            )
            assert not features.loc[:, DEFAULT_PREDICTORS].isna().any().any()
    assert (output_run / "gold" / "data_readiness.json").is_file()


def test_builder_refuses_to_overwrite_source_run(tmp_path: Path) -> None:
    config = SimpleNamespace(training=SimpleNamespace(), features=SimpleNamespace())
    source = tmp_path / "same"

    builder = TrainingGoldBuilder(tmp_path, config, source, source)

    try:
        builder.build()
    except ValueError as exc:
        assert "source_and_output_run_must_differ" in str(exc)
    else:
        raise AssertionError("builder must not overwrite immutable Silver source")
