"""Regression tests for participant-safe DL evaluation boundaries."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from PipelinePython.common.config import get_locked_balanced_accuracy_gate  # noqa: E402
from PipelinePython.common.settings import load_pipeline_config  # noqa: E402
from PipelinePython.common import training_dl_v4, training_dl_v5  # noqa: E402


@pytest.mark.parametrize("gate", [0.0, 0.65, 0.7999, 0.8001, 1.0])
def test_legacy_gate_reader_ignores_noncanonical_sources(
    tmp_path: Path, gate: float
) -> None:
    gates_path = tmp_path / "frozen_gates.yaml"
    gates_path.write_text(
        f"training:\n  min_balanced_accuracy: {gate}\n", encoding="utf-8"
    )

    assert get_locked_balanced_accuracy_gate(gates_path) == 0.70


def test_legacy_gate_reader_uses_canonical_research_minimum(tmp_path: Path) -> None:
    gates_path = tmp_path / "frozen_gates.yaml"
    gates_path.write_text(
        "training:\n  min_balanced_accuracy: 0.80\n", encoding="utf-8"
    )

    assert get_locked_balanced_accuracy_gate(gates_path) == 0.70


def test_runtime_training_config_rejects_subminimum_research_gate() -> None:
    training = load_pipeline_config().training

    with pytest.raises(ValueError, match="0.70/0.70/0.70"):
        replace(training, minimum_research_balanced_accuracy=0.6999)

    assert training.min_balanced_accuracy == 0.80
    assert training.minimum_research_balanced_accuracy == 0.70


def test_local_trainers_use_the_approved_28_plus_4_contract() -> None:
    config = load_pipeline_config()
    assert config.training.development_participant_ids == tuple(
        f"ID-{index:02d}" for index in range(1, 29)
    )
    assert config.training.parity_pool_participant_ids == tuple(
        f"ID-{index:02d}" for index in range(29, 33)
    )


@pytest.mark.parametrize("module", [training_dl_v4, training_dl_v5])
def test_unknown_split_values_fail_closed(module: object) -> None:
    splits = pd.Series(["development", "mystery", "holdout"])

    with pytest.raises(ValueError, match="Unknown split values.*mystery"):
        module._normalize_splits(splits)


@pytest.mark.parametrize("module", [training_dl_v4, training_dl_v5])
def test_early_stopping_split_is_deterministic_and_participant_grouped(
    module: object,
) -> None:
    groups = np.repeat(["ID-01", "ID-02", "ID-03", "ID-04"], 3)

    train_a, val_a = module._participant_group_train_val_indices(
        groups, seed=42, validation_fraction=0.25
    )
    train_b, val_b = module._participant_group_train_val_indices(
        groups, seed=42, validation_fraction=0.25
    )

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert set(groups[train_a]).isdisjoint(set(groups[val_a]))


@pytest.mark.parametrize("module", [training_dl_v4, training_dl_v5])
def test_stable_seed_offset_is_process_independent(module: object) -> None:
    assert module._stable_seed_offset("ID-17") == module._stable_seed_offset("ID-17")
    assert module._stable_seed_offset("ID-17") != module._stable_seed_offset("ID-18")


@pytest.mark.parametrize("module", [training_dl_v4, training_dl_v5])
def test_holdout_labels_cannot_change_frozen_development_threshold(
    module: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_pipeline_config()
    trainer = module.LocalStressDLTrainer(tmp_path, config)
    trainer.input_channels = 1
    trainer.tab_dim = 2

    monkeypatch.setattr(module, "_build_model", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(
        module, "_train_model", lambda model, *args, **kwargs: (model, 0.5)
    )
    monkeypatch.setattr(
        module,
        "_infer_proba",
        lambda model, features, **kwargs: np.array([0.2, 0.4, 0.6, 0.8]),
    )

    dev_waveforms = np.arange(32, dtype=np.float32).reshape(8, 4)
    dev_tabular = np.arange(16, dtype=np.float32).reshape(8, 2)
    holdout_waveforms = np.arange(16, dtype=np.float32).reshape(4, 4)
    holdout_tabular = np.arange(8, dtype=np.float32).reshape(4, 2)
    dev_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    dev_groups = np.repeat(["ID-01", "ID-02", "ID-03", "ID-04"], 2)
    frozen_threshold = 0.63

    def evaluate(holdout_labels: np.ndarray) -> float:
        _, metrics, _, _ = trainer._train_final_and_eval_holdout(
            arch_name="Hybrid_1D_CNN",
            dev_features=(dev_waveforms, dev_tabular),
            dev_labels=dev_labels,
            dev_groups=dev_groups,
            holdout_features=(holdout_waveforms, holdout_tabular),
            holdout_labels=holdout_labels,
            best_params={},
            sensor_channel="ppg_ir",
            dev_threshold=frozen_threshold,
        )
        return metrics.threshold

    assert evaluate(np.array([0, 0, 1, 1])) == frozen_threshold
    assert evaluate(np.array([1, 1, 0, 0])) == frozen_threshold


def test_hpo_does_not_tune_full_training_epoch_budget() -> None:
    class Trial:
        def suggest_float(self, name, *args, **kwargs):
            return 1e-3

        def suggest_categorical(self, name, choices):
            return choices[0]

        def suggest_int(self, *args, **kwargs):
            raise AssertionError("num_epochs must not be an Optuna parameter")

    params = training_dl_v4._suggest_parameters_dl(Trial())
    assert "num_epochs" not in params


def test_cpu_hybrid_train_and_validation_dispatch_both_inputs() -> None:
    class TwoInputModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, waveform, tabular):
            return self.linear(
                torch.stack((waveform.mean(dim=(1, 2)), tabular.mean(dim=1)), dim=1)
            )

    wave = np.arange(32, dtype=np.float32).reshape(8, 4)
    tab = np.arange(16, dtype=np.float32).reshape(8, 2)
    labels = np.array([0, 1] * 4, dtype=np.float32)
    model, _ = training_dl_v4._train_model(
        TwoInputModel(),
        (wave, tab),
        labels,
        (wave, tab),
        labels,
        num_epochs=1,
        batch_size=4,
        device="cpu",
    )
    assert isinstance(model, TwoInputModel)


def test_parity_access_ledger_is_candidate_checksum_keyed_and_one_time(
    tmp_path: Path,
) -> None:
    trainer = training_dl_v4.LocalStressDLTrainer(tmp_path, load_pipeline_config())
    checksum = "a" * 64
    receipt, claim = trainer._claim_parity_access(
        sensor_channel="ppg_ir", candidate_checksum=checksum, threshold=0.57
    )
    assert not receipt.exists()
    assert claim["cohort_access_count"] == 1
    with pytest.raises(RuntimeError, match="already claimed"):
        trainer._claim_parity_access(
            sensor_channel="ppg_ir", candidate_checksum=checksum, threshold=0.57
        )


def test_hybrid_bundle_persists_imputer_statistics(tmp_path: Path) -> None:
    trainer = training_dl_v4.LocalStressDLTrainer(tmp_path, load_pipeline_config())
    bundle = tmp_path / "bundle"
    preprocessing = {
        "mean": np.array([0.0], dtype=np.float32),
        "scale": np.array([1.0], dtype=np.float32),
        "tabular_imputer": np.array([2.0, 3.0], dtype=np.float32),
        "tabular_mean": np.array([2.5, 3.5], dtype=np.float32),
        "tabular_scale": np.array([0.5, 0.5], dtype=np.float32),
    }
    trainer._write_model_bundle(
        bundle_dir=bundle,
        model=nn.Linear(1, 1),
        preprocessing=preprocessing,
        arch_name="Hybrid_1D_CNN",
        arch_params={},
    )
    scaler = json.loads((bundle / "scaler.json").read_text(encoding="utf-8"))
    assert scaler["tabular_imputer"] == [2.0, 3.0]


def test_fusion_rejects_cross_channel_lineage_mismatch() -> None:
    expected = (
        np.array([0, 1]),
        np.array(["ID-01", "ID-02"]),
        np.array(["development", "holdout"]),
    )
    mismatched = (
        np.array([1, 0]),
        expected[1].copy(),
        expected[2].copy(),
    )

    with pytest.raises(ValueError, match="ppg_red: label differs"):
        training_dl_v5._assert_fusion_lineage(expected, {"ppg_red": mismatched})


def test_fusion_rejects_timestamp_or_phase_mismatch() -> None:
    def frame(channel: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "participant_id": ["ID-01"],
                "session_id": ["ID-01"],
                "recording_id": ["ID-01_full"],
                "phase": ["relax"],
                "label": [0],
                "split": ["development"],
                "window_id": [f"ID-01_relax_0001_{channel}"],
                "window_start_utc": ["2026-01-01T00:00:00Z"],
                "window_end_utc": ["2026-01-01T00:03:00Z"],
            }
        )

    frames = {channel: frame(channel) for channel in ("ppg_ir", "ppg_red", "ppg_green")}
    frames["ppg_red"].loc[0, "window_start_utc"] = "2026-01-01T00:00:01Z"

    with pytest.raises(ValueError, match="window_start_utc differs"):
        training_dl_v5._assert_fusion_frame_lineage(frames)
