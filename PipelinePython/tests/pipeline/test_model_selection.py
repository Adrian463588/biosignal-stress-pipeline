from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from PipelinePython.common.model_selection import (
    ClassicalModelFactory,
    GroupedHyperparameterOptimizer,
)


class _FastEstimator:
    def __init__(
        self,
        offset: float,
        observed_splits: list[tuple[set[int], set[int]]],
    ) -> None:
        self.offset = offset
        self.observed_splits = observed_splits
        self.training_indices: set[int] = set()

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> _FastEstimator:
        del labels
        self.training_indices = set(features.index)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.observed_splits.append((self.training_indices, set(features.index)))
        probability = np.clip(
            features["ppi_median_ms"].to_numpy(dtype=float) / 1_000.0 + self.offset,
            0.01,
            0.99,
        )
        return np.column_stack([1.0 - probability, probability])


class _FastFactory:
    def __init__(self) -> None:
        self.observed_splits: list[tuple[set[int], set[int]]] = []

    def suggest(self, model_name: str, trial: optuna.Trial) -> dict[str, Any]:
        assert model_name == "fast"
        return {"offset": trial.suggest_float("offset", -0.05, 0.05)}

    def build(self, model_name: str, parameters: dict[str, Any]) -> _FastEstimator:
        assert model_name == "fast"
        return _FastEstimator(float(parameters["offset"]), self.observed_splits)


def test_inner_optimizer_uses_disjoint_participant_groups() -> None:
    participant_ids = np.repeat([f"ID-{index:02d}" for index in range(1, 11)], 4)
    labels = np.tile([0, 0, 1, 1], 10)
    frame = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "phase": np.tile(["relax", "relax", "task", "task"], 10),
            "sensor_channel": ["ppg_green"] * len(participant_ids),
            "label": labels,
            "ppi_median_ms": np.where(labels == 1, 900.0, 100.0),
        }
    )
    config = SimpleNamespace(
        inner_group_splits=5,
        optuna_trials=3,
        decision_threshold=0.5,
        random_seed=42,
    )
    factory = _FastFactory()

    result = GroupedHyperparameterOptimizer(config, factory).optimize(
        frame, ("ppi_median_ms",), "fast"
    )

    assert result.trials == 3
    assert result.group_splits == 5
    assert len(factory.observed_splits) == 15
    for training_indices, validation_indices in factory.observed_splits:
        training_groups = set(frame.loc[list(training_indices), "participant_id"])
        validation_groups = set(frame.loc[list(validation_indices), "participant_id"])
        assert training_groups.isdisjoint(validation_groups)
        assert training_groups | validation_groups == set(participant_ids)


def test_persistent_optuna_studies_are_isolated_by_channel_and_data(
    tmp_path: Path,
) -> None:
    participant_ids = np.repeat([f"ID-{index:02d}" for index in range(1, 11)], 2)
    labels = np.tile([0, 1], 10)
    frame = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "phase": np.tile(["relax", "task"], 10),
            "sensor_channel": ["ppg_green"] * len(participant_ids),
            "label": labels,
            "ppi_median_ms": np.where(labels == 1, 900.0, 100.0),
        }
    )
    storage = f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"
    config = SimpleNamespace(
        inner_group_splits=5,
        optuna_trials=1,
        decision_threshold=0.5,
        random_seed=42,
        optuna_storage_uri=storage,
    )
    optimizer = GroupedHyperparameterOptimizer(config, _FastFactory())

    optimizer.optimize(frame, ("ppi_median_ms",), "fast")
    red = frame.assign(sensor_channel="ppg_red")
    optimizer.optimize(red, ("ppi_median_ms",), "fast")

    assert len(optuna.get_all_study_summaries(storage=storage)) == 2


def test_persistent_optuna_resume_does_not_repeat_completed_trials(
    tmp_path: Path,
) -> None:
    participant_ids = np.repeat([f"ID-{index:02d}" for index in range(1, 11)], 2)
    labels = np.tile([0, 1], 10)
    frame = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "phase": np.tile(["relax", "task"], 10),
            "sensor_channel": ["ppg_ir"] * len(participant_ids),
            "label": labels,
            "ppi_median_ms": np.where(labels == 1, 900.0, 100.0),
        }
    )
    storage = f"sqlite:///{(tmp_path / 'resume.db').as_posix()}"
    config = SimpleNamespace(
        inner_group_splits=5,
        optuna_trials=2,
        decision_threshold=0.5,
        random_seed=42,
        optuna_storage_uri=storage,
    )
    factory = _FastFactory()
    optimizer = GroupedHyperparameterOptimizer(config, factory)

    first = optimizer.optimize(frame, ("ppi_median_ms",), "fast")
    split_count_after_first = len(factory.observed_splits)
    second = optimizer.optimize(frame, ("ppi_median_ms",), "fast")

    assert first.trials == second.trials == 2
    assert len(factory.observed_splits) == split_count_after_first


def test_group_weighting_is_not_multiplied_by_classifier_class_weight() -> None:
    factory = ClassicalModelFactory(42)

    for name in ("logistic_regression", "rbf_svm", "extra_trees", "lightgbm"):
        classifier = factory.build(name, factory.defaults(name)).named_steps[
            "classifier"
        ]
        assert classifier.get_params().get("class_weight") is None
