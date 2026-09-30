"""Participant-grouped hyperparameter optimization for classical PPG models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .feature_policy import correlation_pruned_features
from .settings import TrainingConfig
from .training_protocol import session_balanced_accuracy


def _participant_phase_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each participant-phase group one total fit weight."""

    required = {"participant_id", "phase"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"weight_groups_missing:{','.join(missing)}")
    groups = frame["participant_id"].astype(str) + "|" + frame["phase"].astype(str)
    counts = groups.value_counts(sort=False)
    return groups.map(lambda group: 1.0 / float(counts[group])).to_numpy(dtype=float)


def _fit_fold_model(
    model: object, features: pd.DataFrame, labels: np.ndarray, frame: pd.DataFrame
) -> None:
    """Fit canonical sklearn pipelines with equal participant-phase weights.

    The small non-Pipeline branch keeps the optimizer's protocol tests and
    third-party factories usable.  All production models are built by
    ``ClassicalModelFactory`` and therefore take the weighted Pipeline branch.
    """

    if isinstance(model, Pipeline):
        model.fit(
            features,
            labels,
            classifier__sample_weight=_participant_phase_weights(frame),
        )
        return
    model.fit(features, labels)


class TunableModelFactory(Protocol):
    """Build and parameterize one exportable estimator family."""

    def build(self, model_name: str, parameters: dict[str, Any]) -> Pipeline:
        """Create a fresh estimator with preprocessing inside the pipeline."""

    def suggest(self, model_name: str, trial: optuna.Trial) -> dict[str, Any]:
        """Return one candidate parameter set for the requested family."""


class ClassicalModelFactory:
    """Create the five independent-channel classical baseline families."""

    def __init__(self, random_seed: int) -> None:
        self.random_seed = random_seed

    def build(self, model_name: str, parameters: dict[str, Any]) -> Pipeline:
        common_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if model_name == "logistic_regression":
            common_steps.append(("scaler", StandardScaler()))
            classifier = LogisticRegression(
                random_state=self.random_seed,
                max_iter=2_000,
                solver="lbfgs",
                **parameters,
            )
        elif model_name == "rbf_svm":
            common_steps.append(("scaler", StandardScaler()))
            classifier = SVC(
                probability=True,
                random_state=self.random_seed,
                kernel="rbf",
                **parameters,
            )
        elif model_name == "extra_trees":
            classifier = ExtraTreesClassifier(
                random_state=self.random_seed,
                n_jobs=1,
                **parameters,
            )
        elif model_name == "lightgbm":
            classifier = LGBMClassifier(
                random_state=self.random_seed,
                n_jobs=1,
                verbosity=-1,
                subsample_freq=1,
                **parameters,
            )
        elif model_name == "xgboost":
            classifier = XGBClassifier(
                eval_metric="logloss",
                random_state=self.random_seed,
                n_jobs=1,
                tree_method="hist",
                **parameters,
            )
        else:
            raise ValueError(f"Unsupported classical model: {model_name}")
        return Pipeline([*common_steps, ("classifier", classifier)])

    def defaults(self, model_name: str) -> dict[str, Any]:
        defaults = {
            "logistic_regression": {"C": 1.0},
            "rbf_svm": {"C": 1.0, "gamma": "scale"},
            "extra_trees": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            "lightgbm": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "max_depth": -1,
                "min_child_samples": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        }
        try:
            return defaults[model_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported classical model: {model_name}") from exc

    def suggest(self, model_name: str, trial: optuna.Trial) -> dict[str, Any]:
        if model_name == "logistic_regression":
            return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
        if model_name == "rbf_svm":
            return {
                "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            }
        if model_name == "extra_trees":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [None, 8, 12, 16, 20]
                ),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.5, 1.0]
                ),
            }
        if model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.10, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [-1, 4, 6, 8, 10, 12]
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.70, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            }
        if model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.10, log=True
                ),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "subsample": trial.suggest_float("subsample", 0.70, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            }
        raise ValueError(f"Unsupported classical model: {model_name}")


@dataclass(frozen=True)
class GroupedOptimizationResult:
    parameters: dict[str, Any]
    balanced_accuracy: float
    trials: int
    group_splits: int


class GroupedHyperparameterOptimizer:
    """Optimize strictly inside participant-grouped development folds."""

    def __init__(
        self,
        config: TrainingConfig,
        factory: TunableModelFactory,
        *,
        storage_uri: str | None = None,
    ) -> None:
        self.config = config
        self.factory = factory
        self.storage_uri = storage_uri

    def optimize(
        self,
        frame: pd.DataFrame,
        feature_order: tuple[str, ...],
        model_name: str,
        *,
        seed_offset: int = 0,
        study_context: Mapping[str, Any] | None = None,
        promoted_experimental: tuple[str, ...] = (),
    ) -> GroupedOptimizationResult:
        groups = frame["participant_id"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < self.config.inner_group_splits:
            raise ValueError("Insufficient participants for inner GroupKFold")
        features = frame.loc[:, feature_order]
        labels = frame["label"].to_numpy(dtype=int)
        splitter = GroupKFold(n_splits=self.config.inner_group_splits)

        def objective(trial: optuna.Trial) -> float:
            parameters = self.factory.suggest(model_name, trial)
            probabilities: list[np.ndarray] = []
            validation_frames: list[pd.DataFrame] = []
            for train_indices, validation_indices in splitter.split(
                features, labels, groups
            ):
                selected_features = correlation_pruned_features(
                    features.iloc[train_indices],
                    feature_order,
                    getattr(self.config, "correlation_pruning_threshold", 0.95),
                    promoted_experimental,
                )
                model = self.factory.build(model_name, parameters)
                train_frame = frame.iloc[train_indices]
                _fit_fold_model(
                    model,
                    features.iloc[train_indices].loc[:, selected_features],
                    labels[train_indices],
                    train_frame,
                )
                probabilities.append(
                    model.predict_proba(
                        features.iloc[validation_indices].loc[:, selected_features]
                    )[:, 1]
                )
                validation_frames.append(frame.iloc[validation_indices])
            return session_balanced_accuracy(
                pd.concat(validation_frames, ignore_index=True),
                np.concatenate(probabilities),
                self.config.decision_threshold,
            )

        sampler = optuna.samplers.TPESampler(seed=self.config.random_seed + seed_offset)
        identity_columns = [
            name
            for name in (
                "participant_id",
                "phase",
                "sensor_channel",
                "label",
                "quality_tier",
                *feature_order,
            )
            if name in frame
        ]
        canonical = frame.loc[:, identity_columns].sort_values(
            identity_columns, kind="mergesort"
        )
        frame_sha256 = hashlib.sha256(
            canonical.to_csv(
                index=False,
                lineterminator="\n",
                float_format="%.17g",
                na_rep="<NA>",
            ).encode("utf-8")
        ).hexdigest()
        identity_payload = {
            "model_name": model_name,
            "seed_offset": seed_offset,
            "feature_order": list(feature_order),
            "groups": sorted(unique_groups.tolist()),
            "frame_sha256": frame_sha256,
            "study_context": dict(study_context or {}),
        }
        identity = hashlib.sha256(
            json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        storage = self.storage_uri or getattr(self.config, "optuna_storage_uri", None)
        if storage and storage.startswith("sqlite:///"):
            Path(storage.removeprefix("sqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=storage,
            study_name=f"{model_name}-{identity}",
            load_if_exists=bool(storage),
        )
        completed_trials = sum(
            trial.state is optuna.trial.TrialState.COMPLETE for trial in study.trials
        )
        remaining_trials = max(self.config.optuna_trials - completed_trials, 0)
        if remaining_trials:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                n_jobs=1,
                show_progress_bar=False,
            )
        return GroupedOptimizationResult(
            parameters=dict(study.best_params),
            balanced_accuracy=float(study.best_value),
            trials=sum(
                trial.state is optuna.trial.TrialState.COMPLETE
                for trial in study.trials
            ),
            group_splits=self.config.inner_group_splits,
        )
