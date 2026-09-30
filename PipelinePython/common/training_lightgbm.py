"""Participant-safe LightGBM baseline for the ADR-015 local protocol.

ID-01..ID-28 are development participants and ID-29..ID-32 are the existing
engineering parity pool. No participant outside the real 32-person dataset is
invented or required.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MIN_BALANCED_ACCURACY_GATE, MIN_F1_GATE, MIN_ROC_AUC_GATE
from .settings import DEFAULT_PATHS, PipelinePaths, PipelineSettings

logger = logging.getLogger(__name__)

DEVELOPMENT_PARTICIPANTS = {f"ID-{number:02d}" for number in range(1, 29)}
PARITY_POOL_PARTICIPANTS = {f"ID-{number:02d}" for number in range(29, 33)}


class LightGBMTrainer:
    """Train a classical baseline with the approved participant boundary."""

    def __init__(
        self,
        paths: PipelinePaths | None = None,
        settings: PipelineSettings | None = None,
    ) -> None:
        self.paths = paths or DEFAULT_PATHS
        self.settings = settings or PipelineSettings()
        self.gate_threshold = MIN_BALANCED_ACCURACY_GATE

    @staticmethod
    def _feature_columns(df: pd.DataFrame, sensor_channel: str) -> list[str]:
        meta_cols = {
            "participant_id",
            "participant_code",
            "session_id",
            "recording_id",
            "phase",
            "sensor_channel",
            "window_id",
            "window_start_utc",
            "window_end_utc",
            "quality_tier",
            "quality_flags",
            "quality_score",
            "source_file",
            "feature_version",
            "label",
            "preprocessing_version",
            "signal_source",
            "window_idx",
            "window_start_ms",
            "window_end_ms",
        }
        other_sensors = {"ecg", "ppg_ir", "ppg_red", "ppg_green"} - {sensor_channel}
        return [
            column
            for column in df.columns
            if column not in meta_cols
            and not column.startswith("ecg_")
            and not any(column.startswith(f"{other}_") for other in other_sensors)
        ]

    @staticmethod
    def _pipeline() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=4,
                        num_leaves=15,
                        random_state=42,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    def run(self, sensor_channel: str = "ppg_green") -> dict[str, Any]:
        """Train on ID-01..ID-28 and evaluate ID-29..ID-32 after freezing."""

        gold_csv = self.paths.gold_dir / "gold_features.csv"
        if not gold_csv.exists():
            gold_csv = self.paths.artifacts_root / "gold" / "gold_features.csv"
        if not gold_csv.exists():
            raise FileNotFoundError(f"Gold feature table not found at {gold_csv}")

        frame = pd.read_csv(gold_csv)
        frame = frame[frame["sensor_channel"] == sensor_channel].copy()
        if frame.empty:
            raise ValueError(f"No rows found for sensor_channel={sensor_channel}")

        participant_column = (
            "participant_code"
            if "participant_code" in frame.columns
            else "participant_id"
        )
        observed = set(frame[participant_column].astype(str))
        missing_development = DEVELOPMENT_PARTICIPANTS - observed
        missing_holdout = PARITY_POOL_PARTICIPANTS - observed
        if missing_development or missing_holdout:
            raise RuntimeError(
                "Corrected LightGBM protocol is blocked: "
                f"missing development={sorted(missing_development)}, "
                f"missing parity_pool={sorted(missing_holdout)}"
            )

        unexpected = observed - DEVELOPMENT_PARTICIPANTS - PARITY_POOL_PARTICIPANTS
        if unexpected:
            raise ValueError(f"Unexpected participant lineage: {sorted(unexpected)}")

        development = frame[
            frame[participant_column].isin(DEVELOPMENT_PARTICIPANTS)
        ].copy()
        parity_pool = frame[
            frame[participant_column].isin(PARITY_POOL_PARTICIPANTS)
        ].copy()
        feature_columns = [
            column
            for column in self._feature_columns(frame, sensor_channel)
            if development[column].notna().any()
        ]
        if not feature_columns:
            raise ValueError(
                "No usable feature columns remain after development-only filtering"
            )

        x_development = development[feature_columns].to_numpy(dtype=np.float32)
        y_development = development["label"].to_numpy(dtype=int)
        groups = development[participant_column].astype(str).to_numpy()
        x_holdout = parity_pool[feature_columns].to_numpy(dtype=np.float32)
        y_holdout = parity_pool["label"].to_numpy(dtype=int)

        fold_metrics: list[dict[str, float | str]] = []
        for participant in sorted(np.unique(groups)):
            validation_mask = groups == participant
            model = self._pipeline()
            model.fit(x_development[~validation_mask], y_development[~validation_mask])
            predictions = model.predict(x_development[validation_mask])
            fold_metrics.append(
                {
                    "participant_id": participant,
                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            y_development[validation_mask], predictions
                        )
                    ),
                    "f1": float(
                        f1_score(
                            y_development[validation_mask], predictions, zero_division=0
                        )
                    ),
                }
            )

        nested_balanced_accuracy = float(
            np.mean([float(fold["balanced_accuracy"]) for fold in fold_metrics])
        )
        if nested_balanced_accuracy < self.gate_threshold:
            return {
                "model_name": "LightGBM",
                "sensor_channel": sensor_channel,
                "nested_balanced_accuracy": nested_balanced_accuracy,
                "release_status": "blocked_nested_gate",
                "holdout_evaluated": False,
                "folds": fold_metrics,
            }

        final_model = self._pipeline()
        final_model.fit(x_development, y_development)
        holdout_predictions = final_model.predict(x_holdout)
        holdout_probabilities = final_model.predict_proba(x_holdout)[:, 1]
        holdout_balanced_accuracy = float(
            balanced_accuracy_score(y_holdout, holdout_predictions)
        )
        holdout_f1 = float(f1_score(y_holdout, holdout_predictions, zero_division=0))
        holdout_roc_auc = float(roc_auc_score(y_holdout, holdout_probabilities))
        passed = all(
            (
                holdout_balanced_accuracy >= self.gate_threshold,
                holdout_f1 >= MIN_F1_GATE,
                holdout_roc_auc >= MIN_ROC_AUC_GATE,
            )
        )

        report: dict[str, Any] = {
            "model_name": "LightGBM",
            "sensor_channel": sensor_channel,
            "feature_version": "v3.0.0_180s",
            "decision_threshold": 0.5,
            "decision_threshold_source": "development_frozen_default",
            "development_participants": sorted(DEVELOPMENT_PARTICIPANTS),
            "holdout_participants": sorted(PARITY_POOL_PARTICIPANTS),
            "nested_balanced_accuracy": nested_balanced_accuracy,
            "holdout_metrics": {
                "balanced_accuracy": holdout_balanced_accuracy,
                "f1": holdout_f1,
                "accuracy": float(accuracy_score(y_holdout, holdout_predictions)),
                "roc_auc": holdout_roc_auc,
            },
            "holdout_evaluated": True,
            "release_status": "candidate_ready" if passed else "blocked_holdout_gate",
            "folds": fold_metrics,
        }

        if passed:
            model_dir = self.paths.models_dir / "stress" / sensor_channel / "classical"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_model, model_dir / "candidate_model.joblib")
            (model_dir / "selected_features.json").write_text(
                json.dumps({"selected_features": feature_columns}, indent=2),
                encoding="utf-8",
            )
            (model_dir / "split_and_metrics.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
        return report
