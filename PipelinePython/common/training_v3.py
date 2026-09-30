"""
Leakage-safe local participant-level stress training with LightGBM and 1D-CNN escalation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

try:  # DL is a Colab-only optional dependency for the local pipeline.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by lean local envs
    if exc.name != "torch":
        raise
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    TensorDataset = None  # type: ignore[assignment,misc]
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Constants
DEV_PARTICIPANTS = [f"ID-{i:02d}" for i in range(1, 29)]
HOLDOUT_PARTICIPANTS = [f"ID-{i:02d}" for i in range(29, 33)]

GATES = {
    "balanced_accuracy": 0.80,
    "f1": 0.78,
    "roc_auc": 0.83,
}
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class TrainingMetrics:
    """Small compatibility contract for the legacy trainer's test boundary."""

    accuracy: float
    balanced_accuracy: float
    f1: float
    roc_auc: float
    threshold: float
    sample_count: int
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def _threshold_for_balanced_accuracy(
    labels: np.ndarray, probability: np.ndarray
) -> float:
    """Choose a threshold on the supplied fold only; never inspect holdout rows."""

    candidates = np.unique(
        np.concatenate(
            ([0.25, 0.5, 0.75], np.quantile(probability, np.linspace(0.05, 0.95, 91)))
        )
    )
    scores = [
        balanced_accuracy_score(labels, (probability >= threshold).astype(int))
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def _inner_oof_probability(
    estimator: Any,
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    """Return deterministic participant-grouped OOF probabilities."""

    probabilities = np.zeros(len(labels), dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    for train_index, validation_index in splitter.split(features, labels, groups):
        model = clone(estimator)
        model.fit(features.iloc[train_index], labels[train_index])
        if hasattr(model, "predict_proba"):
            probabilities[validation_index] = model.predict_proba(
                features.iloc[validation_index]
            )[:, 1]
        else:
            probabilities[validation_index] = model.decision_function(
                features.iloc[validation_index]
            )
    return probabilities


def _metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    prediction: np.ndarray | None = None,
) -> TrainingMetrics:
    """Compute fold-local metrics without changing the locked release gates."""

    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (
        (probability >= threshold).astype(int)
        if prediction is None
        else np.asarray(prediction, dtype=int)
    )
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return TrainingMetrics(
        accuracy=float(accuracy_score(labels, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
        f1=float(f1_score(labels, predicted, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, probability))
        if len(np.unique(labels)) > 1
        else 0.5,
        threshold=float(threshold),
        sample_count=int(labels.size),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


_TorchModuleBase = nn.Module if nn is not None else object


class CNN1DClassifier(_TorchModuleBase):
    """
    1D-CNN DL Escalation Model for Stress Classification from PPG Features.
    """

    def __init__(self, num_features: int):
        if nn is None:
            raise RuntimeError("Deep learning requires the Colab torch environment")
        super().__init__()
        # Input shape: (Batch, 1, Features)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape tabular features to sequence of length features for Conv1D: (B, Features) -> (B, 1, Features)
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def evaluate_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else 0.5,
    }


def passes_gates(metrics: dict) -> bool:
    return all(metrics[k] >= GATES[k] for k in GATES)


def train_lightgbm_cv(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> Tuple[dict, lgb.LGBMClassifier]:
    logger.info("Training LightGBM baseline...")
    gkf = GroupKFold(n_splits=5)

    oof_preds = np.zeros(len(y))
    best_model = None
    best_score = -1

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            random_state=SEED, n_estimators=100, learning_rate=0.05, verbose=-1
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )

        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

        val_score = roc_auc_score(y_val, preds)
        if val_score > best_score:
            best_score = val_score
            best_model = model

    cv_metrics = evaluate_metrics(y.values, oof_preds)
    logger.info(f"LightGBM CV Metrics: {cv_metrics}")
    return cv_metrics, best_model


def train_cnn1d_cv(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> Tuple[dict, CNN1DClassifier, StandardScaler]:
    logger.info("Escalating to 1D-CNN DL model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))

    best_model = None
    best_scaler = None
    best_score = -1

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, y_train = X.iloc[train_idx].values, y.iloc[train_idx].values
        X_val, y_val = X.iloc[val_idx].values, y.iloc[val_idx].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = CNN1DClassifier(num_features=X.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        epochs = 30
        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_x).squeeze(-1)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                out = model(batch_x).squeeze(-1).cpu().numpy()
                # Ensure it's a 1D array
                if out.ndim == 0:
                    out = np.expand_dims(out, axis=0)
                val_preds.extend(out)

        oof_preds[val_idx] = val_preds
        val_score = roc_auc_score(y_val, val_preds)
        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_scaler = scaler

    cv_metrics = evaluate_metrics(y.values, oof_preds)
    logger.info(f"1D-CNN CV Metrics: {cv_metrics}")
    return cv_metrics, best_model, best_scaler


def run_training_pipeline(data_path: Path = None):
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "gold" / "gold_features_v3.parquet"

    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        return

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Isolate features vs metadata
    non_feature_cols = [
        "participant_id",
        "session_id",
        "recording_id",
        "phase",
        "sensor_channel",
        "window_id",
        "label",
        "split",
        "quality_score",
        "quality_tier",
        "quality_flags",
        "feature_version",
        "preprocessing_version",
        "source_file",
        "window_start_utc",
        "window_end_utc",
        "sampling_rate_hz",
    ]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Strictly define development and holdout splits based on requirements
    dev_df = df[df["participant_id"].isin(DEV_PARTICIPANTS)].copy()
    holdout_df = df[df["participant_id"].isin(HOLDOUT_PARTICIPANTS)].copy()

    if dev_df.empty or holdout_df.empty:
        raise ValueError("Data must contain both DEV and HOLDOUT participants.")

    X_dev, y_dev, groups_dev = (
        dev_df[feature_cols],
        dev_df["label"],
        dev_df["participant_id"],
    )
    X_holdout, y_holdout = holdout_df[feature_cols], holdout_df["label"]

    # 1. Baseline LightGBM
    cv_metrics_lgb, model_lgb = train_lightgbm_cv(X_dev, y_dev, groups_dev)

    final_model_type = "lightgbm"
    final_cv_metrics = cv_metrics_lgb

    # 2. DL Escalation if needed
    if not passes_gates(cv_metrics_lgb):
        logger.info("LightGBM did not pass the gates. Escalating to 1D-CNN...")
        cv_metrics_cnn, model_cnn, scaler_cnn = train_cnn1d_cv(X_dev, y_dev, groups_dev)
        final_model_type = "cnn1d"
        final_cv_metrics = cv_metrics_cnn

        if not passes_gates(cv_metrics_cnn):
            logger.warning("1D-CNN also failed to pass all gates on CV.")
    else:
        logger.info("LightGBM passed the gates on CV.")

    logger.info(f"Selected Model: {final_model_type}")

    # 3. Evaluate on Holdout
    logger.info("Evaluating selected model on Holdout set...")
    if final_model_type == "lightgbm":
        holdout_preds = model_lgb.predict_proba(X_holdout)[:, 1]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_holdout_scaled = scaler_cnn.transform(X_holdout.values)
        X_holdout_tensor = torch.tensor(X_holdout_scaled, dtype=torch.float32).to(
            device
        )
        model_cnn.eval()
        with torch.no_grad():
            holdout_preds = model_cnn(X_holdout_tensor).squeeze(-1).cpu().numpy()

    holdout_metrics = evaluate_metrics(y_holdout.values, holdout_preds)
    logger.info(f"Holdout Metrics: {holdout_metrics}")

    if not passes_gates(holdout_metrics):
        logger.warning("Model FAILED to pass gates on Holdout set!")
    else:
        logger.info("Model PASSED all gates on Holdout set!")

    # 4. Export metrics
    reports_dir = PROJECT_ROOT / "reports" / "model_eval"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model_type": final_model_type,
        "gates_required": GATES,
        "cv_metrics": final_cv_metrics,
        "holdout_metrics": holdout_metrics,
        "cv_passed": passes_gates(final_cv_metrics),
        "holdout_passed": passes_gates(holdout_metrics),
    }

    report_path = reports_dir / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Metrics exported to {report_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    # Uncomment to run directly:
    # run_training_pipeline()
