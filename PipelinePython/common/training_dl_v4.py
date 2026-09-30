"""Leakage-safe GPU-accelerated stress training for DL v4.

Optimized for NVIDIA GeForce RTX 2060 Laptop GPU:
  - Automatic CUDA detection & cuDNN benchmark tuning.
  - Automatic Mixed Precision (AMP FP16) + GradScaler.
  - Asynchronous non-blocking host-to-device transfers (`non_blocking=True`).
  - Memory-efficient zero_grad(set_to_none=True).
  - Pinned memory DataLoaders & optional direct VRAM tensor caching.
  - Architecture ladder: Baseline1DCNN → CNNBiLSTM → TCN.
  - Nested LOSO: 28 outer folds.
 - Locked PRD gates: BA 0.80, F1 0.78, ROC-AUC 0.83.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
from joblib import Parallel, delayed
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .contracts import FoldResult
from .settings import PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & CUDA Global Config
# ---------------------------------------------------------------------------
_METADATA_COLUMNS = {
    "participant_id",
    "session_id",
    "recording_id",
    "phase",
    "sensor_channel",
    "window_id",
    "window_start_utc",
    "window_end_utc",
    "sampling_rate_hz",
    "label",
    "quality_score",
    "quality_tier",
    "quality_flags",
    "feature_version",
    "preprocessing_version",
    "split",
    "source_file",
}

_DEFAULT_EPOCHS = 50  # early stopping cuts most runs to 10-25 epochs
_EARLY_STOP_PATIENCE = 7  # patience on inner-val balanced accuracy
_MIN_DELTA = 1e-4
_MAX_OPTUNA_INNER_SPLITS = 5
_HPO_MAX_EPOCHS = 8  # low-fidelity HPO; selected models still train fully


def _setup_cuda_environment() -> str:
    """Detect CUDA and configure global performance flags for Turing (RTX 2060)."""
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("🚀 CUDA GPU Acceleration ENABLED: %s", gpu_name)
    else:
        device = "cpu"
        logger.info("ℹ️ CUDA unavailable; running on CPU.")
    return device


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    balanced_accuracy: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    sensitivity: float
    specificity: float
    threshold: float
    sample_count: int
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def _threshold_for_balanced_accuracy(
    labels: np.ndarray, probability: np.ndarray
) -> float:
    candidates = np.unique(
        np.concatenate(
            ([0.25, 0.5, 0.75], np.quantile(probability, np.linspace(0.05, 0.95, 91)))
        )
    )
    scores = [
        balanced_accuracy_score(labels, (probability >= t).astype(int))
        for t in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def _stable_seed_offset(identifier: str, modulo: int = 1000) -> int:
    """Return a process-independent seed offset for a lineage identifier."""
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _training_data_sha256(
    features: np.ndarray | tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
) -> str:
    """Fingerprint the exact development tensors used by a DL checkpoint."""

    digest = hashlib.sha256()
    arrays = features if isinstance(features, tuple) else (features,)
    for array in (*arrays, labels, groups.astype("U")):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _bundle_sha256(bundle_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("model.pt", "scaler.json"):
        artifact = bundle_dir / name
        if not artifact.is_file():
            raise FileNotFoundError(f"Incomplete model bundle: {artifact}")
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(artifact.read_bytes()).digest())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _normalize_splits(splits: pd.Series) -> pd.Series:
    """Normalize known split aliases and fail closed on every unknown value."""
    split_remap = {
        "train": "development",
        "val": "development",
        "development": "development",
        "test": "holdout",
        "holdout": "holdout",
        "parity_pool": "holdout",
    }
    normalized = splits.map(split_remap)
    if normalized.isna().any():
        unknown = sorted({str(value) for value in splits[normalized.isna()].tolist()})
        raise ValueError(f"Unknown split values: {unknown}")
    return normalized


def _participant_group_train_val_indices(
    groups: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic early-stopping split with no participant overlap."""
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError(
            "Participant-grouped validation requires at least two participants"
        )
    validation_group_count = min(
        unique_groups.size - 1,
        max(1, int(round(validation_fraction * unique_groups.size))),
    )
    rng = np.random.default_rng(seed)
    validation_groups = rng.choice(
        unique_groups, size=validation_group_count, replace=False
    )
    validation_mask = np.isin(groups, validation_groups)
    train_idx = np.flatnonzero(~validation_mask)
    validation_idx = np.flatnonzero(validation_mask)
    if set(groups[train_idx]) & set(groups[validation_idx]):
        raise RuntimeError("Participant overlap in early-stopping split")
    return train_idx, validation_idx


def _metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    prediction: np.ndarray | None = None,
) -> EvaluationMetrics:
    if prediction is None:
        prediction = (probability >= threshold).astype(int)
    else:
        prediction = np.asarray(prediction, dtype=int)
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return EvaluationMetrics(
        accuracy=float(accuracy_score(labels, prediction)),
        balanced_accuracy=float(balanced_accuracy_score(labels, prediction)),
        f1=float(f1_score(labels, prediction, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, probability))
        if np.unique(labels).size == 2
        else 0.5,
        pr_auc=float(average_precision_score(labels, probability)),
        brier_score=float(brier_score_loss(labels, probability)),
        sensitivity=sensitivity,
        specificity=specificity,
        threshold=threshold,
        sample_count=int(labels.size),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


# ---------------------------------------------------------------------------
# Dataset & Memory Pinning
# ---------------------------------------------------------------------------


class WaveformDataset(Dataset):
    """Waveform dataset; expects waveforms shape [N, seq_len] or [N, 1, seq_len]."""

    def __init__(
        self, waveforms: np.ndarray, labels: np.ndarray, tabular: np.ndarray = None
    ) -> None:
        t = torch.tensor(waveforms, dtype=torch.float32)
        if t.ndim == 2:
            t = t.unsqueeze(1)
        elif t.ndim == 3 and t.shape[-1] == 1:
            t = t.permute(0, 2, 1)
        self.waveforms = t
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tabular = (
            torch.tensor(tabular, dtype=torch.float32) if tabular is not None else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.tabular is not None:
            return (self.waveforms[idx], self.tabular[idx]), self.labels[idx]
        return self.waveforms[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Architectures (PRD §9a ladder)
# ---------------------------------------------------------------------------


class Baseline1DCNN(nn.Module):
    """Step 1: 3-block 1D-CNN with global average pooling."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        drop_rate: float = 0.4,
        n_filters_base: int = 32,
    ) -> None:
        super().__init__()
        f1, f2, f3 = n_filters_base, n_filters_base * 2, n_filters_base * 4
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, f1, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(f1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(f1, f2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(f2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(f2, f3, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(f3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(f3, f3 // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(f3 // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNBiLSTM(nn.Module):
    """Step 2: CNN feature extractor followed by a BiLSTM temporal head."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        drop_rate: float = 0.4,
        n_filters_base: int = 32,
        lstm_hidden: int = 64,
    ) -> None:
        super().__init__()
        f1, f2 = n_filters_base, n_filters_base * 2
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, f1, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(f1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(f1, f2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(f2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.bilstm = nn.LSTM(
            input_size=f2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x).permute(0, 2, 1)
        out, _ = self.bilstm(feat)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.head(out)


class _TCNBlock(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int, dilation: int, drop_rate: float
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=pad
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=pad
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(drop_rate)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)[..., : x.shape[-1]]))
        out = self.drop(out)
        out = self.relu(self.bn2(self.conv2(out)[..., : x.shape[-1]]))
        out = self.drop(out)
        return self.relu(out + x)


class TCN(nn.Module):
    """Step 3: Temporal Convolutional Network with dilated causal convolutions."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        drop_rate: float = 0.4,
        n_filters_base: int = 32,
        n_levels: int = 5,
        kernel_size: int = 8,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, n_filters_base, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                _TCNBlock(
                    n_filters_base, kernel_size, dilation=2**i, drop_rate=drop_rate
                )
                for i in range(n_levels)
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(n_filters_base, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.input_proj(x)))


class Hybrid1DCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        drop_rate: float = 0.4,
        n_filters_base: int = 32,
        tab_dim: int = 45,
    ):
        super().__init__()
        self.cnn = Baseline1DCNN(input_channels, num_classes, drop_rate, n_filters_base)
        self.cnn.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )  # Pool and flatten
        f3 = n_filters_base * 4
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(f3 + tab_dim, f3 // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(f3 // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        merged = torch.cat([feat, tab], dim=1)
        return self.classifier(merged)


class HybridCNNBiLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        drop_rate: float = 0.4,
        n_filters_base: int = 32,
        lstm_hidden: int = 64,
        tab_dim: int = 45,
    ):
        super().__init__()
        self.cnn_bilstm = CNNBiLSTM(
            input_channels, num_classes, drop_rate, n_filters_base, lstm_hidden
        )
        self.cnn_bilstm.head = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(lstm_hidden * 2 + tab_dim, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        feat = self.cnn_bilstm(x)
        merged = torch.cat([feat, tab], dim=1)
        return self.classifier(merged)


class HybridTCN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        drop_rate: float = 0.4,
        n_filters_base: int = 64,
        n_levels: int = 4,
        kernel_size: int = 8,
        tab_dim: int = 45,
    ):
        super().__init__()
        self.tcn = TCN(
            input_channels,
            num_classes,
            drop_rate,
            n_filters_base,
            n_levels,
            kernel_size,
        )
        self.tcn.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(n_filters_base + tab_dim, n_filters_base // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(n_filters_base // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        feat = self.tcn(x)
        merged = torch.cat([feat, tab], dim=1)
        return self.classifier(merged)


# ---------------------------------------------------------------------------
# Tabular Features (Late Fusion)
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "n_peaks",
    "n_ibi",
    "mean_hr_bpm",
    "std_hr_bpm",
    "min_hr_bpm",
    "max_hr_bpm",
    "median_hr_bpm",
    "mean_ibi_ms",
    "std_ibi_ms",
    "min_ibi_ms",
    "max_ibi_ms",
    "median_ibi_ms",
    "iqr_ibi_ms",
    "range_ibi_ms",
    "mad_ibi_ms",
    "cv_ibi",
    "rmssd_ms",
    "sdsd_ms",
    "pnn20",
    "pnn50",
    "signal_mean",
    "signal_std",
    "signal_median",
    "signal_q25",
    "signal_q75",
    "signal_iqr",
    "signal_min",
    "signal_max",
    "signal_range",
    "signal_rms",
    "signal_mad",
    "signal_skew",
    "signal_kurtosis",
    "signal_entropy",
    "gradient_mean_abs",
    "gradient_std",
    "gradient_max_abs",
    "peak_amplitude_mean",
    "peak_amplitude_std",
    "peak_amplitude_median",
    "lag1_autocorrelation",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "total_power",
]


def _fit_tabular_scaler(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit median imputer and standard scaler for tabular features."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(features)
    scaler.fit(imputed)
    return imputer.statistics_, scaler.mean_, scaler.scale_


def _apply_tabular_scaler(
    features: np.ndarray, imp_stat: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply median imputer and standard scaler to tabular features."""
    # Manual imputation and scaling
    imputed = features.copy()
    for col in range(imputed.shape[1]):
        mask = np.isnan(imputed[:, col]) | np.isinf(imputed[:, col])
        imputed[mask, col] = imp_stat[col]
    scaled = (imputed - mean) / scale
    return scaled


# ---------------------------------------------------------------------------
# Per-fold normalizer
# ---------------------------------------------------------------------------


def _fit_scaler(waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = waveforms.reshape(waveforms.shape[0], -1)
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def _apply_scaler(
    waveforms: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    flat = waveforms.reshape(waveforms.shape[0], -1)
    scaled = (flat - mean) / scale
    return scaled.reshape(waveforms.shape)


# ---------------------------------------------------------------------------
# GPU-Resident Training: cache entire dataset in VRAM to eliminate PCIe stalls
# ---------------------------------------------------------------------------


def _to_device_tensor(arr: np.ndarray, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Convert numpy array to a pinned tensor and move to device non-blocking."""
    t = torch.as_tensor(arr, dtype=dtype)
    if device == "cuda":
        t = t.pin_memory()
    return t.to(device, non_blocking=True)


def _slice_X(X, idx):
    if isinstance(X, tuple):
        return (X[0][idx], X[1][idx])
    return X[idx]


def _train_model(
    model: nn.Module,
    X_train: tuple[np.ndarray, np.ndarray] | np.ndarray,
    y_train: np.ndarray,
    X_val: tuple[np.ndarray, np.ndarray] | np.ndarray,
    y_val: np.ndarray,
    *,
    num_epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = _EARLY_STOP_PATIENCE,
) -> tuple[nn.Module, float]:
    """Train model with GPU-resident batching, AMP FP16, and early stopping.

    Strategy: pre-load the entire dataset into VRAM as a single tensor.
    With only ~10 MB of waveform data, this eliminates all PCIe transfer
    overhead and DataLoader CPU stalls, saturating RTX 2060 Tensor Cores.
    Falls back to standard DataLoader on CPU.
    """
    use_cuda = device == "cuda"
    n_train = len(X_train[0]) if isinstance(X_train, tuple) else len(X_train)
    effective_bs = min(batch_size if not use_cuda else max(batch_size, 256), n_train)

    # Move model first — this initialises the CUDA context before any
    # pin_memory() calls, preventing an ACCESS_VIOLATION on Windows WDDM.
    model.to(device)
    if use_cuda:
        torch.cuda.synchronize()  # ensure context is fully warm

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    if use_cuda:
        # Pre-cache entire dataset in VRAM — zero PCIe overhead per epoch
        if isinstance(X_train, tuple):
            X_wave_tr, X_tab_tr = X_train
            X_wave_va, X_tab_va = X_val
            X_train_t = (
                _to_device_tensor(
                    X_wave_tr[:, np.newaxis, :] if X_wave_tr.ndim == 2 else X_wave_tr,
                    torch.float32,
                    device,
                ),
                _to_device_tensor(X_tab_tr, torch.float32, device),
            )
            X_val_t = (
                _to_device_tensor(
                    X_wave_va[:, np.newaxis, :] if X_wave_va.ndim == 2 else X_wave_va,
                    torch.float32,
                    device,
                ),
                _to_device_tensor(X_tab_va, torch.float32, device),
            )
        else:
            X_train_t = _to_device_tensor(
                X_train[:, np.newaxis, :] if X_train.ndim == 2 else X_train,
                torch.float32,
                device,
            )
            X_val_t = _to_device_tensor(
                X_val[:, np.newaxis, :] if X_val.ndim == 2 else X_val,
                torch.float32,
                device,
            )

        y_train_t = _to_device_tensor(y_train, torch.float32, device)
        y_val_np = y_val

        def _iter_batches_gpu(
            X: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
            y: torch.Tensor,
            bs: int,
            shuffle: bool,
        ):
            idx = (
                torch.randperm(len(y), device=device)
                if shuffle
                else torch.arange(len(y), device=device)
            )
            for start in range(0, len(y), bs):
                sl = idx[start : start + bs]
                if isinstance(X, tuple):
                    yield (X[0][sl], X[1][sl]), y[sl]
                else:
                    yield X[sl], y[sl]

        best_ba, best_state, no_improve = 0.0, {}, 0

        for _ in range(num_epochs):
            model.train()
            for bx, by in _iter_batches_gpu(
                X_train_t, y_train_t, effective_bs, shuffle=True
            ):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    if isinstance(bx, tuple):
                        loss = criterion(model(bx[0], bx[1]).squeeze(-1), by)
                    else:
                        loss = criterion(model(bx).squeeze(-1), by)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            # Validation: single GPU forward pass — no DataLoader overhead
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                if isinstance(X_val_t, tuple):
                    val_prob = (
                        torch.sigmoid(model(X_val_t[0], X_val_t[1]).squeeze(-1))
                        .cpu()
                        .numpy()
                    )
                else:
                    val_prob = torch.sigmoid(model(X_val_t).squeeze(-1)).cpu().numpy()

            val_ba = float(
                balanced_accuracy_score(y_val_np, (val_prob >= 0.5).astype(int))
            )
            scheduler.step(val_ba)

            if val_ba > best_ba + _MIN_DELTA:
                best_ba = val_ba
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    else:
        # CPU fallback: standard DataLoader path
        if isinstance(X_train, tuple):
            train_ds = WaveformDataset(X_train[0], y_train, tabular=X_train[1])
            val_ds = WaveformDataset(X_val[0], y_val, tabular=X_val[1])
        else:
            train_ds = WaveformDataset(X_train, y_train)
            val_ds = WaveformDataset(X_val, y_val)
        # Windows nested HPO runs inside loky workers.  Spawning additional
        # DataLoader workers from those processes is not reliable on Python
        # 3.12 (the child Process has no inherited env).  Single-process
        # loading is deterministic and avoids that failure; non-Windows keeps
        # the existing worker parallelism.
        loader_workers = 0 if os.name == "nt" else 4
        train_loader = DataLoader(
            train_ds,
            batch_size=effective_bs,
            shuffle=True,
            drop_last=False,
            num_workers=loader_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=512,
            shuffle=False,
            num_workers=loader_workers,
        )

        best_ba, best_state, no_improve = 0.0, {}, 0

        for _ in range(num_epochs):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad(set_to_none=True)
                if isinstance(bx, (tuple, list)):
                    loss = criterion(model(bx[0], bx[1]).squeeze(-1), by)
                else:
                    loss = criterion(model(bx).squeeze(-1), by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            probs: list[float] = []
            with torch.no_grad():
                for bx, _ in val_loader:
                    if isinstance(bx, (tuple, list)):
                        logits = model(bx[0], bx[1]).squeeze(-1)
                    else:
                        logits = model(bx).squeeze(-1)
                    probs.extend(torch.sigmoid(logits).numpy().tolist())
            val_prob = np.array(probs)
            val_ba = float(
                balanced_accuracy_score(y_val, (val_prob >= 0.5).astype(int))
            )
            scheduler.step(val_ba)

            if val_ba > best_ba + _MIN_DELTA:
                best_ba = val_ba
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_ba


def _infer_proba(
    model: nn.Module,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Return class-1 probabilities for waveform array X."""
    use_cuda = device == "cuda"
    model.eval()
    model.to(device)
    if isinstance(X, tuple):
        ds = WaveformDataset(X[0], np.zeros(len(X[0])), tabular=X[1])
    else:
        ds = WaveformDataset(X, np.zeros(len(X)))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0 if os.name == "nt" else 4,
    )
    probs: list[float] = []
    with torch.no_grad():
        for bx, _ in loader:
            if isinstance(bx, (tuple, list)):
                bx = (
                    bx[0].to(device, non_blocking=True),
                    bx[1].to(device, non_blocking=True),
                )
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits = model(bx[0], bx[1]).squeeze(-1)
            else:
                bx = bx.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits = model(bx).squeeze(-1)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(probs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Search
# ---------------------------------------------------------------------------

_ARCH_REGISTRY: dict[str, type[nn.Module]] = {
    "1D_CNN": Baseline1DCNN,
    "CNN_BiLSTM": CNNBiLSTM,
    "TCN": TCN,
    "Hybrid_1D_CNN": Hybrid1DCNN,
    "Hybrid_CNN_BiLSTM": HybridCNNBiLSTM,
    "Hybrid_TCN": HybridTCN,
}


def _suggest_parameters_dl(trial: Any) -> dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "drop_rate": trial.suggest_categorical("drop_rate", [0.3, 0.4, 0.5]),
        "n_filters_base": trial.suggest_categorical("n_filters_base", [16, 32, 64]),
    }


def _build_model(arch_name: str, params: dict[str, Any]) -> nn.Module:
    cls = _ARCH_REGISTRY[arch_name]
    init_kwargs: dict[str, Any] = {}
    if "drop_rate" in params:
        init_kwargs["drop_rate"] = params["drop_rate"]
    if "n_filters_base" in params:
        init_kwargs["n_filters_base"] = params["n_filters_base"]
    return cls(**init_kwargs)


def _tune_within_outer_train_dl(
    *,
    arch_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    splits: int,
    trials: int,
    seed: int,
    device: str = "cpu",
) -> tuple[dict[str, Any], float]:
    """Return (best_params, best_inner_ba) for given architecture."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    effective_splits = min(
        splits, _MAX_OPTUNA_INNER_SPLITS, len(np.unique(groups_train))
    )
    splitter = GroupKFold(n_splits=effective_splits)
    dummy_X = np.zeros(len(y_train))
    fixed_splits = list(splitter.split(dummy_X, y_train, groups_train))

    def objective(trial: Any) -> float:
        params = _suggest_parameters_dl(trial)
        inner_scores: list[float] = []
        for tr_idx, va_idx in fixed_splits:
            if isinstance(X_train, tuple):
                Xtr_w, ytr = X_train[0][tr_idx], y_train[tr_idx]
                Xva_w, yva = X_train[0][va_idx], y_train[va_idx]
                Xtr_t, Xva_t = X_train[1][tr_idx], X_train[1][va_idx]
                mean_w, scale_w = _fit_scaler(Xtr_w)
                Xtr_w_s = _apply_scaler(Xtr_w, mean_w, scale_w)
                Xva_w_s = _apply_scaler(Xva_w, mean_w, scale_w)
                imp_t, mean_t, scale_t = _fit_tabular_scaler(Xtr_t)
                Xtr_t_s = _apply_tabular_scaler(Xtr_t, imp_t, mean_t, scale_t)
                Xva_t_s = _apply_tabular_scaler(Xva_t, imp_t, mean_t, scale_t)
                Xtr_s = (Xtr_w_s, Xtr_t_s)
                Xva_s = (Xva_w_s, Xva_t_s)
            else:
                Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
                Xva, yva = X_train[va_idx], y_train[va_idx]
                mean_i, scale_i = _fit_scaler(Xtr)
                Xtr_s = _apply_scaler(Xtr, mean_i, scale_i)
                Xva_s = _apply_scaler(Xva, mean_i, scale_i)
            model = _build_model(arch_name, params)
            _, best_val_ba = _train_model(
                model,
                Xtr_s,
                ytr,
                Xva_s,
                yva,
                num_epochs=_HPO_MAX_EPOCHS,
                batch_size=params["batch_size"],
                lr=params["lr"],
                device=device,
                patience=5,
            )
            inner_scores.append(best_val_ba)
        return float(np.mean(inner_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _numpy_decimate(signal: np.ndarray, factor: int) -> np.ndarray:
    """Decimate signal using scipy FIR; import is deferred so scipy loads before CUDA.

    IMPORTANT: Call this function BEFORE initialising the CUDA context on Windows
    (WDDM driver conflict between BLAS/pthreads inside scipy FIR and CUDA's
    pinned-memory manager causes an ACCESS_VIOLATION if called after cuda.init()).
    """
    from scipy.signal import decimate as _decimate

    return _decimate(signal, factor, ftype="fir", zero_phase=True).astype(np.float32)


def _load_waveforms(
    gold_dir: Path,
    sensor_channel: str,
    decimate_to_hz: int = 25,
    participant_ids: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw waveform tensors and corresponding labels/splits from gold dir.

    NOTE: Must be called BEFORE CUDA context is initialised on Windows to avoid
    a BLAS/pthreads vs CUDA pinned-memory ACCESS_VIOLATION (WDDM driver bug).
    The LocalStressDLTrainer.run() ensures this ordering.
    """

    gold_path = gold_dir / "gold_features_v3.parquet"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold features not found: {gold_path}")

    gold_df = pd.read_parquet(gold_path)
    gold_df = gold_df[gold_df["sensor_channel"] == sensor_channel].copy()
    allowed_participants = (
        {str(value) for value in participant_ids}
        if participant_ids is not None
        else None
    )
    if allowed_participants is not None:
        gold_df = gold_df[
            gold_df["participant_id"].astype(str).isin(allowed_participants)
        ].copy()

    if gold_df.empty:
        raise ValueError(f"No gold features for sensor: {sensor_channel}")

    gold_df["split_norm"] = _normalize_splits(gold_df["split"])

    label_map = dict(zip(gold_df["window_id"], gold_df["label"]))
    split_map = dict(zip(gold_df["window_id"], gold_df["split_norm"]))

    native_hz_series = gold_df["sampling_rate_hz"]
    native_hz = (
        int(round(native_hz_series.median())) if not native_hz_series.empty else 100
    )
    decimate_factor = max(1, native_hz // decimate_to_hz)
    if decimate_factor > 1:
        logger.info(
            "%s: decimating %d Hz → %d Hz (factor %d)",
            sensor_channel,
            native_hz,
            native_hz // decimate_factor,
            decimate_factor,
        )

    waveform_dir = gold_dir / "waveforms"
    waveforms_list: list[np.ndarray] = []
    labels_list: list[int] = []
    groups_list: list[str] = []
    window_ids_list: list[str] = []

    for npz_file in sorted(waveform_dir.rglob(f"{sensor_channel}.npz")):
        try:
            participant_id = npz_file.parent.parent.name
        except IndexError:
            logger.warning("Unexpected waveform path structure: %s", npz_file)
            continue
        if (
            allowed_participants is not None
            and participant_id not in allowed_participants
        ):
            continue

        data = np.load(npz_file, allow_pickle=True)
        if "waveforms" not in data or "window_ids" not in data:
            logger.warning("Missing keys in %s; skipping.", npz_file)
            continue

        w = data["waveforms"]
        if w.ndim == 3:
            w = w[:, :, 0]
        w_ids = data["window_ids"]

        if decimate_factor > 1:
            w = np.stack(
                [_numpy_decimate(w[i], decimate_factor) for i in range(len(w))]
            )

        for i, w_id in enumerate(w_ids):
            w_id_str = str(w_id)
            if w_id_str not in label_map:
                continue
            waveforms_list.append(w[i])
            labels_list.append(int(label_map[w_id_str]))
            groups_list.append(participant_id)
            window_ids_list.append(w_id_str)

    if not waveforms_list:
        raise ValueError(
            f"No matching waveform windows found for sensor '{sensor_channel}'."
        )

    all_wids = window_ids_list
    missing_split_ids = sorted({w for w in all_wids if w not in split_map})
    if missing_split_ids:
        raise ValueError(
            f"Waveform windows missing split lineage: {missing_split_ids[:5]}"
        )
    splits_arr = np.array([split_map[w] for w in all_wids])

    # Keep the hybrid branch aligned to the exact Gold row represented by each
    # waveform window.  The feature contract intentionally mirrors the
    # classical trainer: numeric, non-label PPG features only.  Never infer a
    # tabular row by position because the NPZ files are independently ordered.
    metadata_columns = {
        "participant_id",
        "participant_code",
        "session_id",
        "recording_id",
        "phase",
        "sensor_channel",
        "window_id",
        "window_start_utc",
        "window_end_utc",
        "label",
        "split",
        "quality_score",
        "quality_tier",
        "quality_flags",
        "feature_version",
        "preprocessing_version",
        "source_file",
        "exclusion_reason",
    }
    tabular_columns = tuple(
        sorted(
            column
            for column in gold_df.columns
            if column not in metadata_columns
            and pd.api.types.is_numeric_dtype(gold_df[column])
            and gold_df[column].notna().any()
            and not column.lower().startswith("ecg_")
        )
    )
    if not tabular_columns:
        raise ValueError(
            f"No numeric tabular features available for sensor '{sensor_channel}'"
        )
    gold_by_window = gold_df.drop_duplicates("window_id").set_index("window_id")
    missing_feature_ids = sorted(set(all_wids) - set(gold_by_window.index))
    if missing_feature_ids:
        raise ValueError(
            f"Waveform windows missing Gold feature lineage: {missing_feature_ids[:5]}"
        )
    tabular = gold_by_window.loc[all_wids, list(tabular_columns)].to_numpy(
        dtype=np.float32
    )

    return (
        np.asarray(waveforms_list, dtype=np.float32),
        tabular,
        np.asarray(labels_list, dtype=np.int64),
        np.asarray(groups_list, dtype=str),
        np.asarray(window_ids_list, dtype=str),
        splits_arr,
    )


# ---------------------------------------------------------------------------


class LocalStressDLTrainer:
    """Nested LOSO deep-learning trainer for PPG stress classification."""

    _ARCH_LADDER = [
        "1D_CNN",
        "CNN_BiLSTM",
        "TCN",
        "Hybrid_1D_CNN",
        "Hybrid_CNN_BiLSTM",
        "Hybrid_TCN",
    ]

    def __init__(self, project_root: Path, config: PipelineConfig) -> None:
        self.root = project_root.resolve()
        self.config = config
        self.gold_dir = self.root / "data" / "gold"
        self.models_dir = self.root / "data" / "artifacts" / "models" / "stress"
        self.reports_dir = self.root / "reports" / "model_eval" / "stress_v4.0.0-dl"
        # NOTE: device is set in run() AFTER data is loaded.
        # On Windows WDDM, CUDA must not be initialised before scipy FIR calls.
        self.device: str = "cpu"

    def _evaluate_architecture(
        self,
        *,
        arch_name: str,
        features: np.ndarray | tuple[np.ndarray, np.ndarray],
        labels: np.ndarray,
        groups: np.ndarray,
        fold_ids: list[str],
        best_arch_params: dict[str, Any],
        sensor_channel: str,
        smoke: bool,
        resume: bool = False,
        parallel_outer: bool = True,
    ) -> tuple[EvaluationMetrics, list[FoldResult], np.ndarray, np.ndarray]:
        checkpoint_dir = self.reports_dir / "checkpoints" / sensor_channel / arch_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_contract = {
            "schema_version": "2.0.0",
            "training_data_sha256": _training_data_sha256(features, labels, groups),
            "model_version": self.config.model_version,
            "feature_version": self.config.feature_version,
            "sensor_channel": sensor_channel,
            "architecture": arch_name,
            "development_participants": list(
                self.config.training.development_participant_ids
            ),
            "parity_pool_participants": list(
                self.config.training.parity_pool_participant_ids
            ),
            "inner_group_splits": self.config.training.inner_group_splits,
            "optuna_trials": self.config.training.optuna_trials,
            "evaluation_mode": "smoke" if smoke else "full",
            "trial_count": 5 if smoke else self.config.training.optuna_trials,
            "hpo_max_epochs": _HPO_MAX_EPOCHS,
            "full_training_epochs": _DEFAULT_EPOCHS,
            "random_seed": self.config.training.random_seed,
        }
        checkpoint_fingerprint = hashlib.sha256(
            json.dumps(checkpoint_contract, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if parallel_outer and not smoke and len(fold_ids) > 1:
            shards = [fold_ids[::2], fold_ids[1::2]]
            Parallel(n_jobs=2, backend="loky")(
                delayed(self._evaluate_architecture)(
                    arch_name=arch_name,
                    features=features,
                    labels=labels,
                    groups=groups,
                    fold_ids=shard,
                    best_arch_params=best_arch_params,
                    sensor_channel=sensor_channel,
                    smoke=smoke,
                    resume=resume,
                    parallel_outer=False,
                )
                for shard in shards
                if shard
            )
            # Re-read the atomically published checkpoints in canonical fold
            # order so aggregation is identical to the serial implementation.
            return self._evaluate_architecture(
                arch_name=arch_name,
                features=features,
                labels=labels,
                groups=groups,
                fold_ids=fold_ids,
                best_arch_params=best_arch_params,
                sensor_channel=sensor_channel,
                smoke=smoke,
                resume=True,
                parallel_outer=False,
            )

        if not parallel_outer and self.device == "cuda":
            _setup_cuda_environment()

        probabilities: list[np.ndarray] = []
        labels_all: list[np.ndarray] = []
        fold_results: list[FoldResult] = []

        for held_out in fold_ids:
            checkpoint = checkpoint_dir / f"{held_out}.json"

            if resume and checkpoint.exists():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                if payload.get("checkpoint_fingerprint") == checkpoint_fingerprint:
                    logger.info(
                        "Fold %s already completed for %s, skipping",
                        held_out,
                        arch_name,
                    )
                    fold_result = FoldResult(**payload["fold_result"])
                    probability = np.array(payload["probability"])
                    y_te = np.array(payload["labels"])
                    probabilities.append(probability)
                    labels_all.append(y_te)
                    fold_results.append(fold_result)
                    continue
                logger.warning(
                    "Ignoring stale DL checkpoint without matching contract: %s",
                    checkpoint,
                )

            test_mask = groups == held_out
            train_mask = ~test_mask
            if isinstance(features, tuple):
                X_tr = (features[0][train_mask], features[1][train_mask])
                X_te = (features[0][test_mask], features[1][test_mask])
            else:
                X_tr = features[train_mask]
                X_te = features[test_mask]
            y_tr, g_tr = labels[train_mask], groups[train_mask]
            y_te = labels[test_mask]

            n_trials = 5 if smoke else self.config.training.optuna_trials
            best_params, _ = _tune_within_outer_train_dl(
                arch_name=arch_name,
                X_train=X_tr,
                y_train=y_tr,
                groups_train=g_tr,
                splits=self.config.training.inner_group_splits,
                trials=n_trials,
                seed=self.config.training.random_seed + _stable_seed_offset(held_out),
                device=self.device,
            )
            merged_params = {**best_arch_params, **best_params}

            if isinstance(X_tr, tuple):
                mean_w, scale_w = _fit_scaler(X_tr[0])
                X_tr_w_s = _apply_scaler(X_tr[0], mean_w, scale_w)
                X_te_w_s = _apply_scaler(X_te[0], mean_w, scale_w)
                imp_t, mean_t, scale_t = _fit_tabular_scaler(X_tr[1])
                X_tr_t_s = _apply_tabular_scaler(X_tr[1], imp_t, mean_t, scale_t)
                X_te_t_s = _apply_tabular_scaler(X_te[1], imp_t, mean_t, scale_t)
                X_tr_s = (X_tr_w_s, X_tr_t_s)
                X_te_s = (X_te_w_s, X_te_t_s)
                mean_f, scale_f = mean_w, scale_w
                imp_t_f, mean_t_f, scale_t_f = imp_t, mean_t, scale_t
            else:
                mean_f, scale_f = _fit_scaler(X_tr)
                X_tr_s = _apply_scaler(X_tr, mean_f, scale_f)
                X_te_s = _apply_scaler(X_te, mean_f, scale_f)

            model = _build_model(arch_name, merged_params)
            tr_idx, val_idx = _participant_group_train_val_indices(
                g_tr,
                seed=self.config.training.random_seed + _stable_seed_offset(held_out),
                validation_fraction=0.2,
            )

            model, _ = _train_model(
                model,
                _slice_X(X_tr_s, tr_idx),
                y_tr[tr_idx],
                _slice_X(X_tr_s, val_idx),
                y_tr[val_idx],
                num_epochs=_DEFAULT_EPOCHS,
                batch_size=merged_params.get("batch_size", 32),
                lr=merged_params.get("lr", 1e-3),
                device=self.device,
                patience=_EARLY_STOP_PATIENCE,
            )

            probability = _infer_proba(model, X_te_s, device=self.device)
            threshold = self.config.training.decision_threshold
            fold_m = _metrics(y_te, probability, threshold)

            fold_result = FoldResult(
                participant_id=held_out,
                model_name=arch_name,
                sensor_channel=sensor_channel,
                balanced_accuracy=fold_m.balanced_accuracy,
                f1=fold_m.f1,
                roc_auc=fold_m.roc_auc,
                threshold=threshold,
                sample_count=int(len(y_te)),
            )
            payload = {
                "checkpoint_contract": checkpoint_contract,
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "fold_result": asdict(fold_result),
                "labels": y_te.tolist(),
                "probability": probability.tolist(),
                "best_params": merged_params,
                "scaler_mean": mean_f.tolist(),
                "scaler_scale": scale_f.tolist(),
            }
            if isinstance(X_tr, tuple):
                payload["tabular_imputer"] = imp_t_f.tolist()
                payload["tabular_mean"] = mean_t_f.tolist()
                payload["tabular_scale"] = scale_t_f.tolist()
            temporary = checkpoint.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(checkpoint)
            probabilities.append(probability)
            labels_all.append(y_te)
            fold_results.append(fold_result)
            logger.info(
                "[%s/%s] fold=%s BA=%.4f",
                arch_name,
                sensor_channel,
                held_out,
                fold_m.balanced_accuracy,
            )

        prob_all = np.concatenate(probabilities)
        lab_all = np.concatenate(labels_all)
        threshold = _threshold_for_balanced_accuracy(lab_all, prob_all)
        agg = _metrics(lab_all, prob_all, threshold)
        return agg, fold_results, lab_all, prob_all

    def _train_frozen_candidate(
        self,
        *,
        arch_name: str,
        dev_features: np.ndarray,
        dev_labels: np.ndarray,
        dev_groups: np.ndarray,
        best_params: dict[str, Any],
    ) -> tuple[nn.Module, dict[str, np.ndarray]]:
        """Train the frozen development-only candidate and its preprocessing."""
        preprocessing: dict[str, np.ndarray] = {}
        if isinstance(dev_features, tuple):
            mean_f, scale_f = _fit_scaler(dev_features[0])
            X_dev_w_s = _apply_scaler(dev_features[0], mean_f, scale_f)
            imp_t, mean_t, scale_t = _fit_tabular_scaler(dev_features[1])
            X_dev_t_s = _apply_tabular_scaler(dev_features[1], imp_t, mean_t, scale_t)
            X_dev_s = (X_dev_w_s, X_dev_t_s)
            preprocessing.update(
                tabular_imputer=imp_t,
                tabular_mean=mean_t,
                tabular_scale=scale_t,
            )
        else:
            mean_f, scale_f = _fit_scaler(dev_features)
            X_dev_s = _apply_scaler(dev_features, mean_f, scale_f)
        preprocessing.update(mean=mean_f, scale=scale_f)

        tr_idx, val_idx = _participant_group_train_val_indices(
            dev_groups,
            seed=self.config.training.random_seed + 9999,
            validation_fraction=0.15,
        )

        model = _build_model(arch_name, best_params)
        model, _ = _train_model(
            model,
            _slice_X(X_dev_s, tr_idx),
            dev_labels[tr_idx],
            _slice_X(X_dev_s, val_idx),
            dev_labels[val_idx],
            num_epochs=_DEFAULT_EPOCHS,
            batch_size=best_params.get("batch_size", 32),
            lr=best_params.get("lr", 1e-3),
            device=self.device,
            patience=_EARLY_STOP_PATIENCE,
        )

        return model, preprocessing

    def _evaluate_frozen_candidate(
        self,
        *,
        model: nn.Module,
        holdout_features: np.ndarray | tuple[np.ndarray, np.ndarray],
        holdout_labels: np.ndarray,
        preprocessing: dict[str, np.ndarray],
        dev_threshold: float,
    ) -> EvaluationMetrics:
        """Apply frozen preprocessing/model/threshold without fitting anything."""
        if isinstance(holdout_features, tuple):
            X_ho_s = (
                _apply_scaler(
                    holdout_features[0], preprocessing["mean"], preprocessing["scale"]
                ),
                _apply_tabular_scaler(
                    holdout_features[1],
                    preprocessing["tabular_imputer"],
                    preprocessing["tabular_mean"],
                    preprocessing["tabular_scale"],
                ),
            )
        else:
            X_ho_s = _apply_scaler(
                holdout_features, preprocessing["mean"], preprocessing["scale"]
            )
        probability = _infer_proba(model, X_ho_s, device=self.device)
        return _metrics(holdout_labels, probability, dev_threshold)

    def _train_final_and_eval_holdout(
        self,
        *,
        arch_name: str,
        dev_features: np.ndarray | tuple[np.ndarray, np.ndarray],
        dev_labels: np.ndarray,
        dev_groups: np.ndarray,
        holdout_features: np.ndarray | tuple[np.ndarray, np.ndarray],
        holdout_labels: np.ndarray,
        best_params: dict[str, Any],
        sensor_channel: str,
        dev_threshold: float,
    ) -> tuple[nn.Module, EvaluationMetrics, np.ndarray, np.ndarray]:
        """Compatibility helper; production orchestration freezes before calling parity."""
        del sensor_channel
        model, preprocessing = self._train_frozen_candidate(
            arch_name=arch_name,
            dev_features=dev_features,
            dev_labels=dev_labels,
            dev_groups=dev_groups,
            best_params=best_params,
        )
        metrics = self._evaluate_frozen_candidate(
            model=model,
            holdout_features=holdout_features,
            holdout_labels=holdout_labels,
            preprocessing=preprocessing,
            dev_threshold=dev_threshold,
        )
        return model, metrics, preprocessing["mean"], preprocessing["scale"]

    def _write_model_bundle(
        self,
        *,
        bundle_dir: Path,
        model: nn.Module,
        preprocessing: dict[str, np.ndarray],
        arch_name: str,
        arch_params: dict[str, Any],
    ) -> str:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), bundle_dir / "model.pt")
        scaler_payload: dict[str, Any] = {
            "mean": preprocessing["mean"].tolist(),
            "scale": preprocessing["scale"].tolist(),
            "arch_name": arch_name,
            "arch_params": arch_params,
        }
        for name in ("tabular_imputer", "tabular_mean", "tabular_scale"):
            if name in preprocessing:
                scaler_payload[name] = preprocessing[name].tolist()
        _atomic_json(bundle_dir / "scaler.json", scaler_payload)
        return _bundle_sha256(bundle_dir)

    def _claim_parity_access(
        self,
        *,
        sensor_channel: str,
        candidate_checksum: str,
        threshold: float,
    ) -> tuple[Path, dict[str, Any]]:
        """Persist a checksum-keyed access claim before parity tensors are loaded."""
        ledger_path = self.reports_dir / "parity_access_ledger.json"
        ledger = (
            json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger_path.exists()
            else {"schema_version": "1.0.0", "entries": []}
        )
        entries = ledger.get("entries")
        if not isinstance(entries, list):
            raise RuntimeError("Parity access ledger is malformed")
        if any(
            entry.get("candidate_bundle_sha256") == candidate_checksum
            for entry in entries
        ):
            raise RuntimeError(
                "Parity pool access already claimed for this frozen candidate checksum"
            )
        receipt_path = (
            self.reports_dir
            / "parity_receipts"
            / sensor_channel
            / f"{candidate_checksum}.json"
        )
        claim = {
            "sensor_channel": sensor_channel,
            "candidate_bundle_sha256": candidate_checksum,
            "candidate_frozen_before_cohort": True,
            "threshold_frozen_before_cohort": True,
            "decision_threshold": float(threshold),
            "threshold_source": "participant_grouped_development_validation",
            "cohort_ids": list(self.config.training.parity_pool_participant_ids),
            "cohort_access_count": 1,
            "claimed_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "receipt_path": str(receipt_path.relative_to(self.root)).replace("\\", "/"),
            "status": "claimed",
        }
        entries.append(claim)
        _atomic_json(ledger_path, ledger)
        return receipt_path, claim

    def _complete_parity_receipt(
        self,
        *,
        receipt_path: Path,
        claim: dict[str, Any],
        metrics: EvaluationMetrics,
    ) -> dict[str, Any]:
        receipt = {
            **claim,
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "metrics": asdict(metrics),
        }
        _atomic_json(receipt_path, receipt)
        ledger_path = self.reports_dir / "parity_access_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        for entry in ledger["entries"]:
            if entry.get("candidate_bundle_sha256") == claim["candidate_bundle_sha256"]:
                entry.update(
                    status="completed",
                    completed_at_utc=receipt["completed_at_utc"],
                )
                break
        _atomic_json(ledger_path, ledger)
        return receipt

    def train_sensor(
        self,
        sensor_channel: str,
        *,
        smoke: bool = False,
        resume: bool = False,
    ) -> dict[str, Any]:
        logger.info("=== DL Training: %s | smoke=%s ===", sensor_channel, smoke)

        development_ids = tuple(self.config.training.development_participant_ids)
        waveforms, tabular, labels, groups, w_ids, splits = _load_waveforms(
            self.gold_dir,
            sensor_channel,
            participant_ids=development_ids,
        )
        if set(np.unique(groups)) != set(development_ids):
            raise ValueError(
                "Development tensors do not contain the configured participants"
            )
        if set(np.unique(splits)) != {"development"}:
            raise ValueError("Development-only load unexpectedly included parity data")
        dev_features = (waveforms, tabular)
        dev_waveforms = waveforms
        dev_labels, dev_groups = labels, groups

        # Initialise CUDA AFTER _load_waveforms so scipy FIR completes first.
        # On Windows WDDM: scipy BLAS/pthreads + CUDA pinned memory = ACCESS_VIOLATION.
        if self.device == "cpu":
            self.device = _setup_cuda_environment()
        logger.info("Device: %s", self.device)

        fold_ids = sorted(np.unique(dev_groups).tolist())
        if smoke:
            fold_ids = fold_ids[:3]

        gate = self.config.training.min_balanced_accuracy
        f1_gate = self.config.training.min_f1
        auc_gate = self.config.training.min_roc_auc
        winning_arch: str | None = None
        winning_metrics: EvaluationMetrics | None = None
        winning_fold_results: list[FoldResult] = []
        winning_params: dict[str, Any] = {}

        ladder = self._ARCH_LADDER if not smoke else ["1D_CNN"]

        for arch_name in ladder:
            logger.info("[Ladder] Trying architecture: %s", arch_name)
            architecture_features = (
                dev_features if arch_name.startswith("Hybrid_") else dev_waveforms
            )
            agg, fold_results, _, _ = self._evaluate_architecture(
                arch_name=arch_name,
                features=architecture_features,
                labels=dev_labels,
                groups=dev_groups,
                fold_ids=fold_ids,
                best_arch_params={},
                sensor_channel=sensor_channel,
                smoke=smoke,
                resume=resume,
            )
            logger.info(
                "[Ladder] %s nested BA=%.4f (gate=%.2f)",
                arch_name,
                agg.balanced_accuracy,
                gate,
            )
            if (
                winning_metrics is None
                or agg.balanced_accuracy > winning_metrics.balanced_accuracy
            ):
                winning_metrics = agg
                winning_arch = arch_name
                winning_fold_results = fold_results

            if not smoke and all(
                (
                    agg.balanced_accuracy >= gate,
                    agg.f1 >= f1_gate,
                    agg.roc_auc >= auc_gate,
                )
            ):
                logger.info("[Ladder] Gate cleared by %s — stopping ladder.", arch_name)
                break

        assert winning_arch is not None
        assert winning_metrics is not None

        gate_passed_nested = (not smoke) and all(
            (
                winning_metrics.balanced_accuracy >= gate,
                winning_metrics.f1 >= f1_gate,
                winning_metrics.roc_auc >= auc_gate,
            )
        )

        checkpoint_dir = (
            self.reports_dir / "checkpoints" / sensor_channel / winning_arch
        )
        fold_checkpoints = sorted(
            checkpoint_dir.glob("ID-*.json"),
            key=lambda p: json.loads(p.read_text())["fold_result"]["balanced_accuracy"],
        )
        if fold_checkpoints:
            median_checkpoint = fold_checkpoints[len(fold_checkpoints) // 2]
            winning_params = json.loads(median_checkpoint.read_text()).get(
                "best_params", {}
            )

        report: dict[str, Any] = {
            "mode": "smoke" if smoke else "full",
            "sensor_channel": sensor_channel,
            "selected_arch": winning_arch,
            "selected_nested_metrics": asdict(winning_metrics),
            "selected_fold_results": [asdict(r) for r in winning_fold_results],
            "nested_gate_passed": gate_passed_nested,
            "selected_hyperparameters": winning_params,
            "release_status": "smoke_only" if smoke else "blocked_nested_gate",
            "holdout_evaluated": False,
            "development_participants": list(
                self.config.training.development_participant_ids
            ),
            "holdout_participants": list(
                self.config.training.parity_pool_participant_ids
            ),
            "threshold_provenance": {
                "source": "participant_grouped_development_validation"
            },
        }

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        suffix = "smoke" if smoke else "full"
        report_path = self.reports_dir / f"{sensor_channel}_{suffix}.json"

        if gate_passed_nested:
            logger.info("[Parity] Freezing the development-only candidate …")
            selected_dev_features = (
                dev_features if winning_arch.startswith("Hybrid_") else dev_waveforms
            )
            candidate_model, candidate_preprocessing = self._train_frozen_candidate(
                arch_name=winning_arch,
                dev_features=selected_dev_features,
                dev_labels=dev_labels,
                dev_groups=dev_groups,
                best_params=winning_params,
            )
            candidate_dir = (
                self.models_dir
                / sensor_channel
                / "deep_learning"
                / self.config.model_version
                / "frozen_candidate"
            )
            candidate_checksum = self._write_model_bundle(
                bundle_dir=candidate_dir,
                model=candidate_model,
                preprocessing=candidate_preprocessing,
                arch_name=winning_arch,
                arch_params=winning_params,
            )
            report["candidate_bundle"] = str(
                candidate_dir.relative_to(self.root)
            ).replace("\\", "/")
            report["frozen_decision_threshold"] = winning_metrics.threshold

            receipt_path, claim = self._claim_parity_access(
                sensor_channel=sensor_channel,
                candidate_checksum=candidate_checksum,
                threshold=winning_metrics.threshold,
            )
            # The isolated parity tensors are not loaded until the candidate,
            # preprocessing, hyperparameters, and threshold are immutable.
            ho_waveforms, ho_tabular, ho_labels, ho_groups, _, ho_splits = (
                _load_waveforms(
                    self.gold_dir,
                    sensor_channel,
                    participant_ids=self.config.training.parity_pool_participant_ids,
                )
            )
            if set(np.unique(ho_groups)) != set(
                self.config.training.parity_pool_participant_ids
            ):
                raise ValueError(
                    "Parity tensors do not contain the configured participants"
                )
            if set(np.unique(ho_splits)) != {"holdout"}:
                raise ValueError(
                    "Parity-only load unexpectedly included development data"
                )
            selected_holdout_features: np.ndarray | tuple[np.ndarray, np.ndarray]
            selected_holdout_features = (
                (ho_waveforms, ho_tabular)
                if winning_arch.startswith("Hybrid_")
                else ho_waveforms
            )
            ho_metrics = self._evaluate_frozen_candidate(
                model=candidate_model,
                holdout_features=selected_holdout_features,
                holdout_labels=ho_labels,
                preprocessing=candidate_preprocessing,
                dev_threshold=winning_metrics.threshold,
            )
            receipt = self._complete_parity_receipt(
                receipt_path=receipt_path,
                claim=claim,
                metrics=ho_metrics,
            )
            holdout_gate_passed = all(
                (
                    ho_metrics.balanced_accuracy >= gate,
                    ho_metrics.f1 >= f1_gate,
                    ho_metrics.roc_auc >= auc_gate,
                )
            )
            report["holdout_evaluated"] = True
            report["holdout_metrics"] = asdict(ho_metrics)
            report["holdout_gate_passed"] = holdout_gate_passed
            report["evaluation_provenance"] = {
                "development_participants": list(development_ids),
                "validation_cohort_ids": receipt["cohort_ids"],
                "cohort_access_count": receipt["cohort_access_count"],
                "candidate_frozen_before_cohort": receipt[
                    "candidate_frozen_before_cohort"
                ],
                "threshold_frozen_before_cohort": receipt[
                    "threshold_frozen_before_cohort"
                ],
                "candidate_bundle_sha256": receipt["candidate_bundle_sha256"],
                "parity_receipt": str(receipt_path.relative_to(self.root)).replace(
                    "\\", "/"
                ),
            }

            if holdout_gate_passed:
                all_waveforms, all_tabular, all_labels, all_groups, _, _ = (
                    _load_waveforms(
                        self.gold_dir,
                        sensor_channel,
                        participant_ids=(
                            *self.config.training.development_participant_ids,
                            *self.config.training.parity_pool_participant_ids,
                        ),
                    )
                )
                production_features = (
                    (all_waveforms, all_tabular)
                    if winning_arch.startswith("Hybrid_")
                    else all_waveforms
                )
                if isinstance(production_features, tuple):
                    all_mean, all_scale = _fit_scaler(production_features[0])
                    X_all_w_s = _apply_scaler(
                        production_features[0], all_mean, all_scale
                    )
                    imp_all, all_mean_t, all_scale_t = _fit_tabular_scaler(
                        production_features[1]
                    )
                    X_all_t_s = _apply_tabular_scaler(
                        production_features[1], imp_all, all_mean_t, all_scale_t
                    )
                    X_all_s = (X_all_w_s, X_all_t_s)
                    production_preprocessing = {
                        "mean": all_mean,
                        "scale": all_scale,
                        "tabular_imputer": imp_all,
                        "tabular_mean": all_mean_t,
                        "tabular_scale": all_scale_t,
                    }
                else:
                    all_mean, all_scale = _fit_scaler(production_features)
                    X_all_s = _apply_scaler(production_features, all_mean, all_scale)
                    production_preprocessing = {"mean": all_mean, "scale": all_scale}
                tr_idx2, val_idx2 = _participant_group_train_val_indices(
                    all_groups,
                    seed=self.config.training.random_seed + 42,
                    validation_fraction=0.1,
                )

                prod_model = _build_model(winning_arch, winning_params)
                prod_model, _ = _train_model(
                    prod_model,
                    _slice_X(X_all_s, tr_idx2),
                    all_labels[tr_idx2],
                    _slice_X(X_all_s, val_idx2),
                    all_labels[val_idx2],
                    num_epochs=_DEFAULT_EPOCHS,
                    batch_size=winning_params.get("batch_size", 32),
                    lr=winning_params.get("lr", 1e-3),
                    device=self.device,
                    patience=_EARLY_STOP_PATIENCE,
                )
                production_dir = (
                    self.models_dir
                    / sensor_channel
                    / "deep_learning"
                    / self.config.model_version
                    / "production_refit"
                )
                production_checksum = self._write_model_bundle(
                    bundle_dir=production_dir,
                    model=prod_model,
                    preprocessing=production_preprocessing,
                    arch_name=winning_arch,
                    arch_params=winning_params,
                )
                report["prod_scaler_mean"] = all_mean.tolist()
                report["prod_scaler_scale"] = all_scale.tolist()
                report["production_bundle"] = str(
                    production_dir.relative_to(self.root)
                ).replace("\\", "/")
                report["production_bundle_sha256"] = production_checksum
                report["production_refit_participants"] = sorted(
                    np.unique(all_groups).tolist()
                )
                report["refit_provenance"] = {
                    "performed_after_parity_pass": True,
                    "participant_ids": sorted(np.unique(all_groups).tolist()),
                    "frozen_candidate_bundle_sha256": candidate_checksum,
                    "production_bundle_sha256": production_checksum,
                }
                report["release_status"] = "candidate_ready"
                logger.info(
                    "✅ %s | holdout BA=%.4f ≥ %.2f — candidate_ready",
                    sensor_channel,
                    ho_metrics.balanced_accuracy,
                    gate,
                )
            else:
                report["release_status"] = "blocked_holdout_gate"
                logger.warning(
                    "❌ %s | holdout BA=%.4f < %.2f — blocked_holdout_gate",
                    sensor_channel,
                    ho_metrics.balanced_accuracy,
                    gate,
                )

        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("Report written: %s", report_path)
        return report

    def run(
        self,
        *,
        sensors: Iterable[str] = ("ppg_ir", "ppg_red"),
        smoke: bool = False,
        resume: bool = False,
    ) -> dict[str, Any]:
        logger.info("PyTorch version: %s | Device: %s", torch.__version__, self.device)
        return {
            sensor: self.train_sensor(sensor, smoke=smoke, resume=resume)
            for sensor in sensors
        }
