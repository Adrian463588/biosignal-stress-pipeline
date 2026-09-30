"""Compatibility-safe DL evaluation surface for the active local protocol.

The old all-participant v5 entry point is intentionally not reintroduced. The
module exposes the audited helpers used by regression tests and delegates
implementation to the participant-safe v4 trainer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import training_dl_v4 as _v4
from .training_dl_v4 import (
    _build_model,
    _apply_scaler,
    _apply_tabular_scaler,
    _fit_scaler,
    _fit_tabular_scaler,
    _infer_proba,
    _metrics,
    _participant_group_train_val_indices,
    _slice_X,
    _train_model,
)

_normalize_splits = _v4._normalize_splits
_stable_seed_offset = _v4._stable_seed_offset


class LocalStressDLTrainer(_v4.LocalStressDLTrainer):
    """V5 compatibility surface using this module's patched helper symbols."""

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
    ) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        del sensor_channel
        preprocessing: dict[str, np.ndarray] = {}
        if isinstance(dev_features, tuple):
            mean, scale = _fit_scaler(dev_features[0])
            dev_wave = _apply_scaler(dev_features[0], mean, scale)
            imputer, tab_mean, tab_scale = _fit_tabular_scaler(dev_features[1])
            dev_tab = _apply_tabular_scaler(
                dev_features[1], imputer, tab_mean, tab_scale
            )
            scaled_dev: Any = (dev_wave, dev_tab)
            preprocessing.update(
                tabular_imputer=imputer, tabular_mean=tab_mean, tabular_scale=tab_scale
            )
        else:
            mean, scale = _fit_scaler(dev_features)
            scaled_dev = _apply_scaler(dev_features, mean, scale)
        preprocessing.update(mean=mean, scale=scale)
        train_idx, validation_idx = _participant_group_train_val_indices(
            dev_groups,
            seed=self.config.training.random_seed + 9999,
            validation_fraction=0.15,
        )
        model = _build_model(arch_name, best_params)
        model, _ = _train_model(
            model,
            _slice_X(scaled_dev, train_idx),
            dev_labels[train_idx],
            _slice_X(scaled_dev, validation_idx),
            dev_labels[validation_idx],
            num_epochs=50,
            batch_size=best_params.get("batch_size", 32),
            lr=best_params.get("lr", 1e-3),
            device=self.device,
            patience=7,
        )
        if isinstance(holdout_features, tuple):
            scaled_holdout: Any = (
                _apply_scaler(holdout_features[0], mean, scale),
                _apply_tabular_scaler(
                    holdout_features[1],
                    preprocessing["tabular_imputer"],
                    preprocessing["tabular_mean"],
                    preprocessing["tabular_scale"],
                ),
            )
        else:
            scaled_holdout = _apply_scaler(holdout_features, mean, scale)
        probability = _infer_proba(model, scaled_holdout, device=self.device)
        return model, _metrics(holdout_labels, probability, dev_threshold), mean, scale


def _assert_fusion_lineage(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    channels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    ref_labels, ref_groups, ref_splits = reference
    for channel, values in channels.items():
        labels, groups, splits = values
        if not np.array_equal(ref_labels, labels):
            raise ValueError(f"{channel}: label differs")
        if not np.array_equal(ref_groups, groups):
            raise ValueError(f"{channel}: participant differs")
        if not np.array_equal(ref_splits, splits):
            raise ValueError(f"{channel}: split differs")


def _assert_fusion_frame_lineage(frames: dict[str, pd.DataFrame]) -> None:
    required = (
        "participant_id",
        "session_id",
        "recording_id",
        "phase",
        "label",
        "split",
        "window_id",
        "window_start_utc",
        "window_end_utc",
    )
    reference_channel, reference = next(iter(frames.items()))
    for column in required:
        if column not in reference:
            raise ValueError(f"{reference_channel}: missing lineage column {column}")
    compared_columns = tuple(column for column in required if column != "window_id")
    for channel, frame in frames.items():
        for column in compared_columns:
            if column not in frame:
                raise ValueError(f"{channel}: missing lineage column {column}")
            if (
                not frame[column]
                .reset_index(drop=True)
                .equals(reference[column].reset_index(drop=True))
            ):
                raise ValueError(
                    f"{column} differs between {reference_channel} and {channel}"
                )
