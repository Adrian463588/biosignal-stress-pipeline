"""Canonical train-fold-only waveform augmentation pipeline."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..contracts import Augmentation
from .transforms import (
    AmplitudeScaling,
    GaussianNoiseInjection,
)

_DEVELOPMENT_PARTICIPANTS = {f"ID-{index:02d}" for index in range(1, 29)}
_TRAIN_ROLE = "train"
_ALLOWED_TRANSFORMS = (AmplitudeScaling, GaussianNoiseInjection)


class AugmentationPipeline:
    """Apply conservative transforms to an already-split training fold."""

    def __init__(
        self,
        transforms: Sequence[Augmentation] | None = None,
        prob: float = 0.5,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.prob = prob
        if transforms is None:
            self.transforms: list[Augmentation] = [
                AmplitudeScaling(rng=self.rng),
                GaussianNoiseInjection(rng=self.rng),
            ]
        else:
            self.transforms = list(transforms)
            self._validate_transforms(self.transforms)
            for transform in self.transforms:
                transform.rng = self.rng

    def _validate_transforms(self, transforms: Sequence[Augmentation]) -> None:
        if not transforms:
            raise ValueError("At least one canonical transform is required")
        for transform in transforms:
            if not isinstance(transform, _ALLOWED_TRANSFORMS):
                raise ValueError(
                    "Forbidden transform detected in AugmentationPipeline: "
                    f"{type(transform).__name__}"
                )

    def augment(
        self,
        signal: np.ndarray,
        label: int = 0,
        *,
        return_provenance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
        """Apply a conservative random subset and optionally return provenance."""
        record: dict[str, object] = {
            "seed": self.seed,
            "label": int(label),
            "applied": False,
            "transforms": [],
        }
        if self.rng.uniform() > self.prob:
            copied = np.copy(signal)
            return (copied, record) if return_provenance else copied

        out = np.copy(signal)
        selected = [
            transform for transform in self.transforms if self.rng.uniform() < 0.5
        ]
        if not selected:
            selected = [
                self.transforms[int(self.rng.integers(0, len(self.transforms)))]
            ]

        transform_records: list[dict[str, object]] = []
        for transform in selected:
            out, parameters = transform.apply_with_metadata(out, label)
            transform_records.append(
                {"name": type(transform).__name__, "parameters": parameters}
            )
        record["applied"] = True
        record["transforms"] = transform_records
        return (out, record) if return_provenance else out

    def augment_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        multiplier: int = 1,
        *,
        dataset_role: str = _TRAIN_ROLE,
        participant_ids: Sequence[str] | None = None,
        return_provenance: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, list[dict[str, object]]]
    ):
        """Augment one training fold after participant-level splitting.

        Args:
            X: Array of shape [N, samples] or [N, channels, samples].
            y: Array of shape [N].
            multiplier: Number of derived copies generated per source window.
            dataset_role: Must be ``train``; validation and holdout fail closed.
            participant_ids: Optional source participant ID for each window.
            return_provenance: Include source, seed, and transform parameters.

        Returns:
            Original training rows followed by derived rows and optional provenance.
        """
        if dataset_role.casefold() != _TRAIN_ROLE:
            raise ValueError("Augmentation is allowed only for dataset_role='train'")
        if (
            not isinstance(multiplier, int)
            or isinstance(multiplier, bool)
            or multiplier < 0
        ):
            raise ValueError("multiplier must be a non-negative integer")

        X_values = np.asarray(X)
        y_values = np.asarray(y)
        if X_values.ndim not in {2, 3}:
            raise ValueError("X must have shape [N, samples] or [N, channels, samples]")
        if y_values.ndim != 1 or len(X_values) != len(y_values):
            raise ValueError("X and y must contain the same number of rows")
        if not np.isfinite(X_values).all():
            raise ValueError("Training waveforms must contain only finite values")
        if not np.issubdtype(X_values.dtype, np.floating):
            raise ValueError("Training waveforms must use a floating-point dtype")

        source_participants = self._validate_participants(
            participant_ids, len(X_values)
        )
        X_out = [np.copy(X_values)]
        y_out = [np.copy(y_values)]
        provenance = [
            {
                "source_index": index,
                "source_participant_id": source_participants[index],
                "copy_index": 0,
                "dataset_role": _TRAIN_ROLE,
                "seed": self.seed,
                "applied": False,
                "transforms": [],
            }
            for index in range(len(X_values))
        ]

        for copy_index in range(1, multiplier + 1):
            X_aug_list = []
            for index, sample in enumerate(X_values):
                if sample.ndim == 1:
                    aug_sample, record = self.augment(
                        sample, y_values[index], return_provenance=True
                    )
                else:
                    channel_results = [
                        self.augment(channel, y_values[index], return_provenance=True)
                        for channel in sample
                    ]
                    aug_channels = [result[0] for result in channel_results]
                    aug_sample = np.stack(aug_channels, axis=0)
                    record = {
                        "seed": self.seed,
                        "label": int(y_values[index]),
                        "applied": any(
                            result[1]["applied"] for result in channel_results
                        ),
                        "channels": [result[1] for result in channel_results],
                    }
                X_aug_list.append(aug_sample)
                provenance.append(
                    {
                        "source_index": index,
                        "source_participant_id": source_participants[index],
                        "copy_index": copy_index,
                        "dataset_role": _TRAIN_ROLE,
                        **record,
                    }
                )
            X_out.append(np.array(X_aug_list, dtype=X_values.dtype))
            y_out.append(np.copy(y_values))

        result = np.concatenate(X_out, axis=0), np.concatenate(y_out, axis=0)
        return (*result, provenance) if return_provenance else result

    @staticmethod
    def _validate_participants(
        participant_ids: Sequence[str] | None, row_count: int
    ) -> list[str | None]:
        if participant_ids is None:
            return [None] * row_count
        participants = list(participant_ids)
        if len(participants) != row_count:
            raise ValueError("participant_ids must contain one ID per training row")
        forbidden = sorted(set(participants) - _DEVELOPMENT_PARTICIPANTS)
        if forbidden:
            raise ValueError(
                "Augmentation participant_ids must be development participants ID-01..ID-28: "
                + ", ".join(forbidden)
            )
        return participants
