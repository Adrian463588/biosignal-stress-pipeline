import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class LeakageDetector:
    """Detect data leakage between train and validation splits."""

    def check_participant_overlap(
        self, train_ids: List[str], val_ids: List[str]
    ) -> bool:
        """Return True if any participant appears in both sets."""
        train_set = set(train_ids)
        val_set = set(val_ids)
        overlap = train_set.intersection(val_set)
        if overlap:
            logger.critical(f"Participant overlap detected: {overlap}")
            return True
        return False

    def check_augmentation_leakage(
        self, train_aug: np.ndarray, val_clean: np.ndarray
    ) -> float:
        """
        Check if augmented training data is too similar to validation data.
        Returns similarity score (higher = more leakage).
        """
        from scipy.spatial.distance import cdist

        # Sample a subset for performance
        sample_size = min(100, len(train_aug), len(val_clean))
        train_sample = train_aug[:sample_size].reshape(sample_size, -1)
        val_sample = val_clean[:sample_size].reshape(sample_size, -1)

        # Compute minimum distance to any training point for each validation point
        distances = cdist(train_sample, val_sample, metric="euclidean")
        min_distances = np.min(distances, axis=0)
        mean_min_distance = np.mean(min_distances)

        # Normalize by std
        std_train = np.std(train_sample)
        std_val = np.std(val_sample)
        normalized_distance = mean_min_distance / (1e-6 + (std_train + std_val) / 2)

        # If normalized_distance < 1.0, there may be leakage
        if normalized_distance < 1.0:
            logger.warning(
                f"Potential augmentation leakage: normalized_distance={normalized_distance:.3f}"
            )

        return float(normalized_distance)
