import logging
from typing import Dict

logger = logging.getLogger(__name__)


class FoldValidator:
    """Validate fold results before aggregation."""

    def __init__(self, min_ba_threshold: float = 0.3):
        self.min_ba_threshold = min_ba_threshold

    def validate(self, fold_id: str, metrics: Dict[str, float]) -> bool:
        """
        Validates fold metrics.
        Returns True if valid, False if it should be skipped/flagged.
        """
        ba = metrics.get("balanced_accuracy", 0.0)

        if ba < self.min_ba_threshold:
            logger.warning(
                f"Fold {fold_id} failed validation. BA={ba:.4f} is below threshold {self.min_ba_threshold}."
            )
            return False

        return True
