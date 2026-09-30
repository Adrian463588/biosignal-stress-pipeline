import torch
from typing import Dict, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class DiagnosticEngine:
    """Monitor training for anomalies and provide diagnostics."""

    def __init__(self, log_interval: int = 10, warn_threshold: float = 0.3):
        self.log_interval = log_interval
        self.warn_threshold = warn_threshold

    def check_predictions(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Check prediction distribution for collapse."""
        preds = torch.argmax(logits, dim=1)
        unique_preds, counts = torch.unique(preds, return_counts=True)

        # Check if model is predicting only one class
        if len(unique_preds) == 1:
            return {
                "status": "COLLAPSED",
                "predicted_class": int(unique_preds[0]),
                "count": int(counts[0]),
                "total": len(preds),
                "message": f"Model predicts only class {int(unique_preds[0])}",
            }

        # Check class distribution
        pred_dist = {
            int(k): int(v) for k, v in zip(unique_preds.tolist(), counts.tolist())
        }
        label_dist = Counter(labels.tolist())

        return {
            "status": "OK",
            "prediction_distribution": pred_dist,
            "label_distribution": dict(label_dist),
            "agreement": float((preds == labels).sum() / len(labels)),
        }

    def check_gradients(self, model: torch.nn.Module) -> Dict[str, float]:
        """Check gradient norms for vanishing/exploding gradients."""
        total_norm = 0.0
        max_norm = 0.0
        min_norm = float("inf")

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
                max_norm = max(max_norm, param_norm)
                min_norm = min(min_norm, param_norm)

        total_norm = total_norm**0.5

        return {
            "total_grad_norm": total_norm,
            "max_grad_norm": max_norm,
            "min_grad_norm": min_norm if min_norm != float("inf") else 0.0,
            "status": "OK" if 1e-6 < total_norm < 1e3 else "SUSPICIOUS",
        }

    def check_fold_validity(self, fold_id: str, balanced_accuracy: float) -> bool:
        """Return True if fold metrics are valid (not collapsed)."""
        if balanced_accuracy < self.warn_threshold:
            logger.warning(
                f"Fold {fold_id} BA={balanced_accuracy:.4f} (< {self.warn_threshold}) — possible label inversion or collapse"
            )
            return False
        if balanced_accuracy > 0.95 and balanced_accuracy < 1.0:
            logger.info(
                f"Fold {fold_id} BA={balanced_accuracy:.4f} (> 0.95) — excellent but check for overfitting"
            )
        return True
