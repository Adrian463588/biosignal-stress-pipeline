import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class ClassWeightCalculator:
    """Compute balanced class weights per fold."""

    def __init__(self, device: torch.device):
        self.device = device

    def compute(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute class weights for balanced loss."""
        labels_np = labels.cpu().numpy()
        classes = np.unique(labels_np)

        if len(classes) < 2:
            # Only one class in this fold — fallback to equal weights
            return torch.ones(2, device=self.device)

        weights = compute_class_weight("balanced", classes=classes, y=labels_np)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def apply_to_loss(
        self, criterion: torch.nn.Module, weights: torch.Tensor
    ) -> torch.nn.Module:
        """Apply class weights to loss function."""
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            criterion.pos_weight = (
                weights[1] / weights[0]
                if len(weights) > 1
                else torch.tensor([1.0], device=self.device)
            )
        elif isinstance(criterion, torch.nn.CrossEntropyLoss):
            criterion.weight = weights
        return criterion
