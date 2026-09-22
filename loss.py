import torch
import torch.nn as nn

class CustomBinaryLoss(nn.Module):
    """
    Penalty Weighted Loss (PWL) with dynamic false-negative / false-positive penalization.
    
    Supports:
    1. Discrete Step Penalty (Paper Formulation):
       penalizes batch-level misclassifications using threshold-based indicator sums.
    2. Continuous Smooth Surrogate:
       uses temperature-scaled sigmoid relaxation to ensure differentiable gradient
       propagation directly to individual false-negative and false-positive samples.
    """
    def __init__(self, alpha=0.87, default_threshold=0.5, temperature=0.1, device='cpu'):
        super(CustomBinaryLoss, self).__init__()
        self.alpha = alpha
        self.default_threshold = default_threshold
        self.temperature = temperature
        self.bce = nn.BCELoss().to(device)

    def forward(self, preds, targets, threshold=None, smooth=False):
        """
        Args:
            preds: Predicted probabilities in [0, 1] of shape (B, 1) or (B,)
            targets: Binary ground truth labels {0, 1} of matching shape
            threshold: Decision boundary threshold (defaults to self.default_threshold)
            smooth: If True, uses differentiable surrogate for backpropagation
        """
        tau = threshold if threshold is not None else self.default_threshold
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1).float()

        # Base Binary Cross-Entropy
        base_loss = self.bce(preds_flat, targets_flat)

        if smooth:
            # Continuous differentiable soft indicator functions
            # High penalty when targets == 1 but preds << tau (False Negative)
            fn_soft = torch.sigmoid((tau - preds_flat) / self.temperature) * targets_flat
            # High penalty when targets == 0 but preds >> tau (False Positive)
            fp_soft = torch.sigmoid((preds_flat - tau) / self.temperature) * (1.0 - targets_flat)

            penalty = self.alpha * fn_soft.sum() + (1.0 - self.alpha) * fp_soft.sum()
        else:
            # Discrete indicator counts (Paper Eq. 9-11)
            false_negatives = ((targets_flat == 1.0) & (preds_flat < tau)).float().sum()
            false_positives = ((targets_flat == 0.0) & (preds_flat >= tau)).float().sum()

            penalty = self.alpha * false_negatives + (1.0 - self.alpha) * false_positives

        # PWL: LPWL = R_penalty * LBCE (Paper Eq. 13)
        return base_loss * (1.0 + penalty)
