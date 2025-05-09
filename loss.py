class CustomBinaryLoss(nn.Module):
    def __init__(self, false_negative_weight=1.0, device='cpu'):
        super(CustomBinaryLoss, self).__init__()
        self.false_negative_weight = false_negative_weight
        self.bce = nn.BCELoss().to(device)

    def forward(self, preds, targets):
        loss = self.bce(preds, targets)

        false_negatives = (targets == 1) & (preds < threshold)
        false_positives = (targets == 0) & (preds > threshold)
        penalty = false_negatives.float().sum() * 0.83 + false_positives.float().sum() * 1.17

        return loss * (1 + penalty)
