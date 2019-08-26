import torch
import torch.nn as nn

# From: https://www.kaggle.com/soulmachine/siim-deeplabv3


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        N = targets.size(0)
        preds = torch.sigmoid(logits)

        EPSILON = 1

        preds_flat = preds.view(N, -1)
        targets_flat = targets.view(N, -1)

        intersection = (preds_flat * targets_flat).sum()
        union = (preds_flat + targets_flat).sum()

        loss = (2.0 * intersection + EPSILON) / (union + EPSILON)
        loss = 1 - loss / N

        return loss
