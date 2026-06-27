import torch.nn as nn

from .dice import DiceLoss


class SegmentationLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, logits, targets):

        return (
            self.bce(logits, targets)
            + self.dice(logits, targets)
        )