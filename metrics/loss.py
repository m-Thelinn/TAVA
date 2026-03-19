import torch.nn as nn
from metrics.metrics import dice_score

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        return 1.0 - dice_score(logits, targets, smooth=self.smooth)