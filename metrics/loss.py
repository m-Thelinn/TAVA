import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.metrics import dice_score

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        return 1.0 - dice_score(logits, targets, smooth=self.smooth)


class CombinedBCEDiceLoss(nn.Module):
    """
    Combines BCEWithLogitsLoss and DiceLoss.
    Optimizes for pixel-wise accuracy (BCE) and regional overlap (Dice) simultaneously.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()  # Internal computation handles applies sigmoid
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedFocalDiceLoss(nn.Module):
    """
    Combines Focal Loss and Dice Loss.
    Optimizes for hard-to-classify pixels (Focal) and regional overlap (Dice).
    """
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss