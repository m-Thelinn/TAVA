import torch
import torch.nn as nn

def dice_score(logits, targets, smooth=1.0, apply_sigmoid=True):
    """
    Devuelve el coeficiente Dice ∈ [0, 1].
    - apply_sigmoid=True  → recibe logits crudos (durante training)
    - apply_sigmoid=False → recibe probabilidades ya calculadas
    """
    probs = torch.sigmoid(logits) if apply_sigmoid else logits
    probs   = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    return (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

def iou_score(logits, targets, smooth=1.0, apply_sigmoid=True):
    """
    Devuelve el índice IoU (Jaccard) ∈ [0, 1].
    """
    probs = torch.sigmoid(logits) if apply_sigmoid else logits
    probs   = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)
