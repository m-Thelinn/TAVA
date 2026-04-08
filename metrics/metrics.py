import torch
import torch.nn as nn

def dice_score(logits, targets, smooth=1.0, apply_sigmoid=True, threshold=0.5):
    """
    Devuelve el coeficiente Dice ∈ [0, 1].
    - apply_sigmoid=True : recibe logits crudos, aplica sigmoid internamente.
    - threshold=None     : Dice suave (para backprop en DiceLoss).
    - threshold=0.5      : Dice binario (para métricas de monitorización reales).
    """
    probs = torch.sigmoid(logits) if apply_sigmoid else logits
    if threshold is not None:
        probs = (probs >= threshold).float()
    probs   = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    return (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

def iou_score(logits, targets, smooth=1.0, apply_sigmoid=True, threshold=0.5):
    """
    Devuelve el índice IoU (Jaccard) ∈ [0, 1].
    - threshold=0.5: binariza las predicciones para métricas honestas.
    """
    probs = torch.sigmoid(logits) if apply_sigmoid else logits
    if threshold is not None:
        probs = (probs >= threshold).float()
    probs   = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)
