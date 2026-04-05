import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt

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


# ──────────────────────────────────────────────────────────
# Numpy-based helpers (used in test.py on flattened arrays)
# ──────────────────────────────────────────────────────────

def pixel_precision_recall_f1(preds: np.ndarray, targets: np.ndarray):
    """Return (precision, recall, f1) from binary arrays."""
    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


def pixel_specificity(preds: np.ndarray, targets: np.ndarray):
    """True-negative rate — critical for cancer screening."""
    tn = np.sum((preds == 0) & (targets == 0))
    fp = np.sum((preds == 1) & (targets == 0))
    return float(tn / (tn + fp + 1e-8))


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray):
    """
    95th-percentile Hausdorff distance between two 2-D binary masks.
    Returns 0.0 when both masks are empty or both are full.
    Returns np.inf when only one mask is empty.
    """
    pred_border   = pred ^ _erode(pred)
    target_border = target ^ _erode(target)

    if pred_border.sum() == 0 and target_border.sum() == 0:
        return 0.0
    if pred_border.sum() == 0 or target_border.sum() == 0:
        return float("inf")

    dt_pred   = distance_transform_edt(~pred_border)
    dt_target = distance_transform_edt(~target_border)

    d_pred_to_target = dt_target[pred_border > 0]
    d_target_to_pred = dt_pred[target_border > 0]

    return float(np.percentile(
        np.concatenate([d_pred_to_target, d_target_to_pred]), 95
    ))


def _erode(mask: np.ndarray):
    """Simple binary erosion by 1 pixel using distance transform."""
    if mask.sum() == 0:
        return mask
    return distance_transform_edt(mask) > 1

