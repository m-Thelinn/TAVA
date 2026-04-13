import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormer(nn.Module):
    """
    SegFormer-B2 for binary breast lesion segmentation.

    Uses a Mix Transformer (MiT-B2) encoder with a lightweight all-MLP decoder,
    pretrained on ImageNet + ADE20k via HuggingFace Transformers.

    Input:  (B, 1, H, W)  — grayscale mammography image, normalized [0, 1]
    Output: (B, 1, H, W)  — raw logits for binary segmentation
    """

    PRETRAINED = "nvidia/segformer-b2-finetuned-ade-512-512"

    def __init__(self, num_classes=1):
        super(SegFormer, self).__init__()

        # Load pretrained B2 backbone + ADE20k head, replace head for binary segmentation
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.PRETRAINED,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # replaces the 150-class ADE20k head
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # SegFormer expects 3-channel input — repeat grayscale to simulate RGB
        x_rgb = x.repeat(1, 3, 1, 1)

        outputs = self.model(pixel_values=x_rgb)
        logits = outputs.logits  # (B, num_classes, H/4, W/4)

        # Upsample to original input resolution
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        return logits
