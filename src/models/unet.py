import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    """
    U-Net 2D con encoder preentrenado en ImageNet via segmentation_models_pytorch.
    Entrada: [B, 1, H, W]  (mamografía monocanal)
    Salida:  [B, 1, H, W]  (logits crudos — sigmoid se aplica en la loss y en métricas)
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        encoder_name: str = "resnet18",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        freeze_encoder: bool = False,
    ):
        super().__init__()

        assert len(decoder_channels) == encoder_depth, (
            f"decoder_channels debe tener {encoder_depth} elementos "
            f"(encoder_depth={encoder_depth}), tiene {len(decoder_channels)}"
        )

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes,
            decoder_channels=decoder_channels,
            activation=None,
        )

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Devuelve logits — NO aplicar sigmoid aquí.
        # Las loss functions lo aplican internamente.
        # Las métricas (dice_score, iou_score) también tienen apply_sigmoid=True.
        return self.model(x)
