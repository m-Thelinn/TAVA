import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinConfig, SwinModel

class SwinUNet(nn.Module):
    """
    Swin-UNet adaptado para segmentación binaria 256x256.
    Encoder: Swin Transformer (Tiny).
    Decoder: Upsampling bilineal con capas de convolución.
    """
    # Usamos la versión base de Microsoft
    PRETRAINED = "microsoft/swin-tiny-patch4-window7-224"

    def __init__(self, num_classes=1):
        super(SwinUNet, self).__init__()

        # Cargamos el encoder preentrenado
        # ignore_mismatched_sizes=True permite que acepte 256x256 aunque fuera entrenado a 224
        self.encoder = SwinModel.from_pretrained(
            self.PRETRAINED, 
            add_pooling_layer=False,
            ignore_mismatched_sizes=True 
        )

        # Decoder jerárquico simple
        # Swin Tiny tiene 768 canales en su última capa (etapa 4)
        self.up1 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(48, num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:]
        
        # Swin requiere 3 canales (RGB)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 1. Forward Encoder
        # last_hidden_state para 256x256 será (B, 64, 768) -> 64 es 8x8
        outputs = self.encoder(x)
        feat = outputs.last_hidden_state 
        
        # Re-formatear a tensor de imagen: (B, C, H_feat, W_feat)
        grid_h, grid_w = h // 32, w // 32
        feat = feat.transpose(1, 2).view(x.shape[0], 768, grid_h, grid_w)

        # 2. Forward Decoder (Upsampling progresivo)
        x = F.relu(self.up1(feat)) # -> 16x16
        x = F.relu(self.up2(x))    # -> 32x32
        x = F.relu(self.up3(x))    # -> 64x64
        x = F.relu(self.up4(x))    # -> 128x128
        
        # 3. Cabeza final y ajuste de tamaño final a 256
        logits = self.final_conv(x)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        return logits