import torch
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for the Segmentation Pipeline."""

    # ── Valid options ──────────────────────────────────────
    VALID_MODELS: tuple = ("segformer", "deeplabv3plus")
    VALID_CRITERIA: tuple = (
        "dice_loss",
        "combined_bce_dice_loss",
        "focal_loss",
        "combined_focal_dice_loss",
        "tversky_loss",
        "combined_focal_tversky_loss",
    )

    # Model settings
    # Options: "deeplabv3plus", "unet"
    MODEL_TYPE: str = "unet"

    # Training hyperparameters
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    IMAGE_SIZE: int = 256

    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 20

    # Gradient clipping
    MAX_GRAD_NORM: float = 1.0

    # Loss function
    # Options: "dice_loss", "combined_bce_dice_loss", "focal_loss", "combined_focal_dice_loss"
    CRITERION: str = "combined_focal_dice_loss"

    # Hardware / Paths
    DEVICE: str = (
        "cuda" if torch.cuda.is_available()
        else "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available()
        else "cpu"
    )
    SAVE_DIR: str = "outputs/experiments"

    def __post_init__(self):
        if self.MODEL_TYPE not in self.VALID_MODELS:
            raise ValueError(
                f"MODEL_TYPE '{self.MODEL_TYPE}' is not supported. "
                f"Choose from: {self.VALID_MODELS}"
            )
        if self.CRITERION not in self.VALID_CRITERIA:
            raise ValueError(
                f"CRITERION '{self.CRITERION}' is not supported. "
                f"Choose from: {self.VALID_CRITERIA}"
            )
        if self.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be a positive integer.")
        if self.NUM_EPOCHS <= 0:
            raise ValueError("NUM_EPOCHS must be a positive integer.")
        if self.LEARNING_RATE <= 0:
            raise ValueError("LEARNING_RATE must be positive.")
        if self.IMAGE_SIZE <= 0:
            raise ValueError("IMAGE_SIZE must be a positive integer.")
