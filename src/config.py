import torch
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for the Segmentation Pipeline."""
    
    # Model settings
    # Options: "deeplabv3plus", "unet"
    MODEL_TYPE: str = "unet"

    # Training hyperparameters
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 5e-5  # Estándar para fine-tuning con backbone preentrenado
    WEIGHT_DECAY: float = 1e-4
    IMAGE_SIZE: int = 256

    # Loss function
    # Options: "dice_loss", "combined_bce_dice_loss", "focal_loss", "combined_focal_dice_loss"
    CRITERION: str = "combined_focal_dice_loss"

    # Hardware / Paths
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR: str = "outputs/experiments"
