import torch
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for the Segmentation Pipeline."""
    
    # Model settings
    MODEL_TYPE: str = "segformer"

    # Training hyperparameters
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    IMAGE_SIZE: int = 256

    # Loss function
    # Options: "dice_loss", "combined_bce_dice_loss", "focal_loss", "combined_focal_dice_loss"
    CRITERION: str = "focal_loss"

    # Hardware / Paths
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR: str = "outputs/experiments"
