import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

# Local imports
from preprocessing.dataloader import get_dataloaders
from train import train
from metrics.loss import DiceLoss, CombinedBCEDiceLoss, FocalLoss, CombinedFocalDiceLoss
from models.deeplabv3plus import DeepLabV3Plus
from metrics.metrics import dice_score, iou_score


# Import configuration dataclass
from config import Config

def main():
    cfg = Config()
    
    print(f"--- Starting Segmentation Training Pipeline ---")
    print(f"Configuration: Model: {cfg.MODEL_TYPE}, Batch Size: {cfg.BATCH_SIZE}, "
          f"Epochs: {cfg.NUM_EPOCHS}, LR: {cfg.LEARNING_RATE}")

    device = torch.device(cfg.DEVICE)

    print("\n--- Setup Dataloaders ---")
    dataloaders = get_dataloaders(
        batch_size=cfg.BATCH_SIZE,
        image_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    )

    print("\n--- Setup Architecture ---")
    if cfg.MODEL_TYPE == "deeplabv3plus":
        print("[Model] Initializing CNN-based model (DeepLabV3+).")
        model = DeepLabV3Plus(num_classes=1).to(device)
    elif cfg.MODEL_TYPE == "unet":
        print("[Model] Initializing CNN-based model (UNet).")
        from models.unet import UNet
        model = UNet(n_channels=1, n_classes=1).to(device)
    else:
        # Placeholder for other models (Transformer, etc.)
        raise ValueError(f"Model {cfg.MODEL_TYPE} not yet configured in pipeline.py.")

    print("\n--- Setup Optimization ---")
    if cfg.CRITERION == "dice_loss":
        criterion = DiceLoss().to(device)
    elif cfg.CRITERION == "combined_bce_dice_loss":
        criterion = CombinedBCEDiceLoss().to(device)
    elif cfg.CRITERION == "focal_loss":
        criterion = FocalLoss().to(device)
    elif cfg.CRITERION == "combined_focal_dice_loss":
        criterion = CombinedFocalDiceLoss().to(device)
    else:
        raise ValueError(f"Criterion {cfg.CRITERION} not yet configured in pipeline.py.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    print("\n--- Setup Learning Rate Scheduler ---")
    # Reduce LR by half if validation dice doesn't improve for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    print("\n--- Begin Training ---")
    history = train(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=cfg.NUM_EPOCHS,
        checkpoint_every=True,
        save_dir=os.path.join(cfg.SAVE_DIR, f"{cfg.MODEL_TYPE}_{cfg.CRITERION}"),
        scheduler=scheduler
    )
    
    print("\n--- Training Complete ---")
    best_dice = max(history["val_dice"])
    best_iou = max(history["val_iou"])
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best Validation IoU:  {best_iou:.4f}")

if __name__ == "__main__":
    main()
