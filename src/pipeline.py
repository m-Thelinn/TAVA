import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Local imports
from preprocessing.dataloader import get_dataloaders
from train import train
from metrics.loss import DiceLoss, CombinedBCEDiceLoss, FocalLoss, CombinedFocalDiceLoss
from models.segformer import SegFormer
from models.deeplabv3plus import DeepLabV3Plus


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
    if cfg.MODEL_TYPE == "segformer":
        print("[Model] Initializing Transformer-based model (SegFormer-B2).")
        model = SegFormer(num_classes=1).to(device)
    elif cfg.MODEL_TYPE == "deeplabv3plus":
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    print("\n--- Setup Learning Rate Scheduler ---")
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=5
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS - 5, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
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
        scheduler=scheduler,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        patience=cfg.EARLY_STOPPING_PATIENCE,
    )
    
    print("\n--- Training Complete ---")
    best_dice = max(history["val_dice"])
    best_iou = max(history["val_iou"])
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best Validation IoU:  {best_iou:.4f}")

if __name__ == "__main__":
    main()
