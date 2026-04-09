import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Local imports
from preprocessing.dataloader import get_dataloaders
from train import train
from metrics.loss import (
    DiceLoss,
    CombinedBCEDiceLoss,
    FocalLoss,
    CombinedFocalDiceLoss,
    TverskyLoss,
    CombinedFocalTverskyLoss,
)
from models.segformer import SegFormer
from models.deeplabv3plus import DeepLabV3Plus


# Import configuration dataclass
from config import Config

def _merge_histories(h1: dict, h2: dict) -> dict:
    """Concatenate two history dicts returned by train()."""
    merged = {}
    for key in h1:
        merged[key] = h1[key] + h2[key]
    return merged


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

    print("\n--- Setup Optimization ---")
    if cfg.CRITERION == "dice_loss":
        criterion = DiceLoss().to(device)
    elif cfg.CRITERION == "combined_bce_dice_loss":
        criterion = CombinedBCEDiceLoss().to(device)
    elif cfg.CRITERION == "focal_loss":
        criterion = FocalLoss().to(device)
    elif cfg.CRITERION == "combined_focal_dice_loss":
        # Mejora 1: updated weights and alpha
        criterion = CombinedFocalDiceLoss(
            focal_weight=0.4, dice_weight=0.6, alpha=0.75, gamma=2.0
        ).to(device)
    elif cfg.CRITERION == "tversky_loss":
        # Mejora 5: Tversky loss support
        criterion = TverskyLoss().to(device)
    elif cfg.CRITERION == "combined_focal_tversky_loss":
        # Mejora 5: Combined Focal + Tversky loss support
        criterion = CombinedFocalTverskyLoss().to(device)

    # Mejora 3: differential LR for DeepLabV3+ backbone vs decoder/ASPP
    is_deeplab = isinstance(model, DeepLabV3Plus)
    if is_deeplab:
        print("[Optimizer] Using differential LR: backbone=LR*0.1, decoder+ASPP=LR")
        backbone_params = list(model.low_level_features.parameters()) + \
                          list(model.high_level_features.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        decoder_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.LEARNING_RATE * 0.1},
                {"params": decoder_params,  "lr": cfg.LEARNING_RATE},
            ],
            weight_decay=cfg.WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
        )

    print("\n--- Setup Learning Rate Scheduler ---")
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=5
    )
    # Mejora 2: T_max dynamic (already was cfg.NUM_EPOCHS - 5), eta_min reduced to 1e-7
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS - 5, eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
    )

    save_dir = os.path.join(cfg.SAVE_DIR, f"{cfg.MODEL_TYPE}_{cfg.CRITERION}")

    print("\n--- Begin Training ---")

    # Mejora 4: freeze backbone for warmup epochs (DeepLabV3+ only)
    WARMUP_FREEZE_EPOCHS = 10
    if is_deeplab and cfg.NUM_EPOCHS > WARMUP_FREEZE_EPOCHS:
        print(f"[Freeze] Freezing backbone for first {WARMUP_FREEZE_EPOCHS} warmup epochs.")
        for p in model.low_level_features.parameters():
            p.requires_grad = False
        for p in model.high_level_features.parameters():
            p.requires_grad = False

        history_warmup = train(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=WARMUP_FREEZE_EPOCHS,
            checkpoint_every=True,
            save_dir=save_dir,
            scheduler=scheduler,
            max_grad_norm=cfg.MAX_GRAD_NORM,
            patience=cfg.EARLY_STOPPING_PATIENCE,
        )

        print(f"[Unfreeze] Unfreezing backbone for remaining {cfg.NUM_EPOCHS - WARMUP_FREEZE_EPOCHS} epochs.")
        for p in model.parameters():
            p.requires_grad = True

        history_main = train(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=cfg.NUM_EPOCHS - WARMUP_FREEZE_EPOCHS,
            checkpoint_every=True,
            save_dir=save_dir,
            scheduler=scheduler,
            max_grad_norm=cfg.MAX_GRAD_NORM,
            patience=cfg.EARLY_STOPPING_PATIENCE,
        )

        history = _merge_histories(history_warmup, history_main)
    else:
        history = train(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=cfg.NUM_EPOCHS,
            checkpoint_every=True,
            save_dir=save_dir,
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
