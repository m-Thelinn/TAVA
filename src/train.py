import torch
import torch.nn as nn
import pandas as pd
import os
import time
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.metrics import dice_score, iou_score


def epoch_train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    start = time.time()
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        with torch.no_grad():
            running_dice += dice_score(outputs, masks, apply_sigmoid=True).item()
            running_iou += iou_score(outputs, masks, apply_sigmoid=True).item()

    epoch_time = time.time() - start
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    return epoch_loss, epoch_dice, epoch_iou, epoch_time


@torch.no_grad()
def epoch_val(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    start = time.time()
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        running_loss += loss.item()
        running_dice += dice_score(outputs, masks, apply_sigmoid=True).item()
        running_iou += iou_score(outputs, masks, apply_sigmoid=True).item()

    epoch_time = time.time() - start
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    return epoch_loss, epoch_dice, epoch_iou, epoch_time


def train(model: nn.Module, dataloaders: dict[str, DataLoader], criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int, checkpoint_every: bool = False, save_dir: str = "outputs", scheduler=None):
    os.makedirs(save_dir, exist_ok=True)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    best_val_dice = 0.0

    history = {
        "epoch": [],
        "train_loss": [], "train_dice": [], "train_iou": [], "train_time_s": [],
        "val_loss":   [], "val_dice":   [], "val_iou":   [], "val_time_s":   [],
    }

    for epoch in range(num_epochs):
        epoch_loss, epoch_dice, epoch_iou, train_time = epoch_train(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou, val_time = epoch_val(model, val_loader, criterion, device)

        # Store metrics
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["train_dice"].append(epoch_dice)
        history["train_iou"].append(epoch_iou)
        history["train_time_s"].append(round(train_time, 2))
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_time_s"].append(round(val_time, 2))

        # Step Scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_dice)
            else:
                scheduler.step()

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        # Checkpoint every epoch
        if checkpoint_every:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "history": history,
            }
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(checkpoint, os.path.join(save_dir, "last_checkpoint.pth"))

        # Persist metrics to CSV incrementally
        pd.DataFrame(history).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train  -> Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f} | IoU: {epoch_iou:.4f} | Time: {train_time:.1f}s")
        print(f"  Val    -> Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | Time: {val_time:.1f}s")

    return history
