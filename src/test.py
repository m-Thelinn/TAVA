import sys
import os
# Append project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from metrics.metrics import dice_score, iou_score


@torch.no_grad()
def epoch_test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_images = 0

    start = time.time()
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        running_loss += loss.item()
        running_dice += dice_score(outputs, masks, apply_sigmoid=True).item()
        running_iou += iou_score(outputs, masks, apply_sigmoid=True).item()
        total_images += images.size(0)

    total_time = time.time() - start
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    time_per_image_ms = (total_time / total_images * 1000) if total_images > 0 else 0.0
    return epoch_loss, epoch_dice, epoch_iou, total_time, time_per_image_ms


def test(model, dataloaders, criterion, device, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    test_loader = dataloaders["test"]
    test_loss, test_dice, test_iou, total_time, time_per_image_ms = epoch_test(model, test_loader, criterion, device)

    # Store and save metrics
    results = {
        "test_loss":              [test_loss],
        "test_dice":              [test_dice],
        "test_iou":               [test_iou],
        "inference_total_s":      [round(total_time, 2)],
        "inference_per_image_ms": [round(time_per_image_ms, 3)],
    }
    pd.DataFrame(results).to_csv(os.path.join(save_dir, "test_metrics.csv"), index=False)

    print(f"Test Loss: {test_loss:.4f} | Dice: {test_dice:.4f} | IoU: {test_iou:.4f}")
    print(f"Inference -> Total: {total_time:.2f}s | Per image: {time_per_image_ms:.1f}ms")

    metrics_csv_path = os.path.join(save_dir, "metrics.csv")
    if os.path.exists(metrics_csv_path):
        df = pd.read_csv(metrics_csv_path)
        if not df.empty and "val_dice" in df.columns:
            # Save the best results apart
            best_idx = df["val_dice"].idxmax()
            best_results = df.iloc[[best_idx]]
            best_results.to_csv(os.path.join(save_dir, "best_metrics.csv"), index=False)

            # Extraer las epoch para los plots
            epochs = df["epoch"]

            # Plot Loss vs Epochs
            plt.figure()
            plt.plot(epochs, df["train_loss"], label="Train Loss")
            plt.plot(epochs, df["val_loss"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Loss vs Epochs")
            plt.savefig(os.path.join(save_dir, "loss_vs_epochs.png"))
            plt.close()

            # Plot Dice vs Epochs
            plt.figure()
            plt.plot(epochs, df["train_dice"], label="Train Dice")
            plt.plot(epochs, df["val_dice"], label="Val Dice")
            plt.xlabel("Epoch")
            plt.ylabel("Dice")
            plt.legend()
            plt.title("Dice vs Epochs")
            plt.savefig(os.path.join(save_dir, "dice_vs_epochs.png"))
            plt.close()

            # Plot IoU vs Epochs
            plt.figure()
            plt.plot(epochs, df["train_iou"], label="Train IoU")
            plt.plot(epochs, df["val_iou"], label="Val IoU")
            plt.xlabel("Epoch")
            plt.ylabel("IoU")
            plt.legend()
            plt.title("IoU vs Epochs")
            plt.savefig(os.path.join(save_dir, "iou_vs_epochs.png"))
            plt.close()

    # Save 3 Real vs Predicted Mask Comparisons
    pred_dir = os.path.join(save_dir, "predicted_masks")
    os.makedirs(pred_dir, exist_ok=True)
    
    model.eval()
    num_saved = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            for i in range(images.size(0)):
                if num_saved >= 3:
                    break
                
                img = images[i].cpu()
                if img.shape[0] == 3:
                    img = img.permute(1, 2, 0).numpy()
                elif img.shape[0] == 1:
                    img = img.squeeze(0).numpy()
                else:
                    img = img.numpy()
                
                mask = masks[i].cpu().squeeze().numpy()
                pred = preds[i].cpu().squeeze().numpy()
                
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                if len(img.shape) == 2:
                    plt.imshow(img, cmap="gray")
                else:
                    plt.imshow(img)
                plt.title("Image")
                plt.axis("off")
                
                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap="gray")
                plt.title("Real Mask")
                plt.axis("off")
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap="gray")
                plt.title("Predicted Mask")
                plt.axis("off")
                
                plt.savefig(os.path.join(pred_dir, f"comparison_{num_saved}.png"), bbox_inches="tight")
                plt.close()
                num_saved += 1
            
            if num_saved >= 3:
                break

    return results

if __name__ == "__main__":
    from config import Config
    from preprocessing.dataloader import get_dataloaders
    from models.segformer import SegFormer
    from metrics.loss import DiceLoss, CombinedBCEDiceLoss, FocalLoss, CombinedFocalDiceLoss

    cfg = Config()
    device = torch.device(cfg.DEVICE)

    print("\n--- Setup Dataloaders ---")
    dataloaders = get_dataloaders(
        batch_size=cfg.BATCH_SIZE,
        image_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    )

    print("\n--- Setup Architecture ---")
    if cfg.MODEL_TYPE == "segformer":
        model = SegFormer(num_classes=1).to(device)
    else:
        raise ValueError(f"Model {cfg.MODEL_TYPE} not supported.")
        
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
        raise ValueError(f"Criterion {cfg.CRITERION} not supported.")
        
    save_dir = os.path.join(cfg.SAVE_DIR, f"{cfg.MODEL_TYPE}_{cfg.CRITERION}")
    checkpoint_path = os.path.join(save_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(save_dir, "last_checkpoint.pth")
        
    if os.path.exists(checkpoint_path):
        print(f"\n--- Loading weights from {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"\n[!] Warning: No checkpoint found at {save_dir}. Testing untrained model...")

    print("\n--- Begin Testing ---")
    test(model, dataloaders, criterion, device, save_dir=save_dir)