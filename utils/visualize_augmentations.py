import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add the project root to sys.path so we can import preprocessing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.dataloader import get_dataloaders

def denormalize(tensor):
    """Convert a [C, H, W] tensor in [0,1] back to a numpy image for plotting."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.numpy()
    if img.shape[0] == 1:
        # Grayscale
        img = img.squeeze(0)
    else:
        # RGB
        img = np.transpose(img, (1, 2, 0))
    return img

def main():
    print("Initializing dataloaders...")
    dataloaders = get_dataloaders(batch_size=1, num_workers=0)
    
    train_loader = dataloaders["train"]
    train_ds = train_loader.dataset
    
    original_size = train_ds._n_real
    expanded_size = len(train_ds)
    
    print("\n" + "="*40)
    print("--- Dataloader Functionality Verification ---")
    print(f"Original unique training images: {original_size}")
    print(f"Total dataset size (augment_ratio={train_ds.augment_ratio}): {expanded_size}")
    print(f"Number of augmented copies: {expanded_size - original_size}")
    print("="*40 + "\n")
    
    if original_size == 0:
        print("No training images found.")
        return

    # We want to visualize the exact same underlying image fetched normally 
    # vs fetched from the "expanded / artificially multiplied" index region.
    # Because __getitem__ uses `real_idx = idx % original_size`, index `idx` 
    # and index `idx + original_size` will fetch the EXACT same original file 
    # but run it through the random Albumentations pipeline a second time independently.
    
    num_samples = 4
    
    # We purposefully select a few indices from the first 50% of the dataset
    # (so we are sure they also appear in the extra 0.7x section of the epoch)
    # We will prioritize those with positive masks to see augmentations on the lesion clearly.
    sample_indices = []
    for i, mask_path in enumerate(train_ds.mask_paths):
        # We only consider `i` if `i + original_size < expanded_size` 
        if mask_path is not None and i in set(train_ds._aug_indices):
            sample_indices.append(i)
        if len(sample_indices) >= num_samples:
            break
            
    if not sample_indices:
        print("Could not find positive mask images valid for duplication. Picking any...")
        for i in range(num_samples):
            if i + original_size < expanded_size:
                sample_indices.append(i)
        
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, 4 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = np.expand_dims(axes, 0)
    
    for row, idx in enumerate(sample_indices):
        aug_pos = train_ds._aug_indices.index(idx)
        idx_expanded = original_size + aug_pos
        
        # --- Fetch from Normal Region (Pass 1) ---
        img1_t, mask1_t = train_ds[idx]
        img1 = denormalize(img1_t)
        mask1 = denormalize(mask1_t)
        
        # --- Fetch from Expanded Region (Pass 2) ---
        img2_t, mask2_t = train_ds[idx_expanded]
        img2 = denormalize(img2_t)
        mask2 = denormalize(mask2_t)
        
        # 1. Image Pass 1
        axes[row, 0].imshow(img1, cmap='gray' if len(img1.shape)==2 else None, vmin=0, vmax=1)
        axes[row, 0].set_title(f"Original (idx={idx})")
        axes[row, 0].axis('off')
        
        # 2. Mask Pass 1
        axes[row, 1].imshow(mask1, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(f"Original Mask")
        axes[row, 1].axis('off')
        
        # 3. Image Pass 2
        axes[row, 2].imshow(img2, cmap='gray' if len(img2.shape)==2 else None, vmin=0, vmax=1)
        axes[row, 2].set_title(f"Augmented (idx={idx_expanded})")
        axes[row, 2].axis('off')
        
        # 4. Mask Pass 2
        axes[row, 3].imshow(mask2, cmap='gray', vmin=0, vmax=1)
        axes[row, 3].set_title("Augmented Mask")
        axes[row, 3].axis('off')

    plt.tight_layout()
    out_path = "dataloader_verification.png"
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
