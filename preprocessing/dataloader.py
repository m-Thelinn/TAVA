"""
Segmentation DataLoader for DMID dermoscopic images.

Loads preprocessed images and their corresponding segmentation masks,
splits into train / validation / test sets with stratified sampling
to preserve the positive/negative ratio across all splits.

Positive = image has a mask file (pathology present)
Negative = image has no mask file (mask is all zeros)
"""

import os
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A

from preprocessing.pipeline import load_image


# ──────────────────────────── Default paths ────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE_DIR = os.path.join(_PROJECT_ROOT, "DMID_PNG", "1024", "TIFF_PREPROCESSED")
DEFAULT_MASK_DIR  = os.path.join(_PROJECT_ROOT, "DMID_PNG", "1024", "Masks")


# ──────────────────────────── Dataset ──────────────────────────────────
class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for image–mask pairs.

    Parameters
    ----------
    image_paths : list[str]
        Absolute paths to the input images.
    mask_paths : list[str | None]
        Absolute paths to the masks.  ``None`` for negative samples
        (a zero-filled mask is generated on the fly).
    image_size : tuple[int, int]
        (H, W) to resize both images and masks.
    """

    def __init__(
        self,
        image_paths: list[str],
        mask_paths: list[Optional[str]],
        image_size: tuple[int, int] = (256, 256),
        transform: Optional[A.Compose] = None,
    ):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # ── Image ──
        try:
            image = load_image(self.image_paths[idx])
        except Exception:
            raise FileNotFoundError(f"Image not found or could not be loaded: {self.image_paths[idx]}")
        
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be loaded: {self.image_paths[idx]}")

        # ── Mask ──
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Apply Albumentations if provided
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, self.image_size[::-1])
            mask = cv2.resize(mask, self.image_size[::-1])

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)  # binarize

        # Convert to Tensor (handling both Grayscale and RGB images safely)
        if len(image.shape) == 2:
            image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
            
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return image, mask


# ──────────────────────────── Factory ──────────────────────────────────
def get_dataloaders(
    image_dir: str = DEFAULT_IMAGE_DIR,
    mask_dir: str = DEFAULT_MASK_DIR,
    batch_size: int = 8,
    image_size: tuple[int, int] = (256, 256),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 2,
    random_state: int = 42,
) -> dict[str, DataLoader]:
    """
    Build stratified train / val / test DataLoaders.

    Stratification key: whether the image has a corresponding mask
    (positive) or not (negative).  This guarantees the positive/negative
    ratio is (approximately) the same across all three splits.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping to a
    :class:`torch.utils.data.DataLoader`.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # ── Collect all images ──
    all_images = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".png")
    )
    mask_set = set(
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    )

    image_paths: list[str] = []
    mask_paths: list[Optional[str]] = []
    labels: list[int] = []  # 1 = positive (has mask), 0 = negative

    for fname in all_images:
        image_paths.append(os.path.join(image_dir, fname))
        if fname in mask_set:
            mask_paths.append(os.path.join(mask_dir, fname))
            labels.append(1)
        else:
            mask_paths.append(None)
            labels.append(0)

    n_total = len(image_paths)
    n_pos = sum(labels)
    n_neg = n_total - n_pos
    print(f"[DataLoader] Found {n_total} images  ({n_pos} positive, {n_neg} negative)")

    # ── Stratified split: train vs (val + test) ──
    val_test_ratio = val_ratio + test_ratio
    img_train, img_valtest, mask_train, mask_valtest, lab_train, lab_valtest = \
        train_test_split(
            image_paths, mask_paths, labels,
            test_size=val_test_ratio,
            stratify=labels,
            random_state=random_state,
        )

    # ── Split val+test into val and test ──
    relative_test = test_ratio / val_test_ratio
    img_val, img_test, mask_val, mask_test = train_test_split(
        img_valtest, mask_valtest,
        test_size=relative_test,
        stratify=lab_valtest,
        random_state=random_state,
    )

    # ── Albumentations Transforms ──
    train_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    ])

    val_test_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
    ])

    splits = {
        "train": (img_train, mask_train),
        "val":   (img_val,   mask_val),
        "test":  (img_test,  mask_test),
    }

    dataloaders: dict[str, DataLoader] = {}
    for split_name, (img_list, msk_list) in splits.items():
        # Apply intense augmentation only for training
        transform = train_transform if split_name == "train" else val_test_transform
        ds = SegmentationDataset(img_list, msk_list, image_size=image_size, transform=transform)
        
        dataloaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        pos = sum(1 for m in msk_list if m is not None)
        neg = len(msk_list) - pos
        print(f"[DataLoader]   {split_name:>5s}: {len(ds):>4d} samples  "
              f"(pos={pos}, neg={neg})")

    return dataloaders
