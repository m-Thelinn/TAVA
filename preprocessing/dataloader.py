import pandas as pd
import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MedicalImageDataset(Dataset):
    def __init__(self, metadata_path: str, image_dir: str, transform=None, label_col_idx: int = 1):
        """
        Args:
            metadata_path (str): Path to the excel metadata file.
            image_dir (str): Directory with all the preprocessed images.
            transform (callable, optional): Optional transform to be applied on a sample.
            label_col_idx (int): Index of the column to use as the label/target.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load and clean metadata
        raw_metadata = pd.read_excel(metadata_path)
        # Drop rows where the first column is null or doesn't start with 'IMG'
        # to skip the explanation headers at the beginning of Metadata.xlsx
        raw_metadata = raw_metadata.dropna(subset=[raw_metadata.columns[0]])
        self.metadata = raw_metadata[raw_metadata.iloc[:, 0].astype(str).str.startswith('IMG')].reset_index(drop=True)

        self.label_col_idx = label_col_idx
        
        # Convert the selected label column to categorical integer IDs
        # For example: 'MLOLT' -> 0, 'CCRT' -> 1, etc...
        # If your objective is predicting Benign/Malignant, you'll want to change label_col_idx
        # to the column that holds those values (e.g., column index 4 for Unnamed: 4).
        labels = self.metadata.iloc[:, self.label_col_idx].astype(str)
        self.classes = sorted(labels.unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE: You must adjust the column indices depending on the Metadata.xlsx structure.
        # Here we assume the first column (index 0) is the filename/id and second (index 1) is the label.
        img_name = str(self.metadata.iloc[idx, 0]).strip()
        
        # Ensure the extension matches the preprocessed images
        if not img_name.endswith('.png'):
            img_name += '.png'
            
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image using OpenCV and normalize
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        image = image.astype(np.float32) / 255.0
        # Add channel dimension (C, H, W) for PyTorch
        image = np.expand_dims(image, axis=0)

        # Target label mapped to integer
        raw_label = str(self.metadata.iloc[idx, self.label_col_idx])
        label = self.class_to_idx[raw_label]

        # Apply optional transforms
        if self.transform:
            # Transforms are typically applied directly if using torchvision,
            # For Albumentations, modify this sequence to: image = self.transform(image=image)["image"]
            image = self.transform(image)
            
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        # Adapt dtype as needed: torch.long for classification, torch.float for regression
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

def get_dataloader(metadata_path: str, image_dir: str, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, label_col_idx: int = 1):
    """
    Creates and returns a PyTorch DataLoader.
    """
    dataset = MedicalImageDataset(metadata_path, image_dir, label_col_idx=label_col_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

    # Example usage:
    # dl = get_dataloader(
    #     metadata_path="../DMID_PNG/Metadata.xlsx",
    #     image_dir="../DMID_PNG/1024/TIFF_PREPROCESSED/",
    #     batch_size=8
    # )
    # for images, labels in dl:
    #     print(images.shape, labels.shape)
    #     break
