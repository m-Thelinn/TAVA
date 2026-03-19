import cv2
import numpy as np
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the given path.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    return image

def gaussian_denoising(image: np.ndarray, ksize:int = 5) -> np.ndarray:
    
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def clahe_contrast(image: np.ndarray, clip_limit:float = 0.5, tile_grid_size:tuple = (8, 8)) -> np.ndarray:
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def preprocess_image(image: str) -> np.ndarray:
    
    image = load_image(image)
    image = gaussian_denoising(image)
    image = clahe_contrast(image)
    
    return image


def preprocess_dataset(dataset_path: str) -> None:
    
    for image_path in os.listdir(dataset_path):
        if image_path.endswith(".png"):
            image = preprocess_image(os.path.join(dataset_path, image_path))
            cv2.imwrite("DMID_PNG/1024/TIFF_PREPROCESSED/" + image_path, image)


if __name__ == "__main__":
    preprocess_dataset("DMID_PNG/1024/TIFF")