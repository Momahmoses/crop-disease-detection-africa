"""
Data augmentation for crop disease images.
Simulates bad phone cameras, variable field lighting, and soil backgrounds.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import os

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS = True
except ImportError:
    ALBUMENTATIONS = False


def get_train_transforms(image_size: int = 224):
    if not ALBUMENTATIONS:
        return None
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=5),
        ], p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.CLAHE(clip_limit=4.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.6),
        A.GaussNoise(var_limit=(10, 80), p=0.3),
        A.RandomShadow(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    if not ALBUMENTATIONS:
        return None
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def simulate_phone_camera_degradation(image: np.ndarray) -> np.ndarray:
    """Simulate poor phone camera quality common in rural field use."""
    pil = Image.fromarray(image.astype(np.uint8))
    degradations = [
        lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0))),
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5)),
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3)),
        lambda img: img.resize((int(img.width * 0.5), int(img.height * 0.5))).resize((img.width, img.height)),
    ]
    n_degradations = random.randint(1, 3)
    for fn in random.sample(degradations, n_degradations):
        pil = fn(pil)
    return np.array(pil)


def generate_synthetic_disease_sample(base_image: np.ndarray, disease_type: str) -> np.ndarray:
    """
    Simple synthetic augmentation for rare disease classes with <500 samples.
    Applies disease-specific color transformations to generate new samples.
    """
    pil = Image.fromarray(base_image.astype(np.uint8))
    color_shifts = {
        "yellow_spot": {"hue": 30, "saturation": 1.3},
        "brown_rust": {"hue": -20, "saturation": 1.5},
        "bacterial_blight": {"hue": 0, "saturation": 0.7},
        "leaf_curl": {"hue": 15, "saturation": 1.1},
    }
    shift = color_shifts.get(disease_type, {"hue": 0, "saturation": 1.0})
    pil = ImageEnhance.Color(pil).enhance(shift["saturation"])
    pil_arr = np.array(pil).astype(np.float32)
    pil_arr[:, :, 0] = np.clip(pil_arr[:, :, 0] + shift["hue"], 0, 255)
    return pil_arr.astype(np.uint8)
