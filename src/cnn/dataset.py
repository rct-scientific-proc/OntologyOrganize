"""PyTorch Dataset for loading labeled images from the application."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.image_utils import normalize_image


# Standard ImageNet normalization for pretrained ResNet
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _prepare_image(img: Image.Image) -> Image.Image:
    """Normalize and convert a PIL image to 3-channel RGB for ResNet input."""
    img = normalize_image(img)
    if img.mode == 'L':
        img = img.convert('RGB')
    elif img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    return img


class LabeledImageDataset(Dataset):
    """Dataset of labeled images for training.

    Args:
        image_paths: List of image file paths
        labels: List of integer class indices corresponding to each image
        image_cache: Optional dict mapping path strings to pre-loaded PIL Images
        transform: Torchvision transform to apply (default: ImageNet normalization)
    """

    def __init__(self, image_paths: list[Path], labels: list[int],
                 image_cache: dict = None,
                 transform: transforms.Compose = None):
        self.image_paths = image_paths
        self.labels = labels
        self.image_cache = image_cache or {}
        self.transform = transform or IMAGENET_TRANSFORM

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]

        # Try cache first, then load from disk
        cached = self.image_cache.get(str(img_path))
        if cached is not None:
            img = cached
        else:
            img = Image.open(img_path)

        img = _prepare_image(img)
        tensor = self.transform(img)
        return tensor, self.labels[idx]


class UnlabeledImageDataset(Dataset):
    """Dataset of unlabeled images for inference.

    Args:
        image_paths: List of image file paths
        image_cache: Optional dict mapping path strings to pre-loaded PIL Images
        transform: Torchvision transform to apply (default: ImageNet normalization)
    """

    def __init__(self, image_paths: list[Path],
                 image_cache: dict = None,
                 transform: transforms.Compose = None):
        self.image_paths = image_paths
        self.image_cache = image_cache or {}
        self.transform = transform or IMAGENET_TRANSFORM

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]

        cached = self.image_cache.get(str(img_path))
        if cached is not None:
            img = cached
        else:
            img = Image.open(img_path)

        img = _prepare_image(img)
        tensor = self.transform(img)
        return tensor, str(img_path)
