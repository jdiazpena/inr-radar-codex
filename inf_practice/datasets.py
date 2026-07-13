# datasets.py

from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from skimage import data as skdata


def get_mgrid(sidelen: int) -> torch.Tensor:
    """
    Create a flattened 2D coordinate grid in [-1, 1].
    Returns shape [sidelen*sidelen, 2].
    """
    y, x = np.mgrid[0:sidelen, 0:sidelen].astype(np.float32)

    # Normalize to [0, 1]
    y = y / (sidelen - 1)
    x = x / (sidelen - 1)

    # Shift to [-1, 1]
    y = 2.0 * (y - 0.5)
    x = 2.0 * (x - 0.5)

    coords = np.stack([y, x], axis=-1)   # [H, W, 2]
    coords = torch.from_numpy(coords).view(-1, 2)
    return coords


def load_builtin_camera() -> Image.Image:
    """
    Load the same built-in 512x512 grayscale camera image used in the SIREN example.
    """
    arr = skdata.camera()   # uint8, shape [512, 512]
    return Image.fromarray(arr).convert("L")


class ImageINRDataset(Dataset):
    """
    Returns the full coordinate-value dataset for one grayscale image.

    coords: [H*W, 2] in [-1, 1]
    values: [H*W, 1] in [-1, 1]
    """
    def __init__(self, sidelength: int = 256, image_path: str | None = None):
        super().__init__()
        self.sidelength = sidelength

        if image_path is None:
            img = load_builtin_camera()
        else:
            img = Image.open(image_path).convert("L")

        self.transform = Compose([
            Resize((sidelength, sidelength)),
            ToTensor(),                         # [1, H, W], values in [0, 1]
            Normalize(mean=[0.5], std=[0.5]),  # map to [-1, 1]
        ])

        img_tensor = self.transform(img)              # [1, H, W]
        self.values = img_tensor.permute(1, 2, 0).reshape(-1, 1)
        self.coords = get_mgrid(sidelength)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "coords": self.coords,   # [N, 2]
            "values": self.values,   # [N, 1]
        }