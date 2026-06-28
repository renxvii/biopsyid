"""
Dataset for biopsy image segmentation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BiopsyDataset(Dataset):
    """
    Dataset for binary semantic segmentation.

    Directory structure:

    images/
        image001.jpg
        image002.jpg

    masks/
        image001.png
        image002.png
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform=None,
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.transform = transform

        self.images = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.suffix.lower() in self.IMAGE_EXTENSIONS
            ]
        )

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):

        image_path = self.images[index]

        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Could not load {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"Could not load {mask_path}"
            )

        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        else:

            image = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            mask = (
                torch.from_numpy(mask)
                .unsqueeze(0)
                .float()
            )

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "filename": image_path.name,
        }