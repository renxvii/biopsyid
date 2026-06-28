"""
PyTorch Dataset for biopsy image segmentation.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BiopsyDataset(Dataset):
    """
    Dataset for biopsy image segmentation.

    Expected directory structure:

    images/
        patient001.jpg
        patient002.jpg

    masks/
        patient001.png
        patient002.png
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
        image_dir: str,
        mask_dir: Optional[str] = None,
        transform=None,
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None

        self.transform = transform

        self.images = sorted(
            [
                file
                for file in self.image_dir.iterdir()
                if file.suffix.lower() in self.IMAGE_EXTENSIONS
            ]
        )

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        image = cv2.imread(str(image_path))

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        if self.mask_dir is not None:

            mask_path = (
                self.mask_dir
                / f"{image_path.stem}.png"
            )

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if mask is None:
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            mask = (mask > 127).astype(np.float32)

        else:

            mask = np.zeros(
                image.shape[:2],
                dtype=np.float32,
            )

        if self.transform:

            augmented = self.transform(
                image=image,
                mask=mask,
            )

            image = augmented["image"]

            mask = augmented["mask"]

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

        return image, mask