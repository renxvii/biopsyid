import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from datasets.biopsy_dataset import BiopsyDataset
from losses.dice import CombinedLoss


def test_biopsy_dataset_loads_matching_image_and_mask(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image_path = image_dir / "sample.jpg"
    mask_path = mask_dir / "sample.png"

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:15, 5:15] = 1

    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    dataset = BiopsyDataset(image_dir=str(image_dir), mask_dir=str(mask_dir), transform=None)

    sample_image, sample_mask = dataset[0]

    assert sample_image.shape[0] == 3
    assert sample_mask.shape[0] == 1
    assert torch.equal(sample_mask.unique(), torch.tensor([0.0, 1.0]))


def test_combined_loss_is_finite_and_reduces():
    logits = torch.tensor([[0.2, -0.4], [0.6, 0.8]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    loss = CombinedLoss()(logits, targets)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
