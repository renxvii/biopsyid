"""
Training entry point for BiopsyID.
"""

from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.datasets import (
    BiopsyDataset,
    get_train_transforms,
    get_val_transforms,
)
from src.losses import SegmentationLoss
from src.models.unet import BiopsyUNet
from src.trainer import Trainer
from src.utils import seed_everything


def load_yaml(path: str):

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    seed_everything()

    train_cfg = load_yaml("configs/train.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    dataset_cfg = load_yaml("configs/dataset.yaml")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    train_dataset = BiopsyDataset(
        image_dir=dataset_cfg["train_images"],
        mask_dir=dataset_cfg["train_masks"],
        transform=get_train_transforms(
            train_cfg["image_size"]
        ),
    )

    val_dataset = BiopsyDataset(
        image_dir=dataset_cfg["val_images"],
        mask_dir=dataset_cfg["val_masks"],
        transform=get_val_transforms(
            train_cfg["image_size"]
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    model = BiopsyUNet(
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
    )

    model.to(device)

    criterion = SegmentationLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=1e-4,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        epochs=train_cfg["epochs"],
    )

    trainer.train()


if __name__ == "__main__":
    main()