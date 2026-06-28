"""
Utility functions for BiopsyID.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    """
    Seed all random number generators for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    """

    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
) -> None:
    """
    Save training checkpoint.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> tuple[int, float]:
    """
    Load a training checkpoint.

    Returns:
        epoch, best_loss
    """

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return (
        checkpoint.get("epoch", 0),
        checkpoint.get("best_loss", float("inf")),
    )


def create_output_dirs(root: str | Path = "outputs") -> None:
    """
    Create project output directories.
    """

    root = Path(root)

    (root / "checkpoints").mkdir(
        parents=True,
        exist_ok=True,
    )

    (root / "logs").mkdir(
        parents=True,
        exist_ok=True,
    )

    (root / "predictions").mkdir(
        parents=True,
        exist_ok=True,
    )


def get_device() -> torch.device:
    """
    Return CUDA if available, otherwise CPU.
    """

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def setup_logger(name: str = "biopsyid") -> logging.Logger:
    """
    Configure and return a logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger