from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class BiopsyUNet(nn.Module):
    """U-Net segmentation model configured for biopsy site segmentation."""

    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_weights: str = "imagenet",
    ) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for a batch of RGB images."""
        return self.model(x)