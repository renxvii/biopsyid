import segmentation_models_pytorch as smp
import torch.nn as nn


class BiopsyUNet(nn.Module):
    """
    U-Net model for biopsy site segmentation.

    Input:
        RGB image (B,3,H,W)

    Output:
        Probability map (B,1,H,W)
    """

    def __init__(
        self,
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)