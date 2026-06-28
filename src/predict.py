from __future__ import annotations

import argparse
import logging
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image

from models.unet import BiopsyUNet
from utils import create_output_dir, load_checkpoint, load_yaml_config, select_device, seed_everything

LOGGER = logging.getLogger("biopsyid")


def build_inference_transform(image_size: int) -> A.Compose:
    """Create preprocessing transforms for inference."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ]
    )


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk and convert it to RGB numpy array."""
    image = Image.open(image_path).convert("RGB")
    return np.array(image, dtype=np.uint8)


def predict_single_image(
    model: torch.nn.Module,
    image_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    image_size: int = 512,
    device: torch.device | None = None,
    threshold: float = 0.5,
    make_overlay: bool = True,
) -> tuple[Path, Path, Path]:
    """Run inference on a single image and save the probability map and mask."""
    device = device or select_device("cpu")
    model.to(device)
    model.eval()

    transform = build_inference_transform(image_size=image_size)
    image_np = load_image(image_path)
    transformed = transform(image=image_np, mask=np.zeros(image_np.shape[:2], dtype=np.uint8))
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    load_checkpoint(checkpoint_path, model, device=device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (probs > threshold).astype(np.uint8) * 255
    output_path = create_output_dir(output_dir)
    prob_path = output_path / f"{Path(image_path).stem}_probability.png"
    mask_path = output_path / f"{Path(image_path).stem}_mask.png"
    overlay_path = output_path / f"{Path(image_path).stem}_overlay.png"

    Image.fromarray((probs * 255).astype(np.uint8)).save(prob_path)
    Image.fromarray(mask).save(mask_path)

    if make_overlay:
        overlay = image_np.copy()
        overlay[mask[..., None] > 0] = [255, 0, 0]
        Image.fromarray(overlay).save(overlay_path)

    LOGGER.info("Saved probability map to %s", prob_path)
    LOGGER.info("Saved binary mask to %s", mask_path)
    if make_overlay:
        LOGGER.info("Saved overlay to %s", overlay_path)
    return prob_path, mask_path, overlay_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run biopsy site prediction")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt", help="Path to the checkpoint")
    parser.add_argument("--output-dir", default="outputs/predictions", help="Directory for saved outputs")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to the training config")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for mask creation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = load_yaml_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = select_device(config.get("device"))

    model = BiopsyUNet(
        encoder_name=str(config.get("encoder", "efficientnet-b3")),
        encoder_weights=str(config.get("encoder_weights", "imagenet")),
    )
    predict_single_image(
        model=model,
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        image_size=int(config.get("image_size", 512)),
        device=device,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()