"""
Metrics for binary semantic segmentation.
"""

from __future__ import annotations

import torch


EPSILON = 1e-7


def _prepare_predictions(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts logits into binary predictions.
    """

    probs = torch.sigmoid(logits)

    return (probs > 0.5).float()


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:

    preds = _prepare_predictions(logits)

    preds = preds.view(-1)

    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    dice = (
        2 * intersection + EPSILON
    ) / (
        preds.sum() + targets.sum() + EPSILON
    )

    return dice.item()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:

    preds = _prepare_predictions(logits)

    preds = preds.view(-1)

    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    union = preds.sum() + targets.sum() - intersection

    iou = (
        intersection + EPSILON
    ) / (
        union + EPSILON
    )

    return iou.item()


def precision(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:

    preds = _prepare_predictions(logits)

    tp = (preds * targets).sum()

    fp = (preds * (1 - targets)).sum()

    return (
        (tp + EPSILON)
        / (tp + fp + EPSILON)
    ).item()


def recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:

    preds = _prepare_predictions(logits)

    tp = (preds * targets).sum()

    fn = ((1 - preds) * targets).sum()

    return (
        (tp + EPSILON)
        / (tp + fn + EPSILON)
    ).item()


def f1_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:

    p = precision(logits, targets)

    r = recall(logits, targets)

    return (
        2 * p * r
    ) / (
        p + r + EPSILON
    )