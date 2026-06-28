"""
Metrics for binary semantic segmentation.
"""

from __future__ import annotations

import torch

EPS = 1e-7


def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert logits to binary predictions.
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).float()


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    preds = _binarize(logits, threshold)

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (preds * targets).sum()

    score = (
        (2 * intersection + EPS)
        / (preds.sum() + targets.sum() + EPS)
    )

    return score.item()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    preds = _binarize(logits, threshold)

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (preds * targets).sum()

    union = preds.sum() + targets.sum() - intersection

    score = (
        (intersection + EPS)
        / (union + EPS)
    )

    return score.item()


def precision(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    preds = _binarize(logits, threshold)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()

    score = (
        (tp + EPS)
        / (tp + fp + EPS)
    )

    return score.item()


def recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    preds = _binarize(logits, threshold)

    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()

    score = (
        (tp + EPS)
        / (tp + fn + EPS)
    )

    return score.item()


def f1_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    p = precision(logits, targets, threshold)
    r = recall(logits, targets, threshold)

    return (2 * p * r) / (p + r + EPS)