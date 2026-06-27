import torch


def dice_score(logits, targets):

    probs = torch.sigmoid(logits)

    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum()

    union = preds.sum() + targets.sum()

    return (
        (2 * intersection + 1)
        / (union + 1)
    ).item()