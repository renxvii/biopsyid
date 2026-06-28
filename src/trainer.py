"""
trainer.py

Generic trainer for binary semantic segmentation.
"""

from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.metrics import (
    dice_score,
    f1_score,
    iou_score,
    precision,
    recall,
)


class Trainer:
    """
    Generic trainer for semantic segmentation models.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        epochs,
        scheduler=None,
        checkpoint_dir="outputs/checkpoints",
        log_dir="outputs/logs",
        mixed_precision=True,
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.criterion = criterion

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.scheduler = scheduler

        self.device = device

        self.epochs = epochs

        self.writer = SummaryWriter(log_dir)

        self.scaler = GradScaler(enabled=mixed_precision)

        self.mixed_precision = mixed_precision

        self.best_loss = float("inf")

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    ###########################################################

    def train(self):

        print("\nBeginning Training\n")

        for epoch in range(1, self.epochs + 1):

            train_loss = self.train_epoch(epoch)

            metrics = self.validate(epoch)

            if self.scheduler is not None:

                self.scheduler.step(metrics["loss"])

            self.writer.add_scalar(
                "Train/Loss",
                train_loss,
                epoch,
            )

            for name, value in metrics.items():

                self.writer.add_scalar(
                    f"Validation/{name}",
                    value,
                    epoch,
                )

            self.writer.add_scalar(
                "Learning Rate",
                self.optimizer.param_groups[0]["lr"],
                epoch,
            )

            self.print_epoch(epoch, train_loss, metrics)

            if metrics["loss"] < self.best_loss:

                self.best_loss = metrics["loss"]

                self.save_checkpoint(
                    "best_model.pth",
                    epoch,
                )

        self.save_checkpoint(
            "last_model.pth",
            self.epochs,
        )

        self.writer.close()

    ###########################################################

    def train_epoch(self, epoch):

        self.model.train()

        running_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
        )

        for images, masks in progress:

            images = images.to(self.device)

            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):

                logits = self.model(images)

                loss = self.criterion(
                    logits,
                    masks,
                )

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            self.scaler.update()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return running_loss / len(self.train_loader)

    ###########################################################

    @torch.no_grad()
    def validate(self, epoch):

        self.model.eval()

        metrics = {

            "loss": 0.0,

            "dice": 0.0,

            "iou": 0.0,

            "precision": 0.0,

            "recall": 0.0,

            "f1": 0.0,

        }

        progress = tqdm(

            self.val_loader,

            desc=f"Epoch {epoch} [Validation]",

        )

        for images, masks in progress:

            images = images.to(self.device)

            masks = masks.to(self.device)

            logits = self.model(images)

            loss = self.criterion(
                logits,
                masks,
            )

            metrics["loss"] += loss.item()

            metrics["dice"] += dice_score(
                logits,
                masks,
            )

            metrics["iou"] += iou_score(
                logits,
                masks,
            )

            metrics["precision"] += precision(
                logits,
                masks,
            )

            metrics["recall"] += recall(
                logits,
                masks,
            )

            metrics["f1"] += f1_score(
                logits,
                masks,
            )

        for key in metrics:

            metrics[key] /= len(self.val_loader)

        return metrics

    ###########################################################

    def save_checkpoint(
        self,
        filename,
        epoch,
    ):

        torch.save(

            {

                "epoch": epoch,

                "model_state_dict": self.model.state_dict(),

                "optimizer_state_dict": self.optimizer.state_dict(),

                "best_loss": self.best_loss,

            },

            self.checkpoint_dir / filename,

        )

    ###########################################################

    @staticmethod
    def print_epoch(
        epoch,
        train_loss,
        metrics,
    ):

        print()

        print("=" * 60)

        print(f"Epoch {epoch}")

        print("=" * 60)

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Val Loss   : {metrics['loss']:.4f}")

        print(f"Dice       : {metrics['dice']:.4f}")

        print(f"IoU        : {metrics['iou']:.4f}")

        print(f"Precision  : {metrics['precision']:.4f}")

        print(f"Recall     : {metrics['recall']:.4f}")

        print(f"F1 Score   : {metrics['f1']:.4f}")

        print()